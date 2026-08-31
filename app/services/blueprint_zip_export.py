from __future__ import annotations

import re
from datetime import datetime
from io import BytesIO
from pathlib import PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from sqlmodel import Session, select

from app.models import ACPPreview, ArtifactRegistryRecord, SessionSnapshot
from app.services.acp_serialization import normalize_text_document, serialize_json_document
from app.services.canonical_export_delivery import build_canonical_export_document
from app.services.diagram_center.persistence import DiagramVersionRecord
from app.services.product_processing.persistence import UncertaintyBacklogRecord


ZIP_FIXED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)

_RENDERING_EXTENSIONS = {
    "svg": "svg",
    "mermaid": "mmd",
    "plantuml": "puml",
    "d2": "d2",
    "bpmn_xml": "bpmn.xml",
}


def _safe_name(value: str, default: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    return normalized or default


def _zip_info(path: str) -> ZipInfo:
    info = ZipInfo(filename=path, date_time=ZIP_FIXED_TIMESTAMP)
    info.compress_type = ZIP_DEFLATED
    return info


def _stable_generated_at(snapshot: SessionSnapshot) -> datetime:
    return snapshot.session.updated_at or snapshot.session.created_at


def _iter_latest_artifacts(snapshot: SessionSnapshot) -> list[ArtifactRegistryRecord]:
    latest: dict[str, ArtifactRegistryRecord] = {}
    for artifact in snapshot.artifact_records:
        key = str(artifact.artifact_key or "").strip()
        if key and key not in latest:
            latest[key] = artifact
    return list(latest.values())


def _artifact_zip_path(artifact: ArtifactRegistryRecord) -> str | None:
    key = str(artifact.artifact_key or "").strip().replace("\\", "/")
    if not key:
        return None
    if key.startswith("Blueprint/"):
        return key
    if key.startswith("ACP/estimation/") or artifact.artifact_kind == "estimation_report":
        return f"Blueprint/estimation/{PurePosixPath(key).name or 'estimation-report.json'}"
    return None


def _iter_latest_diagram_versions(db: Session, *, session_id) -> list[DiagramVersionRecord]:
    rows = db.exec(
        select(DiagramVersionRecord)
        .where(DiagramVersionRecord.session_id == session_id)
        .order_by(DiagramVersionRecord.created_at.desc(), DiagramVersionRecord.version_number.desc())
    ).all()
    latest: dict[str, DiagramVersionRecord] = {}
    for row in rows:
        key = str(row.diagram_key or "").strip()
        if not key or key in latest or str(row.state or "").strip() != "available":
            continue
        latest[key] = row
    return list(latest.values())


def _iter_uncertainty_backlog(db: Session, *, snapshot: SessionSnapshot) -> list[UncertaintyBacklogRecord]:
    statement = select(UncertaintyBacklogRecord).where(UncertaintyBacklogRecord.session_id == snapshot.session.id)
    workspace_id = getattr(snapshot.session, "workspace_id", None)
    if workspace_id is not None:
        statement = statement.where(UncertaintyBacklogRecord.workspace_id == workspace_id)
    return list(
        db.exec(
            statement.order_by(
                UncertaintyBacklogRecord.source_stage.asc(),
                UncertaintyBacklogRecord.uncertainty_key.asc(),
            )
        ).all()
    )


def _decision_status_label(record: UncertaintyBacklogRecord) -> str:
    status = str(record.status or "").strip().lower()
    disposition = str(record.disposition or "").strip().lower()
    if status == "resolved":
        return "respondido"
    if status in {"dismissed", "superseded"}:
        return "descartado"
    if disposition == "defer" or status == "deferred":
        return "delegado"
    if disposition == "infer":
        return "inferido"
    if disposition == "block":
        return "bloqueante"
    return "pendiente"


def _impact_level(record: UncertaintyBacklogRecord) -> str:
    affected_count = len(record.affected_deliverable_keys or [])
    status_label = _decision_status_label(record)
    if status_label == "bloqueante" or affected_count >= 3 or int(record.cost_to_resolve_units or 0) >= 3:
        return "alto"
    if affected_count > 0 or float(record.confidence or 0.0) >= 0.72:
        return "medio"
    return "bajo"


def _normalize_reconciliation_hint(value: str) -> str:
    normalized = str(value or "").strip().lower()
    aliases = {
        "localized_reprocess": "localized_reconciliation",
        "structural_reprocess": "structural_reconciliation",
        "apply_reprocess": "apply_reconciliation",
    }
    return aliases.get(normalized, normalized)


def _reconciliation_hint(record: UncertaintyBacklogRecord) -> str:
    payload = record.payload if isinstance(record.payload, dict) else {}
    explicit = _normalize_reconciliation_hint(
        str(payload.get("reconciliation_decision") or payload.get("reprocess_decision") or "")
    )
    if explicit in {"none", "no_material_impact"}:
        return "ninguna"
    if explicit in {"localized_reconciliation", "structural_reconciliation", "delegated_to_implementation"}:
        return explicit
    affected_count = len(record.affected_deliverable_keys or [])
    if affected_count >= 3:
        return "structural_reconciliation"
    if affected_count > 0:
        return "localized_reconciliation"
    return "ninguna"


def _resolution_moment(record: UncertaintyBacklogRecord) -> str:
    target_stage = str(record.target_stage or "").strip().lower()
    if target_stage in {"acp", "package", "implementation", "implementation_questions", "construction"}:
        return "ACP o implementacion"
    if target_stage:
        return target_stage
    if _decision_status_label(record) == "respondido":
        return "Ya respondida en Blueprint Pro"
    return "Revision del Blueprint Pro"


def _owner_hint(record: UncertaintyBacklogRecord) -> str:
    payload = record.payload if isinstance(record.payload, dict) else {}
    owner = str(payload.get("target_owner") or payload.get("owner") or "").strip()
    if owner:
        return owner
    if _resolution_moment(record) == "ACP o implementacion":
        return "Product owner + implementador"
    return "Product owner"


def _why_later(record: UncertaintyBacklogRecord) -> str:
    target_stage = str(record.target_stage or "").strip().lower()
    if target_stage in {"acp", "package", "implementation", "implementation_questions", "construction"}:
        return (
            "Depende de contexto operativo, credenciales, stack final, owner tecnico o decisiones "
            "que se confirman mejor durante construccion."
        )
    if _decision_status_label(record) == "inferido":
        return "LAB uso un supuesto trazable para mantener continuidad sin bloquear el flujo actual."
    return "Se conserva para trazabilidad y revision del usuario."


def build_blueprint_delegated_decisions_document(db: Session, *, snapshot: SessionSnapshot) -> str:
    records = [
        record
        for record in _iter_uncertainty_backlog(db, snapshot=snapshot)
        if str(record.status or "").strip().lower() not in {"superseded"}
    ]
    title = snapshot.session.title or "Blueprint Profesional"
    lines = [
        "# Decisiones Delegadas y Supuestos de Implementacion",
        "",
        f"Proyecto: {title}",
        f"Sesion: `{snapshot.session.id}`",
        "",
        "Este documento explica las preguntas, supuestos y decisiones que LAB conservo para no bloquear el Blueprint Pro.",
        "No ejecuta reconciliaciones ni modifica entregables; solo documenta trazabilidad, recomendacion e impacto.",
        "",
        "## Resumen",
        "",
    ]
    if not records:
        lines.extend(
            [
                "- No hay decisiones delegadas, preguntas abiertas ni supuestos registrados en el backlog.",
                "- El Blueprint Pro puede revisarse sin pendientes heredados desde Free o Premium.",
                "",
            ]
        )
    else:
        status_counts: dict[str, int] = {}
        for record in records:
            label = _decision_status_label(record)
            status_counts[label] = status_counts.get(label, 0) + 1
        for label in sorted(status_counts):
            lines.append(f"- {label}: {status_counts[label]}")
        lines.append("")

    lines.extend(
        [
            "## Politica de uso",
            "",
            "- Responder una pregunta guarda la decision y calcula impacto.",
            "- Reconciliar entregables requiere una accion explicita del usuario.",
            "- Delegar al ACP no reabre fases aprobadas ni crea jobs ocultos.",
            "- Las decisiones delegadas deben viajar al ACP para resolverse en el momento correcto de implementacion.",
            "",
        ]
    )

    if records:
        lines.extend(["## Items", ""])
        for index, record in enumerate(records, start=1):
            source_refs = record.source_refs or []
            affected = record.affected_deliverable_keys or []
            dependency_keys = record.dependency_keys or []
            recommendation = record.suggested_answer or record.assumed_answer or "Mantener como decision pendiente con owner asignado."
            assumption = record.assumed_answer or "No se aplico supuesto automatico adicional."
            lines.extend(
                [
                    f"### {index}. {record.title or record.uncertainty_key}",
                    "",
                    f"- Clave: `{record.uncertainty_key}`",
                    f"- Estado: `{_decision_status_label(record)}`",
                    f"- Origen: `{record.source_stage or 'unknown'}`",
                    f"- Momento recomendado: `{_resolution_moment(record)}`",
                    f"- Owner sugerido: {_owner_hint(record)}",
                    f"- Nivel de impacto: `{_impact_level(record)}`",
                    f"- Reconciliacion sugerida: `{_reconciliation_hint(record)}`",
                    f"- Pregunta/decision: {record.description or record.title or record.uncertainty_key}",
                    f"- Recomendacion LAB: {recommendation}",
                    f"- Supuesto usado: {assumption}",
                    f"- Por que se resuelve despues: {_why_later(record)}",
                    f"- Impacto si cambia: {record.impact or 'Puede ajustar alcance, controles, integraciones o artefactos relacionados.'}",
                    f"- Entregables afectados: {', '.join(affected) if affected else 'Ninguno identificado'}",
                    f"- Dependencias: {', '.join(dependency_keys) if dependency_keys else 'No registradas'}",
                    f"- Fuentes: {', '.join(source_refs) if source_refs else 'Backlog de incertidumbres LAB'}",
                    "",
                ]
            )
    return "\n".join(lines)


def _rendering_zip_path(diagram_key: str, rendering_key: str) -> str:
    extension = _RENDERING_EXTENSIONS.get(rendering_key)
    if extension:
        return f"Blueprint/diagrams/{diagram_key}/{diagram_key}.{extension}"
    return f"Blueprint/diagrams/{diagram_key}/{diagram_key}-{_safe_name(rendering_key, 'rendering')}.txt"


def build_blueprint_zip(
    db: Session,
    *,
    snapshot: SessionSnapshot,
    preview: ACPPreview,
    overview_markdown: str,
) -> bytes:
    generated_at = _stable_generated_at(snapshot)
    canonical = build_canonical_export_document(
        snapshot,
        "blueprint-core.v1",
        generated_at=generated_at,
    )

    files: dict[str, bytes] = {
        "Blueprint/README.md": normalize_text_document(overview_markdown).encode("utf-8"),
        "Blueprint/contracts/blueprint-core.v1.json": serialize_json_document(canonical.payload).encode("utf-8"),
        "Blueprint/governance/decisiones-delegadas-y-supuestos.md": normalize_text_document(
            build_blueprint_delegated_decisions_document(db, snapshot=snapshot)
        ).encode("utf-8"),
    }

    for artifact in _iter_latest_artifacts(snapshot):
        zip_path = _artifact_zip_path(artifact)
        if not zip_path or zip_path in files:
            continue
        files[zip_path] = normalize_text_document(artifact.content_text or "").encode("utf-8")

    for deliverable in snapshot.blueprint.delivery_package.deliverables if snapshot.blueprint else []:
        zip_path = f"Blueprint/deliverables/{_safe_name(deliverable.key, 'deliverable')}.md"
        if zip_path in files:
            continue
        files[zip_path] = normalize_text_document(deliverable.content_markdown or "").encode("utf-8")

    for version in _iter_latest_diagram_versions(db, session_id=snapshot.session.id):
        diagram_key = _safe_name(version.diagram_key, "diagram")
        files[f"Blueprint/diagrams/{diagram_key}/diagram-model.v1.json"] = serialize_json_document(
            version.diagram_model
        ).encode("utf-8")
        files[f"Blueprint/diagrams/{diagram_key}/diagram-quality.v1.json"] = serialize_json_document(
            version.quality_report
        ).encode("utf-8")
        renderings = version.renderings if isinstance(version.renderings, dict) else {}
        for rendering_key, rendering_value in sorted(renderings.items()):
            if not rendering_value:
                continue
            if rendering_key == "presentation":
                zip_path = f"Blueprint/diagrams/{diagram_key}/diagram-presentation.v1.json"
            else:
                zip_path = _rendering_zip_path(diagram_key, str(rendering_key))
            if isinstance(rendering_value, str):
                files[zip_path] = normalize_text_document(rendering_value).encode("utf-8")
            else:
                files[zip_path] = serialize_json_document(rendering_value).encode("utf-8")

    files["Blueprint/manifest.json"] = serialize_json_document(
        {
            "contract_version": "blueprint-professional-zip.v1",
            "session_id": str(snapshot.session.id),
            "workspace_id": str(snapshot.session.workspace_id) if snapshot.session.workspace_id is not None else None,
            "blueprint_version_number": preview.blueprint_version_number,
            "generated_at": generated_at,
            "files": sorted(files),
        }
    ).encode("utf-8")

    buffer = BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.writestr(_zip_info(path), files[path])
    return buffer.getvalue()
