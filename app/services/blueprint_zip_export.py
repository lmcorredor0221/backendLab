from __future__ import annotations

import re
from html import escape
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


_STAGE_ORDER = {
    "discover": 10,
    "discovery": 10,
    "define": 20,
    "definition": 20,
    "design": 30,
    "architecture": 30,
    "tools": 40,
    "memory": 50,
    "knowledge": 50,
    "validation": 60,
    "validate": 60,
    "estimate": 70,
    "estimation": 70,
    "governance": 80,
    "diagrams": 90,
}

_STAGE_LABELS = {
    "discover": "Problema y contexto",
    "define": "Objetivo y requisitos",
    "design": "Solucion propuesta",
    "tools": "Herramientas",
    "memory": "Memoria y conocimiento",
    "validation": "Validaciones",
    "estimate": "Estimacion",
    "governance": "Decisiones y supuestos",
    "diagrams": "Diagramas",
}

_STAGE_NARRATIVE = {
    "discover": {
        "why": "Alinea el problema, los usuarios afectados y el costo operativo actual antes de disenar.",
        "next": "Revisa como LAB tradujo el problema en objetivos y requisitos.",
    },
    "define": {
        "why": "Convierte la necesidad inicial en objetivos, alcance, criterios de aceptacion y restricciones trazables.",
        "next": "Avanza a la solucion propuesta para entender el patron agentico elegido.",
    },
    "design": {
        "why": "Explica la arquitectura, el patron de agente y las responsabilidades principales de la solucion.",
        "next": "Revisa las herramientas necesarias para operar esa arquitectura.",
    },
    "tools": {
        "why": "Define el set minimo de capacidades, integraciones y contratos requeridos para ejecutar el agente.",
        "next": "Revisa que memoria y conocimiento necesita conservar y recuperar el agente.",
    },
    "memory": {
        "why": "Describe como se conserva contexto, decisiones, fuentes y conocimiento operativo para evitar perdida de informacion.",
        "next": "Revisa las validaciones, estimaciones y siguientes pasos.",
    },
    "validation": {
        "why": "Resume controles, riesgos y criterios usados para revisar si el Blueprint puede avanzar.",
        "next": "Revisa el impacto esperado en tiempo, costo y esfuerzo.",
    },
    "estimate": {
        "why": "Traduce el diseno en esfuerzo, costo, ROI y escenarios de construccion.",
        "next": "Cierra con decisiones delegadas y preparacion hacia ACP o implementacion.",
    },
    "governance": {
        "why": "Muestra preguntas, decisiones, supuestos e inferencias que deben revisarse sin bloquear el Blueprint.",
        "next": "Usa estos pendientes como entrada para Blueprint Pro, ACP o implementacion.",
    },
}


def _canonical_stage(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("_", "-")
    if "discover" in normalized:
        return "discover"
    if "definition" in normalized or "define" in normalized:
        return "define"
    if "design" in normalized or "architecture" in normalized or "contract" in normalized:
        return "design"
    if "tool" in normalized:
        return "tools"
    if "memory" in normalized or "knowledge" in normalized:
        return "memory"
    if "validation" in normalized or "validate" in normalized:
        return "validation"
    if "estimate" in normalized or "estimation" in normalized:
        return "estimate"
    if "governance" in normalized or "decision" in normalized:
        return "governance"
    if "diagram" in normalized:
        return "diagrams"
    return "governance"


def _stage_from_path(path: str) -> str:
    parts = [part.lower() for part in PurePosixPath(path).parts]
    for part in parts:
        stage = _canonical_stage(part)
        if stage != "governance" or part in {"governance", "decisions", "decision"}:
            return stage
    return "governance"


def _path_title(path: str) -> str:
    stem = PurePosixPath(path).name
    if "." in stem:
        stem = ".".join(stem.split(".")[:-1])
    return stem.replace("-", " ").replace("_", " ").strip().title() or "Archivo Blueprint"


def _diagram_item_details(diagram_key: str, filename: str) -> tuple[str, str, str]:
    """
    Retorna (stage, title, description) canonicos para un archivo de diagrama.
    """
    try:
        from app.services.diagram_center.registry_service import load_diagram_registry

        registry = load_diagram_registry()
        entry = registry.entries.get(diagram_key)
    except Exception:
        entry = None

    base_title = entry.title if entry else _path_title(diagram_key)
    base_stage = _canonical_stage(entry.stage if entry else "design")
    base_desc = entry.description if entry else f"Diagrama {base_title}."

    if filename.endswith(".svg"):
        return base_stage, f"{base_title} (SVG)", f"Renderizado visual vectorial SVG para {base_title}."
    if filename.endswith(".mmd"):
        return base_stage, f"{base_title} (Mermaid)", f"Especificacion Mermaid del diagrama {base_title}."
    if filename.endswith(".puml"):
        return base_stage, f"{base_title} (PlantUML)", f"Especificacion PlantUML del diagrama {base_title}."
    if filename.endswith(".bpmn.xml"):
        return base_stage, f"{base_title} (BPMN 2.0 XML)", f"Definicion BPMN 2.0 XML ejecutable de {base_title}."
    if filename == "diagram-model.v1.json":
        return base_stage, f"{base_title} (Modelo Semantico)", f"Modelo canonico estructurado JSON de {base_title}."
    if filename == "diagram-quality.v1.json":
        return base_stage, f"{base_title} (Reporte de Calidad)", f"Metricas de calidad, conectividad y validacion de {base_title}."
    if filename == "diagram-presentation.v1.json":
        return base_stage, f"{base_title} (Presentacion Visual)", f"Configuracion de layout y presentacion de {base_title}."
    if filename.endswith("-json.txt"):
        return base_stage, f"{base_title} (Raw JSON)", f"Representacion textual JSON de {base_title}."

    return base_stage, f"{base_title} ({_path_title(filename)})", base_desc


def _navigation_item(
    *,
    path: str,
    title: str = "",
    stage: str = "",
    item_type: str = "artifact",
    description: str = "",
) -> dict:
    parts = PurePosixPath(path).parts
    relative_path = path.removeprefix("Blueprint/").lstrip("/")

    resolved_stage = stage
    resolved_title = title
    resolved_desc = description

    if not resolved_title or not resolved_stage:
        if len(parts) >= 4 and parts[1] == "diagrams":
            diagram_key = parts[2]
            filename = parts[3]
            d_stage, d_title, d_desc = _diagram_item_details(diagram_key, filename)
            resolved_stage = resolved_stage or d_stage
            resolved_title = resolved_title or d_title
            resolved_desc = resolved_desc or d_desc
        elif len(parts) >= 4 and parts[1] == "commercial" and parts[2] == "diagrams":
            filename = parts[3]
            resolved_stage = resolved_stage or "design"
            resolved_title = resolved_title or f"{_path_title(filename)} (Mermaid)"
            resolved_desc = resolved_desc or f"Diagrama comercial Mermaid de {_path_title(filename)}."
        else:
            resolved_stage = resolved_stage or _stage_from_path(path)
            resolved_title = resolved_title or _path_title(path)
            resolved_desc = resolved_desc or f"Archivo de soporte para {_STAGE_LABELS.get(_canonical_stage(resolved_stage), 'Blueprint')}."

    canonical_stage = _canonical_stage(resolved_stage)
    item_id = re.sub(r"[^a-zA-Z0-9._:-]+", "-", relative_path).strip("-").lower()
    return {
        "id": item_id,
        "title": resolved_title,
        "stage": canonical_stage,
        "type": item_type,
        "description": resolved_desc,
        "path": path,
        "relative_path": relative_path,
        "order": _STAGE_ORDER.get(canonical_stage, 100),
        "related_artifacts": [],
        "related_diagrams": [],
    }


def _build_navigation_decisions(db: Session, *, snapshot: SessionSnapshot) -> list[dict]:
    decisions: list[dict] = []
    for record in _iter_uncertainty_backlog(db, snapshot=snapshot):
        if str(record.status or "").strip().lower() == "superseded":
            continue
        decisions.append(
            {
                "id": f"uncertainty:{record.uncertainty_key}",
                "title": record.title or record.uncertainty_key,
                "stage": _canonical_stage(record.source_stage),
                "status": _decision_status_label(record),
                "disposition": str(record.disposition or ""),
                "impact_level": _impact_level(record),
                "recommendation": record.suggested_answer or record.assumed_answer or "",
                "why_later": _why_later(record),
                "affected_items": list(record.affected_deliverable_keys or []),
                "resolution_moment": _resolution_moment(record),
            }
        )
    return decisions


def _storyline_narrative(snapshot: SessionSnapshot, stage: str, items: list[dict], decisions: list[dict]) -> str:
    discovery = snapshot.discovery
    canvas = snapshot.canvas
    blueprint = snapshot.blueprint
    estimation = snapshot.estimation_report
    if stage == "discover" and discovery is not None:
        return discovery.problem_statement or discovery.current_process or "LAB consolido el problema y el contexto operativo inicial."
    if stage == "define" and canvas is not None:
        return canvas.user_goal or "LAB tradujo el descubrimiento en objetivos, alcance y criterios iniciales."
    if stage == "design" and blueprint is not None:
        return blueprint.narrative or f"LAB propone una arquitectura {blueprint.architecture or 'agentica'} con patron {blueprint.reasoning_pattern or 'gobernado'}."
    if stage == "tools" and blueprint is not None and blueprint.tools:
        return f"LAB identifico {len(blueprint.tools)} herramientas o capacidades necesarias para operar el agente."
    if stage == "memory" and blueprint is not None:
        return f"La estrategia de memoria propuesta es {blueprint.memory_strategy or 'memoria por sesion, decisiones y checkpoints'}."
    if stage == "estimate" and estimation is not None:
        confidence = getattr(estimation.confidence, "score", 0) if estimation.confidence else 0
        return f"LAB estimo esfuerzo, costo y escenarios de construccion con confianza {confidence}%."
    if stage == "governance":
        return f"LAB documento {len(decisions)} decisiones, supuestos o preguntas para mantener trazabilidad sin bloquear el Blueprint."
    return f"Esta seccion agrupa {len(items)} elemento(s) relacionados con {_STAGE_LABELS.get(stage, stage)}."


def _build_storyline(snapshot: SessionSnapshot, *, items: list[dict], decisions: list[dict]) -> list[dict]:
    stages = []
    all_stages = ["discover", "define", "design", "tools", "memory", "validation", "estimate", "governance", "diagrams"]
    for stage in all_stages:
        if stage == "diagrams":
            stage_items = [item for item in items if item["type"] == "diagram"]
        else:
            stage_items = [item for item in items if item["stage"] == stage]
        stage_decisions = [item for item in decisions if item["stage"] == stage]
        if not stage_items and not stage_decisions and stage not in {"discover", "define", "design", "memory", "estimate"}:
            continue
        if stage == "discover" and snapshot.discovery is None and not stage_items:
            continue
        if stage == "define" and snapshot.canvas is None and not stage_items:
            continue
        if stage == "design" and snapshot.blueprint is None and not stage_items:
            continue
        if stage == "memory" and snapshot.blueprint is None and not stage_items:
            continue
        if stage == "estimate" and snapshot.estimation_report is None and not stage_items:
            continue
        stages.append(stage)

    chapters: list[dict] = []
    for index, stage in enumerate(stages):
        if stage == "diagrams":
            stage_items = [item for item in items if item["type"] == "diagram"]
        else:
            stage_items = [item for item in items if item["stage"] == stage]
        stage_decisions = [item for item in decisions if item["stage"] == stage]
        narrative = _storyline_narrative(snapshot, stage, stage_items, stage_decisions)

        evidence_candidates = sorted(
            stage_items,
            key=lambda x: (
                0 if x["path"].endswith(".svg")
                else (1 if x["path"].endswith(".md")
                else (2 if x["path"].endswith(".json") and not x["path"].endswith("diagram-quality.v1.json")
                else 3))
            ),
        )

        chapters.append(
            {
                "id": stage,
                "title": _STAGE_LABELS.get(stage, stage.title()),
                "stage": stage,
                "narrative": narrative,
                "why_it_matters": _STAGE_NARRATIVE.get(stage, {}).get("why", "Ayuda a conectar evidencia, decisiones y entregables del Blueprint."),
                "business_value": "Permite revisar la solucion con trazabilidad y avanzar sin perder contexto.",
                "key_takeaways": [
                    f"{len(stage_items)} artefacto(s) o diagrama(s) relacionados.",
                    f"{len(stage_decisions)} decision(es), supuesto(s) o pendiente(s) relacionados.",
                ],
                "evidence_refs": [item["path"] for item in evidence_candidates[:5]],
                "related_artifacts": [item["id"] for item in stage_items if item["type"] == "artifact"],
                "related_diagrams": [item["id"] for item in stage_items if item["type"] == "diagram"],
                "next_chapter_id": stages[index + 1] if index + 1 < len(stages) else "",
                "next_question": _STAGE_NARRATIVE.get(stage, {}).get("next", "Revisa el siguiente bloque del Blueprint."),
            }
        )
    return chapters


def _build_blueprint_navigation_manifest(
    db: Session,
    *,
    snapshot: SessionSnapshot,
    preview: ACPPreview,
    files: dict[str, bytes],
    generated_at: datetime,
) -> dict:
    items = [
        _navigation_item(path=path, item_type="diagram" if path.startswith("Blueprint/diagrams/") else "artifact")
        for path in sorted(files)
        if path.startswith("Blueprint/")
        and not path.endswith("/")
        and path
        not in {
            "Blueprint/README.md",
            "Blueprint/index.html",
            "Blueprint/manifest.json",
            "Blueprint/navigation-manifest.v1.json",
        }
        and not path.startswith("Blueprint/assets/")
    ]
    decisions = _build_navigation_decisions(db, snapshot=snapshot)
    title = snapshot.session.title or "Blueprint Profesional"
    discovery = snapshot.discovery
    canvas = snapshot.canvas
    blueprint = snapshot.blueprint
    executive_summary = (
        (blueprint.narrative if blueprint is not None else "")
        or (canvas.user_goal if canvas is not None else "")
        or (discovery.desired_outcome if discovery is not None else "")
        or "Blueprint generado por Lean Agent Builder."
    )
    manifest = {
        "contract_version": "blueprint-navigation-manifest.v1",
        "package_type": "blueprint_professional",
        "session_id": str(snapshot.session.id),
        "workspace_id": str(snapshot.session.workspace_id) if snapshot.session.workspace_id is not None else None,
        "title": title,
        "generated_at": generated_at.isoformat(),
        "blueprint_version_number": preview.blueprint_version_number,
        "executive_summary": executive_summary,
        "problem_statement": discovery.problem_statement if discovery is not None else "",
        "north_star_metric": (
            discovery.mvp_definition.north_star_metric
            if discovery is not None and discovery.mvp_definition
            else (canvas.success_metric if canvas is not None else "")
        ),
        "storyline": [],
        "items": items,
        "decisions": decisions,
        "warnings": [
            decision
            for decision in decisions
            if decision.get("status") in {"delegado", "inferido", "pendiente"}
        ],
    }
    manifest["storyline"] = _build_storyline(snapshot, items=items, decisions=decisions)
    return manifest


def _build_blueprint_readme(manifest: dict, *, overview_markdown: str) -> str:
    lines = [
        f"# Blueprint Profesional - {manifest.get('title') or 'Proyecto'}",
        "",
        "Este paquete contiene el Blueprint Profesional generado por Lean Agent Builder. Su objetivo es explicar el problema, la solucion propuesta, los artefactos de soporte, los diagramas, las decisiones y los siguientes pasos para avanzar hacia ACP o implementacion.",
        "",
        "## Por donde empezar",
        "",
        "1. Abre [index.html](index.html) (`Blueprint/index.html`) para recorrer el Blueprint como una experiencia guiada.",
        "2. Usa el mapa lateral del viewer para navegar por problema, contexto, solucion, arquitectura, herramientas, memoria, validacion y estimacion.",
        "3. Consulta [Decisiones Delegadas y Supuestos](governance/decisiones-delegadas-y-supuestos.md) (`Blueprint/governance/decisiones-delegadas-y-supuestos.md`) para revisar preguntas, inferencias o decisiones trasladadas a etapas posteriores.",
        "4. Entra a los artefactos o diagramas vinculados cuando necesites detalle tecnico o evidencia.",
        "",
        "## Resumen ejecutivo",
        "",
        str(manifest.get("executive_summary") or "Blueprint generado con los artefactos disponibles."),
        "",
        "## Problema y metrica",
        "",
        f"- Problema: {manifest.get('problem_statement') or 'No documentado en el snapshot exportado.'}",
        f"- North Star: {manifest.get('north_star_metric') or 'No documentada en el snapshot exportado.'}",
        "",
        "## Recorrido recomendado",
        "",
    ]
    for chapter in manifest.get("storyline", []):
        lines.append(f"- {chapter.get('title')}: {chapter.get('why_it_matters')}")
    lines.extend(["", "## Contenido principal", ""])
    for item in manifest.get("items", []):
        rel_path = str(item.get("relative_path") or item.get("path", "")).removeprefix("Blueprint/").lstrip("/")
        title = item.get("title") or rel_path
        lines.append(f"- [{title}]({rel_path}) (`{rel_path}`)")
    warnings = manifest.get("warnings", [])
    lines.extend(["", "## Decisiones, supuestos y pendientes", ""])
    if warnings:
        lines.append(f"El paquete incluye {len(warnings)} decision(es), supuesto(s) o pendiente(s) documentados para trazabilidad.")
        for decision in warnings[:12]:
            lines.append(
                f"- {decision.get('title')} - estado `{decision.get('status')}`, momento `{decision.get('resolution_moment')}`."
            )
    else:
        lines.append("No hay decisiones, supuestos o pendientes abiertos en el manifest de navegacion.")
    lines.extend(
        [
            "",
            "## Estructura del paquete",
            "",
            "- [index.html](index.html) (`Blueprint/index.html`): viewer autocontenido para navegar el Blueprint.",
            "- [navigation-manifest.v1.json](navigation-manifest.v1.json): indice canonico usado por README y viewer.",
            "- `contracts/`: contratos canonicos del Blueprint.",
            "- [governance/decisiones-delegadas-y-supuestos.md](governance/decisiones-delegadas-y-supuestos.md): decisiones delegadas, supuestos e inferencias.",
            "- `diagrams/`: modelos, calidad y renderings de diagramas.",
            "- `deliverables/`: entregables derivados del Blueprint.",
            "",
            "## Documento profesional completo",
            "",
            normalize_text_document(overview_markdown),
        ]
    )
    return "\n".join(lines)


def _viewer_css() -> str:
    return """
:root { color-scheme: light; --ink:#101525; --muted:#596276; --line:#d8deea; --soft:#f5f7fb; --brand:#2f43bd; --accent:#2f7d52; --warn:#a05a00; }
* { box-sizing: border-box; }
body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); background:linear-gradient(135deg,#f9fbff,#eef3ff); }
a { color:inherit; }
.shell { display:grid; grid-template-columns:280px minmax(0,1fr); min-height:100vh; }
.sidebar { position:sticky; top:0; height:100vh; overflow:auto; padding:24px; border-right:1px solid var(--line); background:rgba(255,255,255,.82); backdrop-filter: blur(10px); }
.brand { font-size:12px; letter-spacing:.18em; text-transform:uppercase; font-weight:800; color:var(--brand); }
.title { margin:10px 0 4px; font-size:26px; line-height:1.05; }
.meta { color:var(--muted); font-size:12px; line-height:1.5; }
.nav { display:grid; gap:8px; margin-top:24px; }
.nav button { border:1px solid var(--line); background:white; border-radius:14px; padding:10px 12px; text-align:left; font-weight:800; cursor:pointer; }
.nav button.active { border-color:var(--brand); color:var(--brand); box-shadow:0 8px 20px rgba(47,67,189,.14); }
.main { padding:34px; }
.hero, .chapter, .card { border:1px solid var(--line); border-radius:24px; background:rgba(255,255,255,.9); box-shadow:0 16px 35px rgba(16,21,37,.07); }
.hero { padding:28px; margin-bottom:22px; }
.hero h1 { margin:0; font-size:34px; }
.hero p { max-width:920px; color:var(--muted); line-height:1.7; }
.chapter { padding:28px; }
.eyebrow { color:var(--brand); font-size:12px; font-weight:900; letter-spacing:.18em; text-transform:uppercase; }
.chapter h2 { margin:8px 0 12px; font-size:30px; }
.chapter p { color:var(--muted); line-height:1.7; }
.section-group { margin-top:24px; padding-top:18px; border-top:1px dashed var(--line); }
.section-group:first-of-type { border-top:none; padding-top:0; margin-top:14px; }
.section-header { font-size:14px; font-weight:850; color:var(--ink); margin-bottom:12px; display:flex; align-items:center; gap:8px; text-transform:uppercase; letter-spacing:.06em; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:14px; margin-top:10px; }
.card { padding:16px; }
.card small { color:var(--muted); text-transform:uppercase; letter-spacing:.12em; font-weight:800; }
.card h3 { margin:8px 0; font-size:16px; }
.card p { margin:0 0 12px; font-size:13px; }
.card a { display:inline-flex; padding:8px 10px; border-radius:12px; background:var(--soft); color:var(--brand); text-decoration:none; font-weight:800; font-size:13px; }
.takeaways { display:flex; flex-wrap:wrap; gap:8px; margin-top:16px; }
.pill { border-radius:999px; background:#eaf0ff; color:var(--brand); padding:7px 10px; font-size:12px; font-weight:800; }
.decision { border-left:4px solid var(--warn); }
.controls { display:flex; justify-content:space-between; gap:12px; margin-top:24px; }
.controls button { border:0; border-radius:14px; padding:12px 16px; background:var(--brand); color:white; font-weight:900; cursor:pointer; }
.controls button.secondary { background:white; color:var(--brand); border:1px solid var(--line); }
.search { width:100%; margin-top:18px; border:1px solid var(--line); border-radius:14px; padding:10px 12px; }
@media (max-width: 860px) { .shell { grid-template-columns:1fr; } .sidebar { position:relative; height:auto; } .main { padding:18px; } .hero h1 { font-size:28px; } }
""".strip()


def _viewer_js() -> str:
    return """
(function(){
  const data = window.BLUEPRINT_NAVIGATION_MANIFEST || {storyline:[], items:[], decisions:[]};
  const nav = document.querySelector('[data-nav]');
  const chapter = document.querySelector('[data-chapter]');
  const search = document.querySelector('[data-search]');
  let current = 0;
  function esc(value){ return String(value || '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch])); }
  function resolveHref(item){
    const raw = item.relative_path || item.path || '';
    return raw.replace(/^Blueprint\\//, '');
  }
  function linkCard(item){
    const href = resolveHref(item);
    const isSvg = href.endsWith('.svg');
    const isMd = href.endsWith('.md');
    const isJson = href.endsWith('.json');
    const isMmd = href.endsWith('.mmd') || href.endsWith('.mermaid');
    const isXml = href.endsWith('.xml');
    let badge = esc(item.type);
    let actionLabel = 'Abrir';
    if (isSvg) { badge = 'DIAGRAMA SVG'; actionLabel = 'Ver diagrama SVG'; }
    else if (isMd) { badge = 'DOCUMENTO'; actionLabel = 'Leer documento'; }
    else if (isJson) { badge = 'CONTRATO JSON'; actionLabel = 'Ver contrato JSON'; }
    else if (isMmd) { badge = 'MERMAID'; actionLabel = 'Ver Mermaid'; }
    else if (isXml) { badge = 'BPMN XML'; actionLabel = 'Ver XML BPMN'; }

    return `<article class="card">
      <small>${badge} · ${esc(item.stage)}</small>
      <h3>${esc(item.title)}</h3>
      <p>${esc(item.description)}</p>
      <a href="${esc(href)}" target="_blank" rel="noreferrer">${esc(actionLabel)}</a>
    </article>`;
  }
  function decisionCard(item){
    return `<article class="card decision"><small>${esc(item.status)} · ${esc(item.impact_level)}</small><h3>${esc(item.title)}</h3><p>${esc(item.recommendation || item.why_later)}</p><p><strong>Momento:</strong> ${esc(item.resolution_moment)}</p></article>`;
  }
  function renderSegmentedContent(artifacts, diagrams, decisionsList){
    const allItems = [...artifacts, ...diagrams];
    const svgItems = [];
    const mmdItems = [];
    const jsonItems = [];
    const mdItems = [];
    const otherItems = [];

    allItems.forEach(item => {
      const href = resolveHref(item).toLowerCase();
      const title = (item.title || '').toLowerCase();
      if (href.endsWith('.svg')) {
        svgItems.push(item);
      } else if (href.endsWith('.mmd') || href.endsWith('.mermaid') || href.endsWith('.xml') || title.includes('mermaid') || title.includes('bpmn')) {
        mmdItems.push(item);
      } else if (href.endsWith('.json')) {
        jsonItems.push(item);
      } else if (href.endsWith('.md')) {
        mdItems.push(item);
      } else {
        otherItems.push(item);
      }
    });

    let html = '';

    if (svgItems.length > 0) {
      html += `<div class="section-group">
        <div class="section-header">🎨 Diagramas Visuales Vectoriales (SVG) (${svgItems.length})</div>
        <div class="grid">${svgItems.map(linkCard).join('')}</div>
      </div>`;
    }

    if (mmdItems.length > 0) {
      html += `<div class="section-group">
        <div class="section-header">📐 Código & Especificaciones de Diagrama (Mermaid / BPMN) (${mmdItems.length})</div>
        <div class="grid">${mmdItems.map(linkCard).join('')}</div>
      </div>`;
    }

    if (jsonItems.length > 0) {
      html += `<div class="section-group">
        <div class="section-header">📜 Contratos Semánticos, Modelos & Calidad (.json) (${jsonItems.length})</div>
        <div class="grid">${jsonItems.map(linkCard).join('')}</div>
      </div>`;
    }

    if (mdItems.length > 0) {
      html += `<div class="section-group">
        <div class="section-header">📄 Documentación & Entregables (.md) (${mdItems.length})</div>
        <div class="grid">${mdItems.map(linkCard).join('')}</div>
      </div>`;
    }

    if (otherItems.length > 0) {
      html += `<div class="section-group">
        <div class="section-header">📦 Otros Artefactos (${otherItems.length})</div>
        <div class="grid">${otherItems.map(linkCard).join('')}</div>
      </div>`;
    }

    if (decisionsList && decisionsList.length > 0) {
      html += `<div class="section-group">
        <div class="section-header">⚠️ Decisiones Delegadas, Supuestos & Gobernanza (${decisionsList.length})</div>
        <div class="grid">${decisionsList.map(decisionCard).join('')}</div>
      </div>`;
    }

    return html || '<p>No hay elementos disponibles en esta sección.</p>';
  }
  function renderNav(){
    nav.innerHTML = data.storyline.map((item, index) => `<button type="button" class="${index===current?'active':''}" data-index="${index}">${esc(index+1)}. ${esc(item.title)}</button>`).join('');
    nav.querySelectorAll('button').forEach(btn => btn.addEventListener('click', () => { current = Number(btn.dataset.index || 0); render(); }));
  }
  function render(){
    const item = data.storyline[current] || data.storyline[0];
    if(!item){ chapter.innerHTML = '<p>No hay capitulos disponibles en este Blueprint.</p>'; return; }
    const artifacts = (data.items || []).filter(entry => (item.related_artifacts || []).includes(entry.id));
    const diagrams = (data.items || []).filter(entry => (item.related_diagrams || []).includes(entry.id));
    const decisions = (data.decisions || []).filter(entry => entry.stage === item.stage);
    chapter.innerHTML = `
      <div class="eyebrow">${esc(item.stage)}</div>
      <h2>${esc(item.title)}</h2>
      <p>${esc(item.narrative)}</p>
      <div class="takeaways">${(item.key_takeaways || []).map(t => `<span class="pill">${esc(t)}</span>`).join('')}</div>
      ${renderSegmentedContent(artifacts, diagrams, decisions)}
      <p style="margin-top:24px;"><strong>Por que importa:</strong> ${esc(item.why_it_matters)}</p>
      <p><strong>Siguiente mirada:</strong> ${esc(item.next_question)}</p>
      <div class="controls"><button class="secondary" type="button" data-prev>Anterior</button><button type="button" data-next>Siguiente</button></div>
    `;
    chapter.querySelector('[data-prev]').addEventListener('click', () => { current = Math.max(0, current - 1); render(); });
    chapter.querySelector('[data-next]').addEventListener('click', () => { current = Math.min(data.storyline.length - 1, current + 1); render(); });
    renderNav();
  }
  if(search){
    search.addEventListener('input', () => {
      const query = search.value.toLowerCase();
      nav.querySelectorAll('button').forEach(btn => { btn.hidden = query && !btn.textContent.toLowerCase().includes(query); });
    });
  }
  render();
})();
""".strip()


def _build_blueprint_viewer_html(manifest: dict) -> str:
    manifest_json = serialize_json_document(manifest).replace("</", "<\\/")
    chapters_count = len(manifest.get("storyline", []))
    items_count = len(manifest.get("items", []))
    warnings_count = len(manifest.get("warnings", []))

    item_map = {item["id"]: item for item in manifest.get("items", [])}
    noscript_chapters: list[str] = []
    for ch in manifest.get("storyline", []):
        ch_title = escape(str(ch.get("title") or ""))
        ch_narrative = escape(str(ch.get("narrative") or ""))
        link_items: list[str] = []
        for it_id in (ch.get("related_artifacts", []) + ch.get("related_diagrams", [])):
            it = item_map.get(it_id)
            if it:
                href = escape(str(it.get("relative_path") or it.get("path", "")).removeprefix("Blueprint/").lstrip("/"))
                title = escape(str(it.get("title") or href))
                link_items.append(f'<li><a href="{href}">{title}</a></li>')
        links_markup = f'<ul style="margin:8px 0 16px 20px; line-height:1.6;">{"".join(link_items)}</ul>' if link_items else '<p style="color:#666;">Sin archivos vinculados.</p>'
        noscript_chapters.append(f'<div style="margin-bottom:16px;"><h3 style="margin:4px 0;">{ch_title}</h3><p style="margin:4px 0; color:#444;">{ch_narrative}</p>{links_markup}</div>')

    noscript_content = "".join(noscript_chapters)

    return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(str(manifest.get('title') or 'Blueprint Profesional'))}</title>
  <link rel="stylesheet" href="assets/blueprint-viewer.css">
</head>
<body>
  <div class="shell">
    <aside class="sidebar">
      <div class="brand">Lean Agent Builder</div>
      <h1 class="title">{escape(str(manifest.get('title') or 'Blueprint Profesional'))}</h1>
      <p class="meta">Blueprint Profesional · {chapters_count} capitulos · {items_count} archivos navegables · {warnings_count} pendientes trazables</p>
      <input class="search" data-search type="search" placeholder="Buscar capitulo">
      <nav class="nav" data-nav aria-label="Mapa del Blueprint"></nav>
    </aside>
    <main class="main">
      <noscript>
        <section class="hero" style="border-left:4px solid var(--brand); margin-bottom:20px;">
          <div class="eyebrow">Modo Portable sin JavaScript</div>
          <h2>Navegación directa de entregables y diagramas</h2>
          <p>Se detectó que JavaScript está desactivado en el entorno local. Puedes acceder directamente a todos los archivos y diagramas usando los siguientes enlaces:</p>
          <div style="margin-top:16px;">
            {noscript_content}
          </div>
        </section>
      </noscript>
      <section class="hero">
        <div class="eyebrow">Blueprint Viewer</div>
        <h1>Recorre la historia de la solucion</h1>
        <p>{escape(str(manifest.get('executive_summary') or 'Este viewer guia la lectura del Blueprint exportado.'))}</p>
      </section>
      <section class="chapter" data-chapter aria-live="polite"></section>
    </main>
  </div>
  <script>window.BLUEPRINT_NAVIGATION_MANIFEST = {manifest_json};</script>
  <script src="assets/blueprint-viewer.js"></script>
</body>
</html>"""


def _rendering_zip_path(diagram_key: str, rendering_key: str) -> str:
    extension = _RENDERING_EXTENSIONS.get(rendering_key)
    if extension:
        return f"Blueprint/diagrams/{diagram_key}/{diagram_key}.{extension}"
    return f"Blueprint/diagrams/{diagram_key}/{diagram_key}-{_safe_name(rendering_key, 'rendering')}.txt"


def _validate_blueprint_package_integrity(
    files: dict[str, bytes],
    manifest: dict[str, Any],
) -> None:
    """
    Valida la integridad y portabilidad del paquete Blueprint Pro antes de comprimirlo:
    1. index.html y assets indispensables existen.
    2. Todo item referenciado en el manifest tiene un archivo fisico en files.
    3. Ningun enlace relativo o interno contiene rutas absolutas, prefijo duplicado Blueprint/,
       ni referencias a localhost o servidores externos.
    4. Todos los IDs referenciados en el storyline existen en manifest['items'].
    5. Todos los enlaces en index.html y README.md resuelven a archivos existentes en el paquete.
    """
    errors: list[str] = []

    # 1. Archivos estructurales obligatorios
    required_files = [
        "Blueprint/index.html",
        "Blueprint/README.md",
        "Blueprint/manifest.json",
        "Blueprint/navigation-manifest.v1.json",
        "Blueprint/assets/blueprint-viewer.css",
        "Blueprint/assets/blueprint-viewer.js",
    ]
    for req in required_files:
        if req not in files or not files[req]:
            errors.append(f"Archivo estructural obligatorio ausente o vacio: {req}")

    # 2. Validar cada item del manifest
    item_ids: set[str] = set()
    for item in manifest.get("items", []):
        item_id = str(item.get("id") or "")
        path = str(item.get("path") or "")
        rel_path = str(item.get("relative_path") or "").strip()
        item_ids.add(item_id)

        if not path.startswith("Blueprint/"):
            errors.append(f"El campo path del item '{item_id}' debe iniciar con 'Blueprint/': {path}")
        if not rel_path:
            errors.append(f"El item '{item_id}' carece de relative_path valido.")
        elif rel_path.startswith(("Blueprint/", "/", "\\", "http://", "https://", "file:")):
            errors.append(f"El relative_path del item '{item_id}' es invalido o no es relativo: {rel_path}")

        # Comprobar que el archivo existe en files
        if path not in files:
            errors.append(f"El item '{item_id}' referencia un path inexistente en el paquete: {path}")
        expected_full_path = f"Blueprint/{rel_path}"
        if expected_full_path not in files:
            errors.append(f"El relative_path '{rel_path}' del item '{item_id}' no resuelve a un archivo: {expected_full_path}")

    # 3. Validar storyline references
    for chapter in manifest.get("storyline", []):
        ch_id = chapter.get("id", "unknown")
        for ref_id in chapter.get("related_artifacts", []):
            if ref_id not in item_ids:
                errors.append(f"Capitulo '{ch_id}' referencia related_artifact desconocido: '{ref_id}'")
        for ref_id in chapter.get("related_diagrams", []):
            if ref_id not in item_ids:
                errors.append(f"Capitulo '{ch_id}' referencia related_diagram desconocido: '{ref_id}'")
        for ev_ref in chapter.get("evidence_refs", []):
            if ev_ref not in files:
                errors.append(f"Capitulo '{ch_id}' referencia evidence_ref inexistente: '{ev_ref}'")

    # 4. Validar enlaces en index.html
    if "Blueprint/index.html" in files:
        html_content = files["Blueprint/index.html"].decode("utf-8", errors="replace")
        for match in re.finditer(r'(?:href|src)=["\']([^"\']+)["\']', html_content):
            target = match.group(1).split("#")[0].strip()
            if not target or target.startswith(("javascript:", "mailto:", "data:")):
                continue
            if target.startswith(("http://", "https://", "//")):
                errors.append(f"index.html contiene enlace absoluto o externo no portable: '{target}'")
            elif target.startswith("Blueprint/"):
                errors.append(f"index.html contiene enlace con prefijo redundante Blueprint/: '{target}'")
            else:
                expected = f"Blueprint/{target}"
                if expected not in files:
                    errors.append(f"index.html contiene enlace roto '{target}' -> falta '{expected}'")

    # 5. Validar enlaces markdown en README.md
    if "Blueprint/README.md" in files:
        readme_content = files["Blueprint/README.md"].decode("utf-8", errors="replace")
        for match in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', readme_content):
            target = match.group(2).split("#")[0].strip()
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            if target.startswith("Blueprint/"):
                errors.append(f"README.md contiene enlace con prefijo redundante Blueprint/: '{target}'")
            else:
                expected = f"Blueprint/{target}"
                if expected not in files:
                    errors.append(f"README.md contiene enlace roto '{target}' -> falta '{expected}'")

    if errors:
        raise ValueError(
            f"Fallo de integridad en el paquete Blueprint Pro ({len(errors)} errores):\n"
            + "\n".join(f"  - {err}" for err in errors[:15])
        )


def build_blueprint_files(
    db: Session,
    *,
    snapshot: SessionSnapshot,
    preview: ACPPreview,
    overview_markdown: str,
) -> dict[str, bytes]:
    generated_at = _stable_generated_at(snapshot)
    canonical = build_canonical_export_document(
        snapshot,
        "blueprint-core.v1",
        generated_at=generated_at,
    )

    files: dict[str, bytes] = {
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

    # Limpiar cualquier clave espuria que termine en '/' o este vacia
    files = {
        path: content
        for path, content in files.items()
        if path and not path.endswith("/") and isinstance(content, (bytes, bytearray))
    }

    navigation_manifest = _build_blueprint_navigation_manifest(
        db,
        snapshot=snapshot,
        preview=preview,
        files=files,
        generated_at=generated_at,
    )
    files["Blueprint/navigation-manifest.v1.json"] = serialize_json_document(navigation_manifest).encode("utf-8")
    files["Blueprint/assets/blueprint-viewer.css"] = _viewer_css().encode("utf-8")
    files["Blueprint/assets/blueprint-viewer.js"] = _viewer_js().encode("utf-8")
    files["Blueprint/index.html"] = _build_blueprint_viewer_html(navigation_manifest).encode("utf-8")
    files["Blueprint/README.md"] = normalize_text_document(
        _build_blueprint_readme(navigation_manifest, overview_markdown=overview_markdown)
    ).encode("utf-8")

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

    # Validacion estricta de integridad y portabilidad antes de comprimir
    _validate_blueprint_package_integrity(files, navigation_manifest)
    return files


def build_blueprint_zip(
    db: Session,
    *,
    snapshot: SessionSnapshot,
    preview: ACPPreview,
    overview_markdown: str,
) -> bytes:
    files = build_blueprint_files(db, snapshot=snapshot, preview=preview, overview_markdown=overview_markdown)
    buffer = BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.writestr(_zip_info(path), files[path])
    return buffer.getvalue()
