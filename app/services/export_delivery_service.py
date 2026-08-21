from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from sqlmodel import Session, select

from app.models import (
    ACPPreview,
    CommercialAccessSnapshotV2,
    ExportCatalogItemResponse,
    ExportCatalogResponse,
    ExportJobCreateRequest,
    ExportJobRecord,
    ExportJobResponse,
    ExportJobStatus,
    SessionRecord,
    SessionSnapshot,
    UserRecord,
    utc_now,
)
from app.services.acp_zip_export import build_acp_zip
from app.services.commerce_service import record_commercial_event


@dataclass(frozen=True)
class ExportDefinition:
    key: str
    label: str
    description: str
    product_key: str
    required_capability: str
    profile: str
    content_type: str
    file_extension: str


EXPORT_DEFINITIONS: tuple[ExportDefinition, ...] = (
    ExportDefinition(
        key="blueprint_professional",
        label="Blueprint Profesional",
        description="Documento profesional con arquitectura, alcance, herramientas, memoria, estimacion y comparativas.",
        product_key="blueprint_pro",
        required_capability="blueprint.download",
        profile="professional",
        content_type="text/markdown; charset=utf-8",
        file_extension="md",
    ),
    ExportDefinition(
        key="estimation_pack",
        label="Estimation Pack",
        description="Estimacion tecnica/comercial y comparativas de construccion basadas en el Blueprint.",
        product_key="blueprint_pro",
        required_capability="export_estimation_pack",
        profile="estimation",
        content_type="application/json",
        file_extension="json",
    ),
    ExportDefinition(
        key="construction_pack",
        label="Construction Pack",
        description="Contratos y artefactos tecnicos de construccion derivados del ACP.",
        product_key="acp",
        required_capability="export_construction_pack",
        profile="construction",
        content_type="application/json",
        file_extension="json",
    ),
    ExportDefinition(
        key="prompt_pack",
        label="Prompt Pack",
        description="Prompts y playbooks preparados para herramientas agenticas.",
        product_key="acp",
        required_capability="export_prompt_pack",
        profile="prompts",
        content_type="application/json",
        file_extension="json",
    ),
    ExportDefinition(
        key="test_pack",
        label="Test Pack",
        description="Casos, rubricas y fixtures para validar el sistema agentico.",
        product_key="acp",
        required_capability="export_test_pack",
        profile="tests",
        content_type="application/json",
        file_extension="json",
    ),
    ExportDefinition(
        key="acp_portable_zip",
        label="ACP portable ZIP",
        description="Paquete tecnico portable con manifest, launcher, prompts, memoria, workflows y contratos.",
        product_key="acp",
        required_capability="export_acp_zip",
        profile="acp-portable",
        content_type="application/zip",
        file_extension="zip",
    ),
)


def _definition(artifact_kind: str) -> ExportDefinition:
    normalized = artifact_kind.strip()
    for item in EXPORT_DEFINITIONS:
        if item.key == normalized:
            return item
    raise ValueError(f"Unknown export artifact: {artifact_kind}")


def _capability_allowed(access: CommercialAccessSnapshotV2, capability: str) -> bool:
    return any(item.capability == capability and item.allowed for item in access.capabilities)


def build_export_catalog(
    *,
    record: SessionRecord,
    access: CommercialAccessSnapshotV2,
) -> ExportCatalogResponse:
    items: list[ExportCatalogItemResponse] = []
    for definition in EXPORT_DEFINITIONS:
        allowed = _capability_allowed(access, definition.required_capability)
        items.append(
            ExportCatalogItemResponse(
                key=definition.key,
                label=definition.label,
                description=definition.description,
                product_key=definition.product_key,
                required_capability=definition.required_capability,
                profile=definition.profile,
                content_type=definition.content_type,
                file_extension=definition.file_extension,
                access_state="allowed" if allowed else "locked",
                locked_reason="" if allowed else f"Requiere capability {definition.required_capability}.",
                cta_label="Generar export" if allowed else ("Adquirir ACP" if definition.product_key == "acp" else "Adquirir Blueprint Pro"),
            )
        )
    return ExportCatalogResponse(session_id=record.id, workspace_id=record.workspace_id, items=items)


def _safe_slug(value: str, default: str = "export") -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    return (normalized or default)[:120]


def _storage_root() -> Path:
    return Path(__file__).resolve().parents[2] / "runtime" / "export-delivery"


def _storage_path(job: ExportJobRecord) -> Path:
    return _storage_root() / str(job.workspace_id) / str(job.session_id) / str(job.id) / job.file_name


def _write_export_bytes(job: ExportJobRecord, payload: bytes) -> None:
    path = _storage_path(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _file_refs(preview: ACPPreview, *, domain: str | None = None, prefix: str | None = None) -> list[dict[str, Any]]:
    refs = []
    for item in preview.files:
        if domain and item.domain != domain:
            continue
        if prefix and not item.path.startswith(prefix):
            continue
        refs.append(
            {
                "path": item.path,
                "domain": item.domain,
                "title": item.title,
                "format": item.format,
                "status": item.status,
                "hash": item.content_hash,
                "content": item.content_text,
            }
        )
    return refs


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")


def _format_currency(value: Any) -> str:
    try:
        numeric = float(value)
        return f"${numeric:,.2f} COP"
    except (TypeError, ValueError):
        return str(value or "$0.00 COP")


def _blueprint_markdown(snapshot: SessionSnapshot, preview: ACPPreview) -> bytes:
    discovery = snapshot.discovery
    canvas = snapshot.canvas
    blueprint = snapshot.blueprint
    estimation = snapshot.estimation_report
    title = snapshot.session.title or "Agente Inteligente"
    version_num = preview.blueprint_version_number or (snapshot.blueprint_versions[-1].version_number if snapshot.blueprint_versions else 1)

    problem = discovery.problem_statement if discovery else "No disponible"
    current_process = discovery.current_process if discovery else "No disponible"
    current_user = discovery.current_user if discovery else (canvas.agent_profile.primary_user if canvas and canvas.agent_profile else "Usuario Operativo")
    desired = discovery.desired_outcome if discovery else "No disponible"
    north_star = (
        discovery.mvp_definition.north_star_metric
        if discovery and discovery.mvp_definition
        else (canvas.success_metric if canvas else "Optimización operativa")
    )
    time_spent = discovery.operational_baseline.current_time_spent if discovery and discovery.operational_baseline else "No especificado"
    cost_spent = discovery.operational_baseline.current_cost if discovery and discovery.operational_baseline else "No especificado"
    frequent_errors = discovery.operational_baseline.frequent_errors if discovery and discovery.operational_baseline else []
    
    in_scope = canvas.mvp_scope if canvas else []
    out_of_scope = canvas.out_of_scope if canvas else []
    non_delegable = discovery.mvp_definition.non_delegable_decisions if discovery and discovery.mvp_definition else []
    constraints = discovery.constraints if discovery else []
    primary_risk = canvas.primary_risk if canvas else "Riesgo operativo general"

    arch = blueprint.architecture if blueprint else "supervisor_with_subagents"
    reasoning = blueprint.reasoning_pattern if blueprint else "Plan-and-Execute"
    memory = blueprint.memory_strategy if blueprint else "session_and_checkpoints"
    tools = blueprint.tools if blueprint and blueprint.tools else []
    guardrails = blueprint.guardrails if blueprint else []
    narrative = blueprint.narrative if blueprint and blueprint.narrative else "Arquitectura agéntica estructurada según la metodología Lean."

    lines = [
        f"# Master Specification Document: {title}",
        "",
        "> **Documento Técnico-Comercial de Blueprint Profesional**  ",
        f"> **Sesión ID:** `{snapshot.session.id}` | **Versión Blueprint:** `v{version_num}` | **Fecha:** `{utc_now().strftime('%Y-%m-%d %H:%M:%S UTC')}`  ",
        f"> **Usuario Objetivo:** {current_user} | **Métrica North Star:** {north_star}",
        "",
        "---",
        "",
        "## 1. Diagnóstico de Negocio y Alineación Estratégica",
        "",
        f"### 1.1 Declaración del Problema",
        f"{problem}",
        "",
        f"### 1.2 Línea Base Operativa Actual",
        f"- **Proceso Actual:** {current_process}",
        f"- **Tiempo Operativo Invertido:** {time_spent}",
        f"- **Costo Operativo Estimado:** {cost_spent}",
        "",
        "### 1.3 Fricciones y Errores Frecuentes",
    ]
    if frequent_errors:
        for err in frequent_errors:
            lines.append(f"- {err}")
    else:
        lines.append("- Reprocesos operativos derivados de gestión manual y dispersión de información.")

    lines.extend(
        [
            "",
            "### 1.4 Resultado Deseado e Impacto Esperado",
            f"{desired}",
            "",
            "---",
            "",
            "## 2. Delimitación del Alcance Operativo",
            "",
            "### 2.1 En Alcance (MVP Construible)",
        ]
    )
    if in_scope:
        for item in in_scope:
            lines.append(f"- **[IN]** {item}")
    else:
        lines.append("- Flujo principal de atención y resolución operativa.")

    lines.extend(
        [
            "",
            "### 2.2 Fuera de Alcance (Exclusiones Explícitas)",
        ]
    )
    if out_of_scope:
        for item in out_of_scope:
            lines.append(f"- **[OUT]** {item}")
    else:
        lines.append("- Integraciones complejas no priorizadas en la primera fase.")

    lines.extend(
        [
            "",
            "### 2.3 Reglas de Negocio y Restricciones No Delegables",
        ]
    )
    if non_delegable:
        for item in non_delegable:
            lines.append(f"- ⚠️ **Decisión No Delegable:** {item} *(Requiere Human-in-the-Loop)*")
    if constraints:
        for item in constraints:
            lines.append(f"- 🔒 **Restricción:** {item}")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Arquitectura Agéntica y Especificación Técnica",
            "",
            f"### 3.1 Topología y Modelo Cognitivo",
            f"- **Topología de Arquitectura:** `{arch}`",
            f"- **Patrón de Razonamiento:** `{reasoning}`",
            f"- **Estrategia de Memoria:** `{memory}`",
            "",
            "### 3.2 Rationale de Arquitectura",
            f"{narrative}",
            "",
            "### 3.3 Catálogo de Herramientas e Integraciones",
            "| Herramienta | Propósito | Requiere Aprobación | Side Effects | Entradas | Salidas |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    if tools:
        for tool in tools:
            req_app = "Sí (HITL)" if tool.requires_approval else "No"
            side_eff = "Sí" if tool.has_side_effects else "No"
            inp = ", ".join(tool.inputs[:2]) if tool.inputs else "none"
            outp = ", ".join(tool.outputs[:2]) if tool.outputs else "none"
            lines.append(f"| `{tool.name}` | {tool.purpose} | {req_app} | {side_eff} | {inp} | {outp} |")
    else:
        lines.append("| `standard_lookup` | Consulta de contexto y validación | No | No | query | result |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Análisis Económico, Estimación y Comparativa de Construcción",
            "",
        ]
    )

    if estimation is not None:
        trad_hours = estimation.traditional.estimated_hours_total
        trad_cost = _format_currency(estimation.traditional.estimated_cost)
        agent_hours = estimation.agentic.estimated_hours_total
        agent_cost = _format_currency(estimation.agentic.estimated_cost)
        savings = _format_currency(estimation.agentic.net_savings_vs_traditional)
        pct_savings = f"{estimation.agentic.percentage_savings_vs_traditional:.1f}%" if estimation.agentic.percentage_savings_vs_traditional else "Calculado"
        aut_cov = f"{estimation.agentic.automation_coverage_percent:.0f}%" if estimation.agentic.automation_coverage_percent else "75%"
        sup_hours = f"{estimation.agentic.human_supervision_hours:.1f} hrs" if estimation.agentic.human_supervision_hours else "15 hrs"
        conf_score = f"{estimation.confidence.score * 100:.0f}%" if estimation.confidence else "85%"
        conf_label = estimation.confidence.label if estimation.confidence else "Alta"
        band = f"±{estimation.confidence.uncertainty_band_percent:.0f}%" if estimation.confidence and estimation.confidence.uncertainty_band_percent else "±15%"

        lines.extend(
            [
                "### 4.1 Cuadro Comparativo de Esfuerzo e Inversión",
                "",
                "| Dimensión de Análisis | Desarrollo Tradicional | Construcción Agéntica LEAN | Diferencial / Impacto |",
                "| :--- | :--- | :--- | :--- |",
                f"| **Esfuerzo Total de Construcción** | {trad_hours:.1f} horas | **{agent_hours:.1f} horas** | Aceleración de entrega |",
                f"| **Costo Total Estimado** | {trad_cost} | **{agent_cost}** | **Ahorro Neto: {savings} ({pct_savings})** |",
                f"| **Grado de Automatización** | 0% (Manual) | **{aut_cov}** | Reducción de sobrecarga |",
                f"| **Supervisión Humana Requerida** | N/A | **{sup_hours}** | Control de calidad gobernado |",
                f"| **Banda de Incertidumbre / Confianza** | ±35% a ±50% | **{band} ({conf_label} - {conf_score})** | Alta certidumbre de alcance |",
                "",
                "### 4.2 Escenarios de Construcción Evaluados",
                "| Escenario | Horas Estimadas | Costo Referencial | Ahorro vs Tradicional |",
                "| :--- | :--- | :--- | :--- |",
            ]
        )
        if estimation.construction_scenarios:
            for sc in estimation.construction_scenarios:
                lines.append(f"| **{sc.scenario_name}** | {sc.estimated_hours:.1f} hrs | {_format_currency(sc.estimated_cost)} | {_format_currency(sc.savings_vs_traditional)} |")
        else:
            lines.append(f"| **MVP Agéntico Base** | {agent_hours:.1f} hrs | {agent_cost} | {savings} |")

        if estimation.assumptions:
            lines.extend(["", "### 4.3 Supuestos de Estimación", *[f"- {item}" for item in estimation.assumptions]])
        if estimation.risk_drivers:
            lines.extend(["", "### 4.4 Drivers de Riesgo de Construcción", *[f"- ⚠️ {item}" for item in estimation.risk_drivers]])
    else:
        lines.extend(
            [
                "### 4.1 Estimación de Construcción",
                "La estimación detallada se calculará al ejecutar el módulo de estimación económica Lean.",
            ]
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 5. Gobernanza de Seguridad y Human-in-the-Loop (HITL)",
            "",
            "### 5.1 Guardrails Activos de Producción",
        ]
    )
    if guardrails:
        for g in guardrails:
            lines.append(f"- 🛡️ {g}")
    else:
        lines.append("- 🛡️ Validación estricta de veracidad y prohibición de inferencias no fundamentadas.")
        lines.append("- 🛡️ Intercepción obligatoria para acciones con efectos colaterales.")

    lines.extend(
        [
            "",
            f"### 5.2 Riesgo Crítico y Protocolo de Mitigación",
            f"- **Riesgo:** {primary_risk}",
            f"- **Protocolo:** Monitoreo continuo frente a la métrica North Star (`{north_star}`) y derivación controlada ante excepciones.",
            "",
            "---",
            "",
            "## 6. Roadmap de Evolución y Siguientes Pasos",
            "",
            "1. **MVP 1 (Inmediato):** Despliegue del agente con el alcance MVP aprobado, herramientas mínimas y gates humanos gobernados.",
            "2. **MVP 2 (Validación & Calidad):** Incorporación de datasets de evaluación y ampliación de contexto documental.",
            "3. **MVP 3 (Escalado Autónomo):** Automatización extendida y conexión con sistemas core bajo observabilidad en tiempo real.",
            "",
            "---",
            f"*Documento generado automáticamente por Lean Agent Builder para la sesión `{snapshot.session.id}`.*",
        ]
    )

    return "\n".join(lines).encode("utf-8")


def _payload_for_definition(definition: ExportDefinition, *, snapshot: SessionSnapshot, preview: ACPPreview) -> bytes:
    if definition.key == "acp_portable_zip":
        return build_acp_zip(preview)
    if definition.key == "blueprint_professional":
        return _blueprint_markdown(snapshot, preview)
    if definition.key == "estimation_pack":
        return _json_bytes(
            {
                "contract_version": "estimation-pack.v1",
                "session_id": str(snapshot.session.id),
                "blueprint_version_number": preview.blueprint_version_number,
                "estimation": snapshot.estimation_report.model_dump(mode="json") if snapshot.estimation_report else None,
            }
        )
    if definition.key == "prompt_pack":
        return _json_bytes(
            {
                "contract_version": "prompt-pack.v1",
                "session_id": str(snapshot.session.id),
                "files": _file_refs(preview, prefix="ACP/prompts/"),
            }
        )
    if definition.key == "test_pack":
        return _json_bytes(
            {
                "contract_version": "test-pack.v1",
                "session_id": str(snapshot.session.id),
                "files": _file_refs(preview, domain="evaluation"),
            }
        )
    return _json_bytes(
        {
            "contract_version": "construction-pack.v1",
            "session_id": str(snapshot.session.id),
            "blueprint_version_number": preview.blueprint_version_number,
            "readiness": preview.construction_readiness.model_dump(mode="json"),
            "validation": preview.validation.model_dump(mode="json"),
            "files": [
                item
                for item in _file_refs(preview)
                if item["path"].startswith(("ACP/runtime/", "ACP/tools/", "ACP/memory/", "ACP/workflows/", "ACP/adapters/"))
            ],
        }
    )


def _serialize_job(job: ExportJobRecord) -> ExportJobResponse:
    download_url = ""
    if job.status == ExportJobStatus.ready:
        download_url = f"/api/v1/sessions/{job.session_id}/exports/jobs/{job.id}/download"
    return ExportJobResponse(
        id=job.id,
        workspace_id=job.workspace_id,
        session_id=job.session_id,
        product_key=job.product_key,
        artifact_kind=job.artifact_kind,
        profile=job.profile,
        status=job.status,
        content_type=job.content_type,
        file_name=job.file_name,
        checksum_sha256=job.checksum_sha256,
        size_bytes=job.size_bytes,
        download_url=download_url,
        expires_at=job.expires_at,
        error_message=job.error_message,
        metadata=job.metadata_payload,
        created_at=job.created_at,
        updated_at=job.updated_at,
        completed_at=job.completed_at,
    )


def _find_export_job(db: Session, *, record: SessionRecord, job_id) -> ExportJobRecord:
    job = db.get(ExportJobRecord, job_id)
    if job is None or job.workspace_id != record.workspace_id or job.session_id != record.id:
        raise LookupError("Export job not found")
    return job


def _conformance_errors(definition: ExportDefinition, preview: ACPPreview) -> list[str]:
    if definition.key != "acp_portable_zip":
        return []
    errors: list[str] = []
    if not preview.validation.can_export_zip:
        errors.append("ACP conformance is not ready for portable ZIP export.")
    if preview.construction_readiness.blocking_gaps:
        errors.append("ACP has blocking implementation gaps.")
    if preview.construction_readiness.open_questions:
        errors.append("ACP has open implementation questions.")
    return errors


def _run_export_generation(
    db: Session,
    *,
    job: ExportJobRecord,
    definition: ExportDefinition,
    current_user: UserRecord,
    snapshot: SessionSnapshot,
    preview: ACPPreview,
) -> None:
    conformance_errors = _conformance_errors(definition, preview)
    if conformance_errors:
        raise ValueError("; ".join(conformance_errors))

    export_bytes = _payload_for_definition(definition, snapshot=snapshot, preview=preview)
    job.checksum_sha256 = hashlib.sha256(export_bytes).hexdigest()
    job.size_bytes = len(export_bytes)
    job.status = ExportJobStatus.ready
    job.error_message = ""
    job.completed_at = utc_now()
    job.updated_at = job.completed_at
    _write_export_bytes(job, export_bytes)
    record_commercial_event(
        db,
        workspace_id=job.workspace_id,
        session_id=job.session_id,
        user_id=current_user.id,
        event_key="export_job_ready",
        product_key=definition.product_key,
        source="export_delivery",
        metadata={
            "artifact_kind": definition.key,
            "job_id": str(job.id),
            "checksum_sha256": job.checksum_sha256,
            "size_bytes": job.size_bytes,
            "retry_count": int(job.metadata_payload.get("retry_count", 0) or 0),
        },
    )


def create_export_job(
    db: Session,
    *,
    record: SessionRecord,
    current_user: UserRecord,
    access: CommercialAccessSnapshotV2,
    snapshot: SessionSnapshot,
    preview: ACPPreview,
    payload: ExportJobCreateRequest,
) -> ExportJobResponse:
    definition = _definition(payload.artifact_kind)
    if not _capability_allowed(access, definition.required_capability):
        raise PermissionError(f"Export requires capability {definition.required_capability}.")

    profile = payload.profile.strip() or definition.profile
    idempotency_key = payload.idempotency_key.strip() or f"{record.id}:{definition.key}:{profile}"
    existing = db.exec(
        select(ExportJobRecord).where(
            ExportJobRecord.workspace_id == record.workspace_id,
            ExportJobRecord.session_id == record.id,
            ExportJobRecord.idempotency_key == idempotency_key,
        )
    ).first()
    if existing is not None:
        return _serialize_job(existing)

    now = utc_now()
    file_name = f"{_safe_slug(record.title)}-{definition.key}.{definition.file_extension}"
    job = ExportJobRecord(
        workspace_id=record.workspace_id,
        session_id=record.id,
        user_id=current_user.id,
        product_key=definition.product_key,
        profile=profile,
        artifact_kind=definition.key,
        status=ExportJobStatus.running,
        idempotency_key=idempotency_key,
        content_type=definition.content_type,
        file_name=file_name,
        storage_key=f"{record.workspace_id}/{record.id}/{idempotency_key}/{file_name}",
        expires_at=now + timedelta(hours=24),
        metadata_payload={
            "contract_version": "export-job.v1",
            "required_capability": definition.required_capability,
            "blueprint_version_number": preview.blueprint_version_number,
        },
    )
    db.add(job)
    db.flush()

    try:
        _run_export_generation(
            db,
            job=job,
            definition=definition,
            current_user=current_user,
            snapshot=snapshot,
            preview=preview,
        )
    except Exception as exc:
        job.status = ExportJobStatus.failed
        job.error_message = str(exc)[:500]
        job.updated_at = utc_now()
    db.add(job)
    return _serialize_job(job)


def get_export_job_response(db: Session, *, record: SessionRecord, job_id) -> ExportJobResponse:
    job = _find_export_job(db, record=record, job_id=job_id)
    if job.expires_at is not None and job.expires_at < utc_now() and job.status == ExportJobStatus.ready:
        job.status = ExportJobStatus.expired
        job.updated_at = utc_now()
        db.add(job)
    return _serialize_job(job)


def cancel_export_job_response(
    db: Session,
    *,
    record: SessionRecord,
    current_user: UserRecord,
    job_id,
) -> ExportJobResponse:
    job = _find_export_job(db, record=record, job_id=job_id)
    if job.status == ExportJobStatus.ready:
        raise ValueError("Ready export jobs cannot be canceled; revoke access or let the short-lived download expire.")
    if job.status == ExportJobStatus.canceled:
        return _serialize_job(job)
    job.status = ExportJobStatus.canceled
    job.error_message = "Canceled by user."
    job.updated_at = utc_now()
    job.completed_at = job.updated_at
    db.add(job)
    record_commercial_event(
        db,
        workspace_id=record.workspace_id,
        session_id=record.id,
        user_id=current_user.id,
        event_key="export_job_canceled",
        product_key=job.product_key,
        source="export_delivery",
        metadata={"artifact_kind": job.artifact_kind, "job_id": str(job.id)},
    )
    return _serialize_job(job)


def retry_export_job_response(
    db: Session,
    *,
    record: SessionRecord,
    current_user: UserRecord,
    access: CommercialAccessSnapshotV2,
    snapshot: SessionSnapshot,
    preview: ACPPreview,
    job_id,
) -> ExportJobResponse:
    job = _find_export_job(db, record=record, job_id=job_id)
    definition = _definition(job.artifact_kind)
    if not _capability_allowed(access, definition.required_capability):
        raise PermissionError(f"Export requires capability {definition.required_capability}.")
    if job.status == ExportJobStatus.running:
        raise ValueError("Export job is already running.")
    if job.status == ExportJobStatus.ready and job.expires_at is not None and job.expires_at >= utc_now():
        return _serialize_job(job)

    retry_count = int(job.metadata_payload.get("retry_count", 0) or 0) + 1
    job.metadata_payload = {**job.metadata_payload, "retry_count": retry_count, "last_retry_at": utc_now().isoformat()}
    job.status = ExportJobStatus.running
    job.error_message = ""
    job.updated_at = utc_now()
    job.completed_at = None
    db.add(job)
    record_commercial_event(
        db,
        workspace_id=record.workspace_id,
        session_id=record.id,
        user_id=current_user.id,
        event_key="export_job_retry_requested",
        product_key=job.product_key,
        source="export_delivery",
        metadata={"artifact_kind": job.artifact_kind, "job_id": str(job.id), "retry_count": retry_count},
    )
    try:
        _run_export_generation(
            db,
            job=job,
            definition=definition,
            current_user=current_user,
            snapshot=snapshot,
            preview=preview,
        )
    except Exception as exc:
        job.status = ExportJobStatus.failed
        job.error_message = str(exc)[:500]
        job.updated_at = utc_now()
    db.add(job)
    return _serialize_job(job)


def read_export_job_bytes(db: Session, *, record: SessionRecord, job_id) -> tuple[ExportJobRecord, bytes]:
    job = _find_export_job(db, record=record, job_id=job_id)
    if job.status != ExportJobStatus.ready:
        raise ValueError(f"Export job is not ready: {job.status.value}")
    if job.expires_at is not None and job.expires_at < utc_now():
        job.status = ExportJobStatus.expired
        job.updated_at = utc_now()
        db.add(job)
        raise ValueError("Export job download expired")
    path = _storage_path(job)
    if not path.exists():
        job.status = ExportJobStatus.failed
        job.error_message = "Export payload not found in storage."
        job.updated_at = utc_now()
        db.add(job)
        raise FileNotFoundError("Export payload not found")
    return job, path.read_bytes()
