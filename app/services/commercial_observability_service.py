from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

from sqlmodel import Session, select

from app.models import (
    ArtifactStatus,
    CommercialAuditEventEntry,
    CommercialAuditFunnelStep,
    CommercialAuditMetric,
    CommercialAuditProductSummary,
    CommercialAuditReport,
    CommercialEventRecord,
    CommercialTier,
    ExecutionLogRecord,
    SessionRecord,
    UserRecord,
)
from app.services.deliverable_catalog.persistence import (
    DeliverableGenerationJobRecord,
    DeliverableGovernanceAuditRecord,
    DeliverablePromptAuditRecord,
)
from app.services.product_processing.persistence import UncertaintyBacklogRecord


SENSITIVE_KEY_FRAGMENTS = (
    "api_key",
    "authorization",
    "content",
    "credential",
    "html",
    "markdown",
    "password",
    "prompt",
    "secret",
    "svg",
    "token",
)

VIEW_EVENT_KEYS = {
    "blueprint_results_viewed",
    "diagram_readable_selected",
    "diagram_locked_selected",
    "invitation_viewed",
}
BLOCKED_EVENT_KEYS = {
    "diagram_download_blocked",
    "diagram_locked_selected",
    "diagram_protected_action_blocked",
}
CTA_EVENT_KEYS = {
    "acquire_clicked",
    "blueprint_pro_acquire_clicked",
    "diagram_stage_redirect_clicked",
    "diagram_upsell_clicked",
    "return_blueprint_clicked",
}
PURCHASE_EVENT_KEYS = {"payment_confirmed", "tier_updated"}
EXPORT_EVENT_KEYS = {
    "acp_exported",
    "blueprint_exported",
    "canonical_exported",
}


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower()
    return any(fragment in normalized for fragment in SENSITIVE_KEY_FRAGMENTS)


def _sanitize_value(value: Any, *, key: str = "", depth: int = 0) -> Any:
    if _is_sensitive_key(key):
        return "[redacted]"

    if depth > 4:
        return "[truncated]"

    if isinstance(value, dict):
        return {
            str(child_key): _sanitize_value(child_value, key=str(child_key), depth=depth + 1)
            for child_key, child_value in list(value.items())[:30]
        }

    if isinstance(value, list):
        return [_sanitize_value(item, depth=depth + 1) for item in value[:30]]

    if isinstance(value, str):
        if len(value) > 180:
            return f"{value[:180]}..."
        return value

    return value


def _safe_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    safe_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"metadata"} and not _is_sensitive_key(str(key))
    }
    if metadata:
        safe_payload["metadata"] = _sanitize_value(metadata)
    return _sanitize_value(safe_payload)


def _infer_product_from_contract(contract_key: str) -> str:
    if contract_key in {"blueprint-core.v1", "estimation-pack.v1"}:
        return "blueprint_pro"
    return "acp"


def _infer_product_from_export(source_action: str, payload: dict[str, Any]) -> str:
    contract_key = str(payload.get("contract_key") or "")
    if contract_key:
        return _infer_product_from_contract(contract_key)
    artifact_key = str(payload.get("artifact_key") or "")
    profile = str(payload.get("profile") or "")
    combined = " ".join([source_action, artifact_key, profile]).lower()
    if "acp" in combined or "construction" in combined or "prompt" in combined or "test" in combined:
        return "acp"
    return "blueprint_pro"


def _normalize_event(log: ExecutionLogRecord) -> CommercialAuditEventEntry | None:
    payload = log.payload or {}
    if "event_key" in payload:
        return CommercialAuditEventEntry(
            created_at=log.created_at,
            event_key=str(payload.get("event_key") or "unknown"),
            message=log.message,
            metadata=_safe_metadata(payload),
            product=str(payload.get("product") or ""),
            source=str(payload.get("source") or ""),
            stage=log.stage,
            status=log.status,
        )

    if log.message == "Tier comercial actualizado":
        tier = str(payload.get("commercial_tier") or "")
        return CommercialAuditEventEntry(
            created_at=log.created_at,
            event_key="tier_updated",
            message=log.message,
            metadata=_safe_metadata(payload),
            product=tier or "unknown",
            source="commercial_tier",
            stage=log.stage,
            status=log.status,
        )

    source_action = str(payload.get("source_action") or "")
    artifact_key = str(payload.get("artifact_key") or "")
    contract_key = str(payload.get("contract_key") or "")
    if log.message == "Export canonico bloqueado por conformance":
        return CommercialAuditEventEntry(
            created_at=log.created_at,
            event_key="canonical_export_blocked",
            message=log.message,
            metadata=_safe_metadata(payload),
            product=_infer_product_from_contract(contract_key),
            source="canonical_export",
            stage=log.stage,
            status=log.status,
        )

    if log.message == "ACP zip exportado":
        return CommercialAuditEventEntry(
            created_at=log.created_at,
            event_key="acp_exported",
            message=log.message,
            metadata=_safe_metadata(payload),
            product="acp",
            source=source_action or artifact_key or "export_acp_zip",
            stage=log.stage,
            status=log.status,
        )

    is_export = (
        "export" in source_action
        or "export" in artifact_key
        or log.message in {"ACP zip exportado", "Blueprint export markdown descargado", "Blueprint export json descargado"}
    )
    if is_export:
        product = _infer_product_from_export(source_action or artifact_key or contract_key, payload)
        event_key = "acp_exported" if product == "acp" else "blueprint_exported"
        if contract_key:
            event_key = "canonical_exported"
        return CommercialAuditEventEntry(
            created_at=log.created_at,
            event_key=event_key,
            message=log.message,
            metadata=_safe_metadata(payload),
            product=product,
            source=source_action or artifact_key or "export",
            stage=log.stage,
            status=log.status,
        )

    if log.message == "ACP preview generado":
        return CommercialAuditEventEntry(
            created_at=log.created_at,
            event_key="acp_preview_generated",
            message=log.message,
            metadata=_safe_metadata(payload),
            product="acp",
            source="acp_preview",
            stage=log.stage,
            status=log.status,
        )

    return None


def _normalize_commerce_event(event: CommercialEventRecord, record: SessionRecord) -> CommercialAuditEventEntry:
    return CommercialAuditEventEntry(
        created_at=event.created_at,
        event_key=event.event_key,
        message=event.event_key.replace("_", " ").capitalize(),
        metadata=_sanitize_value(
            {
                "correlation_id": event.correlation_id,
                "currency": event.currency,
                "metadata": event.metadata_payload or {},
                "revenue_cents": event.revenue_cents,
            }
        ),
        product=event.product_key,
        source=event.source,
        stage=record.current_stage,
        status=ArtifactStatus.ready,
    )


def _count_matching(events: list[CommercialAuditEventEntry], event_keys: set[str], *, product: str | None = None) -> int:
    return sum(1 for event in events if event.event_key in event_keys and (product is None or event.product == product))


def _latest_matching(events: list[CommercialAuditEventEntry], event_keys: set[str], *, product: str | None = None) -> datetime | None:
    for event in events:
        if event.event_key in event_keys and (product is None or event.product == product):
            return event.created_at
    return None


def _build_funnel(events: list[CommercialAuditEventEntry]) -> list[CommercialAuditFunnelStep]:
    step_specs = [
        ("blueprint_value_viewed", "Resultado Blueprint visto", "blueprint", {"blueprint_results_viewed"}),
        ("blueprint_pro_purchased", "Blueprint Profesional adquirido", "blueprint_pro", PURCHASE_EVENT_KEYS),
        ("acp_invitation_viewed", "Invitacion ACP vista", "acp", {"invitation_viewed"}),
        ("acp_purchased", "ACP Premium adquirido", "acp", PURCHASE_EVENT_KEYS),
        ("acp_exported", "ACP exportado", "acp", {"acp_exported", "canonical_exported"}),
    ]
    steps: list[CommercialAuditFunnelStep] = []
    previous_count = 0
    for key, label, product, event_keys in step_specs:
        if key == "blueprint_pro_purchased":
            count = _count_matching(events, event_keys, product="blueprint_pro")
        elif key == "acp_purchased":
            count = _count_matching(events, event_keys, product="acp")
        elif key == "acp_exported":
            count = sum(1 for event in events if event.product == "acp" and event.event_key in event_keys)
        else:
            count = _count_matching(events, event_keys, product=product)

        denominator = max(previous_count, 1)
        steps.append(
            CommercialAuditFunnelStep(
                completed=count > 0,
                conversion_percent=100 if count > 0 and previous_count == 0 else min(100, round(count / denominator * 100)),
                count=count,
                event_keys=sorted(event_keys),
                key=key,
                label=label,
                latest_at=_latest_matching(events, event_keys, product=product),
                product=product,
            )
        )
        previous_count = count
    return steps


def _build_product_summary(events: list[CommercialAuditEventEntry]) -> list[CommercialAuditProductSummary]:
    products = ["blueprint", "blueprint_pro", "acp"]
    rows: list[CommercialAuditProductSummary] = []
    for product in products:
        product_events = [event for event in events if event.product == product]
        rows.append(
            CommercialAuditProductSummary(
                blocked_events=sum(1 for event in product_events if event.event_key in BLOCKED_EVENT_KEYS or "blocked" in event.event_key),
                cta_clicks=sum(1 for event in product_events if event.event_key in CTA_EVENT_KEYS or event.event_key.endswith("_clicked")),
                exports=sum(1 for event in product_events if event.event_key in EXPORT_EVENT_KEYS),
                product=product,
                purchases=sum(1 for event in product_events if event.event_key in PURCHASE_EVENT_KEYS),
                views=sum(1 for event in product_events if event.event_key in VIEW_EVENT_KEYS or event.event_key.endswith("_viewed")),
            )
        )
    return rows


def _build_product_runtime_metrics(db: Session, record: SessionRecord) -> tuple[list[CommercialAuditMetric], list[str]]:
    metrics: list[CommercialAuditMetric] = []
    warnings: list[str] = []
    jobs = db.exec(
        select(DeliverableGenerationJobRecord).where(
            DeliverableGenerationJobRecord.workspace_id == record.workspace_id,
            DeliverableGenerationJobRecord.session_id == record.id,
        )
    ).all()
    uncertainties = db.exec(
        select(UncertaintyBacklogRecord).where(
            UncertaintyBacklogRecord.workspace_id == record.workspace_id,
            UncertaintyBacklogRecord.session_id == record.id,
        )
    ).all()
    governance_audits = db.exec(
        select(DeliverableGovernanceAuditRecord).where(
            (DeliverableGovernanceAuditRecord.workspace_id == record.workspace_id)
            | (DeliverableGovernanceAuditRecord.scope_key == "platform")
        )
    ).all()
    prompt_audits = db.exec(
        select(DeliverablePromptAuditRecord).where(
            (DeliverablePromptAuditRecord.workspace_id == record.workspace_id)
            | (DeliverablePromptAuditRecord.scope_key == "platform")
        )
    ).all()

    total_tokens = sum(max(0, job.tokens_input) + max(0, job.tokens_output) for job in jobs)
    estimated_cost_usd = round(sum(max(0, job.estimated_cost_usd) for job in jobs), 6)
    failed_jobs = [job for job in jobs if job.status in {"failed", "requires_attention"} or job.error_code]
    fallback_jobs = []
    for job in jobs:
        request_metadata = job.request_metadata or {}
        fallback_used = str(request_metadata.get("fallback_used") or "").strip().lower()
        fallback_policy = str(request_metadata.get("fallback_policy") or "").strip().lower()
        if fallback_used in {"true", "1", "yes"} or fallback_policy not in {"", "none", "not_applicable", "disabled"}:
            fallback_jobs.append(job)
    deferred = [item for item in uncertainties if item.disposition == "defer" or item.status == "deferred"]
    blocking = [item for item in uncertainties if item.disposition == "block" and item.status not in {"resolved", "dismissed", "superseded"}]
    premium_items = [item for item in uncertainties if item.product_mode == "premium_enrichment"]
    acp_items = [item for item in uncertainties if item.product_mode == "acp_implementation"]

    metrics.extend(
        [
            CommercialAuditMetric(
                detail="Jobs de generacion de entregables registrados por catalogo y producto.",
                key="deliverable_generation_jobs",
                label="Generaciones",
                tone="green" if jobs else "slate",
                value=len(jobs),
            ),
            CommercialAuditMetric(
                detail="Jobs con error, fallback accionable o estado requires_attention.",
                key="deliverable_generation_errors",
                label="Errores/fallback",
                tone="red" if failed_jobs else "orange" if fallback_jobs else "green",
                value=len(failed_jobs) + len(fallback_jobs),
            ),
            CommercialAuditMetric(
                detail="Tokens agregados de generacion. No expone prompts ni razonamiento interno.",
                key="llm_token_usage",
                label="Tokens LLM",
                tone="blue" if total_tokens else "slate",
                value=total_tokens,
                unit="tokens",
            ),
            CommercialAuditMetric(
                detail="Costo estimado de generacion por entregables, en USD, cuando el proveedor lo reporta.",
                key="llm_estimated_cost_usd",
                label="Costo LLM",
                tone="violet" if estimated_cost_usd else "slate",
                value=estimated_cost_usd,
                unit="USD",
            ),
            CommercialAuditMetric(
                detail="Preguntas/gaps diferidos por Blueprint Basico o flujo no bloqueante.",
                key="uncertainties_deferred",
                label="Diferidas",
                tone="blue" if deferred else "green",
                value=len(deferred),
            ),
            CommercialAuditMetric(
                detail="Preguntas/gaps que bloquean ACP o readiness tecnico hasta resolverse.",
                key="uncertainties_blocking",
                label="Bloqueantes",
                tone="red" if blocking else "green",
                value=len(blocking),
            ),
            CommercialAuditMetric(
                detail="Incertidumbres asociadas al enriquecimiento Premium.",
                key="premium_uncertainties",
                label="Premium",
                tone="violet" if premium_items else "slate",
                value=len(premium_items),
            ),
            CommercialAuditMetric(
                detail="Incertidumbres asociadas al ACP y decisiones de implementacion.",
                key="acp_uncertainties",
                label="ACP",
                tone="orange" if acp_items else "slate",
                value=len(acp_items),
            ),
            CommercialAuditMetric(
                detail="Cambios administrativos de gobernanza de entregables aplicables al workspace/plataforma.",
                key="admin_governance_audits",
                label="Auditoria gov",
                tone="blue" if governance_audits else "slate",
                value=len(governance_audits),
            ),
            CommercialAuditMetric(
                detail="Cambios administrativos de prompts versionados aplicables al workspace/plataforma.",
                key="prompt_audits",
                label="Auditoria prompts",
                tone="blue" if prompt_audits else "slate",
                value=len(prompt_audits),
            ),
        ]
    )
    if failed_jobs:
        warnings.append("Existen generaciones de entregables con error o atencion requerida; revisar antes de exportar comercialmente.")
    if blocking:
        warnings.append("Existen incertidumbres bloqueantes activas para ACP/readiness.")
    return metrics, warnings


def build_commercial_audit_report(
    db: Session,
    *,
    record: SessionRecord,
    current_user: UserRecord,
    limit: int = 40,
) -> CommercialAuditReport:
    logs = db.exec(
        select(ExecutionLogRecord)
        .where(ExecutionLogRecord.session_id == record.id)
        .order_by(ExecutionLogRecord.created_at.desc())
    ).all()
    events = [event for log in logs if (event := _normalize_event(log)) is not None]
    checkout_events = db.exec(
        select(CommercialEventRecord)
        .where(CommercialEventRecord.session_id == record.id)
        .order_by(CommercialEventRecord.created_at.desc())
    ).all()
    events = sorted(
        [*events, *[_normalize_commerce_event(event, record) for event in checkout_events]],
        key=lambda event: event.created_at,
        reverse=True,
    )
    event_counter = Counter(event.event_key for event in events)
    export_count = sum(1 for event in events if event.event_key in EXPORT_EVENT_KEYS)
    blocked_count = sum(1 for event in events if event.event_key in BLOCKED_EVENT_KEYS or "blocked" in event.event_key)
    conformance_errors = sum(1 for event in events if event.event_key == "canonical_export_blocked" or event.status == ArtifactStatus.failed)
    launcher_used = event_counter.get("launcher_used", 0)
    products_touched = len({event.product for event in events if event.product})
    warnings: list[str] = []
    if not events:
        warnings.append("No hay eventos comerciales normalizados para esta sesion todavia.")
    if conformance_errors:
        warnings.append("Existen bloqueos o errores de conformance registrados para revisar antes de vender el paquete como listo.")
    runtime_metrics, runtime_warnings = _build_product_runtime_metrics(db, record)
    warnings.extend(runtime_warnings)

    return CommercialAuditReport(
        current_tier=record.commercial_tier if isinstance(record.commercial_tier, CommercialTier) else CommercialTier(record.commercial_tier),
        funnel=_build_funnel(events),
        metrics=[
            CommercialAuditMetric(
                detail="Eventos comerciales y tecnicos normalizados desde execution_logs.",
                key="total_events",
                label="Eventos auditables",
                tone="blue",
                value=len(events),
            ),
            CommercialAuditMetric(
                detail="Intentos bloqueados, contenido protegido o exportaciones detenidas.",
                key="blocked_events",
                label="Bloqueos",
                tone="orange" if blocked_count else "green",
                value=blocked_count,
            ),
            CommercialAuditMetric(
                detail="Blueprint, ACP y otros productos detectados en la actividad.",
                key="products_touched",
                label="Productos tocados",
                tone="violet",
                value=products_touched,
            ),
            CommercialAuditMetric(
                detail="Exportables Blueprint/ACP generados o descargados.",
                key="exports",
                label="Exportaciones",
                tone="green" if export_count else "slate",
                value=export_count,
            ),
            CommercialAuditMetric(
                detail="Errores o bloqueos asociados a contratos canonicos/conformance.",
                key="conformance_errors",
                label="Conformance",
                tone="red" if conformance_errors else "green",
                value=conformance_errors,
            ),
            CommercialAuditMetric(
                detail="Uso reportado del launcher ACP portable.",
                key="launcher_used",
                label="Launcher",
                tone="green" if launcher_used else "slate",
                value=launcher_used,
            ),
            *runtime_metrics,
        ],
        product_summary=_build_product_summary(events),
        recent_events=events[: max(1, min(limit, 100))],
        requested_by_user_id=current_user.id,
        session_id=record.id,
        warnings=warnings,
        workspace_id=record.workspace_id,
    )
