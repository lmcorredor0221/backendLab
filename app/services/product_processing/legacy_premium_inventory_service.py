from __future__ import annotations

from collections import Counter
from datetime import datetime
from uuid import UUID

from sqlmodel import Session, select

from app.models import CommercialEventRecord, utc_now
from app.services.product_processing.contracts import (
    LegacyPremiumBuildHealth,
    LegacyPremiumEndpointUsage,
    LegacyPremiumInventoryGroup,
    LegacyPremiumInventoryReport,
    LegacyPremiumInventoryRow,
    LegacyPremiumTechnicalAttention,
    LegacyPremiumTechnicalFailureStep,
    ProductBuildLifecycle,
    ProductBuildProductKey,
    ProductProcessingMode,
)
from app.services.deliverable_catalog.persistence import DeliverableGenerationJobRecord
from app.services.product_processing.persistence import (
    ProductBuildRunRecord,
    ProductBuildStepRecord,
    UncertaintyBacklogRecord,
)


LEGACY_PREMIUM_ENDPOINT_EVENT = "legacy_premium_endpoint_invoked"
LEGACY_PREMIUM_ENDPOINT_SOURCE = "premium_enrichment_legacy_endpoint"
BUSINESS_KINDS = {"question", "gap", "assumption", "decision", "hitl"}
TECHNICAL_KINDS = {"runtime_error", "stale_dependency"}
CLOSED_STATUSES = {"resolved", "dismissed", "superseded"}


def record_legacy_premium_endpoint_invocation(
    db: Session,
    *,
    workspace_id: UUID,
    session_id: UUID,
    user_id: UUID,
    operation: str,
) -> None:
    """Record compatibility traffic without storing payload or answer content."""
    db.add(
        CommercialEventRecord(
            workspace_id=workspace_id,
            session_id=session_id,
            user_id=user_id,
            event_key=LEGACY_PREMIUM_ENDPOINT_EVENT,
            product_key="blueprint_pro",
            source=LEGACY_PREMIUM_ENDPOINT_SOURCE,
            metadata_payload={"operation": operation.strip().lower()},
        )
    )


def build_legacy_premium_inventory_report(
    db: Session,
    *,
    workspace_id: UUID | None = None,
    sample_limit: int = 200,
) -> LegacyPremiumInventoryReport:
    """Return a read-only migration inventory for the retired Premium domain."""
    statement = select(UncertaintyBacklogRecord).where(
        UncertaintyBacklogRecord.product_mode == ProductProcessingMode.premium_enrichment.value
    )
    if workspace_id is not None:
        statement = statement.where(UncertaintyBacklogRecord.workspace_id == workspace_id)
    records = list(db.exec(statement.order_by(UncertaintyBacklogRecord.updated_at.desc())).all())
    basic_keys = _basic_backlog_keys(db, workspace_id=workspace_id)
    rows = [_inventory_row(record, basic_keys=basic_keys) for record in records]
    classification_counts = Counter(row.classification for row in rows)
    groups = _build_groups(rows)
    build_health = _build_health(db, workspace_id=workspace_id)
    endpoint_usage = _endpoint_usage(db, workspace_id=workspace_id)
    warnings = _warnings(rows=rows, build_health=build_health, endpoint_usage=endpoint_usage)

    return LegacyPremiumInventoryReport(
        generated_at=utc_now().isoformat(),
        workspace_id=workspace_id,
        total_records=len(rows),
        business_pre_acp_count=classification_counts["business_pre_acp"],
        technical_pro_count=classification_counts["technical_pro"],
        closed_count=classification_counts["closed"],
        acp_managed_count=classification_counts["acp_managed"],
        ambiguous_count=classification_counts["ambiguous"],
        collision_count=sum(1 for row in rows if _is_actionable_collision(row)),
        build_health=build_health,
        endpoint_usage=endpoint_usage,
        groups=groups,
        records=rows[: max(1, min(sample_limit, 1_000))],
        migration_ready=not warnings,
        warnings=warnings,
    )


def _basic_backlog_keys(db: Session, *, workspace_id: UUID | None) -> set[tuple[UUID, str]]:
    statement = select(UncertaintyBacklogRecord.session_id, UncertaintyBacklogRecord.uncertainty_key).where(
        UncertaintyBacklogRecord.product_mode == ProductProcessingMode.basic_free.value
    )
    if workspace_id is not None:
        statement = statement.where(UncertaintyBacklogRecord.workspace_id == workspace_id)
    return {(session_id, key) for session_id, key in db.exec(statement).all()}


def _inventory_row(
    record: UncertaintyBacklogRecord,
    *,
    basic_keys: set[tuple[UUID, str]],
) -> LegacyPremiumInventoryRow:
    classification, migration_action = _classify_record(record)
    payload = record.payload if isinstance(record.payload, dict) else {}
    return LegacyPremiumInventoryRow(
        record_id=record.id,
        workspace_id=record.workspace_id,
        session_id=record.session_id,
        uncertainty_key=record.uncertainty_key,
        source_stage=record.source_stage,
        target_stage=record.target_stage,
        kind=record.kind,
        disposition=record.disposition,
        status=record.status,
        source_tier=str(payload.get("source_tier") or "blueprint_pro"),
        classification=classification,
        migration_action=migration_action,
        collides_with_basic=(record.session_id, record.uncertainty_key) in basic_keys,
    )


def _is_actionable_collision(row: LegacyPremiumInventoryRow) -> bool:
    """Closed Premium sources may share a Basic key as preserved audit history."""
    return row.collides_with_basic and row.classification != "closed"


def _classify_record(record: UncertaintyBacklogRecord) -> tuple[str, str]:
    status = str(record.status or "").strip().lower()
    kind = str(record.kind or "").strip().lower()
    disposition = str(record.disposition or "").strip().lower()
    target_stage = str(record.target_stage or "").strip().lower()
    payload = record.payload if isinstance(record.payload, dict) else {}

    if status in CLOSED_STATUSES:
        if status == "resolved" and (record.assumed_answer or record.suggested_answer):
            return "closed", "preserve_historical_response"
        return "closed", "retain_audit_only"
    if isinstance(payload.get("acp_resolution"), dict) or (target_stage == "acp" and disposition == "defer"):
        return "acp_managed", "merge_acp_metadata"
    if kind in TECHNICAL_KINDS:
        return "technical_pro", "retain_technical_pro"
    if kind in BUSINESS_KINDS:
        return "business_pre_acp", "defer_to_acp"
    return "ambiguous", "manual_review"


def _build_groups(rows: list[LegacyPremiumInventoryRow]) -> list[LegacyPremiumInventoryGroup]:
    counts = Counter((row.workspace_id, row.session_id, row.classification, row.status) for row in rows)
    return [
        LegacyPremiumInventoryGroup(
            workspace_id=workspace_id,
            session_id=session_id,
            classification=classification,
            status=status,
            count=count,
        )
        for (workspace_id, session_id, classification, status), count in sorted(
            counts.items(), key=lambda item: (str(item[0][0]), str(item[0][1]), item[0][2], item[0][3])
        )
    ]


def _build_health(db: Session, *, workspace_id: UUID | None) -> LegacyPremiumBuildHealth:
    statement = select(ProductBuildRunRecord).where(
        (ProductBuildRunRecord.product_key == ProductBuildProductKey.blueprint_pro.value)
        | (ProductBuildRunRecord.product_mode == ProductProcessingMode.premium_enrichment.value)
    )
    if workspace_id is not None:
        statement = statement.where(ProductBuildRunRecord.workspace_id == workspace_id)
    runs = list(db.exec(statement).all())
    run_ids = [run.id for run in runs]
    steps = []
    if run_ids:
        steps = list(db.exec(select(ProductBuildStepRecord).where(ProductBuildStepRecord.run_id.in_(run_ids))).all())
    job_ids = [step.job_id for step in steps if step.job_id is not None]
    jobs_by_id = {}
    if job_ids:
        jobs_by_id = {
            job.id: job
            for job in db.exec(select(DeliverableGenerationJobRecord).where(DeliverableGenerationJobRecord.id.in_(job_ids))).all()
        }
    steps_by_run: dict[UUID, list[ProductBuildStepRecord]] = {}
    for step in steps:
        steps_by_run.setdefault(step.run_id, []).append(step)

    business_attention = 0
    technical_attention = 0
    unattributed_attention = 0
    technical_attention_runs: list[LegacyPremiumTechnicalAttention] = []
    for run in runs:
        if run.lifecycle != ProductBuildLifecycle.requires_attention.value:
            continue
        run_steps = steps_by_run.get(run.id, [])
        legacy_steps = [step for step in run_steps if step.step_key.startswith("premium_backlog:")]
        has_technical_failure = bool(run.error_payload) or any(
            step not in legacy_steps and step.status in {"error", "failed", "requires_attention"}
            for step in run_steps
        )
        if has_technical_failure:
            technical_attention += 1
            technical_attention_runs.append(
                _technical_attention_detail(run, run_steps=run_steps, jobs_by_id=jobs_by_id)
            )
        elif legacy_steps:
            business_attention += 1
        else:
            unattributed_attention += 1
    return LegacyPremiumBuildHealth(
        legacy_run_count=len(runs),
        requires_attention_count=business_attention + technical_attention + unattributed_attention,
        business_backlog_attention_count=business_attention,
        technical_attention_count=technical_attention,
        unattributed_attention_count=unattributed_attention,
        technical_attention_runs=sorted(technical_attention_runs, key=lambda item: item.updated_at, reverse=True),
    )


def _technical_attention_detail(
    run: ProductBuildRunRecord,
    *,
    run_steps: list[ProductBuildStepRecord],
    jobs_by_id: dict[UUID, DeliverableGenerationJobRecord],
) -> LegacyPremiumTechnicalAttention:
    error_payload = run.error_payload if isinstance(run.error_payload, dict) else {}
    failed_steps = [
        _technical_failure_step(step, jobs_by_id=jobs_by_id)
        for step in run_steps
        if not step.step_key.startswith("premium_backlog:")
        and step.status in {"error", "failed", "requires_attention"}
    ]
    return LegacyPremiumTechnicalAttention(
        run_id=run.id,
        session_id=run.session_id,
        lifecycle=run.lifecycle,
        error_code=str(error_payload.get("code") or ""),
        error_title=str(error_payload.get("title") or ""),
        error_message=str(error_payload.get("message") or ""),
        failed_steps=failed_steps[:25],
        updated_at=run.updated_at.isoformat(),
    )


def _technical_failure_step(
    step: ProductBuildStepRecord,
    *,
    jobs_by_id: dict[UUID, DeliverableGenerationJobRecord],
) -> LegacyPremiumTechnicalFailureStep:
    error_payload = step.error_payload if isinstance(step.error_payload, dict) else {}
    job = jobs_by_id.get(step.job_id) if step.job_id is not None else None
    return LegacyPremiumTechnicalFailureStep(
        step_id=step.id,
        step_key=step.step_key,
        deliverable_key=step.deliverable_key,
        status=step.status,
        error_code=str(error_payload.get("code") or getattr(job, "error_code", "") or ""),
        error_message=str(error_payload.get("message") or getattr(job, "error_message", "") or ""),
        updated_at=step.updated_at.isoformat(),
    )


def _endpoint_usage(db: Session, *, workspace_id: UUID | None) -> list[LegacyPremiumEndpointUsage]:
    statement = select(CommercialEventRecord).where(
        CommercialEventRecord.event_key == LEGACY_PREMIUM_ENDPOINT_EVENT,
        CommercialEventRecord.source == LEGACY_PREMIUM_ENDPOINT_SOURCE,
    )
    if workspace_id is not None:
        statement = statement.where(CommercialEventRecord.workspace_id == workspace_id)
    events = list(db.exec(statement).all())
    counts: dict[str, tuple[int, datetime]] = {}
    for event in events:
        metadata = event.metadata_payload if isinstance(event.metadata_payload, dict) else {}
        operation = str(metadata.get("operation") or "unknown")
        count, latest = counts.get(operation, (0, event.created_at))
        counts[operation] = (count + 1, max(latest, event.created_at))
    return [
        LegacyPremiumEndpointUsage(operation=operation, invocation_count=count, last_invoked_at=latest.isoformat())
        for operation, (count, latest) in sorted(counts.items())
    ]


def _warnings(
    *,
    rows: list[LegacyPremiumInventoryRow],
    build_health: LegacyPremiumBuildHealth,
    endpoint_usage: list[LegacyPremiumEndpointUsage],
) -> list[str]:
    warnings: list[str] = []
    if any(_is_actionable_collision(row) for row in rows):
        warnings.append("Existen colisiones Basic/Premium; la migracion requiere fusion por sesion y clave.")
    if any(row.classification == "ambiguous" for row in rows):
        warnings.append("Existen registros Premium ambiguos que requieren regla de negocio antes de migrar.")
    if build_health.business_backlog_attention_count:
        warnings.append("Existen builds Pro historicos en Atencion por backlog de negocio.")
    if build_health.unattributed_attention_count:
        warnings.append("Existen builds Pro historicos en Atencion sin causa tecnica verificable.")
    if endpoint_usage:
        warnings.append("Los endpoints Premium legacy registran uso; no se pueden retirar todavia.")
    return warnings
