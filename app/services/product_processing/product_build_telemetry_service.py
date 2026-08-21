from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlmodel import Session, select

from app.models import CommercialEventRecord, SessionRecord, StageOperationRecord, UserRecord, utc_now
from app.services.deliverable_catalog.persistence import DeliverableGenerationJobRecord
from app.services.product_processing.contracts import (
    ProductBuildProductKey,
    ProductBuildTelemetryEvent,
    ProductBuildTelemetryProductSummary,
    ProductBuildTelemetryReport,
    ProductBuildTelemetryTotals,
)
from app.services.product_processing.persistence import ProductBuildRunRecord, ProductBuildStepRecord


PRODUCT_LABELS: dict[ProductBuildProductKey, str] = {
    ProductBuildProductKey.blueprint_basic: "Blueprint Basico",
    ProductBuildProductKey.blueprint_pro: "Blueprint Pro",
    ProductBuildProductKey.acp: "Agent Construction Package",
}

PRODUCT_MODE_TO_KEY = {
    "basic_free": ProductBuildProductKey.blueprint_basic,
    "blueprint": ProductBuildProductKey.blueprint_basic,
    "blueprint_basic": ProductBuildProductKey.blueprint_basic,
    "premium": ProductBuildProductKey.blueprint_pro,
    "premium_enrichment": ProductBuildProductKey.blueprint_pro,
    "blueprint_pro": ProductBuildProductKey.blueprint_pro,
    "acp": ProductBuildProductKey.acp,
    "acp_implementation": ProductBuildProductKey.acp,
}

CTA_EVENT_KEYS = {
    "acquire_clicked",
    "blueprint_results_viewed",
    "blueprint_pro_acquire_clicked",
    "buy_product_clicked",
    "diagram_upsell_clicked",
    "invitation_viewed",
}
CHECKOUT_EVENT_KEYS = {"checkout_started", "payment_confirmed", "purchase_confirmed", "tier_updated"}
ACTIVATION_EVENT_KEYS = {"product_activated", "build_activated", "tier_updated"}
ERROR_STATES = {"error", "failed", "requires_attention"}
ATTENTION_STATES = {"requires_attention", "blocked"}
SENSITIVE_METADATA_FRAGMENTS = {
    "api_key",
    "authorization",
    "content",
    "credential",
    "html",
    "markdown",
    "password",
    "prompt",
    "raw",
    "secret",
    "svg",
    "token",
}


def build_product_build_telemetry_report(
    db: Session,
    *,
    record: SessionRecord,
    current_user: UserRecord | None = None,
    limit: int = 100,
) -> ProductBuildTelemetryReport:
    runs = _load_runs(db, record)
    steps = _load_steps(db, record)
    jobs = _load_jobs(db, record)
    commercial_events = _load_commercial_events(db, record, limit=limit)
    operations = _load_operations(db, record, limit=limit)

    events = _build_events(
        record=record,
        runs=runs,
        steps=steps,
        jobs=jobs,
        commercial_events=commercial_events,
        operations=operations,
        limit=limit,
    )
    products = [
        _build_product_summary(
            product_key=product_key,
            runs=runs,
            steps=steps,
            jobs=jobs,
            events=events,
        )
        for product_key in (
            ProductBuildProductKey.blueprint_basic,
            ProductBuildProductKey.blueprint_pro,
            ProductBuildProductKey.acp,
        )
    ]
    totals = ProductBuildTelemetryTotals(
        product_count=len(products),
        run_count=sum(item.run_count for item in products),
        step_count=sum(item.step_count for item in products),
        deliverable_count=sum(item.deliverable_count for item in products),
        event_count=len(events),
        requires_attention_count=sum(item.requires_attention_count for item in products),
        retry_count=sum(item.retry_count for item in products),
        resume_count=sum(item.resume_count for item in products),
        tokens_total=sum(item.tokens_total for item in products),
        estimated_cost_usd=round(sum(item.estimated_cost_usd for item in products), 6),
    )
    warnings = _build_warnings(products=products, jobs=jobs)

    return ProductBuildTelemetryReport(
        workspace_id=record.workspace_id,
        session_id=record.id,
        requested_by_user_id=current_user.id if current_user is not None else None,
        generated_at=utc_now().isoformat(),
        products=products,
        events=events,
        totals=totals,
        warnings=warnings,
        source_contracts=[
            "product-build-runs.v1",
            "product-build-steps.v1",
            "commercial-events.v1",
            "stage-operations.v1",
            "deliverable-generation-jobs.v1",
        ],
    )


def _load_runs(db: Session, record: SessionRecord) -> list[ProductBuildRunRecord]:
    return list(
        db.exec(
            select(ProductBuildRunRecord)
            .where(ProductBuildRunRecord.workspace_id == record.workspace_id, ProductBuildRunRecord.session_id == record.id)
            .order_by(ProductBuildRunRecord.updated_at.desc())
        ).all()
    )


def _load_steps(db: Session, record: SessionRecord) -> list[ProductBuildStepRecord]:
    return list(
        db.exec(
            select(ProductBuildStepRecord)
            .where(ProductBuildStepRecord.workspace_id == record.workspace_id, ProductBuildStepRecord.session_id == record.id)
            .order_by(ProductBuildStepRecord.updated_at.desc())
        ).all()
    )


def _load_jobs(db: Session, record: SessionRecord) -> list[DeliverableGenerationJobRecord]:
    return list(
        db.exec(
            select(DeliverableGenerationJobRecord)
            .where(
                DeliverableGenerationJobRecord.workspace_id == record.workspace_id,
                DeliverableGenerationJobRecord.session_id == record.id,
            )
            .order_by(DeliverableGenerationJobRecord.updated_at.desc())
        ).all()
    )


def _load_commercial_events(db: Session, record: SessionRecord, *, limit: int) -> list[CommercialEventRecord]:
    return list(
        db.exec(
            select(CommercialEventRecord)
            .where(CommercialEventRecord.workspace_id == record.workspace_id, CommercialEventRecord.session_id == record.id)
            .order_by(CommercialEventRecord.created_at.desc())
            .limit(limit)
        ).all()
    )


def _load_operations(db: Session, record: SessionRecord, *, limit: int) -> list[StageOperationRecord]:
    return list(
        db.exec(
            select(StageOperationRecord)
            .where(StageOperationRecord.workspace_id == record.workspace_id, StageOperationRecord.session_id == record.id)
            .order_by(StageOperationRecord.updated_at.desc(), StageOperationRecord.created_at.desc())
            .limit(limit)
        ).all()
    )


def _build_events(
    *,
    record: SessionRecord,
    runs: list[ProductBuildRunRecord],
    steps: list[ProductBuildStepRecord],
    jobs: list[DeliverableGenerationJobRecord],
    commercial_events: list[CommercialEventRecord],
    operations: list[StageOperationRecord],
    limit: int,
) -> list[ProductBuildTelemetryEvent]:
    latest_run_by_product = _latest_run_by_product(runs)
    step_by_id = {str(step.id): step for step in steps}
    events: list[ProductBuildTelemetryEvent] = []

    for event in commercial_events:
        product_key = _normalize_product_key(event.product_key)
        run = latest_run_by_product.get(product_key)
        events.append(
            ProductBuildTelemetryEvent(
                event_key=event.event_key or "commercial_event",
                event_type=_classify_event(event.event_key, event.source),
                workspace_id=record.workspace_id,
                session_id=record.id,
                product_key=product_key,
                run_id=str(run.id) if run is not None else "",
                source=event.source or "commercial_event",
                status="recorded",
                created_at=event.created_at.isoformat(),
                metadata_keys=_safe_metadata_keys(event.metadata_payload),
            )
        )

    for run in runs:
        product_key = _normalize_product_key(run.product_key or run.product_mode)
        if run.started_at is not None:
            events.append(
                ProductBuildTelemetryEvent(
                    event_key="run_started",
                    event_type="run",
                    workspace_id=record.workspace_id,
                    session_id=record.id,
                    product_key=product_key,
                    run_id=str(run.id),
                    source="product_build_run",
                    status=run.lifecycle,
                    created_at=run.started_at.isoformat(),
                    metadata_keys=_safe_metadata_keys(run.checkpoint_payload),
                )
            )
        if run.completed_at is not None:
            events.append(
                ProductBuildTelemetryEvent(
                    event_key="run_completed",
                    event_type="run",
                    workspace_id=record.workspace_id,
                    session_id=record.id,
                    product_key=product_key,
                    run_id=str(run.id),
                    source="product_build_run",
                    status=run.lifecycle,
                    created_at=run.completed_at.isoformat(),
                )
            )
        if run.lifecycle in ERROR_STATES:
            created_at = (run.requires_attention_at or run.completed_at or run.updated_at).isoformat()
            events.append(
                ProductBuildTelemetryEvent(
                    event_key="run_requires_attention" if run.lifecycle == "requires_attention" else "run_error",
                    event_type="attention" if run.lifecycle == "requires_attention" else "error",
                    workspace_id=record.workspace_id,
                    session_id=record.id,
                    product_key=product_key,
                    run_id=str(run.id),
                    source="product_build_run",
                    status=run.lifecycle,
                    created_at=created_at,
                    metadata_keys=_safe_metadata_keys(run.error_payload),
                )
            )

    for step in steps:
        if step.status not in ERROR_STATES and step.status not in ATTENTION_STATES:
            continue
        run = next((item for item in runs if item.id == step.run_id), None)
        product_key = _normalize_product_key(run.product_key if run is not None else "")
        events.append(
            ProductBuildTelemetryEvent(
                event_key="step_requires_attention" if step.status in ATTENTION_STATES else "step_error",
                event_type="attention" if step.status in ATTENTION_STATES else "error",
                workspace_id=record.workspace_id,
                session_id=record.id,
                product_key=product_key,
                run_id=str(step.run_id),
                step_id=str(step.id),
                stage_key=step.stage_key,
                deliverable_key=step.deliverable_key,
                source="product_build_step",
                status=step.status,
                created_at=step.updated_at.isoformat(),
                metadata_keys=_safe_metadata_keys(step.error_payload),
            )
        )

    for job in jobs:
        if job.status not in ERROR_STATES and not job.error_code:
            continue
        product_key = _normalize_product_key(job.product_mode or (job.request_metadata or {}).get("product_key", ""))
        related_step = next((step for step in steps if step.job_id == job.id), None)
        events.append(
            ProductBuildTelemetryEvent(
                event_key="deliverable_generation_error",
                event_type="error",
                workspace_id=record.workspace_id,
                session_id=record.id,
                product_key=product_key,
                run_id=str(related_step.run_id) if related_step is not None else "",
                step_id=str(related_step.id) if related_step is not None else "",
                stage_key=related_step.stage_key if related_step is not None else "",
                deliverable_key=job.deliverable_key,
                source="deliverable_generation_job",
                status=job.status,
                created_at=job.updated_at.isoformat(),
                metadata_keys=_safe_metadata_keys(job.request_metadata),
            )
        )

    for operation in operations:
        event_type = _operation_event_type(operation.action)
        if event_type == "other" and operation.status.value not in ERROR_STATES:
            continue
        product_key = _normalize_product_key((operation.request_payload or {}).get("product_key") or operation.action)
        run = latest_run_by_product.get(product_key)
        events.append(
            ProductBuildTelemetryEvent(
                event_key=operation.action or "stage_operation",
                event_type=event_type if operation.status.value not in ERROR_STATES else "error",
                workspace_id=record.workspace_id,
                session_id=record.id,
                product_key=product_key,
                run_id=str(run.id) if run is not None else "",
                stage_key=operation.stage_key,
                source="stage_operation",
                status=operation.status.value,
                created_at=operation.updated_at.isoformat(),
                metadata_keys=_safe_metadata_keys(operation.request_payload),
            )
        )

    events.sort(key=lambda item: item.created_at, reverse=True)
    return events[:limit]


def _build_product_summary(
    *,
    product_key: ProductBuildProductKey,
    runs: list[ProductBuildRunRecord],
    steps: list[ProductBuildStepRecord],
    jobs: list[DeliverableGenerationJobRecord],
    events: list[ProductBuildTelemetryEvent],
) -> ProductBuildTelemetryProductSummary:
    product_runs = [run for run in runs if _normalize_product_key(run.product_key or run.product_mode) == product_key]
    latest_run = product_runs[0] if product_runs else None
    run_ids = {run.id for run in product_runs}
    product_steps = [step for step in steps if step.run_id in run_ids]
    product_jobs = [job for job in jobs if _normalize_product_key(job.product_mode or (job.request_metadata or {}).get("product_key", "")) == product_key]
    product_events = [event for event in events if event.product_key == product_key]
    deliverable_keys = {
        item
        for item in [*(step.deliverable_key for step in product_steps), *(job.deliverable_key for job in product_jobs)]
        if item
    }

    tokens_input = sum(max(0, job.tokens_input) for job in product_jobs)
    tokens_output = sum(max(0, job.tokens_output) for job in product_jobs)
    latest_at = max(
        [
            *(run.updated_at for run in product_runs),
            *(step.updated_at for step in product_steps),
            *(job.updated_at for job in product_jobs),
        ],
        default=None,
    )

    return ProductBuildTelemetryProductSummary(
        product_key=product_key,
        product_label=PRODUCT_LABELS[product_key],
        run_id=str(latest_run.id) if latest_run is not None else "",
        lifecycle=latest_run.lifecycle if latest_run is not None else "not_started",
        run_count=len(product_runs),
        step_count=len(product_steps),
        deliverable_count=len(deliverable_keys),
        event_count=len(product_events),
        cta_count=sum(1 for event in product_events if event.event_type == "cta"),
        checkout_count=sum(1 for event in product_events if event.event_type == "checkout"),
        activation_count=sum(1 for event in product_events if event.event_type == "activation"),
        run_started_count=sum(1 for event in product_events if event.event_key == "run_started"),
        run_completed_count=sum(1 for event in product_events if event.event_key == "run_completed"),
        run_error_count=sum(1 for event in product_events if event.event_type == "error"),
        requires_attention_count=sum(1 for event in product_events if event.event_type == "attention"),
        retry_count=sum(1 for event in product_events if event.event_type == "retry"),
        resume_count=sum(1 for event in product_events if event.event_type == "resume"),
        run_duration_seconds=sum(_duration_seconds(run.started_at, run.completed_at or run.updated_at) for run in product_runs),
        deliverable_duration_seconds=sum(_duration_seconds(job.started_at, job.completed_at or job.updated_at) for job in product_jobs),
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        tokens_total=tokens_input + tokens_output,
        estimated_cost_usd=round(sum(max(0.0, job.estimated_cost_usd) for job in product_jobs), 6),
        latest_at=latest_at.isoformat() if latest_at is not None else "",
    )


def _latest_run_by_product(runs: list[ProductBuildRunRecord]) -> dict[ProductBuildProductKey, ProductBuildRunRecord]:
    by_product: dict[ProductBuildProductKey, ProductBuildRunRecord] = {}
    for run in runs:
        product_key = _normalize_product_key(run.product_key or run.product_mode)
        by_product.setdefault(product_key, run)
    return by_product


def _normalize_product_key(value: object) -> ProductBuildProductKey:
    token = str(value or "").strip().lower()
    if token in PRODUCT_MODE_TO_KEY:
        return PRODUCT_MODE_TO_KEY[token]
    if "acp" in token or "package" in token:
        return ProductBuildProductKey.acp
    if "pro" in token or "premium" in token:
        return ProductBuildProductKey.blueprint_pro
    return ProductBuildProductKey.blueprint_basic


def _classify_event(event_key: str, source: str) -> str:
    normalized = str(event_key or "").strip().lower()
    source_normalized = str(source or "").strip().lower()
    if normalized in CTA_EVENT_KEYS or normalized.endswith("_clicked") or normalized.endswith("_viewed"):
        return "cta"
    if normalized in CHECKOUT_EVENT_KEYS or source_normalized == "commerce_checkout":
        return "checkout"
    if normalized in ACTIVATION_EVENT_KEYS:
        return "activation"
    if "retry" in normalized or "reintentar" in normalized:
        return "retry"
    if "resume" in normalized or "reanudar" in normalized:
        return "resume"
    if "attention" in normalized or "blocked" in normalized:
        return "attention"
    if "error" in normalized or "failed" in normalized:
        return "error"
    return "other"


def _operation_event_type(action: str) -> str:
    normalized = str(action or "").strip().lower()
    if "retry" in normalized or "reintentar" in normalized or "regenerar" in normalized:
        return "retry"
    if "resume" in normalized or "reanudar" in normalized or "continuar" in normalized:
        return "resume"
    if "attention" in normalized:
        return "attention"
    return "other"


def _duration_seconds(started_at: datetime | None, finished_at: datetime | None) -> int:
    if started_at is None or finished_at is None:
        return 0
    return max(0, round((finished_at - started_at).total_seconds()))


def _safe_metadata_keys(value: object) -> list[str]:
    if not isinstance(value, dict):
        return []
    keys: list[str] = []
    for key in value.keys():
        token = str(key or "").strip()
        lowered = token.lower()
        if not token or any(fragment in lowered for fragment in SENSITIVE_METADATA_FRAGMENTS):
            continue
        keys.append(token)
    return sorted(set(keys))


def _build_warnings(
    *,
    products: list[ProductBuildTelemetryProductSummary],
    jobs: list[DeliverableGenerationJobRecord],
) -> list[str]:
    warnings: list[str] = []
    if any(job for job in jobs if (job.tokens_input + job.tokens_output) > 0 and job.estimated_cost_usd <= 0):
        warnings.append("Hay jobs con tokens reportados sin costo estimado; revisa pricing/provider para completar costo operativo.")
    if any(product.run_error_count for product in products):
        warnings.append("Existen errores de ejecucion en product builds; revisar eventos de tipo error antes de analizar conversion.")
    if any(product.requires_attention_count for product in products):
        warnings.append("Hay eventos requires_attention que pueden afectar conversion y finalizacion del funnel.")
    return warnings
