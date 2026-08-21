from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlmodel import Session, select

from app.models import CommercialTier, utc_now
from app.services.product_processing.contracts import (
    ProductBuildLifecycle,
    ProductBuildProductKey,
    ProductProcessingMode,
    calculate_product_build_percent,
)
from app.services.product_processing.persistence import ProductBuildRunRecord, ProductBuildStepRecord


def _normalize_product_key(product_key: ProductBuildProductKey | str) -> str:
    return product_key.value if isinstance(product_key, ProductBuildProductKey) else str(product_key)


def _normalize_product_mode(product_mode: ProductProcessingMode | str) -> str:
    return product_mode.value if isinstance(product_mode, ProductProcessingMode) else str(product_mode)


def _normalize_lifecycle(lifecycle: ProductBuildLifecycle | str) -> str:
    return lifecycle.value if isinstance(lifecycle, ProductBuildLifecycle) else str(lifecycle)


def ensure_product_build_run(
    db: Session,
    *,
    workspace_id: UUID,
    session_id: UUID,
    product_key: ProductBuildProductKey | str,
    product_mode: ProductProcessingMode | str,
    idempotency_key: str,
    entitlement_tier: CommercialTier | str = CommercialTier.blueprint,
    access_state: str = "preview",
    lifecycle: ProductBuildLifecycle | str = ProductBuildLifecycle.ready_to_start,
    created_by_user_id: UUID | None = None,
    checkpoint_payload: dict[str, Any] | None = None,
) -> ProductBuildRunRecord:
    existing = db.exec(
        select(ProductBuildRunRecord).where(
            ProductBuildRunRecord.workspace_id == workspace_id,
            ProductBuildRunRecord.idempotency_key == idempotency_key,
        )
    ).first()
    if existing is not None:
        return existing

    tier_value = entitlement_tier.value if isinstance(entitlement_tier, CommercialTier) else str(entitlement_tier)
    record = ProductBuildRunRecord(
        workspace_id=workspace_id,
        session_id=session_id,
        product_key=_normalize_product_key(product_key),
        product_mode=_normalize_product_mode(product_mode),
        entitlement_tier=tier_value,
        access_state=access_state,
        lifecycle=_normalize_lifecycle(lifecycle),
        idempotency_key=idempotency_key,
        created_by_user_id=created_by_user_id,
        checkpoint_payload=dict(checkpoint_payload or {}),
    )
    db.add(record)
    db.flush()
    return record


def list_product_build_runs(
    db: Session,
    *,
    workspace_id: UUID,
    session_id: UUID,
    product_key: ProductBuildProductKey | str | None = None,
) -> list[ProductBuildRunRecord]:
    statement = select(ProductBuildRunRecord).where(
        ProductBuildRunRecord.workspace_id == workspace_id,
        ProductBuildRunRecord.session_id == session_id,
    )
    if product_key is not None:
        statement = statement.where(ProductBuildRunRecord.product_key == _normalize_product_key(product_key))
    return list(db.exec(statement.order_by(ProductBuildRunRecord.updated_at.desc())).all())


def list_product_build_steps(db: Session, *, run_id: UUID) -> list[ProductBuildStepRecord]:
    return list(
        db.exec(
            select(ProductBuildStepRecord)
            .where(ProductBuildStepRecord.run_id == run_id)
            .order_by(ProductBuildStepRecord.sequence.asc(), ProductBuildStepRecord.updated_at.asc())
        ).all()
    )


def upsert_product_build_step(
    db: Session,
    *,
    run: ProductBuildRunRecord,
    step_key: str,
    status: str,
    stage_key: str = "",
    deliverable_key: str = "",
    job_id: UUID | None = None,
    dependency_key: str = "",
    sequence: int = 0,
    progress_percent: int = 0,
    checkpoint_payload: dict[str, Any] | None = None,
    error_payload: dict[str, Any] | None = None,
) -> ProductBuildStepRecord:
    existing = db.exec(
        select(ProductBuildStepRecord).where(
            ProductBuildStepRecord.run_id == run.id,
            ProductBuildStepRecord.step_key == step_key,
        )
    ).first()
    record = existing or ProductBuildStepRecord(
        run_id=run.id,
        workspace_id=run.workspace_id,
        session_id=run.session_id,
        step_key=step_key,
    )
    record.status = status
    record.stage_key = stage_key
    record.deliverable_key = deliverable_key
    record.job_id = job_id
    record.dependency_key = dependency_key
    record.sequence = sequence
    record.progress_percent = min(max(int(progress_percent or 0), 0), 100)
    record.checkpoint_payload = dict(checkpoint_payload or {})
    record.error_payload = dict(error_payload or {})
    now = utc_now()
    if status in {"running", "generating"} and record.started_at is None:
        record.started_at = now
    if status in {"completed", "available", "skipped"}:
        record.completed_at = record.completed_at or now
    record.updated_at = now
    db.add(record)
    db.flush()
    return record


def update_product_build_run_state(
    db: Session,
    *,
    run: ProductBuildRunRecord,
    lifecycle: ProductBuildLifecycle | str,
    completed_units: float | None = None,
    total_units: float | None = None,
    blocked_units: float | None = None,
    checkpoint_payload: dict[str, Any] | None = None,
    error_payload: dict[str, Any] | None = None,
) -> ProductBuildRunRecord:
    now = utc_now()
    run.lifecycle = _normalize_lifecycle(lifecycle)
    if completed_units is not None:
        run.completed_units = completed_units
    if total_units is not None:
        run.total_units = total_units
    if blocked_units is not None:
        run.blocked_units = blocked_units
    run.progress_percent = calculate_product_build_percent(run.completed_units, run.total_units)
    if checkpoint_payload is not None:
        run.checkpoint_payload = dict(checkpoint_payload)
    if error_payload is not None:
        run.error_payload = dict(error_payload)
    if run.lifecycle in {ProductBuildLifecycle.running.value, ProductBuildLifecycle.preparing.value} and run.started_at is None:
        run.started_at = now
    if run.lifecycle == ProductBuildLifecycle.requires_attention.value:
        run.requires_attention_at = run.requires_attention_at or now
    if run.lifecycle == ProductBuildLifecycle.completed.value:
        run.completed_at = run.completed_at or now
    run.updated_at = now
    db.add(run)
    db.flush()
    return run
