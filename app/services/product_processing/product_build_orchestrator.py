from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sqlmodel import Session

from app.models import CommercialTier, SessionRecord, UserRecord, WorkspaceRole
from app.services.commerce_service import role_for_user, tier_rank
from app.services.commercial_access import build_commercial_access_snapshot_v2
from app.services.deliverable_catalog.catalog_service import build_deliverable_catalog_response
from app.services.deliverable_catalog.contracts import (
    DeliverableCatalogItem,
    DeliverableGenerationResult,
    DeliverableGenerationTask,
)
from app.services.deliverable_catalog.generation_service import run_deliverable_generation_task
from app.services.diagram_center.persistence import DiagramGenerationJobRecord
from app.services.deliverable_catalog.persistence import DeliverableGenerationJobRecord
from app.services.product_processing.contracts import (
    ProductBuildLifecycle,
    ProductBuildProductKey,
    ProductBuildStatus,
)
from app.services.product_processing.persistence import ProductBuildRunRecord
from app.services.product_processing.product_build_run_service import (
    ensure_product_build_run,
    list_product_build_steps,
    update_product_build_run_state,
    upsert_product_build_step,
)
from app.services.product_processing.product_build_status_service import (
    PRODUCT_BUILD_META,
    build_product_build_status,
)


ACTIVE_JOB_STATES = {"queued", "generating", "updating", "running"}
ERROR_JOB_STATES = {"error", "failed", "requires_attention"}
COMPLETED_STEP_STATES = {"available", "completed", "skipped"}

JobRunner = Callable[[Session, DeliverableGenerationTask], tuple[DeliverableGenerationJobRecord, DeliverableGenerationResult | None]]


@dataclass(frozen=True)
class ProductBuildOrchestrationOptions:
    idempotency_key: str = ""
    execute_jobs: bool = False
    allow_llm: bool = False
    current_stage: str = ""
    context_payload: dict[str, Any] | None = None
    activation_payload: dict[str, Any] | None = None
    approved_context_refs: tuple[str, ...] = ()
    job_runner: JobRunner | None = None


def ensure_product_build_orchestration(
    db: Session,
    *,
    record: SessionRecord,
    product_key: ProductBuildProductKey | str,
    current_user: UserRecord | None = None,
    options: ProductBuildOrchestrationOptions | None = None,
    catalog_stage_override: str | None = None,
) -> ProductBuildStatus:
    resolved_options = options or ProductBuildOrchestrationOptions()
    normalized_product_key = _normalize_product_key(product_key)
    meta = PRODUCT_BUILD_META[normalized_product_key]
    access = build_commercial_access_snapshot_v2(db, record, current_user=current_user)
    if tier_rank(access.tier) < tier_rank(meta.required_tier):
        return build_product_build_status(
            db,
            record=record,
            product_key=normalized_product_key,
            current_user=current_user,
            catalog_stage_override=catalog_stage_override,
        )

    workspace_id = record.workspace_id
    if workspace_id is None:
        raise ValueError("Product build orchestration requires a workspace-scoped session.")

    stage_val = getattr(record.current_stage, "value", str(record.current_stage or "discover"))
    current_stage = _normalize_catalog_stage(catalog_stage_override or resolved_options.current_stage or stage_val)
    role = _resolve_role(db, record=record, current_user=current_user)
    catalog = build_deliverable_catalog_response(
        db,
        workspace_id=workspace_id,
        session_id=record.id,
        role=role,
        tier=access.tier,
        current_stage=current_stage,
    )
    expected_items = [item for item in catalog.entries if _is_expected_for_product(item, meta)]
    jobs_by_key = _latest_jobs_by_key(db, session_id=record.id)
    diagram_jobs_by_key = _latest_diagram_jobs_by_key(db, session_id=record.id)
    run_checkpoint = {
        "product_key": meta.product_key.value,
        "product_mode": meta.product_mode.value,
        "expected_deliverables": [item.key for item in expected_items],
        "catalog_stage": current_stage,
    }
    if resolved_options.activation_payload:
        run_checkpoint["activation"] = dict(resolved_options.activation_payload)

    run = ensure_product_build_run(
        db,
        workspace_id=workspace_id,
        session_id=record.id,
        product_key=meta.product_key,
        product_mode=meta.product_mode,
        entitlement_tier=access.tier,
        access_state="allowed",
        lifecycle=ProductBuildLifecycle.preparing,
        idempotency_key=_run_idempotency_key(record=record, product_key=meta.product_key, explicit_key=resolved_options.idempotency_key),
        created_by_user_id=current_user.id if current_user is not None else None,
        checkpoint_payload=run_checkpoint,
    )
    _merge_run_checkpoint(db, run=run, checkpoint_payload=run_checkpoint)

    _sync_expected_steps(
        db,
        run=run,
        expected_items=expected_items,
        jobs_by_key=jobs_by_key,
        diagram_jobs_by_key=diagram_jobs_by_key,
    )

    if resolved_options.execute_jobs:
        _execute_expected_jobs(
            db,
            run=run,
            expected_items=expected_items,
            existing_jobs_by_key=jobs_by_key,
            record=record,
            product_mode=meta.product_mode.value,
            tier=access.tier,
            current_stage=current_stage,
            current_user=current_user,
            options=resolved_options,
        )

    refreshed_jobs = _latest_jobs_by_key(db, session_id=record.id)
    refreshed_diagram_jobs = _latest_diagram_jobs_by_key(db, session_id=record.id)
    _sync_expected_steps(
        db,
        run=run,
        expected_items=expected_items,
        jobs_by_key=refreshed_jobs,
        diagram_jobs_by_key=refreshed_diagram_jobs,
    )
    _finalize_run_from_steps(db, run=run, expected_items=expected_items)
    db.flush()
    return build_product_build_status(
        db,
        record=record,
        product_key=meta.product_key,
        current_user=current_user,
        catalog_stage_override=catalog_stage_override,
    )


def reconcile_product_build_run(
    db: Session,
    *,
    record: SessionRecord,
    product_key: ProductBuildProductKey | str,
    current_user: UserRecord | None = None,
    catalog_stage_override: str | None = None,
) -> ProductBuildStatus:
    from app.services.product_processing.product_build_run_service import list_product_build_runs

    normalized_product_key = _normalize_product_key(product_key)
    meta = PRODUCT_BUILD_META[normalized_product_key]
    access = build_commercial_access_snapshot_v2(db, record, current_user=current_user)
    if tier_rank(access.tier) < tier_rank(meta.required_tier):
        return build_product_build_status(
            db,
            record=record,
            product_key=normalized_product_key,
            current_user=current_user,
            catalog_stage_override=catalog_stage_override,
        )

    workspace_id = record.workspace_id
    if workspace_id is None:
        return build_product_build_status(
            db,
            record=record,
            product_key=normalized_product_key,
            current_user=current_user,
            catalog_stage_override=catalog_stage_override,
        )

    runs = list_product_build_runs(
        db,
        workspace_id=workspace_id,
        session_id=record.id,
        product_key=meta.product_key,
    )
    if not runs:
        return build_product_build_status(
            db,
            record=record,
            product_key=normalized_product_key,
            current_user=current_user,
            catalog_stage_override=catalog_stage_override,
        )

    run = runs[0]
    stage_val = getattr(record.current_stage, "value", str(record.current_stage or "discover"))
    current_stage = str(
        catalog_stage_override
        or (run.checkpoint_payload or {}).get("catalog_stage")
        or _normalize_catalog_stage(stage_val)
    )
    role = _resolve_role(db, record=record, current_user=current_user)
    catalog = build_deliverable_catalog_response(
        db,
        workspace_id=workspace_id,
        session_id=record.id,
        role=role,
        tier=access.tier,
        current_stage=current_stage,
    )
    expected_items = [item for item in catalog.entries if _is_expected_for_product(item, meta)]
    refreshed_jobs = _latest_jobs_by_key(db, session_id=record.id)
    refreshed_diagram_jobs = _latest_diagram_jobs_by_key(db, session_id=record.id)
    _sync_expected_steps(
        db,
        run=run,
        expected_items=expected_items,
        jobs_by_key=refreshed_jobs,
        diagram_jobs_by_key=refreshed_diagram_jobs,
    )
    _finalize_run_from_steps(db, run=run, expected_items=expected_items)
    db.flush()
    return build_product_build_status(
        db,
        record=record,
        product_key=meta.product_key,
        current_user=current_user,
        catalog_stage_override=catalog_stage_override,
    )


def _normalize_product_key(product_key: ProductBuildProductKey | str) -> ProductBuildProductKey:
    return product_key if isinstance(product_key, ProductBuildProductKey) else ProductBuildProductKey(str(product_key))


def _run_idempotency_key(*, record: SessionRecord, product_key: ProductBuildProductKey, explicit_key: str) -> str:
    if explicit_key:
        return explicit_key
    return f"product-build:{record.id}:{product_key.value}"


def _merge_run_checkpoint(
    db: Session,
    *,
    run: ProductBuildRunRecord,
    checkpoint_payload: dict[str, Any],
) -> None:
    merged = {**(run.checkpoint_payload or {}), **checkpoint_payload}
    if merged == (run.checkpoint_payload or {}):
        return
    run.checkpoint_payload = merged
    db.add(run)
    db.flush()


def _resolve_role(db: Session, *, record: SessionRecord, current_user: UserRecord | None) -> WorkspaceRole:
    if current_user is None or record.workspace_id is None:
        return WorkspaceRole.admin
    return role_for_user(db, workspace_id=record.workspace_id, user_id=current_user.id) or WorkspaceRole.viewer


def _is_expected_for_product(item: DeliverableCatalogItem, meta) -> bool:
    return bool(set(item.product_scope).intersection(meta.included_scopes)) and tier_rank(item.required_tier) <= tier_rank(meta.required_tier)


def _normalize_catalog_stage(value: str) -> str:
    stage = str(value or "").strip().lower()
    if stage in {"discover", "define", "design", "tools", "memory", "estimate", "validate", "package"}:
        return stage
    legacy_map = {
        "draft_capture": "discover",
        "input_validation": "discover",
        "normalize_discovery": "discover",
        "build_canvas": "define",
        "build_blueprint": "design",
        "post_validation": "validate",
        "ready_for_export": "package",
    }
    return legacy_map.get(stage, "discover")


def _latest_jobs_by_key(db: Session, *, session_id) -> dict[str, DeliverableGenerationJobRecord]:
    from sqlmodel import select

    jobs = db.exec(
        select(DeliverableGenerationJobRecord)
        .where(DeliverableGenerationJobRecord.session_id == session_id)
        .order_by(DeliverableGenerationJobRecord.updated_at.desc())
    ).all()
    by_key: dict[str, DeliverableGenerationJobRecord] = {}
    for job in jobs:
        by_key.setdefault(job.deliverable_key, job)
    return by_key


def _latest_diagram_jobs_by_key(db: Session, *, session_id) -> dict[str, DiagramGenerationJobRecord]:
    from sqlmodel import select

    jobs = db.exec(
        select(DiagramGenerationJobRecord)
        .where(DiagramGenerationJobRecord.session_id == session_id)
        .order_by(DiagramGenerationJobRecord.updated_at.desc())
    ).all()
    by_key: dict[str, DiagramGenerationJobRecord] = {}
    for job in jobs:
        by_key.setdefault(job.diagram_key, job)
    return by_key


def _sync_expected_steps(
    db: Session,
    *,
    run: ProductBuildRunRecord,
    expected_items: list[DeliverableCatalogItem],
    jobs_by_key: dict[str, DeliverableGenerationJobRecord],
    diagram_jobs_by_key: dict[str, DiagramGenerationJobRecord],
) -> None:
    for index, item in enumerate(sorted(expected_items, key=lambda entry: entry.sort_order), start=1):
        job = jobs_by_key.get(item.key)
        diagram_job = _diagram_job_for_item(item, diagram_jobs_by_key) if job is None else None
        step_state = _step_state_for_item(item, job, diagram_job=diagram_job)
        upsert_product_build_step(
            db,
            run=run,
            step_key=f"deliverable:{item.key}",
            status=step_state,
            stage_key=item.stage,
            deliverable_key=item.key,
            job_id=(job.id if job is not None else diagram_job.id if diagram_job is not None else None),
            sequence=index,
            progress_percent=_progress_for_step_state(step_state),
            checkpoint_payload={
                "title": item.title,
                "type": item.deliverable_type.value,
                "product_scope": list(item.product_scope),
                "access_state": item.access.access_state,
                "job_source": "deliverable_catalog" if job is not None else "diagram_center" if diagram_job is not None else "",
            },
            error_payload=_error_payload_for_job(job or diagram_job),
        )


def _diagram_job_for_item(
    item: DeliverableCatalogItem,
    diagram_jobs_by_key: dict[str, DiagramGenerationJobRecord],
) -> DiagramGenerationJobRecord | None:
    if item.deliverable_type.value != "diagram":
        return None
    return diagram_jobs_by_key.get(item.key.removeprefix("diagram."))


def _step_state_for_item(
    item: DeliverableCatalogItem,
    job: DeliverableGenerationJobRecord | DiagramGenerationJobRecord | None,
    *,
    diagram_job: DiagramGenerationJobRecord | None = None,
) -> str:
    effective_job = job or diagram_job
    access_state = str(item.access.access_state or "")
    job_status = str(effective_job.status or "") if effective_job is not None else ""
    if access_state == "available" or job_status == "available":
        return "available"
    if access_state == "stage_locked":
        return "pending"
    if access_state in {"locked", "disabled"}:
        return "locked"
    if access_state == "quality_failed" or job_status in ERROR_JOB_STATES:
        return "requires_attention" if job_status == "requires_attention" else "error"
    if access_state == "stale":
        return "stale"
    if job_status in ACTIVE_JOB_STATES:
        return "generating" if job_status in {"generating", "updating"} else "queued"
    return "pending"


def _progress_for_step_state(state: str) -> int:
    if state in COMPLETED_STEP_STATES:
        return 100
    if state == "generating":
        return 50
    if state == "queued":
        return 10
    return 0


def _error_payload_for_job(job: DeliverableGenerationJobRecord | DiagramGenerationJobRecord | None) -> dict[str, Any]:
    if job is None or str(job.status or "") not in ERROR_JOB_STATES:
        return {}
    return {
        "code": job.error_code or str(job.status),
        "message": job.error_message or "Deliverable generation did not finish successfully.",
        "job_id": str(job.id),
    }


def _execute_expected_jobs(
    db: Session,
    *,
    run: ProductBuildRunRecord,
    expected_items: list[DeliverableCatalogItem],
    existing_jobs_by_key: dict[str, DeliverableGenerationJobRecord],
    record: SessionRecord,
    product_mode: str,
    tier: CommercialTier,
    current_stage: str,
    current_user: UserRecord | None,
    options: ProductBuildOrchestrationOptions,
) -> None:
    runner = options.job_runner or run_deliverable_generation_task
    for item in sorted(expected_items, key=lambda entry: entry.sort_order):
        existing = existing_jobs_by_key.get(item.key)
        if existing is not None and str(existing.status or "") in {"available", "generating", "queued", "updating"}:
            continue
        if not item.access.can_generate:
            continue
        task = DeliverableGenerationTask(
            workspace_id=run.workspace_id,
            session_id=record.id,
            deliverable_key=item.key,
            product_mode=product_mode,
            current_stage=current_stage,
            tier=tier,
            idempotency_key=f"{run.idempotency_key}:deliverable:{item.key}",
            requested_by_user_id=current_user.id if current_user is not None else None,
            context_payload=dict(options.context_payload or {}),
            approved_context_refs=list(options.approved_context_refs),
            allow_llm=options.allow_llm,
        )
        job, _ = runner(db, task)
        upsert_product_build_step(
            db,
            run=run,
            step_key=f"deliverable:{item.key}",
            status=_step_state_for_item(item, job),
            stage_key=item.stage,
            deliverable_key=item.key,
            job_id=job.id,
            sequence=item.sort_order,
            progress_percent=_progress_for_step_state(_step_state_for_item(item, job)),
            checkpoint_payload={"generation_requested": True, "title": item.title},
            error_payload=_error_payload_for_job(job),
        )


def _finalize_run_from_steps(db: Session, *, run: ProductBuildRunRecord, expected_items: list[DeliverableCatalogItem]) -> None:
    steps = list_product_build_steps(db, run_id=run.id)
    relevant_steps = [step for step in steps if step.deliverable_key]
    total_units = float(len(expected_items))
    completed_units = float(sum(1 for step in relevant_steps if step.status in COMPLETED_STEP_STATES))
    blocked_units = float(sum(1 for step in relevant_steps if step.status in {"error", "requires_attention", "locked"}))
    active_units = sum(1 for step in relevant_steps if step.status in ACTIVE_JOB_STATES or step.status == "generating")
    if blocked_units > 0:
        lifecycle = ProductBuildLifecycle.requires_attention
    elif total_units > 0 and completed_units >= total_units:
        lifecycle = ProductBuildLifecycle.completed
    elif active_units > 0:
        lifecycle = ProductBuildLifecycle.running
    elif completed_units > 0:
        lifecycle = ProductBuildLifecycle.partial
    else:
        lifecycle = ProductBuildLifecycle.ready_to_start
    update_product_build_run_state(
        db,
        run=run,
        lifecycle=lifecycle,
        completed_units=completed_units,
        total_units=total_units,
        blocked_units=blocked_units,
        checkpoint_payload={
            **(run.checkpoint_payload or {}),
            "completed_deliverables": [
                step.deliverable_key
                for step in relevant_steps
                if step.status in COMPLETED_STEP_STATES
            ],
            "blocked_deliverables": [
                step.deliverable_key
                for step in relevant_steps
                if step.status in {"error", "requires_attention", "locked"}
            ],
        },
    )
