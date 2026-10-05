from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from app.core.config import get_settings
from app.db import engine
from app.models import CommercialTier, SessionRecord, UserRecord, WorkspaceRole, utc_now
from app.services.commerce_service import role_for_user, tier_rank
from app.services.commercial_access import build_commercial_access_snapshot_v2
from app.services.deliverable_catalog.catalog_service import build_deliverable_catalog_response
from app.services.deliverable_catalog.contracts import (
    DeliverableCatalogItem,
    DeliverableGenerationResult,
    DeliverableGenerationTask,
)
from app.services.deliverable_catalog.deterministic_builders import supports_deterministic_deliverable
from app.services.deliverable_catalog.generation_service import run_deliverable_generation_task
from app.services.deliverable_catalog.persistence import DeliverableGenerationJobRecord
from app.services.deliverable_catalog.registry_service import (
    get_registry_entry as get_deliverable_registry_entry,
    list_registry_entries as list_deliverable_registry_entries,
    resolve_product_delivery_plan,
)
from app.services.diagram_center.deterministic_builders import supports_deterministic_diagram
from app.services.diagram_center.generation_service import create_generation_job, run_generation_job
from app.services.diagram_center.persistence import DiagramGenerationJobRecord
from app.services.product_processing.contracts import (
    ProductBuildLifecycle,
    ProductBuildProcessingQueueMode,
    ProductBuildProductKey,
    ProductBuildStatus,
)
from app.services.product_processing.persistence import ProductBuildRunRecord, ProductBuildStepRecord
from app.services.product_processing.product_build_run_service import (
    ensure_product_build_run,
    list_product_build_runs,
    list_product_build_steps,
    update_product_build_run_state,
    upsert_product_build_step,
)
from app.services.product_processing.product_build_status_service import (
    PRODUCT_BUILD_META,
    build_product_build_status,
)
from app.services.product_processing.approved_context_service import build_approved_deliverable_context


ACTIVE_JOB_STATES = {"queued", "generating", "updating", "running"}
ERROR_JOB_STATES = {"error", "failed", "requires_attention"}
COMPLETED_STEP_STATES = {"available", "completed", "skipped"}
QUEUE_ACTIVE_STEP_STATES = {"queued", "running", "generating"}
QUEUE_FAILURE_STEP_STATES = {"error", "failed", "requires_attention", "locked"}
QUEUE_ELIGIBLE_STATES = {"pending", "stale"}
QUEUE_RETRY_ONLY_STATES = {"error", "requires_attention"}
QUEUE_ACTIVE_STATUSES = {"queued", "running"}
MAX_PROCESSING_ATTEMPTS = 2
ORPHANED_JOB_TIMEOUT = timedelta(minutes=15)
MAX_PRODUCT_BUILD_BATCH_SIZE = 3
DEFAULT_PARALLEL_TOKEN_BUDGET = 36_000
DEFAULT_PARALLEL_COMPLEXITY_BUDGET = 6
HEAVY_ARTIFACT_DURATION_SECONDS = 180
ACP_NONBLOCKING_DIAGRAM_GROUPS = {"large", "isolated"}
ACP_NONBLOCKING_POLICY_KEY = "acp_core_first_nonblocking_diagrams"

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


@dataclass(frozen=True)
class ProductBuildArtifactEstimate:
    deliverable_key: str
    group: str
    estimated_tokens: int
    estimated_seconds: int
    complexity_score: int
    dependency_count: int
    can_parallelize: bool
    resource_units: int
    isolation_reason: str = ""
    historical_duration_seconds: int = 0
    generation_mode: str = "deterministic"


@dataclass(frozen=True)
class ProductBuildCompletionScope:
    blocking_keys: frozenset[str]
    nonblocking_payload: tuple[dict[str, Any], ...]

    @property
    def nonblocking_keys(self) -> frozenset[str]:
        return frozenset(str(item.get("deliverable_key") or "") for item in self.nonblocking_payload)


def seal_product_build_run(db: Session, *, run: ProductBuildRunRecord) -> None:
    """Mark a blueprint_pro ProductBuildRun as sealed.

    A sealed run prevents further user-triggered regeneration of LEAN work
    stages (start / resume / process_pending / retry_failed actions). The seal
    is set automatically once all steps complete without errors.
    """
    run.is_sealed = True
    run.updated_at = utc_now()
    db.add(run)


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
    run_idempotency_key = _run_idempotency_key(
        record=record,
        product_key=meta.product_key,
        explicit_key=resolved_options.idempotency_key,
    )
    existing_run = _find_product_build_run_by_idempotency(
        db,
        workspace_id=workspace_id,
        idempotency_key=run_idempotency_key,
    )
    expected_items, delivery_plan_payload = _expected_items_for_product_run(
        catalog_entries=catalog.entries,
        meta=meta,
        run=existing_run,
    )
    jobs_by_key = _latest_jobs_by_key(db, session_id=record.id)
    diagram_jobs_by_key = _latest_diagram_jobs_by_key(db, session_id=record.id)
    run_checkpoint = {
        "product_key": meta.product_key.value,
        "product_mode": meta.product_mode.value,
        "expected_deliverables": [item.key for item in expected_items],
        "catalog_stage": current_stage,
    }
    if delivery_plan_payload is not None:
        run_checkpoint["delivery_plan"] = delivery_plan_payload
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
        idempotency_key=run_idempotency_key,
        created_by_user_id=current_user.id if current_user is not None else None,
        checkpoint_payload=run_checkpoint,
    )
    _merge_run_checkpoint(db, run=run, checkpoint_payload=run_checkpoint)
    _recover_orphaned_processing_queue(db, run=run)

    _sync_expected_steps(
        db,
        run=run,
        expected_items=expected_items,
        jobs_by_key=jobs_by_key,
        diagram_jobs_by_key=diagram_jobs_by_key,
    )

    if resolved_options.execute_jobs:
        # If the run is sealed (all steps completed successfully after Blueprint Pro
        # approval), block user-triggered job execution to prevent regeneration.
        if getattr(run, "is_sealed", False):
            return build_product_build_status(
                db,
                record=record,
                product_key=normalized_product_key,
                current_user=current_user,
                catalog_stage_override=catalog_stage_override,
            )
        if resolved_options.job_runner is not None:
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
        else:
            _execute_processing_queue_inline(
                db,
                record=record,
                product_key=normalized_product_key,
                current_user=current_user,
                allow_llm=resolved_options.allow_llm,
                catalog_stage_override=current_stage,
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
    _recover_orphaned_processing_queue(db, run=run)
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
    expected_items, _ = _expected_items_for_product_run(
        catalog_entries=catalog.entries,
        meta=meta,
        run=run,
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
        product_key=normalized_product_key,
        current_user=current_user,
        catalog_stage_override=catalog_stage_override,
    )


def enqueue_product_build_processing(
    db: Session,
    *,
    record: SessionRecord,
    product_key: ProductBuildProductKey | str,
    current_user: UserRecord | None = None,
    mode: ProductBuildProcessingQueueMode | str = ProductBuildProcessingQueueMode.process_pending,
    allow_llm: bool = False,
    activation_payload: dict[str, Any] | None = None,
    catalog_stage_override: str | None = None,
) -> tuple[ProductBuildRunRecord | None, ProductBuildStatus, bool]:
    normalized_product_key = _normalize_product_key(product_key)
    resolved_mode = _normalize_queue_mode(mode)

    # Block re-enqueuing if the blueprint_pro run is already sealed.
    # The seal is set after first successful completion (post access-approval generation).
    if normalized_product_key == ProductBuildProductKey.blueprint_pro and record.workspace_id is not None:
        existing_runs = list_product_build_runs(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            product_key=normalized_product_key,
        )
        if existing_runs and getattr(existing_runs[0], "is_sealed", False):
            return (
                existing_runs[0],
                build_product_build_status(
                    db,
                    record=record,
                    product_key=normalized_product_key,
                    current_user=current_user,
                    catalog_stage_override=catalog_stage_override,
                ),
                False,
            )

    if normalized_product_key == ProductBuildProductKey.acp:
        from app.services.product_processing.acp_product_orchestration_service import ensure_acp_product_orchestration

        status = ensure_acp_product_orchestration(
            db,
            record=record,
            current_user=current_user,
            execute_jobs=False,
            allow_llm=allow_llm,
            activation_payload=activation_payload or {"source": f"product_build_queue:{resolved_mode.value}"},
            catalog_stage_override=catalog_stage_override or "package",
        )
    else:
        status = ensure_product_build_orchestration(
            db,
            record=record,
            product_key=normalized_product_key,
            current_user=current_user,
            options=ProductBuildOrchestrationOptions(
                current_stage=catalog_stage_override or "",
                allow_llm=allow_llm,
                activation_payload=activation_payload,
            ),
            catalog_stage_override=catalog_stage_override,
        )

    if status.entitlement.access_state != "allowed":
        return None, status, False

    if record.workspace_id is None:
        return None, status, False

    run = ensure_product_build_run(
        db,
        workspace_id=record.workspace_id,
        session_id=record.id,
        product_key=normalized_product_key,
        product_mode=PRODUCT_BUILD_META[normalized_product_key].product_mode,
        entitlement_tier=status.entitlement.tier,
        access_state=status.entitlement.access_state,
        lifecycle=ProductBuildLifecycle.preparing,
        idempotency_key=_run_idempotency_key(record=record, product_key=normalized_product_key, explicit_key=""),
        created_by_user_id=current_user.id if current_user is not None else None,
        checkpoint_payload={
            "product_key": normalized_product_key.value,
            "product_mode": PRODUCT_BUILD_META[normalized_product_key].product_mode.value,
            "catalog_stage": _normalize_catalog_stage(catalog_stage_override or "package"),
        },
    )

    _recover_orphaned_processing_queue(db, run=run)
    current_queue = _processing_queue_checkpoint(run)
    if str(run.lifecycle or "") in {
        ProductBuildLifecycle.queued.value,
        ProductBuildLifecycle.preparing.value,
        ProductBuildLifecycle.running.value,
    } and str(current_queue.get("status") or "") in QUEUE_ACTIVE_STATUSES:
        return (
            run,
            _refresh_status_for_product(
                db,
                record=record,
                product_key=normalized_product_key,
                current_user=current_user,
                catalog_stage_override=catalog_stage_override,
            ),
            False,
        )

    access = build_commercial_access_snapshot_v2(db, record, current_user=current_user)
    role = _resolve_role(db, record=record, current_user=current_user)
    current_stage = _normalize_catalog_stage(
        catalog_stage_override
        or (run.checkpoint_payload or {}).get("catalog_stage")
        or getattr(record.current_stage, "value", str(record.current_stage or "discover"))
    )
    catalog = build_deliverable_catalog_response(
        db,
        workspace_id=record.workspace_id,
        session_id=record.id,
        role=role,
        tier=access.tier,
        current_stage=current_stage,
    )
    meta = PRODUCT_BUILD_META[normalized_product_key]
    expected_items, delivery_plan_payload = _expected_items_for_product_run(
        catalog_entries=catalog.entries,
        meta=meta,
        run=run,
    )
    if delivery_plan_payload is not None and not (run.checkpoint_payload or {}).get("delivery_plan"):
        _merge_run_checkpoint(
            db,
            run=run,
            checkpoint_payload={
                "expected_deliverables": [item.key for item in expected_items],
                "delivery_plan": delivery_plan_payload,
            },
        )
    jobs_by_key = _latest_jobs_by_key(db, session_id=record.id)
    diagram_jobs_by_key = _latest_diagram_jobs_by_key(db, session_id=record.id)
    _sync_expected_steps(
        db,
        run=run,
        expected_items=expected_items,
        jobs_by_key=jobs_by_key,
        diagram_jobs_by_key=diagram_jobs_by_key,
    )
    steps_by_key = {step.step_key: step for step in list_product_build_steps(db, run_id=run.id)}
    completion_scope = _build_completion_scope(db, run=run, expected_items=expected_items)
    selected_items = _select_processing_items(
        run=run,
        expected_items=expected_items,
        jobs_by_key=jobs_by_key,
        diagram_jobs_by_key=diagram_jobs_by_key,
        steps_by_key=steps_by_key,
        mode=resolved_mode,
        nonblocking_keys=completion_scope.nonblocking_keys,
    )

    if not selected_items:
        update_product_build_run_state(
            db,
            run=run,
            lifecycle=run.lifecycle,
            checkpoint_payload={
                **(run.checkpoint_payload or {}),
                "processing_queue": {
                    "queue_id": str(uuid4()),
                    "mode": resolved_mode.value,
                    "status": "completed",
                    "selected_deliverable_keys": [],
                    "completion_policy": ACP_NONBLOCKING_POLICY_KEY if _is_acp_run(run) else "all_expected_deliverables",
                    "deferred_nonblocking_deliverable_keys": sorted(completion_scope.nonblocking_keys),
                    "deferred_nonblocking_deliverables": list(completion_scope.nonblocking_payload),
                    "summary": (
                        "No hay entregables bloqueantes pendientes. Los artefactos ACP extendidos quedaron diferidos."
                        if completion_scope.nonblocking_keys
                        else "No hay entregables pendientes, no generados o fallidos para procesar."
                    ),
                },
            },
            error_payload={},
        )
        return (
            run,
            _refresh_status_for_product(
                db,
                record=record,
                product_key=normalized_product_key,
                current_user=current_user,
                catalog_stage_override=catalog_stage_override,
            ),
            False,
        )

    queue_id = str(uuid4())
    items_by_key = {item.key: item for item in expected_items}
    selected_estimates = {
        item.key: _estimate_artifact_processing(db, run=run, item=item, items_by_key=items_by_key)
        for item in selected_items
    }
    for sequence, item in enumerate(selected_items, start=1):
        existing_step = steps_by_key.get(f"deliverable:{item.key}")
        upsert_product_build_step(
            db,
            run=run,
            step_key=f"deliverable:{item.key}",
            status="queued",
            stage_key=item.stage,
            deliverable_key=item.key,
            sequence=sequence,
            progress_percent=10,
            checkpoint_payload={
                **(existing_step.checkpoint_payload or {} if existing_step is not None else {}),
                "title": item.title,
                "type": item.deliverable_type.value,
                "product_scope": list(item.product_scope),
                "access_state": item.access.access_state,
                "attempt_count": 0,
                "retried": False,
                "queue_id": queue_id,
                "queue_mode": resolved_mode.value,
                "queue_selected": True,
                "job_source": "diagram_center" if item.deliverable_type.value == "diagram" else "deliverable_catalog",
                "processing_estimate": _artifact_estimate_payload(selected_estimates[item.key]),
            },
            error_payload={},
        )

    batch_size = _product_build_batch_size(db.get_bind())
    update_product_build_run_state(
        db,
        run=run,
        lifecycle=ProductBuildLifecycle.queued,
        checkpoint_payload={
            **(run.checkpoint_payload or {}),
            "processing_queue": {
                "queue_id": queue_id,
                "mode": resolved_mode.value,
                "status": "queued",
                "selected_deliverable_keys": [item.key for item in selected_items],
                "retry_deliverable_keys": [],
                "allow_llm": allow_llm,
                "strategy": "dynamic_parallelism" if _dynamic_parallelism_enabled() and batch_size > 1 else "sequential",
                "max_parallel": batch_size,
                "parallel_token_budget": _parallel_token_budget(),
                "parallel_complexity_budget": _parallel_complexity_budget(),
                "completion_policy": ACP_NONBLOCKING_POLICY_KEY if _is_acp_run(run) else "all_expected_deliverables",
                "deferred_nonblocking_deliverable_keys": sorted(completion_scope.nonblocking_keys),
                "deferred_nonblocking_deliverables": list(completion_scope.nonblocking_payload),
                "profile_counts": _estimate_profile_counts(list(selected_estimates.values())),
                "summary": (
                    f"Se encolaron {len(selected_items)} entregables con estrategia de paralelizacion dinamica."
                    if _dynamic_parallelism_enabled() and batch_size > 1
                    else f"Se encolaron {len(selected_items)} entregables para procesamiento secuencial."
                ),
            },
        },
        error_payload={},
    )
    return (
        run,
        _refresh_status_for_product(
            db,
            record=record,
            product_key=normalized_product_key,
            current_user=current_user,
            catalog_stage_override=catalog_stage_override,
        ),
        True,
    )


def run_product_build_processing(
    run_id: UUID,
    database_engine: Engine | None = None,
) -> None:
    resolved_engine = database_engine or engine
    with Session(resolved_engine) as db:
        run = db.get(ProductBuildRunRecord, run_id)
        if run is None:
            return
        queue_checkpoint = _processing_queue_checkpoint(run)
        if str(queue_checkpoint.get("status") or "") not in QUEUE_ACTIVE_STATUSES:
            return

        record = db.get(SessionRecord, run.session_id)
        if record is None or record.workspace_id != run.workspace_id:
            _finalize_processing_queue(
                db,
                run=run,
                status="completed_with_errors",
                summary="El proyecto asociado ya no esta disponible para continuar el procesamiento.",
                failed_keys=[str(value) for value in queue_checkpoint.get("selected_deliverable_keys", [])],
            )
            return

        access = build_commercial_access_snapshot_v2(db, record, current_user=None)
        role = _resolve_role(db, record=record, current_user=None)
        current_stage = _normalize_catalog_stage(
            (run.checkpoint_payload or {}).get("catalog_stage")
            or getattr(record.current_stage, "value", str(record.current_stage or "discover"))
        )
        catalog = build_deliverable_catalog_response(
            db,
            workspace_id=run.workspace_id,
            session_id=record.id,
            role=role,
            tier=access.tier,
            current_stage=current_stage,
        )
        meta = PRODUCT_BUILD_META[_normalize_product_key(run.product_key)]
        expected_items, _ = _expected_items_for_product_run(
            catalog_entries=catalog.entries,
            meta=meta,
            run=run,
        )
        items_by_key = {item.key: item for item in expected_items}
        completion_scope = _build_completion_scope(db, run=run, expected_items=expected_items)
        selected_keys = [str(key) for key in queue_checkpoint.get("selected_deliverable_keys", []) if str(key) in items_by_key]
        selected_keys = [key for key in selected_keys if key not in completion_scope.nonblocking_keys]
        ordered_items = _topologically_sort_items([items_by_key[key] for key in selected_keys])

        _update_processing_queue_checkpoint(
            db,
            run=run,
            status="running",
            current_deliverable_key="",
            summary=_processing_batch_summary(total_count=len(selected_keys), batch_size=_product_build_batch_size(db.get_bind())),
        )
        update_product_build_run_state(
            db,
            run=run,
            lifecycle=ProductBuildLifecycle.running,
            checkpoint_payload=run.checkpoint_payload,
        )
        db.commit()

        failed_keys = _process_queue_items_in_batches(
            db,
            database_engine=resolved_engine,
            run=run,
            record=record,
            ordered_items=ordered_items,
            items_by_key=items_by_key,
            allow_llm=bool(queue_checkpoint.get("allow_llm")),
            phase="initial",
            total_count=len(selected_keys),
        )

        retry_items = _topologically_sort_items([items_by_key[key] for key in failed_keys if key in items_by_key])
        _update_processing_queue_checkpoint(
            db,
            run=run,
            retry_deliverable_keys=[item.key for item in retry_items],
            summary=(
                f"Reintentando {len(retry_items)} entregables fallidos."
                if retry_items
                else "La primera pasada finalizo sin fallos que requieran reintento."
            ),
        )
        db.commit()

        remaining_failures = _process_queue_items_in_batches(
            db,
            database_engine=resolved_engine,
            run=run,
            record=record,
            ordered_items=retry_items,
            items_by_key=items_by_key,
            allow_llm=bool(queue_checkpoint.get("allow_llm")),
            phase="retry",
            total_count=len(retry_items),
        )

        terminal_queue_status = "completed_with_errors" if remaining_failures else "completed"
        _update_processing_queue_checkpoint(
            db,
            run=run,
            status=terminal_queue_status,
            current_deliverable_key="",
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
        _finalize_processing_queue(
            db,
            run=run,
            status=terminal_queue_status,
            summary=(
                f"Se completaron {len(selected_keys) - len(remaining_failures)} de {len(selected_keys)} entregables; {len(remaining_failures)} siguen fallando."
                if remaining_failures
                else f"Se completaron correctamente los {len(selected_keys)} entregables seleccionados."
            ),
            failed_keys=remaining_failures,
        )


def _normalize_product_key(product_key: ProductBuildProductKey | str) -> ProductBuildProductKey:
    return product_key if isinstance(product_key, ProductBuildProductKey) else ProductBuildProductKey(str(product_key))


def _normalize_queue_mode(mode: ProductBuildProcessingQueueMode | str) -> ProductBuildProcessingQueueMode:
    return mode if isinstance(mode, ProductBuildProcessingQueueMode) else ProductBuildProcessingQueueMode(str(mode))


def _product_build_batch_size(bind: Any = None) -> int:
    if bind is not None:
        url = getattr(bind, "url", None)
        if url is not None and getattr(url, "drivername", "").startswith("sqlite"):
            db_name = getattr(url, "database", None)
            if not db_name or db_name == ":memory:":
                return 1
    try:
        configured = int(getattr(get_settings(), "product_build_batch_size", 1) or 1)
    except (TypeError, ValueError):
        configured = 1
    return max(1, min(MAX_PRODUCT_BUILD_BATCH_SIZE, configured))


def _dynamic_parallelism_enabled() -> bool:
    return bool(getattr(get_settings(), "product_build_dynamic_parallelism_enabled", True))


def _parallel_token_budget() -> int:
    try:
        configured = int(getattr(get_settings(), "product_build_parallel_token_budget", DEFAULT_PARALLEL_TOKEN_BUDGET) or 0)
    except (TypeError, ValueError):
        configured = DEFAULT_PARALLEL_TOKEN_BUDGET
    return max(8_000, configured)


def _parallel_complexity_budget() -> int:
    try:
        configured = int(
            getattr(get_settings(), "product_build_parallel_complexity_budget", DEFAULT_PARALLEL_COMPLEXITY_BUDGET) or 0
        )
    except (TypeError, ValueError):
        configured = DEFAULT_PARALLEL_COMPLEXITY_BUDGET
    return max(2, configured)


def _processing_batch_summary(*, total_count: int, batch_size: int) -> str:
    if batch_size <= 1:
        return f"Procesando {total_count} entregables de forma secuencial."
    if not _dynamic_parallelism_enabled():
        return f"Procesando {total_count} entregables en lotes fijos de hasta {batch_size}."
    return f"Procesando {total_count} entregables con paralelizacion dinamica de hasta {batch_size}."


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


def _find_product_build_run_by_idempotency(
    db: Session,
    *,
    workspace_id: UUID,
    idempotency_key: str,
) -> ProductBuildRunRecord | None:
    return db.exec(
        select(ProductBuildRunRecord).where(
            ProductBuildRunRecord.workspace_id == workspace_id,
            ProductBuildRunRecord.idempotency_key == idempotency_key,
        )
    ).first()


def _checkpoint_expected_keys(run: ProductBuildRunRecord | None) -> list[str] | None:
    if run is None:
        return None
    checkpoint = run.checkpoint_payload or {}
    delivery_plan = checkpoint.get("delivery_plan")
    if isinstance(delivery_plan, dict) and "generated_deliverable_keys" in delivery_plan:
        return [str(key) for key in delivery_plan.get("generated_deliverable_keys", []) if str(key).strip()]
    if "expected_deliverables" in checkpoint:
        return [str(key) for key in checkpoint.get("expected_deliverables", []) if str(key).strip()]
    return None


def _items_for_expected_keys(
    *,
    catalog_entries: list[DeliverableCatalogItem],
    meta,
    expected_keys: list[str],
) -> list[DeliverableCatalogItem]:
    items_by_key = {item.key: item for item in catalog_entries}
    ordered_items: list[DeliverableCatalogItem] = []
    for key in expected_keys:
        item = items_by_key.get(key)
        if item is None:
            continue
        if not _is_expected_for_product(item, meta):
            continue
        ordered_items.append(item)
    return ordered_items


def _legacy_expected_items(
    *,
    catalog_entries: list[DeliverableCatalogItem],
    meta,
) -> list[DeliverableCatalogItem]:
    return [item for item in catalog_entries if _is_expected_for_product(item, meta)]


def _expected_items_for_product_run(
    *,
    catalog_entries: list[DeliverableCatalogItem],
    meta,
    run: ProductBuildRunRecord | None,
) -> tuple[list[DeliverableCatalogItem], dict[str, Any] | None]:
    frozen_keys = _checkpoint_expected_keys(run)
    if frozen_keys is not None:
        return _items_for_expected_keys(catalog_entries=catalog_entries, meta=meta, expected_keys=frozen_keys), None

    if run is not None:
        return _legacy_expected_items(catalog_entries=catalog_entries, meta=meta), None

    plan = resolve_product_delivery_plan(
        meta.product_key.value,
        registry_entries=list_deliverable_registry_entries(include_inactive=True),
        confirmed_signals=[],
    )
    expected_items = _items_for_expected_keys(
        catalog_entries=catalog_entries,
        meta=meta,
        expected_keys=list(plan.generated_keys),
    )
    plan_payload = plan.checkpoint_payload()
    plan_payload["generated_deliverable_keys"] = [item.key for item in expected_items]
    return expected_items, plan_payload


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
    existing_steps = {step.step_key: step for step in list_product_build_steps(db, run_id=run.id)}
    queue_active = str(_processing_queue_checkpoint(run).get("status") or "") in QUEUE_ACTIVE_STATUSES
    completion_scope = _build_completion_scope(db, run=run, expected_items=expected_items)
    nonblocking_by_key = {
        str(item.get("deliverable_key") or ""): dict(item)
        for item in completion_scope.nonblocking_payload
    }
    for index, item in enumerate(expected_items, start=1):
        step_key = f"deliverable:{item.key}"
        existing_step = existing_steps.get(step_key)
        job = jobs_by_key.get(item.key)
        diagram_job = _diagram_job_for_item(item, diagram_jobs_by_key) if job is None else None
        step_state = _step_state_for_item(item, job, diagram_job=diagram_job, existing_step=existing_step)
        if (
            queue_active
            and existing_step is not None
            and bool((existing_step.checkpoint_payload or {}).get("queue_selected"))
            and str(existing_step.status or "") in QUEUE_ACTIVE_STEP_STATES
        ):
            step_state = str(existing_step.status or step_state)
        job_source = ""
        if job is not None:
            job_source = "deliverable_catalog"
        elif diagram_job is not None:
            job_source = "diagram_center"
        elif existing_step is not None:
            job_source = str((existing_step.checkpoint_payload or {}).get("job_source") or "")
        error_payload = _error_payload_for_job(job or diagram_job)
        nonblocking_payload = nonblocking_by_key.get(item.key)
        if (
            nonblocking_payload is not None
            and str(step_state or "") in {"pending", "stale", "error", "failed", "requires_attention", "locked"}
        ):
            step_state = "skipped"
            error_payload = {}
        if not error_payload and existing_step is not None and step_state in QUEUE_FAILURE_STEP_STATES:
            error_payload = dict(existing_step.error_payload or {})
        upsert_product_build_step(
            db,
            run=run,
            step_key=step_key,
            status=step_state,
            stage_key=item.stage,
            deliverable_key=item.key,
            job_id=(job.id if job is not None else diagram_job.id if diagram_job is not None else None),
            sequence=index,
            progress_percent=_progress_for_step_state(step_state),
            checkpoint_payload={
                **(existing_step.checkpoint_payload or {} if existing_step is not None else {}),
                "title": item.title,
                "type": item.deliverable_type.value,
                "product_scope": list(item.product_scope),
                "access_state": item.access.access_state,
                "job_source": job_source,
                "blocking_for_product": nonblocking_payload is None,
                "deferred_nonblocking": nonblocking_payload is not None,
                "nonblocking_policy_key": str(nonblocking_payload.get("policy_key") or "")
                if nonblocking_payload is not None
                else "",
                "nonblocking_reason": str(nonblocking_payload.get("reason") or "")
                if nonblocking_payload is not None
                else "",
            },
            error_payload=error_payload,
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
    existing_step: ProductBuildStepRecord | None = None,
) -> str:
    effective_job = job or diagram_job
    access_state = str(item.access.access_state or "")
    job_status = str(effective_job.status or "") if effective_job is not None else ""
    existing_status = str(existing_step.status or "") if existing_step is not None else ""
    if access_state == "available" or job_status == "available":
        return "available"
    if access_state in {"locked", "disabled"}:
        return "locked"
    if access_state == "quality_failed" or job_status in ERROR_JOB_STATES:
        return "requires_attention" if job_status == "requires_attention" else "error"
    if access_state == "stale":
        return "stale"
    if job_status in ACTIVE_JOB_STATES:
        return "generating" if job_status in {"generating", "updating", "running"} else "queued"
    if effective_job is None and existing_status in {*QUEUE_ACTIVE_STEP_STATES, *QUEUE_FAILURE_STEP_STATES}:
        return existing_status
    if access_state == "stage_locked":
        return "pending"
    return "pending"


def _progress_for_step_state(state: str) -> int:
    if state in COMPLETED_STEP_STATES:
        return 100
    if state == "generating":
        return 50
    if state == "running":
        return 35
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
    for item in expected_items:
        existing = existing_jobs_by_key.get(item.key)
        if existing is not None and str(existing.status or "") in {"available", "generating", "queued", "updating"}:
            continue
        if not item.access.can_generate:
            continue
        context_payload, approved_context_refs = build_approved_deliverable_context(
            db,
            record=record,
            deliverable_key=item.key,
        )
        task = DeliverableGenerationTask(
            workspace_id=run.workspace_id,
            session_id=record.id,
            deliverable_key=item.key,
            product_mode=product_mode,
            current_stage=current_stage,
            tier=tier,
            idempotency_key=f"{run.idempotency_key}:deliverable:{item.key}",
            requested_by_user_id=current_user.id if current_user is not None else None,
            context_payload=context_payload,
            approved_context_refs=approved_context_refs,
            allow_llm=options.allow_llm,
        )
        job, _ = runner(db, task)
        step_state = _step_state_for_item(item, job)
        upsert_product_build_step(
            db,
            run=run,
            step_key=f"deliverable:{item.key}",
            status=step_state,
            stage_key=item.stage,
            deliverable_key=item.key,
            job_id=job.id,
            sequence=item.sort_order,
            progress_percent=_progress_for_step_state(step_state),
            checkpoint_payload={"generation_requested": True, "title": item.title},
            error_payload=_error_payload_for_job(job),
        )


def _finalize_run_from_steps(db: Session, *, run: ProductBuildRunRecord, expected_items: list[DeliverableCatalogItem]) -> None:
    steps = list_product_build_steps(db, run_id=run.id)
    relevant_steps = [step for step in steps if step.deliverable_key]
    completion_scope = _build_completion_scope(db, run=run, expected_items=expected_items)
    blocking_steps = [step for step in relevant_steps if str(step.deliverable_key or "") in completion_scope.blocking_keys]
    total_units = float(len(completion_scope.blocking_keys))
    completed_units = float(sum(1 for step in blocking_steps if step.status in COMPLETED_STEP_STATES))
    blocked_units = float(sum(1 for step in blocking_steps if step.status in {"error", "requires_attention", "locked"}))
    active_units = sum(1 for step in blocking_steps if step.status in {"queued", "running", "generating"})
    queue_status = str(_processing_queue_checkpoint(run).get("status") or "")

    if queue_status in QUEUE_ACTIVE_STATUSES:
        lifecycle = ProductBuildLifecycle.running if active_units > 0 else ProductBuildLifecycle.queued
    elif blocked_units > 0:
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
                for step in blocking_steps
                if step.status in {"error", "requires_attention", "locked"}
            ],
            "blocking_deliverables": sorted(completion_scope.blocking_keys),
            "nonblocking_deliverables": list(completion_scope.nonblocking_payload),
            "completion_policy": {
                "policy_key": ACP_NONBLOCKING_POLICY_KEY if _is_acp_run(run) else "all_expected_deliverables",
                "blocking_count": len(completion_scope.blocking_keys),
                "nonblocking_count": len(completion_scope.nonblocking_payload),
                "summary": (
                    "El ACP queda listo cuando el nucleo de construccion esta disponible; "
                    "diagramas grandes o aislados se difieren como enriquecimiento."
                    if _is_acp_run(run)
                    else "Todos los entregables esperados bloquean la finalizacion del producto."
                ),
            },
        },
    )

    # Seal blueprint_pro runs once fully completed (no blocked/failed steps).
    # A sealed run blocks further user-triggered regenerations of LEAN work stages.
    if (
        lifecycle == ProductBuildLifecycle.completed
        and blocked_units == 0
        and run.product_key == ProductBuildProductKey.blueprint_pro.value
        and not getattr(run, "is_sealed", False)
    ):
        seal_product_build_run(db, run=run)


def _refresh_status_for_product(
    db: Session,
    *,
    record: SessionRecord,
    product_key: ProductBuildProductKey,
    current_user: UserRecord | None,
    catalog_stage_override: str | None,
) -> ProductBuildStatus:
    if product_key == ProductBuildProductKey.acp:
        from app.services.product_processing.acp_product_orchestration_service import ensure_acp_product_orchestration

        return ensure_acp_product_orchestration(
            db,
            record=record,
            current_user=current_user,
            execute_jobs=False,
            allow_llm=False,
            activation_payload={"source": "product_build_queue_refresh"},
            catalog_stage_override=catalog_stage_override or "package",
        )
    return reconcile_product_build_run(
        db,
        record=record,
        product_key=product_key,
        current_user=current_user,
        catalog_stage_override=catalog_stage_override,
    )


def _execute_processing_queue_inline(
    db: Session,
    *,
    record: SessionRecord,
    product_key: ProductBuildProductKey,
    current_user: UserRecord | None,
    allow_llm: bool,
    catalog_stage_override: str,
) -> None:
    queued_run, _, queued_now = enqueue_product_build_processing(
        db,
        record=record,
        product_key=product_key,
        current_user=current_user,
        mode=ProductBuildProcessingQueueMode.process_pending,
        allow_llm=allow_llm,
        catalog_stage_override=catalog_stage_override,
    )
    db.commit()
    if queued_now and queued_run is not None:
        run_product_build_processing(queued_run.id, db.get_bind())
        db.expire_all()


def _processing_queue_checkpoint(run: ProductBuildRunRecord) -> dict[str, Any]:
    value = (run.checkpoint_payload or {}).get("processing_queue")
    return dict(value) if isinstance(value, dict) else {}


def _update_processing_queue_checkpoint(
    db: Session,
    *,
    run: ProductBuildRunRecord,
    **updates: Any,
) -> dict[str, Any]:
    now = utc_now().isoformat()
    queue_checkpoint = {**_processing_queue_checkpoint(run), **updates}
    if "status" in updates and str(updates["status"] or "") in {"queued", "running"} and not str(queue_checkpoint.get("started_at") or ""):
        queue_checkpoint["started_at"] = now
    if "status" in updates and str(updates["status"] or "").startswith("completed") and not str(queue_checkpoint.get("completed_at") or ""):
        queue_checkpoint["completed_at"] = now
    queue_checkpoint["updated_at"] = now
    run.checkpoint_payload = {
        **(run.checkpoint_payload or {}),
        "processing_queue": queue_checkpoint,
    }
    db.add(run)
    db.flush()
    return queue_checkpoint


def _queue_processing_allowed(run: ProductBuildRunRecord) -> bool:
    resolution = (run.checkpoint_payload or {}).get("acp_direct_resolution")
    if not isinstance(resolution, dict):
        return True
    return bool(resolution.get("can_start_package")) and bool(resolution.get("can_export_package"))


def _select_processing_items(
    *,
    run: ProductBuildRunRecord,
    expected_items: list[DeliverableCatalogItem],
    jobs_by_key: dict[str, DeliverableGenerationJobRecord],
    diagram_jobs_by_key: dict[str, DiagramGenerationJobRecord],
    steps_by_key: dict[str, ProductBuildStepRecord],
    mode: ProductBuildProcessingQueueMode,
    nonblocking_keys: frozenset[str] = frozenset(),
) -> list[DeliverableCatalogItem]:
    if not _queue_processing_allowed(run):
        return []
    eligible_states = QUEUE_RETRY_ONLY_STATES if mode == ProductBuildProcessingQueueMode.retry_failed else QUEUE_ELIGIBLE_STATES
    selected: list[DeliverableCatalogItem] = []
    for item in expected_items:
        if item.key in nonblocking_keys:
            continue
        existing_step = steps_by_key.get(f"deliverable:{item.key}")
        job = jobs_by_key.get(item.key)
        diagram_job = _diagram_job_for_item(item, diagram_jobs_by_key) if job is None else None
        state = _step_state_for_item(item, job, diagram_job=diagram_job, existing_step=existing_step)
        if state not in eligible_states:
            continue
        if not (item.access.can_generate or item.access.can_regenerate):
            continue
        selected.append(item)
    return _topologically_sort_items(selected)


def _retry_budget_exhausted(step: ProductBuildStepRecord | None) -> bool:
    if step is None:
        return False
    checkpoint = step.checkpoint_payload or {}
    try:
        attempt_count = int(checkpoint.get("attempt_count") or 0)
    except (TypeError, ValueError):
        attempt_count = 0
    return attempt_count >= MAX_PROCESSING_ATTEMPTS


def _topologically_sort_items(items: list[DeliverableCatalogItem]) -> list[DeliverableCatalogItem]:
    if len(items) < 2:
        return list(items)
    items_by_key = {item.key: item for item in items}
    order_index = {item.key: index for index, item in enumerate(items)}
    edges: dict[str, set[str]] = defaultdict(set)
    indegree: dict[str, int] = {item.key: 0 for item in items}
    for item in items:
        entry = get_deliverable_registry_entry(item.key)
        depends_on = entry.dependency_policy.depends_on if entry is not None else []
        for dependency_key in depends_on:
            dependency_key = str(dependency_key or "").strip()
            if dependency_key not in items_by_key:
                continue
            if item.key not in edges[dependency_key]:
                edges[dependency_key].add(item.key)
                indegree[item.key] += 1
    queue = deque(sorted((item for item in items if indegree[item.key] == 0), key=lambda entry: (order_index[entry.key], entry.key)))
    ordered: list[DeliverableCatalogItem] = []
    while queue:
        item = queue.popleft()
        ordered.append(item)
        for dependent_key in sorted(edges.get(item.key, set()), key=lambda key: (order_index[key], key)):
            indegree[dependent_key] -= 1
            if indegree[dependent_key] == 0:
                queue.append(items_by_key[dependent_key])
    if len(ordered) != len(items):
        return list(items)
    return ordered


def _artifact_history(
    db: Session,
    *,
    run: ProductBuildRunRecord,
    item: DeliverableCatalogItem,
    generation_mode: str = "",
) -> tuple[int, bool]:
    if item.deliverable_type.value == "diagram":
        rows = list(
            db.exec(
                select(DiagramGenerationJobRecord)
                .where(
                    DiagramGenerationJobRecord.workspace_id == run.workspace_id,
                    DiagramGenerationJobRecord.diagram_key == item.key.removeprefix("diagram."),
                )
                .order_by(DiagramGenerationJobRecord.updated_at.desc())
                .limit(5)
            ).all()
        )
    else:
        rows = list(
            db.exec(
                select(DeliverableGenerationJobRecord)
                .where(
                    DeliverableGenerationJobRecord.workspace_id == run.workspace_id,
                    DeliverableGenerationJobRecord.deliverable_key == item.key,
                )
                .order_by(DeliverableGenerationJobRecord.updated_at.desc())
                .limit(5)
            ).all()
        )
    if generation_mode == "deterministic_python":
        rows = [job for job in rows if str(getattr(job, "provider_key", "") or "") == generation_mode]
    durations = [
        _seconds_between(getattr(job, "started_at", None), getattr(job, "completed_at", None) or getattr(job, "updated_at", None))
        for job in rows
        if getattr(job, "started_at", None) is not None
    ]
    average_duration = round(sum(durations) / len(durations)) if durations else 0
    recent_failure = any(str(getattr(job, "status", "") or "") in ERROR_JOB_STATES or bool(getattr(job, "error_code", "")) for job in rows[:3])
    return average_duration, recent_failure


def _effective_generation_mode(item: DeliverableCatalogItem, entry: Any | None = None) -> str:
    if supports_deterministic_deliverable(item.key):
        return "deterministic_python"
    if item.deliverable_type.value == "diagram" and supports_deterministic_diagram(item.key.removeprefix("diagram.")):
        return "deterministic_python"
    if entry is None:
        entry = get_deliverable_registry_entry(item.key)
    return str(
        getattr(getattr(entry, "generation_mode", ""), "value", getattr(entry, "generation_mode", ""))
        or "deterministic"
    )


def _seconds_between(started_at, finished_at) -> int:
    if started_at is None or finished_at is None:
        return 0
    return max(0, round((finished_at - started_at).total_seconds()))


def _estimate_artifact_processing(
    db: Session,
    *,
    run: ProductBuildRunRecord,
    item: DeliverableCatalogItem,
    items_by_key: dict[str, DeliverableCatalogItem],
) -> ProductBuildArtifactEstimate:
    entry = get_deliverable_registry_entry(item.key)
    type_value = str(getattr(item.deliverable_type, "value", item.deliverable_type) or "artifact")
    generation_mode = _effective_generation_mode(item, entry)
    context_policy = getattr(entry, "context_policy", None)
    try:
        context_tokens = max(0, int(getattr(context_policy, "max_context_tokens", 0) or 0))
    except (TypeError, ValueError):
        context_tokens = 0
    deterministic_python = generation_mode == "deterministic_python"
    if deterministic_python:
        context_tokens = min(context_tokens, 1_500)
    dependency_policy = getattr(entry, "dependency_policy", None)
    dependency_keys = [str(value or "").strip() for value in getattr(dependency_policy, "depends_on", []) or []]
    dependency_count = sum(1 for key in dependency_keys if key in items_by_key)
    historical_duration, recent_failure = _artifact_history(
        db,
        run=run,
        item=item,
        generation_mode=generation_mode,
    )

    type_score = {
        "artifact": 3,
        "contract": 4,
        "diagram": 4,
        "document": 5,
        "lineage": 3,
        "package": 8,
        "prompt": 4,
        "test": 4,
    }.get(type_value, 3)
    if deterministic_python:
        type_score = min(type_score, 2)
    mode_score = {
        "deterministic": 0,
        "deterministic_python": 0,
        "llm_supported": 2,
        "llm_required": 3,
        "llm_with_deterministic_fallback": 2,
        "manual_review_required": 6,
    }.get(generation_mode, 1)
    context_score = min(4, (context_tokens + 5_999) // 6_000)
    dependency_score = min(4, dependency_count)
    history_score = 4 if historical_duration >= HEAVY_ARTIFACT_DURATION_SECONDS else 2 if historical_duration >= 90 else 1 if historical_duration >= 45 else 0
    failure_score = 2 if recent_failure else 0
    complexity_score = type_score + mode_score + context_score + dependency_score + history_score + failure_score

    output_tokens = {
        "artifact": 3_000,
        "contract": 4_000,
        "diagram": 3_500,
        "document": 6_000,
        "lineage": 2_500,
        "package": 1_500,
        "prompt": 3_500,
        "test": 4_500,
    }.get(type_value, 3_000)
    default_seconds = {
        "artifact": 35,
        "contract": 55,
        "diagram": 75,
        "document": 90,
        "lineage": 40,
        "package": 120,
        "prompt": 50,
        "test": 70,
    }.get(type_value, 45)
    if deterministic_python:
        output_tokens = min(output_tokens, 900)
        estimated_tokens = min(6_000, context_tokens + output_tokens + 300)
        default_seconds = 8 if type_value == "diagram" else 10
        token_seconds = max(2, estimated_tokens // 1_500)
    else:
        estimated_tokens = min(120_000, context_tokens + output_tokens + 1_200)
        token_seconds = max(15, estimated_tokens // 350)
    estimated_seconds = max(historical_duration, default_seconds, token_seconds)

    isolation_reason = ""
    if type_value == "package":
        isolation_reason = "package_final_critical_path"
    elif generation_mode == "manual_review_required":
        isolation_reason = "manual_review_required"
    elif dependency_count >= 3:
        isolation_reason = "many_dependencies"
    elif historical_duration >= HEAVY_ARTIFACT_DURATION_SECONDS:
        isolation_reason = "historically_slow"
    elif estimated_tokens >= 30_000:
        isolation_reason = "large_context_budget"
    elif complexity_score >= 11:
        isolation_reason = "high_complexity_score"

    if isolation_reason:
        return ProductBuildArtifactEstimate(
            deliverable_key=item.key,
            group="isolated",
            estimated_tokens=estimated_tokens,
            estimated_seconds=estimated_seconds,
            complexity_score=complexity_score,
            dependency_count=dependency_count,
            can_parallelize=False,
            resource_units=999,
            isolation_reason=isolation_reason,
            historical_duration_seconds=historical_duration,
            generation_mode=generation_mode,
        )
    if complexity_score >= 7 or estimated_tokens >= 16_000 or estimated_seconds >= 120:
        group = "large"
        can_parallelize = False
        resource_units = 4
    elif complexity_score >= 4 or estimated_tokens >= 8_000 or estimated_seconds >= 60:
        group = "medium"
        can_parallelize = True
        resource_units = 2
    else:
        group = "small"
        can_parallelize = True
        resource_units = 1
    return ProductBuildArtifactEstimate(
        deliverable_key=item.key,
        group=group,
        estimated_tokens=estimated_tokens,
        estimated_seconds=estimated_seconds,
        complexity_score=complexity_score,
        dependency_count=dependency_count,
        can_parallelize=can_parallelize,
        resource_units=resource_units,
        historical_duration_seconds=historical_duration,
        generation_mode=generation_mode,
    )


def _artifact_estimate_payload(estimate: ProductBuildArtifactEstimate) -> dict[str, Any]:
    return {
        "group": estimate.group,
        "estimated_tokens": estimate.estimated_tokens,
        "estimated_seconds": estimate.estimated_seconds,
        "complexity_score": estimate.complexity_score,
        "dependency_count": estimate.dependency_count,
        "can_parallelize": estimate.can_parallelize,
        "resource_units": estimate.resource_units,
        "isolation_reason": estimate.isolation_reason,
        "historical_duration_seconds": estimate.historical_duration_seconds,
        "generation_mode": estimate.generation_mode,
    }


def _is_acp_run(run: ProductBuildRunRecord) -> bool:
    return str(run.product_key or "") == ProductBuildProductKey.acp.value


def _nonblocking_acp_payload_for_item(
    db: Session,
    *,
    run: ProductBuildRunRecord,
    item: DeliverableCatalogItem,
    items_by_key: dict[str, DeliverableCatalogItem],
) -> dict[str, Any] | None:
    if not _is_acp_run(run):
        return None
    if item.deliverable_type.value != "diagram":
        return None
    estimate = _estimate_artifact_processing(db, run=run, item=item, items_by_key=items_by_key)
    if estimate.group not in ACP_NONBLOCKING_DIAGRAM_GROUPS:
        return None
    return {
        "deliverable_key": item.key,
        "title": item.title,
        "type": item.deliverable_type.value,
        "stage": item.stage,
        "policy_key": ACP_NONBLOCKING_POLICY_KEY,
        "reason": (
            "Diagrama ACP de alto costo o alta dependencia. No bloquea el paquete inicial; "
            "puede generarse como enriquecimiento extendido."
        ),
        "processing_estimate": _artifact_estimate_payload(estimate),
    }


def _build_completion_scope(
    db: Session,
    *,
    run: ProductBuildRunRecord,
    expected_items: list[DeliverableCatalogItem],
) -> ProductBuildCompletionScope:
    # The delivery plan is already resolved before processing. Once a deliverable
    # is included in that frozen scope, ACP must either complete it or surface an
    # explicit blocker instead of silently classifying it as optional.
    _ = db, run
    nonblocking_payload: tuple[dict[str, Any], ...] = ()
    nonblocking_keys: frozenset[str] = frozenset()
    return ProductBuildCompletionScope(
        blocking_keys=frozenset(item.key for item in expected_items if item.key not in nonblocking_keys),
        nonblocking_payload=nonblocking_payload,
    )


def _estimate_profile_counts(estimates: list[ProductBuildArtifactEstimate]) -> dict[str, int]:
    counts = {"small": 0, "medium": 0, "large": 0, "isolated": 0}
    for estimate in estimates:
        counts[estimate.group] = counts.get(estimate.group, 0) + 1
    return counts


def _next_ready_batch(
    db: Session,
    *,
    run: ProductBuildRunRecord,
    remaining_items: list[DeliverableCatalogItem],
    items_by_key: dict[str, DeliverableCatalogItem],
    batch_size: int,
) -> list[DeliverableCatalogItem]:
    ready: list[DeliverableCatalogItem] = []
    for item in remaining_items:
        if _dependency_error_for_item(db, run=run, item=item, items_by_key=items_by_key) is not None:
            continue
        ready.append(item)
        if not _dynamic_parallelism_enabled() and len(ready) >= batch_size:
            break
    if ready:
        if batch_size <= 1 or not _dynamic_parallelism_enabled():
            return ready[:batch_size]
        estimates = {
            item.key: _estimate_artifact_processing(db, run=run, item=item, items_by_key=items_by_key)
            for item in ready
        }
        first = ready[0]
        first_estimate = estimates[first.key]
        if not first_estimate.can_parallelize:
            return [first]

        selected: list[DeliverableCatalogItem] = []
        total_tokens = 0
        total_units = 0
        token_budget = _parallel_token_budget()
        complexity_budget = _parallel_complexity_budget()
        for item in ready:
            estimate = estimates[item.key]
            if not estimate.can_parallelize:
                if not selected:
                    return [item]
                continue
            next_tokens = total_tokens + estimate.estimated_tokens
            next_units = total_units + estimate.resource_units
            if selected and (next_tokens > token_budget or next_units > complexity_budget):
                continue
            selected.append(item)
            total_tokens = next_tokens
            total_units = next_units
            if len(selected) >= batch_size:
                break
        return selected or [first]
    return remaining_items[:1]


def _batch_execution_summary(
    *,
    batch: list[DeliverableCatalogItem],
    estimates: list[ProductBuildArtifactEstimate],
    positions: dict[str, int],
    total_count: int,
    batch_size: int,
) -> str:
    if len(batch) == 1:
        item = batch[0]
        estimate = estimates[0]
        if batch_size > 1 and not estimate.can_parallelize:
            reason = estimate.isolation_reason or estimate.group
            return f"Procesando {positions[item.key]} de {total_count}: {item.title} de forma individual ({reason})."
        return f"Procesando {positions[item.key]} de {total_count}: {item.title}."
    counts = _estimate_profile_counts(estimates)
    profile = ", ".join(f"{key}:{value}" for key, value in counts.items() if value)
    total_tokens = sum(estimate.estimated_tokens for estimate in estimates)
    return f"Procesando lote paralelo de {len(batch)} de {total_count} entregables ({profile}; {total_tokens} tokens estimados)."


def _process_queue_item_in_new_session(
    *,
    database_engine: Engine,
    run_id: UUID,
    item: DeliverableCatalogItem,
    items_by_key: dict[str, DeliverableCatalogItem],
    allow_llm: bool,
    phase: str,
    position: int,
    total_count: int,
) -> tuple[str, bool]:
    with Session(database_engine) as worker_db:
        run = worker_db.get(ProductBuildRunRecord, run_id)
        if run is None:
            return item.key, False
        record = worker_db.get(SessionRecord, run.session_id)
        if record is None or record.workspace_id != run.workspace_id:
            return item.key, False
        ok = _process_single_queue_item(
            worker_db,
            run=run,
            record=record,
            item=item,
            items_by_key=items_by_key,
            allow_llm=allow_llm,
            phase=phase,
            position=position,
            total_count=total_count,
            update_queue_checkpoint=False,
        )
        return item.key, ok


def _process_queue_items_in_batches(
    db: Session,
    *,
    database_engine: Engine,
    run: ProductBuildRunRecord,
    record: SessionRecord,
    ordered_items: list[DeliverableCatalogItem],
    items_by_key: dict[str, DeliverableCatalogItem],
    allow_llm: bool,
    phase: str,
    total_count: int,
) -> list[str]:
    if not ordered_items:
        return []
    batch_size = _product_build_batch_size(db.get_bind())
    positions = {item.key: index for index, item in enumerate(ordered_items, start=1)}
    failed_keys: list[str] = []
    remaining_items = list(ordered_items)

    while remaining_items:
        run = db.get(ProductBuildRunRecord, run.id)
        if run is None:
            failed_keys.extend(item.key for item in remaining_items)
            break
        batch = _next_ready_batch(
            db,
            run=run,
            remaining_items=remaining_items,
            items_by_key=items_by_key,
            batch_size=batch_size,
        )
        batch_estimates = [
            _estimate_artifact_processing(db, run=run, item=item, items_by_key=items_by_key)
            for item in batch
        ]
        batch_keys = {item.key for item in batch}
        _update_processing_queue_checkpoint(
            db,
            run=run,
            status="running",
            current_deliverable_key=", ".join(item.key for item in batch),
            summary=_batch_execution_summary(
                batch=batch,
                estimates=batch_estimates,
                positions=positions,
                total_count=total_count,
                batch_size=batch_size,
            ),
            current_batch_profile={
                "strategy": "dynamic_parallelism" if _dynamic_parallelism_enabled() and batch_size > 1 else "sequential",
                "items": [
                    {"deliverable_key": item.key, **_artifact_estimate_payload(estimate)}
                    for item, estimate in zip(batch, batch_estimates)
                ],
            },
        )
        db.commit()

        if batch_size <= 1:
            item = batch[0]
            ok = _process_single_queue_item(
                db,
                run=run,
                record=record,
                item=item,
                items_by_key=items_by_key,
                allow_llm=allow_llm,
                phase=phase,
                position=positions[item.key],
                total_count=total_count,
            )
            if not ok:
                failed_keys.append(item.key)
        else:
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = [
                    executor.submit(
                        _process_queue_item_in_new_session,
                        database_engine=database_engine,
                        run_id=run.id,
                        item=item,
                        items_by_key=items_by_key,
                        allow_llm=allow_llm,
                        phase=phase,
                        position=positions[item.key],
                        total_count=total_count,
                    )
                    for item in batch
                ]
                for future in as_completed(futures):
                    item_key, ok = future.result()
                    if not ok:
                        failed_keys.append(item_key)
            db.expire_all()

        remaining_items = [item for item in remaining_items if item.key not in batch_keys]
    return failed_keys


def _process_single_queue_item(
    db: Session,
    *,
    run: ProductBuildRunRecord,
    record: SessionRecord,
    item: DeliverableCatalogItem,
    items_by_key: dict[str, DeliverableCatalogItem],
    allow_llm: bool,
    phase: str,
    position: int,
    total_count: int,
    update_queue_checkpoint: bool = True,
) -> bool:
    if update_queue_checkpoint:
        _update_processing_queue_checkpoint(
            db,
            run=run,
            status="running",
            current_deliverable_key=item.key,
            summary=f"Procesando {position} de {total_count}: {item.title}.",
        )
    step = _step_record(db, run=run, deliverable_key=item.key)
    existing_attempt_count = int((step.checkpoint_payload or {}).get("attempt_count") or 0) if step is not None else 0
    if (
        item.deliverable_type.value == "diagram"
        and not allow_llm
        and not supports_deterministic_diagram(item.key.removeprefix("diagram."))
    ):
        _record_queue_item_failure(
            db,
            run=run,
            item=item,
            position=position,
            attempt_count=existing_attempt_count,
            error_payload={
                "code": "llm_required_for_diagram_generation",
                "message": (
                    "La generacion de diagramas requiere una ejecucion LLM autorizada; "
                    "no se inicio trabajo automatico para preservar costos e idempotencia."
                ),
            },
        )
        return False
    attempt_count = int((step.checkpoint_payload or {}).get("attempt_count") or 0) + 1 if step is not None else 1
    processing_estimate = _estimate_artifact_processing(db, run=run, item=item, items_by_key=items_by_key)
    upsert_product_build_step(
        db,
        run=run,
        step_key=f"deliverable:{item.key}",
        status="running",
        stage_key=item.stage,
        deliverable_key=item.key,
        sequence=position,
        progress_percent=35,
        checkpoint_payload={
            **(step.checkpoint_payload or {} if step is not None else {}),
            "attempt_count": attempt_count,
            "retried": phase == "retry" or attempt_count > 1,
            "last_phase": phase,
            "last_started_at": utc_now().isoformat(),
            "processing_estimate": _artifact_estimate_payload(processing_estimate),
        },
        error_payload={},
    )
    if update_queue_checkpoint:
        update_product_build_run_state(
            db,
            run=run,
            lifecycle=ProductBuildLifecycle.running,
            checkpoint_payload=run.checkpoint_payload,
        )
    db.commit()

    dependency_error = _dependency_error_for_item(db, run=run, item=item, items_by_key=items_by_key)
    if dependency_error is not None:
        _record_queue_item_failure(
            db,
            run=run,
            item=item,
            position=position,
            attempt_count=attempt_count,
            error_payload=dependency_error,
        )
        return False

    try:
        if item.deliverable_type.value == "diagram":
            job = create_generation_job(
                db,
                record=record,
                diagram_key=item.key.removeprefix("diagram."),
                user_id=run.created_by_user_id or record.user_id,
                detail_level="standard",
                reason="regenerate" if phase == "retry" or item.access.can_regenerate else "generate",
                idempotency_key=_queue_job_idempotency_key(run=run, item=item, phase=phase),
            )
            refreshed_step = _step_record(db, run=run, deliverable_key=item.key)
            upsert_product_build_step(
                db,
                run=run,
                step_key=f"deliverable:{item.key}",
                status="generating" if str(job.status or "") in {"queued", "updating", "generating"} else "queued",
                stage_key=item.stage,
                deliverable_key=item.key,
                job_id=job.id,
                sequence=position,
                progress_percent=55,
                checkpoint_payload={
                    **(refreshed_step.checkpoint_payload or {} if refreshed_step is not None else {}),
                    "job_source": "diagram_center",
                },
                error_payload={},
            )
            db.commit()
            run_generation_job(job.id, db_session=db)
            db.expire_all()
            refreshed_job = db.get(DiagramGenerationJobRecord, job.id)
            if refreshed_job is None or str(refreshed_job.status or "") != "available":
                _record_queue_item_failure(
                    db,
                    run=run,
                    item=item,
                    position=position,
                    attempt_count=attempt_count,
                    error_payload=_error_payload_for_job(refreshed_job) if refreshed_job is not None else {
                        "code": "diagram_job_missing",
                        "message": "No se pudo recuperar el job de diagrama despues de ejecutarlo.",
                    },
                    job_id=refreshed_job.id if refreshed_job is not None else None,
                )
                return False
            _record_queue_item_success(
                db,
                run=run,
                item=item,
                position=position,
                attempt_count=attempt_count,
                job_id=refreshed_job.id,
                job_source="diagram_center",
            )
            return True

        context_payload, approved_context_refs = build_approved_deliverable_context(
            db,
            record=record,
            deliverable_key=item.key,
        )
        task = DeliverableGenerationTask(
            workspace_id=run.workspace_id,
            session_id=record.id,
            deliverable_key=item.key,
            product_mode=run.product_mode,
            current_stage=str((run.checkpoint_payload or {}).get("catalog_stage") or item.stage),
            tier=_safe_tier(run.entitlement_tier),
            idempotency_key=_queue_job_idempotency_key(run=run, item=item, phase=phase),
            requested_by_user_id=run.created_by_user_id,
            context_payload=context_payload,
            approved_context_refs=approved_context_refs,
            allow_llm=allow_llm,
        )
        job, result = run_deliverable_generation_task(db, task)
        if result is None and str(job.status or "") == "available":
            _record_queue_item_success(
                db,
                run=run,
                item=item,
                position=position,
                attempt_count=attempt_count,
                job_id=job.id,
                job_source="deliverable_catalog",
            )
            return True
        if result is None or str(job.status or "") != "available":
            _record_queue_item_failure(
                db,
                run=run,
                item=item,
                position=position,
                attempt_count=attempt_count,
                error_payload=_error_payload_for_job(job) or {
                    "code": str(job.status or "deliverable_generation_failed"),
                    "message": job.error_message or "La generacion del entregable no finalizo correctamente.",
                    "job_id": str(job.id),
                },
                job_id=job.id,
            )
            return False
        _record_queue_item_success(
            db,
            run=run,
            item=item,
            position=position,
            attempt_count=attempt_count,
            job_id=job.id,
            job_source="deliverable_catalog",
        )
        return True
    except Exception as exc:
        _record_queue_item_failure(
            db,
            run=run,
            item=item,
            position=position,
            attempt_count=attempt_count,
            error_payload={
                "code": type(exc).__name__,
                "message": str(exc) or "La ejecucion del entregable fallo por una excepcion no controlada.",
            },
        )
        return False


def _dependency_error_for_item(
    db: Session,
    *,
    run: ProductBuildRunRecord,
    item: DeliverableCatalogItem,
    items_by_key: dict[str, DeliverableCatalogItem],
) -> dict[str, Any] | None:
    entry = get_deliverable_registry_entry(item.key)
    if entry is None or not entry.dependency_policy.depends_on:
        return None
    steps_by_key = {step.step_key: step for step in list_product_build_steps(db, run_id=run.id)}
    jobs_by_key = _latest_jobs_by_key(db, session_id=run.session_id)
    diagram_jobs_by_key = _latest_diagram_jobs_by_key(db, session_id=run.session_id)
    for dependency_key in entry.dependency_policy.depends_on:
        dependency_key = str(dependency_key or "").strip()
        dependency_item = items_by_key.get(dependency_key)
        if dependency_item is None:
            continue
        dependency_step = steps_by_key.get(f"deliverable:{dependency_key}")
        dependency_job = jobs_by_key.get(dependency_key)
        dependency_diagram_job = _diagram_job_for_item(dependency_item, diagram_jobs_by_key) if dependency_job is None else None
        dependency_state = _step_state_for_item(
            dependency_item,
            dependency_job,
            diagram_job=dependency_diagram_job,
            existing_step=dependency_step,
        )
        if dependency_state not in COMPLETED_STEP_STATES:
            return {
                "code": "dependency_not_ready",
                "message": f"Depende de {dependency_key}, que aun no esta disponible para continuar.",
                "dependency_key": dependency_key,
            }
    return None


def _queue_job_idempotency_key(
    *,
    run: ProductBuildRunRecord,
    item: DeliverableCatalogItem,
    phase: str,
) -> str:
    queue_id = str(_processing_queue_checkpoint(run).get("queue_id") or "manual")
    return f"{run.idempotency_key}:queue:{queue_id}:{phase}:{item.key}"


def _safe_tier(value: str | CommercialTier) -> CommercialTier:
    if isinstance(value, CommercialTier):
        return value
    try:
        return CommercialTier(str(value))
    except ValueError:
        return CommercialTier.blueprint


def _step_record(
    db: Session,
    *,
    run: ProductBuildRunRecord,
    deliverable_key: str,
) -> ProductBuildStepRecord | None:
    return next((step for step in list_product_build_steps(db, run_id=run.id) if step.deliverable_key == deliverable_key), None)


def _record_queue_item_success(
    db: Session,
    *,
    run: ProductBuildRunRecord,
    item: DeliverableCatalogItem,
    position: int,
    attempt_count: int,
    job_id: UUID | None,
    job_source: str,
) -> None:
    step = _step_record(db, run=run, deliverable_key=item.key)
    upsert_product_build_step(
        db,
        run=run,
        step_key=f"deliverable:{item.key}",
        status="available",
        stage_key=item.stage,
        deliverable_key=item.key,
        job_id=job_id,
        sequence=position,
        progress_percent=100,
        checkpoint_payload={
            **(step.checkpoint_payload or {} if step is not None else {}),
            "attempt_count": attempt_count,
            "retried": attempt_count > 1,
            "job_source": job_source,
            "last_succeeded_at": utc_now().isoformat(),
        },
        error_payload={},
    )
    db.commit()


def _record_queue_item_failure(
    db: Session,
    *,
    run: ProductBuildRunRecord,
    item: DeliverableCatalogItem,
    position: int,
    attempt_count: int,
    error_payload: dict[str, Any],
    job_id: UUID | None = None,
) -> None:
    step = _step_record(db, run=run, deliverable_key=item.key)
    upsert_product_build_step(
        db,
        run=run,
        step_key=f"deliverable:{item.key}",
        status="error",
        stage_key=item.stage,
        deliverable_key=item.key,
        job_id=job_id,
        sequence=position,
        progress_percent=100,
        checkpoint_payload={
            **(step.checkpoint_payload or {} if step is not None else {}),
            "attempt_count": attempt_count,
            "retried": attempt_count > 1,
            "last_failed_at": utc_now().isoformat(),
        },
        error_payload=error_payload,
    )
    db.commit()


def _finalize_processing_queue(
    db: Session,
    *,
    run: ProductBuildRunRecord,
    status: str,
    summary: str,
    failed_keys: list[str],
) -> None:
    if failed_keys:
        _mark_failed_queue_jobs_as_error(db, session_id=run.session_id, failed_keys=failed_keys)
    queue_checkpoint = _update_processing_queue_checkpoint(
        db,
        run=run,
        status=status,
        current_deliverable_key="",
        summary=summary,
    )
    error_payload = {}
    if failed_keys:
        failed_step = next(
            (step for step in list_product_build_steps(db, run_id=run.id) if step.deliverable_key in failed_keys),
            None,
        )
        error_payload = {
            "code": str(failed_step.error_payload.get("code") if failed_step is not None else "product_build_processing_failed"),
            "title": "Persisten entregables fallidos despues del reintento automatico",
            "message": summary,
            "technical_message": str(failed_step.error_payload.get("message") if failed_step is not None else ""),
            "retry_action_key": ProductBuildProcessingQueueMode.retry_failed.value,
            "trace_refs": failed_keys,
        }
    update_product_build_run_state(
        db,
        run=run,
        lifecycle=run.lifecycle,
        checkpoint_payload={
            **(run.checkpoint_payload or {}),
            "processing_queue": queue_checkpoint,
        },
        error_payload=error_payload,
    )
    db.commit()


def _recover_orphaned_processing_queue(db: Session, *, run: ProductBuildRunRecord) -> bool:
    """Close jobs abandoned by a process restart without starting new LLM work."""
    queue_checkpoint = _processing_queue_checkpoint(run)
    queue_status = str(queue_checkpoint.get("status") or "")
    selected_keys = [str(value) for value in queue_checkpoint.get("selected_deliverable_keys", [])]
    active_steps = [
        step
        for step in list_product_build_steps(db, run_id=run.id)
        if step.deliverable_key in selected_keys and str(step.status or "") in QUEUE_ACTIVE_STEP_STATES
    ]
    if queue_status not in QUEUE_ACTIVE_STATUSES:
        all_active_steps = [
            step
            for step in list_product_build_steps(db, run_id=run.id)
            if str(step.status or "") in QUEUE_ACTIVE_STEP_STATES
        ]
        if not all_active_steps:
            return False
        return _mark_queue_steps_as_orphaned(
            db,
            run=run,
            active_steps=all_active_steps,
            summary=(
                f"Se cerraron {len(all_active_steps)} entregables que seguian activos aunque la cola ya no estaba corriendo. "
                "No se reintentaron automaticamente para preservar idempotencia y control de costos."
            ),
        )

    cutoff = utc_now() - ORPHANED_JOB_TIMEOUT
    in_flight_steps = [
        step
        for step in active_steps
        if str(step.status or "").strip().lower() in {"running", "generating"}
    ]
    if in_flight_steps:
        if any(_active_queue_step_updated_at(db, step=step) > cutoff for step in in_flight_steps):
            return False
        stale_steps = [step for step in in_flight_steps if _active_queue_step_updated_at(db, step=step) <= cutoff]
    else:
        all_steps = list_product_build_steps(db, run_id=run.id)
        latest_activity = max(
            [step.updated_at for step in all_steps if step is not None and step.updated_at is not None],
            default=run.updated_at,
        )
        if latest_activity and latest_activity > cutoff:
            return False
        stale_steps = active_steps

    if not stale_steps:
        return False

    return _mark_queue_steps_as_orphaned(
        db,
        run=run,
        active_steps=stale_steps,
        summary=(
            f"Se detectaron {len(stale_steps)} jobs interrumpidos. "
            "No se reintentaron automáticamente para preservar idempotencia y control de costos."
        ),
    )


def _mark_queue_steps_as_orphaned(
    db: Session,
    *,
    run: ProductBuildRunRecord,
    active_steps: list[ProductBuildStepRecord],
    summary: str,
) -> bool:
    orphaned_keys: list[str] = []
    for step in active_steps:
        deliverable_key = str(step.deliverable_key or "")
        if not deliverable_key:
            continue
        orphaned_keys.append(deliverable_key)
        step.status = "error"
        step.progress_percent = 0
        step.error_payload = {
            "code": "processing_queue_orphaned",
            "message": "El procesamiento fue interrumpido antes de persistir un resultado final. Se requiere un reintento controlado.",
        }
        step.checkpoint_payload = {
            **(step.checkpoint_payload or {}),
            "orphaned_at": utc_now().isoformat(),
            "queue_selected": True,
        }
        db.add(step)

    if not orphaned_keys:
        return False
    _mark_failed_queue_jobs_as_error(db, session_id=run.session_id, failed_keys=orphaned_keys, include_started_jobs=True)
    _update_processing_queue_checkpoint(
        db,
        run=run,
        status="completed_with_errors",
        current_deliverable_key="",
        summary=summary,
    )
    update_product_build_run_state(
        db,
        run=run,
        lifecycle=ProductBuildLifecycle.requires_attention,
        checkpoint_payload=run.checkpoint_payload,
        error_payload={
            "code": "processing_queue_orphaned",
            "title": "El procesamiento fue interrumpido",
            "message": "Algunos entregables requieren un reintento controlado.",
            "retry_action_key": ProductBuildProcessingQueueMode.retry_failed.value,
            "trace_refs": orphaned_keys,
        },
    )
    db.commit()
    return True


def _active_queue_step_updated_at(db: Session, *, step: ProductBuildStepRecord):
    deliverable_key = str(step.deliverable_key or "")
    if deliverable_key.startswith("diagram."):
        diagram_key = deliverable_key.removeprefix("diagram.")
        diagram_job = db.exec(
            select(DiagramGenerationJobRecord)
            .where(
                DiagramGenerationJobRecord.session_id == step.session_id,
                DiagramGenerationJobRecord.diagram_key == diagram_key,
                DiagramGenerationJobRecord.status.in_(tuple(ACTIVE_JOB_STATES)),
            )
            .order_by(DiagramGenerationJobRecord.updated_at.desc())
        ).first()
        if diagram_job is not None:
            return diagram_job.updated_at
    deliverable_job = db.exec(
        select(DeliverableGenerationJobRecord)
        .where(
            DeliverableGenerationJobRecord.session_id == step.session_id,
            DeliverableGenerationJobRecord.deliverable_key == deliverable_key,
            DeliverableGenerationJobRecord.status.in_(tuple(ACTIVE_JOB_STATES)),
        )
        .order_by(DeliverableGenerationJobRecord.updated_at.desc())
    ).first()
    if deliverable_job is not None:
        return deliverable_job.updated_at
    return step.updated_at


def _mark_failed_queue_jobs_as_error(
    db: Session,
    *,
    session_id: UUID,
    failed_keys: list[str],
    include_started_jobs: bool = False,
) -> None:
    failure_code = "processing_queue_orphaned"
    for deliverable_key in failed_keys:
        latest_deliverable_job = db.exec(
            select(DeliverableGenerationJobRecord)
            .where(
                DeliverableGenerationJobRecord.session_id == session_id,
                DeliverableGenerationJobRecord.deliverable_key == deliverable_key,
            )
            .order_by(DeliverableGenerationJobRecord.updated_at.desc())
        ).first()
        if (
            latest_deliverable_job is not None
            and str(latest_deliverable_job.status or "") in ACTIVE_JOB_STATES
            and (include_started_jobs or latest_deliverable_job.started_at is None)
            and latest_deliverable_job.completed_at is None
        ):
            latest_deliverable_job.status = "error"
            latest_deliverable_job.error_code = failure_code
            latest_deliverable_job.error_message = (
                "La cola del product build finalizo con error antes de que este job arrancara."
            )
            latest_deliverable_job.completed_at = utc_now()
            latest_deliverable_job.updated_at = utc_now()
            db.add(latest_deliverable_job)

        if not deliverable_key.startswith("diagram."):
            continue
        diagram_key = deliverable_key.removeprefix("diagram.")
        active_diagram_jobs = db.exec(
            select(DiagramGenerationJobRecord)
            .where(
                DiagramGenerationJobRecord.session_id == session_id,
                DiagramGenerationJobRecord.diagram_key == diagram_key,
                DiagramGenerationJobRecord.status.in_(tuple(ACTIVE_JOB_STATES)),
            )
            .order_by(DiagramGenerationJobRecord.updated_at.desc())
        ).all()
        for diagram_job in active_diagram_jobs:
            if (not include_started_jobs and diagram_job.started_at is not None) or diagram_job.completed_at is not None:
                continue
            diagram_job.status = "error"
            diagram_job.error_code = failure_code
            diagram_job.error_message = (
                "La cola del product build finalizo con error antes de que este job arrancara."
            )
            diagram_job.completed_at = utc_now()
            diagram_job.updated_at = utc_now()
            db.add(diagram_job)
    db.flush()
