from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy.engine import Engine
from sqlmodel import Session, select

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
from app.services.deliverable_catalog.generation_service import run_deliverable_generation_task
from app.services.deliverable_catalog.persistence import DeliverableGenerationJobRecord
from app.services.deliverable_catalog.registry_service import get_registry_entry as get_deliverable_registry_entry
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
QUEUE_RETRY_ONLY_STATES = {"error"}
QUEUE_ACTIVE_STATUSES = {"queued", "running"}
MAX_PROCESSING_ATTEMPTS = 2
ORPHANED_JOB_TIMEOUT = timedelta(minutes=5)

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
        idempotency_key=_run_idempotency_key(
            record=record,
            product_key=meta.product_key,
            explicit_key=resolved_options.idempotency_key,
        ),
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
    catalog_stage_override: str | None = None,
) -> tuple[ProductBuildRunRecord | None, ProductBuildStatus, bool]:
    normalized_product_key = _normalize_product_key(product_key)
    resolved_mode = _normalize_queue_mode(mode)

    if normalized_product_key == ProductBuildProductKey.acp:
        from app.services.product_processing.acp_product_orchestration_service import ensure_acp_product_orchestration

        status = ensure_acp_product_orchestration(
            db,
            record=record,
            current_user=current_user,
            execute_jobs=False,
            allow_llm=allow_llm,
            activation_payload={"source": f"product_build_queue:{resolved_mode.value}"},
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
    expected_items = [item for item in catalog.entries if _is_expected_for_product(item, meta)]
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
    selected_items = _select_processing_items(
        run=run,
        expected_items=expected_items,
        jobs_by_key=jobs_by_key,
        diagram_jobs_by_key=diagram_jobs_by_key,
        steps_by_key=steps_by_key,
        mode=resolved_mode,
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
                    "summary": "No hay entregables pendientes, no generados o fallidos para procesar.",
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
            },
            error_payload={},
        )

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
                "summary": f"Se encolaron {len(selected_items)} entregables para procesamiento secuencial.",
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
    with Session(database_engine or engine) as db:
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
        expected_items = [item for item in catalog.entries if _is_expected_for_product(item, meta)]
        items_by_key = {item.key: item for item in expected_items}
        selected_keys = [str(key) for key in queue_checkpoint.get("selected_deliverable_keys", []) if str(key) in items_by_key]
        ordered_items = _topologically_sort_items([items_by_key[key] for key in selected_keys])

        _update_processing_queue_checkpoint(
            db,
            run=run,
            status="running",
            current_deliverable_key="",
            summary=f"Procesando {len(selected_keys)} entregables de forma secuencial.",
        )
        update_product_build_run_state(
            db,
            run=run,
            lifecycle=ProductBuildLifecycle.running,
            checkpoint_payload=run.checkpoint_payload,
        )
        db.commit()

        failed_keys: list[str] = []
        for position, item in enumerate(ordered_items, start=1):
            ok = _process_single_queue_item(
                db,
                run=run,
                record=record,
                item=item,
                items_by_key=items_by_key,
                allow_llm=bool(queue_checkpoint.get("allow_llm")),
                phase="initial",
                position=position,
                total_count=len(selected_keys),
            )
            if not ok:
                failed_keys.append(item.key)

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

        remaining_failures: list[str] = []
        for position, item in enumerate(retry_items, start=1):
            ok = _process_single_queue_item(
                db,
                run=run,
                record=record,
                item=item,
                items_by_key=items_by_key,
                allow_llm=bool(queue_checkpoint.get("allow_llm")),
                phase="retry",
                position=position,
                total_count=len(retry_items),
            )
            if not ok:
                remaining_failures.append(item.key)

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
    for index, item in enumerate(sorted(expected_items, key=lambda entry: entry.sort_order), start=1):
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
    for item in sorted(expected_items, key=lambda entry: entry.sort_order):
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
            context_payload=dict(options.context_payload or {}),
            approved_context_refs=list(options.approved_context_refs),
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
    total_units = float(len(expected_items))
    completed_units = float(sum(1 for step in relevant_steps if step.status in COMPLETED_STEP_STATES))
    blocked_units = float(sum(1 for step in relevant_steps if step.status in {"error", "requires_attention", "locked"}))
    active_units = sum(1 for step in relevant_steps if step.status in {"queued", "running", "generating"})
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
                for step in relevant_steps
                if step.status in {"error", "requires_attention", "locked"}
            ],
        },
    )


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
) -> list[DeliverableCatalogItem]:
    if not _queue_processing_allowed(run):
        return []
    eligible_states = QUEUE_RETRY_ONLY_STATES if mode == ProductBuildProcessingQueueMode.retry_failed else QUEUE_ELIGIBLE_STATES
    selected: list[DeliverableCatalogItem] = []
    for item in expected_items:
        existing_step = steps_by_key.get(f"deliverable:{item.key}")
        job = jobs_by_key.get(item.key)
        diagram_job = _diagram_job_for_item(item, diagram_jobs_by_key) if job is None else None
        state = _step_state_for_item(item, job, diagram_job=diagram_job, existing_step=existing_step)
        if state not in eligible_states:
            continue
        if mode == ProductBuildProcessingQueueMode.retry_failed and _retry_budget_exhausted(existing_step):
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
    queue = deque(sorted((item for item in items if indegree[item.key] == 0), key=lambda entry: (entry.sort_order, entry.key)))
    ordered: list[DeliverableCatalogItem] = []
    while queue:
        item = queue.popleft()
        ordered.append(item)
        for dependent_key in sorted(edges.get(item.key, set()), key=lambda key: (items_by_key[key].sort_order, key)):
            indegree[dependent_key] -= 1
            if indegree[dependent_key] == 0:
                queue.append(items_by_key[dependent_key])
    if len(ordered) != len(items):
        return sorted(items, key=lambda entry: (entry.sort_order, entry.key))
    return ordered


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
) -> bool:
    _update_processing_queue_checkpoint(
        db,
        run=run,
        status="running",
        current_deliverable_key=item.key,
        summary=f"Procesando {position} de {total_count}: {item.title}.",
    )
    step = _step_record(db, run=run, deliverable_key=item.key)
    attempt_count = int((step.checkpoint_payload or {}).get("attempt_count") or 0) + 1 if step is not None else 1
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
        },
        error_payload={},
    )
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
    if str(queue_checkpoint.get("status") or "") not in QUEUE_ACTIVE_STATUSES:
        return False

    cutoff = utc_now() - ORPHANED_JOB_TIMEOUT
    selected_keys = [str(value) for value in queue_checkpoint.get("selected_deliverable_keys", [])]
    active_steps = [
        step
        for step in list_product_build_steps(db, run_id=run.id)
        if step.deliverable_key in selected_keys and str(step.status or "") in QUEUE_ACTIVE_STEP_STATES
    ]
    stale_steps = [step for step in active_steps if step.updated_at <= cutoff]
    if not stale_steps:
        return False

    orphaned_keys: list[str] = []
    for step in stale_steps:
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
        summary=(
            f"Se detectaron {len(orphaned_keys)} jobs interrumpidos. "
            "No se reintentaron automáticamente para preservar idempotencia y control de costos."
        ),
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
