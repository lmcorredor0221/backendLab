from __future__ import annotations

from typing import Any

from sqlmodel import Session

from app.models import SessionRecord, SessionSnapshot, UserRecord
from app.services.product_processing.acp_direct_service import ACP_REQUIRED_STAGE_KEYS, build_acp_direct_resolution
from app.services.product_processing.contracts import (
    ProductBuildLifecycle,
    ProductBuildProductKey,
    ProductBuildStatus,
)
from app.services.product_processing.persistence import ProductBuildRunRecord
from app.services.product_processing.product_build_orchestrator import (
    ProductBuildOrchestrationOptions,
    ensure_product_build_orchestration,
)
from app.services.product_processing.product_build_run_service import (
    list_product_build_runs,
    list_product_build_steps,
    update_product_build_run_state,
    upsert_product_build_step,
)
from app.services.product_processing.product_build_status_service import build_product_build_status


COMPLETED_STEP_STATES = {"available", "completed", "skipped"}
ACTIVE_STEP_STATES = {"queued", "generating", "running", "preparing"}
BLOCKING_STEP_STATES = {"error", "failed", "requires_attention", "locked"}


def ensure_acp_product_orchestration(
    db: Session,
    *,
    record: SessionRecord,
    snapshot: SessionSnapshot | None = None,
    current_user: UserRecord | None = None,
    execute_jobs: bool = False,
    allow_llm: bool = False,
    activation_payload: dict[str, Any] | None = None,
) -> ProductBuildStatus:
    """Synchronize ACP direct readiness with the portable product build run.

    ACP can be purchased before Blueprint Pro is fully enriched. The product run
    must therefore show the remaining Pro/LEAN dependencies as first-class steps,
    instead of forcing the user to discover missing work manually in other views.
    """

    resolution = build_acp_direct_resolution(db, record=record, snapshot=snapshot)
    can_execute_package_jobs = execute_jobs and resolution.can_start_package and resolution.can_export_package
    status = ensure_product_build_orchestration(
        db,
        record=record,
        product_key=ProductBuildProductKey.acp,
        current_user=current_user,
        options=ProductBuildOrchestrationOptions(
            current_stage="package",
            execute_jobs=can_execute_package_jobs,
            allow_llm=allow_llm,
            activation_payload={
                "source": "acp_product_orchestration",
                "route_kind": resolution.route_kind,
                "can_start_package": resolution.can_start_package,
                "can_export_package": resolution.can_export_package,
                **(activation_payload or {}),
            },
        ),
    )

    if record.workspace_id is None:
        return status
    runs = list_product_build_runs(
        db,
        workspace_id=record.workspace_id,
        session_id=record.id,
        product_key=ProductBuildProductKey.acp,
    )
    if not runs:
        return status

    run = runs[0]
    _sync_acp_readiness_steps(db, run=run, resolution=resolution)
    _finalize_acp_run_from_steps(db, run=run, resolution=resolution)
    db.flush()
    return build_product_build_status(db, record=record, product_key=ProductBuildProductKey.acp, current_user=current_user)


def _sync_acp_readiness_steps(db: Session, *, run: ProductBuildRunRecord, resolution) -> None:
    for index, stage in enumerate(resolution.stages, start=1):
        status = _stage_dependency_status(stage)
        upsert_product_build_step(
            db,
            run=run,
            step_key=f"acp_dependency:{stage.stage_key}",
            status=status,
            stage_key=stage.stage_key,
            dependency_key=f"lean_stage:{stage.stage_key}",
            sequence=8_000 + index,
            progress_percent=_progress_for_state(status),
            checkpoint_payload={
                "type": "acp_readiness_dependency",
                "route_kind": resolution.route_kind,
                "label": stage.label,
                "completed": stage.completed,
                "justified": stage.justified,
                "justification": stage.justification,
                "technical_question_count": stage.technical_question_count,
                "blocking_question_count": stage.blocking_question_count,
                "next_action": stage.next_action,
            },
            error_payload=_dependency_error_payload(stage, status=status),
        )


def _stage_dependency_status(stage) -> str:
    if stage.blocking_question_count > 0:
        return "requires_attention"
    if stage.completed or stage.justified:
        return "completed"
    return "requires_attention"


def _progress_for_state(status: str) -> int:
    if status in COMPLETED_STEP_STATES:
        return 100
    if status in BLOCKING_STEP_STATES:
        return 0
    if status in ACTIVE_STEP_STATES:
        return 40
    return 0


def _dependency_error_payload(stage, *, status: str) -> dict[str, Any]:
    if status != "requires_attention":
        return {}
    reasons: list[str] = []
    if not stage.completed and not stage.justified:
        reasons.append(f"missing_stage:{stage.stage_key}")
    if stage.blocking_question_count:
        reasons.append(f"blocking_questions:{stage.stage_key}:{stage.blocking_question_count}")
    return {
        "title": f"{stage.label} requiere cierre antes de Package",
        "message": stage.next_action or "Resuelve las preguntas o decisiones criticas antes de construir el ACP.",
        "reasons": reasons,
    }


def _finalize_acp_run_from_steps(db: Session, *, run: ProductBuildRunRecord, resolution) -> None:
    steps = list_product_build_steps(db, run_id=run.id)
    total_units = max(len(steps), len(ACP_REQUIRED_STAGE_KEYS), 1)
    completed_units = sum(1 for step in steps if step.status in COMPLETED_STEP_STATES)
    blocked_units = sum(1 for step in steps if step.status in BLOCKING_STEP_STATES)
    active_units = sum(1 for step in steps if step.status in ACTIVE_STEP_STATES)

    if blocked_units:
        lifecycle = ProductBuildLifecycle.requires_attention
    elif active_units:
        lifecycle = ProductBuildLifecycle.running
    elif completed_units >= total_units and resolution.can_export_package:
        lifecycle = ProductBuildLifecycle.completed
    elif completed_units:
        lifecycle = ProductBuildLifecycle.partial
    else:
        lifecycle = ProductBuildLifecycle.ready_to_start

    checkpoint = {
        **(run.checkpoint_payload or {}),
        "acp_direct_resolution": {
            "route_kind": resolution.route_kind,
            "required_stage_keys": list(resolution.required_stage_keys),
            "completed_stage_keys": list(resolution.completed_stage_keys),
            "missing_stage_keys": list(resolution.missing_stage_keys),
            "justified_stage_keys": list(resolution.justified_stage_keys),
            "can_start_package": resolution.can_start_package,
            "can_export_package": resolution.can_export_package,
            "total_technical_questions": resolution.total_technical_questions,
            "total_blocking_questions": resolution.total_blocking_questions,
            "readiness_blockers": list(resolution.readiness_blockers),
        },
    }
    update_product_build_run_state(
        db,
        run=run,
        lifecycle=lifecycle,
        completed_units=float(completed_units),
        total_units=float(total_units),
        blocked_units=float(blocked_units),
        checkpoint_payload=checkpoint,
    )
