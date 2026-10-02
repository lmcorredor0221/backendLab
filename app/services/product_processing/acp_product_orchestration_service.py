from __future__ import annotations

from typing import Any

from sqlmodel import Session, select

from app.models import (
    ACPBuildRunRecord,
    ACPPhaseRunRecord,
    ACPWorkflowRunStatus,
    CommercialTier,
    ExportJobRecord,
    ExportJobStatus,
    SessionRecord,
    SessionSnapshot,
    UserRecord,
)
from app.services.commerce_service import tier_rank
from app.services.product_processing.acp_direct_service import ACP_REQUIRED_STAGE_KEYS, build_acp_direct_resolution
from app.services.product_processing.contracts import (
    ProductBuildLifecycle,
    ProductBuildProductKey,
    ProductProcessingMode,
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
    ensure_product_build_run,
    update_product_build_run_state,
    upsert_product_build_step,
)
from app.services.product_processing.product_build_status_service import build_product_build_status


COMPLETED_STEP_STATES = {"available", "completed", "skipped"}
ACTIVE_STEP_STATES = {"queued", "generating", "running", "preparing"}
BLOCKING_STEP_STATES = {"error", "failed", "requires_attention", "locked"}
ACP_PRODUCT_PHASE_STAGE_KEYS = {
    "acp_input_readiness": "validate",
    "acp_questions_resolution": "validate",
    "acp_test_suite": "validate",
    "acp_graphic_simulation": "validate",
    "acp_quality_gates": "validate",
    "acp_artifact_reconciliation": "package",
    "acp_package_build": "package",
    "acp_download_ready": "package",
}
ACP_WORKFLOW_COMPLETED_STATUSES = {
    ACPWorkflowRunStatus.completed.value,
    ACPWorkflowRunStatus.completed_with_observations.value,
}
ACP_WORKFLOW_BLOCKING_STATUSES = {
    ACPWorkflowRunStatus.blocked.value,
    ACPWorkflowRunStatus.waiting_user.value,
}
ACP_WORKFLOW_ACTIVE_STATUSES = {
    ACPWorkflowRunStatus.running.value,
}


def ensure_acp_product_orchestration(
    db: Session,
    *,
    record: SessionRecord,
    snapshot: SessionSnapshot | None = None,
    current_user: UserRecord | None = None,
    execute_jobs: bool = False,
    allow_llm: bool = False,
    activation_payload: dict[str, Any] | None = None,
    catalog_stage_override: str | None = "package",
) -> ProductBuildStatus:
    """Synchronize ACP direct readiness with the portable product build run.

    ACP can be purchased before Blueprint Pro is fully enriched. The product run
    must therefore show the remaining Pro/LEAN dependencies as first-class steps,
    instead of forcing the user to discover missing work manually in other views.
    """
    current_tier = record.commercial_tier if record.commercial_tier is not None else CommercialTier.blueprint
    if tier_rank(current_tier) < tier_rank(CommercialTier.acp):
        return build_product_build_status(
            db,
            record=record,
            product_key=ProductBuildProductKey.acp,
            current_user=current_user,
            catalog_stage_override=catalog_stage_override,
        )

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
        catalog_stage_override=catalog_stage_override,
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
    return build_product_build_status(
        db,
        record=record,
        product_key=ProductBuildProductKey.acp,
        current_user=current_user,
        catalog_stage_override=catalog_stage_override,
    )


def _sync_acp_readiness_steps(db: Session, *, run: ProductBuildRunRecord, resolution) -> None:
    active_dependency_keys = {f"lean_stage:{stage.stage_key}" for stage in resolution.stages}
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
    for step in list_product_build_steps(db, run_id=run.id):
        checkpoint = step.checkpoint_payload or {}
        if checkpoint.get("type") != "acp_readiness_dependency":
            continue
        dependency_key = str(step.dependency_key or "")
        if dependency_key in active_dependency_keys:
            continue
        step.status = "skipped"
        step.progress_percent = 100
        step.error_payload = {}
        step.checkpoint_payload = {
            **checkpoint,
            "obsolete": True,
            "next_action": "",
        }
        db.add(step)


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
    readiness_blocked = bool(getattr(resolution, "readiness_blockers", None))
    if readiness_blocked and not blocked_units:
        blocked_units = 1

    if blocked_units or readiness_blocked:
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


def sync_acp_product_run_from_ready_export(
    db: Session,
    *,
    record: SessionRecord,
    current_user: UserRecord | None = None,
    export_job: ExportJobRecord | None = None,
) -> ProductBuildRunRecord | None:
    """Align the ACP product-build status with the real ACP workspace/export state."""
    if record.workspace_id is None:
        return None
    job = export_job if _is_ready_acp_export(export_job) else _latest_ready_acp_export(db, record=record)
    if job is None:
        return None

    workflow_run = _latest_acp_workflow_run(db, record=record)
    phases = _acp_phase_rows(db, workflow_run=workflow_run)
    product_run = _latest_acp_product_build_run(db, record=record)
    if product_run is None:
        product_run = ensure_product_build_run(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            product_key=ProductBuildProductKey.acp,
            product_mode=ProductProcessingMode.acp_implementation,
            idempotency_key=f"{record.id}:acp-workspace-product-build",
            entitlement_tier=CommercialTier.acp,
            access_state="allowed",
            lifecycle=ProductBuildLifecycle.ready_to_start,
            created_by_user_id=current_user.id if current_user is not None else None,
            checkpoint_payload={
                "product_key": ProductBuildProductKey.acp.value,
                "product_mode": ProductProcessingMode.acp_implementation.value,
                "sync_source": "acp_export_ready",
            },
        )

    completed_units = 0
    blocked_units = 0
    active_units = 0
    total_units = 1
    for phase in phases:
        total_units += 1
        step_status = _product_step_status_for_acp_phase(phase)
        if step_status in COMPLETED_STEP_STATES:
            completed_units += 1
        elif step_status in BLOCKING_STEP_STATES:
            blocked_units += 1
        elif step_status in ACTIVE_STEP_STATES:
            active_units += 1
        upsert_product_build_step(
            db,
            run=product_run,
            step_key=f"acp_phase:{phase.phase_key}",
            status=step_status,
            stage_key=ACP_PRODUCT_PHASE_STAGE_KEYS.get(str(phase.phase_key or ""), "package"),
            dependency_key=f"acp_phase:{phase.phase_key}",
            sequence=20_000 + int(phase.phase_order or 0),
            progress_percent=100 if step_status in COMPLETED_STEP_STATES else (40 if step_status in ACTIVE_STEP_STATES else 0),
            checkpoint_payload={
                "type": "acp_workspace_phase",
                "phase_key": phase.phase_key,
                "phase_label": phase.phase_label,
                "workflow_status": _status_value(phase.status),
                "attempt_count": phase.attempt_count,
                "warnings": phase.warnings,
                "blockers": phase.blockers,
            },
            error_payload=_phase_error_payload(phase, step_status=step_status),
        )

    completed_units += 1
    upsert_product_build_step(
        db,
        run=product_run,
        step_key="export:acp_portable_zip",
        status="available",
        stage_key="package",
        deliverable_key="acp_portable_zip",
        job_id=job.id,
        dependency_key="export_job:acp_portable_zip",
        sequence=20_999,
        progress_percent=100,
        checkpoint_payload={
            "type": "acp_export_job",
            "export_job_id": str(job.id),
            "artifact_kind": job.artifact_kind,
            "file_name": job.file_name,
            "size_bytes": job.size_bytes,
            "checksum_sha256": job.checksum_sha256,
            "status": _status_value(job.status),
        },
    )

    if blocked_units:
        lifecycle = ProductBuildLifecycle.requires_attention
    elif active_units:
        lifecycle = ProductBuildLifecycle.running
    elif completed_units >= total_units and phases:
        lifecycle = ProductBuildLifecycle.completed
    elif completed_units:
        lifecycle = ProductBuildLifecycle.partial
    else:
        lifecycle = ProductBuildLifecycle.ready_to_start

    checkpoint = {
        **(product_run.checkpoint_payload or {}),
        "acp_workspace": {
            "workflow_run_id": str(workflow_run.id) if workflow_run is not None else "",
            "workflow_status": _status_value(workflow_run.status) if workflow_run is not None else "",
            "phase_statuses": {phase.phase_key: _status_value(phase.status) for phase in phases},
        },
        "acp_export_job": {
            "export_job_id": str(job.id),
            "artifact_kind": job.artifact_kind,
            "file_name": job.file_name,
            "size_bytes": job.size_bytes,
            "checksum_sha256": job.checksum_sha256,
        },
        "sync_source": "acp_export_ready",
    }
    update_product_build_run_state(
        db,
        run=product_run,
        lifecycle=lifecycle,
        completed_units=float(completed_units),
        total_units=float(total_units),
        blocked_units=float(blocked_units),
        checkpoint_payload=checkpoint,
    )
    return product_run


def _status_value(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value or "")


def _is_ready_acp_export(job: ExportJobRecord | None) -> bool:
    return bool(
        job is not None
        and str(job.product_key or "") == ProductBuildProductKey.acp.value
        and str(job.artifact_kind or "") == "acp_portable_zip"
        and _status_value(job.status) == ExportJobStatus.ready.value
    )


def _latest_ready_acp_export(db: Session, *, record: SessionRecord) -> ExportJobRecord | None:
    if record.workspace_id is None:
        return None
    return db.exec(
        select(ExportJobRecord)
        .where(
            ExportJobRecord.workspace_id == record.workspace_id,
            ExportJobRecord.session_id == record.id,
            ExportJobRecord.product_key == ProductBuildProductKey.acp.value,
            ExportJobRecord.artifact_kind == "acp_portable_zip",
            ExportJobRecord.status == ExportJobStatus.ready,
        )
        .order_by(ExportJobRecord.updated_at.desc(), ExportJobRecord.created_at.desc())
    ).first()


def _latest_acp_workflow_run(db: Session, *, record: SessionRecord) -> ACPBuildRunRecord | None:
    if record.workspace_id is None:
        return None
    return db.exec(
        select(ACPBuildRunRecord)
        .where(
            ACPBuildRunRecord.workspace_id == record.workspace_id,
            ACPBuildRunRecord.session_id == record.id,
        )
        .order_by(ACPBuildRunRecord.updated_at.desc(), ACPBuildRunRecord.created_at.desc())
    ).first()


def _acp_phase_rows(db: Session, *, workflow_run: ACPBuildRunRecord | None) -> list[ACPPhaseRunRecord]:
    if workflow_run is None:
        return []
    return list(
        db.exec(
            select(ACPPhaseRunRecord)
            .where(ACPPhaseRunRecord.run_id == workflow_run.id)
            .order_by(ACPPhaseRunRecord.phase_order.asc(), ACPPhaseRunRecord.updated_at.asc())
        ).all()
    )


def _latest_acp_product_build_run(db: Session, *, record: SessionRecord) -> ProductBuildRunRecord | None:
    runs = list_product_build_runs(
        db,
        workspace_id=record.workspace_id,
        session_id=record.id,
        product_key=ProductBuildProductKey.acp,
    )
    return runs[0] if runs else None


def _product_step_status_for_acp_phase(phase: ACPPhaseRunRecord) -> str:
    value = _status_value(phase.status)
    if value in ACP_WORKFLOW_COMPLETED_STATUSES:
        return "completed"
    if value in ACP_WORKFLOW_BLOCKING_STATUSES:
        return "requires_attention"
    if value in ACP_WORKFLOW_ACTIVE_STATUSES:
        return "running"
    return "queued"


def _phase_error_payload(phase: ACPPhaseRunRecord, *, step_status: str) -> dict[str, Any]:
    if step_status != "requires_attention":
        return {}
    return {
        "title": f"{phase.phase_label or phase.phase_key} requiere atencion",
        "message": "Completa o desbloquea la fase ACP antes de marcar el paquete como listo.",
        "blockers": phase.blockers,
        "warnings": phase.warnings,
    }
