from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from app.api.routes.sessions import build_snapshot, ensure_commercial_capability, get_or_404, write_log
from app.db import get_session
from app.models import (
    AlertEventRecord,
    ArtifactBrowserResponse,
    ArtifactRegistryRecord,
    ArtifactStatus,
    ExecutionLogRecord,
    FeatureFlagUpdateRequest,
    IntegrationStatusEntry,
    IntegrationStatusRecord,
    MetricSnapshotRecord,
    MonitoringWorkspace,
    SessionSnapshot,
    SessionStage,
    UserRecord,
)
from app.services.auth_service import get_current_user
from app.services.memory_rollout import (
    build_memory_rollout_summary,
    expected_monitoring_stages,
)
from app.services.operations_service import (
    build_artifact_record_entry,
    build_integration_status_entry,
    build_monitoring_workspace,
    capture_operational_state,
    filter_artifact_records,
)
from app.services.memory_observability import build_memory_observability_report
from app.services.release_observability import build_release_observability
from app.services.llm_runtime.runtime_settings_service import load_effective_runtime_settings
from app.services.stage5_service import ensure_local_admin_can_govern, update_feature_flag


router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("/{session_id}/monitoring", response_model=MonitoringWorkspace)
def get_monitoring_workspace_route(
    session_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> MonitoringWorkspace:
    record = get_or_404(db, session_id, current_user.id)
    metric_records = db.exec(
        select(MetricSnapshotRecord)
        .where(MetricSnapshotRecord.session_id == session_id)
        .order_by(MetricSnapshotRecord.created_at.desc())
    ).all()
    if not metric_records:
        capture_operational_state(db, session_id=session_id, source_action="load_monitoring")
        db.commit()
        metric_records = db.exec(
            select(MetricSnapshotRecord)
            .where(MetricSnapshotRecord.session_id == session_id)
            .order_by(MetricSnapshotRecord.created_at.desc())
        ).all()
    alert_records = db.exec(
        select(AlertEventRecord)
        .where(AlertEventRecord.session_id == session_id)
        .order_by(AlertEventRecord.updated_at.desc(), AlertEventRecord.created_at.desc())
    ).all()
    recent_error_records = db.exec(
        select(ExecutionLogRecord)
        .where(
            ExecutionLogRecord.session_id == session_id,
            ExecutionLogRecord.status.in_([ArtifactStatus.failed, ArtifactStatus.needs_review]),
        )
        .order_by(ExecutionLogRecord.created_at.desc())
    ).all()
    integration_records = db.exec(
        select(IntegrationStatusRecord)
        .where(IntegrationStatusRecord.session_id == session_id)
        .order_by(IntegrationStatusRecord.checked_at.desc())
    ).all()
    snapshot = build_snapshot(db, record)
    runtime_settings = load_effective_runtime_settings(db, record.workspace_id)
    rollout_summary = build_memory_rollout_summary(
        runtime_settings,
        session=db,
        workspace_id=record.workspace_id,
    )
    return build_monitoring_workspace(
        metric_records=metric_records,
        alert_records=alert_records,
        recent_error_records=recent_error_records,
        integration_records=integration_records,
        memory_observability=build_memory_observability_report(
            skill_runs=snapshot.skill_runs,
            short_term_memory=snapshot.short_term_memory,
            expected_stages=expected_monitoring_stages(rollout_summary),
        ),
        release_observability=build_release_observability(
            snapshot,
            recent_error_records=recent_error_records,
            estimated_cost_usd=metric_records[0].cost_estimate_usd if metric_records else 0.0,
        ),
    )


@router.get("/{session_id}/artifacts", response_model=ArtifactBrowserResponse)
def list_artifacts_route(
    session_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> ArtifactBrowserResponse:
    record = get_or_404(db, session_id, current_user.id)
    ensure_commercial_capability(record, "library_workspace")
    records = db.exec(
        select(ArtifactRegistryRecord)
        .where(ArtifactRegistryRecord.session_id == session_id)
        .order_by(ArtifactRegistryRecord.created_at.desc())
    ).all()
    return ArtifactBrowserResponse(items=[build_artifact_record_entry(item) for item in records])


@router.get("/{session_id}/library", response_model=ArtifactBrowserResponse)
def query_library_route(
    session_id: UUID,
    q: str = "",
    artifact_kind: str = "",
    stage: SessionStage | None = None,
    blueprint_version_number: int | None = None,
    date_from: str = "",
    date_to: str = "",
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> ArtifactBrowserResponse:
    record = get_or_404(db, session_id, current_user.id)
    ensure_commercial_capability(record, "library_workspace")
    records = db.exec(
        select(ArtifactRegistryRecord)
        .where(ArtifactRegistryRecord.session_id == session_id)
        .order_by(ArtifactRegistryRecord.created_at.desc())
    ).all()
    filtered = filter_artifact_records(
        records,
        query=q,
        artifact_kind=artifact_kind,
        stage=stage,
        blueprint_version_number=blueprint_version_number,
        date_from=date_from,
        date_to=date_to,
    )
    return ArtifactBrowserResponse(items=[build_artifact_record_entry(item) for item in filtered])


@router.get("/{session_id}/integrations", response_model=list[IntegrationStatusEntry])
def list_integrations_route(
    session_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> list[IntegrationStatusEntry]:
    _ = get_or_404(db, session_id, current_user.id)
    records = db.exec(
        select(IntegrationStatusRecord)
        .where(IntegrationStatusRecord.session_id == session_id)
        .order_by(IntegrationStatusRecord.checked_at.desc())
    ).all()
    if not records:
        capture_operational_state(db, session_id=session_id, source_action="load_integrations")
        db.commit()
        records = db.exec(
            select(IntegrationStatusRecord)
            .where(IntegrationStatusRecord.session_id == session_id)
            .order_by(IntegrationStatusRecord.checked_at.desc())
        ).all()
    return [build_integration_status_entry(item) for item in records]


@router.post("/{session_id}/integrations/check", response_model=SessionSnapshot)
def check_integrations_route(
    session_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> SessionSnapshot:
    record = get_or_404(db, session_id, current_user.id)
    capture_operational_state(db, session_id=session_id, source_action="check_integrations")
    write_log(
        db,
        session_id=session_id,
        stage=record.current_stage,
        status_value=record.status,
        message="Integraciones verificadas",
        payload={"session_id": str(session_id)},
    )
    db.commit()
    return build_snapshot(db, record)


@router.patch("/{session_id}/feature-flags/{flag_key}", response_model=SessionSnapshot)
def update_feature_flag_route(
    session_id: UUID,
    flag_key: str,
    payload: FeatureFlagUpdateRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> SessionSnapshot:
    record = get_or_404(db, session_id, current_user.id)
    try:
        ensure_local_admin_can_govern(current_user.email)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    try:
        update_feature_flag(db, workspace_id=record.workspace_id, flag_key=flag_key, enabled=payload.enabled)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    write_log(
        db,
        session_id=session_id,
        stage=record.current_stage,
        status_value=record.status,
        message="Feature flag actualizada",
        payload={"flag_key": flag_key, "enabled": payload.enabled},
    )
    capture_operational_state(db, session_id=session_id, source_action="update_feature_flag")
    db.commit()
    return build_snapshot(db, record)
