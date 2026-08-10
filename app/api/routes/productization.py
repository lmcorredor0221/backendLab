from __future__ import annotations

import io
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlmodel import Session, select

from app.api.routes.sessions import (
    build_construction_readiness_view,
    build_snapshot,
    ensure_commercial_capability,
    get_or_404,
    resolve_acp_preview,
)
from app.db import get_session
from app.models import (
    ACPBuildRunRecord,
    ACPPhaseCommandRequest,
    ACPWorkspaceResponse,
    ActivityResponse,
    ActivityTimelineEntry,
    AttentionActionRequestV2,
    AttentionActionResultV2,
    AttentionResponse,
    AttentionResponseV2,
    CommercialAccessRequestRecord,
    CommercialAccessRequestStatus,
    CommercialEventRecord,
    DiagramCatalogV2Response,
    ExportCatalogResponse,
    ExportJobCreateRequest,
    ExportJobResponse,
    LauncherMetadataResponse,
    LauncherReportResponse,
    LauncherReportSubmitRequest,
    PlanAccessResponse,
    SessionRecord,
    UserRecord,
)
from app.services.acp_continuity import load_construction_question_response_records
from app.services.acp_launcher_service import build_launcher_metadata, submit_launcher_report
from app.services.acp_workflow_service import build_acp_workspace_response, ensure_acp_run, run_acp_phase
from app.services.attention_service import build_attention_response
from app.services.attention_service import (
    apply_attention_action_v2,
    build_attention_metrics_v2,
    build_attention_response_v2,
)
from app.services.auth_service import get_current_user
from app.services.commerce_service import list_active_products, serialize_access_request
from app.services.commercial_access import build_commercial_access_snapshot_v2, resolve_session_entitlement_context
from app.services.commercial_observability_service import build_commercial_audit_report
from app.services.diagram_catalog_service import build_diagram_catalog
from app.services.export_delivery_service import (
    build_export_catalog,
    cancel_export_job_response,
    create_export_job,
    get_export_job_response,
    read_export_job_bytes,
    retry_export_job_response,
)


router = APIRouter(prefix="/sessions", tags=["productization"])


def _context(
    db: Session,
    record: SessionRecord,
    current_user: UserRecord,
):
    snapshot = build_snapshot(db, record, current_user=current_user)
    preview = resolve_acp_preview(db, record)
    response_records = load_construction_question_response_records(db, record.id)
    readiness = build_construction_readiness_view(preview, response_records)
    access = build_commercial_access_snapshot_v2(db, record, current_user=current_user)
    return snapshot, preview, readiness, access


@router.get("/{session_id}/acp/workspace", response_model=ACPWorkspaceResponse)
def get_acp_workspace_route(
    session_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> ACPWorkspaceResponse:
    record = get_or_404(db, session_id, current_user.id)
    ensure_commercial_capability(record, "acp.build", db=db, current_user=current_user)
    snapshot, preview, readiness, access = _context(db, record, current_user)
    response = build_acp_workspace_response(
        db,
        record=record,
        current_user=current_user,
        snapshot=snapshot,
        preview=preview,
        readiness=readiness,
        access=access,
    )
    db.commit()
    return response


@router.post("/{session_id}/acp/workspace/phases/{phase_key}/run", response_model=ACPWorkspaceResponse)
def run_acp_workspace_phase_route(
    session_id: UUID,
    phase_key: str,
    payload: ACPPhaseCommandRequest | None = None,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> ACPWorkspaceResponse:
    record = get_or_404(db, session_id, current_user.id)
    ensure_commercial_capability(record, "acp.build", db=db, current_user=current_user)
    snapshot, preview, readiness, access = _context(db, record, current_user)
    run = ensure_acp_run(db, record=record, current_user=current_user, snapshot=snapshot)
    try:
        run_acp_phase(
            db,
            run=run,
            phase_key=phase_key,
            payload=payload or ACPPhaseCommandRequest(),
            preview=preview,
            readiness=readiness,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    response = build_acp_workspace_response(
        db,
        record=record,
        current_user=current_user,
        snapshot=snapshot,
        preview=preview,
        readiness=readiness,
        access=access,
    )
    db.commit()
    return response


@router.post("/{session_id}/acp/workspace/resume", response_model=ACPWorkspaceResponse)
def resume_acp_workspace_route(
    session_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> ACPWorkspaceResponse:
    record = get_or_404(db, session_id, current_user.id)
    ensure_commercial_capability(record, "acp.build", db=db, current_user=current_user)
    snapshot, preview, readiness, access = _context(db, record, current_user)
    response = build_acp_workspace_response(
        db,
        record=record,
        current_user=current_user,
        snapshot=snapshot,
        preview=preview,
        readiness=readiness,
        access=access,
    )
    db.commit()
    return response


@router.get("/{session_id}/attention", response_model=AttentionResponse)
def get_attention_route(
    session_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> AttentionResponse:
    record = get_or_404(db, session_id, current_user.id)
    snapshot, _, readiness, access = _context(db, record, current_user)
    return build_attention_response(db, record=record, snapshot=snapshot, readiness=readiness, access=access)


@router.get("/{session_id}/attention-v2", response_model=AttentionResponseV2)
def get_attention_v2_route(
    session_id: UUID,
    current_stage: str = Query(default=""),
    stage: str | None = Query(default=None),
    product: str | None = Query(default=None),
    severity: str | None = Query(default=None),
    item_type: str | None = Query(default=None, alias="type"),
    item_status: str | None = Query(default=None, alias="status"),
    cursor: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=100),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> AttentionResponseV2:
    record = get_or_404(db, session_id, current_user.id)
    snapshot, _, readiness, access = _context(db, record, current_user)
    return build_attention_response_v2(
        db,
        record=record,
        snapshot=snapshot,
        readiness=readiness,
        access=access,
        current_stage=current_stage,
        stage=stage,
        product=product,
        severity=severity,
        item_type=item_type,
        item_status=item_status,
        cursor=cursor,
        limit=limit,
    )


@router.get("/{session_id}/attention-v2/metrics", response_model=dict)
def get_attention_v2_metrics_route(
    session_id: UUID,
    current_stage: str = Query(default=""),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> dict:
    record = get_or_404(db, session_id, current_user.id)
    snapshot, _, readiness, access = _context(db, record, current_user)
    return build_attention_metrics_v2(
        db,
        record=record,
        snapshot=snapshot,
        readiness=readiness,
        access=access,
        current_stage=current_stage,
    )


@router.post("/{session_id}/attention-v2/{item_key}/actions", response_model=AttentionActionResultV2)
def post_attention_v2_action_route(
    session_id: UUID,
    item_key: str,
    payload: AttentionActionRequestV2,
    current_stage: str = Query(default=""),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> AttentionActionResultV2:
    record = get_or_404(db, session_id, current_user.id)
    snapshot, _, readiness, access = _context(db, record, current_user)
    result = apply_attention_action_v2(
        db,
        record=record,
        snapshot=snapshot,
        readiness=readiness,
        access=access,
        current_user=current_user,
        item_key=item_key,
        payload=payload,
    )
    db.commit()
    snapshot, _, readiness, access = _context(db, record, current_user)
    attention = build_attention_response_v2(
        db,
        record=record,
        snapshot=snapshot,
        readiness=readiness,
        access=access,
        current_stage=current_stage,
        limit=50,
    )
    return AttentionActionResultV2(
        session_id=record.id,
        workspace_id=record.workspace_id,
        item_key=item_key,
        action_kind=payload.action_kind,
        status=result.status,
        message=result.message,
        attention=attention,
    )


@router.get("/{session_id}/exports/catalog", response_model=ExportCatalogResponse)
def get_export_catalog_route(
    session_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> ExportCatalogResponse:
    record = get_or_404(db, session_id, current_user.id)
    access = build_commercial_access_snapshot_v2(db, record, current_user=current_user)
    return build_export_catalog(record=record, access=access)


@router.post("/{session_id}/exports/jobs", response_model=ExportJobResponse)
def create_export_job_route(
    session_id: UUID,
    payload: ExportJobCreateRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> ExportJobResponse:
    record = get_or_404(db, session_id, current_user.id)
    snapshot, preview, _, access = _context(db, record, current_user)
    try:
        response = create_export_job(
            db,
            record=record,
            current_user=current_user,
            access=access,
            snapshot=snapshot,
            preview=preview,
            payload=payload,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    db.commit()
    return response


@router.get("/{session_id}/exports/jobs/{job_id}", response_model=ExportJobResponse)
def get_export_job_route(
    session_id: UUID,
    job_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> ExportJobResponse:
    record = get_or_404(db, session_id, current_user.id)
    try:
        response = get_export_job_response(db, record=record, job_id=job_id)
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    db.commit()
    return response


@router.post("/{session_id}/exports/jobs/{job_id}/cancel", response_model=ExportJobResponse)
def cancel_export_job_route(
    session_id: UUID,
    job_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> ExportJobResponse:
    record = get_or_404(db, session_id, current_user.id)
    try:
        response = cancel_export_job_response(db, record=record, current_user=current_user, job_id=job_id)
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    db.commit()
    return response


@router.post("/{session_id}/exports/jobs/{job_id}/retry", response_model=ExportJobResponse)
def retry_export_job_route(
    session_id: UUID,
    job_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> ExportJobResponse:
    record = get_or_404(db, session_id, current_user.id)
    snapshot, preview, _, access = _context(db, record, current_user)
    try:
        response = retry_export_job_response(
            db,
            record=record,
            current_user=current_user,
            access=access,
            snapshot=snapshot,
            preview=preview,
            job_id=job_id,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    db.commit()
    return response


@router.get("/{session_id}/exports/jobs/{job_id}/download")
def download_export_job_route(
    session_id: UUID,
    job_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> StreamingResponse:
    record = get_or_404(db, session_id, current_user.id)
    try:
        job, payload = read_export_job_bytes(db, record=record, job_id=job_id)
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    db.commit()
    return StreamingResponse(
        io.BytesIO(payload),
        media_type=job.content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{job.file_name}"',
            "Cache-Control": "private, no-store",
            "X-Export-Job-Id": str(job.id),
            "X-Export-Checksum-SHA256": job.checksum_sha256,
            "X-Export-Expires-At": job.expires_at.isoformat() if job.expires_at else "",
        },
    )


@router.get("/{session_id}/acp/launcher", response_model=LauncherMetadataResponse)
def get_acp_launcher_metadata_route(
    session_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> LauncherMetadataResponse:
    record = get_or_404(db, session_id, current_user.id)
    ensure_commercial_capability(record, "acp.build", db=db, current_user=current_user)
    _, preview, _, _ = _context(db, record, current_user)
    return build_launcher_metadata(record=record, preview=preview)


@router.post("/{session_id}/acp/launcher/report", response_model=LauncherReportResponse)
def submit_acp_launcher_report_route(
    session_id: UUID,
    payload: LauncherReportSubmitRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> LauncherReportResponse:
    record = get_or_404(db, session_id, current_user.id)
    ensure_commercial_capability(record, "acp.build", db=db, current_user=current_user)
    response = submit_launcher_report(db, record=record, current_user=current_user, payload=payload)
    db.commit()
    return response


@router.get("/{session_id}/activity", response_model=ActivityResponse)
def get_activity_route(
    session_id: UUID,
    limit: int = Query(default=40, ge=1, le=100),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> ActivityResponse:
    record = get_or_404(db, session_id, current_user.id)
    audit = build_commercial_audit_report(db, record=record, current_user=current_user, limit=limit)
    events = db.exec(
        select(CommercialEventRecord)
        .where(CommercialEventRecord.workspace_id == record.workspace_id, CommercialEventRecord.session_id == record.id)
        .order_by(CommercialEventRecord.created_at.desc())
        .limit(limit)
    ).all()
    timeline = [
        ActivityTimelineEntry(
            key=str(item.id),
            type="commercial",
            title=item.event_key,
            product_key=item.product_key,
            source=item.source,
            status="recorded",
            revenue_cents=item.revenue_cents,
            currency=item.currency,
            created_at=item.created_at,
            metadata=item.metadata_payload,
        )
        for item in events
    ]
    return ActivityResponse(
        session_id=record.id,
        workspace_id=record.workspace_id,
        metrics=audit.metrics,
        funnel=audit.funnel,
        timeline=timeline,
    )


@router.get("/{session_id}/plan-access", response_model=PlanAccessResponse)
def get_plan_access_route(
    session_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> PlanAccessResponse:
    record = get_or_404(db, session_id, current_user.id)
    access = build_commercial_access_snapshot_v2(db, record, current_user=current_user)
    pending_requests = db.exec(
        select(CommercialAccessRequestRecord).where(
            CommercialAccessRequestRecord.workspace_id == record.workspace_id,
            CommercialAccessRequestRecord.session_id == record.id,
            CommercialAccessRequestRecord.status == CommercialAccessRequestStatus.pending,
        )
    ).all()
    return PlanAccessResponse(
        session_id=record.id,
        workspace_id=record.workspace_id,
        access=access,
        products=list_active_products(db),
        pending_requests=[serialize_access_request(item) for item in pending_requests],
        entitlements=access.entitlements,
    )


@router.get("/{session_id}/diagrams/catalog-v2", response_model=DiagramCatalogV2Response, deprecated=True)
def get_diagram_catalog_v2_route(
    session_id: UUID,
    limit: int = Query(default=20, ge=1, le=50),
    cursor: str | None = None,
    category: str | None = None,
    q: str | None = None,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> DiagramCatalogV2Response:
    record = get_or_404(db, session_id, current_user.id)
    snapshot, preview, _, _ = _context(db, record, current_user)
    context = resolve_session_entitlement_context(db, record, current_user)
    catalog = build_diagram_catalog(snapshot=snapshot, preview=preview, context=context, workspace_id=record.workspace_id)
    entries = catalog.entries
    if category:
        normalized_category = category.strip().lower()
        entries = [item for item in entries if item.category.lower() == normalized_category]
    if q:
        query = q.strip().lower()
        entries = [
            item
            for item in entries
            if query in item.title.lower() or query in item.summary.lower() or query in item.category.lower()
        ]
    start = 0
    if cursor:
        try:
            start = max(0, int(cursor))
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid cursor") from exc
    page = entries[start : start + limit]
    next_index = start + len(page)
    return DiagramCatalogV2Response(
        session_id=catalog.session_id,
        workspace_id=catalog.workspace_id,
        current_stage=catalog.current_stage,
        tier=catalog.tier,
        total_count=len(entries),
        unlocked_count=sum(1 for item in entries if item.access_state == "unlocked"),
        locked_count=sum(1 for item in entries if item.access_state in {"locked_blueprint", "locked_acp", "stage_locked"}),
        sample_count=sum(1 for item in entries if item.access_state == "sample"),
        pending_count=sum(1 for item in entries if item.access_state == "not_generated"),
        entries=page,
        limit=limit,
        next_cursor=str(next_index) if next_index < len(entries) else None,
        has_more=next_index < len(entries),
    )
