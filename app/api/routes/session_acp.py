import io
import hashlib
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from app.api.routes.sessions import (
    build_construction_question_views,
    build_construction_readiness_view,
    build_safe_download_filename,
    build_snapshot,
    ensure_commercial_capability,
    get_acp_file_entry,
    get_acp_knowledge_graph,
    get_construction_question_view,
    get_or_404,
    resolve_acp_preview,
    touch_session,
    upsert_construction_question_response,
    write_log,
)
from app.db import get_session
from app.models import (
    ACPFileEntry,
    ACPPreview,
    ACPValidationReport,
    BlueprintKnowledgeGraph,
    CommercialAuditReport,
    CommercialEventRequest,
    ConstructionGapEntry,
    ConstructionQuestionAnswerRequest,
    ConstructionQuestionViewEntry,
    ConstructionReadinessReport,
    SessionSnapshot,
    SessionStage,
    UserRecord,
)
from app.services.commercial_observability_service import build_commercial_audit_report
from app.services.acp_continuity import (
    build_construction_gap_entries,
    build_continuity_answer_map,
    load_construction_question_response_records,
    sync_construction_question_response_records,
)
from app.services.acp_export_profiles import (
    apply_acp_export_profile,
    normalize_acp_export_profile,
    rebuild_profile_conformance_with_readiness,
)
from app.services.acp_generator import generate_acp_preview
from app.services.acp_validation import derive_acp_export_status, should_block_acp_export
from app.services.acp_zip_export import build_acp_zip
from app.services.auth_service import get_current_user
from app.services.commerce_service import record_commercial_event as record_dedicated_commercial_event
from app.services.estimation_calibration import persist_estimation_run
from app.services.estimation_service import build_estimation_report
from app.services.operations_service import (
    capture_operational_state,
    record_acp_preview_artifacts,
    record_estimation_artifact,
    record_export_artifact,
)
from app.services.stage5_service import FEATURE_FLAG_ESTIMATION, create_export_handoff, is_feature_flag_enabled
from app.services.workspace_bootstrap import apply_workspace_bootstrap


router = APIRouter(prefix="/sessions", tags=["sessions"])


def resolve_acp_profile(profile: str | None) -> str:
    try:
        return normalize_acp_export_profile(profile)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


def resolve_profiled_preview(
    db: Session,
    record,
    *,
    profile: str,
) -> ACPPreview:
    return apply_acp_export_profile(resolve_acp_preview(db, record), profile)


def ensure_acp_build_access(record, *, db: Session, current_user: UserRecord) -> None:
    ensure_commercial_capability(record, "acp.build", db=db, current_user=current_user)


def record_commercial_event(
    session_id: UUID,
    payload: CommercialEventRequest,
    *,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    default_product: str,
    default_source: str,
    log_message: str,
    source_action_prefix: str,
) -> SessionSnapshot:
    record = get_or_404(db, session_id, current_user.id)
    ensure_commercial_capability(record, "acp.invite", db=db, current_user=current_user)
    event_key = payload.event_key.strip() or "unknown"
    source_action = f"{source_action_prefix}:{event_key[:64]}"
    write_log(
        db,
        session_id=session_id,
        stage=record.current_stage,
        status_value=record.status,
        message=log_message,
        payload={
            "event_key": event_key,
            "product": payload.product.strip() or default_product,
            "source": payload.source.strip() or default_source,
            "metadata": payload.metadata,
        },
    )
    record_dedicated_commercial_event(
        db,
        workspace_id=record.workspace_id,
        session_id=record.id,
        user_id=current_user.id,
        event_key=event_key,
        product_key=payload.product.strip() or default_product,
        source=payload.source.strip() or default_source,
        metadata=payload.metadata,
    )
    capture_operational_state(db, session_id=session_id, source_action=source_action)
    db.commit()
    return build_snapshot(db, record, current_user=current_user)


@router.post("/{session_id}/commercial-events", response_model=SessionSnapshot)
def record_commercial_event_route(
    session_id: UUID,
    payload: CommercialEventRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> SessionSnapshot:
    return record_commercial_event(
        session_id,
        payload,
        db=db,
        current_user=current_user,
        default_product="blueprint",
        default_source="commercial_surface",
        log_message="Evento comercial registrado",
        source_action_prefix="commercial_event",
    )


@router.get("/{session_id}/commercial-audit", response_model=CommercialAuditReport)
def get_commercial_audit_report_route(
    session_id: UUID,
    limit: int = 40,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> CommercialAuditReport:
    record = get_or_404(db, session_id, current_user.id)
    return build_commercial_audit_report(db, record=record, current_user=current_user, limit=limit)


@router.post("/{session_id}/acp/invitation-events", response_model=SessionSnapshot)
def record_acp_invitation_event_route(
    session_id: UUID,
    payload: CommercialEventRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> SessionSnapshot:
    return record_commercial_event(
        session_id,
        payload,
        db=db,
        current_user=current_user,
        default_product="acp",
        default_source="acp_gate",
        log_message="Evento comercial ACP registrado",
        source_action_prefix="acp_invitation",
    )


@router.post("/{session_id}/acp/generate", response_model=ACPPreview)
def generate_acp_route(
    session_id: UUID,
    profile: str = "extended",
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> ACPPreview:
    normalized_profile = resolve_acp_profile(profile)
    record = get_or_404(db, session_id, current_user.id)
    ensure_acp_build_access(record, db=db, current_user=current_user)
    snapshot = build_snapshot(db, record, current_user=current_user)
    response_records = load_construction_question_response_records(db, session_id)
    continuity_answers = build_continuity_answer_map(response_records)
    preview = generate_acp_preview(snapshot, continuity_answers or None)
    if snapshot.canvas is not None:
        apply_workspace_bootstrap(db, record.workspace_id)
        if is_feature_flag_enabled(db, FEATURE_FLAG_ESTIMATION, workspace_id=record.workspace_id):
            estimation_report = snapshot.estimation_report
            if estimation_report is None or estimation_report.is_stale:
                estimation_report = build_estimation_report(
                    db,
                    snapshot=snapshot,
                    acp_preview=preview,
                )
                snapshot.estimation_report = estimation_report
                preview = generate_acp_preview(snapshot, continuity_answers or None)
                record_estimation_artifact(
                    db,
                    session_id=session_id,
                    blueprint_version_number=preview.blueprint_version_number,
                    stage=record.current_stage,
                    source_action="generate_acp_preview",
                    estimation_report=estimation_report,
                )
                persist_estimation_run(
                    db,
                    session_id=session_id,
                    blueprint_version_number=preview.blueprint_version_number,
                    source_action="generate_acp_preview",
                    estimation_report=estimation_report,
                )
    sync_construction_question_response_records(db, preview, response_records)
    record_acp_preview_artifacts(
        db,
        session_id=session_id,
        preview=preview,
        source_action="generate_acp_preview",
    )
    write_log(
        db,
        session_id=session_id,
        stage=SessionStage.ready_for_export,
        status_value=derive_acp_export_status(preview.validation),
        message="ACP preview generado",
        payload={
            "file_count": len(preview.files),
            "completeness_percent": preview.validation.completeness_percent,
            "can_export_zip": preview.validation.can_export_zip,
        },
    )
    capture_operational_state(db, session_id=session_id, source_action="generate_acp_preview")
    db.commit()
    return apply_acp_export_profile(preview, normalized_profile)


@router.get("/{session_id}/acp/preview", response_model=ACPPreview)
def get_acp_preview_route(
    session_id: UUID,
    profile: str = "extended",
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> ACPPreview:
    record = get_or_404(db, session_id, current_user.id)
    ensure_acp_build_access(record, db=db, current_user=current_user)
    return resolve_profiled_preview(db, record, profile=resolve_acp_profile(profile))


@router.get("/{session_id}/acp/validate", response_model=ACPValidationReport)
def get_acp_validation_route(
    session_id: UUID,
    profile: str = "extended",
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> ACPValidationReport:
    record = get_or_404(db, session_id, current_user.id)
    ensure_acp_build_access(record, db=db, current_user=current_user)
    preview = resolve_profiled_preview(db, record, profile=resolve_acp_profile(profile))
    return preview.validation


@router.get("/{session_id}/acp/construction-readiness", response_model=ConstructionReadinessReport)
def get_acp_construction_readiness_route(
    session_id: UUID,
    profile: str = "extended",
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> ConstructionReadinessReport:
    record = get_or_404(db, session_id, current_user.id)
    ensure_acp_build_access(record, db=db, current_user=current_user)
    preview = resolve_profiled_preview(db, record, profile=resolve_acp_profile(profile))
    if (
        preview.construction_readiness.can_start_build
        and preview.construction_readiness.open_questions == 0
        and preview.construction_readiness.blocking_gaps == 0
    ):
        return preview.construction_readiness
    response_records = load_construction_question_response_records(db, session_id)
    return build_construction_readiness_view(preview, response_records)


@router.get("/{session_id}/acp/questions", response_model=list[ConstructionQuestionViewEntry])
def get_acp_questions_route(
    session_id: UUID,
    profile: str = "extended",
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> list[ConstructionQuestionViewEntry]:
    record = get_or_404(db, session_id, current_user.id)
    ensure_acp_build_access(record, db=db, current_user=current_user)
    preview = resolve_profiled_preview(db, record, profile=resolve_acp_profile(profile))
    response_records = load_construction_question_response_records(db, session_id)
    return build_construction_question_views(preview, response_records)


@router.patch("/{session_id}/acp/questions/{question_key}", response_model=ConstructionQuestionViewEntry)
def answer_acp_question_route(
    session_id: UUID,
    question_key: str,
    payload: ConstructionQuestionAnswerRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> ConstructionQuestionViewEntry:
    record = get_or_404(db, session_id, current_user.id)
    ensure_acp_build_access(record, db=db, current_user=current_user)
    preview = resolve_acp_preview(db, record)
    response_records = load_construction_question_response_records(db, session_id)
    question = get_construction_question_view(preview, question_key, response_records)
    upsert_construction_question_response(
        db,
        session_id=session_id,
        question=question,
        payload=payload,
        current_user=current_user,
    )
    touch_session(record, record.current_stage, record.status)
    write_log(
        db,
        session_id=session_id,
        stage=record.current_stage,
        status_value=record.status,
        message="Respuesta ACP registrada",
        payload={
            "question_key": question_key,
            "gap_key": question.gap_key,
            "owner_role": payload.owner_role.strip() or question.target_owner,
        },
    )
    db.commit()
    refreshed_records = load_construction_question_response_records(db, session_id)
    return get_construction_question_view(preview, question_key, refreshed_records)


@router.get("/{session_id}/acp/knowledge-graph", response_model=BlueprintKnowledgeGraph)
def get_acp_knowledge_graph_route(
    session_id: UUID,
    profile: str = "extended",
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> BlueprintKnowledgeGraph:
    record = get_or_404(db, session_id, current_user.id)
    ensure_acp_build_access(record, db=db, current_user=current_user)
    preview = resolve_profiled_preview(db, record, profile=resolve_acp_profile(profile))
    return get_acp_knowledge_graph(preview)


@router.get("/{session_id}/acp/gaps/{gap_key}", response_model=ConstructionGapEntry)
def get_acp_gap_route(
    session_id: UUID,
    gap_key: str,
    profile: str = "extended",
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> ConstructionGapEntry:
    record = get_or_404(db, session_id, current_user.id)
    ensure_acp_build_access(record, db=db, current_user=current_user)
    preview = resolve_profiled_preview(db, record, profile=resolve_acp_profile(profile))
    response_records = load_construction_question_response_records(db, session_id)
    for item in build_construction_gap_entries(preview, response_records):
        if item.gap_key == gap_key:
            return item
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ACP gap not found")


@router.get("/{session_id}/acp/files/{file_path:path}", response_model=ACPFileEntry)
def get_acp_file_route(
    session_id: UUID,
    file_path: str,
    profile: str = "extended",
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> ACPFileEntry:
    record = get_or_404(db, session_id, current_user.id)
    ensure_acp_build_access(record, db=db, current_user=current_user)
    preview = resolve_profiled_preview(db, record, profile=resolve_acp_profile(profile))
    return get_acp_file_entry(preview, file_path)


@router.get("/{session_id}/acp/export.zip")
def export_acp_zip_route(
    session_id: UUID,
    profile: str | None = None,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> StreamingResponse:
    legacy_compat_mode = profile is None
    normalized_profile = resolve_acp_profile(profile)
    record = get_or_404(db, session_id, current_user.id)
    ensure_commercial_capability(record, "acp.download", db=db, current_user=current_user)
    preview = resolve_profiled_preview(db, record, profile=normalized_profile)
    response_records = load_construction_question_response_records(db, session_id)
    readiness = build_construction_readiness_view(preview, response_records)
    if should_block_acp_export(preview.validation) or (not legacy_compat_mode and not readiness.can_start_build):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"ACP export profile '{normalized_profile}' is not ready; "
                f"resolve readiness gaps before exporting the zip package"
            ),
        )
    preview = rebuild_profile_conformance_with_readiness(
        preview,
        profile=normalized_profile,
        readiness=readiness,
    )

    zip_bytes = build_acp_zip(preview)
    zip_checksum = hashlib.sha256(zip_bytes).hexdigest()
    export_record = record_export_artifact(
        db,
        session_id=session_id,
        blueprint_version_number=preview.blueprint_version_number,
        artifact_key="acp_zip_export",
        artifact_title="ACP export zip",
        export_format="zip",
        content_text=f"ACP ZIP export with {len(preview.files)} files",
        source_action="export_acp_zip",
        artifact_metadata_extra={
            "bundle_version": preview.package_version,
            "checksum_sha256": zip_checksum,
            "file_count": len(preview.files),
            "manifest_path": preview.manifest_path,
            "completeness_percent": preview.validation.completeness_percent,
            "binary_size_bytes": len(zip_bytes),
            "legacy_compat_mode": legacy_compat_mode,
            "profile": normalized_profile,
            "readiness": readiness.overall_status,
        },
    )
    create_export_handoff(
        db,
        session_record=record,
        blueprint_version_number=export_record.blueprint_version_number,
        source_action="export_acp_zip",
        artifact_key=export_record.artifact_key,
    )
    write_log(
        db,
        session_id=session_id,
        stage=SessionStage.ready_for_export,
        status_value=derive_acp_export_status(preview.validation),
        message="ACP zip exportado",
        payload={
            "source_action": "export_acp_zip",
            "artifact_key": "acp_zip_export",
            "profile": normalized_profile,
            "file_count": len(preview.files),
            "binary_size_bytes": len(zip_bytes),
        },
    )
    capture_operational_state(db, session_id=session_id, source_action="export_acp_zip")
    db.commit()

    safe_filename = build_safe_download_filename(record.title, "-acp.zip")
    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_filename}"',
            "X-Acp-Export-Checksum-SHA256": zip_checksum,
            "X-Acp-Export-Profile": normalized_profile,
            "X-Acp-Export-Readiness": readiness.overall_status,
            "X-Acp-File-Count": str(len(preview.files)),
            "X-Acp-Manifest-Path": preview.manifest_path,
            "X-Acp-Package-Version": preview.package_version,
            "X-Acp-Export-Compatibility": "legacy" if legacy_compat_mode else "strict",
        },
    )
