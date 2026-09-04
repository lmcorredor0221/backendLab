import io
import hashlib
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from app.api.routes.sessions import (
    build_construction_question_views,
    build_construction_readiness_view,
    build_safe_download_filename,
    build_snapshot,
    ensure_acp_evaluation_seed_snapshot,
    ensure_commercial_capability,
    get_acp_file_entry,
    get_acp_knowledge_graph,
    get_construction_question_view,
    get_or_404,
    resolve_acp_preview,
    touch_session,
    upsert_construction_question_response,
    write_react_run,
    write_log,
)
from app.db import get_session
from app.models import (
    ACPFileEntry,
    ACPPreview,
    ACPValidationReport,
    BlueprintKnowledgeGraph,
    CommercialAuditReport,
    CommercialTier,
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
from app.services.blueprint_commercial_result_service import (
    SOURCE_ACTION as BLUEPRINT_COMMERCIAL_RESULT_ACTION,
    record_blueprint_commercial_result_artifacts,
)
from app.services.acp_continuity import (
    apply_uncertainty_backlog_acp_answer,
    build_construction_gaps_from_uncertainty_backlog,
    build_construction_gap_entries,
    build_continuity_answer_map,
    load_construction_question_response_records,
    load_construction_question_response_records_for_preview,
    load_uncertainty_backlog_records,
    merge_construction_question_records_with_uncertainty_backlog,
    sync_construction_question_response_records,
    uncertainty_backlog_id_from_question_key,
)
from app.services.acp_export_profiles import (
    apply_acp_export_profile,
    normalize_acp_export_profile,
    rebuild_profile_conformance_with_readiness,
)
from app.services.acp_generator import generate_acp_preview
from app.services.acp_validation import derive_acp_export_status, should_block_acp_export
from app.services.acp_zip_export import build_acp_zip
from app.services.export_delivery_service import _blueprint_markdown
from app.services.auth_service import get_current_user
from app.services.commerce_service import record_commercial_event as record_dedicated_commercial_event
from app.services.diagram_center.catalog_service import build_catalog_v3
from app.services.diagram_center.generation_service import create_generation_job, run_generation_job
from app.services.deliverable_catalog import (
    DeliverableGenerationMode,
    DeliverableGenerationTask,
    DeliverableType,
    list_registry_entries,
    run_deliverable_generation_task,
)
from app.services.estimation_calibration import persist_estimation_run
from app.services.estimation_service import build_estimation_report
from app.services.operations_service import (
    capture_operational_state,
    record_acp_preview_artifacts,
    record_estimation_artifact,
    record_export_artifact,
)
from app.services.product_processing import (
    AcpDirectRouteResolution,
    ProductBuildOrchestrationOptions,
    ProductBuildProductKey,
    ProductProcessingMode,
    acp_route_blocking_reasons,
    build_acp_direct_resolution,
    ensure_acp_product_orchestration,
    ensure_product_build_orchestration,
)
from app.services.stage5_service import FEATURE_FLAG_ESTIMATION, FEATURE_FLAG_REACT_RUNTIME, create_export_handoff, is_feature_flag_enabled
from app.services.agentic_runtime.stages.extended import ReactCapabilityOutput, run_callable_react
from app.services.workspace_bootstrap import apply_workspace_bootstrap


router = APIRouter(prefix="/sessions", tags=["sessions"])


def _latest_blueprint_version(snapshot: SessionSnapshot) -> int | None:
    if not snapshot.blueprint_versions:
        return None
    return snapshot.blueprint_versions[0].version_number


def _safe_snapshot_payload(value) -> dict:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return dict(value)
    return {"value": str(value)}


def _blueprint_commercial_deliverable_context(snapshot: SessionSnapshot) -> dict:
    return {
        "summary": (
            "Resultado comercial del Blueprint preparado desde contexto aprobado. "
            f"Proyecto: {snapshot.session.title}."
        ),
        "project_title": snapshot.session.title,
        "blueprint_version_number": _latest_blueprint_version(snapshot),
        "approved_context": {
            "discovery": _safe_snapshot_payload(snapshot.discovery),
            "canvas": _safe_snapshot_payload(snapshot.canvas),
            "blueprint": _safe_snapshot_payload(snapshot.blueprint),
            "tools": _safe_snapshot_payload(snapshot.latest_tool_recommendation),
            "estimation": _safe_snapshot_payload(snapshot.estimation_report),
        },
    }


def _is_blueprint_basic_auto_deliverable(entry) -> bool:
    return (
        entry.deliverable_type != DeliverableType.diagram
        and "blueprint" in entry.product_scope
        and entry.required_tier == CommercialTier.blueprint
        and entry.generation_mode
        not in {
            DeliverableGenerationMode.llm_required,
            DeliverableGenerationMode.manual_review_required,
        }
    )


def _generate_blueprint_basic_deliverables(
    db: Session,
    *,
    record,
    snapshot: SessionSnapshot,
    current_user: UserRecord,
) -> tuple[list[str], list[dict[str, str]]]:
    if record.workspace_id is None:
        return [], [{"deliverable_key": "*", "reason": "session_without_workspace"}]
    context_payload = _blueprint_commercial_deliverable_context(snapshot)
    generated_keys: list[str] = []
    skipped: list[dict[str, str]] = []
    for entry in list_registry_entries():
        if not _is_blueprint_basic_auto_deliverable(entry):
            continue
        task = DeliverableGenerationTask(
            workspace_id=record.workspace_id,
            session_id=record.id,
            deliverable_key=entry.deliverable_key,
            product_mode=ProductProcessingMode.basic_free.value,
            current_stage="estimate",
            tier=CommercialTier.blueprint,
            idempotency_key=f"blueprint-commercial-result:{record.id}:deliverable:{entry.deliverable_key}",
            requested_by_user_id=current_user.id,
            context_payload=context_payload,
            approved_context_refs=[
                "session.discovery",
                "session.canvas",
                "session.blueprint",
                "session.tools",
                "session.estimation_report",
            ],
            allow_llm=False,
            max_iterations=entry.prompt_policy.max_iterations or 1,
        )
        try:
            job, _ = run_deliverable_generation_task(db, task)
        except (LookupError, PermissionError, ValueError) as exc:
            skipped.append({"deliverable_key": entry.deliverable_key, "reason": str(exc)})
            continue
        generated_keys.append(f"{entry.deliverable_key}:{job.status}")
    return generated_keys, skipped


def resolve_acp_profile(profile: str | None) -> str:
    try:
        return normalize_acp_export_profile(profile)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


def _construction_question_context(db: Session, session_id: UUID):
    response_records = load_construction_question_response_records(db, session_id)
    backlog_records = load_uncertainty_backlog_records(db, session_id)
    preview_records = merge_construction_question_records_with_uncertainty_backlog(
        response_records,
        backlog_records,
    )
    extra_readiness_gaps = build_construction_gaps_from_uncertainty_backlog(backlog_records)
    continuity_answers = build_continuity_answer_map(preview_records)
    return response_records, preview_records, extra_readiness_gaps, continuity_answers


def resolve_profiled_preview(
    db: Session,
    record,
    *,
    profile: str,
) -> ACPPreview:
    preview = apply_acp_export_profile(resolve_acp_preview(db, record), profile)
    response_records = load_construction_question_response_records_for_preview(db, record.id)
    readiness = build_construction_readiness_view(preview, response_records)
    return rebuild_profile_conformance_with_readiness(
        preview,
        profile=profile,
        readiness=readiness,
    )


def ensure_acp_build_access(record, *, db: Session, current_user: UserRecord) -> None:
    ensure_commercial_capability(record, "acp.build", db=db, current_user=current_user)


def ensure_acp_route_ready_for_package(
    db: Session,
    *,
    record,
    current_user: UserRecord,
    snapshot=None,
) -> AcpDirectRouteResolution:
    resolved_snapshot = snapshot or build_snapshot(db, record, current_user=current_user)
    try:
        resolution = build_acp_direct_resolution(db, record=record, snapshot=resolved_snapshot)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    blockers = acp_route_blocking_reasons(resolution)
    if blockers:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "ACP requires completed LEAN stages or explicit justification before Package.",
                "blocking_reasons": blockers,
                "resolution": resolution.model_dump(mode="json"),
            },
        )
    return resolution


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


@router.post("/{session_id}/blueprint/commercial-result", response_model=SessionSnapshot)
def prepare_blueprint_commercial_result_route(
    session_id: UUID,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> SessionSnapshot:
    from app.services.product_processing.blueprint_basic_service import prepare_blueprint_basic_commercial_result

    record = get_or_404(db, session_id, current_user.id)
    snapshot, _ = prepare_blueprint_basic_commercial_result(
        db,
        record=record,
        current_user=current_user,
        background_tasks=background_tasks,
    )
    return snapshot


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


@router.get("/{session_id}/acp/direct-resolution", response_model=AcpDirectRouteResolution)
def get_acp_direct_resolution_route(
    session_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> AcpDirectRouteResolution:
    record = get_or_404(db, session_id, current_user.id)
    ensure_commercial_capability(record, "acp.invite", db=db, current_user=current_user)
    snapshot, _ = ensure_acp_evaluation_seed_snapshot(
        db,
        record,
        current_user=current_user,
        source_action="acp_direct_resolution",
        allow_write=False,
    )
    try:
        return build_acp_direct_resolution(db, record=record, snapshot=snapshot)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


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
    snapshot, _ = ensure_acp_evaluation_seed_snapshot(
        db,
        record,
        current_user=current_user,
        source_action="generate_acp_preview",
    )
    ensure_acp_route_ready_for_package(db, record=record, current_user=current_user, snapshot=snapshot)
    response_records, preview_records, extra_readiness_gaps, continuity_answers = _construction_question_context(db, session_id)
    react_run = None
    if is_feature_flag_enabled(db, FEATURE_FLAG_REACT_RUNTIME, workspace_id=record.workspace_id):
        react_execution = run_callable_react(
            stage="package",
            capability="generate_acp_preview",
            session_id=session_id,
            workspace_id=record.workspace_id,
            context_refs=["session.blueprint", "session.validate", "session.estimate", "knowledge.acp_portability"],
            runner=lambda: ReactCapabilityOutput(
                value=generate_acp_preview(snapshot, continuity_answers or None, preview_records, extra_readiness_gaps),
                summary="Package valido readiness, preguntas de implementacion y portabilidad del ACP.",
            ),
            validator=lambda value: (
                ["El ACP no contiene artefactos exportables."],
                not bool(getattr(value, "files", [])),
                "Package valido que el ACP tenga artefactos portables." if getattr(value, "files", []) else "Package requiere revision.",
            ),
            effective_language=current_user.preferred_language,
        )
        preview = react_execution.value
        react_run = react_execution.react_run
    else:
        preview = generate_acp_preview(snapshot, continuity_answers or None, preview_records, extra_readiness_gaps)
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
                preview = generate_acp_preview(snapshot, continuity_answers or None, preview_records, extra_readiness_gaps)
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
    write_react_run(
        db,
        session_id=session_id,
        result=react_run,
        stage="package",
        capability="generate_acp_preview",
        source_action="generate_acp_preview",
        blueprint_version_number=preview.blueprint_version_number,
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
    return resolve_profiled_preview(db, record, profile=normalized_profile)


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
    response_records = load_construction_question_response_records_for_preview(db, session_id)
    return build_construction_readiness_view(preview, response_records)


@router.get("/{session_id}/acp/questions", response_model=list[ConstructionQuestionViewEntry])
def get_acp_questions_route(
    session_id: UUID,
    profile: str = "extended",
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> list[ConstructionQuestionViewEntry]:
    record = get_or_404(db, session_id, current_user.id)
    preview = resolve_profiled_preview(db, record, profile=resolve_acp_profile(profile))
    response_records = load_construction_question_response_records_for_preview(db, session_id)
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
    ensure_acp_evaluation_seed_snapshot(
        db,
        record,
        current_user=current_user,
        source_action=f"answer_acp_question:{question_key}",
    )
    preview = resolve_acp_preview(db, record)
    response_records = load_construction_question_response_records_for_preview(db, session_id)
    question = get_construction_question_view(preview, question_key, response_records)
    backlog_id = uncertainty_backlog_id_from_question_key(question_key)
    if backlog_id is not None:
        try:
            apply_uncertainty_backlog_acp_answer(
                db,
                session_id=session_id,
                backlog_id=backlog_id,
                payload=payload,
                current_user=current_user,
            )
        except LookupError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    else:
        upsert_construction_question_response(
            db,
            session_id=session_id,
            question=question,
            payload=payload,
            current_user=current_user,
        )
    touch_session(record, record.current_stage, record.status)
    refreshed_records = load_construction_question_response_records_for_preview(db, session_id)
    refreshed_question = get_construction_question_view(preview, question_key, refreshed_records)
    impact_analysis = (
        refreshed_question.impact_analysis.model_dump(mode="json")
        if refreshed_question.impact_analysis is not None
        else None
    )
    event_key = "acp_question_delegated" if payload.decision == "delegate" else "acp_question_answered"
    write_log(
        db,
        session_id=session_id,
        stage=record.current_stage,
        status_value=record.status,
        message="Respuesta ACP registrada",
        payload={
            "event_key": event_key,
            "product": "acp",
            "source": "acp_question_resolution",
            "question_key": question_key,
            "gap_key": question.gap_key,
            "owner_role": payload.owner_role.strip() or question.target_owner,
            "metadata": {
                "decision_contract_version": "decision-observability.v1",
                "answer_decision": payload.decision,
                "question_origin": "uncertainty_backlog" if backlog_id is not None else "acp_preview",
                "backlog_id": str(backlog_id) if backlog_id is not None else "",
                "status": refreshed_question.status,
                "blocking": refreshed_question.blocking,
                "domain": refreshed_question.domain,
                "impacted_artifacts": list(refreshed_question.impacted_artifacts),
                "impact_analysis": impact_analysis,
                "reconciliation_policy": (
                    "answers_are_accumulated;reconciliation_requires_explicit_queue_action"
                ),
            },
        },
    )
    db.commit()
    return refreshed_question


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
    response_records = load_construction_question_response_records_for_preview(db, session_id)
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
    export_snapshot, _ = ensure_acp_evaluation_seed_snapshot(
        db,
        record,
        current_user=current_user,
        source_action="export_acp_zip",
    )
    direct_resolution = ensure_acp_route_ready_for_package(
        db,
        record=record,
        current_user=current_user,
        snapshot=export_snapshot,
    )
    preview = resolve_profiled_preview(db, record, profile=normalized_profile)
    response_records = load_construction_question_response_records_for_preview(db, session_id)
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

    overview_markdown = _blueprint_markdown(export_snapshot, preview).decode("utf-8")
    zip_bytes = build_acp_zip(
        preview,
        db=db,
        snapshot=export_snapshot,
        overview_markdown=overview_markdown,
    )
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
            "acp_direct_route_kind": direct_resolution.route_kind,
            "acp_required_stage_keys": direct_resolution.required_stage_keys,
            "acp_completed_stage_keys": direct_resolution.completed_stage_keys,
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
