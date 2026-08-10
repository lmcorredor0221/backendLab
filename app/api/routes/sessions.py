import io
import hashlib
import json
import re
import unicodedata
from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from sqlalchemy import func, or_
from sqlmodel import Session, select

from app.db import get_session
from app.models import (
    ACPFileEntry,
    ACPPreview,
    ACPValidationReport,
    AlertEventRecord,
    ApproveToolsSelectionRequest,
    ArtifactBrowserResponse,
    ArtifactRegistryRecord,
    ApprovalGateEntry,
    ApprovalGateRecord,
    ApprovalResolutionRequest,
    ApprovalStatus,
    ArtifactStatus,
    BlueprintArtifact,
    BlueprintEnvelope,
    BlueprintKnowledgeGraph,
    BlueprintPatchRequest,
    BlueprintRecord,
    BlueprintTool,
    BlueprintVersionEntry,
    BlueprintVersionRecord,
    CanvasArtifact,
    CanvasEnvelope,
    CanvasRecord,
    CommercialTierUpdateRequest,
    ConstructionGapEntry,
    ConstructionQuestionAnswerRequest,
    ConstructionQuestionResponseRecord,
    ConstructionQuestionViewEntry,
    ConstructionReadinessReport,
    DesignProposalRequest,
    DesignRecommendationArtifact,
    DiscoveryArtifact,
    DiscoveryEnvelope,
    DiscoveryInput,
    EvidenceItem,
    EstimationEnvelope,
    EstimationAnalysisDecisionRequest,
    EstimationActualsUpsertRequest,
    EstimationReportArtifact,
    ProjectActualsEntry,
    EvaluationArtifact,
    EvaluationCaseResult,
    EvaluationCaseRecord,
    EvaluationDatasetCase,
    EvaluationDatasetArtifact,
    EvaluationDatasetRecord,
    EvaluationDatasetUpdateRequest,
    EvaluationEnvelope,
    EvaluationResultRecord,
    EvaluationRecord,
    EvaluationRubricArtifact,
    EvaluationRubricRecord,
    EvaluationRubricUpdateRequest,
    EvaluationRunEntry,
    EvaluationRunRecord,
    EvaluationRunSummary,
    ExecutionLogEntry,
    ExecutionLogRecord,
    FeatureFlagUpdateRequest,
    GovernancePolicyRecord,
    HandoffRecord,
    HandoffRecordEntry,
    HandoffResolutionRequest,
    IntegrationStatusEntry,
    IntegrationStatusRecord,
    JourneyArtifactEvidenceEntry,
    JourneyArtifactState,
    JourneyStageArtifactApprovalRequest,
    JourneyStageArtifactCreateRequest,
    JourneyStageArtifactEntry,
    JourneyStageArtifactListResponse,
    JourneyStageArtifactRecord,
    JourneyStageArtifactPatchRequest,
    JourneyStageArtifactRejectionRequest,
    KnowledgeProfile,
    LLMContextTrace,
    MemoryRecommendationRequest,
    MemoryRecommendationSourceStageVersions,
    MemoryProfile,
    MetricSnapshotRecord,
    MonitoringWorkspace,
    OpportunityRecord,
    ProjectTitleSource,
    SessionCapabilities,
    SessionCreateResponse,
    SessionDeleteRequest,
    SessionListFacets,
    SessionListPageInfo,
    SessionListResponse,
    SessionOwnerSummary,
    SessionRenameRequest,
    SessionRecord,
    ReviewState,
    SessionSnapshot,
    SessionStage,
    SkillCatalogRecord,
    SkillDefinition,
    SkillRunArtifact,
    SkillRunArtifactRecord,
    SkillRunEntry,
    SkillRunRecord,
    SkillRerunResponse,
    ShortTermMemoryRollbackRequest,
    ShortTermMemoryRuntimeState,
    SubagentRunEntry,
    SubagentRunRecord,
    ToolApiBindRequest,
    ToolRecommendationRequest,
    ToolRecommendationSourceStageVersions,
    ToolRecommendationArtifact,
    ToolRecommendationEnvelope,
    UserRecord,
    ValidationSimulationEventInjectionRequest,
    ValidationSimulationJudgeRequest,
    ValidationScenarioGenerationRequest,
    ValidationSimulationRunRequest,
    ValidationSimulationRunStateRecord,
    ValidationReportRecord,
    WorkspaceMembershipRecord,
    WorkspaceRole,
    WorkflowTemplateApplyRequest,
    WorkflowTemplateEntry,
    WorkflowTemplateRecord,
    SimulationEvent,
    SimulationJudgement,
    SimulationRunRecord,
    SimulationSpecificationArtifact,
    SimulationScenario,
)
from app.services.estimation_analysis_service import (
    apply_estimation_analysis,
    apply_estimation_analysis_decision,
    build_estimation_benchmark_corpus_hash,
    build_estimation_deterministic_inputs,
    build_estimation_pricing_catalog_signature,
    build_estimation_stale_reasons,
    build_estimation_validation_fingerprint,
    build_estimation_validation_fingerprint_from_state,
    run_estimation_analysis,
)
from app.services.acp_continuity import (
    build_construction_gap_entries,
    build_construction_question_views as build_construction_question_views_with_answers,
    build_continuity_answer_map,
    load_construction_question_response_records,
    overlay_construction_readiness,
    sync_construction_question_response_records,
)
from app.services.acp_generator import generate_acp_preview
from app.services.acp_validation import derive_acp_export_status, should_block_acp_export
from app.services.acp_zip_export import build_acp_zip
from app.services.blueprint_consistency_service import (
    ensure_blueprint_consistency_report,
    render_blueprint_consistency_markdown,
)
from app.services.blueprint_hydration import hydrate_blueprint_record
from app.core.config import allow_demo_tier_upgrade
from app.services.commercial_access import (
    resolve_session_commercial_access,
    resolve_session_entitlement_context,
    validate_capability,
)
from app.services.commerce_service import record_commercial_event as record_dedicated_commercial_event
from app.services.operations_service import (
    build_alert_event_entry,
    build_artifact_record_entry,
    build_integration_status_entry,
    build_metric_snapshot_entry,
    build_monitoring_workspace,
    capture_operational_state,
    filter_artifact_records,
    record_acp_preview_artifacts,
    record_delivery_artifacts,
    record_estimation_artifact,
    record_export_artifact,
)
from app.services.auth_service import get_current_user
from app.services.builder_service import patch_blueprint
from app.services.evaluation_workbench import (
    build_default_evaluation_dataset,
    build_default_evaluation_rubric,
    score_evaluation_workbench,
)
from app.services.estimation_calibration import (
    build_estimation_error_metric_entry,
    build_estimation_run_entry,
    build_project_actuals_entry,
    list_estimation_error_metrics,
    list_estimation_runs,
    list_project_actuals,
    persist_estimation_run,
    upsert_project_actuals,
)
from app.services.estimation_service import build_estimation_report
from app.services.journey_stage_migration import JourneyStageMigrationService
from app.services.memory_recommendation_service import build_memory_recommendation_artifact
from app.services.memory_rollout import FEATURE_FLAG_MEMORY_HYBRID_EXTENDED_JOURNEY
from app.services.llm_runtime.runtime_settings_service import load_effective_runtime_settings
from app.services.llm_runtime.stage_context_service import StageContextService
from app.services.llm_runtime.stage_context_types import StageContextBundle
from app.services.stage_proposal_service import (
    StageProposalConflictError,
    StageProposalNotFoundError,
    StageProposalService,
)
from app.services.canonical_export_delivery import (
    CANONICAL_EXPORT_ARTIFACTS,
    CanonicalExportKind,
    build_canonical_export_document,
    build_canonical_export_headers,
    should_block_canonical_export,
)
from app.services.skill_runtime import (
    list_skill_definitions,
    rerun_skill_for_session,
    run_discovery_analysis_stage,
    run_blueprint_stage,
    run_canvas_stage,
    run_design_stage,
    run_definition_stage,
    run_discovery_stage,
    run_enrich_stage,
    run_evaluation_stage,
    run_memory_recommendation_stage,
    run_tool_recommendation_stage,
    validate_definition_artifact,
)
from app.services.llm_runtime.builder_contracts import DiscoveryAnalysisOutput, RequirementsDefinitionOutput
from app.services.rules import derive_knowledge_profile, find_missing_discovery_fields
from app.services.short_term_memory import MAIN_BRANCH_KEY, ShortTermMemoryService
from app.services.tool_recommendation_service import (
    annotate_tool_recommendation_status,
    promote_tool_recommendation_to_blueprint_tools,
)
from app.services.validation_simulation_service import (
    build_validation_simulation_specification,
    execute_validation_simulation,
    judge_validation_simulation_run,
)
from app.services.stage5_service import (
    FEATURE_FLAG_ESTIMATION,
    FEATURE_FLAG_GOVERNANCE,
    FEATURE_FLAG_MULTI_AGENT_RUNTIME,
    FEATURE_FLAG_SUBAGENTS,
    FEATURE_FLAG_TOOL_RECOMMENDATION,
    FEATURE_FLAG_WORKFLOWS,
    HANDOFF_STATUS_COMPLETED,
    HANDOFF_STATUS_RETURNED,
    apply_workflow_template,
    build_handoff_record_entry,
    build_subagent_run_entry,
    build_workflow_template_entry,
    create_export_handoff,
    create_subagent_run,
    ensure_local_admin_can_govern,
    evaluate_governance_policies,
    feature_flag_for_subagent_run,
    is_feature_flag_enabled,
    list_workflow_templates,
    recommend_workflow_template_key,
    resolve_handoff_record,
    sync_evaluation_handoff,
    sync_governance_handoff,
    update_feature_flag,
)
from app.services.workspace_access import WorkspaceAccessContext, get_current_workspace_context
from app.services.workspace_bootstrap import apply_workspace_bootstrap, build_workspace_contract


router = APIRouter(prefix="/sessions", tags=["sessions"])


def utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def ensure_route_feature_flag_enabled(db: Session, *, workspace_id: UUID, flag_key: str, detail: str) -> None:
    apply_workspace_bootstrap(db, workspace_id)
    if not is_feature_flag_enabled(db, flag_key, workspace_id=workspace_id):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)


PROJECT_WRITE_ROLES = {WorkspaceRole.owner, WorkspaceRole.admin, WorkspaceRole.editor}
PROJECT_ADMIN_ROLES = {WorkspaceRole.owner, WorkspaceRole.admin}


def _coerce_title_source(value: object) -> ProjectTitleSource:
    if isinstance(value, ProjectTitleSource):
        return value
    try:
        return ProjectTitleSource(str(value))
    except ValueError:
        return ProjectTitleSource.generated


def _session_progress_percent(record: SessionRecord) -> int:
    stage_progress = {
        SessionStage.draft_capture: 14,
        SessionStage.input_validation: 25,
        SessionStage.normalize_discovery: 30,
        SessionStage.build_canvas: 42,
        SessionStage.build_blueprint: 63,
        SessionStage.post_validation: 82,
        SessionStage.ready_for_export: 100,
    }
    if record.status == ArtifactStatus.failed:
        return min(stage_progress.get(record.current_stage, 0), 70)
    if record.status == ArtifactStatus.ready and record.current_stage == SessionStage.ready_for_export:
        return 100
    return stage_progress.get(record.current_stage, 0)


def _session_capabilities(
    *,
    role: WorkspaceRole | None = None,
    record: SessionRecord,
) -> SessionCapabilities:
    is_deleted = record.deleted_at is not None
    is_archived = record.archived_at is not None
    can_write = role in PROJECT_WRITE_ROLES
    can_admin = role in PROJECT_ADMIN_ROLES
    return SessionCapabilities(
        can_open=not is_deleted,
        can_rename=can_write and not is_deleted,
        can_archive=can_admin and not is_archived and not is_deleted,
        can_restore=can_admin and (is_archived or is_deleted),
        can_delete=can_admin and is_archived and not is_deleted,
    )


def build_session_summary(
    record: SessionRecord,
    *,
    owner: UserRecord | None = None,
    pending_attention_count: int = 0,
    role: WorkspaceRole | None = None,
) -> SessionCreateResponse:
    return SessionCreateResponse(
        id=record.id,
        title=record.title,
        suggested_title=record.suggested_title,
        title_source=_coerce_title_source(record.title_source),
        row_version=record.row_version,
        status=record.status,
        current_stage=record.current_stage,
        commercial_tier=record.commercial_tier,
        workspace_id=record.workspace_id,
        owner=SessionOwnerSummary(id=owner.id, name=owner.full_name) if owner is not None else None,
        pending_attention_count=pending_attention_count,
        progress_percent=_session_progress_percent(record),
        archived_at=record.archived_at,
        deleted_at=record.deleted_at,
        capabilities=_session_capabilities(role=role, record=record),
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def ensure_commercial_capability(
    record: SessionRecord,
    capability: str,
    *,
    db: Session | None = None,
    current_user: UserRecord | None = None,
) -> None:
    context = resolve_session_entitlement_context(db, record, current_user) if db is not None and current_user is not None else None
    violation = validate_capability(record.commercial_tier, capability, context=context)
    if violation is None:
        return
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=violation.message)


def build_approval_entry(record: ApprovalGateRecord) -> ApprovalGateEntry:
    return ApprovalGateEntry(
        id=record.id,
        gate_key=record.gate_key,
        title=record.title,
        rationale=record.rationale,
        instructions=record.instructions,
        requested_in_stage=record.requested_in_stage,
        status=record.status,
        resolution_note=record.resolution_note,
        created_at=record.created_at,
        resolved_at=record.resolved_at,
    )


def serialize_blueprint_artifact(artifact: BlueprintArtifact) -> dict:
    payload = artifact.model_dump(mode="json")
    payload["tools"] = [item.model_dump(mode="json") for item in artifact.tools]
    payload["memory_profile"] = artifact.memory_profile.model_dump(mode="json")
    payload["knowledge_profile"] = artifact.knowledge_profile.model_dump(mode="json")
    payload["safety_checks"] = [item.model_dump(mode="json") for item in artifact.safety_checks]
    payload["delivery_package"] = artifact.delivery_package.model_dump(mode="json")
    return payload


def serialize_estimation_artifact(artifact: EstimationReportArtifact) -> dict:
    return artifact.model_dump(mode="json")


def serialize_tool_recommendation_artifact(artifact: ToolRecommendationArtifact) -> dict:
    return artifact.model_dump(mode="json")


def hydrate_blueprint(record: BlueprintRecord) -> BlueprintArtifact:
    return hydrate_blueprint_record(record)


def hydrate_canvas(record: CanvasRecord) -> CanvasArtifact:
    return CanvasArtifact.model_validate(record.model_dump(exclude={"id", "session_id", "updated_at"}))


def hydrate_discovery(record: OpportunityRecord) -> DiscoveryArtifact:
    return DiscoveryArtifact.model_validate(record.model_dump(exclude={"id", "session_id", "updated_at"}))


def build_journey_evidence_manifest_from_trace(
    *,
    trace,
    source_action: str,
) -> list[JourneyArtifactEvidenceEntry]:
    items: list[JourneyArtifactEvidenceEntry] = []
    llm_trace = getattr(trace, "llm_trace", None)
    for index, source in enumerate(getattr(llm_trace, "context_used_sources", []) if llm_trace is not None else []):
        if not isinstance(source, dict):
            continue
        lineages = source.get("source_lineage", [])
        items.append(
            JourneyArtifactEvidenceEntry(
                source_type=str(source.get("kind", "context_source") or "context_source"),
                source_id=str(source.get("uri", "") or source.get("key", "") or f"{source_action}:source:{index + 1}"),
                source_version=str(source.get("source_version", "") or ""),
                source_lineage=[item for item in lineages if isinstance(item, str)],
                authority_level=str(source.get("authority_level", "") or ""),
                used_for=source_action,
                citation_label=str(source.get("title", "") or source.get("key", "") or f"source-{index + 1}"),
                detail=str(source.get("summary", "") or ""),
            )
        )
    for index, evidence in enumerate(getattr(trace, "evidence", [])):
        items.append(
            JourneyArtifactEvidenceEntry(
                source_type=str(getattr(evidence, "source", "runtime_evidence") or "runtime_evidence"),
                source_id=f"{source_action}:evidence:{index + 1}",
                used_for=source_action,
                citation_label=f"evidence-{index + 1}",
                detail=str(getattr(evidence, "detail", "") or ""),
            )
        )
    return items


def build_journey_evidence_manifest_from_traces(
    *,
    traces: list,
    source_action: str,
) -> list[JourneyArtifactEvidenceEntry]:
    items: list[JourneyArtifactEvidenceEntry] = []
    seen: set[tuple[str, str, str]] = set()
    for trace in traces:
        for item in build_journey_evidence_manifest_from_trace(trace=trace, source_action=source_action):
            signature = (item.source_type, item.source_id, item.citation_label)
            if signature in seen:
                continue
            seen.add(signature)
            items.append(item)
    return items


def resolve_discovery_from_stage_artifact(artifact: JourneyStageArtifactEntry) -> DiscoveryArtifact:
    if artifact.schema_version == "discovery-analysis.v1" or "normalized_discovery_candidate" in artifact.proposal_payload:
        analysis = DiscoveryAnalysisOutput.model_validate(artifact.proposal_payload)
        candidate_payload = analysis.normalized_discovery_candidate.model_dump(mode="json")
        patched_candidate = artifact.user_patch.get("normalized_discovery_candidate")
        if isinstance(patched_candidate, dict) and patched_candidate:
            candidate_payload = patched_candidate
        return DiscoveryArtifact.model_validate(candidate_payload)
    return DiscoveryArtifact.model_validate(artifact.proposal_payload)


def resolve_canvas_from_define_stage_artifact(artifact: JourneyStageArtifactEntry) -> CanvasArtifact:
    if artifact.schema_version == "definition-artifact.v1" or "functional_requirements" in artifact.proposal_payload:
        definition = RequirementsDefinitionOutput.model_validate(artifact.proposal_payload)
        return CanvasArtifact.model_validate(definition.canvas_projection.model_dump(mode="json"))
    return CanvasArtifact.model_validate(artifact.proposal_payload)


def resolve_definition_from_stage_artifact(artifact: JourneyStageArtifactEntry) -> RequirementsDefinitionOutput | None:
    if artifact.schema_version == "definition-artifact.v1" or "functional_requirements" in artifact.proposal_payload:
        return RequirementsDefinitionOutput.model_validate(artifact.proposal_payload)
    return None


def resolve_design_from_stage_artifact(artifact: JourneyStageArtifactEntry) -> DesignRecommendationArtifact | None:
    if artifact.schema_version == "design-recommendation.v1" or "alternatives" in artifact.proposal_payload:
        return DesignRecommendationArtifact.model_validate(artifact.proposal_payload)
    return None


def resolve_validation_specification_from_stage_artifact(
    artifact: JourneyStageArtifactEntry | None,
) -> SimulationSpecificationArtifact | None:
    if artifact is None:
        return None
    schema_version = artifact.schema_version or artifact.proposal_payload.get("schema_version", "")
    if schema_version == "validation-simulation-spec.v1":
        return SimulationSpecificationArtifact.model_validate(artifact.proposal_payload)
    return None


def is_discovery_stage_approved(artifact: JourneyStageArtifactEntry | None) -> bool:
    if artifact is None:
        return False
    return artifact.state in {JourneyArtifactState.approved, JourneyArtifactState.approved_legacy}


def is_define_stage_approved(artifact: JourneyStageArtifactEntry | None) -> bool:
    if artifact is None:
        return False
    return artifact.state in {JourneyArtifactState.approved, JourneyArtifactState.approved_legacy}


def is_design_stage_approved(artifact: JourneyStageArtifactEntry | None) -> bool:
    if artifact is None:
        return False
    return artifact.state in {JourneyArtifactState.approved, JourneyArtifactState.approved_legacy}


def is_tools_stage_approved(artifact: JourneyStageArtifactEntry | None) -> bool:
    if artifact is None:
        return False
    return artifact.state in {JourneyArtifactState.approved, JourneyArtifactState.approved_legacy}


def load_latest_approved_stage_artifact_record(
    session: Session,
    *,
    session_id: UUID,
    stage_key: str,
) -> JourneyStageArtifactRecord | None:
    return session.exec(
        select(JourneyStageArtifactRecord)
        .where(
            JourneyStageArtifactRecord.session_id == session_id,
            JourneyStageArtifactRecord.stage_key == stage_key,
            JourneyStageArtifactRecord.state.in_(
                (JourneyArtifactState.approved, JourneyArtifactState.approved_legacy)
            ),
        )
        .order_by(JourneyStageArtifactRecord.version_number.desc(), JourneyStageArtifactRecord.created_at.desc())
    ).first()


def load_latest_stage_artifact_record(
    session: Session,
    *,
    session_id: UUID,
    stage_key: str,
) -> JourneyStageArtifactRecord | None:
    return session.exec(
        select(JourneyStageArtifactRecord)
        .where(
            JourneyStageArtifactRecord.session_id == session_id,
            JourneyStageArtifactRecord.stage_key == stage_key,
        )
        .order_by(JourneyStageArtifactRecord.version_number.desc(), JourneyStageArtifactRecord.created_at.desc())
    ).first()


def get_or_404(session: Session, session_id: UUID, user_id: UUID) -> SessionRecord:
    record = session.exec(select(SessionRecord).where(SessionRecord.id == session_id)).first()
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    accessible_workspace_ids = set(
        session.exec(
            select(WorkspaceMembershipRecord.workspace_id).where(
                WorkspaceMembershipRecord.user_id == user_id,
                WorkspaceMembershipRecord.is_active == True,  # noqa: E712
            )
        ).all()
    )
    if record.workspace_id not in accessible_workspace_ids:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return record


def touch_session(record: SessionRecord, stage: SessionStage, status_value: ArtifactStatus) -> None:
    record.current_stage = stage
    record.status = status_value
    record.updated_at = utc_now()


def maybe_set_session_title(record: SessionRecord, discovery: DiscoveryArtifact) -> None:
    if not discovery.problem_statement:
        return

    suggested_title = " ".join(discovery.problem_statement.strip().split())[:80]
    if not suggested_title:
        return

    record.suggested_title = suggested_title
    if _coerce_title_source(record.title_source) == ProjectTitleSource.manual:
        return

    if record.title != suggested_title:
        record.title = suggested_title
        record.title_source = ProjectTitleSource.generated
        record.row_version += 1


def write_validation(
    session: Session,
    *,
    session_id: UUID,
    artifact_name: str,
    status_value: ArtifactStatus,
    missing_fields: list[str],
    warnings: list[str],
) -> None:
    report = ValidationReportRecord(
        session_id=session_id,
        artifact_name=artifact_name,
        status=status_value,
        missing_fields=missing_fields,
        warnings=warnings,
    )
    session.add(report)


def write_log(
    session: Session,
    *,
    session_id: UUID,
    stage: SessionStage,
    status_value: ArtifactStatus,
    message: str,
    payload: dict,
) -> None:
    log = ExecutionLogRecord(
        session_id=session_id,
        stage=stage,
        status=status_value,
        message=message,
        payload=payload,
    )
    session.add(log)


def create_blueprint_version(
    session: Session,
    *,
    session_id: UUID,
    source_action: str,
    status_value: ArtifactStatus,
    blueprint: BlueprintArtifact,
) -> int:
    latest = session.exec(
        select(BlueprintVersionRecord)
        .where(BlueprintVersionRecord.session_id == session_id)
        .order_by(BlueprintVersionRecord.version_number.desc())
    ).first()
    next_version = (latest.version_number if latest is not None else 0) + 1
    session.add(
        BlueprintVersionRecord(
            session_id=session_id,
            version_number=next_version,
            source_action=source_action,
            status=status_value,
            blueprint_snapshot=serialize_blueprint_artifact(blueprint),
        )
    )
    session.flush()
    return next_version


def latest_blueprint_version_number(session: Session, session_id: UUID) -> int | None:
    latest = session.exec(
        select(BlueprintVersionRecord)
        .where(BlueprintVersionRecord.session_id == session_id)
        .order_by(BlueprintVersionRecord.version_number.desc())
    ).first()
    return latest.version_number if latest is not None else None


def annotate_estimation_report_status(
    estimation_report: EstimationReportArtifact | None,
    *,
    current_blueprint_version_number: int | None,
    current_validation_fingerprint: str,
    current_pricing_catalog_signature: str,
    current_benchmark_corpus_hash: str = "",
) -> EstimationReportArtifact | None:
    if estimation_report is None:
        return None

    stale_reasons = build_estimation_stale_reasons(
        estimation_report,
        current_blueprint_version_number=current_blueprint_version_number,
        current_validation_fingerprint=current_validation_fingerprint,
        current_pricing_catalog_signature=current_pricing_catalog_signature,
        current_benchmark_corpus_hash=current_benchmark_corpus_hash,
    )
    return estimation_report.model_copy(
        update={
            "current_blueprint_version_number": current_blueprint_version_number,
            "is_stale": bool(stale_reasons),
            "stale_reasons": stale_reasons,
        }
    )


def load_latest_persisted_acp_preview(session: Session, session_id: UUID) -> ACPPreview | None:
    record = session.exec(
        select(ArtifactRegistryRecord)
        .where(
            ArtifactRegistryRecord.session_id == session_id,
            ArtifactRegistryRecord.artifact_kind == "acp_preview",
        )
        .order_by(ArtifactRegistryRecord.created_at.desc())
    ).first()
    if record is None or not record.content_text.strip():
        return None
    return ACPPreview.model_validate(json.loads(record.content_text))


def load_latest_persisted_estimation_report(
    session: Session,
    session_id: UUID,
    *,
    current_blueprint_version_number: int | None = None,
    current_validation_fingerprint: str = "",
    current_pricing_catalog_signature: str = "",
    current_benchmark_corpus_hash: str = "",
) -> EstimationReportArtifact | None:
    record = session.exec(
        select(ArtifactRegistryRecord)
        .where(
            ArtifactRegistryRecord.session_id == session_id,
            ArtifactRegistryRecord.artifact_kind == "estimation_report",
        )
        .order_by(ArtifactRegistryRecord.created_at.desc())
    ).first()
    if record is None or not record.content_text.strip():
        return None
    estimation_report = EstimationReportArtifact.model_validate(json.loads(record.content_text))
    return annotate_estimation_report_status(
        estimation_report,
        current_blueprint_version_number=current_blueprint_version_number,
        current_validation_fingerprint=current_validation_fingerprint,
        current_pricing_catalog_signature=current_pricing_catalog_signature,
        current_benchmark_corpus_hash=current_benchmark_corpus_hash,
    )


def load_latest_tool_recommendation(
    session: Session,
    session_id: UUID,
    *,
    discovery: DiscoveryArtifact | None = None,
    canvas: CanvasArtifact | None = None,
    blueprint: BlueprintArtifact | None = None,
    current_blueprint_version_number: int | None = None,
) -> ToolRecommendationArtifact | None:
    record = session.exec(
        select(ArtifactRegistryRecord)
        .where(
            ArtifactRegistryRecord.session_id == session_id,
            ArtifactRegistryRecord.artifact_kind == "tool_recommendation",
        )
        .order_by(ArtifactRegistryRecord.created_at.desc())
    ).first()
    if record is None or not record.content_text.strip():
        return None
    artifact = ToolRecommendationArtifact.model_validate(json.loads(record.content_text))
    definition_artifact = None
    design_artifact = None
    approved_define_record = load_latest_approved_stage_artifact_record(session, session_id=session_id, stage_key="define")
    if approved_define_record is not None:
        definition_artifact = RequirementsDefinitionOutput.model_validate(approved_define_record.proposal_payload)
    approved_design_record = load_latest_approved_stage_artifact_record(session, session_id=session_id, stage_key="design")
    if approved_design_record is not None:
        design_artifact = DesignRecommendationArtifact.model_validate(approved_design_record.proposal_payload)
    annotated = annotate_tool_recommendation_status(
        artifact,
        discovery=discovery,
        canvas=canvas,
        blueprint=blueprint,
        definition_artifact=definition_artifact,
        design_artifact=design_artifact,
        current_blueprint_version=current_blueprint_version_number,
    )
    latest_tools_record = load_latest_stage_artifact_record(session, session_id=session_id, stage_key="tools")
    if latest_tools_record is None:
        return annotated

    merged_stale_reasons = list(
        dict.fromkeys([*annotated.stale_reasons, *(latest_tools_record.stale_reasons or [])])
    )
    if latest_tools_record.state == JourneyArtifactState.stale or merged_stale_reasons:
        return annotated.model_copy(
            update={
                "is_stale": True,
                "stale_reasons": merged_stale_reasons,
            }
        )
    return annotated


def record_tool_recommendation_artifact(
    session: Session,
    *,
    session_id: UUID,
    blueprint_version_number: int | None,
    stage: SessionStage,
    source_action: str,
    recommendation: ToolRecommendationArtifact,
) -> ArtifactRegistryRecord:
    payload = json.dumps(serialize_tool_recommendation_artifact(recommendation), ensure_ascii=False, indent=2)
    content_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    record = ArtifactRegistryRecord(
        session_id=session_id,
        blueprint_version_number=blueprint_version_number,
        artifact_key="tools/tool-recommendation.v1.json",
        artifact_title="Tool recommendation",
        artifact_kind="tool_recommendation",
        stage=stage,
        source_action=source_action,
        export_format="json",
        content_text=payload,
        content_hash=content_hash,
        artifact_metadata={
            "schema_version": recommendation.schema_version,
            "recommended_count": len(recommendation.recommended_tools),
            "optional_count": len(recommendation.optional_tools),
            "rejected_count": len(recommendation.rejected_tools),
            "coverage_gap_count": len(recommendation.coverage_gaps),
            "needs_information_count": len(recommendation.needs_information),
            "confidence_overall": recommendation.confidence.overall,
            "confidence_band": recommendation.confidence.band,
            "review_state": recommendation.review_state,
            "evaluation_overall_status": recommendation.evaluation.overall_status,
            "promotion_blocked": recommendation.evaluation.promotion_blocked,
            "blocking_findings_count": len(
                [item for item in recommendation.evaluation.findings if item.severity == "blocking"]
            ),
            "warning_findings_count": len(
                [item for item in recommendation.evaluation.findings if item.severity == "warning"]
            ),
            "approved_tools_digest_present": recommendation.approved_tools_digest is not None,
            "approved_tools_count": recommendation.approved_tools_digest.tool_count
            if recommendation.approved_tools_digest is not None
            else 0,
            "approved_tools_digest_sha256": recommendation.approved_tools_digest.digest_sha256
            if recommendation.approved_tools_digest is not None
            else "",
        },
    )
    session.add(record)
    session.flush()
    return record


def acp_preview_supports_graph(preview: ACPPreview) -> bool:
    has_graph_json = any(item.path == "ACP/blueprint.graph.json" for item in preview.files)
    has_svg_exports = any(item.path.startswith("ACP/svg/") and item.path.endswith(".svg") for item in preview.files)
    return has_graph_json and has_svg_exports


def resolve_acp_preview(session: Session, record: SessionRecord) -> ACPPreview:
    continuity_answers = build_continuity_answer_map(load_construction_question_response_records(session, record.id))
    if continuity_answers:
        snapshot = build_snapshot(session, record)
        return generate_acp_preview(snapshot, continuity_answers)
    preview = load_latest_persisted_acp_preview(session, record.id)
    if preview is not None and acp_preview_supports_graph(preview):
        return preview
    snapshot = build_snapshot(session, record)
    return generate_acp_preview(snapshot, continuity_answers or None)


def get_acp_file_entry(preview: ACPPreview, file_path: str) -> ACPFileEntry:
    normalized = file_path.strip().replace("\\", "/")
    for item in preview.files:
        if item.path == normalized:
            return item
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ACP file not found")


def get_acp_gap_entry(preview: ACPPreview, gap_key: str) -> ConstructionGapEntry:
    normalized = gap_key.strip()
    for item in preview.construction_readiness.gaps:
        if item.gap_key == normalized:
            return item
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ACP gap not found")


def get_acp_knowledge_graph(preview: ACPPreview) -> BlueprintKnowledgeGraph:
    graph_file = get_acp_file_entry(preview, "ACP/blueprint.graph.json")
    return BlueprintKnowledgeGraph.model_validate(json.loads(graph_file.content_text))


def build_construction_question_views(
    preview: ACPPreview,
    records: list[ConstructionQuestionResponseRecord] | None = None,
) -> list[ConstructionQuestionViewEntry]:
    return build_construction_question_views_with_answers(preview, records or [])


def build_construction_readiness_view(
    preview: ACPPreview,
    records: list[ConstructionQuestionResponseRecord] | None = None,
) -> ConstructionReadinessReport:
    return overlay_construction_readiness(preview, records or [])


def get_construction_question_view(
    preview: ACPPreview,
    question_key: str,
    records: list[ConstructionQuestionResponseRecord] | None = None,
) -> ConstructionQuestionViewEntry:
    for item in build_construction_question_views(preview, records):
        if item.question_key == question_key and item.status != "resolved":
            return item
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ACP question not found")


def upsert_construction_question_response(
    session: Session,
    *,
    session_id: UUID,
    question: ConstructionQuestionViewEntry,
    payload: ConstructionQuestionAnswerRequest,
    current_user: UserRecord,
) -> ConstructionQuestionResponseRecord:
    record = session.exec(
        select(ConstructionQuestionResponseRecord).where(
            ConstructionQuestionResponseRecord.session_id == session_id,
            ConstructionQuestionResponseRecord.question_key == question.question_key,
        )
    ).first()
    now = utc_now()
    if record is None:
        record = ConstructionQuestionResponseRecord(
            session_id=session_id,
            question_key=question.question_key,
            created_at=now,
        )

    record.gap_key = question.gap_key
    record.gap_title = question.gap_title
    record.domain = question.domain
    record.question_text = question.question_text
    record.rationale = question.rationale
    record.expected_answer_format = question.expected_answer_format
    record.target_owner = question.target_owner
    record.blocking = question.blocking
    record.status = "answered"
    record.answer_text = payload.answer_text.strip()
    record.owner_role = payload.owner_role.strip() or question.owner_role or question.target_owner
    record.impacted_artifacts = payload.impacted_artifacts or question.impacted_artifacts
    record.answered_by_user_id = current_user.id
    record.answered_by_display = current_user.full_name or current_user.email
    record.answered_at = now
    record.resolved_at = None
    record.updated_at = now
    session.add(record)
    session.flush()
    return record


def build_safe_download_filename(title: str | None, suffix: str) -> str:
    base_title = title or "agent"
    ascii_title = unicodedata.normalize("NFKD", base_title).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", ascii_title).strip("-").lower()
    if not slug:
        slug = "agent"
    return f"{slug}{suffix}"


def build_skill_catalog_entries(session: Session) -> list[SkillDefinition]:
    rows = session.exec(
        select(SkillCatalogRecord).order_by(SkillCatalogRecord.stage_hint.asc(), SkillCatalogRecord.label.asc())
    ).all()
    if rows:
        return [
            SkillDefinition(
                skill_key=item.skill_key,
                label=item.label,
                stage_hint=item.stage_hint,
                summary=item.summary,
                evidence_policy=item.evidence_policy,
                input_schema=item.input_schema,
                output_schema=item.output_schema,
                is_active=item.is_active,
            )
            for item in rows
        ]
    return [SkillDefinition.model_validate(item) for item in list_skill_definitions(session)]


def build_skill_run_artifact_entry(record: SkillRunArtifactRecord) -> SkillRunArtifact:
    return SkillRunArtifact(
        artifact_role=record.artifact_role,
        artifact_kind=record.artifact_kind,
        payload=record.payload,
    )


def write_skill_runs(
    session: Session,
    *,
    session_id: UUID,
    traces: list,
    source_action: str,
    blueprint_version_number: int | None = None,
) -> list[SkillRunRecord]:
    created_runs: list[SkillRunRecord] = []
    for trace in traces:
        previous_run = session.exec(
            select(SkillRunRecord)
            .where(
                SkillRunRecord.session_id == session_id,
                SkillRunRecord.skill_key == trace.skill_key,
            )
            .order_by(SkillRunRecord.created_at.desc())
        ).first()
        previous_output_payload: dict = {}
        if previous_run is not None:
            previous_output = session.exec(
                select(SkillRunArtifactRecord)
                .where(
                    SkillRunArtifactRecord.skill_run_id == previous_run.id,
                    SkillRunArtifactRecord.artifact_role == "output",
                )
                .order_by(SkillRunArtifactRecord.created_at.desc())
            ).first()
            if previous_output is not None:
                previous_output_payload = previous_output.payload

        record = SkillRunRecord(
            session_id=session_id,
            skill_key=trace.skill_key,
            stage=trace.stage,
            blueprint_version_number=blueprint_version_number,
            source_action=source_action,
            status=trace.status,
            duration_ms=trace.duration_ms,
            result_summary=trace.result_summary,
            warnings=list(trace.warnings),
            evidence=[item.model_dump(mode="json") for item in trace.evidence],
        )
        session.add(record)
        session.flush()
        session.add(
            SkillRunArtifactRecord(
                skill_run_id=record.id,
                artifact_role="input",
                artifact_kind=trace.input_kind,
                payload=trace.input_payload,
            )
        )
        session.add(
            SkillRunArtifactRecord(
                skill_run_id=record.id,
                artifact_role="output",
                artifact_kind=trace.output_kind,
                payload=trace.output_payload,
            )
        )
        session.add(
            SkillRunArtifactRecord(
                skill_run_id=record.id,
                artifact_role="diff",
                artifact_kind=f"{trace.output_kind}.diff",
                payload={
                    "changed": previous_output_payload != trace.output_payload,
                    "before": previous_output_payload,
                    "after": trace.output_payload,
                },
            )
        )
        if trace.llm_trace is not None:
            session.add(
                SkillRunArtifactRecord(
                    skill_run_id=record.id,
                    artifact_role="llm_trace",
                    artifact_kind="llm_trace.v1",
                    payload=trace.llm_trace.model_dump(mode="json"),
                )
            )
        created_runs.append(record)
    return created_runs


def upsert_opportunity(session: Session, session_id: UUID, envelope: DiscoveryEnvelope) -> OpportunityRecord:
    record = session.exec(select(OpportunityRecord).where(OpportunityRecord.session_id == session_id)).first()
    if record is None:
        record = OpportunityRecord(session_id=session_id, **envelope.data.model_dump())
        session.add(record)
        return record

    for field_name, value in envelope.data.model_dump().items():
        setattr(record, field_name, value)
    record.updated_at = utc_now()
    session.add(record)
    return record


def upsert_canvas(session: Session, session_id: UUID, envelope: CanvasEnvelope) -> CanvasRecord:
    payload = envelope.data.model_dump(mode="json")
    record = session.exec(select(CanvasRecord).where(CanvasRecord.session_id == session_id)).first()
    if record is None:
        record = CanvasRecord(session_id=session_id, **payload)
        session.add(record)
        return record

    for field_name, value in payload.items():
        setattr(record, field_name, value)
    record.updated_at = utc_now()
    session.add(record)
    return record


def upsert_blueprint(session: Session, session_id: UUID, envelope: BlueprintEnvelope) -> BlueprintRecord:
    payload = serialize_blueprint_artifact(envelope.data)
    payload.pop("contract_version", None)
    record = session.exec(select(BlueprintRecord).where(BlueprintRecord.session_id == session_id)).first()
    if record is None:
        record = BlueprintRecord(session_id=session_id, **payload)
        session.add(record)
        return record

    for field_name, value in payload.items():
        setattr(record, field_name, value)
    record.updated_at = utc_now()
    session.add(record)
    return record


def upsert_evaluation(session: Session, session_id: UUID, envelope: EvaluationEnvelope) -> EvaluationRecord:
    payload = envelope.data.model_dump(mode="json")
    record = session.exec(select(EvaluationRecord).where(EvaluationRecord.session_id == session_id)).first()
    if record is None:
        record = EvaluationRecord(session_id=session_id, report=payload, status=envelope.status)
        session.add(record)
        return record

    record.report = payload
    record.status = envelope.status
    record.updated_at = utc_now()
    session.add(record)
    return record


def sync_approval_gates(session: Session, session_id: UUID, blueprint: BlueprintArtifact) -> int:
    existing = {
        item.gate_key: item
        for item in session.exec(select(ApprovalGateRecord).where(ApprovalGateRecord.session_id == session_id)).all()
    }
    pending_count = 0

    for tool in blueprint.tools:
        if not (tool.requires_approval or tool.has_side_effects):
            continue
        gate_key = f"tool:{tool.name}"
        title = f"Aprobacion requerida para {tool.name}"
        rationale = tool.approval_reason or tool.purpose
        instructions = (
            "Validar el contrato, las validaciones y la estrategia de retry/compensacion antes de promover "
            f"esta accion. Modo de ejecucion: {tool.execution_mode or 'workflow_controlled'}."
        )
        record = existing.get(gate_key)
        if record is None:
            record = ApprovalGateRecord(
                session_id=session_id,
                gate_key=gate_key,
                title=title,
                rationale=rationale,
                instructions=instructions,
                requested_in_stage=SessionStage.post_validation,
            )
        else:
            record.title = title
            record.rationale = rationale
            record.instructions = instructions
            record.requested_in_stage = SessionStage.post_validation
            record.status = ApprovalStatus.pending
            record.resolution_note = ""
            record.resolved_at = None
        session.add(record)
        pending_count += 1

    return pending_count


def apply_pending_approvals_to_blueprint(
    envelope: BlueprintEnvelope,
    pending_approvals: int,
) -> BlueprintEnvelope:
    if pending_approvals == 0:
        return envelope
    warnings = list(envelope.warnings)
    warning = "Hay approval gates pendientes antes de promover el blueprint a implementacion."
    if warning not in warnings:
        warnings.append(warning)
    return envelope.model_copy(
        update={
            "status": ArtifactStatus.needs_review,
            "warnings": warnings,
            "next_action": "resolve_approvals",
        }
    )


def apply_pending_approvals_to_evaluation(
    envelope: EvaluationEnvelope,
    pending_approvals: int,
) -> EvaluationEnvelope:
    if pending_approvals == 0:
        return envelope
    data = envelope.data.model_copy(
        update={
            "gaps": [*envelope.data.gaps, "Hay approval gates pendientes"],
            "recommendations": [
                *envelope.data.recommendations,
                "Resolver o rechazar los approval gates antes del handoff a implementacion.",
            ],
            "scores": {**envelope.data.scores, "governance": 40},
        }
    )
    warnings = list(envelope.warnings)
    warning = "La evaluacion detecto gates pendientes por resolver."
    if warning not in warnings:
        warnings.append(warning)
    return envelope.model_copy(
        update={
            "status": ArtifactStatus.needs_review,
            "data": data,
            "missing_fields": data.gaps,
            "warnings": warnings,
            "next_action": "resolve_approvals",
        }
    )


def count_pending_approvals(session: Session, session_id: UUID) -> int:
    approvals = session.exec(
        select(ApprovalGateRecord).where(
            ApprovalGateRecord.session_id == session_id,
            ApprovalGateRecord.status == ApprovalStatus.pending,
        )
    ).all()
    return len(approvals)


def apply_pending_approvals_to_run_summary(
    run_summary: EvaluationRunSummary,
    pending_approvals: int,
    *,
    blueprint_version_number: int | None,
) -> EvaluationRunSummary:
    if pending_approvals == 0:
        return run_summary.model_copy(update={"blueprint_version_number": blueprint_version_number})
    return run_summary.model_copy(
        update={
            "blueprint_version_number": blueprint_version_number,
            "status": ArtifactStatus.needs_review if run_summary.status == ArtifactStatus.ready else run_summary.status,
            "overall_score": min(run_summary.overall_score, 80),
            "blocking_issues": [
                *run_summary.blocking_issues,
                "Hay approval gates pendientes",
            ],
            "recommendations": [
                *run_summary.recommendations,
                "Resolver o rechazar los approval gates antes del handoff a implementacion.",
            ],
        }
    )


def build_markdown_export(snapshot: SessionSnapshot) -> str:
    sections = [
        "# Lean Agent Builder Export",
        "",
        "## Session",
        f"- contract_version: `{snapshot.contract_version}`",
        f"- id: `{snapshot.session.id}`",
        f"- status: `{snapshot.session.status}`",
        f"- current_stage: `{snapshot.session.current_stage}`",
        "",
    ]

    sections.extend(
        [
            "## Workspace Contract",
            f"- contract_version: {snapshot.workspace_contract.contract_version}",
            "### Workspace Sections",
        ]
    )
    for item in snapshot.workspace_contract.sections:
        sections.extend(
            [
                f"- {item.label}",
                f"  - key: {item.key}",
                f"  - view_kind: {item.view_kind}",
                f"  - capability_status: {item.capability_status}",
                f"  - read_only: {item.read_only}",
                f"  - source_of_truth: {item.source_of_truth}",
                f"  - summary: {item.summary}",
            ]
        )
    sections.extend(["", "### Feature Flags"])
    for item in snapshot.workspace_contract.feature_flags:
        sections.extend(
            [
                f"- {item.key}",
                f"  - enabled: {item.enabled}",
                f"  - stage_hint: {item.stage_hint}",
                f"  - description: {item.description}",
            ]
        )
    sections.extend(["", "### Catalog Summary"])
    for item in snapshot.workspace_contract.catalogs:
        sections.extend(
            [
                f"- {item.catalog_key}",
                f"  - version: {item.version}",
                f"  - active_count: {item.active_count}/{item.item_count}",
            ]
        )
        sections.extend(
            [
                f"  - {catalog_item.label} ({catalog_item.item_key}) [{catalog_item.status}] -> {catalog_item.summary}"
                for catalog_item in item.items
            ]
        )
    sections.append("")

    if snapshot.discovery:
        sections.extend(
            [
                "## Discovery",
                f"- problem_statement: {snapshot.discovery.problem_statement}",
                f"- current_user: {snapshot.discovery.current_user}",
                f"- current_process: {snapshot.discovery.current_process}",
                f"- desired_outcome: {snapshot.discovery.desired_outcome}",
                f"- autonomy_level: {snapshot.discovery.autonomy_level}",
                f"- case_type: {snapshot.discovery.case_type}",
                f"- value_statement: {snapshot.discovery.value_statement}",
                "",
                "### Operational Baseline",
                f"- current_time_spent: {snapshot.discovery.operational_baseline.current_time_spent}",
                f"- current_cost: {snapshot.discovery.operational_baseline.current_cost}",
                "- frequent_errors:",
                *[f"  - {item}" for item in snapshot.discovery.operational_baseline.frequent_errors],
                "- automation_opportunities:",
                *[f"  - {item}" for item in snapshot.discovery.operational_baseline.automation_opportunities],
                "",
                "### MVP Definition",
                f"- north_star_metric: {snapshot.discovery.mvp_definition.north_star_metric}",
                "- v1_scope:",
                *[f"  - {item}" for item in snapshot.discovery.mvp_definition.v1_scope],
                "- out_of_scope:",
                *[f"  - {item}" for item in snapshot.discovery.mvp_definition.out_of_scope],
                "- non_delegable_decisions:",
                *[f"  - {item}" for item in snapshot.discovery.mvp_definition.non_delegable_decisions],
                "",
            ]
        )

    if snapshot.canvas:
        sections.extend(
            [
                "## Canvas",
                f"- user_goal: {snapshot.canvas.user_goal}",
                f"- success_metric: {snapshot.canvas.success_metric}",
                f"- primary_risk: {snapshot.canvas.primary_risk}",
                "- mvp_scope:",
                *[f"  - {item}" for item in snapshot.canvas.mvp_scope],
                "- out_of_scope:",
                *[f"  - {item}" for item in snapshot.canvas.out_of_scope],
                "",
                "### Agent Profile",
                f"- mission: {snapshot.canvas.agent_profile.mission}",
                f"- primary_user: {snapshot.canvas.agent_profile.primary_user}",
                f"- agent_task: {snapshot.canvas.agent_profile.agent_task}",
                "- allowed_decisions:",
                *[f"  - {item}" for item in snapshot.canvas.agent_profile.allowed_decisions],
                "- prohibited_decisions:",
                *[f"  - {item}" for item in snapshot.canvas.agent_profile.prohibited_decisions],
                "- key_inputs:",
                *[f"  - {item}" for item in snapshot.canvas.agent_profile.key_inputs],
                "- expected_outputs:",
                *[f"  - {item}" for item in snapshot.canvas.agent_profile.expected_outputs],
                "- human_approvals:",
                *[f"  - {item}" for item in snapshot.canvas.agent_profile.human_approvals],
                "- success_metrics:",
                *[f"  - {item}" for item in snapshot.canvas.agent_profile.success_metrics],
                "",
            ]
        )

    if snapshot.blueprint:
        sections.extend(
            [
                "## Blueprint",
                f"- contract_version: {snapshot.blueprint.contract_version}",
                f"- architecture: {snapshot.blueprint.architecture}",
                f"- reasoning_pattern: {snapshot.blueprint.reasoning_pattern}",
                f"- memory_strategy: {snapshot.blueprint.memory_strategy}",
                f"- readiness_state: {snapshot.blueprint.readiness_state}",
                "",
                "### Narrative",
                snapshot.blueprint.narrative,
                "",
                "### Memory Profile",
                f"- strategy: {snapshot.blueprint.memory_profile.strategy}",
                f"- storage_layers: {', '.join(snapshot.blueprint.memory_profile.storage_layers)}",
                f"- write_policy: {snapshot.blueprint.memory_profile.write_policy}",
                f"- retrieval_policy: {snapshot.blueprint.memory_profile.retrieval_policy}",
                f"- review_trigger: {snapshot.blueprint.memory_profile.review_trigger}",
                f"- goal_drift_guard: {snapshot.blueprint.memory_profile.goal_drift_guard}",
                f"- retention_policy: {snapshot.blueprint.memory_profile.retention_policy}",
                f"- ttl_policy: {snapshot.blueprint.memory_profile.ttl_policy}",
                f"- workspace_scope: {snapshot.blueprint.memory_profile.workspace_scope}",
                f"- agent_scope: {snapshot.blueprint.memory_profile.agent_scope}",
                f"- memory_no_evidence_behavior: {snapshot.blueprint.memory_profile.grounding_policy.no_evidence_behavior}",
                "",
                "### Knowledge Profile",
                f"- mode: {snapshot.blueprint.knowledge_profile.mode}",
                f"- source_count: {len(snapshot.blueprint.knowledge_profile.sources)}",
                f"- source_lineage: {', '.join(f'{item.title}:{item.source_version}' for item in snapshot.blueprint.knowledge_profile.sources)}",
                f"- refresh_frequency: {snapshot.blueprint.knowledge_profile.refresh_policy.frequency}",
                f"- retrieval_fallback: {snapshot.blueprint.knowledge_profile.retrieval_policy.fallback_behavior}",
                f"- knowledge_no_evidence_behavior: {snapshot.blueprint.knowledge_profile.grounding_policy.no_evidence_behavior}",
                "",
                "### Tools",
            ]
        )
        for tool in snapshot.blueprint.tools:
            sections.extend(
                [
                    f"- {tool.name}",
                    f"  - purpose: {tool.purpose}",
                    f"  - risk_level: {tool.risk_level}",
                    f"  - requires_approval: {tool.requires_approval}",
                    f"  - has_side_effects: {tool.has_side_effects}",
                    f"  - execution_mode: {tool.execution_mode}",
                    f"  - inputs: {', '.join(tool.inputs)}",
                    f"  - outputs: {', '.join(tool.outputs)}",
                    f"  - validations: {', '.join(tool.validations)}",
                    f"  - retry_strategy: {tool.retry_strategy}",
                    f"  - compensation_strategy: {tool.compensation_strategy}",
                    f"  - failure_mode: {tool.failure_mode}",
                ]
            )
        sections.extend(["", "### Safety Checks"])
        for item in snapshot.blueprint.safety_checks:
            sections.extend(
                [
                    f"- {item.category}",
                    f"  - risk: {item.risk}",
                    f"  - severity: {item.severity}",
                    f"  - mitigation: {item.mitigation}",
                    f"  - status: {item.status}",
                ]
            )
        sections.extend(["", "### Guardrails", *[f"- {item}" for item in snapshot.blueprint.guardrails], ""])
        sections.extend(
            [
                "### Decision Summary",
                snapshot.blueprint.delivery_package.decision_summary,
                "",
                "### Decision Trace",
            ]
        )
        for item in snapshot.blueprint.delivery_package.decision_trace:
            sections.extend(
                [
                    f"- {item.dimension}",
                    f"  - selected: {item.selected_label} ({item.selected_value})",
                    f"  - recommended: {item.recommended_label} ({item.recommended_value})",
                    f"  - source: {item.decision_source}",
                    f"  - rationale: {item.rationale}",
                    f"  - review_note: {item.review_note}",
                    f"  - evidence: {', '.join(item.evidence)}",
                ]
            )
        sections.extend(["", "### Pattern Catalog"])
        for item in snapshot.blueprint.delivery_package.pattern_catalog:
            sections.extend(
                [
                    f"- {item.family}:{item.label}",
                    f"  - key: {item.key}",
                    f"  - fit_score: {item.fit_score}",
                    f"  - selected: {item.selected}",
                    f"  - summary: {item.summary}",
                ]
            )
        sections.extend(
            [
                "### Workflow Profile",
                f"- contract_version: {snapshot.blueprint.delivery_package.contract_version}",
                f"- execution_pattern: {snapshot.blueprint.delivery_package.workflow_profile.execution_pattern}",
                f"- inbox_strategy: {snapshot.blueprint.delivery_package.workflow_profile.inbox_strategy}",
                f"- outbox_strategy: {snapshot.blueprint.delivery_package.workflow_profile.outbox_strategy}",
                f"- checkpoint_policy: {snapshot.blueprint.delivery_package.workflow_profile.checkpoint_policy}",
                f"- retry_strategy: {snapshot.blueprint.delivery_package.workflow_profile.retry_strategy}",
                f"- compensation_strategy: {snapshot.blueprint.delivery_package.workflow_profile.compensation_strategy}",
                f"- approval_pause: {snapshot.blueprint.delivery_package.workflow_profile.approval_pause}",
                f"- timeout_policy: {snapshot.blueprint.delivery_package.workflow_profile.timeout_policy}",
                "- steps:",
            ]
        )
        for step in snapshot.blueprint.delivery_package.workflow_profile.steps:
            sections.extend(
                [
                    f"  - {step.name}",
                    f"    - actor: {step.actor}",
                    f"    - objective: {step.objective}",
                    f"    - outputs: {', '.join(step.outputs)}",
                    f"    - fallback: {step.fallback}",
                    f"    - requires_approval: {step.requires_approval}",
                ]
            )
        sections.extend(
            [
                "",
                "### Observability Plan",
                f"- captured_signals: {', '.join(snapshot.blueprint.delivery_package.observability_plan.captured_signals)}",
                f"- plan_summary_policy: {snapshot.blueprint.delivery_package.observability_plan.plan_summary_policy}",
                f"- tool_response_logging: {snapshot.blueprint.delivery_package.observability_plan.tool_response_logging}",
                f"- decision_logging: {snapshot.blueprint.delivery_package.observability_plan.decision_logging}",
                f"- cost_tracking: {snapshot.blueprint.delivery_package.observability_plan.cost_tracking}",
                f"- duration_tracking: {snapshot.blueprint.delivery_package.observability_plan.duration_tracking}",
                f"- result_tracking: {snapshot.blueprint.delivery_package.observability_plan.result_tracking}",
                "- alert_triggers:",
                *[f"  - {item}" for item in snapshot.blueprint.delivery_package.observability_plan.alert_triggers],
                "",
                "### Blueprint Coverage",
                f"- overall_status: {snapshot.blueprint.delivery_package.blueprint_coverage.overall_status}",
                f"- covered_sections: {snapshot.blueprint.delivery_package.blueprint_coverage.covered_sections}/{snapshot.blueprint.delivery_package.blueprint_coverage.total_sections}",
            ]
        )
        for item in snapshot.blueprint.delivery_package.blueprint_coverage.sections:
            sections.extend(
                [
                    f"- {item.title}",
                    f"  - key: {item.key}",
                    f"  - status: {item.status}",
                    f"  - source: {item.source}",
                    f"  - note: {item.note}",
                ]
            )
        sections.extend(
            [
                "",
                "### Roadmap Evolution",
                f"- current_release: {snapshot.blueprint.delivery_package.roadmap_evolution.current_release}",
                f"- current_focus: {snapshot.blueprint.delivery_package.roadmap_evolution.current_focus}",
            ]
        )
        for item in snapshot.blueprint.delivery_package.roadmap_evolution.milestones:
            sections.extend(
                [
                    f"- {item.release}: {item.title}",
                    f"  - objective: {item.objective}",
                    f"  - when_to_unlock: {item.when_to_unlock}",
                    "  - capabilities:",
                    *[f"    - {capability}" for capability in item.capabilities],
                ]
            )
        sections.extend(
            [
                "",
                "### Component Readiness",
            ]
        )
        for item in snapshot.blueprint.delivery_package.component_readiness:
            sections.extend(
                [
                    f"- {item.label}",
                    f"  - status: {item.status}",
                    f"  - score: {item.score}",
                    f"  - completed_checks: {item.completed_checks}/{item.total_checks}",
                    "  - checks:",
                    *[f"    - {check.title} [{check.status}] -> {check.detail}" for check in item.checks],
                ]
            )
        sections.extend(
            [
                "",
                "### Risk Summary",
                f"- overall_status: {snapshot.blueprint.delivery_package.risk_summary.overall_status}",
                f"- high_risks: {snapshot.blueprint.delivery_package.risk_summary.high_risks}",
                f"- medium_risks: {snapshot.blueprint.delivery_package.risk_summary.medium_risks}",
                f"- low_risks: {snapshot.blueprint.delivery_package.risk_summary.low_risks}",
                f"- approval_gates_required: {snapshot.blueprint.delivery_package.risk_summary.approval_gates_required}",
                f"- side_effect_tools: {snapshot.blueprint.delivery_package.risk_summary.side_effect_tools}",
                f"- summary: {snapshot.blueprint.delivery_package.risk_summary.summary}",
                "",
                "### Deliverables",
            ]
        )
        for item in snapshot.blueprint.delivery_package.deliverables:
            sections.extend(
                [
                    f"#### {item.title}",
                    f"- key: {item.key}",
                    f"- summary: {item.summary}",
                    "",
                    item.content_markdown,
                    "",
                ]
            )

    if snapshot.approvals:
        sections.extend(["## Approval Gates", ""])
        for item in snapshot.approvals:
            sections.extend(
                [
                    f"- {item.title}",
                    f"  - status: {item.status}",
                    f"  - rationale: {item.rationale}",
                    f"  - instructions: {item.instructions}",
                    f"  - resolution_note: {item.resolution_note or 'pending'}",
                ]
            )
        sections.append("")

    if snapshot.evaluation:
        sections.extend(
            [
                "## Evaluation",
                f"- completeness_status: {snapshot.evaluation.completeness_status}",
                f"- coherence_status: {snapshot.evaluation.coherence_status}",
                "### Scores",
                *[f"- {key}: {value}" for key, value in snapshot.evaluation.scores.items()],
                "### Cases",
                *[
                    f"- {item.name} [{item.category}] -> {item.expected_result}"
                    for item in snapshot.evaluation.cases
                ],
                "- gaps:",
                *[f"  - {item}" for item in snapshot.evaluation.gaps],
                "- recommendations:",
                *[f"  - {item}" for item in snapshot.evaluation.recommendations],
                "",
            ]
        )

    if snapshot.evaluation_dataset or snapshot.evaluation_rubric or snapshot.evaluation_runs:
        sections.extend(["## Evaluation Workbench", ""])
        if snapshot.evaluation_dataset:
            sections.extend(
                [
                    "### Dataset activo",
                    f"- version_number: {snapshot.evaluation_dataset.version_number}",
                    f"- blueprint_version_number: {snapshot.evaluation_dataset.blueprint_version_number}",
                    f"- source_action: {snapshot.evaluation_dataset.source_action}",
                    f"- status: {snapshot.evaluation_dataset.status}",
                    f"- summary: {snapshot.evaluation_dataset.summary}",
                    "- cases:",
                ]
            )
            for item in snapshot.evaluation_dataset.cases:
                sections.extend(
                    [
                        f"  - {item.title} ({item.case_key})",
                        f"    - category: {item.category}",
                        f"    - source: {item.source}",
                        f"    - priority: {item.priority}",
                        f"    - is_active: {item.is_active}",
                        f"    - scenario: {item.scenario}",
                        f"    - expected_result: {item.expected_result}",
                    ]
                )
            sections.append("")
        if snapshot.evaluation_rubric:
            sections.extend(
                [
                    "### Rubrica activa",
                    f"- version_number: {snapshot.evaluation_rubric.version_number}",
                    f"- blueprint_version_number: {snapshot.evaluation_rubric.blueprint_version_number}",
                    f"- source_action: {snapshot.evaluation_rubric.source_action}",
                    f"- summary: {snapshot.evaluation_rubric.summary}",
                ]
            )
            for item in snapshot.evaluation_rubric.dimensions:
                sections.extend(
                    [
                        f"- {item.label} ({item.key})",
                        f"  - weight: {item.weight}",
                        f"  - hard_block: {item.hard_block}",
                        f"  - description: {item.description}",
                    ]
                )
            sections.append("")
        if snapshot.evaluation_runs:
            sections.extend(["### Corridas persistidas"])
            for item in snapshot.evaluation_runs:
                sections.extend(
                    [
                        f"- run {item.id}",
                        f"  - status: {item.status}",
                        f"  - overall_score: {item.overall_score}",
                        f"  - source_action: {item.source_action}",
                        f"  - dataset_version_number: {item.dataset_version_number}",
                        f"  - rubric_version_number: {item.rubric_version_number}",
                        f"  - summary: {item.summary}",
                    ]
                )
                if item.blocking_issues:
                    sections.extend(["  - blocking_issues:", *[f"    - {issue}" for issue in item.blocking_issues]])
                if item.recommendations:
                    sections.extend(["  - recommendations:", *[f"    - {issue}" for issue in item.recommendations]])
                for result in item.results:
                    sections.extend(
                        [
                            f"  - case: {result.title} ({result.case_key})",
                            f"    - category: {result.category}",
                            f"    - status: {result.status}",
                            f"    - score: {result.score}",
                            f"    - summary: {result.summary}",
                            f"    - observed_result: {result.observed_result}",
                        ]
                    )
                sections.append("")

    if snapshot.skill_catalog or snapshot.skill_runs:
        sections.extend(["## Skill Runtime", ""])
        if snapshot.skill_catalog:
            sections.extend(["### Skill Catalog"])
            for item in snapshot.skill_catalog:
                sections.extend(
                    [
                        f"- {item.label}",
                        f"  - key: {item.skill_key}",
                        f"  - stage_hint: {item.stage_hint}",
                        f"  - evidence_policy: {item.evidence_policy}",
                        f"  - is_active: {item.is_active}",
                        f"  - summary: {item.summary}",
                    ]
                )
            sections.append("")
        if snapshot.skill_runs:
            sections.extend(["### Skill Runs"])
            for item in snapshot.skill_runs:
                sections.extend(
                    [
                        f"- {item.label}",
                        f"  - key: {item.skill_key}",
                        f"  - stage: {item.stage}",
                        f"  - status: {item.status}",
                        f"  - source_action: {item.source_action}",
                        f"  - blueprint_version_number: {item.blueprint_version_number}",
                        f"  - duration_ms: {item.duration_ms}",
                        f"  - result_summary: {item.result_summary}",
                    ]
                )
                if item.warnings:
                    sections.extend(["  - warnings:", *[f"    - {warning}" for warning in item.warnings]])
                if item.evidence:
                    sections.extend(
                        [
                            "  - evidence:",
                            *[f"    - {evidence.source}: {evidence.detail}" for evidence in item.evidence],
                        ]
                    )
                for artifact in item.artifacts:
                    if artifact.artifact_role != "diff":
                        continue
                    sections.append(f"  - diff_changed: {artifact.payload.get('changed', False)}")
                sections.append("")

    if snapshot.metric_snapshots or snapshot.alert_events or snapshot.integration_statuses or snapshot.artifact_records:
        sections.extend(["## Operational Modules", ""])
        if snapshot.metric_snapshots:
            latest_metric = snapshot.metric_snapshots[0]
            sections.extend(
                [
                    "### Monitoring",
                    f"- source_action: {latest_metric.source_action}",
                    f"- cost_estimate_usd: {latest_metric.cost_estimate_usd}",
                    f"- total_duration_ms: {latest_metric.total_duration_ms}",
                    f"- error_count: {latest_metric.error_count}",
                    f"- warning_count: {latest_metric.warning_count}",
                    f"- approvals_pending: {latest_metric.approvals_pending}",
                    f"- regenerations_count: {latest_metric.regenerations_count}",
                    f"- needs_review_count: {latest_metric.needs_review_count}",
                    f"- latest_evaluation_score: {latest_metric.latest_evaluation_score}",
                    f"- export_count: {latest_metric.export_count}",
                    f"- artifact_count: {latest_metric.artifact_count}",
                    "",
                ]
            )
        if snapshot.alert_events:
            sections.extend(["### Alerts"])
            for item in snapshot.alert_events:
                sections.extend(
                    [
                        f"- {item.title}",
                        f"  - key: {item.alert_key}",
                        f"  - severity: {item.severity}",
                        f"  - status: {item.status}",
                        f"  - message: {item.message}",
                    ]
                )
            sections.append("")
        if snapshot.integration_statuses:
            sections.extend(["### Integrations"])
            for item in snapshot.integration_statuses:
                sections.extend(
                    [
                        f"- {item.label}",
                        f"  - status: {item.status}",
                        f"  - configured: {item.configured}",
                        f"  - reachable: {item.reachable}",
                        f"  - detail: {item.detail}",
                    ]
                )
            sections.append("")
        if snapshot.artifact_records:
            sections.extend(["### Artifact Browser"])
            for item in snapshot.artifact_records[:20]:
                sections.extend(
                    [
                        f"- {item.artifact_title}",
                        f"  - key: {item.artifact_key}",
                        f"  - kind: {item.artifact_kind}",
                        f"  - blueprint_version_number: {item.blueprint_version_number}",
                        f"  - source_action: {item.source_action}",
                        f"  - created_at: {item.created_at}",
                    ]
                )
            sections.append("")

    if snapshot.workflow_templates or snapshot.handoff_records or snapshot.governance_policies or snapshot.subagent_runs:
        sections.extend(["## MVP 3 Governance", ""])
        if snapshot.workflow_templates:
            sections.extend(
                [
                    "### Workflow Templates",
                    f"- selected_workflow_template_key: {snapshot.selected_workflow_template_key or 'none'}",
                ]
            )
            for item in snapshot.workflow_templates:
                sections.extend(
                    [
                        f"- {item.label}",
                        f"  - key: {item.template_key}",
                        f"  - architecture_scope: {', '.join(item.architecture_scope)}",
                        f"  - supports_approvals: {item.supports_approvals}",
                        f"  - supports_handoffs: {item.supports_handoffs}",
                        f"  - summary: {item.summary}",
                    ]
                )
            sections.append("")
        if snapshot.handoff_records:
            sections.extend(["### Handoffs"])
            for item in snapshot.handoff_records:
                sections.extend(
                    [
                        f"- {item.title}",
                        f"  - key: {item.handoff_key}",
                        f"  - status: {item.status}",
                        f"  - from_stage: {item.from_stage}",
                        f"  - to_stage: {item.to_stage}",
                        f"  - owner_role: {item.owner_role}",
                        f"  - triggered_by: {item.triggered_by}",
                        f"  - resolution_note: {item.resolution_note or 'pending'}",
                    ]
                )
            sections.append("")
        if snapshot.governance_policies:
            sections.extend(["### Governance Policies"])
            for item in snapshot.governance_policies:
                sections.extend(
                    [
                        f"- {item.label}",
                        f"  - key: {item.policy_key}",
                        f"  - scope: {item.scope}",
                        f"  - compliance_status: {item.compliance_status}",
                        f"  - summary: {item.summary}",
                    ]
                )
                if item.evidence:
                    sections.extend(["  - evidence:", *[f"    - {evidence}" for evidence in item.evidence]])
            sections.append("")
        if snapshot.subagent_runs:
            sections.extend(["### Specialized Subprocesses"])
            for item in snapshot.subagent_runs:
                sections.extend(
                    [
                        f"- {item.title}",
                        f"  - run_kind: {item.run_kind}",
                        f"  - status: {item.status}",
                        f"  - feature_flag_key: {item.feature_flag_key}",
                        f"  - summary: {item.summary}",
                    ]
                )
            sections.append("")

    if snapshot.blueprint_versions:
        sections.extend(
            [
                "## Blueprint Versions",
                *[
                    f"- v{item.version_number} [{item.source_action}] {item.status} / {item.readiness_state}"
                    for item in snapshot.blueprint_versions
                ],
                "",
            ]
        )

    consistency_report = ensure_blueprint_consistency_report(snapshot)
    sections.extend([render_blueprint_consistency_markdown(consistency_report), ""])
    if consistency_report.decision_history:
        sections.extend(["## Journey Decisions"])
        for item in consistency_report.decision_history:
            sections.extend(
                [
                    (
                        f"- {item['stage_key']} v{item['version_number']} "
                        f"[{item['source_action']}] {item['state']}"
                    ),
                    f"  - citations: {', '.join(item.get('citation_labels', [])) or 'none'}",
                ]
            )
            for decision in item.get("decisions", [])[:4]:
                sections.append(
                    "  - "
                    f"{decision['decision_type']} -> {decision.get('next_state') or 'n/a'}: "
                    f"{decision.get('note') or 'sin nota'}"
                )
        sections.append("")

    return "\n".join(sections)


def build_json_export(snapshot: SessionSnapshot) -> dict:
    payload = snapshot.model_dump(mode="json")
    payload["generated_at"] = utc_now().isoformat()
    return payload


def build_canonical_export_response(
    *,
    session_id: UUID,
    contract_key: CanonicalExportKind,
    preview: bool,
    db: Session,
    current_user: UserRecord,
) -> JSONResponse:
    record = get_or_404(db, session_id, current_user.id)
    capability_by_contract = {
        "blueprint-core.v1": "export_blueprint_core",
        "construction-pack.v1": "export_construction_pack",
        "agent-construction-package.v2": "export_construction_pack",
        "prompt-pack.v1": "export_prompt_pack",
        "estimation-pack.v1": "export_estimation_pack",
        "test-pack.v1": "export_test_pack",
    }

    if not preview:
        ensure_commercial_capability(record, capability_by_contract[contract_key], db=db, current_user=current_user)

    snapshot = build_snapshot(db, record)
    document = build_canonical_export_document(snapshot, contract_key=contract_key)

    if should_block_canonical_export(document, preview=preview):
        blocking_reasons = ", ".join(document.blocking_reasons) or "critical contract readiness gaps"
        write_log(
            db,
            session_id=session_id,
            stage=record.current_stage,
            status_value=ArtifactStatus.failed,
            message="Export canonico bloqueado por conformance",
            payload={
                "blocking_reasons": document.blocking_reasons,
                "contract_key": contract_key,
                "preview": preview,
                "readiness": document.readiness,
                "source_action": f"export_{contract_key}",
            },
        )
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Canonical export blocked until readiness is resolved: {blocking_reasons}",
        )

    if not preview:
        artifact_key, artifact_title = CANONICAL_EXPORT_ARTIFACTS[contract_key]
        export_record = record_export_artifact(
            db,
            session_id=session_id,
            blueprint_version_number=latest_blueprint_version_number(db, session_id),
            artifact_key=artifact_key,
            artifact_title=artifact_title,
            export_format="json",
            content_text=document.payload_text,
            source_action=f"export_{contract_key}",
            artifact_metadata_extra={
                "canonical_checksum_sha256": document.checksum_sha256,
                "contract_key": contract_key,
                "preview": preview,
                "readiness": document.readiness,
            },
        )
        create_export_handoff(
            db,
            session_record=record,
            blueprint_version_number=export_record.blueprint_version_number,
            source_action=f"export_{contract_key}",
            artifact_key=export_record.artifact_key,
        )
        write_log(
            db,
            session_id=session_id,
            stage=record.current_stage,
            status_value=record.status,
            message="Export canonico generado",
            payload={
                "artifact_key": export_record.artifact_key,
                "canonical_checksum_sha256": document.checksum_sha256,
                "contract_key": contract_key,
                "preview": preview,
                "readiness": document.readiness,
                "source_action": f"export_{contract_key}",
            },
        )
        capture_operational_state(db, session_id=session_id, source_action=f"export_{contract_key}")
        db.commit()

    return JSONResponse(
        content=document.payload,
        headers=build_canonical_export_headers(document, preview=preview),
    )


def build_skill_run_entry(
    record: SkillRunRecord,
    *,
    label: str,
    artifacts: list[SkillRunArtifact],
) -> SkillRunEntry:
    llm_trace_payload = next(
        (item.payload for item in artifacts if item.artifact_role == "llm_trace" and item.payload),
        None,
    )
    return SkillRunEntry(
        id=record.id,
        skill_key=record.skill_key,
        label=label,
        stage=record.stage,
        blueprint_version_number=record.blueprint_version_number,
        source_action=record.source_action,
        status=record.status,
        duration_ms=record.duration_ms,
        result_summary=record.result_summary,
        warnings=record.warnings,
        evidence=[EvidenceItem.model_validate(item) for item in record.evidence],
        llm_trace=LLMContextTrace.model_validate(llm_trace_payload) if isinstance(llm_trace_payload, dict) else None,
        artifacts=[item for item in artifacts if item.artifact_role != "llm_trace"],
        created_at=record.created_at,
    )


def build_evaluation_dataset_case_entry(record: EvaluationCaseRecord) -> EvaluationDatasetCase:
    return EvaluationDatasetCase(
        id=record.id,
        case_key=record.case_key,
        title=record.title,
        category=record.category,
        scenario=record.scenario,
        expected_result=record.expected_result,
        source=record.source,
        priority=record.priority,
        is_active=record.is_active,
    )


def build_evaluation_dataset_artifact(
    record: EvaluationDatasetRecord,
    case_records: list[EvaluationCaseRecord],
) -> EvaluationDatasetArtifact:
    return EvaluationDatasetArtifact(
        id=record.id,
        version_number=record.version_number,
        blueprint_version_number=record.blueprint_version_number,
        source_action=record.source_action,
        status=record.status,
        summary=record.summary,
        cases=[build_evaluation_dataset_case_entry(item) for item in case_records],
    )


def build_evaluation_rubric_artifact(record: EvaluationRubricRecord) -> EvaluationRubricArtifact:
    return EvaluationRubricArtifact(
        id=record.id,
        version_number=record.version_number,
        blueprint_version_number=record.blueprint_version_number,
        source_action=record.source_action,
        summary=record.summary,
        dimensions=record.dimensions,
    )


def build_evaluation_run_entry(
    record: EvaluationRunRecord,
    result_records: list[EvaluationResultRecord],
    dataset_version_number: int,
    rubric_version_number: int,
) -> EvaluationRunEntry:
    return EvaluationRunEntry(
        id=record.id,
        dataset_version_number=dataset_version_number,
        rubric_version_number=rubric_version_number,
        blueprint_version_number=record.blueprint_version_number,
        source_action=record.source_action,
        status=record.status,
        overall_score=record.overall_score,
        summary=record.summary,
        category_scores={key: int(value) for key, value in record.category_scores.items()},
        dimension_scores={key: int(value) for key, value in record.dimension_scores.items()},
        blocking_issues=record.blocking_issues,
        recommendations=record.recommendations,
        results=[
            EvaluationCaseResult(
                case_key=item.case_key,
                title=item.title,
                category=item.category,
                status=item.status,
                score=item.score,
                summary=item.summary,
                observed_result=item.observed_result,
                evidence=item.evidence,
                blocking_issues=item.blocking_issues,
                recommendations=item.recommendations,
            )
            for item in result_records
        ],
        created_at=record.created_at,
    )


def latest_evaluation_dataset(session: Session, session_id: UUID) -> EvaluationDatasetRecord | None:
    return session.exec(
        select(EvaluationDatasetRecord)
        .where(EvaluationDatasetRecord.session_id == session_id)
        .order_by(EvaluationDatasetRecord.version_number.desc())
    ).first()


def latest_evaluation_rubric(session: Session, session_id: UUID) -> EvaluationRubricRecord | None:
    return session.exec(
        select(EvaluationRubricRecord)
        .where(EvaluationRubricRecord.session_id == session_id)
        .order_by(EvaluationRubricRecord.version_number.desc())
    ).first()


def create_evaluation_dataset_version(
    session: Session,
    *,
    session_id: UUID,
    blueprint_version_number: int | None,
    source_action: str,
    dataset: EvaluationDatasetArtifact,
) -> EvaluationDatasetRecord:
    latest = latest_evaluation_dataset(session, session_id)
    next_version = (latest.version_number if latest is not None else 0) + 1
    record = EvaluationDatasetRecord(
        session_id=session_id,
        blueprint_version_number=blueprint_version_number,
        version_number=next_version,
        source_action=source_action,
        status=dataset.status,
        summary=dataset.summary,
    )
    session.add(record)
    session.flush()
    for index, item in enumerate(dataset.cases, start=1):
        session.add(
            EvaluationCaseRecord(
                dataset_id=record.id,
                case_key=item.case_key,
                title=item.title,
                category=item.category,
                scenario=item.scenario,
                expected_result=item.expected_result,
                source=item.source,
                priority=item.priority,
                sort_order=index,
                is_active=item.is_active,
            )
        )
    session.flush()
    return record


def create_evaluation_rubric_version(
    session: Session,
    *,
    session_id: UUID,
    blueprint_version_number: int | None,
    source_action: str,
    rubric: EvaluationRubricArtifact,
) -> EvaluationRubricRecord:
    latest = latest_evaluation_rubric(session, session_id)
    next_version = (latest.version_number if latest is not None else 0) + 1
    record = EvaluationRubricRecord(
        session_id=session_id,
        blueprint_version_number=blueprint_version_number,
        version_number=next_version,
        source_action=source_action,
        summary=rubric.summary,
        dimensions=[item.model_dump(mode="json") for item in rubric.dimensions],
    )
    session.add(record)
    session.flush()
    return record


def hydrate_evaluation_dataset(session: Session, record: EvaluationDatasetRecord) -> EvaluationDatasetArtifact:
    cases = session.exec(
        select(EvaluationCaseRecord)
        .where(EvaluationCaseRecord.dataset_id == record.id)
        .order_by(EvaluationCaseRecord.sort_order.asc(), EvaluationCaseRecord.created_at.asc())
    ).all()
    return build_evaluation_dataset_artifact(record, cases)


def hydrate_evaluation_rubric(record: EvaluationRubricRecord) -> EvaluationRubricArtifact:
    return build_evaluation_rubric_artifact(record)


def ensure_evaluation_workbench_assets(
    session: Session,
    *,
    session_id: UUID,
    discovery: DiscoveryArtifact | None,
    canvas: CanvasArtifact | None,
    blueprint: BlueprintArtifact | None,
    blueprint_version_number: int | None,
) -> tuple[EvaluationDatasetArtifact, EvaluationRubricArtifact]:
    dataset_record = latest_evaluation_dataset(session, session_id)
    rubric_record = latest_evaluation_rubric(session, session_id)

    if dataset_record is None:
        approved_validate_record = session.exec(
            select(JourneyStageArtifactRecord)
            .where(
                JourneyStageArtifactRecord.session_id == session_id,
                JourneyStageArtifactRecord.stage_key == "validate",
                JourneyStageArtifactRecord.state.in_(
                    [JourneyArtifactState.approved, JourneyArtifactState.approved_legacy]
                ),
            )
            .order_by(
                JourneyStageArtifactRecord.version_number.desc(),
                JourneyStageArtifactRecord.created_at.desc(),
            )
        ).first()
        specification = None
        if approved_validate_record is not None:
            schema_version = (
                approved_validate_record.schema_version
                or approved_validate_record.proposal_payload.get("schema_version", "")
            )
            if schema_version == "validation-simulation-spec.v1":
                specification = SimulationSpecificationArtifact.model_validate(
                    approved_validate_record.proposal_payload
                )
        dataset_record = create_evaluation_dataset_version(
            session,
            session_id=session_id,
            blueprint_version_number=blueprint_version_number,
            source_action="bootstrap_dataset",
            dataset=(
                build_evaluation_dataset_from_simulation_specification(
                    specification,
                    blueprint_version_number=blueprint_version_number,
                    source_action="bootstrap_dataset",
                )
                if specification is not None and specification.scenarios
                else build_default_evaluation_dataset(
                    discovery,
                    canvas,
                    blueprint,
                    blueprint_version_number=blueprint_version_number,
                    source_action="bootstrap_dataset",
                )
            ),
        )
    if rubric_record is None:
        rubric_record = create_evaluation_rubric_version(
            session,
            session_id=session_id,
            blueprint_version_number=blueprint_version_number,
            source_action="bootstrap_rubric",
            rubric=build_default_evaluation_rubric(
                blueprint_version_number=blueprint_version_number,
                source_action="bootstrap_rubric",
            ),
        )

    return hydrate_evaluation_dataset(session, dataset_record), hydrate_evaluation_rubric(rubric_record)


def persist_evaluation_run(
    session: Session,
    *,
    session_id: UUID,
    dataset_record: EvaluationDatasetRecord,
    rubric_record: EvaluationRubricRecord,
    run_summary: EvaluationRunSummary,
) -> EvaluationRunRecord:
    run_record = EvaluationRunRecord(
        session_id=session_id,
        dataset_id=dataset_record.id,
        rubric_id=rubric_record.id,
        blueprint_version_number=run_summary.blueprint_version_number,
        source_action=run_summary.source_action,
        status=run_summary.status,
        overall_score=run_summary.overall_score,
        summary=run_summary.summary,
        category_scores=run_summary.category_scores,
        dimension_scores=run_summary.dimension_scores,
        blocking_issues=run_summary.blocking_issues,
        recommendations=run_summary.recommendations,
    )
    session.add(run_record)
    session.flush()

    case_by_key = {
        item.case_key: item
        for item in session.exec(select(EvaluationCaseRecord).where(EvaluationCaseRecord.dataset_id == dataset_record.id)).all()
    }
    for item in run_summary.results:
        session.add(
            EvaluationResultRecord(
                run_id=run_record.id,
                case_id=case_by_key.get(item.case_key).id if case_by_key.get(item.case_key) is not None else None,
                case_key=item.case_key,
                title=item.title,
                category=item.category,
                status=item.status,
                score=item.score,
                summary=item.summary,
                observed_result=item.observed_result,
                evidence=item.evidence,
                blocking_issues=item.blocking_issues,
                recommendations=item.recommendations,
            )
        )
    session.flush()
    return run_record


def build_evaluation_dataset_from_simulation_specification(
    specification: SimulationSpecificationArtifact,
    *,
    blueprint_version_number: int | None,
    source_action: str,
) -> EvaluationDatasetArtifact:
    cases = []
    for index, scenario in enumerate(specification.scenarios, start=1):
        cases.append(
            EvaluationDatasetCase(
                case_key=scenario.scenario_key or f"simulation-case-{index}",
                title=scenario.title or f"Escenario {index}",
                category="simulation",
                scenario=" ".join(
                    [
                        scenario.objective,
                        *(scenario.state_transitions[:2]),
                    ]
                ).strip(),
                expected_result=scenario.expected_outcome or "El escenario concluye en estado controlado.",
                source="validation_simulation",
                priority="core" if scenario.priority == "high" else "extended",
                is_active=True,
            )
        )
    return EvaluationDatasetArtifact(
        blueprint_version_number=blueprint_version_number,
        source_action=source_action,
        status=ArtifactStatus.draft,
        summary="Dataset derivado desde escenarios aprobados de Validate.",
        cases=cases,
    )


def persist_validation_simulation_run(
    session: Session,
    *,
    session_id: UUID,
    run_record: SimulationRunRecord,
) -> ValidationSimulationRunStateRecord:
    db_record = ValidationSimulationRunStateRecord(
        id=run_record.id,
        session_id=session_id,
        specification_artifact_id=run_record.specification_artifact_id,
        blueprint_version_number=run_record.blueprint_version_number,
        scenario_key=run_record.scenario_key,
        scenario_title=run_record.scenario_title,
        scenario_version_number=run_record.scenario_version_number,
        source_action=run_record.source_action,
        status=run_record.status,
        execution_state=run_record.execution_state,
        hard_gate_status=run_record.hard_gate_status,
        final_status=run_record.final_status,
        active_node_key=run_record.active_node_key,
        summary=run_record.summary,
        injected_conditions=list(run_record.injected_conditions),
        deterministic_signature=run_record.deterministic_signature,
        events=[item.model_dump(mode="json") for item in run_record.events],
        judgement=run_record.judgement.model_dump(mode="json") if run_record.judgement is not None else {},
        created_at=run_record.created_at,
        updated_at=run_record.updated_at,
    )
    session.add(db_record)
    session.flush()
    return db_record


def build_validation_simulation_run_entry(
    record: ValidationSimulationRunStateRecord,
    *,
    latest_blueprint_version_number: int | None,
    latest_validate_artifact_id: UUID | None,
) -> SimulationRunRecord:
    stale_reasons: list[str] = []
    if latest_blueprint_version_number is not None and record.blueprint_version_number != latest_blueprint_version_number:
        stale_reasons.append("El blueprint aprobado cambio despues de esta corrida de simulacion.")
    if latest_validate_artifact_id is not None and record.specification_artifact_id != latest_validate_artifact_id:
        stale_reasons.append("La especificacion aprobada de Validate cambio despues de esta corrida.")
    judgement_payload = record.judgement if isinstance(record.judgement, dict) else {}
    return SimulationRunRecord(
        id=record.id,
        specification_artifact_id=record.specification_artifact_id,
        blueprint_version_number=record.blueprint_version_number,
        scenario_key=record.scenario_key,
        scenario_title=record.scenario_title,
        scenario_version_number=record.scenario_version_number,
        source_action=record.source_action,
        status=record.status,
        execution_state=record.execution_state,
        hard_gate_status=record.hard_gate_status,
        final_status=record.final_status,
        active_node_key=record.active_node_key,
        summary=record.summary,
        injected_conditions=list(record.injected_conditions),
        deterministic_signature=record.deterministic_signature,
        is_stale=bool(stale_reasons),
        stale_reasons=stale_reasons,
        events=[SimulationEvent.model_validate(item) for item in (record.events or [])],
        judgement=SimulationJudgement.model_validate(judgement_payload) if judgement_payload else None,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def build_snapshot(
    session: Session,
    record: SessionRecord,
    *,
    include_short_term: bool = True,
    current_user: UserRecord | None = None,
) -> SessionSnapshot:
    JourneyStageMigrationService().backfill_session(session, session_record=record)
    proposal_service = StageProposalService()
    opportunity = session.exec(select(OpportunityRecord).where(OpportunityRecord.session_id == record.id)).first()
    canvas = session.exec(select(CanvasRecord).where(CanvasRecord.session_id == record.id)).first()
    blueprint = session.exec(select(BlueprintRecord).where(BlueprintRecord.session_id == record.id)).first()
    evaluation = session.exec(select(EvaluationRecord).where(EvaluationRecord.session_id == record.id)).first()
    validations = session.exec(
        select(ValidationReportRecord)
        .where(ValidationReportRecord.session_id == record.id)
        .order_by(ValidationReportRecord.created_at.desc())
    ).all()
    activity = session.exec(
        select(ExecutionLogRecord)
        .where(ExecutionLogRecord.session_id == record.id)
        .order_by(ExecutionLogRecord.created_at.desc())
    ).all()
    versions = session.exec(
        select(BlueprintVersionRecord)
        .where(BlueprintVersionRecord.session_id == record.id)
        .order_by(BlueprintVersionRecord.version_number.desc())
    ).all()
    approvals = session.exec(
        select(ApprovalGateRecord)
        .where(ApprovalGateRecord.session_id == record.id)
        .order_by(ApprovalGateRecord.created_at.desc())
    ).all()
    skill_catalog = build_skill_catalog_entries(session)
    skill_run_records = session.exec(
        select(SkillRunRecord)
        .where(SkillRunRecord.session_id == record.id)
        .order_by(SkillRunRecord.created_at.desc())
    ).all()
    skill_artifacts_by_run: dict[UUID, list[SkillRunArtifact]] = {}
    if skill_run_records:
        skill_artifact_records = session.exec(
            select(SkillRunArtifactRecord)
            .where(SkillRunArtifactRecord.skill_run_id.in_([item.id for item in skill_run_records]))
            .order_by(SkillRunArtifactRecord.created_at.asc())
        ).all()
        for artifact in skill_artifact_records:
            skill_artifacts_by_run.setdefault(artifact.skill_run_id, []).append(build_skill_run_artifact_entry(artifact))
    skill_labels = {item.skill_key: item.label for item in skill_catalog}
    evaluation_dataset_record = latest_evaluation_dataset(session, record.id)
    evaluation_rubric_record = latest_evaluation_rubric(session, record.id)
    evaluation_dataset = (
        hydrate_evaluation_dataset(session, evaluation_dataset_record)
        if evaluation_dataset_record is not None
        else None
    )
    evaluation_rubric = (
        hydrate_evaluation_rubric(evaluation_rubric_record)
        if evaluation_rubric_record is not None
        else None
    )
    evaluation_run_records = session.exec(
        select(EvaluationRunRecord)
        .where(EvaluationRunRecord.session_id == record.id)
        .order_by(EvaluationRunRecord.created_at.desc())
    ).all()
    evaluation_result_records_by_run: dict[UUID, list[EvaluationResultRecord]] = {}
    if evaluation_run_records:
        evaluation_result_records = session.exec(
            select(EvaluationResultRecord)
            .where(EvaluationResultRecord.run_id.in_([item.id for item in evaluation_run_records]))
            .order_by(EvaluationResultRecord.created_at.asc())
        ).all()
        for item in evaluation_result_records:
            evaluation_result_records_by_run.setdefault(item.run_id, []).append(item)
    dataset_version_by_id = {}
    rubric_version_by_id = {}
    if evaluation_run_records:
        dataset_ids = {item.dataset_id for item in evaluation_run_records}
        rubric_ids = {item.rubric_id for item in evaluation_run_records}
        for item in session.exec(select(EvaluationDatasetRecord).where(EvaluationDatasetRecord.id.in_(dataset_ids))).all():
            dataset_version_by_id[item.id] = item.version_number
        for item in session.exec(select(EvaluationRubricRecord).where(EvaluationRubricRecord.id.in_(rubric_ids))).all():
            rubric_version_by_id[item.id] = item.version_number
    artifact_records = session.exec(
        select(ArtifactRegistryRecord)
        .where(ArtifactRegistryRecord.session_id == record.id)
        .order_by(ArtifactRegistryRecord.created_at.desc())
    ).all()
    metric_snapshots = session.exec(
        select(MetricSnapshotRecord)
        .where(MetricSnapshotRecord.session_id == record.id)
        .order_by(MetricSnapshotRecord.created_at.desc())
    ).all()
    alert_events = session.exec(
        select(AlertEventRecord)
        .where(AlertEventRecord.session_id == record.id)
        .order_by(AlertEventRecord.updated_at.desc(), AlertEventRecord.created_at.desc())
    ).all()
    integration_statuses = session.exec(
        select(IntegrationStatusRecord)
        .where(IntegrationStatusRecord.session_id == record.id)
        .order_by(IntegrationStatusRecord.checked_at.desc())
    ).all()
    workflow_templates = session.exec(
        select(WorkflowTemplateRecord)
        .where(
            WorkflowTemplateRecord.workspace_id == record.workspace_id,
            WorkflowTemplateRecord.is_active == True,  # noqa: E712
        )
        .order_by(WorkflowTemplateRecord.label.asc())
    ).all()
    handoff_records = session.exec(
        select(HandoffRecord)
        .where(HandoffRecord.session_id == record.id)
        .order_by(HandoffRecord.updated_at.desc(), HandoffRecord.created_at.desc())
    ).all()
    subagent_runs = session.exec(
        select(SubagentRunRecord)
        .where(SubagentRunRecord.session_id == record.id)
        .order_by(SubagentRunRecord.created_at.desc())
    ).all()
    journey_artifacts = proposal_service.list_all(session, session_record=record)
    latest_journey_artifacts = proposal_service.latest_by_stage(session, session_record=record)
    latest_validate_artifact = latest_journey_artifacts.get("validate")
    simulation_run_records = session.exec(
        select(ValidationSimulationRunStateRecord)
        .where(ValidationSimulationRunStateRecord.session_id == record.id)
        .order_by(ValidationSimulationRunStateRecord.updated_at.desc(), ValidationSimulationRunStateRecord.created_at.desc())
    ).all()
    hydrated_discovery = hydrate_discovery(opportunity) if opportunity is not None else None
    hydrated_canvas = hydrate_canvas(canvas) if canvas is not None else None
    hydrated_blueprint = hydrate_blueprint(blueprint) if blueprint is not None else None
    latest_evaluation_run = evaluation_run_records[0] if evaluation_run_records else None
    governance_policies = evaluate_governance_policies(
        session,
        session_record=record,
        blueprint=hydrated_blueprint,
        approvals=approvals,
        latest_evaluation_score=latest_evaluation_run.overall_score if latest_evaluation_run is not None else None,
        latest_evaluation_status=latest_evaluation_run.status if latest_evaluation_run is not None else "",
    )
    selected_workflow_template_key = record.selected_workflow_template_key or recommend_workflow_template_key(
        hydrated_blueprint
    )
    estimation_runs = list_estimation_runs(session, record.id)
    project_actuals = list_project_actuals(session, record.id)
    estimation_error_metrics = list_estimation_error_metrics(session, record.id)
    current_blueprint_version_number = versions[0].version_number if versions else None
    current_validation_fingerprint = build_estimation_validation_fingerprint_from_state(
        latest_validate_artifact=latest_validate_artifact,
        evaluation_runs=[
            build_evaluation_run_entry(
                item,
                evaluation_result_records_by_run.get(item.id, []),
                dataset_version_by_id.get(item.dataset_id, 0),
                rubric_version_by_id.get(item.rubric_id, 0),
            )
            for item in evaluation_run_records
        ],
        simulation_runs=[
            build_validation_simulation_run_entry(
                item,
                latest_blueprint_version_number=current_blueprint_version_number,
                latest_validate_artifact_id=latest_validate_artifact.id if latest_validate_artifact is not None else None,
            )
            for item in simulation_run_records
        ],
    )
    current_pricing_catalog_signature = build_estimation_pricing_catalog_signature(session)
    current_benchmark_corpus_hash = build_estimation_benchmark_corpus_hash(
        session,
        workspace_id=record.workspace_id,
        session_id=record.id,
    )

    current_membership = (
        get_workspace_membership_for_record(session, record=record, user_id=current_user.id) if current_user is not None else None
    )
    owner = session.exec(select(UserRecord).where(UserRecord.id == record.user_id)).first()

    snapshot = SessionSnapshot(
        session=build_session_summary(
            record,
            owner=owner,
            pending_attention_count=len([item for item in approvals if item.status == ApprovalStatus.pending]),
            role=current_membership.role if current_membership is not None else None,
        ),
        commercial_access=resolve_session_commercial_access(session, record, current_user=current_user),
        discovery=hydrated_discovery,
        canvas=hydrated_canvas,
        blueprint=hydrated_blueprint,
        latest_tool_recommendation=load_latest_tool_recommendation(
            session,
            record.id,
            discovery=hydrated_discovery,
            canvas=hydrated_canvas,
            blueprint=hydrated_blueprint,
            current_blueprint_version_number=current_blueprint_version_number,
        ),
        evaluation=evaluation and EvaluationArtifact.model_validate(evaluation.report),
        estimation_report=load_latest_persisted_estimation_report(
            session,
            record.id,
            current_blueprint_version_number=current_blueprint_version_number,
            current_validation_fingerprint=current_validation_fingerprint,
            current_pricing_catalog_signature=current_pricing_catalog_signature,
            current_benchmark_corpus_hash=current_benchmark_corpus_hash,
        ),
        estimation_runs=[build_estimation_run_entry(item) for item in estimation_runs],
        project_actuals=[build_project_actuals_entry(item) for item in project_actuals],
        estimation_error_metrics=[build_estimation_error_metric_entry(item) for item in estimation_error_metrics],
        evaluation_dataset=evaluation_dataset,
        evaluation_rubric=evaluation_rubric,
        evaluation_runs=[
            build_evaluation_run_entry(
                item,
                evaluation_result_records_by_run.get(item.id, []),
                dataset_version_by_id.get(item.dataset_id, 0),
                rubric_version_by_id.get(item.rubric_id, 0),
            )
            for item in evaluation_run_records
        ],
        simulation_runs=[
            build_validation_simulation_run_entry(
                item,
                latest_blueprint_version_number=current_blueprint_version_number,
                latest_validate_artifact_id=latest_validate_artifact.id if latest_validate_artifact is not None else None,
            )
            for item in simulation_run_records
        ],
        validations=[
            {
                "artifact_name": item.artifact_name,
                "status": item.status,
                "missing_fields": item.missing_fields,
                "warnings": item.warnings,
                "created_at": item.created_at,
            }
            for item in validations
        ],
        activity=[
            ExecutionLogEntry(
                stage=item.stage,
                status=item.status,
                message=item.message,
                payload=item.payload,
                created_at=item.created_at,
            )
            for item in activity
        ],
        blueprint_versions=[
            BlueprintVersionEntry(
                version_number=item.version_number,
                source_action=item.source_action,
                status=item.status,
                readiness_state=BlueprintArtifact.model_validate(item.blueprint_snapshot).readiness_state,
                architecture=item.blueprint_snapshot.get("architecture", ""),
                reasoning_pattern=item.blueprint_snapshot.get("reasoning_pattern", ""),
                created_at=item.created_at,
            )
            for item in versions
        ],
        selected_workflow_template_key=selected_workflow_template_key,
        approvals=[build_approval_entry(item) for item in approvals],
        journey_artifacts=journey_artifacts,
        journey_latest_artifacts=latest_journey_artifacts,
        artifact_records=[build_artifact_record_entry(item) for item in artifact_records],
        metric_snapshots=[build_metric_snapshot_entry(item) for item in metric_snapshots],
        alert_events=[build_alert_event_entry(item) for item in alert_events],
        integration_statuses=[build_integration_status_entry(item) for item in integration_statuses],
        workflow_templates=[build_workflow_template_entry(item) for item in workflow_templates],
        handoff_records=[build_handoff_record_entry(item) for item in handoff_records],
        governance_policies=governance_policies,
        subagent_runs=[build_subagent_run_entry(item) for item in subagent_runs],
        workspace_contract=build_workspace_contract(session, record.workspace_id),
        skill_catalog=skill_catalog,
        skill_runs=[
            build_skill_run_entry(
                item,
                label=skill_labels.get(item.skill_key, item.skill_key),
                artifacts=skill_artifacts_by_run.get(item.id, []),
            )
            for item in skill_run_records
        ],
    )
    if include_short_term:
        snapshot.short_term_memory = ShortTermMemoryService().resume_session_state(
            session,
            session_id=record.id,
            snapshot=snapshot,
            source_action="load_session_snapshot",
        )
    return snapshot.model_copy(
        update={
            "blueprint_consistency": ensure_blueprint_consistency_report(snapshot),
        }
    )


def sync_short_term_memory_checkpoint(
    session: Session,
    *,
    record: SessionRecord,
    source_action: str,
    branch_key: str = MAIN_BRANCH_KEY,
) -> ShortTermMemoryRuntimeState:
    snapshot = build_snapshot(session, record, include_short_term=False)
    return ShortTermMemoryService().capture_session_state(
        session,
        session_id=record.id,
        snapshot=snapshot,
        source_action=source_action,
        branch_key=branch_key,
    )


def build_stage_context_bundle(
    session: Session,
    *,
    record: SessionRecord,
    capability: str,
    stage: str,
    effective_language: str = "es",
    role: str = "builder",
    task_source_keys: list[str] | None = None,
    allow_second_page: bool = False,
) -> StageContextBundle:
    snapshot = build_snapshot(session, record)
    return StageContextService().build(
        session,
        workspace_id=record.workspace_id,
        session_id=record.id,
        session_snapshot=snapshot,
        capability=capability,
        effective_language=effective_language,
        role=role,
        stage=stage,
        task_source_keys=task_source_keys,
        allow_second_page=allow_second_page,
    )


@router.get("", response_model=SessionListResponse)
def list_sessions(
    q: str | None = Query(default=None),
    status_filter: ArtifactStatus | None = Query(default=None, alias="status"),
    tier: str | None = Query(default=None),
    lifecycle: str = Query(default="active"),
    sort: str = Query(default="updated_desc"),
    cursor: str | None = Query(default=None),
    limit: int = Query(default=25, ge=1, le=100),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> SessionListResponse:
    offset = 0
    if cursor:
        try:
            offset = max(0, int(cursor))
        except ValueError:
            offset = 0

    filters = [SessionRecord.workspace_id == workspace_context.workspace.id]
    lifecycle_key = lifecycle if lifecycle in {"active", "archived", "trash", "all"} else "active"
    if lifecycle_key == "active":
        filters.extend([SessionRecord.archived_at.is_(None), SessionRecord.deleted_at.is_(None)])
    elif lifecycle_key == "archived":
        filters.extend([SessionRecord.archived_at.is_not(None), SessionRecord.deleted_at.is_(None)])
    elif lifecycle_key == "trash":
        filters.append(SessionRecord.deleted_at.is_not(None))
    else:
        filters.append(SessionRecord.deleted_at.is_(None))

    normalized_q = " ".join((q or "").strip().split())
    if normalized_q:
        pattern = f"%{normalized_q}%"
        filters.append(or_(SessionRecord.title.ilike(pattern), SessionRecord.suggested_title.ilike(pattern)))

    if status_filter is not None:
        filters.append(SessionRecord.status == status_filter)

    if tier:
        filters.append(SessionRecord.commercial_tier == tier)

    order_by = {
        "created_desc": SessionRecord.created_at.desc(),
        "title_asc": SessionRecord.title.asc(),
        "title_desc": SessionRecord.title.desc(),
        "updated_asc": SessionRecord.updated_at.asc(),
        "updated_desc": SessionRecord.updated_at.desc(),
    }.get(sort, SessionRecord.updated_at.desc())

    total = db.exec(select(func.count()).select_from(SessionRecord).where(*filters)).one()
    records = db.exec(
        select(SessionRecord)
        .where(*filters)
        .order_by(order_by, SessionRecord.id.asc())
        .offset(offset)
        .limit(limit)
    ).all()

    owner_ids = {item.user_id for item in records}
    owners = db.exec(select(UserRecord).where(UserRecord.id.in_(owner_ids))).all() if owner_ids else []
    owner_by_id = {item.id: item for item in owners}
    record_ids = [item.id for item in records]
    pending_rows = (
        db.exec(
            select(ApprovalGateRecord.session_id, func.count())
            .where(
                ApprovalGateRecord.session_id.in_(record_ids),
                ApprovalGateRecord.status == ApprovalStatus.pending,
            )
            .group_by(ApprovalGateRecord.session_id)
        ).all()
        if record_ids
        else []
    )
    pending_by_session_id = {session_id: count for session_id, count in pending_rows}

    workspace_filter = SessionRecord.workspace_id == workspace_context.workspace.id
    active_count = db.exec(
        select(func.count()).select_from(SessionRecord).where(
            workspace_filter,
            SessionRecord.archived_at.is_(None),
            SessionRecord.deleted_at.is_(None),
        )
    ).one()
    needs_review_count = db.exec(
        select(func.count()).select_from(SessionRecord).where(
            workspace_filter,
            SessionRecord.archived_at.is_(None),
            SessionRecord.deleted_at.is_(None),
            SessionRecord.status == ArtifactStatus.needs_review,
        )
    ).one()
    archived_count = db.exec(
        select(func.count()).select_from(SessionRecord).where(
            workspace_filter,
            SessionRecord.archived_at.is_not(None),
            SessionRecord.deleted_at.is_(None),
        )
    ).one()
    trash_count = db.exec(
        select(func.count()).select_from(SessionRecord).where(
            workspace_filter,
            SessionRecord.deleted_at.is_not(None),
        )
    ).one()

    next_offset = offset + len(records)
    next_cursor = str(next_offset) if next_offset < total else None
    return SessionListResponse(
        items=[
            build_session_summary(
                item,
                owner=owner_by_id.get(item.user_id),
                pending_attention_count=int(pending_by_session_id.get(item.id, 0)),
                role=workspace_context.membership.role,
            )
            for item in records
        ],
        page=SessionListPageInfo(next_cursor=next_cursor, total=int(total)),
        facets=SessionListFacets(
            active=int(active_count),
            needs_review=int(needs_review_count),
            archived=int(archived_count),
            trash=int(trash_count),
        ),
    )


@router.post("", response_model=SessionCreateResponse, status_code=status.HTTP_201_CREATED)
def create_session(
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> SessionCreateResponse:
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    record = SessionRecord(user_id=current_user.id, workspace_id=workspace_context.workspace.id)
    db.add(record)
    db.flush()
    capture_operational_state(db, session_id=record.id, source_action="create_session")
    sync_short_term_memory_checkpoint(db, record=record, source_action="create_session")
    db.commit()
    db.refresh(record)
    return build_session_summary(record, owner=current_user, role=workspace_context.membership.role)


def get_workspace_membership_for_record(
    session: Session,
    *,
    record: SessionRecord,
    user_id: UUID,
) -> WorkspaceMembershipRecord:
    membership = session.exec(
        select(WorkspaceMembershipRecord).where(
            WorkspaceMembershipRecord.workspace_id == record.workspace_id,
            WorkspaceMembershipRecord.user_id == user_id,
            WorkspaceMembershipRecord.is_active == True,  # noqa: E712
        )
    ).first()
    if membership is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return membership


def ensure_project_role(membership: WorkspaceMembershipRecord, allowed_roles: set[WorkspaceRole], action: str) -> None:
    if membership.role not in allowed_roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"No tienes permiso para {action} este proyecto.")


def touch_project_record(record: SessionRecord) -> None:
    record.row_version += 1
    record.updated_at = utc_now()


@router.patch("/{session_id}", response_model=SessionCreateResponse)
def rename_session(
    session_id: UUID,
    payload: SessionRenameRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> SessionCreateResponse:
    record = get_or_404(db, session_id, current_user.id)
    membership = get_workspace_membership_for_record(db, record=record, user_id=current_user.id)
    ensure_project_role(membership, PROJECT_WRITE_ROLES, "renombrar")
    if record.deleted_at is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="No se puede renombrar un proyecto en papelera.")
    if payload.expected_version is not None and payload.expected_version != record.row_version:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="El proyecto cambio desde que abriste esta vista.")

    before_title = record.title
    record.title = payload.title
    record.title_source = ProjectTitleSource.manual
    touch_project_record(record)
    write_log(
        db,
        session_id=session_id,
        stage=record.current_stage,
        status_value=record.status,
        message="Proyecto renombrado",
        payload={
            "action": "project_rename",
            "before_title": before_title,
            "after_title": record.title,
            "actor_user_id": str(current_user.id),
            "row_version": record.row_version,
        },
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return build_session_summary(record, owner=current_user, role=membership.role)


@router.post("/{session_id}/archive", response_model=SessionCreateResponse)
def archive_session(
    session_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> SessionCreateResponse:
    record = get_or_404(db, session_id, current_user.id)
    membership = get_workspace_membership_for_record(db, record=record, user_id=current_user.id)
    ensure_project_role(membership, PROJECT_ADMIN_ROLES, "archivar")
    if record.deleted_at is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="No se puede archivar un proyecto en papelera.")
    if record.archived_at is None:
        record.archived_at = utc_now()
        record.archived_by_user_id = current_user.id
        touch_project_record(record)
        write_log(
            db,
            session_id=session_id,
            stage=record.current_stage,
            status_value=record.status,
            message="Proyecto archivado",
            payload={"action": "project_archive", "actor_user_id": str(current_user.id)},
        )
        db.add(record)
        db.commit()
        db.refresh(record)
    return build_session_summary(record, owner=current_user, role=membership.role)


@router.post("/{session_id}/restore", response_model=SessionCreateResponse)
def restore_session(
    session_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> SessionCreateResponse:
    record = get_or_404(db, session_id, current_user.id)
    membership = get_workspace_membership_for_record(db, record=record, user_id=current_user.id)
    ensure_project_role(membership, PROJECT_ADMIN_ROLES, "restaurar")
    if record.archived_at is not None or record.deleted_at is not None:
        record.archived_at = None
        record.archived_by_user_id = None
        record.deleted_at = None
        record.deleted_by_user_id = None
        touch_project_record(record)
        write_log(
            db,
            session_id=session_id,
            stage=record.current_stage,
            status_value=record.status,
            message="Proyecto restaurado",
            payload={"action": "project_restore", "actor_user_id": str(current_user.id)},
        )
        db.add(record)
        db.commit()
        db.refresh(record)
    return build_session_summary(record, owner=current_user, role=membership.role)


@router.delete("/{session_id}", response_model=SessionCreateResponse)
def delete_session(
    session_id: UUID,
    payload: SessionDeleteRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> SessionCreateResponse:
    record = get_or_404(db, session_id, current_user.id)
    membership = get_workspace_membership_for_record(db, record=record, user_id=current_user.id)
    ensure_project_role(membership, PROJECT_ADMIN_ROLES, "mover a papelera")
    if record.deleted_at is not None:
        return build_session_summary(record, owner=current_user, role=membership.role)
    if record.archived_at is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Archiva el proyecto antes de moverlo a papelera.")
    if payload.confirm_title != record.title:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="La confirmacion no coincide con el nombre del proyecto.")

    record.deleted_at = utc_now()
    record.deleted_by_user_id = current_user.id
    touch_project_record(record)
    write_log(
        db,
        session_id=session_id,
        stage=record.current_stage,
        status_value=record.status,
        message="Proyecto enviado a papelera",
        payload={"action": "project_delete", "actor_user_id": str(current_user.id)},
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return build_session_summary(record, owner=current_user, role=membership.role)


@router.get("/{session_id}", response_model=SessionSnapshot)
def get_session_snapshot(
    session_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> SessionSnapshot:
    record = get_or_404(db, session_id, current_user.id)
    apply_workspace_bootstrap(db, record.workspace_id)
    has_metrics = db.exec(
        select(MetricSnapshotRecord).where(MetricSnapshotRecord.session_id == session_id)
    ).first()
    has_integrations = db.exec(
        select(IntegrationStatusRecord).where(IntegrationStatusRecord.session_id == session_id)
    ).first()
    if has_metrics is None or has_integrations is None:
        capture_operational_state(db, session_id=session_id, source_action="load_session_snapshot")
        db.commit()
    return build_snapshot(db, record, current_user=current_user)


def _raise_stage_proposal_http_error(exc: Exception) -> None:
    if isinstance(exc, StageProposalNotFoundError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    if isinstance(exc, StageProposalConflictError):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/{session_id}/journey/{stage_key}/artifacts", response_model=JourneyStageArtifactEntry)
def create_stage_artifact_route(
    session_id: UUID,
    stage_key: str,
    payload: JourneyStageArtifactCreateRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> JourneyStageArtifactEntry:
    record = get_or_404(db, session_id, current_user.id)
    service = StageProposalService()
    try:
        artifact = service.create(
            db,
            session_record=record,
            stage_key=stage_key,
            payload=payload,
            actor_user=current_user,
        )
    except Exception as exc:  # noqa: BLE001
        _raise_stage_proposal_http_error(exc)
    capture_operational_state(db, session_id=session_id, source_action=f"journey_create:{stage_key}")
    db.commit()
    return artifact


@router.get("/{session_id}/journey/{stage_key}/artifacts", response_model=JourneyStageArtifactListResponse)
def list_stage_artifacts_route(
    session_id: UUID,
    stage_key: str,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> JourneyStageArtifactListResponse:
    record = get_or_404(db, session_id, current_user.id)
    service = StageProposalService()
    try:
        return service.list(db, session_record=record, stage_key=stage_key)
    except Exception as exc:  # noqa: BLE001
        _raise_stage_proposal_http_error(exc)


@router.get("/{session_id}/journey/{stage_key}/artifacts/latest", response_model=JourneyStageArtifactEntry | None)
def get_latest_stage_artifact_route(
    session_id: UUID,
    stage_key: str,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> JourneyStageArtifactEntry | None:
    record = get_or_404(db, session_id, current_user.id)
    service = StageProposalService()
    try:
        return service.latest(db, session_record=record, stage_key=stage_key)
    except Exception as exc:  # noqa: BLE001
        _raise_stage_proposal_http_error(exc)


@router.patch("/{session_id}/journey/{stage_key}/artifacts/{artifact_id}", response_model=JourneyStageArtifactEntry)
def patch_stage_artifact_route(
    session_id: UUID,
    stage_key: str,
    artifact_id: UUID,
    payload: JourneyStageArtifactPatchRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> JourneyStageArtifactEntry:
    record = get_or_404(db, session_id, current_user.id)
    service = StageProposalService()
    try:
        artifact = service.patch(
            db,
            session_record=record,
            stage_key=stage_key,
            artifact_id=artifact_id,
            payload=payload,
            actor_user=current_user,
        )
    except Exception as exc:  # noqa: BLE001
        _raise_stage_proposal_http_error(exc)
    capture_operational_state(db, session_id=session_id, source_action=f"journey_patch:{stage_key}")
    db.commit()
    return artifact


@router.post(
    "/{session_id}/journey/{stage_key}/artifacts/{artifact_id}/approve",
    response_model=JourneyStageArtifactEntry,
)
def approve_stage_artifact_route(
    session_id: UUID,
    stage_key: str,
    artifact_id: UUID,
    payload: JourneyStageArtifactApprovalRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> JourneyStageArtifactEntry:
    record = get_or_404(db, session_id, current_user.id)
    service = StageProposalService()
    try:
        artifact = service.approve(
            db,
            session_record=record,
            stage_key=stage_key,
            artifact_id=artifact_id,
            payload=payload,
            actor_user=current_user,
        )
    except Exception as exc:  # noqa: BLE001
        _raise_stage_proposal_http_error(exc)
    sync_short_term_memory_checkpoint(db, record=record, source_action=f"journey_approve:{stage_key}")
    capture_operational_state(db, session_id=session_id, source_action=f"journey_approve:{stage_key}")
    db.commit()
    return artifact


@router.post(
    "/{session_id}/journey/{stage_key}/artifacts/{artifact_id}/reject",
    response_model=JourneyStageArtifactEntry,
)
def reject_stage_artifact_route(
    session_id: UUID,
    stage_key: str,
    artifact_id: UUID,
    payload: JourneyStageArtifactRejectionRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> JourneyStageArtifactEntry:
    record = get_or_404(db, session_id, current_user.id)
    service = StageProposalService()
    try:
        artifact = service.reject(
            db,
            session_record=record,
            stage_key=stage_key,
            artifact_id=artifact_id,
            payload=payload,
            actor_user=current_user,
        )
    except Exception as exc:  # noqa: BLE001
        _raise_stage_proposal_http_error(exc)
    capture_operational_state(db, session_id=session_id, source_action=f"journey_reject:{stage_key}")
    db.commit()
    return artifact


@router.post("/{session_id}/tools/bind-api", response_model=BlueprintTool)
def bind_tool_api_route(
    session_id: UUID,
    payload: ToolApiBindRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> BlueprintTool:
    record = get_or_404(db, session_id, current_user.id)
    blueprint_record = db.exec(select(BlueprintRecord).where(BlueprintRecord.session_id == session_id)).first()
    if blueprint_record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Blueprint not found for session")

    tools = list(blueprint_record.tools or [])
    target_index = -1
    for index, tool in enumerate(tools):
        if tool.get("name") == payload.tool_name:
            target_index = index
            break

    if target_index == -1:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tool '{payload.tool_name}' not found in blueprint")

    updated_tool = dict(tools[target_index])
    if payload.registered_api_ref:
        updated_tool["registered_api_ref"] = payload.registered_api_ref
    if payload.endpoint_reference:
        updated_tool["endpoint_reference"] = payload.endpoint_reference
    if payload.auth_reference:
        updated_tool["auth_reference"] = payload.auth_reference
    if payload.openapi_spec:
        spec = payload.openapi_spec
        info = spec.get("info", {})
        if "title" in info:
            updated_tool["purpose"] = f"Integracion con {info['title']}: {info.get('description', '')}".strip()
        paths = spec.get("paths", {})
        if paths:
            first_path = list(paths.keys())[0]
            methods = paths[first_path]
            first_method = list(methods.keys())[0]
            updated_tool["endpoint_reference"] = f"{first_method.upper()} {first_path}"

    tools[target_index] = updated_tool
    blueprint_record.tools = tools
    db.add(blueprint_record)
    touch_session(record, record.current_stage, record.status)
    db.commit()
    db.refresh(blueprint_record)

    return BlueprintTool.model_validate(updated_tool)


@router.patch("/{session_id}/commercial-tier", response_model=SessionSnapshot)
def update_session_commercial_tier(
    session_id: UUID,
    payload: CommercialTierUpdateRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> SessionSnapshot:
    record = get_or_404(db, session_id, current_user.id)
    if not allow_demo_tier_upgrade():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="El cambio directo de tier esta deshabilitado. Usa checkout o grants administrativos.",
        )
    if record.commercial_tier == payload.tier:
        return build_snapshot(db, record, current_user=current_user)
    record.commercial_tier = payload.tier
    touch_session(record, record.current_stage, record.status)
    record_dedicated_commercial_event(
        db,
        workspace_id=record.workspace_id,
        session_id=session_id,
        user_id=current_user.id,
        event_key="legacy_demo_tier_updated",
        product_key=payload.tier.value,
        source="legacy_demo_endpoint",
        correlation_id=f"legacy:{session_id}:{payload.tier.value}",
        metadata={"non_revenue": True},
    )
    write_log(
        db,
        session_id=session_id,
        stage=record.current_stage,
        status_value=record.status,
        message="Tier comercial actualizado",
        payload={
            "commercial_tier": payload.tier,
            "event_key": "legacy_demo_tier_updated",
            "product": payload.tier.value,
            "source": "legacy_demo_endpoint",
            "metadata": {"non_revenue": True},
        },
    )
    capture_operational_state(db, session_id=session_id, source_action="update_commercial_tier")
    db.commit()
    db.refresh(record)
    return build_snapshot(db, record, current_user=current_user)


@router.get("/{session_id}/short-term-memory", response_model=ShortTermMemoryRuntimeState)
def get_short_term_memory_route(
    session_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> ShortTermMemoryRuntimeState:
    record = get_or_404(db, session_id, current_user.id)
    apply_workspace_bootstrap(db, record.workspace_id)
    snapshot = build_snapshot(db, record, include_short_term=False)
    return ShortTermMemoryService().resume_session_state(
        db,
        session_id=session_id,
        snapshot=snapshot,
        source_action="load_short_term_memory",
    )


@router.post("/{session_id}/short-term-memory/reload", response_model=ShortTermMemoryRuntimeState)
def reload_short_term_memory_route(
    session_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> ShortTermMemoryRuntimeState:
    record = get_or_404(db, session_id, current_user.id)
    apply_workspace_bootstrap(db, record.workspace_id)
    snapshot = build_snapshot(db, record, include_short_term=False)
    runtime_state = ShortTermMemoryService().reload_from_snapshot(
        db,
        session_id=session_id,
        snapshot=snapshot,
        source_action="reload_short_term_memory",
    )
    db.commit()
    return runtime_state


@router.post("/{session_id}/short-term-memory/rollback", response_model=ShortTermMemoryRuntimeState)
def rollback_short_term_memory_route(
    session_id: UUID,
    payload: ShortTermMemoryRollbackRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> ShortTermMemoryRuntimeState:
    record = get_or_404(db, session_id, current_user.id)
    apply_workspace_bootstrap(db, record.workspace_id)
    try:
        runtime_state = ShortTermMemoryService().rollback_session_state(
            db,
            session_id=session_id,
            payload=payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    db.commit()
    return runtime_state


@router.post("/{session_id}/workflow-template/apply", response_model=SessionSnapshot)
def apply_workflow_template_route(
    session_id: UUID,
    payload: WorkflowTemplateApplyRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> SessionSnapshot:
    record = get_or_404(db, session_id, current_user.id)
    if not is_feature_flag_enabled(db, FEATURE_FLAG_WORKFLOWS, workspace_id=record.workspace_id):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Workflow templates feature flag is disabled")
    blueprint_record = db.exec(select(BlueprintRecord).where(BlueprintRecord.session_id == session_id)).first()
    if blueprint_record is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Blueprint must exist before applying a workflow template")

    template = db.exec(
        select(WorkflowTemplateRecord).where(
            WorkflowTemplateRecord.workspace_id == record.workspace_id,
            WorkflowTemplateRecord.template_key == payload.template_key,
            WorkflowTemplateRecord.is_active == True,  # noqa: E712
        )
    ).first()
    if template is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workflow template not found")

    updated_blueprint = apply_workflow_template(hydrate_blueprint(blueprint_record), template)
    pending_approvals = sync_approval_gates(db, session_id, updated_blueprint)
    status_value = ArtifactStatus.ready if pending_approvals == 0 and record.status != ArtifactStatus.failed else ArtifactStatus.needs_review
    envelope = BlueprintEnvelope(
        status=status_value,
        stage=SessionStage.post_validation,
        data=updated_blueprint,
        missing_fields=[],
        assumptions=[],
        warnings=[f"Plantilla de workflow aplicada: {template.label}."],
        evidence=[EvidenceItem(source="rule_engine", detail=f"workflow_template={template.template_key}")],
        next_action="review_governance_handoff",
    )
    upsert_blueprint(db, session_id, envelope)
    blueprint_version_number = create_blueprint_version(
        db,
        session_id=session_id,
        source_action="apply_workflow_template",
        status_value=envelope.status,
        blueprint=envelope.data,
    )
    record_delivery_artifacts(
        db,
        session_id=session_id,
        blueprint_version_number=blueprint_version_number,
        source_action="apply_workflow_template",
        stage=envelope.stage,
        blueprint=envelope.data,
    )
    record.selected_workflow_template_key = template.template_key
    sync_governance_handoff(
        db,
        session_record=record,
        blueprint_version_number=blueprint_version_number,
        blueprint=envelope.data,
        source_action="apply_workflow_template",
        pending_approvals=pending_approvals,
    )
    touch_session(record, envelope.stage, envelope.status)
    write_log(
        db,
        session_id=session_id,
        stage=envelope.stage,
        status_value=envelope.status,
        message="Plantilla de workflow aplicada",
        payload={"template_key": template.template_key, "template_label": template.label},
    )
    db.add(record)
    sync_short_term_memory_checkpoint(db, record=record, source_action="apply_workflow_template")
    capture_operational_state(db, session_id=session_id, source_action="apply_workflow_template")
    db.commit()
    return build_snapshot(db, record)


@router.post("/{session_id}/evaluation/bootstrap", response_model=SessionSnapshot)
def bootstrap_evaluation_workbench_route(
    session_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> SessionSnapshot:
    record = get_or_404(db, session_id, current_user.id)
    opportunity = db.exec(select(OpportunityRecord).where(OpportunityRecord.session_id == session_id)).first()
    canvas_record = db.exec(select(CanvasRecord).where(CanvasRecord.session_id == session_id)).first()
    blueprint_record = db.exec(select(BlueprintRecord).where(BlueprintRecord.session_id == session_id)).first()

    discovery = hydrate_discovery(opportunity) if opportunity is not None else None
    canvas = hydrate_canvas(canvas_record) if canvas_record is not None else None
    blueprint = hydrate_blueprint(blueprint_record) if blueprint_record is not None else None
    blueprint_version_number = latest_blueprint_version_number(db, session_id)

    create_evaluation_dataset_version(
        db,
        session_id=session_id,
        blueprint_version_number=blueprint_version_number,
        source_action="bootstrap_dataset",
        dataset=build_default_evaluation_dataset(
            discovery,
            canvas,
            blueprint,
            blueprint_version_number=blueprint_version_number,
            source_action="bootstrap_dataset",
        ),
    )
    create_evaluation_rubric_version(
        db,
        session_id=session_id,
        blueprint_version_number=blueprint_version_number,
        source_action="bootstrap_rubric",
        rubric=build_default_evaluation_rubric(
            blueprint_version_number=blueprint_version_number,
            source_action="bootstrap_rubric",
        ),
    )
    record.updated_at = utc_now()
    db.add(record)
    write_log(
        db,
        session_id=session_id,
        stage=SessionStage.post_validation,
        status_value=record.status,
        message="Workbench de evaluacion regenerado",
        payload={
            "blueprint_version_number": blueprint_version_number,
            "has_blueprint": blueprint is not None,
        },
    )
    sync_short_term_memory_checkpoint(db, record=record, source_action="bootstrap_evaluation_workbench")
    capture_operational_state(db, session_id=session_id, source_action="bootstrap_evaluation_workbench")
    db.commit()
    return build_snapshot(db, record)


@router.patch("/{session_id}/evaluation/dataset", response_model=SessionSnapshot)
def update_evaluation_dataset_route(
    session_id: UUID,
    payload: EvaluationDatasetUpdateRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> SessionSnapshot:
    record = get_or_404(db, session_id, current_user.id)
    latest_dataset_record = latest_evaluation_dataset(db, session_id)
    base_summary = (
        hydrate_evaluation_dataset(db, latest_dataset_record).summary
        if latest_dataset_record is not None
        else "Dataset ajustado manualmente por el usuario."
    )
    blueprint_version_number = latest_blueprint_version_number(db, session_id)
    create_evaluation_dataset_version(
        db,
        session_id=session_id,
        blueprint_version_number=blueprint_version_number,
        source_action="manual_dataset_edit",
        dataset=EvaluationDatasetArtifact(
            blueprint_version_number=blueprint_version_number,
            source_action="manual_dataset_edit",
            status=ArtifactStatus.ready if payload.cases else ArtifactStatus.needs_review,
            summary=base_summary,
            cases=[
                item.model_copy(update={"source": item.source or "manual"})
                for item in payload.cases
            ],
        ),
    )
    record.updated_at = utc_now()
    db.add(record)
    write_log(
        db,
        session_id=session_id,
        stage=SessionStage.post_validation,
        status_value=record.status,
        message="Dataset de evaluacion actualizado",
        payload={"case_count": len(payload.cases)},
    )
    sync_short_term_memory_checkpoint(db, record=record, source_action="update_evaluation_dataset")
    capture_operational_state(db, session_id=session_id, source_action="update_evaluation_dataset")
    db.commit()
    return build_snapshot(db, record)


@router.patch("/{session_id}/evaluation/rubric", response_model=SessionSnapshot)
def update_evaluation_rubric_route(
    session_id: UUID,
    payload: EvaluationRubricUpdateRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> SessionSnapshot:
    record = get_or_404(db, session_id, current_user.id)
    latest_rubric_record = latest_evaluation_rubric(db, session_id)
    current_rubric = hydrate_evaluation_rubric(latest_rubric_record) if latest_rubric_record is not None else None
    blueprint_version_number = latest_blueprint_version_number(db, session_id)
    create_evaluation_rubric_version(
        db,
        session_id=session_id,
        blueprint_version_number=blueprint_version_number,
        source_action="manual_rubric_edit",
        rubric=EvaluationRubricArtifact(
            blueprint_version_number=blueprint_version_number,
            source_action="manual_rubric_edit",
            summary=payload.summary or (current_rubric.summary if current_rubric is not None else "Rubrica ajustada manualmente."),
            dimensions=payload.dimensions or (current_rubric.dimensions if current_rubric is not None else []),
        ),
    )
    record.updated_at = utc_now()
    db.add(record)
    write_log(
        db,
        session_id=session_id,
        stage=SessionStage.post_validation,
        status_value=record.status,
        message="Rubrica de evaluacion actualizada",
        payload={"dimension_count": len(payload.dimensions)},
    )
    sync_short_term_memory_checkpoint(db, record=record, source_action="update_evaluation_rubric")
    capture_operational_state(db, session_id=session_id, source_action="update_evaluation_rubric")
    db.commit()
    return build_snapshot(db, record)


@router.post("/{session_id}/normalize-discovery", response_model=DiscoveryEnvelope)
def normalize_discovery_route(
    session_id: UUID,
    payload: DiscoveryInput,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> DiscoveryEnvelope:
    record = get_or_404(db, session_id, current_user.id)
    runtime_settings = load_effective_runtime_settings(db, record.workspace_id)
    stage_context = build_stage_context_bundle(
        db,
        record=record,
        capability="normalize_discovery",
        stage="discover",
        effective_language=current_user.preferred_language,
        task_source_keys=["discovery_capture"],
    )
    envelope, traces = run_discovery_stage(
        payload,
        runtime_settings=runtime_settings,
        stage_context=stage_context,
    )

    upsert_opportunity(db, session_id, envelope)
    maybe_set_session_title(record, envelope.data)
    touch_session(record, envelope.stage, envelope.status)
    write_skill_runs(
        db,
        session_id=session_id,
        traces=traces,
        source_action="normalize_discovery",
    )
    write_validation(
        db,
        session_id=session_id,
        artifact_name="discovery",
        status_value=envelope.status,
        missing_fields=envelope.missing_fields,
        warnings=envelope.warnings,
    )
    write_log(
        db,
        session_id=session_id,
        stage=envelope.stage,
        status_value=envelope.status,
        message="Discovery normalizado",
        payload=envelope.model_dump(mode="json"),
    )
    db.add(record)
    sync_short_term_memory_checkpoint(db, record=record, source_action="normalize_discovery")
    capture_operational_state(db, session_id=session_id, source_action="normalize_discovery")
    db.commit()
    return envelope


@router.post("/{session_id}/analyze-discovery", response_model=JourneyStageArtifactEntry)
def analyze_discovery_route(
    session_id: UUID,
    payload: DiscoveryInput,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> JourneyStageArtifactEntry:
    record = get_or_404(db, session_id, current_user.id)
    runtime_settings = load_effective_runtime_settings(db, record.workspace_id)
    stage_context = build_stage_context_bundle(
        db,
        record=record,
        capability="analyze_discovery",
        stage="discover",
        effective_language=current_user.preferred_language,
        task_source_keys=["discovery_analysis_input"],
    )
    analysis, traces = run_discovery_analysis_stage(
        payload,
        runtime_settings=runtime_settings,
        stage_context=stage_context,
    )
    trace = traces[0]
    proposal_payload = analysis.model_dump(mode="json")
    proposal_payload["schema_version"] = "discovery-analysis.v1"
    service = StageProposalService()
    try:
        artifact = service.create(
            db,
            session_record=record,
            stage_key="discover",
            payload=JourneyStageArtifactCreateRequest(
                artifact_kind="discovery_analysis_artifact",
                source_action="analyze_discovery",
                proposal_payload=proposal_payload,
                provider_key=trace.llm_trace.provider_key if trace.llm_trace is not None else "",
                model=trace.llm_trace.model_name if trace.llm_trace is not None else "",
                execution_backend=trace.llm_trace.execution_backend if trace.llm_trace is not None else "",
                prompt_version=trace.llm_trace.prompt_version if trace.llm_trace is not None else "",
                schema_version="discovery-analysis.v1",
                confidence=analysis.confidence,
                missing_information=list(analysis.missing_information),
                warnings=list(trace.warnings),
                evidence_manifest=build_journey_evidence_manifest_from_trace(
                    trace=trace,
                    source_action="analyze_discovery",
                ),
            ),
            actor_user=current_user,
        )
    except Exception as exc:  # noqa: BLE001
        _raise_stage_proposal_http_error(exc)
    write_skill_runs(
        db,
        session_id=session_id,
        traces=traces,
        source_action="analyze_discovery",
    )
    write_validation(
        db,
        session_id=session_id,
        artifact_name="discovery_analysis",
        status_value=ArtifactStatus.ready if not analysis.missing_information else ArtifactStatus.needs_review,
        missing_fields=list(analysis.missing_information),
        warnings=list(trace.warnings),
    )
    write_log(
        db,
        session_id=session_id,
        stage=SessionStage.normalize_discovery,
        status_value=ArtifactStatus.ready if not analysis.missing_information else ArtifactStatus.needs_review,
        message="Discovery analizado",
        payload=proposal_payload,
    )
    sync_short_term_memory_checkpoint(db, record=record, source_action="analyze_discovery")
    capture_operational_state(db, session_id=session_id, source_action="analyze_discovery")
    db.commit()
    return artifact


@router.post("/{session_id}/build-canvas", response_model=CanvasEnvelope)
def build_canvas_route(
    session_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> CanvasEnvelope:
    record = get_or_404(db, session_id, current_user.id)
    JourneyStageMigrationService().backfill_session(db, session_record=record)
    proposal_service = StageProposalService()
    opportunity = db.exec(select(OpportunityRecord).where(OpportunityRecord.session_id == session_id)).first()
    if opportunity is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Discovery must exist before canvas")
    latest_discover_artifact = proposal_service.latest(db, session_record=record, stage_key="discover")
    if not is_discovery_stage_approved(latest_discover_artifact):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Discover must be approved before canvas",
        )
    canonical_discovery = hydrate_discovery(opportunity)
    approved_discovery = resolve_discovery_from_stage_artifact(latest_discover_artifact)
    if approved_discovery.model_dump(mode="json") != canonical_discovery.model_dump(mode="json"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Discovery draft differs from the approved discover proposal. Re-run and approve Discover before canvas.",
        )

    runtime_settings = load_effective_runtime_settings(db, record.workspace_id)
    stage_context = build_stage_context_bundle(
        db,
        record=record,
        capability="build_canvas",
        stage="define",
        effective_language=current_user.preferred_language,
        task_source_keys=["normalized_discovery"],
    )
    envelope, traces = run_canvas_stage(
        approved_discovery,
        runtime_settings=runtime_settings,
        stage_context=stage_context,
    )
    upsert_canvas(db, session_id, envelope)
    touch_session(record, envelope.stage, envelope.status)
    write_skill_runs(
        db,
        session_id=session_id,
        traces=traces,
        source_action="build_canvas",
    )
    write_validation(
        db,
        session_id=session_id,
        artifact_name="canvas",
        status_value=envelope.status,
        missing_fields=envelope.missing_fields,
        warnings=envelope.warnings,
    )
    write_log(
        db,
        session_id=session_id,
        stage=envelope.stage,
        status_value=envelope.status,
        message="Canvas construido",
        payload=envelope.model_dump(mode="json"),
    )
    db.add(record)
    sync_short_term_memory_checkpoint(db, record=record, source_action="build_canvas")
    capture_operational_state(db, session_id=session_id, source_action="build_canvas")
    db.commit()
    return envelope


@router.post("/{session_id}/define-requirements", response_model=JourneyStageArtifactEntry)
def define_requirements_route(
    session_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> JourneyStageArtifactEntry:
    record = get_or_404(db, session_id, current_user.id)
    JourneyStageMigrationService().backfill_session(db, session_record=record)
    proposal_service = StageProposalService()
    opportunity = db.exec(select(OpportunityRecord).where(OpportunityRecord.session_id == session_id)).first()
    if opportunity is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Discovery must exist before define requirements")
    latest_discover_artifact = proposal_service.latest(db, session_record=record, stage_key="discover")
    if not is_discovery_stage_approved(latest_discover_artifact):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Discover must be approved before define requirements",
        )
    canonical_discovery = hydrate_discovery(opportunity)
    approved_discovery = resolve_discovery_from_stage_artifact(latest_discover_artifact)
    if approved_discovery.model_dump(mode="json") != canonical_discovery.model_dump(mode="json"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Discovery draft differs from the approved discover proposal. Re-run and approve Discover before Define.",
        )

    runtime_settings = load_effective_runtime_settings(db, record.workspace_id)
    canvas_record = db.exec(select(CanvasRecord).where(CanvasRecord.session_id == session_id)).first()
    working_canvas = hydrate_canvas(canvas_record) if canvas_record is not None else None
    all_traces = []
    warnings: list[str] = []
    if working_canvas is None:
        canvas_stage_context = build_stage_context_bundle(
            db,
            record=record,
            capability="build_canvas",
            stage="define",
            effective_language=current_user.preferred_language,
            task_source_keys=["normalized_discovery"],
        )
        canvas_envelope, canvas_traces = run_canvas_stage(
            approved_discovery,
            runtime_settings=runtime_settings,
            stage_context=canvas_stage_context,
        )
        working_canvas = canvas_envelope.data
        all_traces.extend(canvas_traces)
        warnings.extend(canvas_envelope.warnings)

    stage_context = build_stage_context_bundle(
        db,
        record=record,
        capability="define_requirements",
        stage="define",
        effective_language=current_user.preferred_language,
        task_source_keys=["requirements_definition_input"],
        allow_second_page=True,
    )
    definition, traces = run_definition_stage(
        approved_discovery,
        working_canvas,
        runtime_settings=runtime_settings,
        stage_context=stage_context,
    )
    all_traces.extend(traces)
    warnings.extend(traces[0].warnings)
    proposal_payload = definition.model_dump(mode="json")
    proposal_payload["schema_version"] = "definition-artifact.v1"
    trace = traces[0]
    try:
        artifact = proposal_service.create(
            db,
            session_record=record,
            stage_key="define",
            payload=JourneyStageArtifactCreateRequest(
                artifact_kind="definition_artifact",
                source_action="define_requirements",
                proposal_payload=proposal_payload,
                provider_key=trace.llm_trace.provider_key if trace.llm_trace is not None else "",
                model=trace.llm_trace.model_name if trace.llm_trace is not None else "",
                execution_backend=trace.llm_trace.execution_backend if trace.llm_trace is not None else "",
                prompt_version=trace.llm_trace.prompt_version if trace.llm_trace is not None else "",
                schema_version="definition-artifact.v1",
                confidence=definition.confidence,
                missing_information=list(definition.validation.blocking_issues),
                warnings=list(dict.fromkeys(warnings)),
                evidence_manifest=build_journey_evidence_manifest_from_traces(
                    traces=all_traces,
                    source_action="define_requirements",
                ),
            ),
            actor_user=current_user,
        )
    except Exception as exc:  # noqa: BLE001
        _raise_stage_proposal_http_error(exc)

    write_skill_runs(
        db,
        session_id=session_id,
        traces=all_traces,
        source_action="define_requirements",
    )
    write_validation(
        db,
        session_id=session_id,
        artifact_name="definition",
        status_value=ArtifactStatus.ready if not definition.validation.blocking_issues else ArtifactStatus.needs_review,
        missing_fields=list(definition.validation.blocking_issues),
        warnings=list(dict.fromkeys(warnings)),
    )
    write_log(
        db,
        session_id=session_id,
        stage=SessionStage.build_canvas,
        status_value=ArtifactStatus.ready if not definition.validation.blocking_issues else ArtifactStatus.needs_review,
        message="Definition consolidada",
        payload=proposal_payload,
    )
    sync_short_term_memory_checkpoint(db, record=record, source_action="define_requirements")
    capture_operational_state(db, session_id=session_id, source_action="define_requirements")
    db.commit()
    return artifact


@router.post("/{session_id}/propose-design", response_model=JourneyStageArtifactEntry)
def propose_design_route(
    session_id: UUID,
    payload: DesignProposalRequest | None = None,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> JourneyStageArtifactEntry:
    record = get_or_404(db, session_id, current_user.id)
    JourneyStageMigrationService().backfill_session(db, session_record=record)
    proposal_service = StageProposalService()
    latest_define_artifact = proposal_service.latest(db, session_record=record, stage_key="define")
    if not is_define_stage_approved(latest_define_artifact):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Define must be approved before design",
        )

    opportunity = db.exec(select(OpportunityRecord).where(OpportunityRecord.session_id == session_id)).first()
    canvas_record = db.exec(select(CanvasRecord).where(CanvasRecord.session_id == session_id)).first()
    if opportunity is None or canvas_record is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Discovery and canvas must exist before design",
        )

    approved_discovery = hydrate_discovery(opportunity)
    approved_canvas = hydrate_canvas(canvas_record)
    expected_canvas = resolve_canvas_from_define_stage_artifact(latest_define_artifact)
    if expected_canvas.model_dump(mode="json") != approved_canvas.model_dump(mode="json"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Canvas differs from the approved define proposal. Re-run and approve Define before Design.",
        )

    definition_payload = latest_define_artifact.proposal_payload if latest_define_artifact is not None else {}
    definition_artifact = validate_definition_artifact(RequirementsDefinitionOutput.model_validate(definition_payload))
    runtime_settings = load_effective_runtime_settings(db, record.workspace_id)
    proposal_stage_context = build_stage_context_bundle(
        db,
        record=record,
        capability="propose_agent_design",
        stage="design",
        effective_language=current_user.preferred_language,
        task_source_keys=["agent_design_input"],
        allow_second_page=True,
    )
    critique_stage_context = build_stage_context_bundle(
        db,
        record=record,
        capability="critique_agent_design",
        stage="design",
        effective_language=current_user.preferred_language,
        task_source_keys=["agent_design_critique_input"],
        allow_second_page=True,
    )
    artifact_payload, traces = run_design_stage(
        approved_discovery,
        approved_canvas,
        definition_artifact,
        instructions=payload.instructions if payload is not None else "",
        runtime_settings=runtime_settings,
        proposal_stage_context=proposal_stage_context,
        critique_stage_context=critique_stage_context,
    )
    if payload is not None and payload.instructions.strip():
        artifact_payload = artifact_payload.model_copy(
            update={
                "decision_rationale": (
                    f"{artifact_payload.decision_rationale} Instrucciones de regeneracion: {payload.instructions.strip()}"
                ).strip(),
            }
        )
    stage_trace = next((trace.llm_trace for trace in traces if trace.llm_trace is not None), None)
    warnings = list(dict.fromkeys([warning for trace in traces for warning in trace.warnings]))
    if artifact_payload.review_state == ReviewState.blocked:
        warnings.append("Design detecto findings bloqueantes; resuelvelos antes de aprobar la arquitectura.")
    proposal_payload = artifact_payload.model_dump(mode="json")
    try:
        artifact = proposal_service.create(
            db,
            session_record=record,
            stage_key="design",
            payload=JourneyStageArtifactCreateRequest(
                artifact_kind="design_recommendation_artifact",
                source_action="propose_design",
                proposal_payload=proposal_payload,
                provider_key=stage_trace.provider_key if stage_trace is not None else "",
                model=stage_trace.model_name if stage_trace is not None else "",
                execution_backend=stage_trace.execution_backend if stage_trace is not None else "",
                prompt_version=stage_trace.prompt_version if stage_trace is not None else "",
                schema_version="design-recommendation.v1",
                confidence=artifact_payload.confidence.overall,
                missing_information=list(artifact_payload.missing_information),
                warnings=warnings,
                evidence_manifest=build_journey_evidence_manifest_from_traces(
                    traces=traces,
                    source_action="propose_design",
                ),
            ),
            actor_user=current_user,
        )
    except Exception as exc:  # noqa: BLE001
        _raise_stage_proposal_http_error(exc)

    write_skill_runs(
        db,
        session_id=session_id,
        traces=traces,
        source_action="propose_design",
    )
    write_validation(
        db,
        session_id=session_id,
        artifact_name="design",
        status_value=ArtifactStatus.ready if artifact_payload.review_state == ReviewState.complete else ArtifactStatus.needs_review,
        missing_fields=list(artifact_payload.missing_information),
        warnings=warnings,
    )
    write_log(
        db,
        session_id=session_id,
        stage=SessionStage.build_blueprint,
        status_value=ArtifactStatus.ready if artifact_payload.review_state == ReviewState.complete else ArtifactStatus.needs_review,
        message="Design comparado y criticado",
        payload=proposal_payload,
    )
    sync_short_term_memory_checkpoint(db, record=record, source_action="propose_design")
    capture_operational_state(db, session_id=session_id, source_action="propose_design")
    db.commit()
    return artifact


@router.post("/{session_id}/build-blueprint", response_model=BlueprintEnvelope)
def build_blueprint_route(
    session_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> BlueprintEnvelope:
    record = get_or_404(db, session_id, current_user.id)
    JourneyStageMigrationService().backfill_session(db, session_record=record)
    proposal_service = StageProposalService()
    latest_define_artifact = proposal_service.latest(db, session_record=record, stage_key="define")
    if not is_define_stage_approved(latest_define_artifact):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Define must be approved before blueprint",
        )
    opportunity = db.exec(select(OpportunityRecord).where(OpportunityRecord.session_id == session_id)).first()
    canvas = db.exec(select(CanvasRecord).where(CanvasRecord.session_id == session_id)).first()
    if opportunity is None or canvas is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Discovery and canvas must exist before blueprint",
        )

    discovery_artifact = hydrate_discovery(opportunity)
    discovery_gaps = [
        item
        for item in find_missing_discovery_fields(discovery_artifact.model_dump(mode="json"))
        if item.startswith("operational_baseline.") or item.startswith("mvp_definition.")
    ]
    if discovery_gaps:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Discovery must include operational baseline and MVP definition before blueprint: "
                + ", ".join(discovery_gaps)
            ),
        )

    canvas_artifact = hydrate_canvas(canvas)
    approved_canvas = resolve_canvas_from_define_stage_artifact(latest_define_artifact)
    if approved_canvas.model_dump(mode="json") != canvas_artifact.model_dump(mode="json"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Canvas differs from the approved define proposal. Re-run and approve Define before Design.",
        )
    if not canvas_artifact.success_metric or not canvas_artifact.mvp_scope or not canvas_artifact.out_of_scope:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Canvas must include success_metric, mvp_scope and out_of_scope before blueprint",
        )

    runtime_settings = load_effective_runtime_settings(db, record.workspace_id)
    stage_context = build_stage_context_bundle(
        db,
        record=record,
        capability="synthesize_blueprint_narrative",
        stage="design",
        effective_language=current_user.preferred_language,
        task_source_keys=["narrative_discovery", "narrative_canvas", "narrative_blueprint"],
        allow_second_page=True,
    )
    envelope, traces = run_blueprint_stage(
        discovery_artifact,
        canvas_artifact,
        runtime_settings=runtime_settings,
        stage_context=stage_context,
    )
    pending_approvals = sync_approval_gates(db, session_id, envelope.data)
    envelope = apply_pending_approvals_to_blueprint(envelope, pending_approvals)
    upsert_blueprint(db, session_id, envelope)
    blueprint_version_number = create_blueprint_version(
        db,
        session_id=session_id,
        source_action="build_blueprint",
        status_value=envelope.status,
        blueprint=envelope.data,
    )
    record_delivery_artifacts(
        db,
        session_id=session_id,
        blueprint_version_number=blueprint_version_number,
        source_action="build_blueprint",
        stage=envelope.stage,
        blueprint=envelope.data,
    )
    if not record.selected_workflow_template_key:
        record.selected_workflow_template_key = recommend_workflow_template_key(envelope.data)
    sync_governance_handoff(
        db,
        session_record=record,
        blueprint_version_number=blueprint_version_number,
        blueprint=envelope.data,
        source_action="build_blueprint",
        pending_approvals=pending_approvals,
    )
    touch_session(record, envelope.stage, envelope.status)
    write_skill_runs(
        db,
        session_id=session_id,
        traces=traces,
        source_action="build_blueprint",
        blueprint_version_number=blueprint_version_number,
    )
    write_validation(
        db,
        session_id=session_id,
        artifact_name="blueprint",
        status_value=envelope.status,
        missing_fields=envelope.missing_fields,
        warnings=envelope.warnings,
    )
    write_log(
        db,
        session_id=session_id,
        stage=envelope.stage,
        status_value=envelope.status,
        message="Blueprint construido",
        payload=envelope.model_dump(mode="json"),
    )
    db.add(record)
    sync_short_term_memory_checkpoint(db, record=record, source_action="build_blueprint")
    capture_operational_state(db, session_id=session_id, source_action="build_blueprint")
    db.commit()
    return envelope


@router.post("/{session_id}/enrich-blueprint", response_model=BlueprintEnvelope)
def enrich_blueprint_route(
    session_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> BlueprintEnvelope:
    record = get_or_404(db, session_id, current_user.id)
    opportunity = db.exec(select(OpportunityRecord).where(OpportunityRecord.session_id == session_id)).first()
    canvas = db.exec(select(CanvasRecord).where(CanvasRecord.session_id == session_id)).first()
    blueprint = db.exec(select(BlueprintRecord).where(BlueprintRecord.session_id == session_id)).first()
    if opportunity is None or blueprint is None or canvas is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Discovery, canvas and blueprint must exist first")

    runtime_settings = load_effective_runtime_settings(db, record.workspace_id)
    stage_context = build_stage_context_bundle(
        db,
        record=record,
        capability="synthesize_blueprint_narrative",
        stage="design",
        effective_language=current_user.preferred_language,
        task_source_keys=["narrative_discovery", "narrative_canvas", "narrative_blueprint"],
        allow_second_page=True,
    )
    envelope, traces = run_enrich_stage(
        hydrate_blueprint(blueprint),
        hydrate_discovery(opportunity),
        hydrate_canvas(canvas),
        runtime_settings=runtime_settings,
        stage_context=stage_context,
    )
    pending_approvals = sync_approval_gates(db, session_id, envelope.data)
    envelope = apply_pending_approvals_to_blueprint(envelope, pending_approvals)
    upsert_blueprint(db, session_id, envelope)
    blueprint_version_number = create_blueprint_version(
        db,
        session_id=session_id,
        source_action="enrich_blueprint",
        status_value=envelope.status,
        blueprint=envelope.data,
    )
    record_delivery_artifacts(
        db,
        session_id=session_id,
        blueprint_version_number=blueprint_version_number,
        source_action="enrich_blueprint",
        stage=envelope.stage,
        blueprint=envelope.data,
    )
    if not record.selected_workflow_template_key:
        record.selected_workflow_template_key = recommend_workflow_template_key(envelope.data)
    sync_governance_handoff(
        db,
        session_record=record,
        blueprint_version_number=blueprint_version_number,
        blueprint=envelope.data,
        source_action="enrich_blueprint",
        pending_approvals=pending_approvals,
    )
    touch_session(record, envelope.stage, envelope.status)
    write_skill_runs(
        db,
        session_id=session_id,
        traces=traces,
        source_action="enrich_blueprint",
        blueprint_version_number=blueprint_version_number,
    )
    write_validation(
        db,
        session_id=session_id,
        artifact_name="blueprint_details",
        status_value=envelope.status,
        missing_fields=envelope.missing_fields,
        warnings=envelope.warnings,
    )
    write_log(
        db,
        session_id=session_id,
        stage=envelope.stage,
        status_value=envelope.status,
        message="Blueprint enriquecido",
        payload=envelope.model_dump(mode="json"),
    )
    db.add(record)
    sync_short_term_memory_checkpoint(db, record=record, source_action="enrich_blueprint")
    capture_operational_state(db, session_id=session_id, source_action="enrich_blueprint")
    db.commit()
    return envelope


@router.patch("/{session_id}/blueprint", response_model=BlueprintEnvelope)
def patch_blueprint_route(
    session_id: UUID,
    payload: BlueprintPatchRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> BlueprintEnvelope:
    record = get_or_404(db, session_id, current_user.id)
    opportunity = db.exec(select(OpportunityRecord).where(OpportunityRecord.session_id == session_id)).first()
    canvas = db.exec(select(CanvasRecord).where(CanvasRecord.session_id == session_id)).first()
    blueprint = db.exec(select(BlueprintRecord).where(BlueprintRecord.session_id == session_id)).first()
    if blueprint is None or opportunity is None or canvas is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Blueprint context not found")

    envelope = patch_blueprint(
        hydrate_blueprint(blueprint),
        payload,
        hydrate_discovery(opportunity),
        hydrate_canvas(canvas),
    )
    pending_approvals = sync_approval_gates(db, session_id, envelope.data)
    envelope = apply_pending_approvals_to_blueprint(envelope, pending_approvals)
    upsert_blueprint(db, session_id, envelope)
    blueprint_version_number = create_blueprint_version(
        db,
        session_id=session_id,
        source_action="manual_patch",
        status_value=envelope.status,
        blueprint=envelope.data,
    )
    record_delivery_artifacts(
        db,
        session_id=session_id,
        blueprint_version_number=blueprint_version_number,
        source_action="manual_patch",
        stage=envelope.stage,
        blueprint=envelope.data,
    )
    if not record.selected_workflow_template_key:
        record.selected_workflow_template_key = recommend_workflow_template_key(envelope.data)
    sync_governance_handoff(
        db,
        session_record=record,
        blueprint_version_number=blueprint_version_number,
        blueprint=envelope.data,
        source_action="manual_patch",
        pending_approvals=pending_approvals,
    )
    touch_session(record, envelope.stage, envelope.status)
    write_validation(
        db,
        session_id=session_id,
        artifact_name="blueprint_manual_patch",
        status_value=envelope.status,
        missing_fields=envelope.missing_fields,
        warnings=envelope.warnings,
    )
    write_log(
        db,
        session_id=session_id,
        stage=envelope.stage,
        status_value=envelope.status,
        message="Blueprint ajustado manualmente",
        payload=envelope.model_dump(mode="json"),
    )
    db.add(record)
    sync_short_term_memory_checkpoint(db, record=record, source_action="manual_patch")
    capture_operational_state(db, session_id=session_id, source_action="manual_patch")
    db.commit()
    return envelope


@router.post("/{session_id}/recommend-tools", response_model=ToolRecommendationEnvelope)
def recommend_tools_route(
    session_id: UUID,
    payload: ToolRecommendationRequest | None = None,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> ToolRecommendationEnvelope:
    record = get_or_404(db, session_id, current_user.id)
    JourneyStageMigrationService().backfill_session(db, session_record=record)
    proposal_service = StageProposalService()
    ensure_route_feature_flag_enabled(
        db,
        workspace_id=record.workspace_id,
        flag_key=FEATURE_FLAG_TOOL_RECOMMENDATION,
        detail="Tool recommendation feature flag is disabled",
    )
    latest_discover_artifact = proposal_service.latest(db, session_record=record, stage_key="discover")
    latest_define_artifact = proposal_service.latest(db, session_record=record, stage_key="define")
    if not is_define_stage_approved(latest_define_artifact):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Define must be approved before recommending tools",
        )
    latest_design_artifact = proposal_service.latest(db, session_record=record, stage_key="design")
    if not is_design_stage_approved(latest_design_artifact):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Design must be approved before recommending tools",
        )

    opportunity = db.exec(select(OpportunityRecord).where(OpportunityRecord.session_id == session_id)).first()
    canvas = db.exec(select(CanvasRecord).where(CanvasRecord.session_id == session_id)).first()
    blueprint = db.exec(select(BlueprintRecord).where(BlueprintRecord.session_id == session_id)).first()
    if opportunity is None or canvas is None or blueprint is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Discovery, canvas and blueprint must exist before recommending tools",
        )

    blueprint_version_number = latest_blueprint_version_number(db, session_id)
    runtime_settings = load_effective_runtime_settings(db, record.workspace_id)
    definition_artifact = validate_definition_artifact(
        RequirementsDefinitionOutput.model_validate(latest_define_artifact.proposal_payload)
    )
    design_artifact = resolve_design_from_stage_artifact(latest_design_artifact)
    if design_artifact is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The approved design proposal could not be resolved before Tools",
        )
    stage_context = build_stage_context_bundle(
        db,
        record=record,
        capability="recommend_minimal_tools",
        stage="tools",
        effective_language=current_user.preferred_language,
        task_source_keys=["tool_recommendation_case", "tool_recommendation_catalog"],
        allow_second_page=True,
    )
    envelope, traces = run_tool_recommendation_stage(
        session_id,
        hydrate_discovery(opportunity),
        hydrate_canvas(canvas),
        hydrate_blueprint(blueprint),
        definition_artifact=definition_artifact,
        design_artifact=design_artifact,
        instructions=payload.instructions if payload is not None else "",
        blueprint_version_number=blueprint_version_number,
        runtime_settings=runtime_settings,
        stage_context=stage_context,
    )
    envelope = envelope.model_copy(
        update={
            "data": envelope.data.model_copy(
                update={
                    "source_stage_versions": ToolRecommendationSourceStageVersions(
                        discover=latest_discover_artifact.version_number if latest_discover_artifact is not None else None,
                        define=latest_define_artifact.version_number if latest_define_artifact is not None else None,
                        design=latest_design_artifact.version_number if latest_design_artifact is not None else None,
                    )
                }
            )
        }
    )

    record_tool_recommendation_artifact(
        db,
        session_id=session_id,
        blueprint_version_number=blueprint_version_number,
        stage=envelope.stage,
        source_action="recommend_tools",
        recommendation=envelope.data,
    )
    write_skill_runs(
        db,
        session_id=session_id,
        traces=traces,
        source_action="recommend_tools",
        blueprint_version_number=blueprint_version_number,
    )
    write_validation(
        db,
        session_id=session_id,
        artifact_name="tool_recommendation",
        status_value=envelope.status,
        missing_fields=envelope.missing_fields,
        warnings=envelope.warnings,
    )
    write_log(
        db,
        session_id=session_id,
        stage=envelope.stage,
        status_value=envelope.status,
        message="Recomendacion de tools generada",
        payload=envelope.model_dump(mode="json"),
    )
    stage_trace = envelope.llm_trace
    try:
        proposal_service.create(
            db,
            session_record=record,
            stage_key="tools",
            payload=JourneyStageArtifactCreateRequest(
                artifact_kind="tool_recommendation_artifact",
                source_action="recommend_tools",
                proposal_payload=envelope.data.model_dump(mode="json"),
                provider_key=stage_trace.provider_key if stage_trace is not None else "",
                model=stage_trace.model_name if stage_trace is not None else "",
                execution_backend=stage_trace.execution_backend if stage_trace is not None else "",
                prompt_version=stage_trace.prompt_version if stage_trace is not None else "",
                schema_version="tool-recommendation.v1",
                confidence=envelope.data.confidence.overall,
                missing_information=[
                    *dict.fromkeys(
                        [
                            item.title
                            for item in [*envelope.data.coverage_gaps, *envelope.data.needs_information]
                            if item.title
                        ]
                    )
                ],
                warnings=list(dict.fromkeys(envelope.warnings)),
                corpus_hash=stage_context.corpus_hash if stage_context is not None else "",
                evidence_manifest=build_journey_evidence_manifest_from_traces(
                    traces=traces,
                    source_action="recommend_tools",
                ),
            ),
            actor_user=current_user,
        )
    except Exception as exc:  # noqa: BLE001
        _raise_stage_proposal_http_error(exc)
    db.add(record)
    sync_short_term_memory_checkpoint(db, record=record, source_action="recommend_tools")
    capture_operational_state(db, session_id=session_id, source_action="recommend_tools")
    db.commit()
    return envelope


@router.post("/{session_id}/approve-tools-selection", response_model=SessionSnapshot)
def approve_tools_selection_route(
    session_id: UUID,
    payload: ApproveToolsSelectionRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> SessionSnapshot:
    record = get_or_404(db, session_id, current_user.id)
    JourneyStageMigrationService().backfill_session(db, session_record=record)
    proposal_service = StageProposalService()
    latest_design_artifact = proposal_service.latest(db, session_record=record, stage_key="design")
    if not is_design_stage_approved(latest_design_artifact):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Design must be approved before promoting tools",
        )
    latest_tools_artifact = proposal_service.latest(db, session_record=record, stage_key="tools")
    if latest_tools_artifact is not None and latest_tools_artifact.schema_version == "tool-recommendation.v1":
        try:
            proposal_service.approve(
                db,
                session_record=record,
                stage_key="tools",
                artifact_id=latest_tools_artifact.id,
                payload=JourneyStageArtifactApprovalRequest(
                    note="Promocion legacy de tools delegada al lifecycle del journey.",
                    decision_payload={"include_optional_tool_keys": payload.include_optional_tool_keys},
                ),
                actor_user=current_user,
            )
        except Exception as exc:  # noqa: BLE001
            _raise_stage_proposal_http_error(exc)
        sync_short_term_memory_checkpoint(db, record=record, source_action="approve_tools_selection")
        capture_operational_state(db, session_id=session_id, source_action="approve_tools_selection")
        db.commit()
        return build_snapshot(db, record)

    opportunity = db.exec(select(OpportunityRecord).where(OpportunityRecord.session_id == session_id)).first()
    canvas_record = db.exec(select(CanvasRecord).where(CanvasRecord.session_id == session_id)).first()
    blueprint_record = db.exec(select(BlueprintRecord).where(BlueprintRecord.session_id == session_id)).first()
    if opportunity is None or canvas_record is None or blueprint_record is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Discovery, canvas and blueprint must exist before promoting approved tools",
        )

    discovery = hydrate_discovery(opportunity)
    canvas = hydrate_canvas(canvas_record)
    current_blueprint = hydrate_blueprint(blueprint_record)
    latest_recommendation = load_latest_tool_recommendation(
        db,
        session_id,
        discovery=discovery,
        canvas=canvas,
        blueprint=current_blueprint,
        current_blueprint_version_number=latest_blueprint_version_number(db, session_id),
    )
    if latest_recommendation is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A tool recommendation must exist before approving the selection",
        )
    if latest_recommendation.is_stale:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The latest tool recommendation is stale; rerun Tools before approving the selection",
        )

    try:
        approved_tools, review_decisions, digest = promote_tool_recommendation_to_blueprint_tools(
            latest_recommendation,
            include_optional_tool_keys=payload.include_optional_tool_keys,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    envelope = patch_blueprint(
        current_blueprint,
        BlueprintPatchRequest(
            tools=approved_tools,
            knowledge_profile=derive_knowledge_profile(
                discovery,
                approved_tools,
                current_blueprint.memory_strategy,
            ),
        ),
        discovery,
        canvas,
    )
    pending_approvals = sync_approval_gates(db, session_id, envelope.data)
    envelope = apply_pending_approvals_to_blueprint(envelope, pending_approvals)
    upsert_blueprint(db, session_id, envelope)
    blueprint_version_number = create_blueprint_version(
        db,
        session_id=session_id,
        source_action="approve_tools_selection",
        status_value=envelope.status,
        blueprint=envelope.data,
    )
    record_delivery_artifacts(
        db,
        session_id=session_id,
        blueprint_version_number=blueprint_version_number,
        source_action="approve_tools_selection",
        stage=envelope.stage,
        blueprint=envelope.data,
    )
    if not record.selected_workflow_template_key:
        record.selected_workflow_template_key = recommend_workflow_template_key(envelope.data)
    sync_governance_handoff(
        db,
        session_record=record,
        blueprint_version_number=blueprint_version_number,
        blueprint=envelope.data,
        source_action="approve_tools_selection",
        pending_approvals=pending_approvals,
    )

    promoted_digest = digest.model_copy(update={"promoted_blueprint_version": blueprint_version_number})
    promoted_recommendation = latest_recommendation.model_copy(
        update={
            "review_decisions": review_decisions,
            "approved_tools_digest": promoted_digest,
            "review_state": ReviewState.complete,
            "summary": (
                f"Seleccion aprobada y promovida a blueprint.tools con {len(approved_tools)} tools canonicamente activas."
            ),
        }
    )
    record_tool_recommendation_artifact(
        db,
        session_id=session_id,
        blueprint_version_number=blueprint_version_number,
        stage=envelope.stage,
        source_action="approve_tools_selection",
        recommendation=promoted_recommendation,
    )
    touch_session(record, envelope.stage, envelope.status)
    write_validation(
        db,
        session_id=session_id,
        artifact_name="approved_tools_selection",
        status_value=envelope.status,
        missing_fields=[],
        warnings=envelope.warnings,
    )
    write_log(
        db,
        session_id=session_id,
        stage=envelope.stage,
        status_value=envelope.status,
        message="Seleccion de tools promovida a blueprint.tools",
        payload={
            "approved_tool_keys": promoted_digest.approved_tool_keys,
            "digest_sha256": promoted_digest.digest_sha256,
            "promoted_blueprint_version": blueprint_version_number,
        },
    )
    db.add(record)
    sync_short_term_memory_checkpoint(db, record=record, source_action="approve_tools_selection")
    capture_operational_state(db, session_id=session_id, source_action="approve_tools_selection")
    db.commit()
    return build_snapshot(db, record)


@router.post("/{session_id}/recommend-memory", response_model=JourneyStageArtifactEntry)
def recommend_memory_route(
    session_id: UUID,
    payload: MemoryRecommendationRequest | None = None,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> JourneyStageArtifactEntry:
    record = get_or_404(db, session_id, current_user.id)
    ensure_route_feature_flag_enabled(
        db,
        workspace_id=record.workspace_id,
        flag_key=FEATURE_FLAG_MEMORY_HYBRID_EXTENDED_JOURNEY,
        detail="Memory hybrid journey feature flag is disabled",
    )
    JourneyStageMigrationService().backfill_session(db, session_record=record)
    proposal_service = StageProposalService()
    latest_discover_artifact = proposal_service.latest(db, session_record=record, stage_key="discover")
    latest_define_artifact = proposal_service.latest(db, session_record=record, stage_key="define")
    latest_design_artifact = proposal_service.latest(db, session_record=record, stage_key="design")
    latest_tools_artifact = proposal_service.latest(db, session_record=record, stage_key="tools")
    if not is_discovery_stage_approved(latest_discover_artifact):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Discover must be approved before Memory")
    if not is_define_stage_approved(latest_define_artifact):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Define must be approved before Memory")
    if not is_design_stage_approved(latest_design_artifact):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Design must be approved before Memory")
    if not is_tools_stage_approved(latest_tools_artifact):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Tools must be approved before Memory")

    opportunity = db.exec(select(OpportunityRecord).where(OpportunityRecord.session_id == session_id)).first()
    canvas_record = db.exec(select(CanvasRecord).where(CanvasRecord.session_id == session_id)).first()
    blueprint_record = db.exec(select(BlueprintRecord).where(BlueprintRecord.session_id == session_id)).first()
    if opportunity is None or canvas_record is None or blueprint_record is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Discovery, canvas and blueprint must exist before recommending memory",
        )

    discovery = hydrate_discovery(opportunity)
    canvas = hydrate_canvas(canvas_record)
    blueprint = hydrate_blueprint(blueprint_record)
    blueprint_version_number = latest_blueprint_version_number(db, session_id)
    latest_recommendation = load_latest_tool_recommendation(
        db,
        session_id,
        discovery=discovery,
        canvas=canvas,
        blueprint=blueprint,
        current_blueprint_version_number=blueprint_version_number,
    )
    if latest_recommendation is None or latest_recommendation.approved_tools_digest is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Memory requires an approved tools digest before generating the recommendation",
        )
    if latest_recommendation.is_stale:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Memory requires regenerating Tools after the design context changed",
        )

    definition_artifact = (
        resolve_definition_from_stage_artifact(latest_define_artifact) if latest_define_artifact is not None else None
    )
    design_artifact = (
        resolve_design_from_stage_artifact(latest_design_artifact) if latest_design_artifact is not None else None
    )
    if design_artifact is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The approved design proposal could not be resolved before Memory",
        )

    source_stage_versions = MemoryRecommendationSourceStageVersions(
        discover=latest_discover_artifact.version_number if latest_discover_artifact is not None else None,
        define=latest_define_artifact.version_number if latest_define_artifact is not None else None,
        design=latest_design_artifact.version_number if latest_design_artifact is not None else None,
        tools=latest_tools_artifact.version_number if latest_tools_artifact is not None else None,
    )
    runtime_settings = load_effective_runtime_settings(db, record.workspace_id)
    proposal_stage_context = build_stage_context_bundle(
        db,
        record=record,
        capability="recommend_memory_architecture",
        stage="memory",
        effective_language=current_user.preferred_language,
        task_source_keys=["memory_architecture_input"],
        allow_second_page=True,
    )
    critique_stage_context = build_stage_context_bundle(
        db,
        record=record,
        capability="critique_memory_architecture",
        stage="memory",
        effective_language=current_user.preferred_language,
        task_source_keys=["memory_architecture_critique_input"],
        allow_second_page=True,
    )
    session_snapshot = build_snapshot(db, record)
    artifact, traces = run_memory_recommendation_stage(
        session_id=session_id,
        discovery=discovery,
        canvas=canvas,
        blueprint=blueprint,
        definition_artifact=definition_artifact,
        design_artifact=design_artifact,
        approved_tools_digest=latest_recommendation.approved_tools_digest,
        session_snapshot=session_snapshot,
        instructions=payload.instructions if payload is not None else "",
        blueprint_version_number=blueprint_version_number,
        source_stage_versions=source_stage_versions,
        runtime_settings=runtime_settings,
        proposal_stage_context=proposal_stage_context,
        critique_stage_context=critique_stage_context,
    )
    stage_trace = next((trace.llm_trace for trace in reversed(traces) if trace.llm_trace is not None), None)
    combined_warnings = list(
        dict.fromkeys(
            [
                warning
                for trace in traces
                for warning in trace.warnings
                if warning
            ]
        )
    )
    try:
        journey_artifact = proposal_service.create(
            db,
            session_record=record,
            stage_key="memory",
            payload=JourneyStageArtifactCreateRequest(
                artifact_kind="memory_recommendation_artifact",
                source_action="recommend_memory",
                proposal_payload=artifact.model_dump(mode="json"),
                provider_key=stage_trace.provider_key if stage_trace is not None else "",
                model=stage_trace.model_name if stage_trace is not None else "",
                execution_backend=stage_trace.execution_backend if stage_trace is not None else "",
                prompt_version=stage_trace.prompt_version if stage_trace is not None else "",
                schema_version="memory-recommendation.v1",
                confidence=artifact.confidence.overall,
                source_stage_versions=source_stage_versions.model_dump(mode="json"),
                missing_information=list(dict.fromkeys(artifact.missing_information)),
                warnings=combined_warnings,
                corpus_hash=proposal_stage_context.corpus_hash or critique_stage_context.corpus_hash,
                evidence_manifest=build_journey_evidence_manifest_from_traces(
                    traces=traces,
                    source_action="recommend_memory",
                ),
            ),
            actor_user=current_user,
        )
    except Exception as exc:  # noqa: BLE001
        _raise_stage_proposal_http_error(exc)

    write_skill_runs(
        db,
        session_id=session_id,
        traces=traces,
        source_action="recommend_memory",
        blueprint_version_number=blueprint_version_number,
    )
    write_validation(
        db,
        session_id=session_id,
        artifact_name="memory_recommendation",
        status_value=ArtifactStatus.ready if artifact.review_state == ReviewState.complete else ArtifactStatus.needs_review,
        missing_fields=list(artifact.missing_information),
        warnings=combined_warnings,
    )
    write_log(
        db,
        session_id=session_id,
        stage=SessionStage.build_blueprint,
        status_value=ArtifactStatus.ready if artifact.review_state == ReviewState.complete else ArtifactStatus.needs_review,
        message="Recomendacion de memoria generada",
        payload=artifact.model_dump(mode="json"),
    )
    db.add(record)
    sync_short_term_memory_checkpoint(db, record=record, source_action="recommend_memory")
    capture_operational_state(db, session_id=session_id, source_action="recommend_memory")
    db.commit()
    return journey_artifact


@router.post("/{session_id}/approve-memory-profile", response_model=SessionSnapshot)
def approve_memory_profile_route(
    session_id: UUID,
    payload: JourneyStageArtifactApprovalRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> SessionSnapshot:
    record = get_or_404(db, session_id, current_user.id)
    ensure_route_feature_flag_enabled(
        db,
        workspace_id=record.workspace_id,
        flag_key=FEATURE_FLAG_MEMORY_HYBRID_EXTENDED_JOURNEY,
        detail="Memory hybrid journey feature flag is disabled",
    )
    JourneyStageMigrationService().backfill_session(db, session_record=record)
    proposal_service = StageProposalService()
    latest_tools_artifact = proposal_service.latest(db, session_record=record, stage_key="tools")
    if not is_tools_stage_approved(latest_tools_artifact):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Tools must be approved before Memory")
    latest_memory_artifact = proposal_service.latest(db, session_record=record, stage_key="memory")
    if latest_memory_artifact is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A memory recommendation must exist before approving the memory profile",
        )
    opportunity = db.exec(select(OpportunityRecord).where(OpportunityRecord.session_id == session_id)).first()
    canvas_record = db.exec(select(CanvasRecord).where(CanvasRecord.session_id == session_id)).first()
    blueprint_record = db.exec(select(BlueprintRecord).where(BlueprintRecord.session_id == session_id)).first()
    if opportunity is None or canvas_record is None or blueprint_record is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Discovery, canvas and blueprint must exist before approving the memory profile",
        )
    discovery = hydrate_discovery(opportunity)
    canvas = hydrate_canvas(canvas_record)
    blueprint = hydrate_blueprint(blueprint_record)
    blueprint_version_number = latest_blueprint_version_number(db, session_id)
    latest_tool_recommendation = load_latest_tool_recommendation(
        db,
        session_id,
        discovery=discovery,
        canvas=canvas,
        blueprint=blueprint,
        current_blueprint_version_number=blueprint_version_number,
    )
    if latest_tool_recommendation is None or latest_tool_recommendation.approved_tools_digest is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Memory requires an approved tools digest before approval",
        )
    latest_define_artifact = proposal_service.latest(db, session_record=record, stage_key="define")
    latest_design_artifact = proposal_service.latest(db, session_record=record, stage_key="design")
    if latest_memory_artifact.schema_version == "memory-recommendation.v1":
        memory_payload = latest_memory_artifact.proposal_payload
        proposed_memory_profile = memory_payload.get("proposed_memory_profile")
        proposed_knowledge_profile = memory_payload.get("proposed_knowledge_profile")
        if isinstance(proposed_memory_profile, dict) and isinstance(proposed_knowledge_profile, dict):
            refreshed_artifact = build_memory_recommendation_artifact(
                discovery=discovery,
                canvas=canvas,
                blueprint=blueprint,
                approved_tools_digest=latest_tool_recommendation.approved_tools_digest,
                source_session_id=session_id,
                source_blueprint_version=memory_payload.get("source_blueprint_version") or blueprint_version_number,
                current_blueprint_version=blueprint_version_number,
                source_stage_versions=MemoryRecommendationSourceStageVersions.model_validate(
                    memory_payload.get("source_stage_versions") or {}
                ),
                instructions=str(memory_payload.get("generation_instructions", "") or ""),
                definition_artifact=(
                    resolve_definition_from_stage_artifact(latest_define_artifact)
                    if latest_define_artifact is not None
                    else None
                ),
                design_artifact=(
                    resolve_design_from_stage_artifact(latest_design_artifact)
                    if latest_design_artifact is not None
                    else None
                ),
                session_snapshot=build_snapshot(db, record),
                proposed_memory_profile=MemoryProfile.model_validate(proposed_memory_profile),
                proposed_knowledge_profile=KnowledgeProfile.model_validate(proposed_knowledge_profile),
            )
            try:
                proposal_service.patch(
                    db,
                    session_record=record,
                    stage_key="memory",
                    artifact_id=latest_memory_artifact.id,
                    payload=JourneyStageArtifactPatchRequest(
                        note="Refresh Memory artifact before approval.",
                        proposal_payload=refreshed_artifact.model_dump(mode="json"),
                    ),
                    actor_user=current_user,
                )
                latest_memory_artifact = proposal_service.latest(db, session_record=record, stage_key="memory")
            except Exception as exc:  # noqa: BLE001
                _raise_stage_proposal_http_error(exc)
    try:
        proposal_service.approve(
            db,
            session_record=record,
            stage_key="memory",
            artifact_id=latest_memory_artifact.id,
            payload=payload,
            actor_user=current_user,
        )
    except Exception as exc:  # noqa: BLE001
        _raise_stage_proposal_http_error(exc)
    sync_short_term_memory_checkpoint(db, record=record, source_action="approve_memory_profile")
    capture_operational_state(db, session_id=session_id, source_action="approve_memory_profile")
    db.commit()
    return build_snapshot(db, record)


@router.post("/{session_id}/generate-validation-scenarios", response_model=JourneyStageArtifactEntry)
def generate_validation_scenarios_route(
    session_id: UUID,
    payload: ValidationScenarioGenerationRequest | None = None,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> JourneyStageArtifactEntry:
    record = get_or_404(db, session_id, current_user.id)
    JourneyStageMigrationService().backfill_session(db, session_record=record)
    proposal_service = StageProposalService()
    latest_discover_artifact = proposal_service.latest(db, session_record=record, stage_key="discover")
    latest_define_artifact = proposal_service.latest(db, session_record=record, stage_key="define")
    latest_design_artifact = proposal_service.latest(db, session_record=record, stage_key="design")
    latest_tools_artifact = proposal_service.latest(db, session_record=record, stage_key="tools")
    latest_memory_artifact = proposal_service.latest(db, session_record=record, stage_key="memory")
    if not is_discovery_stage_approved(latest_discover_artifact):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Discover must be approved before Validate")
    if not is_define_stage_approved(latest_define_artifact):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Define must be approved before Validate")
    if not is_design_stage_approved(latest_design_artifact):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Design must be approved before Validate")
    if not is_tools_stage_approved(latest_tools_artifact):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Tools must be approved before Validate")
    if latest_memory_artifact is None or latest_memory_artifact.state not in {
        JourneyArtifactState.approved,
        JourneyArtifactState.approved_legacy,
    }:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Memory must be approved before Validate")

    opportunity = db.exec(select(OpportunityRecord).where(OpportunityRecord.session_id == session_id)).first()
    canvas_record = db.exec(select(CanvasRecord).where(CanvasRecord.session_id == session_id)).first()
    blueprint_record = db.exec(select(BlueprintRecord).where(BlueprintRecord.session_id == session_id)).first()
    if canvas_record is None or blueprint_record is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Canvas and blueprint must exist before generating validation scenarios",
        )

    discovery = hydrate_discovery(opportunity) if opportunity is not None else None
    canvas = hydrate_canvas(canvas_record)
    blueprint = hydrate_blueprint(blueprint_record)
    definition_artifact = (
        resolve_definition_from_stage_artifact(latest_define_artifact) if latest_define_artifact is not None else None
    )
    blueprint_version_number = latest_blueprint_version_number(db, session_id)
    runtime_settings = load_effective_runtime_settings(db, record.workspace_id)
    stage_context = build_stage_context_bundle(
        db,
        record=record,
        capability="generate_validation_scenarios",
        stage="validate",
        effective_language=current_user.preferred_language,
        task_source_keys=["validation_scenario_generation_input"],
        allow_second_page=True,
    )
    source_stage_versions = {
        "discover": latest_discover_artifact.version_number if latest_discover_artifact is not None else None,
        "define": latest_define_artifact.version_number if latest_define_artifact is not None else None,
        "design": latest_design_artifact.version_number if latest_design_artifact is not None else None,
        "tools": latest_tools_artifact.version_number if latest_tools_artifact is not None else None,
        "memory": latest_memory_artifact.version_number if latest_memory_artifact is not None else None,
    }
    artifact, traces = build_validation_simulation_specification(
        discovery=discovery,
        canvas=canvas,
        blueprint=blueprint,
        definition_artifact=definition_artifact,
        session_snapshot=build_snapshot(db, record),
        blueprint_version_number=blueprint_version_number,
        source_stage_versions=source_stage_versions,
        instructions=payload.instructions if payload is not None else "",
        runtime_settings=runtime_settings,
        stage_context=stage_context,
    )
    stage_trace = next((trace.llm_trace for trace in reversed(traces) if trace.llm_trace is not None), None)
    combined_warnings = list(dict.fromkeys([warning for trace in traces for warning in trace.warnings if warning]))
    try:
        journey_artifact = proposal_service.create(
            db,
            session_record=record,
            stage_key="validate",
            payload=JourneyStageArtifactCreateRequest(
                artifact_kind="validation_simulation_artifact",
                source_action="generate_validation_scenarios",
                proposal_payload=artifact.model_dump(mode="json"),
                provider_key=stage_trace.provider_key if stage_trace is not None else "",
                model=stage_trace.model_name if stage_trace is not None else "",
                execution_backend=stage_trace.execution_backend if stage_trace is not None else "",
                prompt_version=stage_trace.prompt_version if stage_trace is not None else "",
                schema_version="validation-simulation-spec.v1",
                confidence=artifact.confidence,
                source_stage_versions=source_stage_versions,
                missing_information=list(artifact.missing_information),
                warnings=combined_warnings,
                corpus_hash=stage_context.corpus_hash,
                evidence_manifest=build_journey_evidence_manifest_from_traces(
                    traces=traces,
                    source_action="generate_validation_scenarios",
                ),
            ),
            actor_user=current_user,
        )
    except Exception as exc:  # noqa: BLE001
        _raise_stage_proposal_http_error(exc)

    write_skill_runs(
        db,
        session_id=session_id,
        traces=traces,
        source_action="generate_validation_scenarios",
        blueprint_version_number=blueprint_version_number,
    )
    write_validation(
        db,
        session_id=session_id,
        artifact_name="validation_simulation_specification",
        status_value=ArtifactStatus.ready if artifact.review_state == ReviewState.complete else ArtifactStatus.needs_review,
        missing_fields=list(artifact.missing_information),
        warnings=combined_warnings,
    )
    write_log(
        db,
        session_id=session_id,
        stage=SessionStage.post_validation,
        status_value=ArtifactStatus.ready if artifact.review_state == ReviewState.complete else ArtifactStatus.needs_review,
        message="Escenarios de validacion generados",
        payload=artifact.model_dump(mode="json"),
    )
    db.add(record)
    sync_short_term_memory_checkpoint(db, record=record, source_action="generate_validation_scenarios")
    capture_operational_state(db, session_id=session_id, source_action="generate_validation_scenarios")
    db.commit()
    return journey_artifact


@router.post("/{session_id}/approve-validation-scenarios", response_model=SessionSnapshot)
def approve_validation_scenarios_route(
    session_id: UUID,
    payload: JourneyStageArtifactApprovalRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> SessionSnapshot:
    record = get_or_404(db, session_id, current_user.id)
    JourneyStageMigrationService().backfill_session(db, session_record=record)
    proposal_service = StageProposalService()
    latest_validate_artifact = proposal_service.latest(db, session_record=record, stage_key="validate")
    specification = resolve_validation_specification_from_stage_artifact(latest_validate_artifact)
    if latest_validate_artifact is None or specification is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A validation simulation specification must exist before approval",
        )
    try:
        proposal_service.approve(
            db,
            session_record=record,
            stage_key="validate",
            artifact_id=latest_validate_artifact.id,
            payload=payload,
            actor_user=current_user,
        )
    except Exception as exc:  # noqa: BLE001
        _raise_stage_proposal_http_error(exc)
    write_validation(
        db,
        session_id=session_id,
        artifact_name="validation_scenarios_approved",
        status_value=ArtifactStatus.ready,
        missing_fields=[],
        warnings=list(specification.warnings),
    )
    write_log(
        db,
        session_id=session_id,
        stage=SessionStage.post_validation,
        status_value=ArtifactStatus.ready,
        message="Escenarios de validacion aprobados",
        payload={"scenario_keys": [item.scenario_key for item in specification.scenarios]},
    )
    sync_short_term_memory_checkpoint(db, record=record, source_action="approve_validation_scenarios")
    capture_operational_state(db, session_id=session_id, source_action="approve_validation_scenarios")
    db.commit()
    return build_snapshot(db, record)


@router.post("/{session_id}/run-validation-simulation", response_model=SimulationRunRecord)
def run_validation_simulation_route(
    session_id: UUID,
    payload: ValidationSimulationRunRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> SimulationRunRecord:
    record = get_or_404(db, session_id, current_user.id)
    JourneyStageMigrationService().backfill_session(db, session_record=record)
    proposal_service = StageProposalService()
    latest_validate_artifact = proposal_service.latest(db, session_record=record, stage_key="validate")
    specification = resolve_validation_specification_from_stage_artifact(latest_validate_artifact)
    if latest_validate_artifact is None or specification is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Generate validation scenarios before executing a simulation",
        )
    if latest_validate_artifact.state in {JourneyArtifactState.stale, JourneyArtifactState.rejected}:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The current validation specification is stale or rejected; regenerate Validate before simulating",
        )

    blueprint_record = db.exec(select(BlueprintRecord).where(BlueprintRecord.session_id == session_id)).first()
    if blueprint_record is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Blueprint must exist before simulation")
    blueprint = hydrate_blueprint(blueprint_record)
    blueprint_version_number = latest_blueprint_version_number(db, session_id)
    scenario = next((item for item in specification.scenarios if item.scenario_key == payload.scenario_key), None)
    if scenario is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Validation scenario not found")

    runtime_settings = load_effective_runtime_settings(db, record.workspace_id)
    stage_context = build_stage_context_bundle(
        db,
        record=record,
        capability="simulate_validation_scenario",
        stage="validate",
        effective_language=current_user.preferred_language,
        task_source_keys=["validation_scenario_simulation_input"],
        allow_second_page=True,
    )
    generated_run, traces = execute_validation_simulation(
        blueprint=blueprint,
        scenario=scenario,
        request=payload,
        blueprint_version_number=blueprint_version_number,
        specification_artifact_id=latest_validate_artifact.id,
        scenario_version_number=latest_validate_artifact.version_number,
        runtime_settings=runtime_settings,
        stage_context=stage_context,
        source_action="run_validation_simulation",
    )
    db_run = persist_validation_simulation_run(db, session_id=session_id, run_record=generated_run)
    persisted = build_validation_simulation_run_entry(
        db_run,
        latest_blueprint_version_number=blueprint_version_number,
        latest_validate_artifact_id=latest_validate_artifact.id,
    )
    combined_warnings = list(dict.fromkeys([warning for trace in traces for warning in trace.warnings if warning]))
    write_skill_runs(
        db,
        session_id=session_id,
        traces=traces,
        source_action="run_validation_simulation",
        blueprint_version_number=blueprint_version_number,
    )
    write_validation(
        db,
        session_id=session_id,
        artifact_name="validation_simulation_run",
        status_value=persisted.status,
        missing_fields=[],
        warnings=combined_warnings,
    )
    write_log(
        db,
        session_id=session_id,
        stage=SessionStage.post_validation,
        status_value=persisted.status,
        message="Simulacion de validacion ejecutada",
        payload={
            "run_id": str(db_run.id),
            "scenario_key": persisted.scenario_key,
            "hard_gate_status": persisted.hard_gate_status,
            "injected_conditions": persisted.injected_conditions,
        },
    )
    touch_session(record, SessionStage.post_validation, persisted.status)
    db.add(record)
    sync_short_term_memory_checkpoint(db, record=record, source_action="run_validation_simulation")
    capture_operational_state(db, session_id=session_id, source_action="run_validation_simulation")
    db.commit()
    return build_validation_simulation_run_entry(
        db_run,
        latest_blueprint_version_number=blueprint_version_number,
        latest_validate_artifact_id=latest_validate_artifact.id,
    )


@router.post("/{session_id}/inject-validation-event", response_model=SimulationRunRecord)
def inject_validation_event_route(
    session_id: UUID,
    payload: ValidationSimulationEventInjectionRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> SimulationRunRecord:
    record = get_or_404(db, session_id, current_user.id)
    JourneyStageMigrationService().backfill_session(db, session_record=record)
    proposal_service = StageProposalService()
    latest_validate_artifact = proposal_service.latest(db, session_record=record, stage_key="validate")
    specification = resolve_validation_specification_from_stage_artifact(latest_validate_artifact)
    if latest_validate_artifact is None or specification is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Validate must exist before fault injection")
    source_run = db.exec(
        select(ValidationSimulationRunStateRecord).where(
            ValidationSimulationRunStateRecord.id == payload.run_id,
            ValidationSimulationRunStateRecord.session_id == session_id,
        )
    ).first()
    if source_run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Validation simulation run not found")
    source_run_entry = build_validation_simulation_run_entry(
        source_run,
        latest_blueprint_version_number=latest_blueprint_version_number(db, session_id),
        latest_validate_artifact_id=latest_validate_artifact.id,
    )
    if source_run_entry.is_stale:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The selected simulation run is stale; regenerate Validate or rerun the scenario first",
        )

    blueprint_record = db.exec(select(BlueprintRecord).where(BlueprintRecord.session_id == session_id)).first()
    if blueprint_record is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Blueprint must exist before fault injection")
    blueprint = hydrate_blueprint(blueprint_record)
    scenario = next((item for item in specification.scenarios if item.scenario_key == source_run.scenario_key), None)
    if scenario is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Validation scenario not found for the run")

    initial_input = next((item.detail for item in source_run_entry.events if item.event_type == "input"), scenario.initial_input)
    injected_conditions = list(dict.fromkeys([*source_run_entry.injected_conditions, payload.injection_type]))
    runtime_settings = load_effective_runtime_settings(db, record.workspace_id)
    stage_context = build_stage_context_bundle(
        db,
        record=record,
        capability="simulate_validation_scenario",
        stage="validate",
        effective_language=current_user.preferred_language,
        task_source_keys=["validation_scenario_simulation_input"],
        allow_second_page=True,
    )
    generated_run, traces = execute_validation_simulation(
        blueprint=blueprint,
        scenario=scenario,
        request=ValidationSimulationRunRequest(
            scenario_key=scenario.scenario_key,
            scenario_version_number=source_run.scenario_version_number,
            initial_input_override=initial_input,
            injected_conditions=injected_conditions,
        ),
        blueprint_version_number=latest_blueprint_version_number(db, session_id),
        specification_artifact_id=latest_validate_artifact.id,
        scenario_version_number=latest_validate_artifact.version_number,
        runtime_settings=runtime_settings,
        stage_context=stage_context,
        source_action="inject_validation_event",
    )
    db_run = persist_validation_simulation_run(db, session_id=session_id, run_record=generated_run)
    combined_warnings = list(dict.fromkeys([warning for trace in traces for warning in trace.warnings if warning]))
    write_skill_runs(
        db,
        session_id=session_id,
        traces=traces,
        source_action="inject_validation_event",
        blueprint_version_number=latest_blueprint_version_number(db, session_id),
    )
    write_validation(
        db,
        session_id=session_id,
        artifact_name="validation_simulation_injection",
        status_value=db_run.status,
        missing_fields=[],
        warnings=combined_warnings,
    )
    write_log(
        db,
        session_id=session_id,
        stage=SessionStage.post_validation,
        status_value=db_run.status,
        message="Fallo inyectado sobre simulacion de validacion",
        payload={
            "source_run_id": str(source_run.id),
            "new_run_id": str(db_run.id),
            "scenario_key": db_run.scenario_key,
            "injection_type": payload.injection_type,
            "note": payload.note,
        },
    )
    touch_session(record, SessionStage.post_validation, db_run.status)
    db.add(record)
    sync_short_term_memory_checkpoint(db, record=record, source_action="inject_validation_event")
    capture_operational_state(db, session_id=session_id, source_action="inject_validation_event")
    db.commit()
    return build_validation_simulation_run_entry(
        db_run,
        latest_blueprint_version_number=latest_blueprint_version_number(db, session_id),
        latest_validate_artifact_id=latest_validate_artifact.id,
    )


@router.post("/{session_id}/judge-validation-run", response_model=SimulationRunRecord)
def judge_validation_run_route(
    session_id: UUID,
    payload: ValidationSimulationJudgeRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> SimulationRunRecord:
    record = get_or_404(db, session_id, current_user.id)
    JourneyStageMigrationService().backfill_session(db, session_record=record)
    proposal_service = StageProposalService()
    latest_validate_artifact = proposal_service.latest(db, session_record=record, stage_key="validate")
    specification = resolve_validation_specification_from_stage_artifact(latest_validate_artifact)
    if latest_validate_artifact is None or specification is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Validate must exist before judging a run")
    db_run = db.exec(
        select(ValidationSimulationRunStateRecord).where(
            ValidationSimulationRunStateRecord.id == payload.run_id,
            ValidationSimulationRunStateRecord.session_id == session_id,
        )
    ).first()
    if db_run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Validation simulation run not found")
    blueprint_record = db.exec(select(BlueprintRecord).where(BlueprintRecord.session_id == session_id)).first()
    if blueprint_record is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Blueprint must exist before judging a run")
    current_blueprint_version_number = latest_blueprint_version_number(db, session_id)
    run_entry = build_validation_simulation_run_entry(
        db_run,
        latest_blueprint_version_number=current_blueprint_version_number,
        latest_validate_artifact_id=latest_validate_artifact.id,
    )
    if run_entry.is_stale:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The selected simulation run is stale; rerun the scenario on the current Validate specification",
        )
    scenario = next((item for item in specification.scenarios if item.scenario_key == db_run.scenario_key), None)
    if scenario is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Validation scenario not found for the run")

    runtime_settings = load_effective_runtime_settings(db, record.workspace_id)
    stage_context = build_stage_context_bundle(
        db,
        record=record,
        capability="judge_validation_run",
        stage="validate",
        effective_language=current_user.preferred_language,
        task_source_keys=["validation_run_judgment_input"],
        allow_second_page=True,
    )
    judgement, traces = judge_validation_simulation_run(
        run_record=run_entry,
        blueprint=hydrate_blueprint(blueprint_record),
        scenario=scenario,
        runtime_settings=runtime_settings,
        stage_context=stage_context,
    )
    db_run.final_status = judgement.final_status
    db_run.status = ArtifactStatus.ready if judgement.final_status == "pass" else ArtifactStatus.needs_review
    db_run.summary = judgement.summary
    db_run.judgement = judgement.model_dump(mode="json")
    db_run.updated_at = utc_now()
    db.add(db_run)
    combined_warnings = list(dict.fromkeys([warning for trace in traces for warning in trace.warnings if warning]))
    write_skill_runs(
        db,
        session_id=session_id,
        traces=traces,
        source_action="judge_validation_run",
        blueprint_version_number=current_blueprint_version_number,
    )
    write_validation(
        db,
        session_id=session_id,
        artifact_name="validation_run_judgement",
        status_value=db_run.status,
        missing_fields=[],
        warnings=combined_warnings,
    )
    write_log(
        db,
        session_id=session_id,
        stage=SessionStage.post_validation,
        status_value=db_run.status,
        message="Juicio de simulacion de validacion emitido",
        payload={
            "run_id": str(db_run.id),
            "scenario_key": db_run.scenario_key,
            "hard_gate_status": run_entry.hard_gate_status,
            "final_status": judgement.final_status,
            "score": judgement.score,
        },
    )
    touch_session(record, SessionStage.post_validation, db_run.status)
    db.add(record)
    sync_short_term_memory_checkpoint(db, record=record, source_action="judge_validation_run")
    capture_operational_state(db, session_id=session_id, source_action="judge_validation_run")
    db.commit()
    return build_validation_simulation_run_entry(
        db_run,
        latest_blueprint_version_number=current_blueprint_version_number,
        latest_validate_artifact_id=latest_validate_artifact.id,
    )


@router.post("/{session_id}/evaluate", response_model=EvaluationEnvelope)
def evaluate_blueprint_route(
    session_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> EvaluationEnvelope:
    record = get_or_404(db, session_id, current_user.id)
    opportunity = db.exec(select(OpportunityRecord).where(OpportunityRecord.session_id == session_id)).first()
    canvas = db.exec(select(CanvasRecord).where(CanvasRecord.session_id == session_id)).first()
    blueprint = db.exec(select(BlueprintRecord).where(BlueprintRecord.session_id == session_id)).first()

    discovery_artifact = opportunity and hydrate_discovery(opportunity)
    canvas_artifact = canvas and hydrate_canvas(canvas)
    blueprint_artifact = blueprint and hydrate_blueprint(blueprint)
    blueprint_version_number = latest_blueprint_version_number(db, session_id)
    evaluation_dataset, evaluation_rubric = ensure_evaluation_workbench_assets(
        db,
        session_id=session_id,
        discovery=discovery_artifact,
        canvas=canvas_artifact,
        blueprint=blueprint_artifact,
        blueprint_version_number=blueprint_version_number,
    )
    dataset_record = latest_evaluation_dataset(db, session_id)
    rubric_record = latest_evaluation_rubric(db, session_id)
    if dataset_record is None or rubric_record is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Evaluation assets not available")

    runtime_settings = load_effective_runtime_settings(db, record.workspace_id)
    envelope, traces = run_evaluation_stage(
        discovery_artifact,
        canvas_artifact,
        blueprint_artifact,
        evaluation_dataset,
        evaluation_rubric,
        runtime_settings=runtime_settings,
    )
    pending_approvals = count_pending_approvals(db, session_id)
    envelope = apply_pending_approvals_to_evaluation(envelope, pending_approvals)
    run_summary = score_evaluation_workbench(
        evaluation_dataset,
        evaluation_rubric,
        discovery_artifact,
        canvas_artifact,
        blueprint_artifact,
        source_action="evaluate_blueprint",
    )
    run_summary = apply_pending_approvals_to_run_summary(
        run_summary,
        pending_approvals,
        blueprint_version_number=blueprint_version_number,
    )
    persist_evaluation_run(
        db,
        session_id=session_id,
        dataset_record=dataset_record,
        rubric_record=rubric_record,
        run_summary=run_summary,
    )
    sync_evaluation_handoff(
        db,
        session_record=record,
        blueprint_version_number=blueprint_version_number,
        source_action="evaluate_blueprint",
        overall_score=run_summary.overall_score,
        status=run_summary.status,
    )
    upsert_evaluation(db, session_id, envelope)
    touch_session(record, SessionStage.post_validation, envelope.status)
    write_skill_runs(
        db,
        session_id=session_id,
        traces=traces,
        source_action="evaluate_blueprint",
        blueprint_version_number=blueprint_version_number,
    )
    write_validation(
        db,
        session_id=session_id,
        artifact_name="evaluation",
        status_value=envelope.status,
        missing_fields=envelope.missing_fields,
        warnings=envelope.warnings,
    )
    write_log(
        db,
        session_id=session_id,
        stage=envelope.stage,
        status_value=envelope.status,
        message="Evaluacion generada",
        payload=envelope.model_dump(mode="json"),
    )
    db.add(record)
    sync_short_term_memory_checkpoint(db, record=record, source_action="evaluate_blueprint")
    capture_operational_state(db, session_id=session_id, source_action="evaluate_blueprint")
    db.commit()
    return envelope


@router.post("/{session_id}/estimate", response_model=EstimationEnvelope)
def generate_estimation_report_route(
    session_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> EstimationEnvelope:
    record = get_or_404(db, session_id, current_user.id)
    ensure_route_feature_flag_enabled(
        db,
        workspace_id=record.workspace_id,
        flag_key=FEATURE_FLAG_ESTIMATION,
        detail="Comparative estimation feature flag is disabled",
    )
    snapshot = build_snapshot(db, record)
    if snapshot.canvas is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Canvas must exist before generating an estimation report",
        )

    acp_preview = load_latest_persisted_acp_preview(db, session_id)
    estimation_report = build_estimation_report(
        db,
        snapshot=snapshot,
        acp_preview=acp_preview,
    )
    blueprint_version_number = latest_blueprint_version_number(db, session_id)
    benchmark_corpus_hash = build_estimation_benchmark_corpus_hash(
        db,
        workspace_id=record.workspace_id,
        session_id=record.id,
    )
    estimation_report = estimation_report.model_copy(
        update={
            "blueprint_version_number": blueprint_version_number,
            "current_blueprint_version_number": blueprint_version_number,
            "is_stale": False,
            "stale_reasons": [],
        }
    )
    deterministic_inputs = build_estimation_deterministic_inputs(
        db,
        snapshot=snapshot,
        report=estimation_report,
        benchmark_ids=[],
        benchmark_corpus_hash=benchmark_corpus_hash,
    )
    estimation_report = estimation_report.model_copy(update={"deterministic_inputs": deterministic_inputs})
    stage_context = build_stage_context_bundle(
        db,
        record=record,
        capability="analyze_estimation_risks",
        stage="estimate",
        effective_language=current_user.preferred_language,
        task_source_keys=["estimation_risk_analysis_input"],
        allow_second_page=True,
    )
    analysis, trace = run_estimation_analysis(
        db,
        snapshot=snapshot,
        report=estimation_report,
        stage_context=stage_context,
    )
    deterministic_inputs = build_estimation_deterministic_inputs(
        db,
        snapshot=snapshot,
        report=estimation_report,
        benchmark_ids=[
            item.benchmark_key or item.source_ref
            for item in analysis.benchmark_refs
            if item.benchmark_key or item.source_ref
        ],
        benchmark_corpus_hash=benchmark_corpus_hash,
    )
    estimation_report = estimation_report.model_copy(update={"deterministic_inputs": deterministic_inputs})
    estimation_report = apply_estimation_analysis(
        estimation_report,
        analysis=analysis,
    )
    missing_fields: list[str] = []
    if snapshot.blueprint is None:
        missing_fields.append("blueprint")
    if acp_preview is None:
        missing_fields.append("acp_preview")

    status_value = (
        ArtifactStatus.ready
        if estimation_report.package_policy.can_continue_to_package
        else ArtifactStatus.needs_review
    )
    next_action = "review_estimation_report"
    if estimation_report.analysis_decision.decision == "pending" and estimation_report.analysis is not None:
        proposal = estimation_report.analysis.confidence_adjustment_proposal
        if proposal.proposed_score_delta != 0 or proposal.proposed_uncertainty_band_delta != 0:
            next_action = "review_estimation_adjustment"
    if estimation_report.package_policy.preliminary:
        next_action = "close_validate_before_package"
    envelope = EstimationEnvelope(
        status=status_value,
        stage=record.current_stage,
        data=estimation_report,
        missing_fields=missing_fields,
        assumptions=estimation_report.assumptions,
        warnings=list(
            dict.fromkeys(
                estimation_report.traditional.warnings
                + estimation_report.agentic.warnings
                + trace.warnings
                + estimation_report.notes
            )
        ),
        evidence=[
            EvidenceItem(source="rule_engine", detail="estimation_engine=deterministic_v1"),
            EvidenceItem(source="rule_engine", detail=f"maturity_stage={estimation_report.maturity_stage}"),
            EvidenceItem(source="rule_engine", detail=f"active_provider={estimation_report.agentic.active_provider}"),
            EvidenceItem(source="rule_engine", detail=f"confidence_score={estimation_report.confidence.score}"),
            *trace.evidence,
        ],
        llm_trace=trace.llm_trace.model_copy(deep=True) if trace.llm_trace is not None else None,
        next_action=(
            "build_blueprint"
            if estimation_report.maturity_stage == "canvas"
            else next_action
        ),
    )

    record_estimation_artifact(
        db,
        session_id=session_id,
        blueprint_version_number=blueprint_version_number,
        stage=record.current_stage,
        source_action="generate_estimation_report",
        estimation_report=estimation_report,
    )
    persist_estimation_run(
        db,
        session_id=session_id,
        blueprint_version_number=blueprint_version_number,
        source_action="generate_estimation_report",
        estimation_report=estimation_report,
    )
    write_skill_runs(
        db,
        session_id=session_id,
        traces=[trace],
        source_action="analyze_estimation_risks",
        blueprint_version_number=blueprint_version_number,
    )
    record.updated_at = utc_now()
    db.add(record)
    write_log(
        db,
        session_id=session_id,
        stage=record.current_stage,
        status_value=status_value,
        message="Estimacion comparativa generada",
        payload={
            "maturity_stage": estimation_report.maturity_stage,
            "confidence_score": estimation_report.confidence.score,
            "active_provider": estimation_report.agentic.active_provider,
            "traditional_hours": estimation_report.traditional.estimated_hours_total,
            "agentic_hours": estimation_report.agentic.estimated_hours_total,
            "package_ready": estimation_report.package_policy.can_continue_to_package,
            "preliminary": estimation_report.package_policy.preliminary,
        },
    )
    capture_operational_state(db, session_id=session_id, source_action="generate_estimation_report")
    db.commit()
    return envelope


@router.post("/{session_id}/estimate/analysis-decision", response_model=EstimationEnvelope)
def apply_estimation_analysis_decision_route(
    session_id: UUID,
    payload: EstimationAnalysisDecisionRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> EstimationEnvelope:
    record = get_or_404(db, session_id, current_user.id)
    ensure_route_feature_flag_enabled(
        db,
        workspace_id=record.workspace_id,
        flag_key=FEATURE_FLAG_ESTIMATION,
        detail="Comparative estimation feature flag is disabled",
    )
    snapshot = build_snapshot(db, record)
    estimation_report = snapshot.estimation_report
    if estimation_report is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An estimation report must exist before deciding the confidence adjustment",
        )
    if estimation_report.analysis is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The current estimation report does not contain an analysis proposal to review",
        )

    updated_report = apply_estimation_analysis_decision(
        estimation_report,
        decision=payload.decision,
        note=payload.note,
    )
    blueprint_version_number = latest_blueprint_version_number(db, session_id)
    record_estimation_artifact(
        db,
        session_id=session_id,
        blueprint_version_number=blueprint_version_number,
        stage=record.current_stage,
        source_action="apply_estimation_analysis_decision",
        estimation_report=updated_report,
    )
    status_value = ArtifactStatus.ready if updated_report.package_policy.can_continue_to_package else ArtifactStatus.needs_review
    record.updated_at = utc_now()
    db.add(record)
    write_validation(
        db,
        session_id=session_id,
        artifact_name="estimation_report",
        status_value=status_value,
        missing_fields=[],
        warnings=updated_report.package_policy.package_block_reasons,
    )
    write_log(
        db,
        session_id=session_id,
        stage=record.current_stage,
        status_value=status_value,
        message="Decision de ajuste de confianza aplicada",
        payload={
            "decision": payload.decision,
            "note": payload.note,
            "confidence_score": updated_report.confidence.score,
            "uncertainty_band_percent": updated_report.confidence.uncertainty_band_percent,
            "package_ready": updated_report.package_policy.can_continue_to_package,
        },
    )
    sync_short_term_memory_checkpoint(db, record=record, source_action="apply_estimation_analysis_decision")
    capture_operational_state(db, session_id=session_id, source_action="apply_estimation_analysis_decision")
    db.commit()
    return EstimationEnvelope(
        status=status_value,
        stage=record.current_stage,
        data=updated_report,
        missing_fields=[],
        assumptions=updated_report.assumptions,
        warnings=list(
            dict.fromkeys(
                [
                    *updated_report.package_policy.package_block_reasons,
                    *updated_report.notes,
                ]
            )
        ),
        evidence=[
            EvidenceItem(source="rule_engine", detail="decision=estimation_confidence_adjustment"),
        ],
        next_action="package_agent_construction" if updated_report.package_policy.can_continue_to_package else "review_estimation_report",
    )


@router.post("/{session_id}/estimate/actuals", response_model=ProjectActualsEntry)
def upsert_estimation_actuals_route(
    session_id: UUID,
    payload: EstimationActualsUpsertRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> ProjectActualsEntry:
    record = get_or_404(db, session_id, current_user.id)
    ensure_route_feature_flag_enabled(
        db,
        workspace_id=record.workspace_id,
        flag_key=FEATURE_FLAG_ESTIMATION,
        detail="Comparative estimation feature flag is disabled",
    )
    try:
        actuals_record, metric_record = upsert_project_actuals(
            db,
            session_id=session_id,
            current_user_id=current_user.id,
            payload=payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    touch_session(record, record.current_stage, record.status)
    db.add(record)
    write_log(
        db,
        session_id=session_id,
        stage=record.current_stage,
        status_value=record.status,
        message="Actuals de estimacion actualizados",
        payload={
            "estimation_run_id": str(payload.estimation_run_id),
            "delivery_mode": payload.delivery_mode,
            "actual_hours_total": actuals_record.actual_hours_total,
            "actual_duration_weeks": actuals_record.actual_duration_weeks,
            "actual_cost_total": actuals_record.actual_cost_total,
            "absolute_percentage_error_cost": metric_record.absolute_percentage_error_cost,
            "band_hit_overall": metric_record.band_hit_overall,
        },
    )
    capture_operational_state(db, session_id=session_id, source_action="upsert_estimation_actuals")
    db.commit()
    return build_project_actuals_entry(actuals_record)


@router.post("/{session_id}/skills/{skill_key}/rerun", response_model=SkillRerunResponse)
def rerun_skill_route(
    session_id: UUID,
    skill_key: str,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> SkillRerunResponse:
    record = get_or_404(db, session_id, current_user.id)
    opportunity = db.exec(select(OpportunityRecord).where(OpportunityRecord.session_id == session_id)).first()
    canvas_record = db.exec(select(CanvasRecord).where(CanvasRecord.session_id == session_id)).first()
    blueprint_record = db.exec(select(BlueprintRecord).where(BlueprintRecord.session_id == session_id)).first()

    discovery = hydrate_discovery(opportunity) if opportunity is not None else None
    canvas = hydrate_canvas(canvas_record) if canvas_record is not None else None
    blueprint = hydrate_blueprint(blueprint_record) if blueprint_record is not None else None
    evaluation_dataset_record = latest_evaluation_dataset(db, session_id)
    evaluation_rubric_record = latest_evaluation_rubric(db, session_id)
    evaluation_dataset = (
        hydrate_evaluation_dataset(db, evaluation_dataset_record)
        if evaluation_dataset_record is not None
        else None
    )
    evaluation_rubric = (
        hydrate_evaluation_rubric(evaluation_rubric_record)
        if evaluation_rubric_record is not None
        else None
    )
    if skill_key == "evaluation_skill" and (evaluation_dataset is None or evaluation_rubric is None):
        blueprint_version_number = latest_blueprint_version_number(db, session_id)
        evaluation_dataset, evaluation_rubric = ensure_evaluation_workbench_assets(
            db,
            session_id=session_id,
            discovery=discovery,
            canvas=canvas,
            blueprint=blueprint,
            blueprint_version_number=blueprint_version_number,
        )
        evaluation_dataset_record = latest_evaluation_dataset(db, session_id)
        evaluation_rubric_record = latest_evaluation_rubric(db, session_id)
    if skill_key == "tool_recommendation_skill":
        ensure_route_feature_flag_enabled(
            db,
            workspace_id=record.workspace_id,
            flag_key=FEATURE_FLAG_TOOL_RECOMMENDATION,
            detail="Tool recommendation feature flag is disabled",
        )
    if skill_key == "memory_design_skill":
        tools_llm_enabled = is_feature_flag_enabled(
            db,
            FEATURE_FLAG_TOOL_RECOMMENDATION,
            workspace_id=record.workspace_id,
        )
        if tools_llm_enabled:
            latest_recommendation = load_latest_tool_recommendation(
                db,
                session_id,
                discovery=discovery,
                canvas=canvas,
                blueprint=blueprint,
                current_blueprint_version_number=latest_blueprint_version_number(db, session_id),
            )
            if latest_recommendation is not None and latest_recommendation.approved_tools_digest is None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Memory requires an approved tools digest before rerunning the memory skill",
                )
            if latest_recommendation is not None and latest_recommendation.is_stale:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Memory requires regenerating Tools after the design context changed",
                )

    try:
        runtime_settings = load_effective_runtime_settings(db, record.workspace_id)
        trace, next_discovery, next_canvas, next_blueprint, next_evaluation = rerun_skill_for_session(
            skill_key,
            discovery=discovery,
            canvas=canvas,
            blueprint=blueprint,
            evaluation_dataset=evaluation_dataset,
            evaluation_rubric=evaluation_rubric,
            runtime_settings=runtime_settings,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    source_action = f"rerun:{skill_key}"
    resulting_stage = trace.stage
    resulting_status = trace.status
    blueprint_version_number = latest_blueprint_version_number(db, session_id)

    if next_discovery is not None:
        discovery_missing_fields = find_missing_discovery_fields(next_discovery.model_dump(mode="json"))
        discovery_envelope = DiscoveryEnvelope(
            status=ArtifactStatus.ready if not discovery_missing_fields else ArtifactStatus.needs_review,
            stage=SessionStage.normalize_discovery,
            data=next_discovery,
            missing_fields=discovery_missing_fields,
            assumptions=[],
            warnings=list(trace.warnings),
            evidence=list(trace.evidence),
            next_action="build_canvas" if not discovery_missing_fields else "collect_missing_fields",
        )
        upsert_opportunity(db, session_id, discovery_envelope)
        maybe_set_session_title(record, next_discovery)
        resulting_stage = discovery_envelope.stage
        resulting_status = discovery_envelope.status
        discovery = next_discovery

    if next_canvas is not None:
        canvas_missing_fields = find_missing_discovery_fields(
            discovery.model_dump(mode="json") if discovery is not None else {}
        )
        canvas_envelope = CanvasEnvelope(
            status=ArtifactStatus.ready if not canvas_missing_fields else ArtifactStatus.needs_review,
            stage=SessionStage.build_canvas,
            data=next_canvas,
            missing_fields=canvas_missing_fields,
            assumptions=[],
            warnings=list(trace.warnings),
            evidence=list(trace.evidence),
            next_action="build_blueprint" if not canvas_missing_fields else "review_canvas",
        )
        upsert_canvas(db, session_id, canvas_envelope)
        resulting_stage = canvas_envelope.stage
        resulting_status = canvas_envelope.status
        canvas = next_canvas

    if next_blueprint is not None:
        blueprint_envelope = BlueprintEnvelope(
            status=ArtifactStatus.ready
            if next_blueprint.readiness_state == "complete"
            else ArtifactStatus.needs_review,
            stage=SessionStage.post_validation,
            data=next_blueprint,
            missing_fields=[],
            assumptions=[],
            warnings=list(trace.warnings),
            evidence=list(trace.evidence),
            next_action="evaluate_blueprint"
            if next_blueprint.readiness_state == "complete"
            else "review_blueprint_details",
        )
        pending_approvals = sync_approval_gates(db, session_id, blueprint_envelope.data)
        blueprint_envelope = apply_pending_approvals_to_blueprint(blueprint_envelope, pending_approvals)
        upsert_blueprint(db, session_id, blueprint_envelope)
        blueprint_version_number = create_blueprint_version(
            db,
            session_id=session_id,
            source_action=source_action,
            status_value=blueprint_envelope.status,
            blueprint=blueprint_envelope.data,
        )
        record_delivery_artifacts(
            db,
            session_id=session_id,
            blueprint_version_number=blueprint_version_number,
            source_action=source_action,
            stage=blueprint_envelope.stage,
            blueprint=blueprint_envelope.data,
        )
        if not record.selected_workflow_template_key:
            record.selected_workflow_template_key = recommend_workflow_template_key(blueprint_envelope.data)
        sync_governance_handoff(
            db,
            session_record=record,
            blueprint_version_number=blueprint_version_number,
            blueprint=blueprint_envelope.data,
            source_action=source_action,
            pending_approvals=pending_approvals,
        )
        resulting_stage = blueprint_envelope.stage
        resulting_status = blueprint_envelope.status
        blueprint = next_blueprint

    if next_evaluation is not None:
        evaluation_envelope = EvaluationEnvelope(
            status=ArtifactStatus.ready
            if next_evaluation.completeness_status == "complete"
            else ArtifactStatus.needs_review,
            stage=SessionStage.post_validation,
            data=next_evaluation,
            missing_fields=list(next_evaluation.gaps),
            assumptions=[],
            warnings=list(trace.warnings),
            evidence=list(trace.evidence),
            next_action="ready_for_export"
            if next_evaluation.completeness_status == "complete"
            else "resolve_gaps",
        )
        evaluation_envelope = apply_pending_approvals_to_evaluation(
            evaluation_envelope,
            count_pending_approvals(db, session_id),
        )
        if evaluation_dataset_record is None or evaluation_rubric_record is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Evaluation assets not available",
            )
        run_summary = score_evaluation_workbench(
            evaluation_dataset
            if evaluation_dataset is not None
            else hydrate_evaluation_dataset(db, evaluation_dataset_record),
            evaluation_rubric
            if evaluation_rubric is not None
            else hydrate_evaluation_rubric(evaluation_rubric_record),
            discovery,
            canvas,
            blueprint,
            source_action=source_action,
        )
        run_summary = apply_pending_approvals_to_run_summary(
            run_summary,
            count_pending_approvals(db, session_id),
            blueprint_version_number=blueprint_version_number,
        )
        persist_evaluation_run(
            db,
            session_id=session_id,
            dataset_record=evaluation_dataset_record,
            rubric_record=evaluation_rubric_record,
            run_summary=run_summary,
        )
        sync_evaluation_handoff(
            db,
            session_record=record,
            blueprint_version_number=blueprint_version_number,
            source_action=source_action,
            overall_score=run_summary.overall_score,
            status=run_summary.status,
        )
        upsert_evaluation(db, session_id, evaluation_envelope)
        resulting_stage = evaluation_envelope.stage
        resulting_status = evaluation_envelope.status

    created_runs = write_skill_runs(
        db,
        session_id=session_id,
        traces=[trace],
        source_action=source_action,
        blueprint_version_number=blueprint_version_number,
    )
    write_log(
        db,
        session_id=session_id,
        stage=resulting_stage,
        status_value=resulting_status,
        message=f"Skill reejecutada: {skill_key}",
        payload={
            "skill_key": skill_key,
            "source_action": source_action,
            "status": resulting_status,
            "blueprint_version_number": blueprint_version_number,
        },
    )
    touch_session(record, resulting_stage, resulting_status)
    db.add(record)
    sync_short_term_memory_checkpoint(db, record=record, source_action=source_action)
    capture_operational_state(db, session_id=session_id, source_action=source_action)
    db.commit()

    snapshot = build_snapshot(db, record)
    skill_run = next((item for item in snapshot.skill_runs if item.id == created_runs[0].id), None)
    if skill_run is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Skill run snapshot unavailable")
    return SkillRerunResponse(skill_run=skill_run, snapshot=snapshot)


@router.post("/{session_id}/approvals/{approval_id}/resolve", response_model=ApprovalGateEntry)
def resolve_approval_route(
    session_id: UUID,
    approval_id: UUID,
    payload: ApprovalResolutionRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> ApprovalGateEntry:
    record = get_or_404(db, session_id, current_user.id)
    if is_feature_flag_enabled(db, FEATURE_FLAG_GOVERNANCE, workspace_id=record.workspace_id):
        try:
            ensure_local_admin_can_govern(current_user.email)
        except PermissionError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    approval = db.exec(
        select(ApprovalGateRecord).where(
            ApprovalGateRecord.id == approval_id,
            ApprovalGateRecord.session_id == session_id,
        )
    ).first()
    if approval is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Approval gate not found")
    if payload.decision == ApprovalStatus.pending:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Pending is not a valid resolution")

    approval.status = payload.decision
    approval.resolution_note = payload.resolution_note
    approval.resolved_at = utc_now()
    db.add(approval)
    remaining_pending_approvals = count_pending_approvals(db, session_id)

    next_status = record.status
    if payload.decision == ApprovalStatus.rejected:
        next_status = ArtifactStatus.needs_review
    elif remaining_pending_approvals <= 1 and record.status != ArtifactStatus.failed:
        next_status = ArtifactStatus.ready

    governance_handoff = db.exec(
        select(HandoffRecord)
        .where(
            HandoffRecord.session_id == session_id,
            HandoffRecord.handoff_key == "governance_review",
        )
        .order_by(HandoffRecord.updated_at.desc(), HandoffRecord.created_at.desc())
    ).first()
    if governance_handoff is not None:
        resolve_handoff_record(
            db,
            handoff_record=governance_handoff,
            decision=HANDOFF_STATUS_RETURNED if payload.decision == ApprovalStatus.rejected else HANDOFF_STATUS_COMPLETED,
            resolution_note=payload.resolution_note or (
                "Retorno al blueprint por rechazo de gate."
                if payload.decision == ApprovalStatus.rejected
                else "Gate resuelto y handoff listo."
            ),
        )

    touch_session(record, SessionStage.post_validation, next_status)
    write_log(
        db,
        session_id=session_id,
        stage=SessionStage.post_validation,
        status_value=next_status,
        message="Approval gate resuelto",
        payload={
            "approval_id": str(approval_id),
            "gate_key": approval.gate_key,
            "decision": payload.decision,
            "resolution_note": payload.resolution_note,
        },
    )
    db.add(record)
    sync_short_term_memory_checkpoint(db, record=record, source_action="resolve_approval")
    capture_operational_state(db, session_id=session_id, source_action="resolve_approval")
    db.commit()
    db.refresh(approval)
    return build_approval_entry(approval)


@router.post("/{session_id}/handoffs/{handoff_id}/resolve", response_model=SessionSnapshot)
def resolve_handoff_route(
    session_id: UUID,
    handoff_id: UUID,
    payload: HandoffResolutionRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> SessionSnapshot:
    record = get_or_404(db, session_id, current_user.id)
    if not is_feature_flag_enabled(db, FEATURE_FLAG_GOVERNANCE, workspace_id=record.workspace_id):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Governance console feature flag is disabled")
    try:
        ensure_local_admin_can_govern(current_user.email)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc

    handoff = db.exec(
        select(HandoffRecord).where(HandoffRecord.id == handoff_id, HandoffRecord.session_id == session_id)
    ).first()
    if handoff is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Handoff not found")

    try:
        resolve_handoff_record(
            db,
            handoff_record=handoff,
            decision=payload.decision,
            resolution_note=payload.resolution_note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    next_status = record.status
    next_stage = handoff.to_stage
    if payload.decision == HANDOFF_STATUS_RETURNED:
        next_status = ArtifactStatus.needs_review
        next_stage = handoff.from_stage
    elif count_pending_approvals(db, session_id) == 0 and record.status != ArtifactStatus.failed:
        next_status = ArtifactStatus.ready

    touch_session(record, next_stage, next_status)
    write_log(
        db,
        session_id=session_id,
        stage=next_stage,
        status_value=next_status,
        message="Handoff resuelto",
        payload={
            "handoff_id": str(handoff_id),
            "handoff_key": handoff.handoff_key,
            "decision": payload.decision,
            "resolution_note": payload.resolution_note,
        },
    )
    db.add(record)
    sync_short_term_memory_checkpoint(
        db,
        record=record,
        source_action="resolve_handoff",
        branch_key=f"handoff:{handoff.handoff_key or handoff.id}",
    )
    capture_operational_state(db, session_id=session_id, source_action="resolve_handoff")
    db.commit()
    return build_snapshot(db, record)


@router.post("/{session_id}/subagents/{run_kind}/run", response_model=SessionSnapshot)
def run_subagent_route(
    session_id: UUID,
    run_kind: str,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> SessionSnapshot:
    record = get_or_404(db, session_id, current_user.id)
    apply_workspace_bootstrap(db, record.workspace_id)
    required_flag = feature_flag_for_subagent_run(run_kind)
    if not is_feature_flag_enabled(db, required_flag, workspace_id=record.workspace_id):
        detail = (
            "Multi-agent runtime feature flag is disabled"
            if required_flag == FEATURE_FLAG_MULTI_AGENT_RUNTIME
            else "Specialized subagents feature flag is disabled"
        )
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)
    blueprint_record = db.exec(select(BlueprintRecord).where(BlueprintRecord.session_id == session_id)).first()
    if blueprint_record is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Blueprint must exist before running a specialized subprocess")

    blueprint = hydrate_blueprint(blueprint_record)
    approvals = db.exec(select(ApprovalGateRecord).where(ApprovalGateRecord.session_id == session_id)).all()
    latest_evaluation_run = db.exec(
        select(EvaluationRunRecord)
        .where(EvaluationRunRecord.session_id == session_id)
        .order_by(EvaluationRunRecord.created_at.desc())
    ).first()
    blueprint_version_number = latest_blueprint_version_number(db, session_id)
    snapshot_for_run = build_snapshot(db, record) if run_kind == "supervisor_orchestrator" else None
    try:
        subagent_run = create_subagent_run(
            db,
            session_record=record,
            blueprint_version_number=blueprint_version_number,
            run_kind=run_kind,
            blueprint=blueprint,
            approvals=approvals,
            latest_evaluation_score=latest_evaluation_run.overall_score if latest_evaluation_run is not None else None,
            snapshot=snapshot_for_run,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    next_status = ArtifactStatus.needs_review if subagent_run.status == ArtifactStatus.needs_review else record.status
    touch_session(record, record.current_stage, next_status)
    log_message = "Orquestacion multiagente ejecutada" if run_kind == "supervisor_orchestrator" else "Subproceso especializado ejecutado"
    write_log(
        db,
        session_id=session_id,
        stage=record.current_stage,
        status_value=next_status,
        message=log_message,
        payload={"run_kind": run_kind, "subagent_run_id": str(subagent_run.id)},
    )
    db.add(record)
    sync_short_term_memory_checkpoint(
        db,
        record=record,
        source_action=f"run_subagent:{run_kind}",
        branch_key=f"subagent_run:{subagent_run.id}",
    )
    capture_operational_state(db, session_id=session_id, source_action=f"run_subagent:{run_kind}")
    db.commit()
    return build_snapshot(db, record)


@router.get("/{session_id}/export/markdown", response_class=PlainTextResponse)
def export_markdown_route(
    session_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> str:
    record = get_or_404(db, session_id, current_user.id)
    ensure_commercial_capability(record, "export_markdown", db=db, current_user=current_user)
    snapshot = build_snapshot(db, record)
    markdown = build_markdown_export(snapshot)
    export_record = record_export_artifact(
        db,
        session_id=session_id,
        blueprint_version_number=latest_blueprint_version_number(db, session_id),
        artifact_key="markdown_export",
        artifact_title="Blueprint export markdown",
        export_format="markdown",
        content_text=markdown,
        source_action="export_markdown",
    )
    create_export_handoff(
        db,
        session_record=record,
        blueprint_version_number=export_record.blueprint_version_number,
        source_action="export_markdown",
        artifact_key=export_record.artifact_key,
    )
    write_log(
        db,
        session_id=session_id,
        stage=record.current_stage,
        status_value=record.status,
        message="Blueprint export markdown descargado",
        payload={
            "artifact_key": export_record.artifact_key,
            "blueprint_version_number": export_record.blueprint_version_number,
            "export_format": "markdown",
            "product": "blueprint_pro",
            "source_action": "export_markdown",
        },
    )
    capture_operational_state(db, session_id=session_id, source_action="export_markdown")
    db.commit()
    return markdown


@router.get("/{session_id}/export/json")
def export_json_route(
    session_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> JSONResponse:
    record = get_or_404(db, session_id, current_user.id)
    ensure_commercial_capability(record, "export_json", db=db, current_user=current_user)
    snapshot = build_snapshot(db, record)
    payload = build_json_export(snapshot)
    export_record = record_export_artifact(
        db,
        session_id=session_id,
        blueprint_version_number=latest_blueprint_version_number(db, session_id),
        artifact_key="json_export",
        artifact_title="Blueprint export json",
        export_format="json",
        content_text=json.dumps(payload, ensure_ascii=True),
        source_action="export_json",
    )
    create_export_handoff(
        db,
        session_record=record,
        blueprint_version_number=export_record.blueprint_version_number,
        source_action="export_json",
        artifact_key=export_record.artifact_key,
    )
    write_log(
        db,
        session_id=session_id,
        stage=record.current_stage,
        status_value=record.status,
        message="Blueprint export json descargado",
        payload={
            "artifact_key": export_record.artifact_key,
            "blueprint_version_number": export_record.blueprint_version_number,
            "export_format": "json",
            "product": "blueprint_pro",
            "source_action": "export_json",
        },
    )
    capture_operational_state(db, session_id=session_id, source_action="export_json")
    db.commit()
    return JSONResponse(content=payload)


@router.get("/{session_id}/export/blueprint-core")
def export_blueprint_core_route(
    session_id: UUID,
    preview: bool = False,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> JSONResponse:
    return build_canonical_export_response(
        session_id=session_id,
        contract_key="blueprint-core.v1",
        preview=preview,
        db=db,
        current_user=current_user,
    )


@router.get("/{session_id}/export/construction-pack")
def export_construction_pack_route(
    session_id: UUID,
    preview: bool = False,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> JSONResponse:
    return build_canonical_export_response(
        session_id=session_id,
        contract_key="construction-pack.v1",
        preview=preview,
        db=db,
        current_user=current_user,
    )


@router.get("/{session_id}/export/agent-construction-package")
def export_agent_construction_package_route(
    session_id: UUID,
    preview: bool = False,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> JSONResponse:
    return build_canonical_export_response(
        session_id=session_id,
        contract_key="agent-construction-package.v2",
        preview=preview,
        db=db,
        current_user=current_user,
    )


@router.get("/{session_id}/export/prompt-pack")
def export_prompt_pack_route(
    session_id: UUID,
    preview: bool = False,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> JSONResponse:
    return build_canonical_export_response(
        session_id=session_id,
        contract_key="prompt-pack.v1",
        preview=preview,
        db=db,
        current_user=current_user,
    )


@router.get("/{session_id}/export/estimation-pack")
def export_estimation_pack_route(
    session_id: UUID,
    preview: bool = False,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> JSONResponse:
    return build_canonical_export_response(
        session_id=session_id,
        contract_key="estimation-pack.v1",
        preview=preview,
        db=db,
        current_user=current_user,
    )


@router.get("/{session_id}/export/test-pack")
def export_test_pack_route(
    session_id: UUID,
    preview: bool = False,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> JSONResponse:
    return build_canonical_export_response(
        session_id=session_id,
        contract_key="test-pack.v1",
        preview=preview,
        db=db,
        current_user=current_user,
    )
