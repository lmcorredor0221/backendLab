from __future__ import annotations

import copy
import hashlib
import json
from typing import Any
from uuid import UUID

from sqlmodel import Session, select

from app.models import (
    ApprovedToolsDigest,
    ArtifactRegistryRecord,
    ArtifactStatus,
    BlueprintArtifact,
    BlueprintRecord,
    BlueprintTool,
    BlueprintVersionRecord,
    CanvasArtifact,
    CanvasRecord,
    CommercialTier,
    DesignRecommendationArtifact,
    DiscoveryArtifact,
    EstimationReportArtifact,
    EvaluationArtifact,
    EvaluationRecord,
    JourneyArtifactState,
    JourneyDecisionType,
    JourneyStageArtifactApprovalRequest,
    JourneyStageArtifactCreateRequest,
    JourneyStageArtifactEntry,
    JourneyStageArtifactListResponse,
    JourneyStageArtifactPatchRequest,
    JourneyStageArtifactRecord,
    JourneyStageArtifactRejectionRequest,
    JourneyStageDecisionEntry,
    JourneyStageDecisionRecord,
    MemoryRecommendationArtifact,
    OpportunityRecord,
    ReviewState,
    SessionRecord,
    SessionStage,
    SimulationSpecificationArtifact,
    ToolRecommendationArtifact,
    UserRecord,
    utc_now,
)
from app.services.llm_runtime.builder_contracts import DiscoveryAnalysisOutput, RequirementsDefinitionOutput
from app.services.product_processing.policy import resolve_product_processing_mode
from app.services.product_processing.contracts import ProductProcessingMode
from app.services.skill_runtime import validate_definition_artifact
from app.services.estimation_calibration import persist_estimation_run
from app.services.journey_stage_contract import get_journey_stage_boundary, list_journey_stage_boundaries
from app.services.blueprint_hydration import hydrate_blueprint_record
from app.services.operations_service import record_estimation_artifact
from app.services.tool_recommendation_service import (
    build_approved_tools_digest_from_blueprint_tools,
    evaluate_tool_recommendation_artifact,
    promote_tool_recommendation_to_blueprint_tools,
)


IMMUTABLE_ARTIFACT_STATES = {
    JourneyArtifactState.approved,
    JourneyArtifactState.approved_legacy,
    JourneyArtifactState.rejected,
    JourneyArtifactState.stale,
    JourneyArtifactState.needs_review_legacy,
}
APPROVED_ARTIFACT_STATES = {
    JourneyArtifactState.approved,
    JourneyArtifactState.approved_legacy,
}
STAGE_ORDER = tuple(boundary.stage_key for boundary in list_journey_stage_boundaries())
DEFAULT_ARTIFACT_KIND_BY_STAGE = {
    "discover": "discovery_artifact",
    "define": "definition_artifact",
    "design": "design_recommendation_artifact",
    "tools": "tool_recommendation_artifact",
    "memory": "memory_recommendation_artifact",
    "validate": "evaluation_artifact",
    "estimate": "estimation_report_artifact",
    "build": "build_package_artifact",
}
DESIGN_BLUEPRINT_FIELDS = {"architecture", "reasoning_pattern", "safety_checks", "guardrails", "narrative"}
MEMORY_BLUEPRINT_FIELDS = {"memory_strategy", "memory_profile", "knowledge_profile"}
SESSION_STAGE_BY_JOURNEY_STAGE = {
    "discover": SessionStage.normalize_discovery,
    "define": SessionStage.build_canvas,
    "design": SessionStage.build_blueprint,
    "tools": SessionStage.build_blueprint,
    "memory": SessionStage.build_blueprint,
    "validate": SessionStage.post_validation,
    "estimate": SessionStage.ready_for_export,
    "build": SessionStage.ready_for_export,
}


class StageProposalError(RuntimeError):
    pass


class StageProposalNotFoundError(StageProposalError):
    pass


class StageProposalConflictError(StageProposalError):
    pass


def _stable_hash(payload: Any) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _journey_stage_or_raise(stage_key: str) -> str:
    return get_journey_stage_boundary(stage_key).stage_key


def _default_artifact_kind(stage_key: str, artifact_kind: str | None = None) -> str:
    normalized = _journey_stage_or_raise(stage_key)
    candidate = (artifact_kind or "").strip()
    return candidate or DEFAULT_ARTIFACT_KIND_BY_STAGE[normalized]


def _hydrate_blueprint(record: BlueprintRecord) -> BlueprintArtifact:
    return hydrate_blueprint_record(record)


def _assign_blueprint_record(record: BlueprintRecord, artifact: BlueprintArtifact) -> None:
    record.architecture = artifact.architecture
    record.reasoning_pattern = artifact.reasoning_pattern
    record.memory_strategy = artifact.memory_strategy
    record.tools = [item.model_dump(mode="json") for item in artifact.tools]
    record.llm_policy = artifact.llm_policy.model_dump(mode="json")
    record.memory_profile = artifact.memory_profile.model_dump(mode="json")
    record.knowledge_profile = artifact.knowledge_profile.model_dump(mode="json")
    record.safety_checks = [item.model_dump(mode="json") for item in artifact.safety_checks]
    record.guardrails = list(artifact.guardrails)
    record.delivery_package = artifact.delivery_package.model_dump(mode="json")
    record.readiness_state = artifact.readiness_state
    record.narrative = artifact.narrative
    record.updated_at = utc_now()


def _serialize_tool_recommendation(artifact: ToolRecommendationArtifact) -> str:
    return json.dumps(artifact.model_dump(mode="json"), ensure_ascii=False, indent=2)


def _resolve_discovery_artifact_payload(artifact_record: JourneyStageArtifactRecord) -> tuple[DiscoveryArtifact, dict[str, Any]]:
    if artifact_record.schema_version == "discovery-analysis.v1" or "normalized_discovery_candidate" in artifact_record.proposal_payload:
        analysis = DiscoveryAnalysisOutput.model_validate(artifact_record.proposal_payload)
        candidate_payload = analysis.normalized_discovery_candidate.model_dump(mode="json")
        patched_candidate = artifact_record.user_patch.get("normalized_discovery_candidate")
        if isinstance(patched_candidate, dict) and patched_candidate:
            candidate_payload = copy.deepcopy(patched_candidate)
        artifact = DiscoveryArtifact.model_validate(candidate_payload)
        return artifact, {
            "analysis_confidence": analysis.confidence,
            "open_question_count": len(analysis.open_questions),
            "missing_information": list(analysis.missing_information),
        }

    artifact = DiscoveryArtifact.model_validate(artifact_record.proposal_payload)
    return artifact, {}


def _resolve_definition_artifact_payload(
    artifact_record: JourneyStageArtifactRecord,
    *,
    commercial_tier: CommercialTier | str = CommercialTier.blueprint,
) -> tuple[CanvasArtifact, RequirementsDefinitionOutput | None, dict[str, Any]]:
    if artifact_record.schema_version == "definition-artifact.v1" or "functional_requirements" in artifact_record.proposal_payload:
        definition_payload = copy.deepcopy(artifact_record.proposal_payload)
        patched_definition = artifact_record.user_patch.get("definition")
        if isinstance(patched_definition, dict) and patched_definition:
            definition_payload = patched_definition
        definition = validate_definition_artifact(RequirementsDefinitionOutput.model_validate(definition_payload))
        approval_blockers = _definition_approval_blocking_issues(definition, commercial_tier)
        if approval_blockers:
            raise StageProposalConflictError(
                "Define still has blocking issues for the active product tier. Resolve questions, acceptance criteria or contradictions before approval."
            )
        canvas_projection = CanvasArtifact.model_validate(definition.canvas_projection.model_dump(mode="json"))
        return canvas_projection, definition, {
            "definition_confidence": definition.confidence,
            "functional_requirement_count": len(definition.functional_requirements),
            "blocking_issue_count": len(definition.validation.blocking_issues),
            "approval_blocking_issue_count": len(approval_blockers),
            "open_question_count": len(definition.open_questions),
        }

    artifact = CanvasArtifact.model_validate(artifact_record.proposal_payload)
    return artifact, None, {}


def _definition_approval_blocking_issues(
    definition: RequirementsDefinitionOutput,
    commercial_tier: CommercialTier | str,
) -> list[str]:
    """Apply product-tier approval semantics to Define validation findings.

    Blueprint Basico is intentionally an infer/defer/continue product. Quality
    gaps detected during Define remain traceable in the artifact and can feed
    Premium enrichment, but they must not block the first-value funnel unless
    the artifact itself is invalid, which is handled before this function.
    """
    issues = list(definition.validation.blocking_issues)
    mode = resolve_product_processing_mode(commercial_tier)
    if mode == ProductProcessingMode.basic_free:
        return []
    return issues


def _resolve_design_artifact_payload(
    artifact_record: JourneyStageArtifactRecord,
    *,
    commercial_tier: CommercialTier | str = CommercialTier.blueprint,
    decision_payload: dict[str, Any],
) -> tuple[dict[str, Any], DesignRecommendationArtifact | None, dict[str, Any]]:
    if artifact_record.schema_version == "design-recommendation.v1" or "alternatives" in artifact_record.proposal_payload:
        design_payload = copy.deepcopy(artifact_record.proposal_payload)
        artifact = DesignRecommendationArtifact.model_validate(design_payload)
        selected_key = str(
            decision_payload.get("selected_alternative_key")
            or (artifact.selected_design.alternative_key if artifact.selected_design is not None else "")
            or artifact.recommended_alternative_key
        ).strip()
        selected_design = next(
            (item for item in artifact.alternatives if item.alternative_key == selected_key),
            artifact.selected_design or (artifact.alternatives[0] if artifact.alternatives else None),
        )
        if selected_design is None:
            raise StageProposalConflictError(
                "Design must keep at least one selectable alternative before approval."
            )
        approval_blockers = _design_approval_blocking_issues(artifact, commercial_tier)
        if approval_blockers:
            raise StageProposalConflictError(
                "Design still has blocking issues for the active product tier. Resolve them before approval."
            )
        projection = selected_design.blueprint_projection.model_dump(mode="json")
        promoted = artifact.model_copy(
            update={
                "recommended_alternative_key": selected_design.alternative_key,
                "selected_design": selected_design,
            }
        )
        return projection, promoted, {
            "selected_alternative_key": selected_design.alternative_key,
            "alternative_count": len(artifact.alternatives),
            "critic_finding_count": len(artifact.critic_findings),
            "approval_blocking_issue_count": len(approval_blockers),
            "review_state": artifact.review_state,
        }

    payload = artifact_record.proposal_payload
    projection = {
        key: payload.get(key)
        for key in DESIGN_BLUEPRINT_FIELDS
        if key in payload
    }
    return projection, None, {"projection_mode": "legacy_design_sections"}


def _design_approval_blocking_issues(
    artifact: DesignRecommendationArtifact,
    commercial_tier: CommercialTier | str,
) -> list[str]:
    mode = resolve_product_processing_mode(commercial_tier)
    if mode == ProductProcessingMode.basic_free:
        return []

    issues: list[str] = []
    if artifact.review_state == ReviewState.blocked:
        issues.append("design_review_state:blocked")
    issues.extend(
        f"blocking_finding:{item.finding_key or item.title}"
        for item in artifact.critic_findings
        if item.severity == "blocking"
    )
    issues.extend(f"open_question:{item}" for item in artifact.open_questions if item)
    issues.extend(f"missing_information:{item}" for item in artifact.missing_information if item)
    return issues


def _memory_approval_blocking_issues(
    artifact: MemoryRecommendationArtifact,
    commercial_tier: CommercialTier | str,
) -> list[str]:
    issues: list[str] = []
    if artifact.dry_compile_status.status == "blocked":
        issues.extend(f"dry_compile:{item}" for item in artifact.dry_compile_status.blocking_issues)
        if not artifact.dry_compile_status.blocking_issues:
            issues.append("dry_compile:blocked")
    issues.extend(
        f"missing_required_tool:{item.tool_key}"
        for item in artifact.tool_dependencies
        if item.required and item.status == "missing"
    )

    mode = resolve_product_processing_mode(commercial_tier)
    if mode == ProductProcessingMode.basic_free:
        return issues

    if artifact.review_state == ReviewState.blocked:
        issues.append("memory_review_state:blocked")
    issues.extend(
        f"blocking_finding:{item.finding_key or item.title}"
        for item in artifact.critic_findings
        if item.severity == "blocking"
    )
    issues.extend(f"open_question:{item}" for item in artifact.open_questions if item)
    issues.extend(f"missing_information:{item}" for item in artifact.missing_information if item)
    return issues


class StageProposalService:
    def list_all(
        self,
        session: Session,
        *,
        session_record: SessionRecord,
    ) -> list[JourneyStageArtifactEntry]:
        records = session.exec(
            select(JourneyStageArtifactRecord)
            .where(
                JourneyStageArtifactRecord.workspace_id == session_record.workspace_id,
                JourneyStageArtifactRecord.session_id == session_record.id,
            )
            .order_by(
                JourneyStageArtifactRecord.stage_key.asc(),
                JourneyStageArtifactRecord.version_number.desc(),
                JourneyStageArtifactRecord.created_at.desc(),
            )
        ).all()
        return self._build_artifact_entries(session, records)

    def latest_by_stage(
        self,
        session: Session,
        *,
        session_record: SessionRecord,
    ) -> dict[str, JourneyStageArtifactEntry]:
        entries = self.list_all(session, session_record=session_record)
        latest: dict[str, JourneyStageArtifactEntry] = {}
        for entry in entries:
            if entry.stage_key not in latest:
                latest[entry.stage_key] = entry
        return latest

    def list(
        self,
        session: Session,
        *,
        session_record: SessionRecord,
        stage_key: str,
    ) -> JourneyStageArtifactListResponse:
        normalized_stage = _journey_stage_or_raise(stage_key)
        records = session.exec(
            select(JourneyStageArtifactRecord)
            .where(
                JourneyStageArtifactRecord.workspace_id == session_record.workspace_id,
                JourneyStageArtifactRecord.session_id == session_record.id,
                JourneyStageArtifactRecord.stage_key == normalized_stage,
            )
            .order_by(JourneyStageArtifactRecord.version_number.desc(), JourneyStageArtifactRecord.created_at.desc())
        ).all()
        latest = records[0] if records else None
        return JourneyStageArtifactListResponse(
            items=self._build_artifact_entries(session, records),
            latest=self._build_artifact_entry(session, latest) if latest is not None else None,
        )

    def latest(
        self,
        session: Session,
        *,
        session_record: SessionRecord,
        stage_key: str,
    ) -> JourneyStageArtifactEntry | None:
        record = self._latest_record(session, session_record=session_record, stage_key=stage_key)
        return self._build_artifact_entry(session, record) if record is not None else None

    def get(
        self,
        session: Session,
        *,
        session_record: SessionRecord,
        stage_key: str,
        artifact_id: UUID,
    ) -> JourneyStageArtifactEntry:
        record = self._artifact_or_raise(
            session,
            session_record=session_record,
            stage_key=stage_key,
            artifact_id=artifact_id,
        )
        entry = self._build_artifact_entry(session, record)
        if entry is None:
            raise StageProposalNotFoundError("Journey stage artifact not found.")
        return entry

    def create(
        self,
        session: Session,
        *,
        session_record: SessionRecord,
        stage_key: str,
        payload: JourneyStageArtifactCreateRequest,
        actor_user: UserRecord,
    ) -> JourneyStageArtifactEntry:
        normalized_stage = _journey_stage_or_raise(stage_key)
        previous_latest = self._latest_record(session, session_record=session_record, stage_key=normalized_stage)
        now = utc_now()
        record = JourneyStageArtifactRecord(
            workspace_id=session_record.workspace_id,
            session_id=session_record.id,
            artifact_kind=_default_artifact_kind(normalized_stage, payload.artifact_kind),
            stage_key=normalized_stage,
            version_number=self._next_version(session, session_record=session_record, stage_key=normalized_stage),
            state=JourneyArtifactState.generated,
            source_action=payload.source_action or "manual_draft",
            proposal_payload=copy.deepcopy(payload.proposal_payload),
            user_patch=copy.deepcopy(payload.user_patch),
            source_stage_versions=self._coalesce_source_stage_versions(
                session,
                session_record=session_record,
                provided=payload.source_stage_versions,
            ),
            input_fingerprint=payload.input_fingerprint,
            context_fingerprint=payload.context_fingerprint,
            output_fingerprint=payload.output_fingerprint,
            corpus_hash=payload.corpus_hash,
            provider_key=payload.provider_key,
            model=payload.model,
            execution_backend=payload.execution_backend,
            prompt_version=payload.prompt_version,
            schema_version=payload.schema_version,
            confidence=payload.confidence,
            missing_information=list(payload.missing_information),
            warnings=list(dict.fromkeys(payload.warnings)),
            evidence_manifest=[item.model_dump(mode="json") for item in payload.evidence_manifest],
            created_at=now,
            updated_at=now,
        )
        self._ensure_fingerprints(record)
        session.add(record)
        session.flush()
        self._record_decision(
            session,
            session_record=session_record,
            artifact_record=record,
            actor_user_id=actor_user.id,
            decision_type=JourneyDecisionType.create,
            previous_state=None,
            next_state=record.state,
            note=payload.note,
            payload={"artifact_kind": record.artifact_kind},
        )
        session_record.updated_at = now
        session.add(session_record)
        session.flush()
        if previous_latest is not None:
            self._invalidate_downstream(
                session,
                session_record=session_record,
                upstream_artifact=record,
                actor_user_id=actor_user.id,
                reason_action="regenerated",
            )
            session.flush()
        return self._build_artifact_entry(session, record)

    def patch(
        self,
        session: Session,
        *,
        session_record: SessionRecord,
        stage_key: str,
        artifact_id: UUID,
        payload: JourneyStageArtifactPatchRequest,
        actor_user: UserRecord,
    ) -> JourneyStageArtifactEntry:
        record = self._artifact_or_raise(
            session,
            session_record=session_record,
            stage_key=stage_key,
            artifact_id=artifact_id,
        )
        if record.state in IMMUTABLE_ARTIFACT_STATES:
            return self._replace(
                session,
                session_record=session_record,
                original=record,
                payload=payload,
                actor_user=actor_user,
            )

        previous_state = record.state
        now = utc_now()
        if payload.artifact_kind is not None:
            record.artifact_kind = _default_artifact_kind(record.stage_key, payload.artifact_kind)
        if payload.proposal_payload is not None:
            record.proposal_payload = copy.deepcopy(payload.proposal_payload)
        if payload.user_patch is not None:
            merged_patch = dict(record.user_patch)
            merged_patch.update(payload.user_patch)
            record.user_patch = merged_patch
        if payload.source_stage_versions is not None:
            record.source_stage_versions = self._coalesce_source_stage_versions(
                session,
                session_record=session_record,
                provided=payload.source_stage_versions,
            )
        for field_name in (
            "input_fingerprint",
            "context_fingerprint",
            "output_fingerprint",
            "corpus_hash",
            "provider_key",
            "model",
            "execution_backend",
            "prompt_version",
            "schema_version",
            "confidence",
        ):
            value = getattr(payload, field_name)
            if value is not None:
                setattr(record, field_name, value)
        if payload.missing_information is not None:
            record.missing_information = list(payload.missing_information)
        if payload.warnings is not None:
            record.warnings = list(dict.fromkeys(payload.warnings))
        if payload.evidence_manifest is not None:
            record.evidence_manifest = [item.model_dump(mode="json") for item in payload.evidence_manifest]
        if previous_state == JourneyArtifactState.generated:
            record.state = JourneyArtifactState.reviewed
            record.reviewed_at = now
        record.updated_at = now
        self._ensure_fingerprints(record)
        session.add(record)
        self._record_decision(
            session,
            session_record=session_record,
            artifact_record=record,
            actor_user_id=actor_user.id,
            decision_type=JourneyDecisionType.patch,
            previous_state=previous_state,
            next_state=record.state,
            note=payload.note,
            payload={"artifact_kind": record.artifact_kind},
        )
        session_record.updated_at = now
        session.add(session_record)
        session.flush()
        self._invalidate_downstream(
            session,
            session_record=session_record,
            upstream_artifact=record,
            actor_user_id=actor_user.id,
            reason_action="regenerated",
        )
        session.flush()
        return self._build_artifact_entry(session, record)

    def approve(
        self,
        session: Session,
        *,
        session_record: SessionRecord,
        stage_key: str,
        artifact_id: UUID,
        payload: JourneyStageArtifactApprovalRequest,
        actor_user: UserRecord,
    ) -> JourneyStageArtifactEntry:
        record = self._artifact_or_raise(
            session,
            session_record=session_record,
            stage_key=stage_key,
            artifact_id=artifact_id,
        )
        self._ensure_required_predecessors_approved(
            session,
            session_record=session_record,
            stage_key=record.stage_key,
        )
        latest = self._latest_record(session, session_record=session_record, stage_key=stage_key)
        if latest is None or latest.id != record.id:
            raise StageProposalConflictError("Only the latest version can be approved.")
        if record.state == JourneyArtifactState.stale:
            raise StageProposalConflictError("The selected proposal is stale and must be regenerated before approval.")
        if record.state == JourneyArtifactState.rejected:
            raise StageProposalConflictError("Rejected proposals cannot be approved.")

        projection_payload = self._project_to_canonical(
            session,
            session_record=session_record,
            artifact_record=record,
            decision_payload=payload.decision_payload,
        )
        previous_state = record.state
        now = utc_now()
        record.state = JourneyArtifactState.approved
        record.reviewed_at = record.reviewed_at or now
        record.approved_at = now
        record.approved_by_user_id = actor_user.id
        record.stale_reasons = []
        record.stale_at = None
        record.updated_at = now
        session.add(record)
        self._mark_prior_versions_superseded(session, record)
        self._record_decision(
            session,
            session_record=session_record,
            artifact_record=record,
            actor_user_id=actor_user.id,
            decision_type=JourneyDecisionType.approve,
            previous_state=previous_state,
            next_state=record.state,
            note=payload.note,
            payload=projection_payload,
        )
        self._invalidate_downstream(
            session,
            session_record=session_record,
            upstream_artifact=record,
            actor_user_id=actor_user.id,
        )
        session_record.updated_at = now
        session.add(session_record)
        session.flush()
        return self._build_artifact_entry(session, record)

    def reject(
        self,
        session: Session,
        *,
        session_record: SessionRecord,
        stage_key: str,
        artifact_id: UUID,
        payload: JourneyStageArtifactRejectionRequest,
        actor_user: UserRecord,
    ) -> JourneyStageArtifactEntry:
        record = self._artifact_or_raise(
            session,
            session_record=session_record,
            stage_key=stage_key,
            artifact_id=artifact_id,
        )
        previous_state = record.state
        now = utc_now()
        record.state = JourneyArtifactState.rejected
        record.rejected_at = now
        record.updated_at = now
        session.add(record)
        self._record_decision(
            session,
            session_record=session_record,
            artifact_record=record,
            actor_user_id=actor_user.id,
            decision_type=JourneyDecisionType.reject,
            previous_state=previous_state,
            next_state=record.state,
            note=payload.note,
            payload=payload.decision_payload,
        )
        session_record.updated_at = now
        session.add(session_record)
        session.flush()
        self._invalidate_downstream(
            session,
            session_record=session_record,
            upstream_artifact=record,
            actor_user_id=actor_user.id,
            reason_action="rejected",
        )
        session.flush()
        return self._build_artifact_entry(session, record)

    def backfill_legacy(
        self,
        session: Session,
        *,
        session_record: SessionRecord,
        stage_key: str,
        artifact_kind: str,
        proposal_payload: dict[str, Any],
        state: JourneyArtifactState,
        source_action: str,
        note: str = "",
    ) -> JourneyStageArtifactEntry:
        normalized_stage = _journey_stage_or_raise(stage_key)
        existing = self._latest_record(session, session_record=session_record, stage_key=normalized_stage)
        if existing is not None:
            return self._build_artifact_entry(session, existing)

        now = utc_now()
        record = JourneyStageArtifactRecord(
            workspace_id=session_record.workspace_id,
            session_id=session_record.id,
            artifact_kind=_default_artifact_kind(normalized_stage, artifact_kind),
            stage_key=normalized_stage,
            version_number=1,
            state=state,
            source_action=source_action,
            proposal_payload=copy.deepcopy(proposal_payload),
            user_patch={},
            source_stage_versions=self._derive_source_stage_versions(session, session_record=session_record),
            created_at=now,
            updated_at=now,
            reviewed_at=now if state in {JourneyArtifactState.reviewed, JourneyArtifactState.approved_legacy} else None,
            approved_at=now if state == JourneyArtifactState.approved_legacy else None,
        )
        self._ensure_fingerprints(record)
        session.add(record)
        session.flush()
        self._record_decision(
            session,
            session_record=session_record,
            artifact_record=record,
            actor_user_id=None,
            decision_type=JourneyDecisionType.backfill_legacy,
            previous_state=None,
            next_state=state,
            note=note,
            payload={"artifact_kind": record.artifact_kind},
        )
        session.flush()
        return self._build_artifact_entry(session, record)

    def _replace(
        self,
        session: Session,
        *,
        session_record: SessionRecord,
        original: JourneyStageArtifactRecord,
        payload: JourneyStageArtifactPatchRequest,
        actor_user: UserRecord,
    ) -> JourneyStageArtifactEntry:
        now = utc_now()
        proposal_payload = copy.deepcopy(original.proposal_payload)
        if payload.proposal_payload is not None:
            proposal_payload = copy.deepcopy(payload.proposal_payload)
        user_patch = dict(original.user_patch)
        if payload.user_patch is not None:
            user_patch.update(payload.user_patch)
        record = JourneyStageArtifactRecord(
            workspace_id=session_record.workspace_id,
            session_id=session_record.id,
            artifact_kind=_default_artifact_kind(original.stage_key, payload.artifact_kind or original.artifact_kind),
            stage_key=original.stage_key,
            version_number=self._next_version(session, session_record=session_record, stage_key=original.stage_key),
            state=JourneyArtifactState.reviewed,
            source_action=original.source_action,
            proposal_payload=proposal_payload,
            user_patch=user_patch,
            source_stage_versions=self._coalesce_source_stage_versions(
                session,
                session_record=session_record,
                provided=payload.source_stage_versions or original.source_stage_versions,
            ),
            input_fingerprint=payload.input_fingerprint if payload.input_fingerprint is not None else original.input_fingerprint,
            context_fingerprint=(
                payload.context_fingerprint if payload.context_fingerprint is not None else original.context_fingerprint
            ),
            output_fingerprint=payload.output_fingerprint if payload.output_fingerprint is not None else original.output_fingerprint,
            corpus_hash=payload.corpus_hash if payload.corpus_hash is not None else original.corpus_hash,
            provider_key=payload.provider_key if payload.provider_key is not None else original.provider_key,
            model=payload.model if payload.model is not None else original.model,
            execution_backend=(
                payload.execution_backend if payload.execution_backend is not None else original.execution_backend
            ),
            prompt_version=payload.prompt_version if payload.prompt_version is not None else original.prompt_version,
            schema_version=payload.schema_version if payload.schema_version is not None else original.schema_version,
            confidence=payload.confidence if payload.confidence is not None else original.confidence,
            missing_information=(
                list(payload.missing_information)
                if payload.missing_information is not None
                else list(original.missing_information)
            ),
            warnings=list(dict.fromkeys(payload.warnings)) if payload.warnings is not None else list(original.warnings),
            evidence_manifest=(
                [item.model_dump(mode="json") for item in payload.evidence_manifest]
                if payload.evidence_manifest is not None
                else list(original.evidence_manifest)
            ),
            based_on_artifact_id=original.id,
            created_at=now,
            updated_at=now,
            reviewed_at=now,
        )
        self._ensure_fingerprints(record)
        session.add(record)
        session.flush()
        original.superseded_by_artifact_id = record.id
        original.updated_at = now
        session.add(original)
        self._record_decision(
            session,
            session_record=session_record,
            artifact_record=record,
            actor_user_id=actor_user.id,
            decision_type=JourneyDecisionType.replace,
            previous_state=original.state,
            next_state=record.state,
            note=payload.note,
            payload={"source_artifact_id": str(original.id)},
        )
        session_record.updated_at = now
        session.add(session_record)
        session.flush()
        self._invalidate_downstream(
            session,
            session_record=session_record,
            upstream_artifact=record,
            actor_user_id=actor_user.id,
            reason_action="regenerated",
        )
        session.flush()
        return self._build_artifact_entry(session, record)

    def _project_to_canonical(
        self,
        session: Session,
        *,
        session_record: SessionRecord,
        artifact_record: JourneyStageArtifactRecord,
        decision_payload: dict[str, Any],
    ) -> dict[str, Any]:
        if artifact_record.stage_key == "discover":
            artifact, projection_meta = _resolve_discovery_artifact_payload(artifact_record)
            record = session.exec(
                select(OpportunityRecord).where(OpportunityRecord.session_id == session_record.id)
            ).first() or OpportunityRecord(session_id=session_record.id)
            record.problem_statement = artifact.problem_statement
            record.current_user = artifact.current_user
            record.current_process = artifact.current_process
            record.desired_outcome = artifact.desired_outcome
            record.autonomy_level = artifact.autonomy_level
            record.constraints = list(artifact.constraints)
            record.operational_baseline = artifact.operational_baseline.model_dump(mode="json")
            record.mvp_definition = artifact.mvp_definition.model_dump(mode="json")
            record.case_type = artifact.case_type
            record.value_statement = artifact.value_statement
            record.updated_at = utc_now()
            session.add(record)
            session_record.title = artifact.problem_statement[:80] or session_record.title
            return {
                "projected_artifact": "opportunity",
                **projection_meta,
            }

        if artifact_record.stage_key == "define":
            artifact, definition, projection_meta = _resolve_definition_artifact_payload(
                artifact_record,
                commercial_tier=session_record.commercial_tier,
            )
            record = session.exec(select(CanvasRecord).where(CanvasRecord.session_id == session_record.id)).first() or CanvasRecord(
                session_id=session_record.id,
                user_goal="",
                success_metric="",
                primary_risk="",
            )
            record.user_goal = artifact.user_goal
            record.mvp_scope = list(artifact.mvp_scope)
            record.out_of_scope = list(artifact.out_of_scope)
            record.success_metric = artifact.success_metric
            record.primary_risk = artifact.primary_risk
            record.agent_profile = artifact.agent_profile.model_dump(mode="json")
            record.updated_at = utc_now()
            session.add(record)
            return {
                "projected_artifact": "canvas",
                "projection_mode": "definition_canvas_projection" if definition is not None else "legacy_canvas",
                **projection_meta,
            }

        if artifact_record.stage_key in {"design", "memory", "tools"}:
            return self._project_blueprint_stage(
                session,
                session_record=session_record,
                artifact_record=artifact_record,
                decision_payload=decision_payload,
            )

        if artifact_record.stage_key == "validate":
            schema_version = artifact_record.schema_version or artifact_record.proposal_payload.get("schema_version", "")
            if schema_version == "validation-simulation-spec.v1":
                artifact = SimulationSpecificationArtifact.model_validate(artifact_record.proposal_payload)
                return {
                    "projected_artifact": "validation_simulation_specification",
                    "scenario_count": len(artifact.scenarios),
                    "source_blueprint_version": artifact.source_blueprint_version,
                }
            artifact = EvaluationArtifact.model_validate(artifact_record.proposal_payload)
            record = session.exec(
                select(EvaluationRecord).where(EvaluationRecord.session_id == session_record.id)
            ).first() or EvaluationRecord(session_id=session_record.id)
            record.report = artifact.model_dump(mode="json")
            record.status = ArtifactStatus.ready
            record.updated_at = utc_now()
            session.add(record)
            return {"projected_artifact": "evaluation"}

        if artifact_record.stage_key == "estimate":
            artifact = EstimationReportArtifact.model_validate(artifact_record.proposal_payload)
            blueprint_version_number = self._latest_blueprint_version_number(session, session_id=session_record.id)
            record_estimation_artifact(
                session,
                session_id=session_record.id,
                blueprint_version_number=blueprint_version_number,
                stage=SESSION_STAGE_BY_JOURNEY_STAGE["estimate"],
                source_action=artifact_record.source_action or "approve_estimation_proposal",
                estimation_report=artifact,
            )
            persist_estimation_run(
                session,
                session_id=session_record.id,
                blueprint_version_number=blueprint_version_number,
                source_action=artifact_record.source_action or "approve_estimation_proposal",
                estimation_report=artifact,
            )
            return {
                "projected_artifact": "estimation_report",
                "blueprint_version_number": blueprint_version_number,
            }

        return {"projected_artifact": "journey_only"}

    def _project_blueprint_stage(
        self,
        session: Session,
        *,
        session_record: SessionRecord,
        artifact_record: JourneyStageArtifactRecord,
        decision_payload: dict[str, Any],
    ) -> dict[str, Any]:
        existing = session.exec(select(BlueprintRecord).where(BlueprintRecord.session_id == session_record.id)).first()
        current = _hydrate_blueprint(existing) if existing is not None else BlueprintArtifact()
        merged = current.model_dump(mode="json")
        projected_sections: list[str] = []

        if artifact_record.stage_key == "design":
            projected_design, promoted_design_artifact, projection_meta = _resolve_design_artifact_payload(
                artifact_record,
                commercial_tier=session_record.commercial_tier,
                decision_payload=decision_payload,
            )
            for key in DESIGN_BLUEPRINT_FIELDS:
                if key in projected_design:
                    merged[key] = projected_design[key]
                    projected_sections.append(key)
            if promoted_design_artifact is not None:
                artifact_record.proposal_payload = promoted_design_artifact.model_dump(mode="json")
                session.add(artifact_record)
            projection = self._persist_blueprint_projection(
                session,
                session_record=session_record,
                blueprint_payload=merged,
                existing_record=existing,
                source_action=artifact_record.source_action or "approve_design_proposal",
                projected_sections=projected_sections,
                approved_tools_digest=None,
            )
            projection.update(projection_meta)
            return projection

        elif artifact_record.stage_key == "memory":
            projected_memory, promoted_memory_artifact, projection_meta = self._resolve_memory_projection_payload(
                artifact_record=artifact_record,
                commercial_tier=session_record.commercial_tier,
            )
            for key in MEMORY_BLUEPRINT_FIELDS:
                if key in projected_memory:
                    merged[key] = projected_memory[key]
                    projected_sections.append(key)
            if promoted_memory_artifact is not None:
                artifact_record.proposal_payload = promoted_memory_artifact.model_dump(mode="json")
                session.add(artifact_record)
            projection = self._persist_blueprint_projection(
                session,
                session_record=session_record,
                blueprint_payload=merged,
                existing_record=existing,
                source_action=artifact_record.source_action or "approve_memory_profile",
                projected_sections=projected_sections,
                approved_tools_digest=None,
            )
            projection.update(projection_meta)
            return projection

        elif artifact_record.stage_key == "tools":
            approved_tools, digest, promoted_payload, recommendation = self._resolve_tools_projection_payload(
                artifact_record=artifact_record,
                decision_payload=decision_payload,
                session_record=session_record,
            )
            merged["tools"] = [item.model_dump(mode="json") for item in approved_tools]
            projected_sections.append("tools")
            projection = self._persist_blueprint_projection(
                session,
                session_record=session_record,
                blueprint_payload=merged,
                existing_record=existing,
                source_action=artifact_record.source_action or "approve_tools_proposal",
                projected_sections=projected_sections,
                approved_tools_digest=digest,
            )
            promoted_digest = digest.model_copy(update={"promoted_blueprint_version": projection["blueprint_version_number"]})
            promoted_payload["approved_tools_digest"] = promoted_digest.model_dump(mode="json")
            artifact_record.proposal_payload = promoted_payload
            session.add(artifact_record)
            projection["approved_tools_digest"] = promoted_digest.model_dump(mode="json")
            if recommendation is not None:
                recommendation = recommendation.model_copy(update={"approved_tools_digest": promoted_digest})
            self._record_tool_recommendation_legacy_artifact(
                session,
                session_record=session_record,
                recommendation=recommendation,
                blueprint_version_number=projection["blueprint_version_number"],
            )
            return projection

        return self._persist_blueprint_projection(
            session,
            session_record=session_record,
            blueprint_payload=merged,
            existing_record=existing,
            source_action=artifact_record.source_action or f"approve_{artifact_record.stage_key}_proposal",
            projected_sections=projected_sections,
            approved_tools_digest=None,
        )

    def _persist_blueprint_projection(
        self,
        session: Session,
        *,
        session_record: SessionRecord,
        blueprint_payload: dict[str, Any],
        existing_record: BlueprintRecord | None,
        source_action: str,
        projected_sections: list[str],
        approved_tools_digest: ApprovedToolsDigest | None,
    ) -> dict[str, Any]:
        artifact = BlueprintArtifact.model_validate(blueprint_payload)
        record = existing_record or BlueprintRecord(
            session_id=session_record.id,
            architecture="",
            reasoning_pattern="",
            memory_strategy="",
            narrative="",
        )
        _assign_blueprint_record(record, artifact)
        session.add(record)
        blueprint_version_number = self._create_blueprint_version(
            session,
            session_id=session_record.id,
            source_action=source_action,
            blueprint=artifact,
        )
        return {
            "projected_artifact": "blueprint",
            "projected_sections": projected_sections,
            "blueprint_version_number": blueprint_version_number,
            "approved_tools_digest": approved_tools_digest.model_dump(mode="json")
            if approved_tools_digest is not None
            else None,
        }

    def _resolve_tools_projection_payload(
        self,
        *,
        artifact_record: JourneyStageArtifactRecord,
        decision_payload: dict[str, Any],
        session_record: SessionRecord,
    ) -> tuple[list[BlueprintTool], ApprovedToolsDigest, dict[str, Any], ToolRecommendationArtifact | None]:
        if artifact_record.proposal_payload.get("schema_version") == "tool-recommendation.v1":
            artifact = evaluate_tool_recommendation_artifact(
                ToolRecommendationArtifact.model_validate(artifact_record.proposal_payload)
            )
            approved_tools, review_decisions, digest = promote_tool_recommendation_to_blueprint_tools(
                artifact,
                include_optional_tool_keys=list(decision_payload.get("include_optional_tool_keys", [])),
            )
            promoted = artifact.model_copy(
                update={
                    "review_decisions": review_decisions,
                    "approved_tools_digest": digest,
                    "review_state": ReviewState.complete
                    if not artifact.coverage_gaps and not artifact.needs_information
                    else ReviewState.partial,
                }
            )
            return approved_tools, digest, promoted.model_dump(mode="json"), promoted

        approved_tools = [
            BlueprintTool.model_validate(item) for item in artifact_record.proposal_payload.get("approved_tools", [])
        ]
        digest_payload = artifact_record.proposal_payload.get("approved_tools_digest")
        digest = (
            ApprovedToolsDigest.model_validate(digest_payload)
            if digest_payload
            else build_approved_tools_digest_from_blueprint_tools(
                approved_tools,
                source_session_id=session_record.id,
            )
        )
        promoted_payload = copy.deepcopy(artifact_record.proposal_payload)
        promoted_payload["approved_tools_digest"] = digest.model_dump(mode="json")
        return approved_tools, digest, promoted_payload, None

    def _resolve_memory_projection_payload(
        self,
        *,
        artifact_record: JourneyStageArtifactRecord,
        commercial_tier: CommercialTier | str = CommercialTier.blueprint,
    ) -> tuple[dict[str, Any], MemoryRecommendationArtifact | None, dict[str, Any]]:
        if artifact_record.proposal_payload.get("schema_version") == "memory-recommendation.v1":
            artifact = MemoryRecommendationArtifact.model_validate(artifact_record.proposal_payload)
            approval_blockers = _memory_approval_blocking_issues(artifact, commercial_tier)
            if approval_blockers:
                raise StageProposalConflictError(
                    "Memory still has blocking issues for the active product tier. Resolve them before approval."
                )
            projection = {
                "memory_strategy": artifact.proposed_memory_profile.strategy,
                "memory_profile": artifact.proposed_memory_profile.model_dump(mode="json"),
                "knowledge_profile": artifact.proposed_knowledge_profile.model_dump(mode="json"),
            }
            promoted = artifact.model_copy(
                update={
                    "review_state": ReviewState.complete
                    if not artifact.missing_information and not artifact.critic_findings
                    else ReviewState.partial,
                }
            )
            return projection, promoted, {
                "projection_mode": "memory_recommendation_artifact",
                "critic_finding_count": len(artifact.critic_findings),
                "tool_dependency_count": len(artifact.tool_dependencies),
                "dry_compile_status": artifact.dry_compile_status.status,
                "approval_blocking_issue_count": len(approval_blockers),
            }

        payload = artifact_record.proposal_payload
        projection = {key: payload.get(key) for key in MEMORY_BLUEPRINT_FIELDS if key in payload}
        return projection, None, {"projection_mode": "legacy_memory_sections"}

    def _record_tool_recommendation_legacy_artifact(
        self,
        session: Session,
        *,
        session_record: SessionRecord,
        recommendation: ToolRecommendationArtifact | None,
        blueprint_version_number: int,
    ) -> None:
        if recommendation is None:
            return
        payload = _serialize_tool_recommendation(recommendation)
        session.add(
            ArtifactRegistryRecord(
                session_id=session_record.id,
                blueprint_version_number=blueprint_version_number,
                artifact_key="tools/tool-recommendation.v1.json",
                artifact_title="Tool recommendation",
                artifact_kind="tool_recommendation",
                stage=SESSION_STAGE_BY_JOURNEY_STAGE["tools"],
                source_action="approve_tools_proposal",
                export_format="json",
                content_text=payload,
                content_hash=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                artifact_metadata={
                    "schema_version": recommendation.schema_version,
                    "recommended_count": len(recommendation.recommended_tools),
                    "optional_count": len(recommendation.optional_tools),
                    "approved_tools_digest_present": recommendation.approved_tools_digest is not None,
                },
            )
        )

    def _latest_record(
        self,
        session: Session,
        *,
        session_record: SessionRecord,
        stage_key: str,
    ) -> JourneyStageArtifactRecord | None:
        normalized_stage = _journey_stage_or_raise(stage_key)
        return session.exec(
            select(JourneyStageArtifactRecord)
            .where(
                JourneyStageArtifactRecord.workspace_id == session_record.workspace_id,
                JourneyStageArtifactRecord.session_id == session_record.id,
                JourneyStageArtifactRecord.stage_key == normalized_stage,
            )
            .order_by(JourneyStageArtifactRecord.version_number.desc(), JourneyStageArtifactRecord.created_at.desc())
        ).first()

    def _artifact_or_raise(
        self,
        session: Session,
        *,
        session_record: SessionRecord,
        stage_key: str,
        artifact_id: UUID,
    ) -> JourneyStageArtifactRecord:
        normalized_stage = _journey_stage_or_raise(stage_key)
        record = session.get(JourneyStageArtifactRecord, artifact_id)
        if (
            record is None
            or record.workspace_id != session_record.workspace_id
            or record.session_id != session_record.id
            or record.stage_key != normalized_stage
        ):
            raise StageProposalNotFoundError("Journey stage artifact not found.")
        return record

    def _next_version(self, session: Session, *, session_record: SessionRecord, stage_key: str) -> int:
        latest = self._latest_record(session, session_record=session_record, stage_key=stage_key)
        return 1 if latest is None else latest.version_number + 1

    def _build_artifact_entries(
        self,
        session: Session,
        records: list[JourneyStageArtifactRecord],
    ) -> list[JourneyStageArtifactEntry]:
        if not records:
            return []
        artifact_ids = [item.id for item in records]
        decision_records = session.exec(
            select(JourneyStageDecisionRecord)
            .where(JourneyStageDecisionRecord.artifact_id.in_(artifact_ids))
            .order_by(JourneyStageDecisionRecord.created_at.asc())
        ).all()
        decisions_by_artifact: dict[UUID, list[JourneyStageDecisionRecord]] = {}
        for decision in decision_records:
            decisions_by_artifact.setdefault(decision.artifact_id, []).append(decision)
        return [
            self._build_artifact_entry(session, item, decisions=decisions_by_artifact.get(item.id, [])) for item in records
        ]

    def _build_artifact_entry(
        self,
        session: Session,
        record: JourneyStageArtifactRecord | None,
        *,
        decisions: list[JourneyStageDecisionRecord] | None = None,
    ) -> JourneyStageArtifactEntry | None:
        if record is None:
            return None
        if decisions is None:
            decisions = session.exec(
                select(JourneyStageDecisionRecord)
                .where(JourneyStageDecisionRecord.artifact_id == record.id)
                .order_by(JourneyStageDecisionRecord.created_at.asc())
            ).all()
        return JourneyStageArtifactEntry(
            id=record.id,
            workspace_id=record.workspace_id,
            session_id=record.session_id,
            artifact_kind=record.artifact_kind,
            stage_key=record.stage_key,
            version_number=record.version_number,
            state=record.state,
            source_action=record.source_action,
            proposal_payload=record.proposal_payload,
            user_patch=record.user_patch,
            source_stage_versions=record.source_stage_versions,
            input_fingerprint=record.input_fingerprint,
            context_fingerprint=record.context_fingerprint,
            output_fingerprint=record.output_fingerprint,
            corpus_hash=record.corpus_hash,
            provider_key=record.provider_key,
            model=record.model,
            execution_backend=record.execution_backend,
            prompt_version=record.prompt_version,
            schema_version=record.schema_version,
            confidence=record.confidence,
            missing_information=record.missing_information,
            warnings=record.warnings,
            evidence_manifest=record.evidence_manifest,
            stale_reasons=record.stale_reasons,
            based_on_artifact_id=record.based_on_artifact_id,
            superseded_by_artifact_id=record.superseded_by_artifact_id,
            approved_by_user_id=record.approved_by_user_id,
            reviewed_at=record.reviewed_at,
            approved_at=record.approved_at,
            rejected_at=record.rejected_at,
            stale_at=record.stale_at,
            created_at=record.created_at,
            updated_at=record.updated_at,
            decisions=[
                JourneyStageDecisionEntry(
                    id=item.id,
                    artifact_id=item.artifact_id,
                    stage_key=item.stage_key,
                    decision_type=item.decision_type,
                    previous_state=item.previous_state,
                    next_state=item.next_state,
                    actor_user_id=item.actor_user_id,
                    note=item.note,
                    payload=item.payload,
                    created_at=item.created_at,
                )
                for item in decisions
            ],
        )

    def _record_decision(
        self,
        session: Session,
        *,
        session_record: SessionRecord,
        artifact_record: JourneyStageArtifactRecord,
        actor_user_id: UUID | None,
        decision_type: JourneyDecisionType,
        previous_state: JourneyArtifactState | None,
        next_state: JourneyArtifactState | None,
        note: str,
        payload: dict[str, Any],
    ) -> None:
        session.add(
            JourneyStageDecisionRecord(
                workspace_id=session_record.workspace_id,
                session_id=session_record.id,
                artifact_id=artifact_record.id,
                stage_key=artifact_record.stage_key,
                decision_type=decision_type,
                previous_state=previous_state,
                next_state=next_state,
                actor_user_id=actor_user_id,
                note=note,
                payload=payload,
            )
        )

    def _ensure_fingerprints(self, record: JourneyStageArtifactRecord) -> None:
        record.input_fingerprint = record.input_fingerprint or _stable_hash(
            {
                "stage_key": record.stage_key,
                "artifact_kind": record.artifact_kind,
                "source_stage_versions": record.source_stage_versions,
                "proposal_payload": record.proposal_payload,
            }
        )
        record.context_fingerprint = record.context_fingerprint or _stable_hash(
            {
                "stage_key": record.stage_key,
                "source_stage_versions": record.source_stage_versions,
                "evidence_manifest": record.evidence_manifest,
                "warnings": record.warnings,
            }
        )
        record.output_fingerprint = record.output_fingerprint or _stable_hash(record.proposal_payload)
        record.schema_version = record.schema_version or "journey-stage-artifact.v1"

    def _derive_source_stage_versions(
        self,
        session: Session,
        *,
        session_record: SessionRecord,
    ) -> dict[str, Any]:
        versions: dict[str, Any] = {}
        for stage_key in STAGE_ORDER:
            latest = session.exec(
                select(JourneyStageArtifactRecord)
                .where(
                    JourneyStageArtifactRecord.workspace_id == session_record.workspace_id,
                    JourneyStageArtifactRecord.session_id == session_record.id,
                    JourneyStageArtifactRecord.stage_key == stage_key,
                    JourneyStageArtifactRecord.state.in_(tuple(APPROVED_ARTIFACT_STATES)),
                )
                .order_by(JourneyStageArtifactRecord.version_number.desc())
            ).first()
            if latest is not None:
                versions[stage_key] = latest.version_number
        return versions

    def _ensure_required_predecessors_approved(
        self,
        session: Session,
        *,
        session_record: SessionRecord,
        stage_key: str,
    ) -> None:
        boundary = get_journey_stage_boundary(stage_key)
        missing: list[str] = []
        for predecessor in boundary.required_predecessors:
            approved = session.exec(
                select(JourneyStageArtifactRecord)
                .where(
                    JourneyStageArtifactRecord.workspace_id == session_record.workspace_id,
                    JourneyStageArtifactRecord.session_id == session_record.id,
                    JourneyStageArtifactRecord.stage_key == predecessor,
                    JourneyStageArtifactRecord.state.in_(tuple(APPROVED_ARTIFACT_STATES)),
                )
                .order_by(JourneyStageArtifactRecord.version_number.desc())
            ).first()
            if approved is None:
                missing.append(predecessor)
        if missing:
            raise StageProposalConflictError(
                f"Cannot approve '{stage_key}' before approved predecessors exist: {', '.join(missing)}."
            )

    def _coalesce_source_stage_versions(
        self,
        session: Session,
        *,
        session_record: SessionRecord,
        provided: dict[str, Any],
    ) -> dict[str, Any]:
        merged = self._derive_source_stage_versions(session, session_record=session_record)
        merged.update(copy.deepcopy(provided))
        return merged

    def _mark_prior_versions_superseded(
        self,
        session: Session,
        approved_record: JourneyStageArtifactRecord,
    ) -> None:
        now = utc_now()
        prior_records = session.exec(
            select(JourneyStageArtifactRecord).where(
                JourneyStageArtifactRecord.workspace_id == approved_record.workspace_id,
                JourneyStageArtifactRecord.session_id == approved_record.session_id,
                JourneyStageArtifactRecord.stage_key == approved_record.stage_key,
                JourneyStageArtifactRecord.version_number < approved_record.version_number,
            )
        ).all()
        for item in prior_records:
            if item.superseded_by_artifact_id is None and item.state in APPROVED_ARTIFACT_STATES:
                item.superseded_by_artifact_id = approved_record.id
                item.updated_at = now
                session.add(item)

    def _invalidate_downstream(
        self,
        session: Session,
        *,
        session_record: SessionRecord,
        upstream_artifact: JourneyStageArtifactRecord,
        actor_user_id: UUID,
        reason_action: str = "approved",
    ) -> None:
        upstream_index = STAGE_ORDER.index(upstream_artifact.stage_key)
        downstream_stage_keys = STAGE_ORDER[upstream_index + 1 :]
        now = utc_now()
        for stage_key in downstream_stage_keys:
            records = session.exec(
                select(JourneyStageArtifactRecord).where(
                    JourneyStageArtifactRecord.workspace_id == session_record.workspace_id,
                    JourneyStageArtifactRecord.session_id == session_record.id,
                    JourneyStageArtifactRecord.stage_key == stage_key,
                )
            ).all()
            for item in records:
                previous_state = item.state
                reason = f"upstream_{upstream_artifact.stage_key}_artifact_v{upstream_artifact.version_number}_{reason_action}"
                reason_missing = reason not in item.stale_reasons
                if reason not in item.stale_reasons:
                    item.stale_reasons = [*item.stale_reasons, reason]
                changed = reason_missing or item.state != JourneyArtifactState.stale
                if item.state != JourneyArtifactState.stale:
                    item.state = JourneyArtifactState.stale
                    item.stale_at = now
                if not changed:
                    continue
                item.updated_at = now
                session.add(item)
                self._record_decision(
                    session,
                    session_record=session_record,
                    artifact_record=item,
                    actor_user_id=actor_user_id,
                    decision_type=JourneyDecisionType.mark_stale,
                    previous_state=previous_state,
                    next_state=item.state,
                    note=f"{upstream_artifact.stage_key} {reason_action} version {upstream_artifact.version_number}",
                    payload={
                        "upstream_stage": upstream_artifact.stage_key,
                        "upstream_artifact_id": str(upstream_artifact.id),
                        "upstream_version_number": upstream_artifact.version_number,
                        "reason_action": reason_action,
                    },
                )

    def _create_blueprint_version(
        self,
        session: Session,
        *,
        session_id: UUID,
        source_action: str,
        blueprint: BlueprintArtifact,
    ) -> int:
        latest = session.exec(
            select(BlueprintVersionRecord)
            .where(BlueprintVersionRecord.session_id == session_id)
            .order_by(BlueprintVersionRecord.version_number.desc())
        ).first()
        version_number = 1 if latest is None else latest.version_number + 1
        session.add(
            BlueprintVersionRecord(
                session_id=session_id,
                version_number=version_number,
                source_action=source_action,
                status=ArtifactStatus.ready,
                blueprint_snapshot=blueprint.model_dump(mode="json"),
            )
        )
        session.flush()
        return version_number

    def _latest_blueprint_version_number(self, session: Session, *, session_id: UUID) -> int | None:
        latest = session.exec(
            select(BlueprintVersionRecord)
            .where(BlueprintVersionRecord.session_id == session_id)
            .order_by(BlueprintVersionRecord.version_number.desc())
        ).first()
        return latest.version_number if latest is not None else None
