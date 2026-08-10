from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import UUID

from sqlalchemy import inspect
from sqlmodel import SQLModel, Session, select

from app.models import (
    ArtifactStatus,
    ArtifactRegistryRecord,
    BlueprintRecord,
    BlueprintTool,
    CanvasRecord,
    EstimationReportArtifact,
    EvaluationRecord,
    JourneyArtifactState,
    JourneyStageArtifactRecord,
    JourneyStageDecisionRecord,
    OpportunityRecord,
    ReviewState,
    SchemaMigrationRecord,
    SessionRecord,
)
from app.services.stage_proposal_service import StageProposalService
from app.services.tool_recommendation_service import build_approved_tools_digest_from_blueprint_tools


MIGRATION_KEY_JOURNEY_STAGE_CI1 = "2026-07-22-ci1-journey-stage-lifecycle"


@dataclass
class JourneyStageMigrationSummary:
    migration_key: str = MIGRATION_KEY_JOURNEY_STAGE_CI1
    already_recorded: bool = False
    created_tables: list[str] = field(default_factory=list)
    sessions_scanned: int = 0
    artifacts_backfilled: int = 0
    stages_backfilled: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JourneyStageMigrationService:
    def __init__(self) -> None:
        self._proposal_service = StageProposalService()

    def apply(self, session: Session) -> JourneyStageMigrationSummary:
        summary = JourneyStageMigrationSummary()
        bind = session.get_bind()
        if bind is None:
            summary.notes.append("No database bind available for CI1 journey migration.")
            return summary

        inspector = inspect(bind)
        existing_tables = set(inspector.get_table_names())
        required_tables = {
            JourneyStageArtifactRecord.__tablename__,
            JourneyStageDecisionRecord.__tablename__,
        }
        missing_tables = sorted(required_tables - existing_tables)
        if missing_tables:
            SQLModel.metadata.create_all(
                bind,
                tables=[
                    JourneyStageArtifactRecord.__table__,
                    JourneyStageDecisionRecord.__table__,
                ],
            )
            summary.created_tables.extend(missing_tables)

        migration_record = session.exec(
            select(SchemaMigrationRecord).where(
                SchemaMigrationRecord.migration_key == MIGRATION_KEY_JOURNEY_STAGE_CI1
            )
        ).first()
        if migration_record is not None:
            summary.already_recorded = True

        session_records = session.exec(select(SessionRecord)).all()
        summary.sessions_scanned = len(session_records)
        for session_record in session_records:
            counts = self.backfill_session(session, session_record=session_record)
            summary.artifacts_backfilled += sum(counts.values())
            for stage_key, count in counts.items():
                summary.stages_backfilled[stage_key] = summary.stages_backfilled.get(stage_key, 0) + count

        if migration_record is None:
            session.add(
                SchemaMigrationRecord(
                    migration_key=MIGRATION_KEY_JOURNEY_STAGE_CI1,
                    description=(
                        "Crea el ciclo de vida de artefactos por etapa, backfillea discovery/canvas/blueprint/tools/"
                        "memory/evaluation/estimation y deja compatibilidad con las tablas canonicas legacy."
                    ),
                )
            )
        session.commit()
        return summary

    def backfill_session(
        self,
        session: Session,
        *,
        session_record: SessionRecord,
    ) -> dict[str, int]:
        counts = {key: 0 for key in ("discover", "define", "design", "tools", "memory", "validate", "estimate")}

        discovery = session.exec(
            select(OpportunityRecord).where(OpportunityRecord.session_id == session_record.id)
        ).first()
        if (
            discovery is not None
            and self._can_backfill_discovery(discovery)
            and not self._has_stage_artifact(session, session_record=session_record, stage_key="discover")
        ):
            self._proposal_service.backfill_legacy(
                session,
                session_record=session_record,
                stage_key="discover",
                artifact_kind="discovery_artifact",
                proposal_payload={
                    key: value
                    for key, value in discovery.model_dump(exclude={"id", "session_id", "updated_at"}).items()
                },
                state=JourneyArtifactState.approved_legacy,
                source_action="legacy_backfill_discovery",
                note="Backfill legacy discovery into CI1 lifecycle.",
            )
            counts["discover"] += 1

        canvas_record = session.exec(
            select(CanvasRecord).where(CanvasRecord.session_id == session_record.id)
        ).first()
        if canvas_record is not None and not self._has_stage_artifact(session, session_record=session_record, stage_key="define"):
            self._proposal_service.backfill_legacy(
                session,
                session_record=session_record,
                stage_key="define",
                artifact_kind="canvas_artifact",
                proposal_payload=canvas_record.model_dump(exclude={"id", "session_id", "updated_at"}),
                state=JourneyArtifactState.approved_legacy,
                source_action="legacy_backfill_canvas",
                note="Backfill legacy canvas into CI1 lifecycle.",
            )
            counts["define"] += 1

        blueprint = session.exec(
            select(BlueprintRecord).where(BlueprintRecord.session_id == session_record.id)
        ).first()
        if blueprint is not None:
            readiness_state = blueprint.readiness_state
            default_state = (
                JourneyArtifactState.approved_legacy
                if readiness_state == ReviewState.complete
                else JourneyArtifactState.needs_review_legacy
            )
            if not self._has_stage_artifact(session, session_record=session_record, stage_key="design"):
                self._proposal_service.backfill_legacy(
                    session,
                    session_record=session_record,
                    stage_key="design",
                    artifact_kind="design_recommendation_artifact",
                    proposal_payload={
                        "architecture": blueprint.architecture,
                        "reasoning_pattern": blueprint.reasoning_pattern,
                        "safety_checks": blueprint.safety_checks,
                        "guardrails": blueprint.guardrails,
                        "narrative": blueprint.narrative,
                    },
                    state=default_state,
                    source_action="legacy_backfill_design",
                    note="Backfill legacy blueprint design-owned sections into CI1 lifecycle.",
                )
                counts["design"] += 1

            if not self._has_stage_artifact(session, session_record=session_record, stage_key="memory"):
                self._proposal_service.backfill_legacy(
                    session,
                    session_record=session_record,
                    stage_key="memory",
                    artifact_kind="memory_recommendation_artifact",
                    proposal_payload={
                        "memory_strategy": blueprint.memory_strategy,
                        "memory_profile": blueprint.memory_profile,
                        "knowledge_profile": blueprint.knowledge_profile,
                    },
                    state=default_state,
                    source_action="legacy_backfill_memory",
                    note="Backfill legacy blueprint memory-owned sections into CI1 lifecycle.",
                )
                counts["memory"] += 1

        if not self._has_stage_artifact(session, session_record=session_record, stage_key="tools"):
            tool_payload, tool_state = self._load_legacy_tools_payload(session, session_id=session_record.id)
            if tool_payload is not None:
                self._proposal_service.backfill_legacy(
                    session,
                    session_record=session_record,
                    stage_key="tools",
                    artifact_kind="tool_recommendation_artifact",
                    proposal_payload=tool_payload,
                    state=tool_state,
                    source_action="legacy_backfill_tools",
                    note="Backfill legacy tools recommendation/selection into CI1 lifecycle.",
                )
                counts["tools"] += 1

        evaluation = session.exec(
            select(EvaluationRecord).where(EvaluationRecord.session_id == session_record.id)
        ).first()
        if evaluation is not None and not self._has_stage_artifact(session, session_record=session_record, stage_key="validate"):
            evaluation_state = (
                JourneyArtifactState.approved_legacy
                if evaluation.status == ArtifactStatus.ready
                else JourneyArtifactState.needs_review_legacy
            )
            self._proposal_service.backfill_legacy(
                session,
                session_record=session_record,
                stage_key="validate",
                artifact_kind="evaluation_artifact",
                proposal_payload=evaluation.report,
                state=evaluation_state,
                source_action="legacy_backfill_validation",
                note="Backfill legacy evaluation report into CI1 lifecycle.",
            )
            counts["validate"] += 1

        if not self._has_stage_artifact(session, session_record=session_record, stage_key="estimate"):
            estimation_record = self._latest_artifact_record(
                session,
                session_id=session_record.id,
                artifact_kind="estimation_report",
            )
            if estimation_record is not None and estimation_record.content_text.strip():
                estimation = EstimationReportArtifact.model_validate(json.loads(estimation_record.content_text))
                estimation_state = (
                    JourneyArtifactState.approved_legacy
                    if estimation.confidence.blocking_gaps == 0 and not estimation.is_stale
                    else JourneyArtifactState.needs_review_legacy
                )
                self._proposal_service.backfill_legacy(
                    session,
                    session_record=session_record,
                    stage_key="estimate",
                    artifact_kind="estimation_report_artifact",
                    proposal_payload=estimation.model_dump(mode="json"),
                    state=estimation_state,
                    source_action="legacy_backfill_estimation",
                    note="Backfill legacy estimation report into CI1 lifecycle.",
                )
                counts["estimate"] += 1

        return counts

    def _can_backfill_discovery(self, discovery: OpportunityRecord) -> bool:
        operational_baseline = discovery.operational_baseline if isinstance(discovery.operational_baseline, dict) else {}
        mvp_definition = discovery.mvp_definition if isinstance(discovery.mvp_definition, dict) else {}

        required_scalars = (
            discovery.problem_statement,
            discovery.current_user,
            discovery.current_process,
            discovery.desired_outcome,
            operational_baseline.get("current_time_spent", ""),
            operational_baseline.get("current_cost", ""),
            mvp_definition.get("north_star_metric", ""),
        )
        if not all(isinstance(value, str) and value.strip() for value in required_scalars):
            return False

        required_lists = (
            operational_baseline.get("frequent_errors", []),
            operational_baseline.get("automation_opportunities", []),
            mvp_definition.get("v1_scope", []),
            mvp_definition.get("out_of_scope", []),
            mvp_definition.get("non_delegable_decisions", []),
        )
        return all(isinstance(items, list) and len(items) > 0 for items in required_lists)

    def _has_stage_artifact(self, session: Session, *, session_record: SessionRecord, stage_key: str) -> bool:
        record = session.exec(
            select(JourneyStageArtifactRecord).where(
                JourneyStageArtifactRecord.workspace_id == session_record.workspace_id,
                JourneyStageArtifactRecord.session_id == session_record.id,
                JourneyStageArtifactRecord.stage_key == stage_key,
            )
        ).first()
        return record is not None

    def _latest_artifact_record(
        self,
        session: Session,
        *,
        session_id: UUID,
        artifact_kind: str,
    ) -> ArtifactRegistryRecord | None:
        return session.exec(
            select(ArtifactRegistryRecord)
            .where(
                ArtifactRegistryRecord.session_id == session_id,
                ArtifactRegistryRecord.artifact_kind == artifact_kind,
            )
            .order_by(ArtifactRegistryRecord.created_at.desc())
        ).first()

    def _load_legacy_tools_payload(
        self,
        session: Session,
        *,
        session_id: UUID,
    ) -> tuple[dict[str, Any] | None, JourneyArtifactState]:
        recommendation_record = self._latest_artifact_record(
            session,
            session_id=session_id,
            artifact_kind="tool_recommendation",
        )
        if recommendation_record is not None and recommendation_record.content_text.strip():
            recommendation_payload = json.loads(recommendation_record.content_text)
            review_state = recommendation_payload.get("review_state")
            tool_state = (
                JourneyArtifactState.approved_legacy
                if review_state == ReviewState.complete or review_state == "complete"
                else JourneyArtifactState.needs_review_legacy
            )
            return recommendation_payload, tool_state

        blueprint = session.exec(select(BlueprintRecord).where(BlueprintRecord.session_id == session_id)).first()
        if blueprint is None or not blueprint.tools:
            return None, JourneyArtifactState.needs_review_legacy

        digest = build_approved_tools_digest_from_blueprint_tools(
            [BlueprintTool.model_validate(item) for item in blueprint.tools],
            source_session_id=session_id,
        )
        return {
            "approved_tools": blueprint.tools,
            "approved_tools_digest": digest.model_dump(mode="json"),
        }, JourneyArtifactState.needs_review_legacy
