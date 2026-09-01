from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.models import (
    ArtifactStatus,
    CommercialEventRecord,
    CommercialTier,
    ConstructionQuestionResponseRecord,
    JourneyArtifactState,
    JourneyStageArtifactRecord,
    JourneyStageDecisionRecord,
    SessionRecord,
    SessionStage,
    UserRecord,
    WorkspaceRecord,
)
from app.services.acp_handoff_service import (
    BLUEPRINT_ACP_HANDOFF_EVENT_KEY,
    finalize_blueprint_for_acp_handoff,
    has_blueprint_acp_handoff_finalized,
)


def build_session() -> tuple[Session, SessionRecord, UserRecord]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    db = Session(engine)
    workspace = WorkspaceRecord(name="LAB Test Workspace", slug="lab-test-workspace")
    user = UserRecord(email="owner@example.com", password_hash="hash", full_name="Owner")
    db.add(workspace)
    db.add(user)
    db.flush()
    session_record = SessionRecord(
        user_id=user.id,
        workspace_id=workspace.id,
        title="Proyecto ACP",
        status=ArtifactStatus.ready,
        current_stage=SessionStage.ready_for_export,
        commercial_tier=CommercialTier.blueprint_pro,
    )
    db.add(session_record)
    db.flush()
    return db, session_record, user


def test_finalize_blueprint_for_acp_handoff_closes_process_stale_artifacts() -> None:
    db, session_record, user = build_session()
    stale_artifact = JourneyStageArtifactRecord(
        workspace_id=session_record.workspace_id,
        session_id=session_record.id,
        artifact_kind="tool_recommendation_artifact",
        stage_key="tools",
        version_number=1,
        state=JourneyArtifactState.stale,
        stale_reasons=["tool_recommendation_context_changed"],
    )
    design_artifact = JourneyStageArtifactRecord(
        workspace_id=session_record.workspace_id,
        session_id=session_record.id,
        artifact_kind="design_recommendation_artifact",
        stage_key="design",
        version_number=1,
        state=JourneyArtifactState.stale,
        stale_reasons=["real_design_drift"],
    )
    db.add(stale_artifact)
    db.add(design_artifact)
    db.commit()

    result = finalize_blueprint_for_acp_handoff(
        db,
        session_record=session_record,
        actor_user_id=user.id,
        source="test",
        correlation_id="handoff-test",
    )
    db.commit()

    db.refresh(stale_artifact)
    db.refresh(design_artifact)
    assert result["status"] == "finalized"
    assert stale_artifact.state == JourneyArtifactState.approved
    assert stale_artifact.stale_reasons == []
    assert stale_artifact.stale_at is None
    assert design_artifact.state == JourneyArtifactState.stale
    assert design_artifact.stale_reasons == ["real_design_drift"]
    assert has_blueprint_acp_handoff_finalized(db, session_id=session_record.id) is True

    event = db.exec(
        select(CommercialEventRecord).where(
            CommercialEventRecord.session_id == session_record.id,
            CommercialEventRecord.event_key == BLUEPRINT_ACP_HANDOFF_EVENT_KEY,
        )
    ).one()
    assert event.product_key == "acp"
    assert event.metadata_payload["closed_process_items"][0]["stage_key"] == "tools"

    decision = db.exec(
        select(JourneyStageDecisionRecord).where(
            JourneyStageDecisionRecord.artifact_id == stale_artifact.id,
        )
    ).one()
    assert decision.payload["debt_kind"] == "process_debt"


def test_finalize_blueprint_for_acp_handoff_is_idempotent() -> None:
    db, session_record, user = build_session()

    first = finalize_blueprint_for_acp_handoff(
        db,
        session_record=session_record,
        actor_user_id=user.id,
        source="test",
        correlation_id="handoff-test",
    )
    second = finalize_blueprint_for_acp_handoff(
        db,
        session_record=session_record,
        actor_user_id=user.id,
        source="test",
        correlation_id="handoff-test",
    )
    db.commit()

    events = db.exec(
        select(CommercialEventRecord).where(
            CommercialEventRecord.session_id == session_record.id,
            CommercialEventRecord.event_key == BLUEPRINT_ACP_HANDOFF_EVENT_KEY,
        )
    ).all()
    assert first["status"] == "finalized"
    assert second["status"] == "already_finalized"
    assert len(events) == 1


def test_finalize_blueprint_for_acp_handoff_preserves_real_delegated_questions() -> None:
    db, session_record, user = build_session()
    delegated_question = ConstructionQuestionResponseRecord(
        session_id=session_record.id,
        question_key="deployment_target",
        gap_key="deployment_target_unknown",
        gap_title="Definir entorno de despliegue",
        domain="deployment",
        question_text="Donde se desplegara el agente?",
        blocking=True,
        status="deferred",
        answer_text="Delegado a implementacion.",
        impacted_artifacts=["ACP/deployment/docker-compose.yaml"],
    )
    db.add(delegated_question)
    db.commit()

    result = finalize_blueprint_for_acp_handoff(
        db,
        session_record=session_record,
        actor_user_id=user.id,
        source="manual_approval",
        correlation_id="manual-handoff",
    )
    db.commit()
    db.refresh(delegated_question)

    assert result["status"] == "finalized"
    assert delegated_question.status == "deferred"
    assert delegated_question.answer_text == "Delegado a implementacion."
    event = db.exec(
        select(CommercialEventRecord).where(
            CommercialEventRecord.session_id == session_record.id,
            CommercialEventRecord.event_key == BLUEPRINT_ACP_HANDOFF_EVENT_KEY,
        )
    ).one()
    assert "construction_readiness_gaps" in event.metadata_payload["preserved_real_debt"]
