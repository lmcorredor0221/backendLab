from __future__ import annotations

from uuid import uuid4

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from app.models import (
    ArtifactRegistryRecord,
    ConstructionQuestionResponseRecord,
    JourneyArtifactState,
    JourneyStageArtifactRecord,
    SessionRecord,
    SessionStage,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceRecord,
    WorkspaceRole,
)
from app.services.auth_service import hash_password
from app.services.product_processing.approved_context_service import build_approved_deliverable_context


def _engine():
    return create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)


def _seed_session(db: Session) -> SessionRecord:
    user = UserRecord(
        email=f"context-{uuid4()}@leanbuilder.local",
        full_name="Context Tester",
        password_hash=hash_password("Secret123!"),
    )
    db.add(user)
    db.flush()
    workspace = WorkspaceRecord(
        name="Context Workspace",
        slug=f"context-{str(user.id)[:8]}",
        created_by_user_id=user.id,
    )
    db.add(workspace)
    db.flush()
    db.add(WorkspaceMembershipRecord(workspace_id=workspace.id, user_id=user.id, role=WorkspaceRole.owner))
    record = SessionRecord(
        user_id=user.id,
        workspace_id=workspace.id,
        title="Context Project",
        current_stage=SessionStage.ready_for_export,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def test_stage_context_refs_resolve_to_approved_journey_artifacts() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        record = _seed_session(db)
        db.add(
            JourneyStageArtifactRecord(
                workspace_id=record.workspace_id,
                session_id=record.id,
                artifact_kind="definition_artifact",
                stage_key="define",
                version_number=1,
                state=JourneyArtifactState.approved,
                proposal_payload={
                    "schema_version": "definition-artifact.v1",
                    "functional_requirements": [
                        {"id": "FR-001", "description": "Normalize operational questions."}
                    ],
                },
                source_action="unit_test",
            )
        )
        db.commit()

        context, refs = build_approved_deliverable_context(
            db,
            record=record,
            deliverable_key="definition.requirements",
        )

    assert refs
    assert refs[0].startswith("journey:")
    assert context["approved_context"]["stages"]["define"]["functional_requirements"][0]["id"] == "FR-001"
    assert "stage.define" in context["context_policy"]["requested_refs"]


def test_artifact_refs_resolve_generated_records_by_deliverable_key_metadata() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        record = _seed_session(db)
        db.add(
            ArtifactRegistryRecord(
                session_id=record.id,
                artifact_key="Blueprint/architecture/architecture.md",
                artifact_title="Arquitectura del agente",
                artifact_kind="contract",
                content_text="Arquitectura aprobada para el agente.",
                artifact_metadata={"deliverable_key": "blueprint.architecture_spec"},
                source_action="unit_test",
            )
        )
        db.commit()

        context, refs = build_approved_deliverable_context(
            db,
            record=record,
            deliverable_key="diagram.prompt_reasoning_playbook",
        )

    assert refs
    assert refs[0].startswith("artifact:")
    assert "blueprint.architecture_spec" in context["approved_context"]["artifacts"]
    assert "Blueprint/architecture/architecture.md" not in context["approved_context"]["artifacts"]


def test_acp_package_context_uses_construction_questions_without_package_stage() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        record = _seed_session(db)
        db.add(
            ConstructionQuestionResponseRecord(
                session_id=record.id,
                question_key="objective_validation",
                gap_key="objective_validation",
                gap_title="Confirmar objetivo",
                domain="governance",
                question_text="Confirma el objetivo de construccion.",
                blocking=True,
                status="answered",
                answer_text="Confirmado.",
                impacted_artifacts=["acp.prompt_pack"],
            )
        )
        db.commit()

        context, refs = build_approved_deliverable_context(
            db,
            record=record,
            deliverable_key="acp.prompt_pack",
        )

    assert refs
    assert refs[0].startswith("acp-question:")
    assert context["context_policy"]["retrieval_strategy"] == (
        "acp_package_context_from_product_artifacts_and_readiness_questions"
    )
    questions = context["approved_context"]["construction_questions"]
    assert questions[0]["question_key"] == "objective_validation"
    assert questions[0]["answer_text"] == "Confirmado."
