from __future__ import annotations

from uuid import uuid4

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from app.models import (
    CommercialTier,
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
from app.services.product_processing import (
    ACP_REQUIRED_STAGE_KEYS,
    ProductBuildLifecycle,
    ProductBuildProductKey,
    ProductProcessingMode,
    UncertaintyBacklogStatus,
    UncertaintyDisposition,
    ensure_acp_product_orchestration,
    list_product_build_runs,
    list_product_build_steps,
)
from app.services.product_processing.persistence import UncertaintyBacklogRecord


def _engine():
    return create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)


def _seed_acp_session(db: Session) -> tuple[UserRecord, SessionRecord]:
    user = UserRecord(
        email=f"eov8-{uuid4()}@leanbuilder.local",
        full_name="EOV8 Tester",
        password_hash=hash_password("Secret123!"),
    )
    db.add(user)
    db.flush()
    workspace = WorkspaceRecord(name="EOV8 Workspace", slug=f"eov8-{str(user.id)[:8]}", created_by_user_id=user.id)
    db.add(workspace)
    db.flush()
    db.add(WorkspaceMembershipRecord(workspace_id=workspace.id, user_id=user.id, role=WorkspaceRole.owner))
    record = SessionRecord(
        user_id=user.id,
        workspace_id=workspace.id,
        title="EOV8 ACP direct project",
        current_stage=SessionStage.ready_for_export,
        commercial_tier=CommercialTier.acp,
    )
    db.add(record)
    db.commit()
    db.refresh(user)
    db.refresh(record)
    return user, record


def _approve_stage(db: Session, record: SessionRecord, stage_key: str) -> None:
    db.add(
        JourneyStageArtifactRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            artifact_kind=f"{stage_key}_artifact",
            stage_key=stage_key,
            state=JourneyArtifactState.approved,
            source_action="eov8_test",
        )
    )


def test_acp_direct_run_tracks_missing_pro_dependencies() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_acp_session(db)
        _approve_stage(db, record, "discover")
        _approve_stage(db, record, "define")
        db.commit()

        status = ensure_acp_product_orchestration(db, record=record, current_user=user)
        db.commit()
        run = list_product_build_runs(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            product_key=ProductBuildProductKey.acp,
        )[0]
        steps = list_product_build_steps(db, run_id=run.id)
        dependency_steps = [step for step in steps if step.step_key.startswith("acp_dependency:")]

    assert status.lifecycle == ProductBuildLifecycle.requires_attention
    assert len(dependency_steps) == len(ACP_REQUIRED_STAGE_KEYS)
    assert next(step for step in dependency_steps if step.step_key == "acp_dependency:discover").status == "completed"
    assert next(step for step in dependency_steps if step.step_key == "acp_dependency:design").status == "requires_attention"
    assert run.checkpoint_payload["acp_direct_resolution"]["missing_stage_keys"] == [
        "design",
        "tools",
        "memory",
        "estimate",
        "validate",
    ]


def test_acp_direct_run_blocks_on_acp_questions_even_with_approved_stages() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_acp_session(db)
        for stage in ACP_REQUIRED_STAGE_KEYS:
            _approve_stage(db, record, stage)
        db.add(
            UncertaintyBacklogRecord(
                workspace_id=record.workspace_id,
                session_id=record.id,
                uncertainty_key="acp-runtime-secret-owner",
                product_mode=ProductProcessingMode.acp_implementation.value,
                source_stage="package",
                target_stage="package",
                disposition=UncertaintyDisposition.block.value,
                status=UncertaintyBacklogStatus.open.value,
                title="Confirmar owner de secretos de integracion",
                description="La implementacion requiere definir responsable de credenciales.",
            )
        )
        db.commit()

        status = ensure_acp_product_orchestration(db, record=record, current_user=user)
        run = list_product_build_runs(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            product_key=ProductBuildProductKey.acp,
        )[0]
        package_step = next(step for step in list_product_build_steps(db, run_id=run.id) if step.step_key == "acp_dependency:validate")

    assert status.lifecycle == ProductBuildLifecycle.requires_attention
    assert package_step.status == "requires_attention"
    assert "blocking_questions:validate:1" in package_step.error_payload["reasons"]


def test_acp_direct_dependencies_complete_when_readiness_is_closed() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_acp_session(db)
        for stage in ACP_REQUIRED_STAGE_KEYS:
            _approve_stage(db, record, stage)
        db.commit()

        status = ensure_acp_product_orchestration(db, record=record, current_user=user)
        run = list_product_build_runs(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            product_key=ProductBuildProductKey.acp,
        )[0]
        dependency_steps = [
            step for step in list_product_build_steps(db, run_id=run.id) if step.step_key.startswith("acp_dependency:")
        ]

    assert status.lifecycle != ProductBuildLifecycle.requires_attention
    assert {step.status for step in dependency_steps} == {"completed"}
    assert run.checkpoint_payload["acp_direct_resolution"]["can_start_package"] is True
    assert run.checkpoint_payload["acp_direct_resolution"]["can_export_package"] is True
