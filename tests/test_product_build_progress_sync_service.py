from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi import BackgroundTasks
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.api.routes.productization import post_product_build_action_route
from app.models import (
    CommercialTier,
    JourneyStateRecord,
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
from app.services.deliverable_catalog.contracts import DeliverableGenerationResult, DeliverableGenerationTask
from app.services.deliverable_catalog.persistence import DeliverableGenerationJobRecord
from app.services.diagram_center.persistence import DiagramGenerationJobRecord
from app.services.product_processing import (
    ACP_REQUIRED_STAGE_KEYS,
    ProductBuildCommandRequest,
    ProductBuildProductKey,
    build_product_build_status,
    sync_product_builds_after_stage_approval,
)
from app.services.product_processing.contracts import JourneyStateKey


def _engine():
    return create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)


def _seed_session(
    db: Session,
    *,
    tier: CommercialTier,
    stage: SessionStage = SessionStage.ready_for_export,
) -> tuple[UserRecord, SessionRecord]:
    user = UserRecord(
        email=f"eov9-{uuid4()}@leanbuilder.local",
        full_name="EOV9 Tester",
        password_hash=hash_password("Secret123!"),
    )
    db.add(user)
    db.flush()
    workspace = WorkspaceRecord(name="EOV9 Workspace", slug=f"eov9-{str(user.id)[:8]}", created_by_user_id=user.id)
    db.add(workspace)
    db.flush()
    db.add(WorkspaceMembershipRecord(workspace_id=workspace.id, user_id=user.id, role=WorkspaceRole.owner))
    record = SessionRecord(
        user_id=user.id,
        workspace_id=workspace.id,
        title="EOV9 Project",
        current_stage=stage,
        commercial_tier=tier,
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
            source_action="eov9_test",
        )
    )


def _fake_runner_factory(generated_tasks: list[tuple[str, str]]):
    def fake_runner(db: Session, task: DeliverableGenerationTask):
        generated_tasks.append((task.product_mode, task.deliverable_key))
        job = DeliverableGenerationJobRecord(
            workspace_id=task.workspace_id,
            session_id=task.session_id,
            deliverable_key=task.deliverable_key,
            status="available",
            product_mode=task.product_mode,
            idempotency_key=task.idempotency_key,
        )
        db.add(job)
        db.flush()
        return job, DeliverableGenerationResult(deliverable_key=task.deliverable_key, status="available")

    return fake_runner


def _fake_create_diagram_job(
    db: Session,
    *,
    record: SessionRecord,
    diagram_key: str,
    user_id,
    detail_level: str,
    reason: str,
    idempotency_key: str,
):
    job = DiagramGenerationJobRecord(
        workspace_id=record.workspace_id,
        session_id=record.id,
        diagram_key=diagram_key,
        requested_by_user_id=user_id,
        detail_level=detail_level,
        reason=reason,
        idempotency_key=idempotency_key,
        status="queued",
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def _fake_run_diagram_job(job_id, database_engine=None, *, db_session=None):
    if db_session is not None:
        db = db_session
    else:
        db = Session(database_engine)
    try:
        job = db.get(DiagramGenerationJobRecord, job_id)
        assert job is not None
        job.status = "available"
        job.completed_at = job.completed_at or job.updated_at
        db.add(job)
        db.commit()
    finally:
        if db_session is None:
            db.close()


def test_stage_approval_sync_auto_executes_blueprint_pro_build(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)
    generated_tasks: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "app.services.product_processing.product_build_orchestrator.run_deliverable_generation_task",
        _fake_runner_factory(generated_tasks),
    )
    monkeypatch.setattr("app.services.product_processing.product_build_orchestrator.create_generation_job", _fake_create_diagram_job)
    monkeypatch.setattr("app.services.product_processing.product_build_orchestrator.run_generation_job", _fake_run_diagram_job)

    with Session(engine) as db:
        user, record = _seed_session(db, tier=CommercialTier.blueprint_pro)
        statuses = sync_product_builds_after_stage_approval(
            db,
            record=record,
            stage_key="package",
            current_user=user,
        )

    assert any(status.product_key == ProductBuildProductKey.blueprint_pro for status in statuses)
    assert any(product_mode == "premium_enrichment" for product_mode, _ in generated_tasks)


def test_stage_approval_persists_the_next_actionable_journey_state() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_session(db, tier=CommercialTier.blueprint)
        sync_product_builds_after_stage_approval(
            db,
            record=record,
            stage_key="tools",
            current_user=user,
        )
        db.commit()
        current = db.exec(
            select(JourneyStateRecord).where(JourneyStateRecord.session_id == record.id)
        ).one()

    assert current.state_key == JourneyStateKey.memory.value
    assert current.stage_key == "memory"


def test_stage_approval_sync_auto_executes_acp_when_package_is_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)
    generated_tasks: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "app.services.product_processing.product_build_orchestrator.run_deliverable_generation_task",
        _fake_runner_factory(generated_tasks),
    )
    monkeypatch.setattr("app.services.product_processing.product_build_orchestrator.create_generation_job", _fake_create_diagram_job)
    monkeypatch.setattr("app.services.product_processing.product_build_orchestrator.run_generation_job", _fake_run_diagram_job)

    with Session(engine) as db:
        user, record = _seed_session(db, tier=CommercialTier.acp)
        for stage_key in ACP_REQUIRED_STAGE_KEYS:
            _approve_stage(db, record, stage_key)
        db.commit()

        statuses = sync_product_builds_after_stage_approval(
            db,
            record=record,
            stage_key="package",
            current_user=user,
        )

    assert any(status.product_key == ProductBuildProductKey.acp for status in statuses)
    assert any(product_mode == "acp_implementation" for product_mode, _ in generated_tasks)


def test_post_product_build_action_route_uses_acp_orchestration(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)
    captured: dict[str, object] = {}

    def fake_ensure_acp_product_orchestration(
        db: Session,
        *,
        record: SessionRecord,
        snapshot=None,
        current_user: UserRecord | None = None,
        execute_jobs: bool = False,
        allow_llm: bool = False,
        activation_payload=None,
        catalog_stage_override: str | None = "package",
    ):
        captured["execute_jobs"] = execute_jobs
        captured["allow_llm"] = allow_llm
        captured["activation_payload"] = dict(activation_payload or {})
        return build_product_build_status(
            db,
            record=record,
            product_key=ProductBuildProductKey.acp,
            current_user=current_user,
            catalog_stage_override=catalog_stage_override,
        )

    monkeypatch.setattr(
        "app.services.product_processing.acp_product_orchestration_service.ensure_acp_product_orchestration",
        fake_ensure_acp_product_orchestration,
    )

    with Session(engine) as db:
        user, record = _seed_session(db, tier=CommercialTier.acp)
        response = post_product_build_action_route(
            record.id,
            ProductBuildProductKey.acp,
            ProductBuildCommandRequest(action="start", allow_llm=True),
            BackgroundTasks(),
            db,
            user,
        )

    assert response.product_key == ProductBuildProductKey.acp
    assert captured["execute_jobs"] is True
    assert captured["allow_llm"] is True
    assert captured["activation_payload"] == {
        "source": "product_build_action:start",
        "idempotency_key": "",
    }
