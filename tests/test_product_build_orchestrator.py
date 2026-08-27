from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.models import (
    CommercialTier,
    SessionRecord,
    SessionStage,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceRecord,
    WorkspaceRole,
    utc_now,
)
from app.services.auth_service import hash_password
from app.services.commerce_service import tier_rank
from app.services.deliverable_catalog.catalog_service import build_deliverable_catalog_response
from app.services.deliverable_catalog.contracts import DeliverableGenerationResult, DeliverableGenerationTask
from app.services.deliverable_catalog.persistence import DeliverableGenerationJobRecord
from app.services.diagram_center.persistence import DiagramGenerationJobRecord
from app.services.product_processing import (
    ProductBuildLifecycle,
    ProductBuildOrchestrationOptions,
    ProductBuildProductKey,
    build_product_build_status,
    enqueue_product_build_processing,
    ensure_product_build_orchestration,
    run_product_build_processing,
)
from app.services.product_processing.persistence import ProductBuildRunRecord, ProductBuildStepRecord
from app.services.product_processing.product_build_run_service import list_product_build_runs, list_product_build_steps


def _engine():
    return create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)


def _seed_session(db: Session, *, tier: CommercialTier = CommercialTier.blueprint) -> tuple[UserRecord, SessionRecord]:
    user = UserRecord(
        email=f"eov4-{uuid4()}@leanbuilder.local",
        full_name="EOV4 Tester",
        password_hash=hash_password("Secret123!"),
    )
    db.add(user)
    db.flush()
    workspace = WorkspaceRecord(name="EOV4 Workspace", slug=f"eov4-{str(user.id)[:8]}", created_by_user_id=user.id)
    db.add(workspace)
    db.flush()
    db.add(WorkspaceMembershipRecord(workspace_id=workspace.id, user_id=user.id, role=WorkspaceRole.owner))
    record = SessionRecord(
        user_id=user.id,
        workspace_id=workspace.id,
        title="EOV4 Project",
        current_stage=SessionStage.ready_for_export,
        commercial_tier=tier,
    )
    db.add(record)
    db.commit()
    db.refresh(user)
    db.refresh(record)
    return user, record


def test_orchestrator_creates_run_and_expected_deliverable_steps() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_session(db)
        catalog = build_deliverable_catalog_response(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            role=WorkspaceRole.owner,
            tier=CommercialTier.blueprint,
            current_stage="package",
        )
        expected_count = len(
            [
                item
                for item in catalog.entries
                if "blueprint" in item.product_scope and tier_rank(item.required_tier) <= tier_rank(CommercialTier.blueprint)
            ]
        )

        status = ensure_product_build_orchestration(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_basic,
            current_user=user,
        )
        db.commit()
        runs = list_product_build_runs(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            product_key=ProductBuildProductKey.blueprint_basic,
        )
        steps = list_product_build_steps(db, run_id=runs[0].id)

    assert len(runs) == 1
    assert len(steps) == expected_count
    assert status.progress.total_units == float(expected_count)
    assert status.lifecycle == ProductBuildLifecycle.ready_to_start


def test_orchestrator_reuses_active_run_for_same_product() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_session(db)
        first = ensure_product_build_orchestration(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_basic,
            current_user=user,
            options=ProductBuildOrchestrationOptions(idempotency_key="eov4-blueprint-basic"),
        )
        second = ensure_product_build_orchestration(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_basic,
            current_user=user,
            options=ProductBuildOrchestrationOptions(idempotency_key="eov4-blueprint-basic"),
        )
        runs = list_product_build_runs(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            product_key=ProductBuildProductKey.blueprint_basic,
        )
        steps = db.exec(select(ProductBuildStepRecord).where(ProductBuildStepRecord.run_id == runs[0].id)).all()

    assert len(runs) == 1
    assert first.progress.total_units == second.progress.total_units
    assert len({step.step_key for step in steps}) == len(steps)


def test_orchestrator_reflects_existing_deliverable_job_state() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_session(db)
        catalog = build_deliverable_catalog_response(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            role=WorkspaceRole.owner,
            tier=CommercialTier.blueprint,
            current_stage="package",
        )
        deliverable_key = next(item.key for item in catalog.entries if "blueprint" in item.product_scope)
        db.add(
            DeliverableGenerationJobRecord(
                workspace_id=record.workspace_id,
                session_id=record.id,
                deliverable_key=deliverable_key,
                status="available",
                product_mode="basic_free",
                idempotency_key=f"eov4-job-{uuid4()}",
            )
        )
        db.commit()

        status = ensure_product_build_orchestration(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_basic,
            current_user=user,
        )
        run = list_product_build_runs(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            product_key=ProductBuildProductKey.blueprint_basic,
        )[0]
        step = next(step for step in list_product_build_steps(db, run_id=run.id) if step.deliverable_key == deliverable_key)

    assert step.status == "available"
    assert step.progress_percent == 100
    assert any(item.deliverable_key == deliverable_key and item.state.value == "available" for item in status.deliverables)


def test_orchestrator_can_execute_jobs_with_injected_runner() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)
    generated_keys: list[str] = []

    def fake_runner(db: Session, task: DeliverableGenerationTask):
        generated_keys.append(task.deliverable_key)
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

    with Session(engine) as db:
        user, record = _seed_session(db)
        status = ensure_product_build_orchestration(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_basic,
            current_user=user,
            options=ProductBuildOrchestrationOptions(execute_jobs=True, job_runner=fake_runner),
        )
        runs = db.exec(select(ProductBuildRunRecord)).all()

    assert generated_keys
    assert len(runs) == 1
    assert status.lifecycle == ProductBuildLifecycle.completed


def test_orchestrator_supports_blueprint_pro_with_enrichment_and_reconciliation() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_session(db, tier=CommercialTier.blueprint_pro)
        status = ensure_product_build_orchestration(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_pro,
            current_user=user,
        )
        db.commit()
        runs = list_product_build_runs(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            product_key=ProductBuildProductKey.blueprint_pro,
        )

    assert len(runs) == 1
    assert runs[0].product_key == ProductBuildProductKey.blueprint_pro
    assert status.product_key == ProductBuildProductKey.blueprint_pro
    assert status.lifecycle in {ProductBuildLifecycle.ready_to_start, ProductBuildLifecycle.preparing}


def test_activate_product_builds_for_paid_order_queues_blueprint_pro_run() -> None:
    from app.models import CommercialOrderRecord, CommercialOrderLineRecord, CommercialOrderStatus
    from app.services.product_processing.product_build_activation_service import activate_product_builds_for_paid_order

    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_session(db, tier=CommercialTier.blueprint_pro)
        order = CommercialOrderRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            buyer_user_id=user.id,
            status=CommercialOrderStatus.paid,
            currency="USD",
            subtotal_cents=9900,
            total_cents=9900,
            provider="sandbox",
            checkout_ref=f"sandbox_{uuid4().hex}",
            idempotency_key=f"idemp_{uuid4().hex}",
        )
        db.add(order)
        db.flush()
        line = CommercialOrderLineRecord(
            order_id=order.id,
            product_key="blueprint_pro",
            price_code="price_pro",
            quantity=1,
            unit_amount_cents=9900,
            total_amount_cents=9900,
        )
        db.add(line)
        db.commit()

        statuses = activate_product_builds_for_paid_order(
            db,
            order=order,
            current_user=user,
        )
        db.commit()

    assert len(statuses) == 1
    assert statuses[0].product_key == ProductBuildProductKey.blueprint_pro


def test_enqueue_product_build_processing_persists_queue_selection() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_session(db, tier=CommercialTier.blueprint_pro)
        catalog = build_deliverable_catalog_response(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            role=WorkspaceRole.owner,
            tier=CommercialTier.blueprint_pro,
            current_stage="package",
        )
        failed_item = next(item for item in catalog.entries if item.access.can_generate or item.access.can_regenerate)
        db.add(
            DeliverableGenerationJobRecord(
                workspace_id=record.workspace_id,
                session_id=record.id,
                deliverable_key=failed_item.key,
                status="failed",
                product_mode="premium_enrichment",
                idempotency_key=f"queue-failed-{uuid4()}",
            )
        )
        db.commit()

        run, status, queued_now = enqueue_product_build_processing(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_pro,
            current_user=user,
            mode="process_pending",
            allow_llm=False,
            catalog_stage_override="package",
        )
        db.commit()
        run_id = run.id if run is not None else None

    assert queued_now is True
    assert run is not None
    assert status.processing_queue is not None
    assert status.processing_queue.active is True
    assert status.processing_queue.total_count > 0
    with Session(engine) as db:
        persisted_run = db.get(ProductBuildRunRecord, run_id)
        assert persisted_run is not None
        assert failed_item.key in (persisted_run.checkpoint_payload or {}).get("processing_queue", {}).get("selected_deliverable_keys", [])
        selected_steps = [
            step
            for step in list_product_build_steps(db, run_id=run_id)
            if step.checkpoint_payload.get("queue_selected")
        ]
    assert selected_steps
    assert all(step.status == "queued" for step in selected_steps)


def test_run_product_build_processing_retries_failed_items_once(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    def fake_runner(db: Session, task: DeliverableGenerationTask):
        job = DeliverableGenerationJobRecord(
            workspace_id=task.workspace_id,
            session_id=task.session_id,
            deliverable_key=task.deliverable_key,
            status="error",
            product_mode=task.product_mode,
            idempotency_key=task.idempotency_key,
            error_code="forced_failure",
            error_message=f"Forced failure for {task.deliverable_key}",
        )
        db.add(job)
        db.flush()
        return job, DeliverableGenerationResult(
            deliverable_key=task.deliverable_key,
            status="failed",
            error_code="forced_failure",
            error_message=f"Forced failure for {task.deliverable_key}",
        )

    def fake_create_diagram_job(
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

    def fake_run_diagram_job(job_id, database_engine=None):
        with Session(database_engine or engine) as db:
            job = db.get(DiagramGenerationJobRecord, job_id)
            assert job is not None
            job.status = "error"
            job.error_code = "forced_diagram_failure"
            job.error_message = f"Forced failure for {job.diagram_key}"
            job.completed_at = utc_now()
            job.updated_at = utc_now()
            db.add(job)
            db.commit()

    monkeypatch.setattr("app.services.product_processing.product_build_orchestrator.run_deliverable_generation_task", fake_runner)
    monkeypatch.setattr("app.services.product_processing.product_build_orchestrator.create_generation_job", fake_create_diagram_job)
    monkeypatch.setattr("app.services.product_processing.product_build_orchestrator.run_generation_job", fake_run_diagram_job)

    with Session(engine) as db:
        user, record = _seed_session(db, tier=CommercialTier.blueprint_pro)
        run, _, queued_now = enqueue_product_build_processing(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_pro,
            current_user=user,
            mode="process_pending",
            allow_llm=False,
            catalog_stage_override="package",
        )
        assert run is not None
        assert queued_now is True

        run_product_build_processing(run.id, engine)
        db.expire_all()
        status = build_product_build_status(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_pro,
            current_user=user,
            catalog_stage_override="package",
        )
        steps = [
            step
            for step in list_product_build_steps(db, run_id=run.id)
            if step.checkpoint_payload.get("queue_selected")
        ]

    assert status.processing_queue is not None
    assert status.processing_queue.active is False
    assert status.processing_queue.failed_count == status.processing_queue.total_count
    assert status.processing_queue.retried_count == status.processing_queue.total_count
    assert status.processing_queue.status == "completed_with_errors"
    assert status.lifecycle == ProductBuildLifecycle.requires_attention
    assert steps
    assert all(step.checkpoint_payload.get("attempt_count") == 2 for step in steps)
