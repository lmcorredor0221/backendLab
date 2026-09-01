from __future__ import annotations

from datetime import timedelta
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
from app.services.product_processing.product_build_orchestrator import _finalize_processing_queue
from app.services.product_processing.product_build_orchestrator import _process_single_queue_item
from app.services.product_processing.product_build_orchestrator import _recover_orphaned_processing_queue
from app.services.product_processing.product_build_run_service import (
    list_product_build_runs,
    list_product_build_steps,
    upsert_product_build_step,
)


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
        ensure_product_build_orchestration(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_pro,
            current_user=user,
            options=ProductBuildOrchestrationOptions(current_stage="package"),
            catalog_stage_override="package",
        )
        catalog = build_deliverable_catalog_response(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            role=WorkspaceRole.owner,
            tier=CommercialTier.blueprint_pro,
            current_stage="package",
        )
        queued_item = next(item for item in catalog.entries if item.access.can_generate or item.access.can_regenerate)

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
        assert queued_item.key in (persisted_run.checkpoint_payload or {}).get("processing_queue", {}).get("selected_deliverable_keys", [])
        selected_steps = [
            step
            for step in list_product_build_steps(db, run_id=run_id)
            if step.checkpoint_payload.get("queue_selected")
        ]
    assert selected_steps
    assert all(step.status == "queued" for step in selected_steps)


def test_enqueue_product_build_processing_skips_exhausted_failed_deliverables() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_session(db, tier=CommercialTier.blueprint_pro)
        ensure_product_build_orchestration(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_pro,
            current_user=user,
            options=ProductBuildOrchestrationOptions(current_stage="package"),
            catalog_stage_override="package",
        )
        catalog = build_deliverable_catalog_response(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            role=WorkspaceRole.owner,
            tier=CommercialTier.blueprint_pro,
            current_stage="package",
        )
        failed_item = next(
            item
            for item in catalog.entries
            if item.deliverable_type.value != "diagram" and (item.access.can_generate or item.access.can_regenerate)
        )
        run = list_product_build_runs(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            product_key=ProductBuildProductKey.blueprint_pro,
        )[0]
        step = next(
            candidate for candidate in list_product_build_steps(db, run_id=run.id) if candidate.deliverable_key == failed_item.key
        )
        step.status = "error"
        step.error_payload = {"code": "forced_failure", "message": "El entregable ya agotó los intentos permitidos."}
        step.checkpoint_payload = {
            **(step.checkpoint_payload or {}),
            "queue_selected": True,
            "attempt_count": 2,
            "retried": True,
            "last_failed_at": utc_now().isoformat(),
            "job_source": "deliverable_catalog",
        }
        db.add(step)
        db.add(
            DeliverableGenerationJobRecord(
                workspace_id=record.workspace_id,
                session_id=record.id,
                deliverable_key=failed_item.key,
                status="error",
                product_mode="premium_enrichment",
                idempotency_key=f"exhausted-deliverable-{uuid4()}",
                error_code="forced_failure",
                error_message="El entregable ya agotó los intentos permitidos.",
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

    assert run is not None
    assert status.processing_queue is not None
    assert failed_item.key not in status.processing_queue.current_deliverable_key
    assert failed_item.key not in [item.deliverable_key for item in status.processing_queue.completed_items]
    assert failed_item.key not in [item.deliverable_key for item in status.processing_queue.failed_items]
    with Session(engine) as db:
        persisted_run = db.get(ProductBuildRunRecord, run_id)
        assert persisted_run is not None
        selected_keys = (persisted_run.checkpoint_payload or {}).get("processing_queue", {}).get("selected_deliverable_keys", [])
    assert failed_item.key not in selected_keys
    assert queued_now is (len(selected_keys) > 0)


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

    def fake_run_diagram_job(job_id, database_engine=None, *, db_session=None):
        if db_session is not None:
            db = db_session
            job = db.get(DiagramGenerationJobRecord, job_id)
            assert job is not None
            job.status = "error"
            job.error_code = "forced_diagram_failure"
            job.error_message = f"Forced failure for {job.diagram_key}"
            job.completed_at = utc_now()
            job.updated_at = utc_now()
            db.add(job)
            db.commit()
            return
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
    assert status.processing_queue.retried_count == sum(
        1 for step in steps if int((step.checkpoint_payload or {}).get("attempt_count") or 0) > 1
    )
    assert status.processing_queue.status == "completed_with_errors"
    assert status.lifecycle == ProductBuildLifecycle.requires_attention
    assert steps
    assert all(
        int((step.checkpoint_payload or {}).get("attempt_count") or 0) == 0
        for step in steps
        if str(step.deliverable_key or "").startswith("diagram.")
    )
    assert all(
        int((step.checkpoint_payload or {}).get("attempt_count") or 0) == 2
        for step in steps
        if not str(step.deliverable_key or "").startswith("diagram.")
    )


def test_process_single_queue_item_sends_approved_context_to_deliverable_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)
    captured_task: DeliverableGenerationTask | None = None

    def fake_context_builder(db: Session, *, record: SessionRecord, deliverable_key: str):
        return {"approved": True, "deliverable_key": deliverable_key}, ["journey:approved:v1"]

    def fake_runner(db: Session, task: DeliverableGenerationTask):
        nonlocal captured_task
        captured_task = task
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

    monkeypatch.setattr(
        "app.services.product_processing.product_build_orchestrator.build_approved_deliverable_context",
        fake_context_builder,
    )
    monkeypatch.setattr(
        "app.services.product_processing.product_build_orchestrator.run_deliverable_generation_task",
        fake_runner,
    )
    monkeypatch.setattr(
        "app.services.product_processing.product_build_orchestrator._dependency_error_for_item",
        lambda *args, **kwargs: None,
    )

    with Session(engine) as db:
        user, record = _seed_session(db, tier=CommercialTier.blueprint_pro)
        ensure_product_build_orchestration(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_pro,
            current_user=user,
            options=ProductBuildOrchestrationOptions(current_stage="package"),
            catalog_stage_override="package",
        )
        catalog = build_deliverable_catalog_response(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            role=WorkspaceRole.owner,
            tier=CommercialTier.blueprint_pro,
            current_stage="package",
        )
        item = next(
            entry
            for entry in catalog.entries
            if entry.deliverable_type.value != "diagram" and (entry.access.can_generate or entry.access.can_regenerate)
        )
        items_by_key = {entry.key: entry for entry in catalog.entries}
        run = list_product_build_runs(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            product_key=ProductBuildProductKey.blueprint_pro,
        )[0]

        processed = _process_single_queue_item(
            db,
            run=run,
            record=record,
            item=item,
            items_by_key=items_by_key,
            allow_llm=False,
            phase="initial",
            position=1,
            total_count=1,
        )

    assert processed is True
    assert captured_task is not None
    assert captured_task.context_payload == {"approved": True, "deliverable_key": item.key}
    assert captured_task.approved_context_refs == ["journey:approved:v1"]


def test_execute_jobs_sends_built_approved_context_to_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)
    captured_tasks: list[DeliverableGenerationTask] = []

    def fake_context_builder(db: Session, *, record: SessionRecord, deliverable_key: str):
        return {"approved": True, "deliverable_key": deliverable_key}, ["journey:approved:v1"]

    def fake_runner(db: Session, task: DeliverableGenerationTask):
        captured_tasks.append(task)
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

    monkeypatch.setattr(
        "app.services.product_processing.product_build_orchestrator.build_approved_deliverable_context",
        fake_context_builder,
    )

    with Session(engine) as db:
        user, record = _seed_session(db, tier=CommercialTier.blueprint_pro)
        ensure_product_build_orchestration(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_pro,
            current_user=user,
            options=ProductBuildOrchestrationOptions(
                current_stage="package",
                execute_jobs=True,
                job_runner=fake_runner,
            ),
            catalog_stage_override="package",
        )

    assert captured_tasks
    assert all(task.context_payload == {"approved": True, "deliverable_key": task.deliverable_key} for task in captured_tasks)
    assert all(task.approved_context_refs == ["journey:approved:v1"] for task in captured_tasks)


def test_diagram_queue_item_does_not_start_provider_when_llm_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    monkeypatch.setattr(
        "app.services.product_processing.product_build_orchestrator._dependency_error_for_item",
        lambda *args, **kwargs: None,
    )

    def fail_create_generation_job(*args, **kwargs):
        pytest.fail("diagram generation job should not be created when allow_llm is false")

    def fail_run_generation_job(*args, **kwargs):
        pytest.fail("diagram provider should not run when allow_llm is false")

    monkeypatch.setattr(
        "app.services.product_processing.product_build_orchestrator.create_generation_job",
        fail_create_generation_job,
    )
    monkeypatch.setattr(
        "app.services.product_processing.product_build_orchestrator.run_generation_job",
        fail_run_generation_job,
    )

    with Session(engine) as db:
        user, record = _seed_session(db, tier=CommercialTier.blueprint_pro)
        ensure_product_build_orchestration(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_pro,
            current_user=user,
            options=ProductBuildOrchestrationOptions(current_stage="package"),
            catalog_stage_override="package",
        )
        catalog = build_deliverable_catalog_response(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            role=WorkspaceRole.owner,
            tier=CommercialTier.blueprint_pro,
            current_stage="package",
        )
        item = next(entry for entry in catalog.entries if entry.deliverable_type.value == "diagram")
        run = list_product_build_runs(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            product_key=ProductBuildProductKey.blueprint_pro,
        )[0]

        processed = _process_single_queue_item(
            db,
            run=run,
            record=record,
            item=item,
            items_by_key={entry.key: entry for entry in catalog.entries},
            allow_llm=False,
            phase="retry",
            position=1,
            total_count=1,
        )

        step = next(step for step in list_product_build_steps(db, run_id=run.id) if step.deliverable_key == item.key)
        diagram_jobs = db.exec(
            select(DiagramGenerationJobRecord).where(DiagramGenerationJobRecord.session_id == record.id)
        ).all()

    assert processed is False
    assert step.status == "error"
    assert step.error_payload["code"] == "llm_required_for_diagram_generation"
    assert int((step.checkpoint_payload or {}).get("attempt_count") or 0) == 0
    assert diagram_jobs == []


def test_retry_failed_selects_requires_attention_steps() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_session(db, tier=CommercialTier.blueprint_pro)
        ensure_product_build_orchestration(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_pro,
            current_user=user,
            options=ProductBuildOrchestrationOptions(current_stage="package"),
            catalog_stage_override="package",
        )
        catalog = build_deliverable_catalog_response(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            role=WorkspaceRole.owner,
            tier=CommercialTier.blueprint_pro,
            current_stage="package",
        )
        item = next(entry for entry in catalog.entries if entry.deliverable_type.value != "diagram")
        run = list_product_build_runs(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            product_key=ProductBuildProductKey.blueprint_pro,
        )[0]
        upsert_product_build_step(
            db,
            run=run,
            step_key=f"deliverable:{item.key}",
            status="requires_attention",
            stage_key=item.stage,
            deliverable_key=item.key,
            sequence=item.sort_order,
            progress_percent=0,
            checkpoint_payload={"attempt_count": 1},
            error_payload={"code": "context_missing"},
        )
        db.commit()

        _, status, queued_now = enqueue_product_build_processing(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_pro,
            current_user=user,
            mode="retry_failed",
            allow_llm=False,
            catalog_stage_override="package",
        )
        db.refresh(run)

    assert queued_now is True
    assert status.processing_queue is not None
    assert status.processing_queue.total_count >= 1
    assert item.key in (run.checkpoint_payload.get("processing_queue") or {}).get("selected_deliverable_keys", [])


def test_run_product_build_processing_marks_orphaned_diagram_jobs_as_error(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_session(db, tier=CommercialTier.blueprint_pro)
        ensure_product_build_orchestration(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_pro,
            current_user=user,
            options=ProductBuildOrchestrationOptions(current_stage="package"),
            catalog_stage_override="package",
        )
        catalog = build_deliverable_catalog_response(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            role=WorkspaceRole.owner,
            tier=CommercialTier.blueprint_pro,
            current_stage="package",
        )
        diagrams = [item for item in catalog.entries if item.deliverable_type.value == "diagram"]
        queued_diagram = diagrams[0]
        pending_diagram = diagrams[1]
        run = list_product_build_runs(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            product_key=ProductBuildProductKey.blueprint_pro,
        )[0]
        step = next(
            candidate for candidate in list_product_build_steps(db, run_id=run.id) if candidate.deliverable_key == queued_diagram.key
        )
        step.status = "error"
        step.error_payload = {"code": "TimeoutError", "message": "QueuePool limit reached before the diagram job could start."}
        step.checkpoint_payload = {
            **(step.checkpoint_payload or {}),
            "queue_selected": True,
            "attempt_count": 2,
            "retried": True,
            "last_failed_at": utc_now().isoformat(),
            "job_source": "diagram_center",
        }
        db.add(step)
        queued_job = DiagramGenerationJobRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            diagram_key=queued_diagram.key.removeprefix("diagram."),
            requested_by_user_id=user.id,
            detail_level="standard",
            reason="regenerate",
            idempotency_key=f"queued-diagram-{uuid4()}",
            status="queued",
        )
        db.add(queued_job)
        run.checkpoint_payload = {
            **(run.checkpoint_payload or {}),
            "processing_queue": {
                "queue_id": f"queue-{uuid4()}",
                "mode": "retry_failed",
                "status": "running",
                "selected_deliverable_keys": [queued_diagram.key],
                "retry_deliverable_keys": [queued_diagram.key],
                "allow_llm": False,
                "summary": "Procesando 0 de 1 entregables elegibles.",
            },
        }
        db.add(run)
        db.commit()

        _finalize_processing_queue(
            db,
            run=run,
            status="completed_with_errors",
            summary="Se completaron 0 de 1 entregables; 1 siguen fallando.",
            failed_keys=[queued_diagram.key],
        )
        db.expire_all()
        status = build_product_build_status(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_pro,
            current_user=user,
            catalog_stage_override="package",
        )
        latest_diagram_job = db.exec(
            select(DiagramGenerationJobRecord)
            .where(
                DiagramGenerationJobRecord.session_id == record.id,
                DiagramGenerationJobRecord.diagram_key == queued_diagram.key.removeprefix("diagram."),
            )
            .order_by(DiagramGenerationJobRecord.updated_at.desc())
        ).first()
        step = next(
            step for step in list_product_build_steps(db, run_id=run.id) if step.deliverable_key == queued_diagram.key
        )

    assert latest_diagram_job is not None
    assert latest_diagram_job.status == "error"
    assert latest_diagram_job.error_code == "processing_queue_orphaned"
    assert latest_diagram_job.started_at is None
    assert status.processing_queue is not None
    assert status.processing_queue.active is False
    assert status.processing_queue.failed_count == 1
    assert status.processing_queue.processing_count == 0
    assert step.status == "error"


def test_reconcile_marks_started_stale_diagram_job_as_orphaned() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_session(db, tier=CommercialTier.blueprint_pro)
        ensure_product_build_orchestration(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_pro,
            current_user=user,
            options=ProductBuildOrchestrationOptions(current_stage="package"),
            catalog_stage_override="package",
        )
        catalog = build_deliverable_catalog_response(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            role=WorkspaceRole.owner,
            tier=CommercialTier.blueprint_pro,
            current_stage="package",
        )
        diagrams = [item for item in catalog.entries if item.deliverable_type.value == "diagram"]
        queued_diagram = diagrams[0]
        pending_diagram = diagrams[1]
        run = list_product_build_runs(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            product_key=ProductBuildProductKey.blueprint_pro,
        )[0]
        stale_at = utc_now() - timedelta(minutes=10)
        step = next(
            candidate for candidate in list_product_build_steps(db, run_id=run.id) if candidate.deliverable_key == queued_diagram.key
        )
        step.status = "generating"
        step.updated_at = stale_at
        step.checkpoint_payload = {
            **(step.checkpoint_payload or {}),
            "queue_selected": True,
            "job_source": "diagram_center",
        }
        db.add(step)
        pending_step = next(
            candidate for candidate in list_product_build_steps(db, run_id=run.id) if candidate.deliverable_key == pending_diagram.key
        )
        pending_step.status = "queued"
        pending_step.updated_at = utc_now()
        pending_step.checkpoint_payload = {
            **(pending_step.checkpoint_payload or {}),
            "queue_selected": True,
            "job_source": "diagram_center",
        }
        db.add(pending_step)
        job = DiagramGenerationJobRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            diagram_key=queued_diagram.key.removeprefix("diagram."),
            requested_by_user_id=user.id,
            detail_level="standard",
            reason="regenerate",
            idempotency_key=f"stale-diagram-{uuid4()}",
            status="generating",
            started_at=stale_at,
            updated_at=stale_at,
        )
        db.add(job)
        run.checkpoint_payload = {
            **(run.checkpoint_payload or {}),
            "processing_queue": {
                "queue_id": f"queue-{uuid4()}",
                "mode": "process_pending",
                "status": "running",
                "selected_deliverable_keys": [queued_diagram.key, pending_diagram.key],
                "retry_deliverable_keys": [],
                "allow_llm": False,
                "summary": "Procesando 1 de 1 entregables elegibles.",
            },
        }
        db.add(run)
        db.commit()

        recovered = _recover_orphaned_processing_queue(db, run=run)
        db.expire_all()
        refreshed_run = db.get(ProductBuildRunRecord, run.id)
        refreshed_job = db.get(DiagramGenerationJobRecord, job.id)
        refreshed_step = db.get(ProductBuildStepRecord, step.id)

    assert recovered is True
    assert refreshed_run is not None
    assert refreshed_run.lifecycle == ProductBuildLifecycle.requires_attention
    assert refreshed_job is not None
    assert refreshed_job.status == "error"
    assert refreshed_job.error_code == "processing_queue_orphaned"
    assert refreshed_job.started_at is not None
    assert refreshed_job.completed_at is not None
    assert refreshed_step is not None
    assert refreshed_step.status == "error"


def test_ensure_product_build_orchestration_recovers_stale_processing_queue() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_session(db, tier=CommercialTier.blueprint_pro)
        ensure_product_build_orchestration(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_pro,
            current_user=user,
            options=ProductBuildOrchestrationOptions(current_stage="package"),
            catalog_stage_override="package",
        )
        catalog = build_deliverable_catalog_response(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            role=WorkspaceRole.owner,
            tier=CommercialTier.blueprint_pro,
            current_stage="package",
        )
        diagrams = [item for item in catalog.entries if item.deliverable_type.value == "diagram"]
        queued_diagram = diagrams[0]
        pending_diagram = diagrams[1]
        run = list_product_build_runs(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            product_key=ProductBuildProductKey.blueprint_pro,
        )[0]
        stale_at = utc_now() - timedelta(minutes=10)
        step = next(
            candidate for candidate in list_product_build_steps(db, run_id=run.id) if candidate.deliverable_key == queued_diagram.key
        )
        step.status = "generating"
        step.updated_at = stale_at
        step.checkpoint_payload = {
            **(step.checkpoint_payload or {}),
            "queue_selected": True,
            "job_source": "diagram_center",
        }
        db.add(step)
        pending_step = next(
            candidate for candidate in list_product_build_steps(db, run_id=run.id) if candidate.deliverable_key == pending_diagram.key
        )
        pending_step.status = "queued"
        pending_step.updated_at = utc_now()
        pending_step.checkpoint_payload = {
            **(pending_step.checkpoint_payload or {}),
            "queue_selected": True,
            "job_source": "diagram_center",
        }
        db.add(pending_step)
        job = DiagramGenerationJobRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            diagram_key=queued_diagram.key.removeprefix("diagram."),
            requested_by_user_id=user.id,
            detail_level="standard",
            reason="regenerate",
            idempotency_key=f"stale-ensure-diagram-{uuid4()}",
            status="generating",
            started_at=stale_at,
            updated_at=stale_at,
        )
        db.add(job)
        run.checkpoint_payload = {
            **(run.checkpoint_payload or {}),
            "processing_queue": {
                "queue_id": f"queue-{uuid4()}",
                "mode": "process_pending",
                "status": "running",
                "selected_deliverable_keys": [queued_diagram.key, pending_diagram.key],
                "retry_deliverable_keys": [],
                "allow_llm": False,
                "summary": "Procesando 1 de 1 entregables elegibles.",
            },
        }
        db.add(run)
        db.commit()

        status = ensure_product_build_orchestration(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_pro,
            current_user=user,
            options=ProductBuildOrchestrationOptions(current_stage="package"),
            catalog_stage_override="package",
        )
        db.expire_all()
        refreshed_job = db.get(DiagramGenerationJobRecord, job.id)

    assert status.lifecycle == ProductBuildLifecycle.requires_attention
    assert status.processing_queue is not None
    assert status.processing_queue.active is False
    assert refreshed_job is not None
    assert refreshed_job.status == "error"
    assert refreshed_job.error_code == "processing_queue_orphaned"


def test_recover_orphaned_queue_uses_active_job_timestamp_even_when_step_was_refreshed() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_session(db, tier=CommercialTier.blueprint_pro)
        ensure_product_build_orchestration(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_pro,
            current_user=user,
            options=ProductBuildOrchestrationOptions(current_stage="package"),
            catalog_stage_override="package",
        )
        catalog = build_deliverable_catalog_response(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            role=WorkspaceRole.owner,
            tier=CommercialTier.blueprint_pro,
            current_stage="package",
        )
        queued_diagram = next(item for item in catalog.entries if item.deliverable_type.value == "diagram")
        run = list_product_build_runs(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            product_key=ProductBuildProductKey.blueprint_pro,
        )[0]
        stale_at = utc_now() - timedelta(minutes=10)
        step = next(
            candidate for candidate in list_product_build_steps(db, run_id=run.id) if candidate.deliverable_key == queued_diagram.key
        )
        step.status = "generating"
        step.updated_at = utc_now()
        step.checkpoint_payload = {
            **(step.checkpoint_payload or {}),
            "queue_selected": True,
            "job_source": "diagram_center",
        }
        db.add(step)
        job = DiagramGenerationJobRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            diagram_key=queued_diagram.key.removeprefix("diagram."),
            requested_by_user_id=user.id,
            detail_level="standard",
            reason="regenerate",
            idempotency_key=f"stale-job-fresh-step-{uuid4()}",
            status="generating",
            started_at=stale_at,
            updated_at=stale_at,
        )
        db.add(job)
        run.checkpoint_payload = {
            **(run.checkpoint_payload or {}),
            "processing_queue": {
                "queue_id": f"queue-{uuid4()}",
                "mode": "process_pending",
                "status": "running",
                "selected_deliverable_keys": [queued_diagram.key],
                "retry_deliverable_keys": [],
                "allow_llm": False,
                "summary": "Procesando 1 de 1 entregables elegibles.",
            },
        }
        db.add(run)
        db.commit()

        recovered = _recover_orphaned_processing_queue(db, run=run)
        db.expire_all()
        refreshed_job = db.get(DiagramGenerationJobRecord, job.id)
        refreshed_step = db.get(ProductBuildStepRecord, step.id)

    assert recovered is True
    assert refreshed_job is not None
    assert refreshed_job.status == "error"
    assert refreshed_job.error_code == "processing_queue_orphaned"
    assert refreshed_step is not None
    assert refreshed_step.status == "error"


def test_ensure_product_build_orchestration_execute_jobs_skips_exhausted_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    def fail_deliverable_runner(db: Session, task: DeliverableGenerationTask):
        pytest.fail(f"deliverable execution should not restart for exhausted item {task.deliverable_key}")

    def fail_create_diagram_job(
        db: Session,
        *,
        record: SessionRecord,
        diagram_key: str,
        user_id,
        detail_level: str,
        reason: str,
        idempotency_key: str,
    ):
        pytest.fail(f"diagram execution should not restart for exhausted item {diagram_key}")

    def fail_run_diagram_job(job_id, database_engine=None, *, db_session=None):
        pytest.fail(f"diagram worker should not restart for exhausted job {job_id}")

    monkeypatch.setattr("app.services.product_processing.product_build_orchestrator.run_deliverable_generation_task", fail_deliverable_runner)
    monkeypatch.setattr("app.services.product_processing.product_build_orchestrator.create_generation_job", fail_create_diagram_job)
    monkeypatch.setattr("app.services.product_processing.product_build_orchestrator.run_generation_job", fail_run_diagram_job)

    with Session(engine) as db:
        user, record = _seed_session(db, tier=CommercialTier.blueprint_pro)
        ensure_product_build_orchestration(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_pro,
            current_user=user,
            options=ProductBuildOrchestrationOptions(current_stage="package"),
            catalog_stage_override="package",
        )
        catalog = build_deliverable_catalog_response(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            role=WorkspaceRole.owner,
            tier=CommercialTier.blueprint_pro,
            current_stage="package",
        )
        by_key = {item.key: item for item in catalog.entries}
        run = list_product_build_runs(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            product_key=ProductBuildProductKey.blueprint_pro,
        )[0]

        for step in list_product_build_steps(db, run_id=run.id):
            item = by_key.get(step.deliverable_key or "")
            if item is None:
                continue
            step.status = "error"
            step.error_payload = {"code": "forced_failure", "message": f"{item.key} agotó sus intentos."}
            step.checkpoint_payload = {
                **(step.checkpoint_payload or {}),
                "queue_selected": True,
                "attempt_count": 2,
                "retried": True,
                "last_failed_at": utc_now().isoformat(),
                "job_source": "diagram_center" if item.deliverable_type.value == "diagram" else "deliverable_catalog",
            }
            db.add(step)
            if item.deliverable_type.value == "diagram":
                db.add(
                    DiagramGenerationJobRecord(
                        workspace_id=record.workspace_id,
                        session_id=record.id,
                        diagram_key=item.key.removeprefix("diagram."),
                        requested_by_user_id=user.id,
                        detail_level="standard",
                        reason="regenerate",
                        idempotency_key=f"exhausted-diagram-{uuid4()}",
                        status="error",
                        error_code="forced_diagram_failure",
                        error_message=f"{item.key} agotó sus intentos.",
                    )
                )
            else:
                db.add(
                    DeliverableGenerationJobRecord(
                        workspace_id=record.workspace_id,
                        session_id=record.id,
                        deliverable_key=item.key,
                        status="error",
                        product_mode="premium_enrichment",
                        idempotency_key=f"exhausted-deliverable-{uuid4()}",
                        error_code="forced_failure",
                        error_message=f"{item.key} agotó sus intentos.",
                    )
                )
        db.commit()

        status = ensure_product_build_orchestration(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_pro,
            current_user=user,
            options=ProductBuildOrchestrationOptions(
                current_stage="package",
                execute_jobs=True,
                allow_llm=False,
            ),
            catalog_stage_override="package",
        )

    assert status.lifecycle == ProductBuildLifecycle.requires_attention
    assert status.processing_queue is not None
    assert status.processing_queue.total_count == 0
