from uuid import uuid4

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from app.models import CommercialTier
from app.services.product_processing import (
    ProductBuildLifecycle,
    ProductBuildProductKey,
    ProductProcessingMode,
    ensure_product_build_run,
    list_product_build_runs,
    list_product_build_steps,
    update_product_build_run_state,
    upsert_product_build_step,
)
from app.services.product_processing.persistence import ProductBuildRunRecord, ProductBuildStepRecord  # noqa: F401


def _engine():
    return create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)


def test_product_build_run_is_idempotent_by_workspace_and_key() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)
    workspace_id = uuid4()
    session_id = uuid4()

    with Session(engine) as db:
        first = ensure_product_build_run(
            db,
            workspace_id=workspace_id,
            session_id=session_id,
            product_key=ProductBuildProductKey.blueprint_basic,
            product_mode=ProductProcessingMode.basic_free,
            entitlement_tier=CommercialTier.blueprint,
            access_state="allowed",
            idempotency_key="blueprint-basic:session-1",
        )
        second = ensure_product_build_run(
            db,
            workspace_id=workspace_id,
            session_id=session_id,
            product_key=ProductBuildProductKey.blueprint_basic,
            product_mode=ProductProcessingMode.basic_free,
            entitlement_tier=CommercialTier.blueprint,
            access_state="allowed",
            idempotency_key="blueprint-basic:session-1",
        )
        runs = list_product_build_runs(db, workspace_id=workspace_id, session_id=session_id)

    assert first.id == second.id
    assert len(runs) == 1


def test_product_build_steps_are_idempotent_and_survive_refresh() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)
    workspace_id = uuid4()
    session_id = uuid4()

    with Session(engine) as db:
        run = ensure_product_build_run(
            db,
            workspace_id=workspace_id,
            session_id=session_id,
            product_key=ProductBuildProductKey.blueprint_basic,
            product_mode=ProductProcessingMode.basic_free,
            idempotency_key="blueprint-basic:session-1",
        )
        upsert_product_build_step(
            db,
            run=run,
            step_key="commercial_result",
            status="running",
            stage_key="estimate",
            sequence=1,
            progress_percent=25,
            checkpoint_payload={"started": True},
        )
        upsert_product_build_step(
            db,
            run=run,
            step_key="commercial_result",
            status="completed",
            stage_key="estimate",
            sequence=1,
            progress_percent=100,
            checkpoint_payload={"completed": True},
        )
        update_product_build_run_state(
            db,
            run=run,
            lifecycle=ProductBuildLifecycle.running,
            completed_units=1,
            total_units=4,
            checkpoint_payload={"current_step": "commercial_result"},
        )
        db.commit()
        run_id = run.id

    with Session(engine) as db:
        refreshed = db.get(ProductBuildRunRecord, run_id)
        steps = list_product_build_steps(db, run_id=run_id)

    assert refreshed is not None
    assert refreshed.progress_percent == 25
    assert refreshed.checkpoint_payload == {"current_step": "commercial_result"}
    assert len(steps) == 1
    assert steps[0].status == "completed"
    assert steps[0].checkpoint_payload == {"completed": True}


def test_product_build_run_can_require_attention_without_losing_steps() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)
    workspace_id = uuid4()
    session_id = uuid4()

    with Session(engine) as db:
        run = ensure_product_build_run(
            db,
            workspace_id=workspace_id,
            session_id=session_id,
            product_key=ProductBuildProductKey.acp,
            product_mode=ProductProcessingMode.acp_implementation,
            idempotency_key="acp:session-1",
        )
        upsert_product_build_step(
            db,
            run=run,
            step_key="validate_readiness",
            status="completed",
            stage_key="validate",
            sequence=1,
            progress_percent=100,
        )
        update_product_build_run_state(
            db,
            run=run,
            lifecycle=ProductBuildLifecycle.requires_attention,
            completed_units=1,
            total_units=2,
            blocked_units=1,
            error_payload={"code": "attention_required"},
        )
        db.commit()
        run_id = run.id

    with Session(engine) as db:
        refreshed = db.get(ProductBuildRunRecord, run_id)
        steps = list_product_build_steps(db, run_id=run_id)

    assert refreshed is not None
    assert refreshed.lifecycle == "requires_attention"
    assert refreshed.requires_attention_at is not None
    assert refreshed.error_payload == {"code": "attention_required"}
    assert [step.step_key for step in steps] == ["validate_readiness"]
