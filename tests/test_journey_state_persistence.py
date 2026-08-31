from __future__ import annotations

from uuid import uuid4

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.models import (
    JourneyStateRecord,
    JourneyStateTransitionRecord,
    SessionRecord,
    SessionStage,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceRecord,
    WorkspaceRole,
)
from app.services.auth_service import hash_password
from app.services.product_processing.contracts import JourneyStateKey, JourneyStateSubstate
from app.services.product_processing.journey_state_machine_service import (
    initialize_journey_state,
    transition_journey_state,
)
from app.services.product_processing.product_journey_overview_service import build_product_journey_overview


def _engine():
    return create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)


def _seed_session(db: Session) -> tuple[UserRecord, SessionRecord]:
    user = UserRecord(
        email=f"journey-state-{uuid4()}@leanbuilder.local",
        full_name="Journey State Tester",
        password_hash=hash_password("Secret123!"),
    )
    db.add(user)
    db.flush()
    workspace = WorkspaceRecord(name="Journey State Workspace", slug=f"journey-state-{str(user.id)[:8]}")
    db.add(workspace)
    db.flush()
    db.add(WorkspaceMembershipRecord(workspace_id=workspace.id, user_id=user.id, role=WorkspaceRole.owner))
    record = SessionRecord(user_id=user.id, workspace_id=workspace.id, current_stage=SessionStage.ready_for_export)
    db.add(record)
    db.commit()
    db.refresh(user)
    db.refresh(record)
    return user, record


def test_journey_state_transitions_are_persisted_sequential_and_idempotent() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_session(db)
        initialize_journey_state(
            db,
            record=record,
            state_key=JourneyStateKey.blueprint_free_ready,
            actor_type="user",
            actor_user_id=user.id,
            correlation_id=f"session-created:{record.id}",
        )
        transition_journey_state(
            db,
            record=record,
            event_key="request_blueprint_pro_access",
            target_state_key=JourneyStateKey.blueprint_pro_access_requested,
            target_substate=JourneyStateSubstate.waiting_dependency,
            actor_type="user",
            actor_user_id=user.id,
            correlation_id="access-request:pro-1",
        )
        duplicate = transition_journey_state(
            db,
            record=record,
            event_key="request_blueprint_pro_access",
            target_state_key=JourneyStateKey.blueprint_pro_access_requested,
            target_substate=JourneyStateSubstate.waiting_dependency,
            actor_type="user",
            actor_user_id=user.id,
            correlation_id="access-request:pro-1",
        )
        db.commit()

        current = db.exec(select(JourneyStateRecord).where(JourneyStateRecord.session_id == record.id)).one()
        transitions = db.exec(
            select(JourneyStateTransitionRecord)
            .where(JourneyStateTransitionRecord.session_id == record.id)
            .order_by(JourneyStateTransitionRecord.sequence)
        ).all()

    assert current.state_key == JourneyStateKey.blueprint_pro_access_requested.value
    assert current.revision == 2
    assert [item.sequence for item in transitions] == [1, 2]
    assert [item.event_key for item in transitions] == ["journey_initialized", "request_blueprint_pro_access"]
    assert duplicate.state_source == "canonical"
    assert duplicate.current.substate == JourneyStateSubstate.waiting_dependency
    assert [item.event_key for item in duplicate.history] == ["journey_initialized", "request_blueprint_pro_access"]


def test_persisted_journey_state_is_authoritative_for_product_overview() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_session(db)
        initialize_journey_state(
            db,
            record=record,
            state_key=JourneyStateKey.blueprint_pro_active,
            substate=JourneyStateSubstate.idle,
            actor_user_id=user.id,
            correlation_id=f"session-created:{record.id}",
        )
        record.current_stage = SessionStage.draft_capture
        db.add(record)
        db.commit()

        overview = build_product_journey_overview(db, record=record, current_user=user)

    assert overview.journey_state_machine is not None
    assert overview.journey_state_machine.state_source == "canonical"
    assert overview.journey_state_machine.current.state_key == JourneyStateKey.blueprint_pro_active
    assert overview.journey_state_machine.current.href.endswith("/blueprint/pro")
    assert overview.current_stage.stage_key == "blueprint_pro"
    assert overview.current_stage.product_key.value == "blueprint_pro"


def test_legacy_overview_fallback_is_read_only_until_explicit_backfill() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_session(db)
        overview = build_product_journey_overview(db, record=record, current_user=user)
        persisted_rows = db.exec(select(JourneyStateRecord).where(JourneyStateRecord.session_id == record.id)).all()

    assert overview.journey_state_machine is not None
    assert overview.journey_state_machine.state_source == "legacy_projection"
    assert persisted_rows == []
