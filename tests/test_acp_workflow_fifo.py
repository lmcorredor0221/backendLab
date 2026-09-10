from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from app.api.routes.sessions import build_construction_readiness_view, build_snapshot, resolve_acp_preview
from app.models import (
    ACPPhaseCommandRequest,
    ACPWorkflowRunStatus,
    CommercialEntitlementRecord,
    CommercialEntitlementSource,
    CommercialEntitlementStatus,
    CommercialTier,
    ProjectTitleSource,
    SessionRecord,
    SessionStage,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceRecord,
    WorkspaceRole,
)
from app.services.acp_continuity import load_construction_question_response_records
from app.services.acp_workflow_service import (
    ACPPhaseSequenceError,
    ensure_acp_run,
    run_acp_phase,
)
from app.services.auth_service import hash_password


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _seed_user_and_session(session: Session) -> tuple[UserRecord, SessionRecord]:
    user = UserRecord(
        email=f"acp-fifo-{uuid4().hex[:6]}@leanbuilder.local",
        full_name="ACP FIFO Tester",
        password_hash=hash_password("Secret123!"),
    )
    session.add(user)
    session.flush()

    workspace = WorkspaceRecord(
        name="ACP FIFO Workspace",
        slug=f"acp-fifo-{str(user.id)[:8]}",
        created_by_user_id=user.id,
    )
    session.add(workspace)
    session.flush()

    session.add(
        WorkspaceMembershipRecord(
            workspace_id=workspace.id,
            user_id=user.id,
            role=WorkspaceRole.owner,
        )
    )

    record = SessionRecord(
        user_id=user.id,
        workspace_id=workspace.id,
        title="ACP FIFO Project",
        title_source=ProjectTitleSource.manual,
        current_stage=SessionStage.ready_for_export,
        commercial_tier=CommercialTier.acp,
    )
    session.add(record)
    session.flush()

    entitlement = CommercialEntitlementRecord(
        workspace_id=workspace.id,
        session_id=record.id,
        user_id=user.id,
        product_key="acp",
        tier=CommercialTier.acp,
        status=CommercialEntitlementStatus.active,
        source=CommercialEntitlementSource.checkout,
    )
    session.add(entitlement)
    session.commit()
    session.refresh(user)
    session.refresh(record)
    return user, record


def test_acp_phase_sequence_blocks_skipping_phases(db_session: Session) -> None:
    user, record = _seed_user_and_session(db_session)
    snapshot = build_snapshot(db_session, record, current_user=user)
    preview = resolve_acp_preview(db_session, record)
    response_records = load_construction_question_response_records(db_session, record.id)
    readiness = build_construction_readiness_view(preview, response_records)

    run = ensure_acp_run(db_session, record=record, current_user=user, snapshot=snapshot)

    # Attempting to run phase 5 without running the previous ACP phases must fail.
    with pytest.raises(ACPPhaseSequenceError) as excinfo:
        run_acp_phase(
            db_session,
            run=run,
            phase_key="acp_quality_gates",
            payload=ACPPhaseCommandRequest(),
            preview=preview,
            readiness=readiness,
        )

    assert excinfo.value.phase_key == "acp_quality_gates"
    assert excinfo.value.blocking_phase_key == "acp_graphic_simulation"


def test_blocked_phase_prevents_subsequent_phases(db_session: Session) -> None:
    user, record = _seed_user_and_session(db_session)
    snapshot = build_snapshot(db_session, record, current_user=user)
    preview = resolve_acp_preview(db_session, record)
    response_records = load_construction_question_response_records(db_session, record.id)
    readiness = build_construction_readiness_view(preview, response_records)

    run = ensure_acp_run(db_session, record=record, current_user=user, snapshot=snapshot)

    # Phase 1 runs and becomes blocked because session lacks approved discovery
    p1 = run_acp_phase(
        db_session,
        run=run,
        phase_key="acp_input_readiness",
        payload=ACPPhaseCommandRequest(),
        preview=preview,
        readiness=readiness,
    )
    db_session.commit()
    assert p1.status == ACPWorkflowRunStatus.blocked

    # Attempting to run Phase 2 without force must raise ACPPhaseSequenceError
    with pytest.raises(ACPPhaseSequenceError) as excinfo:
        run_acp_phase(
            db_session,
            run=run,
            phase_key="acp_questions_resolution",
            payload=ACPPhaseCommandRequest(),
            preview=preview,
            readiness=readiness,
        )

    assert excinfo.value.phase_key == "acp_questions_resolution"
    assert excinfo.value.blocking_phase_key == "acp_input_readiness"
    assert excinfo.value.blocking_phase_status == "blocked"


def test_force_flag_allows_override_when_explicitly_requested(db_session: Session) -> None:
    user, record = _seed_user_and_session(db_session)
    snapshot = build_snapshot(db_session, record, current_user=user)
    preview = resolve_acp_preview(db_session, record)
    response_records = load_construction_question_response_records(db_session, record.id)
    readiness = build_construction_readiness_view(preview, response_records)

    run = ensure_acp_run(db_session, record=record, current_user=user, snapshot=snapshot)

    # Phase 2 with force=True bypasses the strict sequence block.
    p2 = run_acp_phase(
        db_session,
        run=run,
        phase_key="acp_questions_resolution",
        payload=ACPPhaseCommandRequest(force=True),
        preview=preview,
        readiness=readiness,
    )
    db_session.commit()
    assert p2.phase_key == "acp_questions_resolution"


def test_legacy_acp_phase_keys_are_canonicalized(db_session: Session) -> None:
    user, record = _seed_user_and_session(db_session)
    snapshot = build_snapshot(db_session, record, current_user=user)
    preview = resolve_acp_preview(db_session, record)
    response_records = load_construction_question_response_records(db_session, record.id)
    readiness = build_construction_readiness_view(preview, response_records)

    run = ensure_acp_run(db_session, record=record, current_user=user, snapshot=snapshot)

    phase = run_acp_phase(
        db_session,
        run=run,
        phase_key="blueprint_validation",
        payload=ACPPhaseCommandRequest(force=True),
        preview=preview,
        readiness=readiness,
    )
    db_session.commit()

    assert phase.phase_key == "acp_input_readiness"
