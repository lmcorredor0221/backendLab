from __future__ import annotations

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.models import (
    AccessRequestCreateRequest,
    CommercialAccessRequestRecord,
    CommercialEventRecord,
    CommercialAccessRequestStatus,
    CommercialEntitlementRecord,
    CommercialQuotaSourceKind,
    CommercialTier,
    JourneyArtifactState,
    JourneyStageArtifactRecord,
    JourneyStateRecord,
    JourneyStateTransitionRecord,
    SessionRecord,
    UserRecord,
)
from app.services.auth_service import hash_password
from app.services.commerce_service import request_access
from app.services.commercial_quota_service import get_balance_snapshot, grant_balance_units, upsert_quota_product_config
from app.services.workspace_access import ensure_personal_workspace


def _db_session() -> Session:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def _seed_project_context(session: Session, *, email: str) -> tuple[UserRecord, SessionRecord]:
    user = UserRecord(
        email=email,
        full_name=email.split("@")[0],
        password_hash=hash_password("Secret123!"),
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    workspace = ensure_personal_workspace(session, user).workspace
    record = SessionRecord(
        user_id=user.id,
        workspace_id=workspace.id,
        title="Premium Approval Request",
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return user, record


def test_request_access_auto_approves_when_workspace_has_available_balance() -> None:
    with _db_session() as session:
        user, record = _seed_project_context(session, email="quota-autoapprove@leanbuilder.local")
        tools_artifact = JourneyStageArtifactRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            stage_key="tools",
            artifact_kind="tool_recommendation",
            version_number=1,
            state=JourneyArtifactState.stale,
            stale_reasons=["memory_reprocessed"],
            proposal_payload={"summary": "Herramientas previas aprobadas"},
        )
        session.add(tools_artifact)
        session.commit()
        upsert_quota_product_config(
            session,
            product_key="acp",
            display_name="ACP",
            initial_free_units=1,
        )

        response = request_access(
            session,
            payload=AccessRequestCreateRequest(session_id=record.id, capability="acp.build", reason="Continuar ACP"),
            record=record,
            current_user=user,
            product_key="acp",
            target_tier=CommercialTier.acp,
        )

        access_request = session.exec(select(CommercialAccessRequestRecord)).one()
        entitlements = session.exec(select(CommercialEntitlementRecord)).all()
        handoff_event = session.exec(
            select(CommercialEventRecord).where(
                CommercialEventRecord.session_id == record.id,
                CommercialEventRecord.event_key == "blueprint_acp_handoff_finalized",
            )
        ).one()
        snapshot = get_balance_snapshot(session, workspace_id=record.workspace_id, product_key="acp")
        journey_state = session.exec(select(JourneyStateRecord).where(JourneyStateRecord.session_id == record.id)).one()
        journey_events = session.exec(
            select(JourneyStateTransitionRecord)
            .where(JourneyStateTransitionRecord.session_id == record.id)
            .order_by(JourneyStateTransitionRecord.sequence)
        ).all()

        assert response.status == CommercialAccessRequestStatus.approved
        assert access_request.status == CommercialAccessRequestStatus.approved
        assert access_request.resolution_note == "Autoaprobada por saldo disponible del workspace."
        assert len(entitlements) == 1
        session.refresh(tools_artifact)
        assert tools_artifact.state == JourneyArtifactState.approved
        assert tools_artifact.stale_reasons == []
        assert handoff_event.metadata_payload["closed_process_items"][0]["stage_key"] == "tools"
        assert snapshot.total_available_units == 0
        assert journey_state.state_key == "acp_prep"
        assert [item.event_key for item in journey_events] == [
            "journey_initialized",
            "request_acp_access",
            "approve_acp_access",
        ]


def test_request_access_stays_pending_when_workspace_has_no_available_balance() -> None:
    with _db_session() as session:
        user, record = _seed_project_context(session, email="quota-pending@leanbuilder.local")
        upsert_quota_product_config(
            session,
            product_key="blueprint_pro",
            display_name="Blueprint Pro",
            initial_free_units=0,
        )

        response = request_access(
            session,
            payload=AccessRequestCreateRequest(session_id=record.id, capability="blueprint.download", reason="Continuar Blueprint"),
            record=record,
            current_user=user,
            product_key="blueprint_pro",
            target_tier=CommercialTier.blueprint_pro,
        )

        access_request = session.exec(select(CommercialAccessRequestRecord)).one()
        entitlements = session.exec(select(CommercialEntitlementRecord)).all()
        snapshot = get_balance_snapshot(session, workspace_id=record.workspace_id, product_key="blueprint_pro")
        journey_state = session.exec(select(JourneyStateRecord).where(JourneyStateRecord.session_id == record.id)).one()

        assert response.status == CommercialAccessRequestStatus.pending
        assert access_request.status == CommercialAccessRequestStatus.pending
        assert entitlements == []
        assert snapshot.total_available_units == 0
        assert journey_state.state_key == "blueprint_pro_access_requested"


def test_grant_balance_auto_approves_oldest_pending_requests_in_fifo_order() -> None:
    with _db_session() as session:
        user, record_a = _seed_project_context(session, email="quota-fifo@leanbuilder.local")
        record_b = SessionRecord(
            user_id=user.id,
            workspace_id=record_a.workspace_id,
            title="Second Premium Approval Request",
        )
        session.add(record_b)
        session.commit()
        session.refresh(record_b)
        upsert_quota_product_config(
            session,
            product_key="acp",
            display_name="ACP",
            initial_free_units=0,
        )

        first = request_access(
            session,
            payload=AccessRequestCreateRequest(session_id=record_a.id, capability="acp.build", reason="Primera"),
            record=record_a,
            current_user=user,
            product_key="acp",
            target_tier=CommercialTier.acp,
        )
        second = request_access(
            session,
            payload=AccessRequestCreateRequest(session_id=record_b.id, capability="acp.build", reason="Segunda"),
            record=record_b,
            current_user=user,
            product_key="acp",
            target_tier=CommercialTier.acp,
        )
        assert first.status == CommercialAccessRequestStatus.pending
        assert second.status == CommercialAccessRequestStatus.pending

        grant_balance_units(
            session,
            workspace_id=record_a.workspace_id,
            product_key="acp",
            source_kind=CommercialQuotaSourceKind.one_time,
            units=1,
            bucket_key="otp-fifo",
            source_ref="order:otp-fifo",
            actor_user_id=user.id,
        )

        requests = session.exec(select(CommercialAccessRequestRecord).order_by(CommercialAccessRequestRecord.created_at.asc())).all()
        entitlements = session.exec(select(CommercialEntitlementRecord).order_by(CommercialEntitlementRecord.created_at.asc())).all()
        snapshot = get_balance_snapshot(session, workspace_id=record_a.workspace_id, product_key="acp")

        assert requests[0].status == CommercialAccessRequestStatus.approved
        assert requests[1].status == CommercialAccessRequestStatus.pending
        assert len(entitlements) == 1
        assert snapshot.total_available_units == 0
