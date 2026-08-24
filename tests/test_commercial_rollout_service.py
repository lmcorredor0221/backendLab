from __future__ import annotations

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.models import (
    AccessRequestCreateRequest,
    CommercialAccessRequestRecord,
    CommercialOrderRecord,
    CommercialOrderStatus,
    CommercialTier,
    SchemaMigrationRecord,
    SessionRecord,
    UserRecord,
)
from app.services.auth_service import hash_password
from app.services.commerce_service import request_access
from app.services.commercial_rollout_service import (
    MIGRATION_KEY_COMMERCIAL_QUOTA_ROLLOUT,
    apply_commercial_quota_rollout,
)
from app.services.commercial_quota_service import get_balance_snapshot, upsert_quota_product_config
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
        title="Commercial rollout test",
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return user, record


def test_apply_commercial_quota_rollout_initializes_existing_workspaces_recalculates_pending_and_cancels_legacy_orders() -> None:
    with _db_session() as session:
        upsert_quota_product_config(
            session,
            product_key="blueprint_pro",
            display_name="Blueprint Pro",
            initial_free_units=0,
        )
        session.commit()
        user, record = _seed_project_context(session, email="commercial-rollout@leanbuilder.local")

        pending_request = request_access(
            session,
            payload=AccessRequestCreateRequest(
                session_id=record.id,
                capability="blueprint.download",
                reason="Continuar Blueprint",
            ),
            record=record,
            current_user=user,
            product_key="blueprint_pro",
            target_tier=CommercialTier.blueprint_pro,
        )
        assert pending_request.status == "pending"

        legacy_order = CommercialOrderRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            buyer_user_id=user.id,
            status=CommercialOrderStatus.pending,
            currency="USD",
            subtotal_cents=4900,
            total_cents=4900,
            provider="hotmart",
            checkout_ref="legacy-checkout-1",
            checkout_url="https://checkout.example.com/legacy-1",
            idempotency_key="legacy-checkout-1",
            metadata_payload={"product_key": "blueprint_pro"},
        )
        session.add(legacy_order)
        session.commit()

        upsert_quota_product_config(
            session,
            product_key="blueprint_pro",
            display_name="Blueprint Pro",
            initial_free_units=1,
        )

        summary = apply_commercial_quota_rollout(session)

        session.refresh(legacy_order)
        pending_row = session.exec(
            select(CommercialAccessRequestRecord).where(
                CommercialAccessRequestRecord.id == pending_request.id
            )
        ).one()
        snapshot = get_balance_snapshot(session, workspace_id=record.workspace_id, product_key="blueprint_pro")
        migration = session.exec(
            select(SchemaMigrationRecord).where(
                SchemaMigrationRecord.migration_key == MIGRATION_KEY_COMMERCIAL_QUOTA_ROLLOUT
            )
        ).one()

        assert summary.workspaces_initialized == 1
        assert summary.pending_requests_before == 1
        assert summary.pending_requests_after == 0
        assert summary.pending_requests_auto_approved == 1
        assert summary.legacy_orders_canceled == 1
        assert pending_row.status.value == "approved"
        assert legacy_order.status == CommercialOrderStatus.canceled
        assert snapshot.total_available_units == 0
        assert migration.migration_key == MIGRATION_KEY_COMMERCIAL_QUOTA_ROLLOUT


def test_apply_commercial_quota_rollout_is_idempotent_after_migration_record_exists() -> None:
    with _db_session() as session:
        session.add(
            SchemaMigrationRecord(
                migration_key=MIGRATION_KEY_COMMERCIAL_QUOTA_ROLLOUT,
                description="already applied",
            )
        )
        session.commit()

        summary = apply_commercial_quota_rollout(session)

        assert summary.already_recorded is True
        assert summary.workspaces_scanned == 0
