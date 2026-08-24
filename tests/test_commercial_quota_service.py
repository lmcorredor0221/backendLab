from __future__ import annotations

from datetime import timedelta
from uuid import UUID

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from app.models import (
    CommercialQuotaBucketStatus,
    CommercialQuotaLedgerMovementType,
    CommercialQuotaSourceKind,
    UserRecord,
    WorkspaceRecord,
)
from app.services.auth_service import hash_password
from app.services.commercial_quota_service import (
    consume_balance_units,
    get_balance_snapshot,
    initialize_workspace_commercial_quota,
    list_balance_buckets,
    list_balance_ledger,
    resolve_effective_quota_config,
    sync_workspace_free_bucket,
    upsert_quota_product_config,
    upsert_workspace_quota_override,
    grant_balance_units,
)
from app.services.workspace_access import ensure_personal_workspace


def _db_session() -> Session:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def _create_user(session: Session, *, email: str) -> UserRecord:
    user = UserRecord(
        email=email,
        full_name=email.split("@")[0],
        password_hash=hash_password("Secret123!"),
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def _workspace_id(session: Session, user: UserRecord) -> UUID:
    context = ensure_personal_workspace(session, user)
    return context.workspace.id


def test_workspace_creation_initializes_quota_buckets_for_seeded_products() -> None:
    with _db_session() as session:
        user = _create_user(session, email="quota-init@leanbuilder.local")
        workspace_id = _workspace_id(session, user)

        pro_buckets = list_balance_buckets(session, workspace_id=workspace_id, product_key="blueprint_pro")
        acp_buckets = list_balance_buckets(session, workspace_id=workspace_id, product_key="acp")

        assert len(pro_buckets) == 1
        assert len(acp_buckets) == 1
        assert pro_buckets[0].source_kind == CommercialQuotaSourceKind.free
        assert acp_buckets[0].source_kind == CommercialQuotaSourceKind.free
        assert pro_buckets[0].status in {CommercialQuotaBucketStatus.exhausted, CommercialQuotaBucketStatus.canceled}
        assert acp_buckets[0].status in {CommercialQuotaBucketStatus.exhausted, CommercialQuotaBucketStatus.canceled}


def test_effective_quota_config_applies_workspace_override() -> None:
    with _db_session() as session:
        user = _create_user(session, email="quota-override@leanbuilder.local")
        workspace_id = _workspace_id(session, user)
        upsert_quota_product_config(
            session,
            product_key="blueprint_pro",
            display_name="Blueprint Pro",
            initial_free_units=1,
            consumption_priority=["free", "subscription", "one_time"],
            default_checkout_ttl_minutes=30,
        )
        upsert_workspace_quota_override(
            session,
            workspace_id=workspace_id,
            product_key="blueprint_pro",
            free_units_override=3,
            consumption_priority_override=["free", "one_time", "subscription"],
            default_checkout_ttl_minutes_override=45,
            debt_enabled_override=False,
            updated_by_user_id=user.id,
        )

        resolved = resolve_effective_quota_config(session, workspace_id=workspace_id, product_key="blueprint_pro")

        assert resolved.initial_free_units == 3
        assert [item.value for item in resolved.consumption_priority] == ["free", "one_time", "subscription"]
        assert resolved.default_checkout_ttl_minutes == 45
        assert resolved.debt_enabled is False


def test_consume_balance_respects_priority_and_earliest_expiry() -> None:
    with _db_session() as session:
        user = _create_user(session, email="quota-consume@leanbuilder.local")
        workspace_id = _workspace_id(session, user)
        now = session.get(WorkspaceRecord, workspace_id).created_at
        upsert_quota_product_config(
            session,
            product_key="blueprint_pro",
            display_name="Blueprint Pro",
            initial_free_units=1,
        )
        initialize_workspace_commercial_quota(session, workspace_id=workspace_id, actor_user_id=user.id, at_time=now)
        grant_balance_units(
            session,
            workspace_id=workspace_id,
            product_key="blueprint_pro",
            source_kind=CommercialQuotaSourceKind.subscription,
            units=2,
            bucket_key="sub-current",
            source_ref="subscription:current",
            actor_user_id=user.id,
            starts_at=now,
            ends_at=now + timedelta(days=30),
        )
        grant_balance_units(
            session,
            workspace_id=workspace_id,
            product_key="blueprint_pro",
            source_kind=CommercialQuotaSourceKind.one_time,
            units=1,
            bucket_key="otp-earlier",
            source_ref="order:earlier",
            actor_user_id=user.id,
            starts_at=now,
            ends_at=now + timedelta(days=10),
        )
        grant_balance_units(
            session,
            workspace_id=workspace_id,
            product_key="blueprint_pro",
            source_kind=CommercialQuotaSourceKind.one_time,
            units=1,
            bucket_key="otp-later",
            source_ref="order:later",
            actor_user_id=user.id,
            starts_at=now,
            ends_at=now + timedelta(days=20),
        )

        touched_first = consume_balance_units(
            session,
            workspace_id=workspace_id,
            product_key="blueprint_pro",
            units=3,
            actor_user_id=user.id,
            source_ref="request:first",
        )
        snapshot_after_first = get_balance_snapshot(session, workspace_id=workspace_id, product_key="blueprint_pro")

        assert [bucket.bucket_key for bucket in touched_first] == [
            "free:blueprint_pro:default",
            "sub-current",
        ]
        assert snapshot_after_first.total_available_units == 2

        touched_second = consume_balance_units(
            session,
            workspace_id=workspace_id,
            product_key="blueprint_pro",
            units=2,
            actor_user_id=user.id,
            source_ref="request:second",
        )
        snapshot_after_second = get_balance_snapshot(session, workspace_id=workspace_id, product_key="blueprint_pro")

        assert [bucket.bucket_key for bucket in touched_second] == ["otp-earlier", "otp-later"]
        assert snapshot_after_second.total_available_units == 0
        ledger = list_balance_ledger(session, workspace_id=workspace_id, product_key="blueprint_pro")
        consume_entries = [entry for entry in ledger if entry.movement_type == CommercialQuotaLedgerMovementType.consume]
        assert len(consume_entries) == 4


def test_sync_free_bucket_overwrite_floors_available_at_zero_and_writes_ledger() -> None:
    with _db_session() as session:
        user = _create_user(session, email="quota-floor@leanbuilder.local")
        workspace_id = _workspace_id(session, user)
        upsert_quota_product_config(
            session,
            product_key="blueprint_pro",
            display_name="Blueprint Pro",
            initial_free_units=3,
        )
        initialize_workspace_commercial_quota(session, workspace_id=workspace_id, actor_user_id=user.id)
        consume_balance_units(
            session,
            workspace_id=workspace_id,
            product_key="blueprint_pro",
            units=2,
            actor_user_id=user.id,
            source_ref="request:consume-two",
        )
        upsert_workspace_quota_override(
            session,
            workspace_id=workspace_id,
            product_key="blueprint_pro",
            free_units_override=1,
            updated_by_user_id=user.id,
        )

        sync_workspace_free_bucket(session, workspace_id=workspace_id, product_key="blueprint_pro", actor_user_id=user.id)
        snapshot = get_balance_snapshot(session, workspace_id=workspace_id, product_key="blueprint_pro")
        ledger = list_balance_ledger(session, workspace_id=workspace_id, product_key="blueprint_pro")

        assert snapshot.total_available_units == 0
        assert ledger[-1].movement_type == CommercialQuotaLedgerMovementType.overwrite
        assert ledger[-1].delta_units == -1
