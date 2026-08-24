from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

from sqlmodel import Session, select

from app.models import (
    CommercialBalanceBucketRecord,
    CommercialBalanceLedgerRecord,
    CommercialQuotaBucketStatus,
    CommercialQuotaLedgerMovementType,
    CommercialQuotaProductConfigRecord,
    CommercialQuotaSourceKind,
    CommercialQuotaWorkspaceOverrideRecord,
    utc_now,
)


DEFAULT_CONSUMPTION_PRIORITY: tuple[CommercialQuotaSourceKind, ...] = (
    CommercialQuotaSourceKind.free,
    CommercialQuotaSourceKind.subscription,
    CommercialQuotaSourceKind.one_time,
)

WORKSPACE_FREE_BUCKET_SUFFIX = "default"

QUOTA_PRODUCT_SEED: tuple[dict[str, object], ...] = (
    {
        "product_key": "blueprint_pro",
        "display_name": "Blueprint Pro",
        "enabled": True,
        "initial_free_units": 0,
    },
    {
        "product_key": "acp",
        "display_name": "ACP",
        "enabled": True,
        "initial_free_units": 0,
    },
)


@dataclass(frozen=True)
class EffectiveCommercialQuotaConfig:
    product_key: str
    display_name: str
    enabled: bool
    initial_free_units: int
    consumption_priority: tuple[CommercialQuotaSourceKind, ...]
    checkout_required_on_zero_balance: bool
    fifo_auto_approval_enabled: bool
    default_blocked_request_ttl_hours: int
    default_checkout_ttl_minutes: int
    debt_enabled: bool
    allow_manual_override_without_charge: bool
    allow_courtesy: bool
    allow_debt_pending: bool
    catalog_priority_strategy: str
    sync_retry_limit: int
    duplicate_conflict_visibility: str
    override_id: UUID | None = None


@dataclass(frozen=True)
class CommercialBucketBalance:
    bucket_id: UUID
    bucket_key: str
    source_kind: CommercialQuotaSourceKind
    status: CommercialQuotaBucketStatus
    units_granted: int
    units_consumed: int
    available_units: int
    starts_at: datetime
    ends_at: datetime | None
    source_ref: str


@dataclass(frozen=True)
class CommercialBalanceSnapshot:
    workspace_id: UUID
    product_key: str
    total_available_units: int
    by_source_kind: dict[CommercialQuotaSourceKind, int]
    buckets: tuple[CommercialBucketBalance, ...]


def _normalize_consumption_priority(
    values: list[str] | tuple[str, ...] | tuple[CommercialQuotaSourceKind, ...] | None,
) -> tuple[CommercialQuotaSourceKind, ...]:
    if not values:
        return DEFAULT_CONSUMPTION_PRIORITY
    resolved: list[CommercialQuotaSourceKind] = []
    for raw_value in values:
        try:
            candidate = raw_value if isinstance(raw_value, CommercialQuotaSourceKind) else CommercialQuotaSourceKind(str(raw_value).strip())
        except ValueError:
            continue
        if candidate not in resolved:
            resolved.append(candidate)
    for default_kind in DEFAULT_CONSUMPTION_PRIORITY:
        if default_kind not in resolved:
            resolved.append(default_kind)
    return tuple(resolved)


def _priority_to_storage(values: tuple[CommercialQuotaSourceKind, ...]) -> list[str]:
    return [item.value for item in values]


def _workspace_free_bucket_key(product_key: str) -> str:
    return f"free:{product_key}:{WORKSPACE_FREE_BUCKET_SUFFIX}"


def _is_override_effective(record: CommercialQuotaWorkspaceOverrideRecord, *, at_time: datetime) -> bool:
    if not record.is_active:
        return False
    if record.effective_from is not None and record.effective_from > at_time:
        return False
    if record.effective_to is not None and record.effective_to <= at_time:
        return False
    return True


def _derive_bucket_status(bucket: CommercialBalanceBucketRecord, *, at_time: datetime) -> CommercialQuotaBucketStatus:
    if bucket.status == CommercialQuotaBucketStatus.canceled:
        return CommercialQuotaBucketStatus.canceled
    if bucket.starts_at > at_time:
        return CommercialQuotaBucketStatus.scheduled
    if bucket.ends_at is not None and bucket.ends_at <= at_time:
        return CommercialQuotaBucketStatus.expired
    if bucket.units_granted - bucket.units_consumed <= 0:
        return CommercialQuotaBucketStatus.exhausted
    return CommercialQuotaBucketStatus.active


def _refresh_bucket_state(bucket: CommercialBalanceBucketRecord, *, at_time: datetime) -> bool:
    next_status = _derive_bucket_status(bucket, at_time=at_time)
    if next_status == bucket.status:
        return False
    bucket.status = next_status
    bucket.updated_at = at_time
    return True


def _bucket_available_units(bucket: CommercialBalanceBucketRecord, *, at_time: datetime) -> int:
    if bucket.status != CommercialQuotaBucketStatus.active:
        return 0
    if bucket.starts_at > at_time:
        return 0
    if bucket.ends_at is not None and bucket.ends_at <= at_time:
        return 0
    return max(0, bucket.units_granted - bucket.units_consumed)


def _record_balance_ledger(
    session: Session,
    *,
    workspace_id: UUID,
    product_key: str,
    bucket_id: UUID | None,
    movement_type: CommercialQuotaLedgerMovementType,
    source_kind: CommercialQuotaSourceKind,
    delta_units: int,
    balance_before_units: int,
    balance_after_units: int,
    bucket_balance_before_units: int,
    bucket_balance_after_units: int,
    source_ref: str,
    actor_user_id: UUID | None = None,
    order_id: UUID | None = None,
    payment_id: UUID | None = None,
    access_request_id: UUID | None = None,
    metadata: dict[str, object] | None = None,
) -> CommercialBalanceLedgerRecord:
    entry = CommercialBalanceLedgerRecord(
        workspace_id=workspace_id,
        product_key=product_key,
        bucket_id=bucket_id,
        movement_type=movement_type,
        source_kind=source_kind,
        delta_units=delta_units,
        balance_before_units=balance_before_units,
        balance_after_units=balance_after_units,
        bucket_balance_before_units=bucket_balance_before_units,
        bucket_balance_after_units=bucket_balance_after_units,
        source_ref=source_ref,
        actor_user_id=actor_user_id,
        order_id=order_id,
        payment_id=payment_id,
        access_request_id=access_request_id,
        metadata_payload=dict(metadata or {}),
    )
    session.add(entry)
    return entry


def ensure_quota_seed(session: Session) -> None:
    for item in QUOTA_PRODUCT_SEED:
        existing = session.exec(
            select(CommercialQuotaProductConfigRecord).where(
                CommercialQuotaProductConfigRecord.product_key == str(item["product_key"])
            )
        ).first()
        if existing is None:
            session.add(
                CommercialQuotaProductConfigRecord(
                    product_key=str(item["product_key"]),
                    display_name=str(item["display_name"]),
                    enabled=bool(item["enabled"]),
                    initial_free_units=int(item["initial_free_units"]),
                    consumption_priority=_priority_to_storage(DEFAULT_CONSUMPTION_PRIORITY),
                )
            )
            continue
        changed = False
        if not existing.display_name:
            existing.display_name = str(item["display_name"])
            changed = True
        if not existing.consumption_priority:
            existing.consumption_priority = _priority_to_storage(DEFAULT_CONSUMPTION_PRIORITY)
            changed = True
        if changed:
            existing.updated_at = utc_now()
            session.add(existing)


def list_quota_product_configs(session: Session) -> list[CommercialQuotaProductConfigRecord]:
    ensure_quota_seed(session)
    return session.exec(
        select(CommercialQuotaProductConfigRecord).order_by(CommercialQuotaProductConfigRecord.product_key.asc())
    ).all()


def get_quota_product_config(session: Session, *, product_key: str) -> CommercialQuotaProductConfigRecord:
    ensure_quota_seed(session)
    record = session.exec(
        select(CommercialQuotaProductConfigRecord).where(CommercialQuotaProductConfigRecord.product_key == product_key)
    ).first()
    if record is None:
        raise ValueError(f"Quota product config not found for {product_key}.")
    return record


def upsert_quota_product_config(
    session: Session,
    *,
    product_key: str,
    display_name: str,
    enabled: bool = True,
    initial_free_units: int = 0,
    consumption_priority: list[str] | tuple[str, ...] | tuple[CommercialQuotaSourceKind, ...] | None = None,
    checkout_required_on_zero_balance: bool = True,
    fifo_auto_approval_enabled: bool = True,
    default_blocked_request_ttl_hours: int = 72,
    default_checkout_ttl_minutes: int = 30,
    debt_enabled: bool = True,
    allow_manual_override_without_charge: bool = True,
    allow_courtesy: bool = True,
    allow_debt_pending: bool = True,
    catalog_priority_strategy: str = "minimum_sufficient",
    sync_retry_limit: int = 5,
    duplicate_conflict_visibility: str = "platform_admin_only",
    metadata: dict[str, object] | None = None,
) -> CommercialQuotaProductConfigRecord:
    ensure_quota_seed(session)
    record = session.exec(
        select(CommercialQuotaProductConfigRecord).where(CommercialQuotaProductConfigRecord.product_key == product_key)
    ).first()
    if record is None:
        record = CommercialQuotaProductConfigRecord(product_key=product_key)
    record.display_name = display_name
    record.enabled = enabled
    record.initial_free_units = max(0, initial_free_units)
    record.consumption_priority = _priority_to_storage(_normalize_consumption_priority(consumption_priority))
    record.checkout_required_on_zero_balance = checkout_required_on_zero_balance
    record.fifo_auto_approval_enabled = fifo_auto_approval_enabled
    record.default_blocked_request_ttl_hours = default_blocked_request_ttl_hours
    record.default_checkout_ttl_minutes = default_checkout_ttl_minutes
    record.debt_enabled = debt_enabled
    record.allow_manual_override_without_charge = allow_manual_override_without_charge
    record.allow_courtesy = allow_courtesy
    record.allow_debt_pending = allow_debt_pending
    record.catalog_priority_strategy = catalog_priority_strategy
    record.sync_retry_limit = sync_retry_limit
    record.duplicate_conflict_visibility = duplicate_conflict_visibility
    record.metadata_payload = dict(metadata or {})
    record.updated_at = utc_now()
    session.add(record)
    session.flush()
    return record


def upsert_workspace_quota_override(
    session: Session,
    *,
    workspace_id: UUID,
    product_key: str,
    is_active: bool = True,
    enabled_override: bool | None = None,
    free_units_override: int | None = None,
    consumption_priority_override: list[str] | tuple[str, ...] | None = None,
    checkout_required_on_zero_balance_override: bool | None = None,
    fifo_auto_approval_enabled_override: bool | None = None,
    default_blocked_request_ttl_hours_override: int | None = None,
    default_checkout_ttl_minutes_override: int | None = None,
    debt_enabled_override: bool | None = None,
    effective_from: datetime | None = None,
    effective_to: datetime | None = None,
    notes: str = "",
    updated_by_user_id: UUID | None = None,
    metadata: dict[str, object] | None = None,
) -> CommercialQuotaWorkspaceOverrideRecord:
    record = session.exec(
        select(CommercialQuotaWorkspaceOverrideRecord).where(
            CommercialQuotaWorkspaceOverrideRecord.workspace_id == workspace_id,
            CommercialQuotaWorkspaceOverrideRecord.product_key == product_key,
        )
    ).first()
    if record is None:
        record = CommercialQuotaWorkspaceOverrideRecord(
            workspace_id=workspace_id,
            product_key=product_key,
        )
    record.is_active = is_active
    record.enabled_override = enabled_override
    record.free_units_override = free_units_override
    record.consumption_priority_override = _priority_to_storage(
        _normalize_consumption_priority(consumption_priority_override)
    ) if consumption_priority_override is not None else []
    record.checkout_required_on_zero_balance_override = checkout_required_on_zero_balance_override
    record.fifo_auto_approval_enabled_override = fifo_auto_approval_enabled_override
    record.default_blocked_request_ttl_hours_override = default_blocked_request_ttl_hours_override
    record.default_checkout_ttl_minutes_override = default_checkout_ttl_minutes_override
    record.debt_enabled_override = debt_enabled_override
    record.effective_from = effective_from
    record.effective_to = effective_to
    record.notes = notes
    record.updated_by_user_id = updated_by_user_id
    record.metadata_payload = dict(metadata or {})
    record.updated_at = utc_now()
    session.add(record)
    session.flush()
    return record


def resolve_effective_quota_config(
    session: Session,
    *,
    workspace_id: UUID,
    product_key: str,
    at_time: datetime | None = None,
) -> EffectiveCommercialQuotaConfig:
    ensure_quota_seed(session)
    now = at_time or utc_now()
    base = get_quota_product_config(session, product_key=product_key)
    override = session.exec(
        select(CommercialQuotaWorkspaceOverrideRecord).where(
            CommercialQuotaWorkspaceOverrideRecord.workspace_id == workspace_id,
            CommercialQuotaWorkspaceOverrideRecord.product_key == product_key,
        )
    ).first()
    effective_override = override if override is not None and _is_override_effective(override, at_time=now) else None
    return EffectiveCommercialQuotaConfig(
        product_key=base.product_key,
        display_name=base.display_name,
        enabled=effective_override.enabled_override if effective_override and effective_override.enabled_override is not None else base.enabled,
        initial_free_units=(
            effective_override.free_units_override
            if effective_override and effective_override.free_units_override is not None
            else base.initial_free_units
        ),
        consumption_priority=(
            _normalize_consumption_priority(effective_override.consumption_priority_override)
            if effective_override and effective_override.consumption_priority_override
            else _normalize_consumption_priority(base.consumption_priority)
        ),
        checkout_required_on_zero_balance=(
            effective_override.checkout_required_on_zero_balance_override
            if effective_override and effective_override.checkout_required_on_zero_balance_override is not None
            else base.checkout_required_on_zero_balance
        ),
        fifo_auto_approval_enabled=(
            effective_override.fifo_auto_approval_enabled_override
            if effective_override and effective_override.fifo_auto_approval_enabled_override is not None
            else base.fifo_auto_approval_enabled
        ),
        default_blocked_request_ttl_hours=(
            effective_override.default_blocked_request_ttl_hours_override
            if effective_override and effective_override.default_blocked_request_ttl_hours_override is not None
            else base.default_blocked_request_ttl_hours
        ),
        default_checkout_ttl_minutes=(
            effective_override.default_checkout_ttl_minutes_override
            if effective_override and effective_override.default_checkout_ttl_minutes_override is not None
            else base.default_checkout_ttl_minutes
        ),
        debt_enabled=(
            effective_override.debt_enabled_override
            if effective_override and effective_override.debt_enabled_override is not None
            else base.debt_enabled
        ),
        allow_manual_override_without_charge=base.allow_manual_override_without_charge,
        allow_courtesy=base.allow_courtesy,
        allow_debt_pending=base.allow_debt_pending,
        catalog_priority_strategy=base.catalog_priority_strategy,
        sync_retry_limit=base.sync_retry_limit,
        duplicate_conflict_visibility=base.duplicate_conflict_visibility,
        override_id=effective_override.id if effective_override is not None else None,
    )


def refresh_workspace_balance_bucket_states(
    session: Session,
    *,
    workspace_id: UUID,
    product_key: str | None = None,
    at_time: datetime | None = None,
) -> None:
    now = at_time or utc_now()
    statement = select(CommercialBalanceBucketRecord).where(CommercialBalanceBucketRecord.workspace_id == workspace_id)
    if product_key:
        statement = statement.where(CommercialBalanceBucketRecord.product_key == product_key)
    buckets = session.exec(statement).all()
    for bucket in buckets:
        if _refresh_bucket_state(bucket, at_time=now):
            session.add(bucket)


def list_balance_buckets(
    session: Session,
    *,
    workspace_id: UUID,
    product_key: str,
    at_time: datetime | None = None,
) -> list[CommercialBalanceBucketRecord]:
    now = at_time or utc_now()
    refresh_workspace_balance_bucket_states(session, workspace_id=workspace_id, product_key=product_key, at_time=now)
    return session.exec(
        select(CommercialBalanceBucketRecord)
        .where(
            CommercialBalanceBucketRecord.workspace_id == workspace_id,
            CommercialBalanceBucketRecord.product_key == product_key,
        )
        .order_by(
            CommercialBalanceBucketRecord.starts_at.asc(),
            CommercialBalanceBucketRecord.created_at.asc(),
            CommercialBalanceBucketRecord.bucket_key.asc(),
        )
    ).all()


def list_balance_ledger(
    session: Session,
    *,
    workspace_id: UUID,
    product_key: str,
) -> list[CommercialBalanceLedgerRecord]:
    return session.exec(
        select(CommercialBalanceLedgerRecord)
        .where(
            CommercialBalanceLedgerRecord.workspace_id == workspace_id,
            CommercialBalanceLedgerRecord.product_key == product_key,
        )
        .order_by(CommercialBalanceLedgerRecord.created_at.asc(), CommercialBalanceLedgerRecord.id.asc())
    ).all()


def get_balance_snapshot(
    session: Session,
    *,
    workspace_id: UUID,
    product_key: str,
    at_time: datetime | None = None,
) -> CommercialBalanceSnapshot:
    now = at_time or utc_now()
    buckets = list_balance_buckets(session, workspace_id=workspace_id, product_key=product_key, at_time=now)
    by_source_kind: dict[CommercialQuotaSourceKind, int] = {
        CommercialQuotaSourceKind.free: 0,
        CommercialQuotaSourceKind.subscription: 0,
        CommercialQuotaSourceKind.one_time: 0,
        CommercialQuotaSourceKind.adjustment: 0,
    }
    resolved_buckets: list[CommercialBucketBalance] = []
    total_available = 0
    for bucket in buckets:
        available_units = _bucket_available_units(bucket, at_time=now)
        by_source_kind[bucket.source_kind] = by_source_kind.get(bucket.source_kind, 0) + available_units
        total_available += available_units
        resolved_buckets.append(
            CommercialBucketBalance(
                bucket_id=bucket.id,
                bucket_key=bucket.bucket_key,
                source_kind=bucket.source_kind,
                status=bucket.status,
                units_granted=bucket.units_granted,
                units_consumed=bucket.units_consumed,
                available_units=available_units,
                starts_at=bucket.starts_at,
                ends_at=bucket.ends_at,
                source_ref=bucket.source_ref,
            )
        )
    return CommercialBalanceSnapshot(
        workspace_id=workspace_id,
        product_key=product_key,
        total_available_units=total_available,
        by_source_kind=by_source_kind,
        buckets=tuple(resolved_buckets),
    )


def initialize_workspace_commercial_quota(
    session: Session,
    *,
    workspace_id: UUID,
    actor_user_id: UUID | None = None,
    at_time: datetime | None = None,
) -> None:
    ensure_quota_seed(session)
    for config in list_quota_product_configs(session):
        sync_workspace_free_bucket(
            session,
            workspace_id=workspace_id,
            product_key=config.product_key,
            actor_user_id=actor_user_id,
            at_time=at_time,
        )


def sync_workspace_free_bucket(
    session: Session,
    *,
    workspace_id: UUID,
    product_key: str,
    actor_user_id: UUID | None = None,
    at_time: datetime | None = None,
) -> CommercialBalanceBucketRecord:
    now = at_time or utc_now()
    config = resolve_effective_quota_config(session, workspace_id=workspace_id, product_key=product_key, at_time=now)
    before_snapshot = get_balance_snapshot(session, workspace_id=workspace_id, product_key=product_key, at_time=now)
    bucket_key = _workspace_free_bucket_key(product_key)
    bucket = session.exec(
        select(CommercialBalanceBucketRecord).where(
            CommercialBalanceBucketRecord.workspace_id == workspace_id,
            CommercialBalanceBucketRecord.product_key == product_key,
            CommercialBalanceBucketRecord.bucket_key == bucket_key,
        )
    ).first()
    before_bucket_available = 0
    created = False
    if bucket is None:
        created = True
        bucket = CommercialBalanceBucketRecord(
            workspace_id=workspace_id,
            product_key=product_key,
            bucket_key=bucket_key,
            source_kind=CommercialQuotaSourceKind.free,
            status=CommercialQuotaBucketStatus.scheduled,
            units_granted=0,
            units_consumed=0,
            source_ref="workspace_free_policy",
            granted_by_user_id=actor_user_id,
            starts_at=now,
        )
        session.add(bucket)
    else:
        before_bucket_available = _bucket_available_units(bucket, at_time=now)
    bucket.units_granted = max(0, config.initial_free_units if config.enabled else 0)
    if not config.enabled:
        bucket.status = CommercialQuotaBucketStatus.canceled
    else:
        if bucket.status == CommercialQuotaBucketStatus.canceled:
            bucket.status = CommercialQuotaBucketStatus.scheduled
        if bucket.starts_at > now:
            bucket.starts_at = now
        _refresh_bucket_state(bucket, at_time=now)
    bucket.granted_by_user_id = actor_user_id or bucket.granted_by_user_id
    bucket.updated_at = now
    session.add(bucket)
    session.flush()
    if config.enabled:
        _refresh_bucket_state(bucket, at_time=now)
        session.add(bucket)
        session.flush()
    after_snapshot = get_balance_snapshot(session, workspace_id=workspace_id, product_key=product_key, at_time=now)
    after_bucket_available = _bucket_available_units(bucket, at_time=now)
    delta_units = after_bucket_available - before_bucket_available
    if created and delta_units == 0:
        return bucket
    if delta_units != 0:
        _record_balance_ledger(
            session,
            workspace_id=workspace_id,
            product_key=product_key,
            bucket_id=bucket.id,
            movement_type=CommercialQuotaLedgerMovementType.seed if created else CommercialQuotaLedgerMovementType.overwrite,
            source_kind=CommercialQuotaSourceKind.free,
            delta_units=delta_units,
            balance_before_units=before_snapshot.total_available_units,
            balance_after_units=after_snapshot.total_available_units,
            bucket_balance_before_units=before_bucket_available,
            bucket_balance_after_units=after_bucket_available,
            source_ref=bucket.source_ref,
            actor_user_id=actor_user_id,
            metadata={"override_id": str(config.override_id) if config.override_id else ""},
        )
    if delta_units > 0 and config.fifo_auto_approval_enabled:
        from app.services.commerce_service import process_pending_access_requests_fifo

        process_pending_access_requests_fifo(
            session,
            workspace_id=workspace_id,
            product_key=product_key,
            actor_user=None,
            approval_mode="workspace_quota_replenishment",
        )
    return bucket


def grant_balance_units(
    session: Session,
    *,
    workspace_id: UUID,
    product_key: str,
    source_kind: CommercialQuotaSourceKind,
    units: int,
    bucket_key: str = "",
    source_ref: str = "",
    actor_user_id: UUID | None = None,
    order_id: UUID | None = None,
    payment_id: UUID | None = None,
    starts_at: datetime | None = None,
    ends_at: datetime | None = None,
    overwrite_existing: bool = False,
    reset_consumed_on_overwrite: bool = False,
    movement_type: CommercialQuotaLedgerMovementType | None = None,
    metadata: dict[str, object] | None = None,
    at_time: datetime | None = None,
) -> CommercialBalanceBucketRecord:
    now = at_time or utc_now()
    resolved_units = max(0, units)
    before_snapshot = get_balance_snapshot(session, workspace_id=workspace_id, product_key=product_key, at_time=now)
    resolved_bucket_key = bucket_key.strip() or f"{source_kind.value}:{uuid4().hex}"
    bucket = session.exec(
        select(CommercialBalanceBucketRecord).where(
            CommercialBalanceBucketRecord.workspace_id == workspace_id,
            CommercialBalanceBucketRecord.product_key == product_key,
            CommercialBalanceBucketRecord.bucket_key == resolved_bucket_key,
        )
    ).first()
    before_bucket_available = 0
    if bucket is None:
        bucket = CommercialBalanceBucketRecord(
            workspace_id=workspace_id,
            product_key=product_key,
            bucket_key=resolved_bucket_key,
            source_kind=source_kind,
            status=CommercialQuotaBucketStatus.scheduled,
            units_granted=resolved_units,
            units_consumed=0,
            source_ref=source_ref.strip() or resolved_bucket_key,
            granted_by_user_id=actor_user_id,
            order_id=order_id,
            payment_id=payment_id,
            starts_at=starts_at or now,
            ends_at=ends_at,
            metadata_payload=dict(metadata or {}),
        )
        session.add(bucket)
    else:
        before_bucket_available = _bucket_available_units(bucket, at_time=now)
        if overwrite_existing:
            bucket.units_granted = resolved_units
            if reset_consumed_on_overwrite:
                bucket.units_consumed = 0
        else:
            bucket.units_granted += resolved_units
        bucket.source_kind = source_kind
        bucket.source_ref = source_ref.strip() or bucket.source_ref or resolved_bucket_key
        bucket.granted_by_user_id = actor_user_id or bucket.granted_by_user_id
        bucket.order_id = order_id or bucket.order_id
        bucket.payment_id = payment_id or bucket.payment_id
        if starts_at is not None:
            bucket.starts_at = starts_at
        if ends_at is not None or overwrite_existing:
            bucket.ends_at = ends_at
        if bucket.status == CommercialQuotaBucketStatus.canceled:
            bucket.status = CommercialQuotaBucketStatus.scheduled
        if metadata is not None:
            bucket.metadata_payload = dict(metadata)
        bucket.updated_at = now
        session.add(bucket)
    _refresh_bucket_state(bucket, at_time=now)
    session.add(bucket)
    session.flush()
    after_snapshot = get_balance_snapshot(session, workspace_id=workspace_id, product_key=product_key, at_time=now)
    after_bucket_available = _bucket_available_units(bucket, at_time=now)
    if movement_type is None:
        movement_type = CommercialQuotaLedgerMovementType.overwrite if overwrite_existing else CommercialQuotaLedgerMovementType.credit
    delta_units = resolved_units if bucket.starts_at > now and before_bucket_available == 0 and after_bucket_available == 0 else after_bucket_available - before_bucket_available
    if delta_units != 0 or before_bucket_available != after_bucket_available:
        _record_balance_ledger(
            session,
            workspace_id=workspace_id,
            product_key=product_key,
            bucket_id=bucket.id,
            movement_type=movement_type,
            source_kind=source_kind,
            delta_units=delta_units,
            balance_before_units=before_snapshot.total_available_units,
            balance_after_units=after_snapshot.total_available_units,
            bucket_balance_before_units=before_bucket_available,
            bucket_balance_after_units=after_bucket_available,
            source_ref=bucket.source_ref,
            actor_user_id=actor_user_id,
            order_id=order_id,
            payment_id=payment_id,
            metadata=metadata,
        )
    if after_snapshot.total_available_units > before_snapshot.total_available_units:
        config = resolve_effective_quota_config(session, workspace_id=workspace_id, product_key=product_key, at_time=now)
        if config.fifo_auto_approval_enabled:
            from app.services.commerce_service import process_pending_access_requests_fifo

            process_pending_access_requests_fifo(
                session,
                workspace_id=workspace_id,
                product_key=product_key,
                actor_user=None,
                approval_mode="workspace_quota_replenishment",
            )
    return bucket


def consume_balance_units(
    session: Session,
    *,
    workspace_id: UUID,
    product_key: str,
    units: int = 1,
    actor_user_id: UUID | None = None,
    access_request_id: UUID | None = None,
    source_ref: str = "",
    metadata: dict[str, object] | None = None,
    at_time: datetime | None = None,
) -> list[CommercialBalanceBucketRecord]:
    now = at_time or utc_now()
    if units <= 0:
        return []
    config = resolve_effective_quota_config(session, workspace_id=workspace_id, product_key=product_key, at_time=now)
    snapshot = get_balance_snapshot(session, workspace_id=workspace_id, product_key=product_key, at_time=now)
    if snapshot.total_available_units < units:
        raise ValueError("Insufficient available balance.")
    priority_index = {kind: idx for idx, kind in enumerate(config.consumption_priority)}
    buckets = [
        bucket
        for bucket in list_balance_buckets(session, workspace_id=workspace_id, product_key=product_key, at_time=now)
        if _bucket_available_units(bucket, at_time=now) > 0
    ]
    buckets.sort(
        key=lambda bucket: (
            priority_index.get(bucket.source_kind, len(priority_index)),
            bucket.ends_at or datetime.max.replace(tzinfo=None),
            bucket.created_at,
            bucket.bucket_key,
        )
    )
    remaining = units
    current_total = snapshot.total_available_units
    touched: list[CommercialBalanceBucketRecord] = []
    for bucket in buckets:
        if remaining <= 0:
            break
        available = _bucket_available_units(bucket, at_time=now)
        if available <= 0:
            continue
        take = min(available, remaining)
        before_bucket_available = available
        bucket.units_consumed += take
        bucket.updated_at = now
        _refresh_bucket_state(bucket, at_time=now)
        session.add(bucket)
        current_total_after = current_total - take
        _record_balance_ledger(
            session,
            workspace_id=workspace_id,
            product_key=product_key,
            bucket_id=bucket.id,
            movement_type=CommercialQuotaLedgerMovementType.consume,
            source_kind=bucket.source_kind,
            delta_units=-take,
            balance_before_units=current_total,
            balance_after_units=current_total_after,
            bucket_balance_before_units=before_bucket_available,
            bucket_balance_after_units=max(0, before_bucket_available - take),
            source_ref=source_ref.strip() or bucket.source_ref,
            actor_user_id=actor_user_id,
            access_request_id=access_request_id,
            metadata=metadata,
        )
        touched.append(bucket)
        current_total = current_total_after
        remaining -= take
    if remaining > 0:
        raise ValueError("Unable to consume the requested balance units.")
    return touched
