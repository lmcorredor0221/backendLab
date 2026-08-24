from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from sqlmodel import Session, select

from app.models import (
    CommercialAccessRequestRecord,
    CommercialAccessRequestStatus,
    CommercialOrderLineRecord,
    CommercialOrderRecord,
    CommercialOrderStatus,
    SchemaMigrationRecord,
    WorkspaceRecord,
    utc_now,
)
from app.services.commerce_service import ensure_commercial_seed, process_pending_access_requests_fifo, record_commercial_event
from app.services.commercial_quota_service import list_quota_product_configs, sync_workspace_free_bucket


MIGRATION_KEY_COMMERCIAL_QUOTA_ROLLOUT = "2026-08-23-commercial-quota-rollout"
ROLLOUT_PRODUCT_KEYS: tuple[str, ...] = ("blueprint_pro", "acp")


@dataclass
class CommercialQuotaRolloutSummary:
    migration_key: str = MIGRATION_KEY_COMMERCIAL_QUOTA_ROLLOUT
    already_recorded: bool = False
    workspaces_scanned: int = 0
    workspaces_initialized: int = 0
    product_buckets_synced: int = 0
    pending_requests_before: int = 0
    pending_requests_after: int = 0
    pending_requests_auto_approved: int = 0
    legacy_orders_canceled: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_commercial_quota_rollout(
    session: Session,
    *,
    at_time: datetime | None = None,
) -> CommercialQuotaRolloutSummary:
    summary = CommercialQuotaRolloutSummary()
    migration_record = session.exec(
        select(SchemaMigrationRecord).where(
            SchemaMigrationRecord.migration_key == MIGRATION_KEY_COMMERCIAL_QUOTA_ROLLOUT
        )
    ).first()
    if migration_record is not None:
        summary.already_recorded = True
        return summary

    ensure_commercial_seed(session)
    quota_configs = [config for config in list_quota_product_configs(session) if config.product_key in ROLLOUT_PRODUCT_KEYS]
    active_workspaces = session.exec(
        select(WorkspaceRecord).where(WorkspaceRecord.is_active == True)  # noqa: E712
    ).all()
    summary.workspaces_scanned = len(active_workspaces)
    summary.pending_requests_before = _count_pending_requests(session)

    for workspace in active_workspaces:
        summary.workspaces_initialized += 1
        for config in quota_configs:
            sync_workspace_free_bucket(
                session,
                workspace_id=workspace.id,
                product_key=config.product_key,
                actor_user_id=None,
                at_time=at_time,
            )
            summary.product_buckets_synced += 1
            process_pending_access_requests_fifo(
                session,
                workspace_id=workspace.id,
                product_key=config.product_key,
                actor_user=None,
                approval_mode="commercial_rollout_recalculation",
            )

    summary.legacy_orders_canceled = _cancel_legacy_pending_orders(session, at_time=at_time)
    summary.pending_requests_after = _count_pending_requests(session)
    summary.pending_requests_auto_approved = max(0, summary.pending_requests_before - summary.pending_requests_after)

    session.add(
        SchemaMigrationRecord(
            migration_key=MIGRATION_KEY_COMMERCIAL_QUOTA_ROLLOUT,
            description=(
                "Inicializa saldo comercial por workspace, recalcula solicitudes pendientes y cancela "
                "ordenes legacy pendientes sin snapshot comercial congelado."
            ),
        )
    )
    session.commit()
    return summary


def _count_pending_requests(session: Session) -> int:
    pending = session.exec(
        select(CommercialAccessRequestRecord).where(
            CommercialAccessRequestRecord.status == CommercialAccessRequestStatus.pending,
            CommercialAccessRequestRecord.product_key.in_(ROLLOUT_PRODUCT_KEYS),
        )
    ).all()
    return len(pending)


def _cancel_legacy_pending_orders(
    session: Session,
    *,
    at_time: datetime | None = None,
) -> int:
    now = at_time or utc_now()
    orders = session.exec(
        select(CommercialOrderRecord).where(CommercialOrderRecord.status == CommercialOrderStatus.pending)
    ).all()
    canceled = 0
    for order in orders:
        snapshot = order.metadata_payload.get("commercial_snapshot") if isinstance(order.metadata_payload, dict) else None
        if snapshot:
            continue
        order.status = CommercialOrderStatus.canceled
        order.updated_at = now
        metadata_payload = dict(order.metadata_payload or {})
        metadata_payload["rollout_canceled"] = True
        metadata_payload["rollout_migration_key"] = MIGRATION_KEY_COMMERCIAL_QUOTA_ROLLOUT
        order.metadata_payload = metadata_payload
        session.add(order)
        record_commercial_event(
            session,
            workspace_id=order.workspace_id,
            session_id=order.session_id,
            user_id=None,
            event_key="commercial_rollout_legacy_order_canceled",
            product_key=_resolve_order_product_key(session, order),
            source="commercial_rollout",
            metadata={"order_id": str(order.id), "reason": "missing_commercial_snapshot"},
            correlation_id=f"rollout:{order.id}",
        )
        canceled += 1
    return canceled


def _resolve_order_product_key(session: Session, order: CommercialOrderRecord) -> str:
    product_key = ""
    if isinstance(order.metadata_payload, dict):
        product_key = str(order.metadata_payload.get("product_key") or "").strip()
    if product_key:
        return product_key
    line = session.exec(
        select(CommercialOrderLineRecord).where(CommercialOrderLineRecord.order_id == order.id)
    ).first()
    return line.product_key if line is not None else ""
