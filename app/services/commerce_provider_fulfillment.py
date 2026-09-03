from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from sqlmodel import Session, select

from app.models import (
    CommercialEntitlementRecord,
    CommercialEntitlementSource,
    CommercialEntitlementStatus,
    CommercialOrderLineRecord,
    CommercialOrderRecord,
    CommercialOrderStatus,
    CommercialPaymentRecord,
    CommercialPaymentStatus,
    SessionRecord,
    utc_now,
)


@dataclass(frozen=True)
class ProviderFulfillmentResult:
    payment: CommercialPaymentRecord
    entitlement: CommercialEntitlementRecord | None = None


@dataclass(frozen=True)
class ProviderPaymentEvent:
    provider_key: str
    provider_payment_id: str
    event_id: str
    event_type: str
    amount_cents: int
    currency: str
    metadata: dict[str, Any] = field(default_factory=dict)


def apply_provider_payment_success(
    session: Session,
    *,
    order: CommercialOrderRecord,
    event: ProviderPaymentEvent,
    actor_user_id: UUID | None = None,
    event_key: str = "payment_confirmed",
    source: str = "commerce_provider_webhook",
) -> ProviderFulfillmentResult:
    from app.services.commerce_service import (
        apply_package_credits_from_paid_order,
        get_product,
        record_commercial_event,
        settle_open_debts_from_paid_order,
        tier_rank,
    )

    line = _order_line(session, order)
    if line is None:
        raise ValueError("Commerce order has no order line.")
    product = get_product(session, line.product_key)
    provider_payment_id = event.provider_payment_id or f"{event.provider_key}:{event.event_id}"
    payment = session.exec(
        select(CommercialPaymentRecord).where(
            CommercialPaymentRecord.provider == event.provider_key,
            CommercialPaymentRecord.provider_payment_id == provider_payment_id,
        )
    ).first()
    if payment is None:
        payment = CommercialPaymentRecord(
            workspace_id=order.workspace_id,
            session_id=order.session_id,
            order_id=order.id,
            provider=event.provider_key,
            provider_payment_id=provider_payment_id,
            provider_checkout_ref=order.checkout_ref,
            idempotency_key=f"{event.provider_key}:{provider_payment_id}:success",
        )
    payment.status = CommercialPaymentStatus.succeeded
    payment.amount_cents = max(0, event.amount_cents)
    payment.currency = (event.currency or order.currency or "USD").upper()
    payment.metadata_payload = {
        **dict(payment.metadata_payload or {}),
        "provider": event.provider_key,
        "event_id": event.event_id,
        "event_type": event.event_type,
        **event.metadata,
    }
    payment.updated_at = utc_now()
    session.add(payment)
    session.flush()

    settle_open_debts_from_paid_order(
        session,
        order=order,
        payment=payment,
        actor_user_id=actor_user_id,
    )
    apply_package_credits_from_paid_order(
        session,
        order=order,
        payment=payment,
        actor_user_id=actor_user_id,
    )

    entitlement = session.exec(
        select(CommercialEntitlementRecord).where(CommercialEntitlementRecord.order_id == order.id)
    ).first()
    if entitlement is None:
        entitlement = CommercialEntitlementRecord(
            workspace_id=order.workspace_id,
            session_id=order.session_id,
            product_key=product.product_key,
            tier=product.tier,
            status=CommercialEntitlementStatus.active,
            source=CommercialEntitlementSource.checkout,
            order_id=order.id,
            order_line_id=line.id,
            payment_id=payment.id,
            granted_by_user_id=actor_user_id,
            metadata_payload={"non_revenue": False, "provider": event.provider_key, "event_id": event.event_id},
        )
    entitlement.status = CommercialEntitlementStatus.active
    entitlement.payment_id = payment.id
    entitlement.updated_at = utc_now()
    session.add(entitlement)

    if order.session_id is not None:
        session_record = session.get(SessionRecord, order.session_id)
        if session_record is not None and tier_rank(product.tier) > tier_rank(session_record.commercial_tier):
            session_record.commercial_tier = product.tier
            session_record.updated_at = utc_now()
            session.add(session_record)

    order.status = CommercialOrderStatus.paid
    order.paid_at = order.paid_at or utc_now()
    order.updated_at = utc_now()
    session.add(order)
    record_commercial_event(
        session,
        workspace_id=order.workspace_id,
        session_id=order.session_id,
        user_id=actor_user_id or order.buyer_user_id,
        event_key=event_key,
        product_key=product.product_key,
        source=source,
        revenue_cents=payment.amount_cents,
        currency=payment.currency,
        metadata={"order_id": str(order.id), "payment_id": str(payment.id), **event.metadata},
        correlation_id=event.event_id or order.checkout_ref,
    )
    from app.services.product_processing.product_build_activation_service import activate_product_builds_for_paid_order

    activate_product_builds_for_paid_order(
        session,
        order=order,
        current_user=None,
        source=source,
    )
    session.flush()
    return ProviderFulfillmentResult(payment=payment, entitlement=entitlement)


def apply_provider_payment_revocation(
    session: Session,
    *,
    order: CommercialOrderRecord,
    event: ProviderPaymentEvent,
    payment_status: CommercialPaymentStatus = CommercialPaymentStatus.refunded,
    order_status: CommercialOrderStatus = CommercialOrderStatus.refunded,
    entitlement_status: CommercialEntitlementStatus = CommercialEntitlementStatus.refunded,
    actor_user_id: UUID | None = None,
    event_key: str = "payment_refunded",
    source: str = "commerce_provider_webhook",
) -> ProviderFulfillmentResult:
    from app.services.commerce_service import record_commercial_event, tier_rank

    provider_payment_id = event.provider_payment_id or f"{event.provider_key}:{event.event_id}"
    payment = session.exec(
        select(CommercialPaymentRecord).where(
            CommercialPaymentRecord.provider == event.provider_key,
            CommercialPaymentRecord.provider_payment_id == provider_payment_id,
        )
    ).first()
    if payment is None:
        payment = session.exec(select(CommercialPaymentRecord).where(CommercialPaymentRecord.order_id == order.id)).first()
    if payment is None:
        payment = CommercialPaymentRecord(
            workspace_id=order.workspace_id,
            session_id=order.session_id,
            order_id=order.id,
            provider=event.provider_key,
            provider_payment_id=provider_payment_id,
            provider_checkout_ref=order.checkout_ref,
            amount_cents=max(0, event.amount_cents),
            currency=(event.currency or order.currency or "USD").upper(),
            idempotency_key=f"{event.provider_key}:{provider_payment_id}:revocation",
        )
    payment.status = payment_status
    payment.metadata_payload = {
        **dict(payment.metadata_payload or {}),
        "provider": event.provider_key,
        "event_id": event.event_id,
        "event_type": event.event_type,
        **event.metadata,
    }
    payment.updated_at = utc_now()
    session.add(payment)
    session.flush()

    entitlement: CommercialEntitlementRecord | None = None
    entitlements = session.exec(
        select(CommercialEntitlementRecord).where(CommercialEntitlementRecord.order_id == order.id)
    ).all()
    for entitlement_record in entitlements:
        entitlement_record.status = entitlement_status
        entitlement_record.payment_id = payment.id
        entitlement_record.updated_at = utc_now()
        entitlement = entitlement_record
        session.add(entitlement_record)

    order.status = order_status
    order.updated_at = utc_now()
    session.add(order)
    _recompute_session_tier(session, order.session_id, tier_rank=tier_rank)
    record_commercial_event(
        session,
        workspace_id=order.workspace_id,
        session_id=order.session_id,
        user_id=actor_user_id or order.buyer_user_id,
        event_key=event_key,
        product_key=_order_product_key(session, order),
        source=source,
        revenue_cents=0,
        currency=payment.currency,
        metadata={"order_id": str(order.id), "payment_id": str(payment.id), **event.metadata},
        correlation_id=event.event_id or order.checkout_ref,
    )
    session.flush()
    return ProviderFulfillmentResult(payment=payment, entitlement=entitlement)


def _order_line(session: Session, order: CommercialOrderRecord) -> CommercialOrderLineRecord | None:
    return session.exec(select(CommercialOrderLineRecord).where(CommercialOrderLineRecord.order_id == order.id)).first()


def _order_product_key(session: Session, order: CommercialOrderRecord) -> str:
    line = _order_line(session, order)
    if line is not None and line.product_key:
        return line.product_key
    return str(order.metadata_payload.get("product_key") or "")


def _recompute_session_tier(session: Session, session_id: UUID | None, *, tier_rank) -> None:
    if session_id is None:
        return
    session_record = session.get(SessionRecord, session_id)
    if session_record is None:
        return
    active_entitlements = session.exec(
        select(CommercialEntitlementRecord).where(
            CommercialEntitlementRecord.session_id == session_id,
            CommercialEntitlementRecord.status == CommercialEntitlementStatus.active,
        )
    ).all()
    effective_tier = session_record.commercial_tier.__class__.blueprint
    for entitlement in active_entitlements:
        if tier_rank(entitlement.tier) > tier_rank(effective_tier):
            effective_tier = entitlement.tier
    session_record.commercial_tier = effective_tier
    session_record.updated_at = utc_now()
    session.add(session_record)
