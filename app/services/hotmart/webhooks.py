from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any
from uuid import UUID

from sqlmodel import Session, select

from app.core.config import get_settings
from app.models import (
    CommercialEntitlementRecord,
    CommercialEntitlementSource,
    CommercialEntitlementStatus,
    CommercialOrderLineRecord,
    CommercialOrderRecord,
    CommercialOrderStatus,
    CommercialPaymentRecord,
    CommercialPaymentStatus,
    HotmartIntegrationConfigRecord,
    HotmartPaymentLinkRecord,
    HotmartWebhookEventRecord,
    HotmartWebhookIngestResponse,
    HotmartPendingActivationRecord,
    SessionRecord,
    utc_now,
)
from app.services.commerce_service import (
    apply_package_credits_from_paid_order,
    get_product,
    record_commercial_event,
    settle_open_debts_from_paid_order,
    tier_rank,
)
from app.services.hotmart.auth import normalize_hotmart_environment
from app.services.hotmart.pending_activations import register_pending_hotmart_activation
from app.services.hotmart.redaction import redact_headers, redact_payload
from app.services.hotmart.secrets import load_hotmart_hottok


APPROVAL_EVENTS = {"PURCHASE_APPROVED", "PURCHASE_COMPLETE", "PURCHASE_COMPLETED"}
REFUND_EVENTS = {"PURCHASE_REFUNDED", "PURCHASE_REFUND_REQUEST"}
CHARGEBACK_EVENTS = {"PURCHASE_CHARGEBACK"}
CANCEL_EVENTS = {"PURCHASE_CANCELED", "PURCHASE_CANCELLED", "PURCHASE_EXPIRED"}
DELAY_EVENTS = {"PURCHASE_DELAYED", "PURCHASE_BILLET_PRINTED", "PURCHASE_WAITING_PAYMENT"}


def process_hotmart_webhook(
    session: Session,
    *,
    payload: dict[str, Any],
    hottok_header: str,
    request_headers: dict[str, str] | None = None,
    environment: str = "sandbox",
) -> HotmartWebhookIngestResponse:
    env = normalize_hotmart_environment(environment)
    event_type = _extract_event_type(payload)
    transaction = _extract_transaction(payload)
    payload_hash = _payload_hash(payload)
    event_id = _extract_event_id(payload, event_type=event_type, transaction=transaction, payload_hash=payload_hash)

    existing_event = session.exec(
        select(HotmartWebhookEventRecord).where(
            HotmartWebhookEventRecord.event_id == event_id,
            HotmartWebhookEventRecord.event_type == event_type,
        )
    ).first()
    if existing_event is not None:
        pending_activation = _pending_activation_for_event(session, existing_event.id)
        existing_event.retries += 1
        if existing_event.payload_hash != payload_hash:
            existing_event.processing_status = "observed"
            existing_event.error_code = "payload_conflict"
            existing_event.error_message = "Same Hotmart event id/type arrived with a different payload hash."
            existing_event.processed_at = utc_now()
            session.add(existing_event)
            from app.services.hotmart.sync import _open_or_update_issue

            if existing_event.workspace_id is not None:
                _open_or_update_issue(
                    session,
                    workspace_id=existing_event.workspace_id,
                    environment=env,
                    issue_type="webhook_payload_conflict",
                    provider_ref=existing_event.transaction or existing_event.event_id,
                    internal_ref=str(existing_event.order_id or ""),
                    severity="medium",
                    summary=f"Webhook {existing_event.event_id} arrived with a conflicting payload hash.",
                    suggested_action="Review both payloads and resolve manually from the reconciliation console.",
                    metadata={
                        "event_id": existing_event.event_id,
                        "event_type": existing_event.event_type,
                        "stored_payload_hash": existing_event.payload_hash,
                        "incoming_payload_hash": payload_hash,
                        "retries": existing_event.retries,
                    },
                )
            session.flush()
            return HotmartWebhookIngestResponse(
                event_id=event_id,
                event_type=existing_event.event_type,
                transaction=existing_event.transaction,
                processing_status="observed",
                duplicate=True,
                workspace_id=existing_event.workspace_id,
                order_id=existing_event.order_id,
                payment_id=existing_event.payment_id,
                pending_activation_id=pending_activation.id if pending_activation is not None else None,
                message="Hotmart webhook conflict observed and sent to reconciliation.",
            )
        session.add(existing_event)
        session.flush()
        return HotmartWebhookIngestResponse(
            event_id=event_id,
            event_type=existing_event.event_type,
            transaction=existing_event.transaction,
            processing_status=existing_event.processing_status,
            duplicate=True,
            workspace_id=existing_event.workspace_id,
            order_id=existing_event.order_id,
            payment_id=existing_event.payment_id,
            pending_activation_id=pending_activation.id if pending_activation is not None else None,
            message="Duplicate Hotmart webhook ignored.",
        )

    order = _resolve_order(session, payload=payload, transaction=transaction)
    workspace_id = order.workspace_id if order is not None else _fallback_workspace_id(session, environment=env)
    hottok_validated = _validate_hottok(
        session,
        workspace_id=workspace_id,
        environment=env,
        hottok_header=hottok_header,
    )
    payload_redacted = redact_payload(payload)
    diagnostics = _request_diagnostics(
        request_headers=request_headers,
        hottok_header=hottok_header,
    )
    if diagnostics:
        payload_redacted["_lab_request_diagnostics"] = diagnostics
    webhook_event = HotmartWebhookEventRecord(
        event_id=event_id,
        event_type=event_type,
        transaction=transaction,
        workspace_id=workspace_id,
        order_id=order.id if order is not None else None,
        hottok_validated=hottok_validated,
        processing_status="received",
        payload_hash=payload_hash,
        payload_redacted=payload_redacted,
    )
    session.add(webhook_event)
    session.flush()

    if not hottok_validated:
        webhook_event.processing_status = "rejected"
        webhook_event.error_code = "invalid_hottok"
        webhook_event.error_message = "Invalid or missing X-HOTMART-HOTTOK."
        webhook_event.processed_at = utc_now()
        session.add(webhook_event)
        session.flush()
        raise PermissionError("Invalid Hotmart webhook token.")

    if order is None and workspace_id is not None and event_type in APPROVAL_EVENTS:
        pending_activation = register_pending_hotmart_activation(
            session,
            payload=payload,
            webhook_event_id=webhook_event.id,
            event_id=event_id,
            source_workspace_id=workspace_id,
            environment=env,
            transaction=transaction or event_id,
        )
        webhook_event.processing_status = "pending_activation"
        webhook_event.processed_at = utc_now()
        session.add(webhook_event)
        session.flush()
        return HotmartWebhookIngestResponse(
            event_id=event_id,
            event_type=event_type,
            transaction=transaction,
            processing_status="pending_activation",
            workspace_id=workspace_id,
            pending_activation_id=pending_activation.id,
            message="Hotmart approved purchase recorded pending activation.",
        )

    if order is None or workspace_id is None:
        webhook_event.processing_status = "unresolved"
        webhook_event.error_code = "order_not_found"
        webhook_event.error_message = "Could not resolve internal order from Hotmart webhook."
        webhook_event.processed_at = utc_now()
        session.add(webhook_event)
        session.flush()
        return HotmartWebhookIngestResponse(
            event_id=event_id,
            event_type=event_type,
            transaction=transaction,
            processing_status="unresolved",
            workspace_id=workspace_id,
            message=webhook_event.error_message,
        )

    if event_type in APPROVAL_EVENTS:
        response = _process_approved_purchase(
            session,
            order=order,
            webhook_event=webhook_event,
            payload=payload,
            transaction=transaction or event_id,
        )
    elif event_type in REFUND_EVENTS | CHARGEBACK_EVENTS | CANCEL_EVENTS:
        response = _process_revocation_event(
            session,
            order=order,
            webhook_event=webhook_event,
            payload=payload,
            transaction=transaction or event_id,
            event_type=event_type,
        )
    elif event_type in DELAY_EVENTS:
        webhook_event.processing_status = "processed"
        webhook_event.processed_at = utc_now()
        session.add(webhook_event)
        response = HotmartWebhookIngestResponse(
            event_id=event_id,
            event_type=event_type,
            transaction=transaction,
            processing_status="processed",
            workspace_id=workspace_id,
            order_id=order.id,
            message="Hotmart delayed/waiting payment event recorded.",
        )
    else:
        webhook_event.processing_status = "ignored"
        webhook_event.processed_at = utc_now()
        session.add(webhook_event)
        response = HotmartWebhookIngestResponse(
            event_id=event_id,
            event_type=event_type,
            transaction=transaction,
            processing_status="ignored",
            workspace_id=workspace_id,
            order_id=order.id,
            message=f"Hotmart event {event_type or 'unknown'} does not change commerce state.",
        )
    session.flush()
    return response


def _process_approved_purchase(
    session: Session,
    *,
    order: CommercialOrderRecord,
    webhook_event: HotmartWebhookEventRecord,
    payload: dict[str, Any],
    transaction: str,
) -> HotmartWebhookIngestResponse:
    line = _order_line(session, order)
    if line is None:
        raise ValueError("Hotmart order has no order line.")
    product = get_product(session, line.product_key)
    payment = session.exec(
        select(CommercialPaymentRecord).where(
            CommercialPaymentRecord.provider == "hotmart",
            CommercialPaymentRecord.provider_payment_id == transaction,
        )
    ).first()
    if payment is None:
        payment = CommercialPaymentRecord(
            workspace_id=order.workspace_id,
            session_id=order.session_id,
            order_id=order.id,
            provider="hotmart",
            provider_payment_id=transaction,
            provider_checkout_ref=order.checkout_ref,
            status=CommercialPaymentStatus.succeeded,
            amount_cents=_extract_amount_cents(payload, fallback_cents=order.total_cents),
            currency=_extract_currency(payload, fallback=order.currency),
            idempotency_key=f"hotmart:{transaction}:approved",
            metadata_payload={"event_id": webhook_event.event_id},
        )
    payment.status = CommercialPaymentStatus.succeeded
    payment.updated_at = utc_now()
    session.add(payment)
    session.flush()
    settle_open_debts_from_paid_order(
        session,
        order=order,
        payment=payment,
        actor_user_id=order.buyer_user_id,
    )
    apply_package_credits_from_paid_order(
        session,
        order=order,
        payment=payment,
        actor_user_id=order.buyer_user_id,
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
            metadata_payload={"provider": "hotmart", "event_id": webhook_event.event_id},
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
    webhook_event.payment_id = payment.id
    webhook_event.processing_status = "processed"
    webhook_event.processed_at = utc_now()
    session.add(webhook_event)
    record_commercial_event(
        session,
        workspace_id=order.workspace_id,
        session_id=order.session_id,
        user_id=order.buyer_user_id,
        event_key="hotmart_payment_approved",
        product_key=product.product_key,
        source="hotmart_webhook",
        revenue_cents=payment.amount_cents,
        currency=payment.currency,
        metadata={"order_id": str(order.id), "payment_id": str(payment.id), "transaction": transaction},
        correlation_id=webhook_event.event_id,
    )
    from app.services.product_processing.product_build_activation_service import activate_product_builds_for_paid_order

    activate_product_builds_for_paid_order(
        session,
        order=order,
        current_user=None,
        source="hotmart_webhook",
    )
    return HotmartWebhookIngestResponse(
        event_id=webhook_event.event_id,
        event_type=webhook_event.event_type,
        transaction=transaction,
        processing_status="processed",
        workspace_id=order.workspace_id,
        order_id=order.id,
        payment_id=payment.id,
        entitlement_id=entitlement.id,
        message="Hotmart approved purchase processed.",
    )


def _process_revocation_event(
    session: Session,
    *,
    order: CommercialOrderRecord,
    webhook_event: HotmartWebhookEventRecord,
    payload: dict[str, Any],
    transaction: str,
    event_type: str,
) -> HotmartWebhookIngestResponse:
    payment_status = CommercialPaymentStatus.refunded
    order_status = CommercialOrderStatus.refunded
    entitlement_status = CommercialEntitlementStatus.refunded
    event_key = "hotmart_payment_refunded"
    if event_type in CHARGEBACK_EVENTS:
        entitlement_status = CommercialEntitlementStatus.revoked
        event_key = "hotmart_chargeback_received"
    elif event_type in CANCEL_EVENTS:
        payment_status = CommercialPaymentStatus.canceled
        order_status = CommercialOrderStatus.canceled
        entitlement_status = CommercialEntitlementStatus.revoked
        event_key = "hotmart_payment_canceled"

    payment = session.exec(
        select(CommercialPaymentRecord).where(
            CommercialPaymentRecord.provider == "hotmart",
            CommercialPaymentRecord.provider_payment_id == transaction,
        )
    ).first()
    if payment is None:
        payment = session.exec(
            select(CommercialPaymentRecord).where(CommercialPaymentRecord.order_id == order.id)
        ).first()
    if payment is None:
        payment = CommercialPaymentRecord(
            workspace_id=order.workspace_id,
            session_id=order.session_id,
            order_id=order.id,
            provider="hotmart",
            provider_payment_id=transaction,
            provider_checkout_ref=order.checkout_ref,
            amount_cents=_extract_amount_cents(payload, fallback_cents=order.total_cents),
            currency=_extract_currency(payload, fallback=order.currency),
            idempotency_key=f"hotmart:{transaction}:revocation",
        )
    payment.status = payment_status
    payment.updated_at = utc_now()
    session.add(payment)
    session.flush()

    entitlements = session.exec(
        select(CommercialEntitlementRecord).where(CommercialEntitlementRecord.order_id == order.id)
    ).all()
    entitlement_id: UUID | None = None
    for entitlement in entitlements:
        entitlement.status = entitlement_status
        entitlement.payment_id = payment.id
        entitlement.updated_at = utc_now()
        entitlement_id = entitlement.id
        session.add(entitlement)

    order.status = order_status
    order.updated_at = utc_now()
    session.add(order)
    _recompute_session_tier(session, order.session_id)
    webhook_event.payment_id = payment.id
    webhook_event.processing_status = "processed"
    webhook_event.processed_at = utc_now()
    session.add(webhook_event)
    product_key = _order_product_key(session, order)
    record_commercial_event(
        session,
        workspace_id=order.workspace_id,
        session_id=order.session_id,
        user_id=order.buyer_user_id,
        event_key=event_key,
        product_key=product_key,
        source="hotmart_webhook",
        revenue_cents=0,
        currency=payment.currency,
        metadata={"order_id": str(order.id), "payment_id": str(payment.id), "transaction": transaction},
        correlation_id=webhook_event.event_id,
    )
    return HotmartWebhookIngestResponse(
        event_id=webhook_event.event_id,
        event_type=webhook_event.event_type,
        transaction=transaction,
        processing_status="processed",
        workspace_id=order.workspace_id,
        order_id=order.id,
        payment_id=payment.id,
        entitlement_id=entitlement_id,
        message=f"Hotmart {event_type} processed.",
    )


def _validate_hottok(
    session: Session,
    *,
    workspace_id: UUID | None,
    environment: str,
    hottok_header: str,
) -> bool:
    provided = (hottok_header or "").strip()
    if not provided:
        return False
    expected = ""
    if workspace_id is not None:
        expected = load_hotmart_hottok(session, workspace_id=workspace_id, environment=environment)
    if not expected:
        settings = get_settings()
        if normalize_hotmart_environment(settings.hotmart_environment) == environment:
            expected = settings.hotmart_hottok.strip()
    return bool(expected and hmac.compare_digest(provided, expected))


def _request_diagnostics(
    *,
    request_headers: dict[str, str] | None,
    hottok_header: str,
) -> dict[str, Any]:
    if request_headers is None:
        return {}
    normalized_headers = {str(key): str(value) for key, value in request_headers.items()}
    provided = (hottok_header or "").strip()
    return {
        "headers_redacted": redact_headers(normalized_headers),
        "header_names": sorted(normalized_headers.keys(), key=str.lower),
        "hottok_header_present": bool(provided),
        "hottok_header_length": len(provided),
        "hottok_header_sha256_prefix": hashlib.sha256(provided.encode("utf-8")).hexdigest()[:12]
        if provided
        else "",
    }


def _resolve_order(
    session: Session,
    *,
    payload: dict[str, Any],
    transaction: str,
) -> CommercialOrderRecord | None:
    order_id = _first_string(
        payload,
        ("order_id",),
        ("data", "order_id"),
        ("data", "purchase", "order_id"),
        ("data", "purchase", "metadata", "order_id"),
        ("data", "metadata", "order_id"),
    )
    if order_id:
        try:
            order = session.get(CommercialOrderRecord, UUID(order_id))
            if order is not None:
                return order
        except ValueError:
            pass
    checkout_ref = _first_string(
        payload,
        ("checkout_ref",),
        ("data", "checkout_ref"),
        ("data", "purchase", "checkout_ref"),
        ("data", "purchase", "metadata", "checkout_ref"),
        ("data", "purchase", "sck"),
        ("data", "sck"),
    )
    if checkout_ref:
        order = session.exec(select(CommercialOrderRecord).where(CommercialOrderRecord.checkout_ref == checkout_ref)).first()
        if order is not None:
            return order
    provider_ref = _first_string(
        payload,
        ("data", "payment_link", "ucode"),
        ("data", "purchase", "payment_link_ucode"),
        ("data", "purchase", "payment_link", "ucode"),
    )
    if provider_ref:
        link = session.exec(
            select(HotmartPaymentLinkRecord).where(
                (HotmartPaymentLinkRecord.provider_ref == provider_ref)
                | (HotmartPaymentLinkRecord.hotmart_payment_link_id == provider_ref)
            )
        ).first()
        if link is not None:
            return session.get(CommercialOrderRecord, link.order_id)
    if transaction:
        payment = session.exec(
            select(CommercialPaymentRecord).where(
                CommercialPaymentRecord.provider == "hotmart",
                CommercialPaymentRecord.provider_payment_id == transaction,
            )
        ).first()
        if payment is not None:
            return session.get(CommercialOrderRecord, payment.order_id)
    return None


def _fallback_workspace_id(session: Session, *, environment: str) -> UUID | None:
    config = session.exec(
        select(HotmartIntegrationConfigRecord).where(HotmartIntegrationConfigRecord.environment == environment)
    ).first()
    return config.workspace_id if config is not None else None


def _pending_activation_for_event(
    session: Session,
    webhook_event_id: UUID,
) -> HotmartPendingActivationRecord | None:
    return session.exec(
        select(HotmartPendingActivationRecord).where(HotmartPendingActivationRecord.webhook_event_id == webhook_event_id)
    ).first()


def _payload_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _extract_event_type(payload: dict[str, Any]) -> str:
    raw = payload.get("event") or payload.get("event_type") or _first_string(payload, ("data", "event"))
    if isinstance(raw, dict):
        raw = raw.get("type") or raw.get("name")
    return str(raw or "").strip().upper()


def _extract_transaction(payload: dict[str, Any]) -> str:
    return _first_string(
        payload,
        ("transaction",),
        ("data", "transaction"),
        ("data", "purchase", "transaction"),
        ("data", "purchase", "transaction_code"),
        ("data", "purchase", "code"),
    )


def _extract_event_id(payload: dict[str, Any], *, event_type: str, transaction: str, payload_hash: str) -> str:
    explicit = _first_string(payload, ("id",), ("event_id",), ("data", "id"), ("data", "event_id"))
    if explicit:
        return explicit[:160]
    if event_type and transaction:
        return f"{event_type}:{transaction}"[:160]
    return f"payload:{payload_hash}"[:160]


def _first_string(payload: Any, *paths: tuple[str, ...]) -> str:
    for path in paths:
        value = payload
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _extract_amount_cents(payload: dict[str, Any], *, fallback_cents: int) -> int:
    amount_cents = _first_string(
        payload,
        ("amount_cents",),
        ("data", "amount_cents"),
        ("data", "purchase", "amount_cents"),
    )
    if amount_cents:
        try:
            return max(0, int(float(amount_cents)))
        except ValueError:
            pass
    amount_value = _first_string(
        payload,
        ("data", "purchase", "price", "value"),
        ("data", "purchase", "full_price", "value"),
        ("data", "purchase", "approved_transaction", "amount"),
        ("data", "price", "value"),
    )
    if amount_value:
        try:
            return max(0, int(round(float(amount_value) * 100)))
        except ValueError:
            pass
    return fallback_cents


def _extract_currency(payload: dict[str, Any], *, fallback: str) -> str:
    return (
        _first_string(
            payload,
            ("currency",),
            ("data", "currency"),
            ("data", "purchase", "currency"),
            ("data", "purchase", "price", "currency_code"),
            ("data", "purchase", "full_price", "currency_code"),
        )
        or fallback
    ).upper()


def _order_line(session: Session, order: CommercialOrderRecord) -> CommercialOrderLineRecord | None:
    return session.exec(select(CommercialOrderLineRecord).where(CommercialOrderLineRecord.order_id == order.id)).first()


def _order_product_key(session: Session, order: CommercialOrderRecord) -> str:
    line = _order_line(session, order)
    if line is not None and line.product_key:
        return line.product_key
    return str(order.metadata_payload.get("product_key") or "")


def _recompute_session_tier(session: Session, session_id: UUID | None) -> None:
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
