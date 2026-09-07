from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import hmac
import json
from typing import Any
from uuid import UUID

from sqlmodel import Session, select

from app.core.config import get_settings
from app.models import (
    CommerceProviderCheckoutRecord,
    CommerceProviderConfigRecord,
    CommerceProviderWebhookEventRecord,
    CommerceProviderWebhookIngestResponse,
    CommercialEntitlementStatus,
    CommercialOrderRecord,
    CommercialOrderStatus,
    CommercialPaymentRecord,
    CommercialPaymentStatus,
    utc_now,
)
from app.services.commerce_provider_fulfillment import (
    ProviderPaymentEvent,
    apply_provider_payment_revocation,
    apply_provider_payment_success,
)
from app.services.commerce_provider_redaction import redact_headers, redact_payload
from app.services.commerce_provider_secrets import load_commerce_provider_secret
from app.services.commerce_provider_utils import normalize_commerce_provider_environment
from app.services.rapyd.signatures import verify_rapyd_webhook_signature


SUCCESS_EVENTS = {"PAYMENT_SUCCEEDED", "PAYMENT_COMPLETED"}
FAILED_EVENTS = {"PAYMENT_FAILED", "PAYMENT_EXPIRED", "PAYMENT_CANCELED", "PAYMENT_CANCELLED"}
REVOCATION_EVENTS = {"PAYMENT_REFUNDED", "PAYMENT_REVERSED", "PAYMENT_CHARGEBACK_CREATED"}
SUCCESS_STATUSES = {"CLO", "SUCCESS", "SUCCEEDED", "COMPLETED", "PAID"}
PENDING_STATUSES = {"ACT", "NEW", "PENDING", "PENDING_OFFLINE_CAPTURE"}
FAILED_STATUSES = {"CAN", "ERR", "EXP", "FAILED", "CANCELED", "CANCELLED", "DECLINED", "EXPIRED"}
REVOCATION_STATUSES = {"REV", "REF", "REFUNDED", "REVERSED", "CHARGEBACK"}


def process_rapyd_webhook(
    session: Session,
    *,
    raw_body: bytes,
    request_headers: dict[str, str],
    url_secret: str = "",
    environment: str = "sandbox",
) -> CommerceProviderWebhookIngestResponse:
    env = normalize_commerce_provider_environment(environment)
    payload = _decode_payload(raw_body)
    event_type = _extract_event_type(payload)
    data = _extract_data(payload)
    metadata = _extract_metadata(data)
    payload_hash = hashlib.sha256(raw_body).hexdigest()
    provider_resource_id = _extract_provider_resource_id(payload, data)
    event_id = _extract_event_id(
        payload,
        event_type=event_type,
        provider_resource_id=provider_resource_id,
        payload_hash=payload_hash,
    )
    order = _resolve_order(session, data=data, metadata=metadata, provider_resource_id=provider_resource_id)
    workspace_id = order.workspace_id if order is not None else _workspace_id_from_metadata_or_config(session, metadata, environment=env)
    webhook_event = _find_existing_event(session, event_id=event_id, event_type=event_type)
    signature_validated = _validate_rapyd_signature(
        session,
        workspace_id=workspace_id,
        environment=env,
        raw_body=raw_body,
        request_headers=request_headers,
    )
    url_secret_validated = _validate_url_secret(
        session,
        workspace_id=workspace_id,
        environment=env,
        provided=url_secret,
    )

    if webhook_event is not None:
        webhook_event.retries += 1
        webhook_event.signature_validated = webhook_event.signature_validated or (signature_validated and url_secret_validated)
        webhook_event.payload_redacted = {
            **dict(webhook_event.payload_redacted or {}),
            "_lab_last_duplicate_payload_hash": payload_hash,
            "_lab_last_duplicate_headers_redacted": redact_headers(request_headers),
        }
        session.add(webhook_event)
        session.flush()
        if not signature_validated or not url_secret_validated:
            webhook_event.processing_status = "rejected"
            webhook_event.error_code = "invalid_signature"
            webhook_event.error_message = "Invalid Rapyd webhook signature or URL secret."
            webhook_event.processed_at = utc_now()
            session.add(webhook_event)
            session.flush()
            raise PermissionError("Invalid Rapyd webhook signature or URL secret.")
        return CommerceProviderWebhookIngestResponse(
            provider_key="rapyd",
            event_id=webhook_event.event_id,
            event_type=webhook_event.event_type,
            provider_resource_id=webhook_event.provider_resource_id,
            processing_status=webhook_event.processing_status,
            duplicate=True,
            workspace_id=webhook_event.workspace_id,
            order_id=webhook_event.order_id,
            payment_id=webhook_event.payment_id,
            message="Duplicate Rapyd webhook ignored.",
        )

    webhook_event = CommerceProviderWebhookEventRecord(
        provider_key="rapyd",
        environment=env,
        event_id=event_id,
        event_type=event_type,
        provider_resource_id=provider_resource_id,
        workspace_id=workspace_id,
        order_id=order.id if order is not None else None,
        signature_validated=signature_validated and url_secret_validated,
        processing_status="received",
        payload_hash=payload_hash,
        payload_redacted={
            **redact_payload(payload),
            "_lab_request_headers_redacted": redact_headers(request_headers),
        },
    )
    session.add(webhook_event)
    session.flush()

    if not signature_validated or not url_secret_validated:
        webhook_event.processing_status = "rejected"
        webhook_event.error_code = "invalid_signature"
        webhook_event.error_message = "Invalid Rapyd webhook signature or URL secret."
        webhook_event.processed_at = utc_now()
        session.add(webhook_event)
        session.flush()
        raise PermissionError("Invalid Rapyd webhook signature or URL secret.")

    if order is None or workspace_id is None:
        webhook_event.processing_status = "unresolved"
        webhook_event.error_code = "order_not_found"
        webhook_event.error_message = "Could not resolve internal order from Rapyd webhook."
        webhook_event.processed_at = utc_now()
        session.add(webhook_event)
        session.flush()
        return _response_from_event(webhook_event, message=webhook_event.error_message)

    payment_status = _extract_payment_status(data)
    if event_type in SUCCESS_EVENTS or payment_status in SUCCESS_STATUSES:
        result = apply_provider_payment_success(
            session,
            order=order,
            event=ProviderPaymentEvent(
                provider_key="rapyd",
                provider_payment_id=provider_resource_id or event_id,
                event_id=event_id,
                event_type=event_type,
                amount_cents=_extract_amount_cents(data, fallback_cents=order.total_cents),
                currency=_extract_currency(data, fallback=order.currency),
                metadata={"rapyd_event_type": event_type, "rapyd_status": payment_status},
            ),
            actor_user_id=order.buyer_user_id,
            event_key="rapyd_payment_approved",
            source="rapyd_webhook",
        )
        webhook_event.payment_id = result.payment.id
        webhook_event.processing_status = "processed"
        webhook_event.processed_at = utc_now()
        session.add(webhook_event)
        session.flush()
        return CommerceProviderWebhookIngestResponse(
            provider_key="rapyd",
            event_id=event_id,
            event_type=event_type,
            provider_resource_id=provider_resource_id,
            processing_status="processed",
            workspace_id=workspace_id,
            order_id=order.id,
            payment_id=result.payment.id,
            entitlement_id=result.entitlement.id if result.entitlement is not None else None,
            message="Rapyd approved payment processed.",
        )

    if event_type in REVOCATION_EVENTS or payment_status in REVOCATION_STATUSES:
        result = apply_provider_payment_revocation(
            session,
            order=order,
            event=ProviderPaymentEvent(
                provider_key="rapyd",
                provider_payment_id=provider_resource_id or event_id,
                event_id=event_id,
                event_type=event_type,
                amount_cents=_extract_amount_cents(data, fallback_cents=order.total_cents),
                currency=_extract_currency(data, fallback=order.currency),
                metadata={"rapyd_event_type": event_type, "rapyd_status": payment_status},
            ),
            payment_status=CommercialPaymentStatus.refunded,
            order_status=CommercialOrderStatus.refunded,
            entitlement_status=CommercialEntitlementStatus.refunded,
            actor_user_id=order.buyer_user_id,
            event_key="rapyd_payment_refunded",
            source="rapyd_webhook",
        )
        webhook_event.payment_id = result.payment.id
        webhook_event.processing_status = "processed"
        webhook_event.processed_at = utc_now()
        session.add(webhook_event)
        session.flush()
        return _response_from_event(webhook_event, message="Rapyd revocation event processed.")

    if event_type in FAILED_EVENTS or payment_status in FAILED_STATUSES:
        if order.status == CommercialOrderStatus.pending:
            order.status = CommercialOrderStatus.failed
            order.updated_at = utc_now()
            session.add(order)
        webhook_event.processing_status = "processed"
    else:
        webhook_event.processing_status = "processed" if payment_status in PENDING_STATUSES else "ignored"
    webhook_event.processed_at = utc_now()
    session.add(webhook_event)
    session.flush()
    return _response_from_event(webhook_event, message="Rapyd webhook recorded without granting access.")


def _response_from_event(
    event: CommerceProviderWebhookEventRecord,
    *,
    message: str,
) -> CommerceProviderWebhookIngestResponse:
    return CommerceProviderWebhookIngestResponse(
        provider_key=event.provider_key,
        event_id=event.event_id,
        event_type=event.event_type,
        provider_resource_id=event.provider_resource_id,
        processing_status=event.processing_status,
        workspace_id=event.workspace_id,
        order_id=event.order_id,
        payment_id=event.payment_id,
        message=message,
    )


def _validate_rapyd_signature(
    session: Session,
    *,
    workspace_id: UUID | None,
    environment: str,
    raw_body: bytes,
    request_headers: dict[str, str],
) -> bool:
    if workspace_id is None:
        return False
    access_key = load_commerce_provider_secret(
        session,
        workspace_id=workspace_id,
        provider_key="rapyd",
        environment=environment,
        secret_kind="access_key",
    )
    secret_key = load_commerce_provider_secret(
        session,
        workspace_id=workspace_id,
        provider_key="rapyd",
        environment=environment,
        secret_kind="secret_key",
    )
    return verify_rapyd_webhook_signature(
        raw_body=raw_body,
        request_headers=request_headers,
        candidate_urls=_webhook_candidate_urls(session, workspace_id=workspace_id, environment=environment),
        access_key=access_key,
        secret_key=secret_key,
    )


def _validate_url_secret(
    session: Session,
    *,
    workspace_id: UUID | None,
    environment: str,
    provided: str,
) -> bool:
    if workspace_id is None:
        return False
    expected = load_commerce_provider_secret(
        session,
        workspace_id=workspace_id,
        provider_key="rapyd",
        environment=environment,
        secret_kind="webhook_url_secret",
    )
    if not expected:
        return True
    return hmac.compare_digest((provided or "").strip(), expected)


def _webhook_candidate_urls(session: Session, *, workspace_id: UUID, environment: str) -> list[str]:
    config = session.exec(
        select(CommerceProviderConfigRecord).where(
            CommerceProviderConfigRecord.workspace_id == workspace_id,
            CommerceProviderConfigRecord.provider_key == "rapyd",
            CommerceProviderConfigRecord.environment == environment,
        )
    ).first()
    candidates = []
    if config is not None and config.webhook_public_url:
        candidates.append(config.webhook_public_url)
    settings_url = get_settings().rapyd_webhook_public_url.strip()
    if settings_url:
        candidates.append(settings_url)
    return candidates


def _decode_payload(raw_body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid Rapyd webhook JSON payload.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Invalid Rapyd webhook payload.")
    return payload


def _extract_data(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    if isinstance(data, dict):
        payment = data.get("payment")
        if isinstance(payment, dict):
            return {**payment, "_rapyd_parent_data": data}
        return data
    payment = payload.get("payment")
    if isinstance(payment, dict):
        return payment
    return payload


def _extract_metadata(data: dict[str, Any]) -> dict[str, Any]:
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    parent = data.get("_rapyd_parent_data")
    if isinstance(parent, dict) and isinstance(parent.get("metadata"), dict):
        return parent["metadata"]
    return {}


def _extract_event_type(payload: dict[str, Any]) -> str:
    raw = payload.get("type") or payload.get("event") or payload.get("event_type")
    return str(raw or "").strip().upper()


def _extract_provider_resource_id(payload: dict[str, Any], data: dict[str, Any]) -> str:
    return _first_string(
        data,
        ("id",),
        ("payment_id",),
        ("paymentId",),
        ("transaction_id",),
        ("transactionId",),
        ("_rapyd_parent_data", "id"),
        ("_rapyd_parent_data", "payment_id"),
        ("_rapyd_parent_data", "paymentId"),
        ("_rapyd_parent_data", "payment", "id"),
    ) or _first_string(payload, ("data", "id"), ("payment", "id"))


def _extract_event_id(
    payload: dict[str, Any],
    *,
    event_type: str,
    provider_resource_id: str,
    payload_hash: str,
) -> str:
    explicit = _first_string(
        payload,
        ("id",),
        ("event_id",),
        ("eventId",),
        ("webhook_id",),
        ("webhookId",),
        ("data", "event_id"),
        ("data", "eventId"),
    )
    if explicit:
        return explicit[:160]
    if event_type and provider_resource_id:
        return f"{event_type}:{provider_resource_id}"[:160]
    return f"payload:{payload_hash}"[:160]


def _extract_payment_status(data: dict[str, Any]) -> str:
    return _first_string(
        data,
        ("status",),
        ("payment_status",),
        ("paymentStatus",),
        ("state",),
        ("_rapyd_parent_data", "status"),
    ).strip().upper()


def _extract_amount_cents(data: dict[str, Any], *, fallback_cents: int) -> int:
    cents = _first_string(data, ("amount_cents",), ("amountCents",), ("total_cents",), ("totalCents",))
    if cents:
        try:
            return max(0, int(Decimal(cents)))
        except (InvalidOperation, ValueError):
            pass
    amount = _first_string(data, ("amount",), ("total",), ("paid_amount",), ("paidAmount",))
    if amount:
        try:
            return max(0, int((Decimal(str(amount)) * Decimal(100)).quantize(Decimal("1"))))
        except (InvalidOperation, ValueError):
            pass
    return fallback_cents


def _extract_currency(data: dict[str, Any], *, fallback: str) -> str:
    return (
        _first_string(
            data,
            ("currency",),
            ("currency_code",),
            ("currencyCode",),
            ("_rapyd_parent_data", "currency"),
            ("_rapyd_parent_data", "currency_code"),
        )
        or fallback
        or "USD"
    ).upper()


def _find_existing_event(
    session: Session,
    *,
    event_id: str,
    event_type: str,
) -> CommerceProviderWebhookEventRecord | None:
    return session.exec(
        select(CommerceProviderWebhookEventRecord).where(
            CommerceProviderWebhookEventRecord.provider_key == "rapyd",
            CommerceProviderWebhookEventRecord.event_id == event_id,
            CommerceProviderWebhookEventRecord.event_type == event_type,
        )
    ).first()


def _resolve_order(
    session: Session,
    *,
    data: dict[str, Any],
    metadata: dict[str, Any],
    provider_resource_id: str,
) -> CommercialOrderRecord | None:
    order_id = str(metadata.get("lab_order_id") or data.get("lab_order_id") or "").strip()
    if order_id:
        try:
            order = session.get(CommercialOrderRecord, UUID(order_id))
            if order is not None and order.provider == "rapyd":
                return order
        except ValueError:
            pass
    checkout_ref = str(
        metadata.get("lab_checkout_ref")
        or data.get("lab_checkout_ref")
        or data.get("merchant_reference_id")
        or data.get("merchantReferenceId")
        or ""
    ).strip()
    if checkout_ref:
        order = session.exec(
            select(CommercialOrderRecord).where(
                CommercialOrderRecord.provider == "rapyd",
                CommercialOrderRecord.checkout_ref == checkout_ref,
            )
        ).first()
        if order is not None:
            return order
    checkout_id = _first_string(data, ("checkout_id",), ("checkoutId",), ("checkout", "id"), ("_rapyd_parent_data", "checkout_id"))
    for candidate in (provider_resource_id, checkout_id):
        if not candidate:
            continue
        checkout_record = session.exec(
            select(CommerceProviderCheckoutRecord).where(
                CommerceProviderCheckoutRecord.provider_key == "rapyd",
                (
                    (CommerceProviderCheckoutRecord.provider_checkout_id == candidate)
                    | (CommerceProviderCheckoutRecord.provider_payment_link_id == candidate)
                ),
            )
        ).first()
        if checkout_record is not None:
            return session.get(CommercialOrderRecord, checkout_record.order_id)
    if provider_resource_id:
        payment = session.exec(
            select(CommercialPaymentRecord).where(
                CommercialPaymentRecord.provider == "rapyd",
                CommercialPaymentRecord.provider_payment_id == provider_resource_id,
            )
        ).first()
        if payment is not None:
            return session.get(CommercialOrderRecord, payment.order_id)
    return None


def _workspace_id_from_metadata_or_config(
    session: Session,
    metadata: dict[str, Any],
    *,
    environment: str,
) -> UUID | None:
    workspace_id = str(metadata.get("lab_workspace_id") or "").strip()
    if workspace_id:
        try:
            return UUID(workspace_id)
        except ValueError:
            pass
    config = session.exec(
        select(CommerceProviderConfigRecord).where(
            CommerceProviderConfigRecord.provider_key == "rapyd",
            CommerceProviderConfigRecord.environment == environment,
        )
    ).first()
    return config.workspace_id if config is not None else None


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
