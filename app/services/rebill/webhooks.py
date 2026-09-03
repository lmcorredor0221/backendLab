from __future__ import annotations

import base64
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
from app.services.rebill.client import RebillApiError, RebillClient, RebillClientConfig


SUCCESS_STATUSES = {"approved", "captured", "completed", "paid", "succeeded", "success"}
PENDING_STATUSES = {"created", "in_process", "pending", "processing", "requires_action"}
FAILED_STATUSES = {"cancelled", "canceled", "declined", "failed", "rejected"}
REVOCATION_STATUSES = {"chargeback", "refunded", "refund"}


def process_rebill_webhook(
    session: Session,
    *,
    raw_body: bytes,
    request_headers: dict[str, str],
    url_secret: str = "",
    environment: str = "sandbox",
    client_factory=RebillClient,
) -> CommerceProviderWebhookIngestResponse:
    env = normalize_commerce_provider_environment(environment)
    payload = _decode_payload(raw_body)
    event_type = _extract_event_type(payload)
    data = _extract_data(payload)
    metadata = _extract_metadata(data)
    payload_hash = hashlib.sha256(raw_body).hexdigest()
    provider_resource_id = _extract_provider_resource_id(data)
    event_id = _extract_event_id(
        payload,
        event_type=event_type,
        provider_resource_id=provider_resource_id,
        payload_hash=payload_hash,
    )
    order = _resolve_order(session, data=data, metadata=metadata, provider_resource_id=provider_resource_id)
    workspace_id = order.workspace_id if order is not None else _workspace_id_from_metadata_or_config(session, metadata, environment=env)
    webhook_event = _find_existing_event(session, event_id=event_id, event_type=event_type)
    signature_header = _header_value(request_headers, "x-rebill-signature")
    signature_validated = _validate_rebill_signature(
        session,
        workspace_id=workspace_id,
        environment=env,
        raw_body=raw_body,
        signature_header=signature_header,
    )
    url_secret_validated = _validate_url_secret(
        session,
        workspace_id=workspace_id,
        environment=env,
        provided=url_secret,
    )

    if webhook_event is not None:
        webhook_event.retries += 1
        webhook_event.signature_validated = webhook_event.signature_validated or signature_validated
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
            webhook_event.error_message = "Invalid Rebill webhook signature or URL secret."
            webhook_event.processed_at = utc_now()
            session.add(webhook_event)
            session.flush()
            raise PermissionError("Invalid Rebill webhook signature or URL secret.")
        return CommerceProviderWebhookIngestResponse(
            provider_key="rebill",
            event_id=webhook_event.event_id,
            event_type=webhook_event.event_type,
            provider_resource_id=webhook_event.provider_resource_id,
            processing_status=webhook_event.processing_status,
            duplicate=True,
            workspace_id=webhook_event.workspace_id,
            order_id=webhook_event.order_id,
            payment_id=webhook_event.payment_id,
            message="Duplicate Rebill webhook ignored.",
        )

    webhook_event = CommerceProviderWebhookEventRecord(
        provider_key="rebill",
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
        webhook_event.error_message = "Invalid Rebill webhook signature or URL secret."
        webhook_event.processed_at = utc_now()
        session.add(webhook_event)
        session.flush()
        raise PermissionError("Invalid Rebill webhook signature or URL secret.")

    if order is None or workspace_id is None:
        webhook_event.processing_status = "unresolved"
        webhook_event.error_code = "order_not_found"
        webhook_event.error_message = "Could not resolve internal order from Rebill webhook."
        webhook_event.processed_at = utc_now()
        session.add(webhook_event)
        session.flush()
        return CommerceProviderWebhookIngestResponse(
            provider_key="rebill",
            event_id=event_id,
            event_type=event_type,
            provider_resource_id=provider_resource_id,
            processing_status="unresolved",
            workspace_id=workspace_id,
            message=webhook_event.error_message,
        )

    confirmed_data = data
    if _event_may_change_access(event_type, data) and provider_resource_id:
        secret_key = load_commerce_provider_secret(
            session,
            workspace_id=workspace_id,
            provider_key="rebill",
            environment=env,
            secret_kind="secret_key",
        )
        if not secret_key:
            webhook_event.processing_status = "failed"
            webhook_event.error_code = "missing_secret_key"
            webhook_event.error_message = "Rebill secret key is required to confirm payment before fulfillment."
            webhook_event.processed_at = utc_now()
            session.add(webhook_event)
            session.flush()
            return _response_from_event(webhook_event, message=webhook_event.error_message)
        client = client_factory(
            RebillClientConfig(
                api_base_url=_api_base_url_for_workspace(session, workspace_id=workspace_id, environment=env),
                timeout_seconds=get_settings().rebill_request_timeout_seconds,
            )
        )
        try:
            confirmed_data = _extract_data(client.get_payment(secret_key=secret_key, payment_id=provider_resource_id))
        except RebillApiError as exc:
            webhook_event.processing_status = "failed"
            webhook_event.error_code = exc.code
            webhook_event.error_message = str(exc)
            webhook_event.processed_at = utc_now()
            session.add(webhook_event)
            session.flush()
            return _response_from_event(webhook_event, message=webhook_event.error_message)

    payment_status = _extract_payment_status(confirmed_data)
    if payment_status in SUCCESS_STATUSES:
        result = apply_provider_payment_success(
            session,
            order=order,
            event=ProviderPaymentEvent(
                provider_key="rebill",
                provider_payment_id=provider_resource_id or event_id,
                event_id=event_id,
                event_type=event_type,
                amount_cents=_extract_amount_cents(confirmed_data, fallback_cents=order.total_cents),
                currency=_extract_currency(confirmed_data, fallback=order.currency),
                metadata={"rebill_event_type": event_type, "rebill_status": payment_status},
            ),
            actor_user_id=order.buyer_user_id,
            event_key="rebill_payment_approved",
            source="rebill_webhook",
        )
        webhook_event.payment_id = result.payment.id
        webhook_event.processing_status = "processed"
        webhook_event.processed_at = utc_now()
        session.add(webhook_event)
        session.flush()
        return CommerceProviderWebhookIngestResponse(
            provider_key="rebill",
            event_id=event_id,
            event_type=event_type,
            provider_resource_id=provider_resource_id,
            processing_status="processed",
            workspace_id=workspace_id,
            order_id=order.id,
            payment_id=result.payment.id,
            entitlement_id=result.entitlement.id if result.entitlement is not None else None,
            message="Rebill approved payment processed.",
        )

    if payment_status in REVOCATION_STATUSES or _event_is_revocation(event_type):
        result = apply_provider_payment_revocation(
            session,
            order=order,
            event=ProviderPaymentEvent(
                provider_key="rebill",
                provider_payment_id=provider_resource_id or event_id,
                event_id=event_id,
                event_type=event_type,
                amount_cents=_extract_amount_cents(confirmed_data, fallback_cents=order.total_cents),
                currency=_extract_currency(confirmed_data, fallback=order.currency),
                metadata={"rebill_event_type": event_type, "rebill_status": payment_status},
            ),
            payment_status=CommercialPaymentStatus.refunded,
            order_status=CommercialOrderStatus.refunded,
            entitlement_status=CommercialEntitlementStatus.refunded,
            actor_user_id=order.buyer_user_id,
            event_key="rebill_payment_refunded",
            source="rebill_webhook",
        )
        webhook_event.payment_id = result.payment.id
        webhook_event.processing_status = "processed"
        webhook_event.processed_at = utc_now()
        session.add(webhook_event)
        session.flush()
        return CommerceProviderWebhookIngestResponse(
            provider_key="rebill",
            event_id=event_id,
            event_type=event_type,
            provider_resource_id=provider_resource_id,
            processing_status="processed",
            workspace_id=workspace_id,
            order_id=order.id,
            payment_id=result.payment.id,
            entitlement_id=result.entitlement.id if result.entitlement is not None else None,
            message="Rebill revocation event processed.",
        )

    webhook_event.processing_status = "processed" if payment_status in PENDING_STATUSES | FAILED_STATUSES else "ignored"
    webhook_event.processed_at = utc_now()
    session.add(webhook_event)
    session.flush()
    return _response_from_event(
        webhook_event,
        message="Rebill webhook recorded without granting access.",
    )


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


def verify_rebill_signature(*, raw_body: bytes, signature_header: str, signing_secret: str) -> bool:
    signature = (signature_header or "").strip()
    secret = (signing_secret or "").strip()
    if not signature or not secret:
        return False
    expected_digest = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).digest()
    expected_hex = expected_digest.hex()
    expected_base64 = base64.b64encode(expected_digest).decode("ascii")
    candidates = {
        signature,
        signature.removeprefix("sha256=").strip(),
    }
    return any(
        hmac.compare_digest(candidate, expected_hex) or hmac.compare_digest(candidate, expected_base64)
        for candidate in candidates
    )


def _validate_rebill_signature(
    session: Session,
    *,
    workspace_id: UUID | None,
    environment: str,
    raw_body: bytes,
    signature_header: str,
) -> bool:
    if workspace_id is None:
        return False
    signing_secret = load_commerce_provider_secret(
        session,
        workspace_id=workspace_id,
        provider_key="rebill",
        environment=environment,
        secret_kind="webhook_signing_secret",
    )
    return verify_rebill_signature(
        raw_body=raw_body,
        signature_header=signature_header,
        signing_secret=signing_secret,
    )


def _validate_url_secret(
    session: Session,
    *,
    workspace_id: UUID | None,
    environment: str,
    provided: str,
) -> bool:
    if workspace_id is None:
        return True
    expected = load_commerce_provider_secret(
        session,
        workspace_id=workspace_id,
        provider_key="rebill",
        environment=environment,
        secret_kind="webhook_url_secret",
    )
    if not expected:
        return True
    return hmac.compare_digest((provided or "").strip(), expected)


def _decode_payload(raw_body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid Rebill webhook JSON payload.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Invalid Rebill webhook payload.")
    return payload


def _extract_data(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    if isinstance(data, dict):
        nested_payment = data.get("payment")
        if isinstance(nested_payment, dict):
            return {**nested_payment, "_rebill_parent_data": data}
        return data
    payment = payload.get("payment")
    if isinstance(payment, dict):
        return payment
    return payload


def _extract_metadata(data: dict[str, Any]) -> dict[str, Any]:
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    parent = data.get("_rebill_parent_data")
    if isinstance(parent, dict) and isinstance(parent.get("metadata"), dict):
        return parent["metadata"]
    return {}


def _extract_event_type(payload: dict[str, Any]) -> str:
    raw = payload.get("event") or payload.get("event_type") or payload.get("type")
    webhook = payload.get("webhook")
    if isinstance(webhook, dict):
        raw = raw or webhook.get("event") or webhook.get("type")
    return str(raw or "").strip().lower()


def _extract_provider_resource_id(data: dict[str, Any]) -> str:
    return _first_string(
        data,
        ("id",),
        ("uuid",),
        ("payment_id",),
        ("paymentId",),
        ("transaction_id",),
        ("transactionId",),
        ("_rebill_parent_data", "id"),
        ("_rebill_parent_data", "payment_id"),
    )


def _extract_event_id(
    payload: dict[str, Any],
    *,
    event_type: str,
    provider_resource_id: str,
    payload_hash: str,
) -> str:
    webhook = payload.get("webhook")
    explicit = _first_string(
        payload,
        ("id",),
        ("event_id",),
        ("eventId",),
        ("data", "event_id"),
        ("data", "eventId"),
    )
    if not explicit and isinstance(webhook, dict):
        explicit = _first_string(webhook, ("id",), ("logId",), ("log_id",))
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
        ("_rebill_parent_data", "status"),
    ).strip().lower()


def _extract_amount_cents(data: dict[str, Any], *, fallback_cents: int) -> int:
    cents = _first_string(data, ("amount_cents",), ("amountCents",), ("total_cents",), ("totalCents",))
    if cents:
        try:
            return max(0, int(float(cents)))
        except ValueError:
            pass
    amount = _first_string(data, ("amount",), ("total",), ("paid_amount",), ("paidAmount",))
    if amount:
        try:
            return max(0, int(round(float(amount) * 100)))
        except ValueError:
            pass
    return fallback_cents


def _extract_currency(data: dict[str, Any], *, fallback: str) -> str:
    return (
        _first_string(
            data,
            ("currency",),
            ("currency_code",),
            ("currencyCode",),
            ("_rebill_parent_data", "currency"),
        )
        or fallback
        or "USD"
    ).upper()


def _event_may_change_access(event_type: str, data: dict[str, Any]) -> bool:
    status = _extract_payment_status(data)
    return event_type.startswith("payment.") or event_type.startswith("subscription.") or status in SUCCESS_STATUSES | REVOCATION_STATUSES


def _event_is_revocation(event_type: str) -> bool:
    lowered = event_type.lower()
    return "refund" in lowered or "chargeback" in lowered or "cancel" in lowered


def _find_existing_event(
    session: Session,
    *,
    event_id: str,
    event_type: str,
) -> CommerceProviderWebhookEventRecord | None:
    return session.exec(
        select(CommerceProviderWebhookEventRecord).where(
            CommerceProviderWebhookEventRecord.provider_key == "rebill",
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
            if order is not None:
                return order
        except ValueError:
            pass
    checkout_ref = str(metadata.get("lab_checkout_ref") or data.get("lab_checkout_ref") or "").strip()
    if checkout_ref:
        order = session.exec(select(CommercialOrderRecord).where(CommercialOrderRecord.checkout_ref == checkout_ref)).first()
        if order is not None:
            return order
    if provider_resource_id:
        checkout_record = session.exec(
            select(CommerceProviderCheckoutRecord).where(
                CommerceProviderCheckoutRecord.provider_key == "rebill",
                (
                    (CommerceProviderCheckoutRecord.provider_checkout_id == provider_resource_id)
                    | (CommerceProviderCheckoutRecord.provider_payment_link_id == provider_resource_id)
                ),
            )
        ).first()
        if checkout_record is not None:
            return session.get(CommercialOrderRecord, checkout_record.order_id)
    if provider_resource_id:
        payment = session.exec(
            select(CommercialPaymentRecord).where(
                CommercialPaymentRecord.provider == "rebill",
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
            CommerceProviderConfigRecord.provider_key == "rebill",
            CommerceProviderConfigRecord.environment == environment,
        )
    ).first()
    return config.workspace_id if config is not None else None


def _api_base_url_for_workspace(session: Session, *, workspace_id: UUID, environment: str) -> str:
    config = session.exec(
        select(CommerceProviderConfigRecord).where(
            CommerceProviderConfigRecord.workspace_id == workspace_id,
            CommerceProviderConfigRecord.provider_key == "rebill",
            CommerceProviderConfigRecord.environment == environment,
        )
    ).first()
    return (config.api_base_url if config is not None and config.api_base_url else get_settings().rebill_api_base_url).rstrip("/")


def _header_value(headers: dict[str, str], key: str) -> str:
    expected = key.lower()
    for header_key, value in headers.items():
        if header_key.lower() == expected:
            return str(value)
    return ""


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
