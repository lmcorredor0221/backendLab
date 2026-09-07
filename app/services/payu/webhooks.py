from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any
from urllib.parse import parse_qs
from uuid import UUID

from sqlmodel import Session, select

from app.models import (
    CommerceProviderCheckoutRecord,
    CommerceProviderWebhookEventRecord,
    CommerceProviderWebhookIngestResponse,
    CommercialEntitlementStatus,
    CommercialOrderRecord,
    CommercialOrderStatus,
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
from app.services.payu.signatures import amount_cents_from_payu_value, verify_payu_confirmation_signature


SUCCESS_STATE_POL = {"4"}
FAILED_STATE_POL = {"5", "6"}
PENDING_STATE_POL = {"7"}
REVOCATION_CODES = {"REFUND", "REVERSED", "CHARGEBACK", "REFUNDED"}


def process_payu_webhook(
    session: Session,
    *,
    raw_body: bytes,
    request_headers: dict[str, str],
    url_secret: str = "",
    environment: str = "sandbox",
) -> CommerceProviderWebhookIngestResponse:
    env = normalize_commerce_provider_environment(environment)
    payload = _decode_payload(raw_body)
    payload_hash = hashlib.sha256(raw_body).hexdigest()
    reference_sale = _payload_value(payload, "reference_sale", "referenceCode")
    state_pol = _payload_value(payload, "state_pol", "transactionState", "polTransactionState")
    reference_pol = _payload_value(payload, "reference_pol")
    transaction_id = _payload_value(payload, "transaction_id", "transactionId")
    response_code = _payload_value(payload, "response_code_pol", "polResponseCode", "lapResponseCode").upper()
    event_type = f"payu.confirmation.state_{state_pol or 'unknown'}"
    event_id = transaction_id or (f"{reference_pol}:{state_pol}" if reference_pol else payload_hash)
    provider_resource_id = transaction_id or reference_pol or reference_sale
    order = _resolve_order(session, payload=payload, reference_sale=reference_sale)
    workspace_id = order.workspace_id if order is not None else _workspace_id_from_payload(payload)
    webhook_event = _find_existing_event(session, event_id=event_id, event_type=event_type)
    signature_validated = _validate_payu_signature(
        session,
        workspace_id=workspace_id,
        environment=env,
        payload=payload,
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
            webhook_event.error_message = "Invalid PayU confirmation signature or URL secret."
            webhook_event.processed_at = utc_now()
            session.add(webhook_event)
            session.flush()
            raise PermissionError("Invalid PayU confirmation signature or URL secret.")
        return CommerceProviderWebhookIngestResponse(
            provider_key="payu",
            event_id=webhook_event.event_id,
            event_type=webhook_event.event_type,
            provider_resource_id=webhook_event.provider_resource_id,
            processing_status=webhook_event.processing_status,
            duplicate=True,
            workspace_id=webhook_event.workspace_id,
            order_id=webhook_event.order_id,
            payment_id=webhook_event.payment_id,
            message="Duplicate PayU confirmation ignored.",
        )

    webhook_event = CommerceProviderWebhookEventRecord(
        provider_key="payu",
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
        webhook_event.error_message = "Invalid PayU confirmation signature or URL secret."
        webhook_event.processed_at = utc_now()
        session.add(webhook_event)
        session.flush()
        raise PermissionError("Invalid PayU confirmation signature or URL secret.")

    if order is None or workspace_id is None:
        webhook_event.processing_status = "unresolved"
        webhook_event.error_code = "order_not_found"
        webhook_event.error_message = "Could not resolve internal order from PayU confirmation."
        webhook_event.processed_at = utc_now()
        session.add(webhook_event)
        session.flush()
        return _response_from_event(webhook_event, message=webhook_event.error_message)

    amount_cents = amount_cents_from_payu_value(_payload_value(payload, "value", "TX_VALUE"), fallback_cents=order.total_cents)
    currency = _payload_value(payload, "currency") or order.currency
    if state_pol in SUCCESS_STATE_POL:
        result = apply_provider_payment_success(
            session,
            order=order,
            event=ProviderPaymentEvent(
                provider_key="payu",
                provider_payment_id=provider_resource_id or event_id,
                event_id=event_id,
                event_type=event_type,
                amount_cents=amount_cents,
                currency=currency,
                metadata={
                    "payu_state_pol": state_pol,
                    "payu_reference_sale": reference_sale,
                    "payu_reference_pol": reference_pol,
                    "payu_response_code": response_code,
                },
            ),
            actor_user_id=order.buyer_user_id,
            event_key="payu_payment_approved",
            source="payu_confirmation_url",
        )
        webhook_event.payment_id = result.payment.id
        webhook_event.processing_status = "processed"
        webhook_event.processed_at = utc_now()
        session.add(webhook_event)
        session.flush()
        return CommerceProviderWebhookIngestResponse(
            provider_key="payu",
            event_id=event_id,
            event_type=event_type,
            provider_resource_id=provider_resource_id,
            processing_status="processed",
            workspace_id=workspace_id,
            order_id=order.id,
            payment_id=result.payment.id,
            entitlement_id=result.entitlement.id if result.entitlement is not None else None,
            message="PayU approved payment processed.",
        )

    if response_code in REVOCATION_CODES:
        result = apply_provider_payment_revocation(
            session,
            order=order,
            event=ProviderPaymentEvent(
                provider_key="payu",
                provider_payment_id=provider_resource_id or event_id,
                event_id=event_id,
                event_type=event_type,
                amount_cents=amount_cents,
                currency=currency,
                metadata={
                    "payu_state_pol": state_pol,
                    "payu_reference_sale": reference_sale,
                    "payu_reference_pol": reference_pol,
                    "payu_response_code": response_code,
                },
            ),
            payment_status=CommercialPaymentStatus.refunded,
            order_status=CommercialOrderStatus.refunded,
            entitlement_status=CommercialEntitlementStatus.refunded,
            actor_user_id=order.buyer_user_id,
            event_key="payu_payment_refunded",
            source="payu_confirmation_url",
        )
        webhook_event.payment_id = result.payment.id
        webhook_event.processing_status = "processed"
        webhook_event.processed_at = utc_now()
        session.add(webhook_event)
        session.flush()
        return _response_from_event(webhook_event, message="PayU revocation event processed.")

    if state_pol in FAILED_STATE_POL and order.status == CommercialOrderStatus.pending:
        order.status = CommercialOrderStatus.failed
        order.updated_at = utc_now()
        session.add(order)
    webhook_event.processing_status = "processed" if state_pol in FAILED_STATE_POL | PENDING_STATE_POL else "ignored"
    webhook_event.processed_at = utc_now()
    session.add(webhook_event)
    session.flush()
    return _response_from_event(webhook_event, message="PayU confirmation recorded without granting access.")


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


def _decode_payload(raw_body: bytes) -> dict[str, Any]:
    text = raw_body.decode("utf-8", errors="replace")
    parsed = parse_qs(text, keep_blank_values=True)
    if parsed:
        return {key: values[-1] if values else "" for key, values in parsed.items()}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("PayU confirmation payload is not valid form data or JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("PayU confirmation payload must be an object.")
    return payload


def _payload_value(payload: dict[str, Any], *keys: str) -> str:
    lowered = {str(key).lower(): value for key, value in payload.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _resolve_order(
    session: Session,
    *,
    payload: dict[str, Any],
    reference_sale: str,
) -> CommercialOrderRecord | None:
    order_id = _payload_value(payload, "extra2", "lab_order_id")
    if order_id:
        try:
            order = session.get(CommercialOrderRecord, UUID(order_id))
        except ValueError:
            order = None
        if order is not None and order.provider == "payu":
            return order
    checkout_ref = _payload_value(payload, "extra1", "lab_checkout_ref")
    if checkout_ref:
        order = session.exec(
            select(CommercialOrderRecord).where(
                CommercialOrderRecord.provider == "payu",
                CommercialOrderRecord.checkout_ref == checkout_ref,
            )
        ).first()
        if order is not None:
            return order
    if reference_sale:
        checkout_record = session.exec(
            select(CommerceProviderCheckoutRecord).where(
                CommerceProviderCheckoutRecord.provider_key == "payu",
                CommerceProviderCheckoutRecord.provider_checkout_id == reference_sale,
            )
        ).first()
        if checkout_record is not None:
            return session.get(CommercialOrderRecord, checkout_record.order_id)
    return None


def _workspace_id_from_payload(payload: dict[str, Any]) -> UUID | None:
    workspace_id = _payload_value(payload, "extra3", "lab_workspace_id")
    if not workspace_id:
        return None
    try:
        return UUID(workspace_id)
    except ValueError:
        return None


def _find_existing_event(
    session: Session,
    *,
    event_id: str,
    event_type: str,
) -> CommerceProviderWebhookEventRecord | None:
    return session.exec(
        select(CommerceProviderWebhookEventRecord).where(
            CommerceProviderWebhookEventRecord.provider_key == "payu",
            CommerceProviderWebhookEventRecord.event_id == event_id,
            CommerceProviderWebhookEventRecord.event_type == event_type,
        )
    ).first()


def _validate_payu_signature(
    session: Session,
    *,
    workspace_id: UUID | None,
    environment: str,
    payload: dict[str, Any],
) -> bool:
    if workspace_id is None:
        return False
    api_key = load_commerce_provider_secret(
        session,
        workspace_id=workspace_id,
        provider_key="payu",
        environment=environment,
        secret_kind="secret_key",
    )
    hmac_secret = load_commerce_provider_secret(
        session,
        workspace_id=workspace_id,
        provider_key="payu",
        environment=environment,
        secret_kind="webhook_signing_secret",
    )
    return verify_payu_confirmation_signature(payload, api_key=api_key, hmac_secret=hmac_secret)


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
        provider_key="payu",
        environment=environment,
        secret_kind="webhook_url_secret",
    )
    if not expected:
        return True
    return hmac.compare_digest((provided or "").strip(), expected)
