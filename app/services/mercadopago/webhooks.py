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
from app.services.mercadopago.client import MercadoPagoApiError, MercadoPagoClient, MercadoPagoClientConfig
from app.services.mercadopago.signatures import verify_mercadopago_webhook_signature


SUCCESS_STATUSES = {"processed"}
SUCCESS_DETAILS = {"accredited"}
PENDING_STATUSES = {"action_required", "authorized", "created", "in_process", "pending", "processing"}
FAILED_STATUSES = {"cancelled", "canceled", "declined", "expired", "failed", "rejected"}
REVOCATION_STATUSES = {"chargeback", "charged_back", "partially_refunded", "refunded"}


def process_mercadopago_webhook(
    session: Session,
    *,
    raw_body: bytes,
    request_headers: dict[str, str],
    query_params: dict[str, str] | None = None,
    url_secret: str = "",
    environment: str = "sandbox",
    client_factory=MercadoPagoClient,
) -> CommerceProviderWebhookIngestResponse:
    env = normalize_commerce_provider_environment(environment)
    payload = _decode_payload(raw_body)
    query = query_params or {}
    payload_hash = hashlib.sha256(raw_body).hexdigest()
    data = _extract_data(payload)
    metadata = _extract_metadata(data)
    provider_resource_id = _query_value(query, "data.id", "data_id") or _extract_provider_resource_id(payload, data)
    event_type = _query_value(query, "type", "topic") or _extract_event_type(payload)
    event_id = _extract_event_id(
        payload,
        event_type=event_type,
        provider_resource_id=provider_resource_id,
        payload_hash=payload_hash,
    )
    order = _resolve_order(session, data=data, metadata=metadata, provider_resource_id=provider_resource_id)
    workspace_id = order.workspace_id if order is not None else _workspace_id_from_metadata_or_config(session, metadata, environment=env)
    webhook_event = _find_existing_event(session, event_id=event_id, event_type=event_type)
    validation_workspace_id = workspace_id or (webhook_event.workspace_id if webhook_event is not None else None)
    signature_validated = _validate_signature(
        session,
        workspace_id=validation_workspace_id,
        environment=env,
        provider_resource_id=provider_resource_id,
        request_headers=request_headers,
    )
    url_secret_validated = _validate_url_secret(
        session,
        workspace_id=validation_workspace_id,
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
            raise PermissionError("Invalid Mercado Pago webhook signature or URL secret.")
        return CommerceProviderWebhookIngestResponse(
            provider_key="mercadopago",
            event_id=webhook_event.event_id,
            event_type=webhook_event.event_type,
            provider_resource_id=webhook_event.provider_resource_id,
            processing_status=webhook_event.processing_status,
            duplicate=True,
            workspace_id=webhook_event.workspace_id,
            order_id=webhook_event.order_id,
            payment_id=webhook_event.payment_id,
            message="Duplicate Mercado Pago webhook ignored.",
        )

    webhook_event = CommerceProviderWebhookEventRecord(
        provider_key="mercadopago",
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
        webhook_event.error_message = "Invalid Mercado Pago webhook signature or URL secret."
        webhook_event.processed_at = utc_now()
        session.add(webhook_event)
        session.flush()
        raise PermissionError("Invalid Mercado Pago webhook signature or URL secret.")

    confirmed_data = data
    if _event_may_change_access(event_type, data) and provider_resource_id:
        access_token = load_commerce_provider_secret(
            session,
            workspace_id=workspace_id,
            provider_key="mercadopago",
            environment=env,
            secret_kind="secret_key",
        )
        if not access_token:
            webhook_event.processing_status = "failed"
            webhook_event.error_code = "missing_access_token"
            webhook_event.error_message = "Mercado Pago access token is required to confirm payment before fulfillment."
            webhook_event.processed_at = utc_now()
            session.add(webhook_event)
            session.flush()
            return _response_from_event(webhook_event, message=webhook_event.error_message)
        client = client_factory(
            MercadoPagoClientConfig(
                api_base_url=_api_base_url_for_workspace(session, workspace_id=workspace_id, environment=env),
                timeout_seconds=get_settings().mercadopago_webhook_timeout_seconds,
                environment=env,
            )
        )
        try:
            confirmed_data = client.get_order(access_token=access_token, order_id=provider_resource_id)
        except MercadoPagoApiError as exc:
            webhook_event.processing_status = "failed"
            webhook_event.error_code = exc.code
            webhook_event.error_message = str(exc)
            webhook_event.processed_at = utc_now()
            session.add(webhook_event)
            session.flush()
            return _response_from_event(webhook_event, message=webhook_event.error_message)
        metadata = _extract_metadata(confirmed_data)
        order = _resolve_order(session, data=confirmed_data, metadata=metadata, provider_resource_id=provider_resource_id) or order
        if order is not None:
            workspace_id = order.workspace_id
            webhook_event.workspace_id = workspace_id
            webhook_event.order_id = order.id

    if order is None or workspace_id is None:
        webhook_event.processing_status = "unresolved"
        webhook_event.error_code = "order_not_found"
        webhook_event.error_message = "Could not resolve internal order from Mercado Pago webhook."
        webhook_event.processed_at = utc_now()
        session.add(webhook_event)
        session.flush()
        return _response_from_event(webhook_event, message=webhook_event.error_message)

    order_status = _extract_order_status(confirmed_data)
    status_detail = _extract_status_detail(confirmed_data)
    status_markers = {order_status, status_detail}
    provider_payment_id = _extract_provider_payment_id(confirmed_data, fallback=provider_resource_id or event_id)
    if _is_successful_order(confirmed_data, order_status=order_status, status_detail=status_detail):
        result = apply_provider_payment_success(
            session,
            order=order,
            event=ProviderPaymentEvent(
                provider_key="mercadopago",
                provider_payment_id=provider_payment_id,
                event_id=event_id,
                event_type=event_type,
                amount_cents=_extract_amount_cents(confirmed_data, fallback_cents=order.total_cents),
                currency=_extract_currency(confirmed_data, fallback=order.currency),
                metadata={
                    "mercadopago_event_type": event_type,
                    "mercadopago_order_id": provider_resource_id,
                    "mercadopago_status": order_status,
                    "mercadopago_status_detail": status_detail,
                },
            ),
            actor_user_id=order.buyer_user_id,
            event_key="mercadopago_order_processed",
            source="mercadopago_webhook",
        )
        webhook_event.payment_id = result.payment.id
        webhook_event.processing_status = "processed"
        webhook_event.processed_at = utc_now()
        session.add(webhook_event)
        session.flush()
        return CommerceProviderWebhookIngestResponse(
            provider_key="mercadopago",
            event_id=event_id,
            event_type=event_type,
            provider_resource_id=provider_resource_id,
            processing_status="processed",
            workspace_id=workspace_id,
            order_id=order.id,
            payment_id=result.payment.id,
            entitlement_id=result.entitlement.id if result.entitlement is not None else None,
            message="Mercado Pago processed order recorded.",
        )

    if status_markers & REVOCATION_STATUSES or _has_refunds_or_chargebacks(confirmed_data):
        result = apply_provider_payment_revocation(
            session,
            order=order,
            event=ProviderPaymentEvent(
                provider_key="mercadopago",
                provider_payment_id=provider_payment_id,
                event_id=event_id,
                event_type=event_type,
                amount_cents=_extract_amount_cents(confirmed_data, fallback_cents=order.total_cents),
                currency=_extract_currency(confirmed_data, fallback=order.currency),
                metadata={
                    "mercadopago_event_type": event_type,
                    "mercadopago_order_id": provider_resource_id,
                    "mercadopago_status": order_status,
                    "mercadopago_status_detail": status_detail,
                },
            ),
            payment_status=CommercialPaymentStatus.refunded,
            order_status=CommercialOrderStatus.refunded,
            entitlement_status=CommercialEntitlementStatus.refunded,
            actor_user_id=order.buyer_user_id,
            event_key="mercadopago_order_revoked",
            source="mercadopago_webhook",
        )
        webhook_event.payment_id = result.payment.id
        webhook_event.processing_status = "processed"
        webhook_event.processed_at = utc_now()
        session.add(webhook_event)
        session.flush()
        return _response_from_event(webhook_event, message="Mercado Pago revocation event processed.")

    if status_markers & FAILED_STATUSES and order.status == CommercialOrderStatus.pending:
        order.status = CommercialOrderStatus.failed
        order.updated_at = utc_now()
        session.add(order)
    known_non_success = bool(status_markers & (PENDING_STATUSES | FAILED_STATUSES))
    webhook_event.processing_status = "processed" if known_non_success else "ignored"
    webhook_event.processed_at = utc_now()
    session.add(webhook_event)
    session.flush()
    return _response_from_event(webhook_event, message="Mercado Pago webhook recorded without granting access.")


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
    try:
        payload = json.loads(raw_body.decode("utf-8") or "{}")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid Mercado Pago webhook JSON payload.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Invalid Mercado Pago webhook payload.")
    return payload


def _query_value(query_params: dict[str, str], *keys: str) -> str:
    lowered = {str(key).lower(): value for key, value in query_params.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _extract_data(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    if isinstance(data, dict):
        nested_payment = data.get("payment")
        if isinstance(nested_payment, dict):
            return {**nested_payment, "_mercadopago_parent_data": data}
        return data
    payment = payload.get("payment")
    if isinstance(payment, dict):
        return payment
    return payload


def _extract_metadata(data: dict[str, Any]) -> dict[str, Any]:
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    parent = data.get("_mercadopago_parent_data")
    if isinstance(parent, dict) and isinstance(parent.get("metadata"), dict):
        return parent["metadata"]
    return {}


def _extract_event_type(payload: dict[str, Any]) -> str:
    raw = payload.get("type") or payload.get("topic") or payload.get("action") or payload.get("event")
    return str(raw or "").strip().lower()


def _extract_provider_resource_id(payload: dict[str, Any], data: dict[str, Any]) -> str:
    return _first_string(
        data,
        ("id",),
        ("payment_id",),
        ("paymentId",),
        ("_mercadopago_parent_data", "id"),
    ) or _first_string(payload, ("data", "id"), ("payment", "id"), ("resource", "id"))


def _extract_event_id(
    payload: dict[str, Any],
    *,
    event_type: str,
    provider_resource_id: str,
    payload_hash: str,
) -> str:
    explicit = _first_string(payload, ("id",), ("event_id",), ("eventId",), ("notification_id",), ("notificationId",))
    if explicit:
        return explicit[:160]
    if event_type and provider_resource_id:
        return f"{event_type}:{provider_resource_id}"[:160]
    return f"payload:{payload_hash}"[:160]


def _extract_order_status(data: dict[str, Any]) -> str:
    return _first_string(data, ("status",), ("payment_status",), ("paymentStatus",), ("state",)).strip().lower()


def _extract_status_detail(data: dict[str, Any]) -> str:
    return _first_string(data, ("status_detail",), ("statusDetail",), ("payment_status_detail",)).strip().lower()


def _is_successful_order(data: dict[str, Any], *, order_status: str, status_detail: str) -> bool:
    status_markers = {order_status, status_detail}
    if status_markers & REVOCATION_STATUSES or status_markers & FAILED_STATUSES:
        return False
    if order_status in SUCCESS_STATUSES or status_detail in SUCCESS_DETAILS:
        return True
    for payment in _transaction_payments(data):
        payment_status = _extract_order_status(payment)
        payment_detail = _extract_status_detail(payment)
        if payment_status in SUCCESS_STATUSES and payment_detail in SUCCESS_DETAILS:
            return True
    return False


def _extract_provider_payment_id(data: dict[str, Any], *, fallback: str) -> str:
    for payment in _transaction_payments(data):
        payment_id = _first_string(payment, ("id",), ("payment_id",), ("paymentId",), ("reference_id",))
        if payment_id:
            return payment_id
    return _first_string(data, ("payment_id",), ("paymentId",), ("payment", "id")) or fallback


def _extract_amount_cents(data: dict[str, Any], *, fallback_cents: int) -> int:
    cents = _first_string(data, ("amount_cents",), ("amountCents",), ("total_cents",), ("totalCents",))
    if cents:
        try:
            return max(0, int(Decimal(cents)))
        except (InvalidOperation, ValueError):
            pass
    amount = _first_string(data, ("total_paid_amount",), ("transaction_amount",), ("paid_amount",), ("amount",), ("total_amount",))
    if amount:
        try:
            return max(0, int((Decimal(str(amount)) * Decimal(100)).quantize(Decimal("1"))))
        except (InvalidOperation, ValueError):
            pass
    for payment in _transaction_payments(data):
        payment_amount = _first_string(payment, ("paid_amount",), ("amount",), ("transaction_amount",))
        if payment_amount:
            try:
                return max(0, int((Decimal(str(payment_amount)) * Decimal(100)).quantize(Decimal("1"))))
            except (InvalidOperation, ValueError):
                pass
    return fallback_cents


def _extract_currency(data: dict[str, Any], *, fallback: str) -> str:
    return (
        _first_string(data, ("currency_id",), ("currency",), ("currency_code",), ("currencyCode",))
        or fallback
        or "USD"
    ).upper()


def _event_may_change_access(event_type: str, data: dict[str, Any]) -> bool:
    status = _extract_order_status(data)
    status_detail = _extract_status_detail(data)
    return (
        event_type in {"merchant_order", "order", "orders", "payment"}
        or event_type.startswith(("merchant_order", "order", "payment"))
        or status in SUCCESS_STATUSES | REVOCATION_STATUSES
        or status_detail in SUCCESS_DETAILS | REVOCATION_STATUSES
    )


def _validate_signature(
    session: Session,
    *,
    workspace_id: UUID | None,
    environment: str,
    provider_resource_id: str,
    request_headers: dict[str, str],
) -> bool:
    if workspace_id is None:
        return False
    signing_secret = load_commerce_provider_secret(
        session,
        workspace_id=workspace_id,
        provider_key="mercadopago",
        environment=environment,
        secret_kind="webhook_signing_secret",
    )
    return verify_mercadopago_webhook_signature(
        data_id=provider_resource_id,
        request_headers=request_headers,
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
        return False
    expected = load_commerce_provider_secret(
        session,
        workspace_id=workspace_id,
        provider_key="mercadopago",
        environment=environment,
        secret_kind="webhook_url_secret",
    )
    if not expected:
        return True
    return hmac.compare_digest((provided or "").strip(), expected)


def _find_existing_event(
    session: Session,
    *,
    event_id: str,
    event_type: str,
) -> CommerceProviderWebhookEventRecord | None:
    return session.exec(
        select(CommerceProviderWebhookEventRecord).where(
            CommerceProviderWebhookEventRecord.provider_key == "mercadopago",
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
            if order is not None and order.provider == "mercadopago":
                return order
        except ValueError:
            pass
    checkout_ref = str(
        metadata.get("lab_checkout_ref")
        or data.get("lab_checkout_ref")
        or data.get("external_reference")
        or data.get("externalReference")
        or ""
    ).strip()
    if checkout_ref:
        order = session.exec(
            select(CommercialOrderRecord).where(
                CommercialOrderRecord.provider == "mercadopago",
                CommercialOrderRecord.checkout_ref == checkout_ref,
            )
        ).first()
        if order is not None:
            return order
    checkout_id = _first_string(data, ("id",), ("order_id",), ("orderId",), ("order", "id"), ("preference_id",), ("preferenceId"))
    for candidate in (provider_resource_id, checkout_id):
        if not candidate:
            continue
        checkout_record = session.exec(
            select(CommerceProviderCheckoutRecord).where(
                CommerceProviderCheckoutRecord.provider_key == "mercadopago",
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
                CommercialPaymentRecord.provider == "mercadopago",
                CommercialPaymentRecord.provider_payment_id == provider_resource_id,
            )
        ).first()
        if payment is not None:
            return session.get(CommercialOrderRecord, payment.order_id)
    return None


def _transaction_payments(data: dict[str, Any]) -> list[dict[str, Any]]:
    transactions = data.get("transactions")
    if not isinstance(transactions, dict):
        return []
    payments = transactions.get("payments")
    if not isinstance(payments, list):
        return []
    return [payment for payment in payments if isinstance(payment, dict)]


def _has_refunds_or_chargebacks(data: dict[str, Any]) -> bool:
    transactions = data.get("transactions")
    if not isinstance(transactions, dict):
        return False
    for key in ("refunds", "chargebacks"):
        values = transactions.get(key)
        if isinstance(values, list) and values:
            return True
    return False


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
            CommerceProviderConfigRecord.provider_key == "mercadopago",
            CommerceProviderConfigRecord.environment == environment,
        )
    ).first()
    return config.workspace_id if config is not None else None


def _api_base_url_for_workspace(session: Session, *, workspace_id: UUID | None, environment: str) -> str:
    if workspace_id is not None:
        config = session.exec(
            select(CommerceProviderConfigRecord).where(
                CommerceProviderConfigRecord.workspace_id == workspace_id,
                CommerceProviderConfigRecord.provider_key == "mercadopago",
                CommerceProviderConfigRecord.environment == environment,
            )
        ).first()
        if config is not None and config.api_base_url:
            return config.api_base_url.rstrip("/")
    configured = get_settings().mercadopago_api_base_url.strip()
    return configured.rstrip("/") if configured else "https://api.mercadopago.com"


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
