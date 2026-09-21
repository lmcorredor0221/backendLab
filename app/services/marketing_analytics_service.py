from __future__ import annotations

import re
from datetime import timedelta
from typing import Any
from uuid import UUID

import httpx
from sqlmodel import Session, select

from app.core.config import get_settings
from app.models import (
    CommercialEventRecord,
    CommercialOrderLineRecord,
    CommercialOrderRecord,
    MarketingAnalyticsOutboxRecord,
    SessionRecord,
    utc_now,
)

SAFE_TEXT = re.compile(r"^[a-zA-Z0-9 _.\-:/|+%]{1,180}$")
ALLOWED_ATTRIBUTION_KEYS = {"utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term", "gclid", "gbraid", "wbraid"}
PURCHASE_EVENT_KEYS = {"mercadopago_order_processed", "payment_confirmed", "hotmart_payment_approved", "hotmart_external_sale_claimed"}
REFUND_EVENT_KEYS = {"mercadopago_order_revoked", "payment_refunded"}


def sanitize_marketing_context(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    consent = raw.get("consent") if isinstance(raw.get("consent"), dict) else {}
    sanitized: dict[str, Any] = {
        "consent": {
            "analytics_storage": _consent_value(consent.get("analytics_storage")),
            "ad_storage": _consent_value(consent.get("ad_storage")),
            "ad_user_data": _consent_value(consent.get("ad_user_data")),
            "ad_personalization": _consent_value(consent.get("ad_personalization")),
            "version": _safe_int(consent.get("version"), fallback=1),
        }
    }
    decided_at = _safe_string(consent.get("decided_at"), max_length=40)
    if decided_at:
        sanitized["consent"]["decided_at"] = decided_at

    attribution = raw.get("attribution") if isinstance(raw.get("attribution"), dict) else {}
    attribution_snapshot = _sanitize_attribution(attribution)
    if attribution_snapshot:
        sanitized["attribution"] = attribution_snapshot

    ga_client_id = _safe_string(raw.get("ga_client_id"), max_length=80, pattern=re.compile(r"^[0-9]+\.[0-9]+$"))
    ga_session_id = _safe_string(raw.get("ga_session_id"), max_length=40, pattern=re.compile(r"^[0-9]+$"))
    if ga_client_id:
        sanitized["ga_client_id"] = ga_client_id
    if ga_session_id:
        sanitized["ga_session_id"] = ga_session_id
    return sanitized


def enqueue_blueprint_completed(
    db: Session,
    *,
    record: SessionRecord,
    actor_user_id: UUID | None,
    correlation_id: str,
) -> MarketingAnalyticsOutboxRecord | None:
    context = sanitize_marketing_context(record.marketing_context)
    if not _has_analytics_consent(context):
        return None
    client_id = str(context.get("ga_client_id") or "")
    if not client_id:
        return None
    payload = _ga_payload(
        client_id=client_id,
        context=context,
        event_name="blueprint_completed",
        params={
            "product_key": "blueprint",
            "engagement_time_msec": 1,
        },
    )
    return _enqueue(
        db,
        event_name="blueprint_completed",
        business_key=f"blueprint_completed:{record.id}",
        source="journey_state",
        workspace_id=record.workspace_id,
        session_id=record.id,
        user_id=actor_user_id or record.user_id,
        order_id=None,
        payload=payload,
    )


def enqueue_from_commercial_event(db: Session, event: CommercialEventRecord) -> MarketingAnalyticsOutboxRecord | None:
    if event.event_key in PURCHASE_EVENT_KEYS:
        return _enqueue_purchase(db, event)
    if event.event_key in REFUND_EVENT_KEYS:
        return _enqueue_refund(db, event)
    return None


def process_pending_marketing_events(db: Session, *, limit: int | None = None) -> dict[str, int]:
    settings = get_settings()
    if not settings.marketing_analytics_enabled:
        return {"processed": 0, "sent": 0, "failed": 0, "skipped": 0}
    batch_size = max(1, min(limit or settings.marketing_analytics_batch_size, 100))
    now = utc_now()
    records = db.exec(
        select(MarketingAnalyticsOutboxRecord)
        .where(
            MarketingAnalyticsOutboxRecord.status == "pending",
            MarketingAnalyticsOutboxRecord.next_attempt_at <= now,
        )
        .order_by(MarketingAnalyticsOutboxRecord.created_at.asc())
        .limit(batch_size)
    ).all()
    summary = {"processed": 0, "sent": 0, "failed": 0, "skipped": 0}
    for record in records:
        summary["processed"] += 1
        if not settings.ga4_measurement_id or not settings.ga4_api_secret:
            record.status = "failed"
            record.error_code = "missing_ga4_config"
            record.error_message = "GA4 measurement id and api secret are required."
            record.updated_at = utc_now()
            db.add(record)
            summary["failed"] += 1
            continue
        try:
            _send_ga4_payload(record.payload, settings=settings)
        except Exception as exc:  # pragma: no cover - exercised through tests with monkeypatch.
            record.attempts += 1
            record.last_attempt_at = utc_now()
            if record.attempts >= settings.marketing_analytics_max_attempts:
                record.status = "failed"
            record.error_code = exc.__class__.__name__[:80]
            record.error_message = _safe_string(str(exc), max_length=240)
            record.next_attempt_at = utc_now() + timedelta(minutes=min(60, 2 ** max(0, record.attempts)))
            record.updated_at = utc_now()
            db.add(record)
            summary["failed"] += 1
            continue
        record.status = "sent"
        record.sent_at = utc_now()
        record.last_attempt_at = record.sent_at
        record.updated_at = record.sent_at
        record.error_code = ""
        record.error_message = ""
        db.add(record)
        summary["sent"] += 1
    db.commit()
    return summary


def _enqueue_purchase(db: Session, event: CommercialEventRecord) -> MarketingAnalyticsOutboxRecord | None:
    order = _order_from_event(db, event)
    if order is None or order.provider == "sandbox":
        return None
    context = sanitize_marketing_context(order.marketing_context or _session_context(db, order.session_id))
    if not _has_analytics_consent(context) or not context.get("ga_client_id"):
        return None
    line = _order_line(db, order)
    transaction_id = _transaction_id(order)
    payload = _ga_payload(
        client_id=str(context["ga_client_id"]),
        context=context,
        event_name="purchase",
        params={
            "transaction_id": transaction_id,
            "currency": (event.currency or order.currency or "USD").upper(),
            "value": round(max(0, event.revenue_cents or order.total_cents) / 100, 2),
            "items": [_item_payload(line, order)],
            "engagement_time_msec": 1,
        },
    )
    return _enqueue(
        db,
        event_name="purchase",
        business_key=f"purchase:{transaction_id}",
        source=event.source,
        workspace_id=event.workspace_id,
        session_id=event.session_id,
        user_id=event.user_id,
        order_id=order.id,
        payload=payload,
    )


def _enqueue_refund(db: Session, event: CommercialEventRecord) -> MarketingAnalyticsOutboxRecord | None:
    order = _order_from_event(db, event)
    if order is None or order.provider == "sandbox":
        return None
    context = sanitize_marketing_context(order.marketing_context or _session_context(db, order.session_id))
    if not _has_analytics_consent(context) or not context.get("ga_client_id"):
        return None
    line = _order_line(db, order)
    amount_cents = _safe_int(event.metadata_payload.get("refund_amount_cents"), fallback=0) or event.revenue_cents or order.total_cents
    payload = _ga_payload(
        client_id=str(context["ga_client_id"]),
        context=context,
        event_name="refund",
        params={
            "transaction_id": _transaction_id(order),
            "currency": (event.currency or order.currency or "USD").upper(),
            "value": round(max(0, amount_cents) / 100, 2),
            "items": [_item_payload(line, order)],
            "engagement_time_msec": 1,
        },
    )
    return _enqueue(
        db,
        event_name="refund",
        business_key=f"refund:{_transaction_id(order)}:{event.correlation_id or event.id}",
        source=event.source,
        workspace_id=event.workspace_id,
        session_id=event.session_id,
        user_id=event.user_id,
        order_id=order.id,
        payload=payload,
    )


def _enqueue(
    db: Session,
    *,
    event_name: str,
    business_key: str,
    source: str,
    workspace_id: UUID | None,
    session_id: UUID | None,
    user_id: UUID | None,
    order_id: UUID | None,
    payload: dict[str, Any],
) -> MarketingAnalyticsOutboxRecord | None:
    existing = db.exec(
        select(MarketingAnalyticsOutboxRecord).where(MarketingAnalyticsOutboxRecord.business_key == business_key)
    ).first()
    if existing is not None:
        return existing
    record = MarketingAnalyticsOutboxRecord(
        workspace_id=workspace_id,
        session_id=session_id,
        user_id=user_id,
        order_id=order_id,
        event_name=event_name,
        business_key=business_key[:180],
        source=source[:120],
        payload=payload,
    )
    db.add(record)
    db.flush()
    return record


def _send_ga4_payload(payload: dict[str, Any], *, settings) -> None:
    endpoint = settings.ga4_mp_endpoint.rstrip("/")
    params = {"measurement_id": settings.ga4_measurement_id, "api_secret": settings.ga4_api_secret}
    with httpx.Client(timeout=10) as client:
        response = client.post(endpoint, params=params, json=payload)
    response.raise_for_status()


def _ga_payload(*, client_id: str, context: dict[str, Any], event_name: str, params: dict[str, Any]) -> dict[str, Any]:
    event_params = {**_traffic_params(context), **params}
    ga_session_id = context.get("ga_session_id")
    if ga_session_id:
        event_params["session_id"] = int(ga_session_id)
    return {"client_id": client_id, "events": [{"name": event_name, "params": event_params}]}


def _traffic_params(context: dict[str, Any]) -> dict[str, str]:
    touch = {}
    attribution = context.get("attribution") if isinstance(context.get("attribution"), dict) else {}
    for key in ("last_touch", "first_touch"):
        candidate = attribution.get(key)
        if isinstance(candidate, dict):
            touch = candidate
            break
    mapped: dict[str, str] = {}
    for key, value in touch.items():
        if key in ALLOWED_ATTRIBUTION_KEYS:
            safe = _safe_string(value)
            if safe:
                mapped[key] = safe
    return mapped


def _sanitize_attribution(raw: dict[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for date_key in ("captured_at", "expires_at"):
        safe = _safe_string(raw.get(date_key), max_length=40)
        if safe:
            snapshot[date_key] = safe
    for touch_key in ("first_touch", "last_touch"):
        value = raw.get(touch_key)
        if not isinstance(value, dict):
            continue
        clean = {key: _safe_string(item) for key, item in value.items() if key in ALLOWED_ATTRIBUTION_KEYS}
        clean = {key: item for key, item in clean.items() if item}
        if clean:
            snapshot[touch_key] = clean
    return snapshot


def _has_analytics_consent(context: dict[str, Any]) -> bool:
    consent = context.get("consent")
    return isinstance(consent, dict) and consent.get("analytics_storage") == "granted"


def _order_from_event(db: Session, event: CommercialEventRecord) -> CommercialOrderRecord | None:
    order_id = _safe_string(event.metadata_payload.get("order_id"), max_length=80)
    if not order_id:
        return None
    try:
        return db.get(CommercialOrderRecord, UUID(order_id))
    except ValueError:
        return None


def _order_line(db: Session, order: CommercialOrderRecord) -> CommercialOrderLineRecord | None:
    return db.exec(select(CommercialOrderLineRecord).where(CommercialOrderLineRecord.order_id == order.id)).first()


def _session_context(db: Session, session_id: UUID | None) -> dict[str, Any]:
    if session_id is None:
        return {}
    record = db.get(SessionRecord, session_id)
    return dict(record.marketing_context or {}) if record is not None else {}


def _transaction_id(order: CommercialOrderRecord) -> str:
    provider_id = _safe_string(order.metadata_payload.get("provider_checkout_id"), max_length=80)
    return provider_id or str(order.id)


def _item_payload(line: CommercialOrderLineRecord | None, order: CommercialOrderRecord) -> dict[str, Any]:
    product_key = line.product_key if line is not None else _safe_string(order.metadata_payload.get("product_key"))
    return {
        "item_id": product_key or "unknown",
        "item_name": product_key or "unknown",
        "price": round(max(0, (line.total_amount_cents if line is not None else order.total_cents)) / 100, 2),
        "quantity": line.quantity if line is not None else 1,
    }


def _consent_value(value: Any) -> str:
    return "granted" if str(value).strip().lower() == "granted" else "denied"


def _safe_int(value: Any, *, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _safe_string(value: Any, *, max_length: int = 160, pattern: re.Pattern[str] = SAFE_TEXT) -> str:
    if value is None:
        return ""
    candidate = str(value).strip()[:max_length]
    return candidate if candidate and pattern.match(candidate) else ""
