from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import httpx
from sqlalchemy import func
from sqlmodel import Session, select

from app.core.config import get_settings
from app.models import (
    HotmartProductMappingRecord,
    HotmartPromotionCreateRequest,
    HotmartPromotionDeleteResponse,
    HotmartPromotionMetricsResponse,
    HotmartPromotionRecord,
    HotmartPromotionResponse,
    utc_now,
)
from app.services.commerce_service import record_commercial_event
from app.services.hotmart.auth import (
    HotmartAuthClient,
    HotmartAuthError,
    default_hotmart_api_base_url,
    normalize_hotmart_environment,
)
from app.services.hotmart.redaction import redact_payload
from app.services.hotmart.secrets import build_hotmart_status, load_hotmart_credentials


class HotmartCouponError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        http_status: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        self.payload = redact_payload(payload or {})


@dataclass(frozen=True)
class HotmartCouponApiResult:
    http_status: int
    payload_redacted: dict[str, Any]


class HotmartCouponApiClient:
    def __init__(
        self,
        *,
        api_base_url: str,
        create_path_template: str = "",
        list_path_template: str = "",
        delete_path_template: str = "",
        timeout_seconds: int = 30,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        settings = get_settings()
        self.api_base_url = api_base_url.rstrip("/") or default_hotmart_api_base_url("sandbox")
        self.create_path_template = create_path_template or settings.hotmart_coupon_create_path_template
        self.list_path_template = list_path_template or settings.hotmart_coupon_list_path_template
        self.delete_path_template = delete_path_template or settings.hotmart_coupon_delete_path_template
        self.timeout_seconds = max(1, timeout_seconds)
        self.transport = transport

    def _url(self, path_template: str, **path_values: str) -> str:
        path = path_template.format(**{key: str(value).strip() for key, value in path_values.items()})
        normalized_path = path if path.startswith("/") else f"/{path}"
        return f"{self.api_base_url}{normalized_path}"

    def create_coupon(
        self,
        *,
        access_token: str,
        product_id: str,
        payload: dict[str, Any],
    ) -> HotmartCouponApiResult:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout_seconds, transport=self.transport) as client:
            response = client.post(
                self._url(self.create_path_template, product_id=product_id),
                headers=headers,
                json=payload,
            )
        response_payload = _read_response_payload(response)
        redacted = redact_payload(response_payload)

        if response.status_code >= 400:
            raise HotmartCouponError(
                "coupon_create_rejected",
                "Hotmart rejected the coupon creation request.",
                http_status=response.status_code,
                payload=redacted,
            )
        if response.status_code not in {200, 201, 202, 204}:
            raise HotmartCouponError(
                "unexpected_coupon_create_status",
                f"Hotmart returned unexpected HTTP status {response.status_code} while creating a coupon.",
                http_status=response.status_code,
                payload=redacted,
            )

        return HotmartCouponApiResult(http_status=response.status_code, payload_redacted=redacted)

    def list_coupons(
        self,
        *,
        access_token: str,
        product_id: str,
        code: str = "",
        page_token: str = "",
    ) -> dict[str, Any]:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}",
        }
        params: dict[str, str] = {}
        if code.strip():
            params["code"] = code.strip()
        if page_token.strip():
            params["page_token"] = page_token.strip()

        with httpx.Client(timeout=self.timeout_seconds, transport=self.transport) as client:
            response = client.get(
                self._url(self.list_path_template, product_id=product_id),
                headers=headers,
                params=params,
            )
        response_payload = _read_response_payload(response)
        redacted = redact_payload(response_payload)
        if response.status_code >= 400:
            raise HotmartCouponError(
                "coupon_list_rejected",
                "Hotmart rejected the coupon listing request.",
                http_status=response.status_code,
                payload=redacted,
            )
        return redacted

    def delete_coupon(self, *, access_token: str, coupon_id: str) -> HotmartCouponApiResult:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}",
        }
        with httpx.Client(timeout=self.timeout_seconds, transport=self.transport) as client:
            response = client.delete(
                self._url(self.delete_path_template, coupon_id=coupon_id),
                headers=headers,
            )
        response_payload = _read_response_payload(response)
        redacted = redact_payload(response_payload)
        if response.status_code >= 400:
            raise HotmartCouponError(
                "coupon_delete_rejected",
                "Hotmart rejected the coupon deletion request.",
                http_status=response.status_code,
                payload=redacted,
            )
        if response.status_code not in {200, 202, 204}:
            raise HotmartCouponError(
                "unexpected_coupon_delete_status",
                f"Hotmart returned unexpected HTTP status {response.status_code} while deleting a coupon.",
                http_status=response.status_code,
                payload=redacted,
            )
        return HotmartCouponApiResult(http_status=response.status_code, payload_redacted=redacted)


def _read_response_payload(response: httpx.Response) -> dict[str, Any]:
    if not response.text:
        return {}
    try:
        payload = response.json()
    except ValueError:
        return {"raw": response.text[:500]}
    return payload if isinstance(payload, dict) else {"payload": payload}


def _normalize_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def _datetime_to_epoch_ms(value: datetime) -> int:
    if value.tzinfo is None:
        aware = value.replace(tzinfo=UTC)
    else:
        aware = value.astimezone(UTC)
    return int(aware.timestamp() * 1000)


def _normalize_coupon_code(value: str) -> str:
    return value.strip().upper()


def _normalize_offer_codes(values: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        candidate = str(value or "").strip()
        if not candidate or candidate in seen:
            continue
        normalized.append(candidate)
        seen.add(candidate)
    return normalized


def _coupon_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("items", "results", "data", "coupons", "content"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return [payload] if _extract_coupon_code(payload) or _extract_coupon_id(payload) else []


def _extract_coupon_item(payload: Any, coupon_code: str) -> dict[str, Any]:
    expected = coupon_code.strip().upper()
    for item in _coupon_items(payload):
        item_code = _extract_coupon_code(item).upper()
        if expected and item_code == expected:
            return item
    items = _coupon_items(payload)
    return items[0] if len(items) == 1 else {}


def _extract_coupon_code(payload: dict[str, Any]) -> str:
    for key in ("coupon_code", "code", "couponCode"):
        value = payload.get(key)
        if str(value or "").strip():
            return str(value).strip()
    return ""


def _extract_coupon_id(payload: dict[str, Any]) -> str:
    for key in ("id", "coupon_id", "couponId"):
        value = payload.get(key)
        if str(value or "").strip():
            return str(value).strip()
    return ""


def _promotion_status_from_remote(
    *,
    item: dict[str, Any],
    starts_at: datetime | None,
    ends_at: datetime | None,
) -> str:
    now = utc_now()
    raw_status = str(item.get("status") or "").strip().lower()
    active_value = item.get("active")

    if raw_status in {"deleted", "removed"}:
        return "deleted"
    if raw_status in {"expired", "invalid"}:
        return "expired"
    if starts_at and starts_at > now:
        return "scheduled"
    if ends_at and ends_at < now:
        return "expired"
    if active_value is False:
        return "draft"
    if active_value is True:
        return "active"
    if raw_status in {"active", "valid", "enabled", "published"}:
        return "active"
    return "active" if item else "scheduled" if starts_at and starts_at > now else "active"


def _local_promotion_status(*, publish: bool, starts_at: datetime | None, ends_at: datetime | None) -> str:
    now = utc_now()
    if ends_at and ends_at < now:
        return "expired"
    if starts_at and starts_at > now:
        return "scheduled"
    return "active" if publish else "draft"


def _discount_fraction(discount_percent: float) -> float:
    return round(discount_percent / 100.0, 6)


def _validate_coupon_request(
    payload: HotmartPromotionCreateRequest,
    *,
    coupon_code: str,
    starts_at: datetime | None,
    ends_at: datetime | None,
) -> None:
    if not payload.internal_product_key.strip():
        raise ValueError("internal_product_key is required.")
    if not coupon_code:
        raise ValueError("coupon_code is required.")
    if len(coupon_code) > 25:
        raise ValueError("Hotmart coupon code must have 25 characters or fewer.")
    if payload.discount_percent <= 0 or payload.discount_percent >= 99:
        raise ValueError("discount_percent must be greater than 0 and lower than 99.")
    if starts_at and ends_at and starts_at >= ends_at:
        raise ValueError("Promotion end date must be after start date.")


def _get_active_mapping(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str,
    product_key: str,
) -> HotmartProductMappingRecord:
    mapping = session.exec(
        select(HotmartProductMappingRecord).where(
            HotmartProductMappingRecord.workspace_id == workspace_id,
            HotmartProductMappingRecord.environment == environment,
            HotmartProductMappingRecord.internal_product_key == product_key,
            HotmartProductMappingRecord.is_active == True,  # noqa: E712
        )
    ).first()
    if mapping is None:
        raise ValueError(f"Hotmart mapping is required for product {product_key}.")
    if not mapping.hotmart_product_id.strip():
        raise ValueError(f"Hotmart coupon creation for {product_key} requires Hotmart product id.")
    return mapping


def _validate_not_subscription(mapping: HotmartProductMappingRecord) -> None:
    billing_mode = mapping.billing_mode.strip().lower()
    if "subscription" in billing_mode or "recurr" in billing_mode:
        raise ValueError(
            "Hotmart coupon creation is not supported for subscription products. "
            "Use an internal platform discount or choose a one-time Hotmart mapping."
        )


def _find_existing_promotion(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str,
    coupon_code: str,
) -> HotmartPromotionRecord | None:
    return session.exec(
        select(HotmartPromotionRecord).where(
            HotmartPromotionRecord.workspace_id == workspace_id,
            HotmartPromotionRecord.environment == environment,
            HotmartPromotionRecord.coupon_code == coupon_code,
        )
    ).first()


def _get_or_create_promotion(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str,
    coupon_code: str,
) -> HotmartPromotionRecord:
    record = _find_existing_promotion(
        session,
        workspace_id=workspace_id,
        environment=environment,
        coupon_code=coupon_code,
    )
    if record is not None:
        return record
    return HotmartPromotionRecord(
        workspace_id=workspace_id,
        environment=environment,
        coupon_code=coupon_code,
        discount_origin="provider_coupon",
        discount_type="percent",
    )


def _build_coupon_payload(
    *,
    payload: HotmartPromotionCreateRequest,
    coupon_code: str,
    offer_codes: list[str],
    starts_at: datetime | None,
    ends_at: datetime | None,
) -> dict[str, Any]:
    request_payload: dict[str, Any] = {
        "code": coupon_code,
        "discount": _discount_fraction(payload.discount_percent),
    }
    if starts_at is not None:
        request_payload["start_date"] = _datetime_to_epoch_ms(starts_at)
    if ends_at is not None:
        request_payload["end_date"] = _datetime_to_epoch_ms(ends_at)
    if payload.affiliate_id.strip():
        request_payload["affiliate"] = payload.affiliate_id.strip()
    if offer_codes:
        request_payload["offer_ids"] = offer_codes
    return request_payload


def serialize_hotmart_promotion(record: HotmartPromotionRecord) -> HotmartPromotionResponse:
    return HotmartPromotionResponse(
        id=record.id,
        workspace_id=record.workspace_id,
        environment=record.environment,  # type: ignore[arg-type]
        internal_campaign_key=record.internal_campaign_key,
        internal_product_key=record.internal_product_key,
        hotmart_product_id=record.hotmart_product_id,
        offer_codes=record.offer_codes,
        coupon_id=record.coupon_id,
        coupon_code=record.coupon_code,
        discount_percent=record.discount_percent,
        discount_origin=record.discount_origin,
        discount_type=record.discount_type,
        discount_amount_cents=record.discount_amount_cents,
        starts_at=record.starts_at,
        ends_at=record.ends_at,
        status=record.status,
        published_at=record.published_at,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def list_hotmart_promotions(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str = "sandbox",
) -> list[HotmartPromotionResponse]:
    env = normalize_hotmart_environment(environment)
    rows = session.exec(
        select(HotmartPromotionRecord)
        .where(
            HotmartPromotionRecord.workspace_id == workspace_id,
            HotmartPromotionRecord.environment == env,
        )
        .order_by(HotmartPromotionRecord.updated_at.desc())
    ).all()
    return [serialize_hotmart_promotion(row) for row in rows]


def build_hotmart_promotion_metrics(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str = "sandbox",
) -> HotmartPromotionMetricsResponse:
    env = normalize_hotmart_environment(environment)
    rows = session.exec(
        select(
            HotmartPromotionRecord.status,
            HotmartPromotionRecord.discount_origin,
            func.count(),
        )
        .where(
            HotmartPromotionRecord.workspace_id == workspace_id,
            HotmartPromotionRecord.environment == env,
        )
        .group_by(HotmartPromotionRecord.status, HotmartPromotionRecord.discount_origin)
    ).all()
    statuses = {status: 0 for status in ("active", "scheduled", "expired", "deleted", "sync_error")}
    provider_coupon_count = 0
    internal_upgrade_credit_count = 0
    total = 0
    for status_value, discount_origin, count in rows:
        normalized_count = int(count)
        total += normalized_count
        if status_value in statuses:
            statuses[status_value] += normalized_count
        if discount_origin == "provider_coupon":
            provider_coupon_count += normalized_count
        if discount_origin == "internal_upgrade_credit":
            internal_upgrade_credit_count += normalized_count
    return HotmartPromotionMetricsResponse(
        total=total,
        active=statuses["active"],
        scheduled=statuses["scheduled"],
        expired=statuses["expired"],
        deleted=statuses["deleted"],
        sync_error=statuses["sync_error"],
        provider_coupon_count=provider_coupon_count,
        internal_upgrade_credit_count=internal_upgrade_credit_count,
    )


def create_hotmart_coupon_promotion(
    session: Session,
    *,
    workspace_id: UUID,
    payload: HotmartPromotionCreateRequest,
    actor_user_id: UUID | None = None,
    transport: httpx.BaseTransport | None = None,
) -> HotmartPromotionResponse:
    env = normalize_hotmart_environment(payload.environment)
    coupon_code = _normalize_coupon_code(payload.coupon_code)
    starts_at = _normalize_datetime(payload.starts_at)
    ends_at = _normalize_datetime(payload.ends_at)
    _validate_coupon_request(payload, coupon_code=coupon_code, starts_at=starts_at, ends_at=ends_at)

    product_key = payload.internal_product_key.strip()
    mapping = _get_active_mapping(session, workspace_id=workspace_id, environment=env, product_key=product_key)
    _validate_not_subscription(mapping)
    offer_codes = _normalize_offer_codes(payload.offer_codes or ([mapping.offer_code] if mapping.offer_code else []))
    local_status = _local_promotion_status(publish=payload.publish, starts_at=starts_at, ends_at=ends_at)
    promotion = _get_or_create_promotion(
        session,
        workspace_id=workspace_id,
        environment=env,
        coupon_code=coupon_code,
    )
    promotion.internal_campaign_key = payload.internal_campaign_key.strip()
    promotion.internal_product_key = product_key
    promotion.hotmart_product_id = mapping.hotmart_product_id.strip()
    promotion.offer_codes = offer_codes
    promotion.discount_percent = payload.discount_percent
    promotion.discount_origin = "provider_coupon"
    promotion.discount_type = "percent"
    promotion.starts_at = starts_at
    promotion.ends_at = ends_at
    promotion.created_by_user_id = actor_user_id
    promotion.updated_at = utc_now()

    request_payload = _build_coupon_payload(
        payload=payload,
        coupon_code=coupon_code,
        offer_codes=offer_codes,
        starts_at=starts_at,
        ends_at=ends_at,
    )
    promotion.metadata_payload = {
        **payload.metadata,
        "hotmart_request_redacted": redact_payload(request_payload),
        "provider_contract": "hotmart-coupon.v1",
    }

    if not payload.publish:
        promotion.status = "draft"
        promotion.published_at = None
        session.add(promotion)
        session.flush()
        return serialize_hotmart_promotion(promotion)

    status = build_hotmart_status(session, workspace_id=workspace_id, environment=env)
    credentials = load_hotmart_credentials(session, workspace_id=workspace_id, environment=env)
    if credentials is None:
        raise ValueError("Hotmart OAuth credentials are required before creating coupons.")

    session.add(promotion)
    session.flush()

    api_client = HotmartCouponApiClient(
        api_base_url=status.api_base_url or default_hotmart_api_base_url(env),
        timeout_seconds=get_settings().hotmart_request_timeout_seconds,
        transport=transport,
    )
    try:
        token = HotmartAuthClient(
            environment=env,
            auth_base_url=status.auth_base_url,
            timeout_seconds=get_settings().hotmart_request_timeout_seconds,
            transport=transport,
        ).fetch_access_token(credentials)
        create_result = api_client.create_coupon(
            access_token=token.access_token,
            product_id=promotion.hotmart_product_id,
            payload=request_payload,
        )
        list_payload = api_client.list_coupons(
            access_token=token.access_token,
            product_id=promotion.hotmart_product_id,
            code=coupon_code,
        )
    except HotmartAuthError as exc:
        promotion.status = "sync_error"
        promotion.metadata_payload = {
            **promotion.metadata_payload,
            "last_error_code": exc.code,
            "last_error_http_status": exc.http_status,
            "last_error_payload_redacted": exc.payload,
        }
        promotion.updated_at = utc_now()
        session.add(promotion)
        session.flush()
        raise HotmartCouponError(
            "coupon_auth_failed",
            "Hotmart OAuth failed while creating the coupon.",
            http_status=exc.http_status,
            payload=exc.payload,
        ) from exc
    except HotmartCouponError as exc:
        promotion.status = "sync_error"
        promotion.metadata_payload = {
            **promotion.metadata_payload,
            "last_error_code": exc.code,
            "last_error_http_status": exc.http_status,
            "last_error_payload_redacted": exc.payload,
        }
        promotion.updated_at = utc_now()
        session.add(promotion)
        session.flush()
        raise

    coupon_item = _extract_coupon_item(list_payload, coupon_code) or _extract_coupon_item(
        create_result.payload_redacted,
        coupon_code,
    )
    promotion.coupon_id = _extract_coupon_id(coupon_item) or promotion.coupon_id
    promotion.coupon_code = _extract_coupon_code(coupon_item) or coupon_code
    promotion.status = _promotion_status_from_remote(item=coupon_item, starts_at=starts_at, ends_at=ends_at) or local_status
    promotion.published_at = promotion.published_at or utc_now()
    promotion.metadata_payload = {
        **promotion.metadata_payload,
        "hotmart_create_http_status": create_result.http_status,
        "hotmart_create_response_redacted": create_result.payload_redacted,
        "hotmart_list_response_redacted": list_payload,
    }
    promotion.updated_at = utc_now()
    session.add(promotion)
    record_commercial_event(
        session,
        workspace_id=workspace_id,
        session_id=None,
        user_id=actor_user_id,
        event_key="hotmart_coupon_created",
        product_key=product_key,
        source="hotmart_coupons",
        metadata={"coupon_code": promotion.coupon_code, "coupon_id": promotion.coupon_id, "status": promotion.status},
        correlation_id=promotion.coupon_code,
    )
    session.flush()
    return serialize_hotmart_promotion(promotion)


def _resolve_coupon_record(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str,
    coupon_ref: str,
) -> HotmartPromotionRecord:
    record: HotmartPromotionRecord | None = None
    try:
        record = session.get(HotmartPromotionRecord, UUID(coupon_ref))
    except ValueError:
        record = None
    if record is not None and record.workspace_id == workspace_id and record.environment == environment:
        return record

    stripped_ref = coupon_ref.strip()
    if stripped_ref:
        record = session.exec(
            select(HotmartPromotionRecord).where(
                HotmartPromotionRecord.workspace_id == workspace_id,
                HotmartPromotionRecord.environment == environment,
                HotmartPromotionRecord.coupon_id == stripped_ref,
            )
        ).first()
        if record is not None:
            return record
        record = session.exec(
            select(HotmartPromotionRecord).where(
                HotmartPromotionRecord.workspace_id == workspace_id,
                HotmartPromotionRecord.environment == environment,
                HotmartPromotionRecord.coupon_code == _normalize_coupon_code(stripped_ref),
            )
        ).first()
        if record is not None:
            return record
    raise ValueError("Hotmart coupon was not found in this workspace.")


def delete_hotmart_coupon_promotion(
    session: Session,
    *,
    workspace_id: UUID,
    coupon_ref: str,
    environment: str = "sandbox",
    actor_user_id: UUID | None = None,
    transport: httpx.BaseTransport | None = None,
) -> HotmartPromotionDeleteResponse:
    env = normalize_hotmart_environment(environment)
    promotion = _resolve_coupon_record(
        session,
        workspace_id=workspace_id,
        environment=env,
        coupon_ref=coupon_ref,
    )
    deleted_remote = False
    delete_payload: dict[str, Any] = {}

    if promotion.coupon_id.strip():
        status = build_hotmart_status(session, workspace_id=workspace_id, environment=env)
        credentials = load_hotmart_credentials(session, workspace_id=workspace_id, environment=env)
        if credentials is None:
            raise ValueError("Hotmart OAuth credentials are required before deleting coupons.")
        try:
            token = HotmartAuthClient(
                environment=env,
                auth_base_url=status.auth_base_url,
                timeout_seconds=get_settings().hotmart_request_timeout_seconds,
                transport=transport,
            ).fetch_access_token(credentials)
            delete_result = HotmartCouponApiClient(
                api_base_url=status.api_base_url or default_hotmart_api_base_url(env),
                timeout_seconds=get_settings().hotmart_request_timeout_seconds,
                transport=transport,
            ).delete_coupon(access_token=token.access_token, coupon_id=promotion.coupon_id)
            deleted_remote = True
            delete_payload = delete_result.payload_redacted
        except HotmartAuthError as exc:
            raise HotmartCouponError(
                "coupon_auth_failed",
                "Hotmart OAuth failed while deleting the coupon.",
                http_status=exc.http_status,
                payload=exc.payload,
            ) from exc

    promotion.status = "deleted"
    promotion.updated_at = utc_now()
    promotion.metadata_payload = {
        **promotion.metadata_payload,
        "hotmart_delete_response_redacted": delete_payload,
        "hotmart_deleted_remote": deleted_remote,
    }
    session.add(promotion)
    record_commercial_event(
        session,
        workspace_id=workspace_id,
        session_id=None,
        user_id=actor_user_id,
        event_key="hotmart_coupon_deleted",
        product_key=promotion.internal_product_key,
        source="hotmart_coupons",
        metadata={
            "coupon_code": promotion.coupon_code,
            "coupon_id": promotion.coupon_id,
            "deleted_remote": deleted_remote,
        },
        correlation_id=promotion.coupon_code or promotion.coupon_id,
    )
    session.flush()
    return HotmartPromotionDeleteResponse(
        id=promotion.id,
        coupon_id=promotion.coupon_id,
        coupon_code=promotion.coupon_code,
        deleted_remote=deleted_remote,
        message="Coupon deleted in Hotmart and marked deleted locally."
        if deleted_remote
        else "Coupon marked deleted locally because no Hotmart coupon id was stored.",
    )
