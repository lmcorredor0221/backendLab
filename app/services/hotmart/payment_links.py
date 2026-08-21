from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import httpx
from sqlmodel import Session, select

from app.core.config import get_settings
from app.models import (
    CommercialEventRecord,
    CommercialOrderLineRecord,
    CommercialOrderRecord,
    CommercialOrderStatus,
    HotmartPaymentLinkCreateRequest,
    HotmartPaymentLinkRecord,
    HotmartPaymentLinkResponse,
    HotmartProductMappingRecord,
    HotmartProductMappingResponse,
    HotmartProductMappingUpsertRequest,
    ProductPriceRecord,
    utc_now,
)
from app.services.commerce_service import get_price, record_commercial_event
from app.services.hotmart.auth import HotmartAuthClient, default_hotmart_api_base_url, normalize_hotmart_environment
from app.services.hotmart.redaction import redact_payload
from app.services.hotmart.secrets import build_hotmart_status, load_hotmart_credentials


class HotmartPaymentLinkError(RuntimeError):
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
class HotmartPaymentLinkApiResult:
    provider_ref: str
    checkout_url: str
    http_status: int
    payload_redacted: dict[str, Any]


class HotmartPaymentLinkApiClient:
    def __init__(
        self,
        *,
        api_base_url: str,
        create_path: str = "",
        list_path: str = "",
        timeout_seconds: int = 30,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_base_url = api_base_url.rstrip("/") or default_hotmart_api_base_url("sandbox")
        self.create_path = create_path or get_settings().hotmart_payment_link_create_path
        self.list_path = list_path or get_settings().hotmart_payment_link_list_path
        self.timeout_seconds = max(1, timeout_seconds)
        self.transport = transport

    def _url(self, path: str) -> str:
        normalized_path = path if path.startswith("/") else f"/{path}"
        return f"{self.api_base_url}{normalized_path}"

    def create_payment_link(self, *, access_token: str, payload: dict[str, Any]) -> HotmartPaymentLinkApiResult:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}",
        }
        with httpx.Client(timeout=self.timeout_seconds, transport=self.transport) as client:
            response = client.post(self._url(self.create_path), headers=headers, json=payload)

        try:
            response_payload = response.json() if response.text else {}
        except ValueError:
            response_payload = {"raw": response.text[:500]}
        redacted = redact_payload(response_payload if isinstance(response_payload, dict) else {"payload": response_payload})

        if response.status_code >= 400:
            raise HotmartPaymentLinkError(
                "payment_link_rejected",
                "Hotmart rejected the payment link creation request.",
                http_status=response.status_code,
                payload=redacted,
            )

        if response.status_code not in {200, 201, 202}:
            raise HotmartPaymentLinkError(
                "unexpected_payment_link_status",
                f"Hotmart returned unexpected HTTP status {response.status_code}.",
                http_status=response.status_code,
                payload=redacted,
            )

        provider_ref = _extract_provider_ref(response_payload)
        checkout_url = _extract_checkout_url(response_payload)
        if not provider_ref and not checkout_url:
            raise HotmartPaymentLinkError(
                "invalid_payment_link_response",
                "Hotmart payment link response did not include a link identifier or checkout URL.",
                http_status=response.status_code,
                payload=redacted,
            )

        return HotmartPaymentLinkApiResult(
            provider_ref=provider_ref or checkout_url,
            checkout_url=checkout_url,
            http_status=response.status_code,
            payload_redacted=redacted,
        )

    def list_payment_links(self, *, access_token: str) -> dict[str, Any]:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}",
        }
        with httpx.Client(timeout=self.timeout_seconds, transport=self.transport) as client:
            response = client.get(self._url(self.list_path), headers=headers)
        try:
            payload = response.json() if response.text else {}
        except ValueError:
            payload = {"raw": response.text[:500]}
        redacted = redact_payload(payload if isinstance(payload, dict) else {"payload": payload})
        if response.status_code >= 400:
            raise HotmartPaymentLinkError(
                "payment_link_refresh_rejected",
                "Hotmart rejected the payment link refresh request.",
                http_status=response.status_code,
                payload=redacted,
            )
        return redacted


def _extract_provider_ref(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    candidates = [
        payload.get("ucode"),
        payload.get("id"),
        payload.get("code"),
        payload.get("payment_link_ucode"),
        payload.get("payment_link_id"),
    ]
    nested = payload.get("payment_link")
    if isinstance(nested, dict):
        candidates.extend([nested.get("ucode"), nested.get("id"), nested.get("code")])
    for candidate in candidates:
        if str(candidate or "").strip():
            return str(candidate).strip()
    return ""


def _extract_checkout_url(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    candidates = [
        payload.get("url"),
        payload.get("checkout_url"),
        payload.get("payment_url"),
        payload.get("link"),
    ]
    nested = payload.get("payment_link")
    if isinstance(nested, dict):
        candidates.extend([nested.get("url"), nested.get("checkout_url"), nested.get("payment_url")])
    offers = payload.get("offers")
    if isinstance(offers, list):
        for offer in offers:
            if isinstance(offer, dict):
                candidates.extend([offer.get("url"), offer.get("checkout_url")])
    for candidate in candidates:
        if str(candidate or "").strip():
            return str(candidate).strip()
    return ""


def serialize_hotmart_mapping(record: HotmartProductMappingRecord) -> HotmartProductMappingResponse:
    return HotmartProductMappingResponse(
        id=record.id,
        workspace_id=record.workspace_id,
        environment=record.environment,  # type: ignore[arg-type]
        internal_product_key=record.internal_product_key,
        hotmart_product_id=record.hotmart_product_id,
        hotmart_product_ucode=record.hotmart_product_ucode,
        offer_code=record.offer_code,
        plan_code=record.plan_code,
        billing_mode=record.billing_mode,
        currency=record.currency,
        internal_unit_amount_usd_cents=record.internal_unit_amount_usd_cents,
        hotmart_price_strategy=record.hotmart_price_strategy,
        trm_policy=record.trm_policy,
        grants_tier=record.grants_tier,
        entitlement_scope=record.entitlement_scope,
        is_active=record.is_active,
        updated_at=record.updated_at,
    )


def serialize_hotmart_payment_link(record: HotmartPaymentLinkRecord) -> HotmartPaymentLinkResponse:
    return HotmartPaymentLinkResponse(
        id=record.id,
        workspace_id=record.workspace_id,
        order_id=record.order_id,
        internal_product_key=record.internal_product_key,
        hotmart_payment_link_id=record.hotmart_payment_link_id,
        provider_ref=record.provider_ref,
        checkout_url=record.checkout_url,
        activation_status=record.activation_status,
        gross_amount_cents=record.gross_amount_cents,
        discount_amount_cents=record.discount_amount_cents,
        net_amount_cents=record.net_amount_cents,
        currency=record.currency,
        discount_origin=record.discount_origin,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def upsert_hotmart_product_mapping(
    session: Session,
    *,
    workspace_id: UUID,
    payload: HotmartProductMappingUpsertRequest,
) -> HotmartProductMappingResponse:
    env = normalize_hotmart_environment(payload.environment)
    product_key = payload.internal_product_key.strip()
    if not product_key:
        raise ValueError("internal_product_key is required.")
    price: ProductPriceRecord = get_price(session, product_key)
    mapping = session.exec(
        select(HotmartProductMappingRecord).where(
            HotmartProductMappingRecord.workspace_id == workspace_id,
            HotmartProductMappingRecord.environment == env,
            HotmartProductMappingRecord.internal_product_key == product_key,
        )
    ).first()
    if mapping is None:
        mapping = HotmartProductMappingRecord(
            workspace_id=workspace_id,
            environment=env,
            internal_product_key=product_key,
        )
    mapping.hotmart_product_id = payload.hotmart_product_id.strip()
    mapping.hotmart_product_ucode = payload.hotmart_product_ucode.strip()
    mapping.offer_code = payload.offer_code.strip()
    mapping.plan_code = payload.plan_code.strip()
    mapping.billing_mode = payload.billing_mode.strip() or "one_time"
    mapping.currency = payload.currency.strip().upper() or price.currency
    mapping.internal_base_currency = "USD"
    mapping.internal_unit_amount_usd_cents = (
        price.unit_amount_usd_cents if price.unit_amount_usd_cents > 0 else price.unit_amount_cents
    )
    mapping.hotmart_price_strategy = payload.hotmart_price_strategy.strip() or "net_order_amount"
    mapping.trm_policy = payload.trm_policy.strip() or "internal_usd"
    mapping.grants_tier = payload.grants_tier
    mapping.entitlement_scope = payload.entitlement_scope.strip() or "project"
    mapping.is_active = payload.is_active
    mapping.metadata_payload = payload.metadata
    mapping.updated_at = utc_now()
    session.add(mapping)
    session.flush()
    return serialize_hotmart_mapping(mapping)


def list_hotmart_product_mappings(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str = "sandbox",
) -> list[HotmartProductMappingResponse]:
    env = normalize_hotmart_environment(environment)
    rows = session.exec(
        select(HotmartProductMappingRecord)
        .where(
            HotmartProductMappingRecord.workspace_id == workspace_id,
            HotmartProductMappingRecord.environment == env,
        )
        .order_by(HotmartProductMappingRecord.internal_product_key)
    ).all()
    return [serialize_hotmart_mapping(row) for row in rows]


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
    if not (mapping.hotmart_product_id.strip() or mapping.hotmart_product_ucode.strip()):
        raise ValueError(f"Hotmart mapping for {product_key} must include a Hotmart product id or ucode.")
    return mapping


def _resolve_order(
    session: Session,
    *,
    workspace_id: UUID,
    payload: HotmartPaymentLinkCreateRequest,
) -> CommercialOrderRecord:
    order: CommercialOrderRecord | None = None
    if payload.order_id is not None:
        order = session.get(CommercialOrderRecord, payload.order_id)
    elif payload.checkout_ref.strip():
        order = session.exec(
            select(CommercialOrderRecord).where(CommercialOrderRecord.checkout_ref == payload.checkout_ref.strip())
        ).first()
    if order is None or order.workspace_id != workspace_id:
        raise ValueError("Hotmart checkout order was not found in this workspace.")
    if order.provider != "hotmart":
        raise ValueError("Payment links can only be created for Hotmart checkout orders.")
    if order.status != CommercialOrderStatus.pending:
        raise ValueError("Payment link can only be created for pending orders.")
    return order


def _order_product_key(session: Session, order: CommercialOrderRecord) -> str:
    line = session.exec(select(CommercialOrderLineRecord).where(CommercialOrderLineRecord.order_id == order.id)).first()
    if line is not None and line.product_key:
        return line.product_key
    return str(order.metadata_payload.get("product_key") or "")


def _build_payment_link_payload(
    *,
    order: CommercialOrderRecord,
    product_key: str,
    mapping: HotmartProductMappingRecord,
    payload: HotmartPaymentLinkCreateRequest,
    callback_url: str,
) -> dict[str, Any]:
    link_name = payload.link_name.strip() or f"{product_key}-{order.checkout_ref}"
    return {
        "name": link_name[:120],
        "value": round(order.total_cents / 100.0, 2),
        "currency": (mapping.currency or order.currency).upper(),
        "link_configuration": {
            "link_callback_url": callback_url,
        },
    }


def _existing_payment_link(
    session: Session,
    *,
    order_id: UUID,
) -> HotmartPaymentLinkRecord | None:
    return session.exec(
        select(HotmartPaymentLinkRecord)
        .where(
            HotmartPaymentLinkRecord.order_id == order_id,
            HotmartPaymentLinkRecord.activation_status != "failed",
        )
        .order_by(HotmartPaymentLinkRecord.created_at.desc())
    ).first()


def create_hotmart_payment_link_for_order(
    session: Session,
    *,
    workspace_id: UUID,
    payload: HotmartPaymentLinkCreateRequest,
    transport: httpx.BaseTransport | None = None,
) -> HotmartPaymentLinkResponse:
    env = normalize_hotmart_environment(payload.environment)
    order = _resolve_order(session, workspace_id=workspace_id, payload=payload)
    existing = _existing_payment_link(session, order_id=order.id)
    if existing is not None and not payload.force_new:
        return serialize_hotmart_payment_link(existing)

    product_key = _order_product_key(session, order)
    mapping = _get_active_mapping(session, workspace_id=workspace_id, environment=env, product_key=product_key)
    status = build_hotmart_status(session, workspace_id=workspace_id, environment=env)
    callback_url = payload.callback_url.strip() or status.webhook_public_url.strip()
    if not callback_url:
        raise ValueError("Hotmart payment link creation requires link_callback_url/webhook_public_url.")
    credentials = load_hotmart_credentials(session, workspace_id=workspace_id, environment=env)
    if credentials is None:
        raise ValueError("Hotmart OAuth credentials are required before creating payment links.")

    request_payload = _build_payment_link_payload(
        order=order,
        product_key=product_key,
        mapping=mapping,
        payload=payload,
        callback_url=callback_url,
    )
    api_client = HotmartPaymentLinkApiClient(
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
        api_result = api_client.create_payment_link(access_token=token.access_token, payload=request_payload)
    except HotmartPaymentLinkError as exc:
        failed = HotmartPaymentLinkRecord(
            workspace_id=workspace_id,
            order_id=order.id,
            created_by_user_id=order.buyer_user_id,
            internal_product_key=product_key,
            hotmart_payment_link_id=f"failed_{order.checkout_ref}",
            provider_ref=f"failed:{order.checkout_ref}",
            activation_status="failed",
            gross_amount_cents=order.subtotal_cents,
            discount_amount_cents=max(0, order.subtotal_cents - order.total_cents),
            net_amount_cents=order.total_cents,
            currency=(mapping.currency or order.currency).upper(),
            internal_unit_amount_usd_cents=mapping.internal_unit_amount_usd_cents,
            discount_origin=str(order.metadata_payload.get("discount_origin") or "internal_upgrade_credit")
            if order.subtotal_cents != order.total_cents
            else "none",
            request_payload_redacted=redact_payload(request_payload),
            response_payload_redacted=exc.payload,
            metadata_payload={"error_code": exc.code, "http_status": exc.http_status},
        )
        session.add(failed)
        session.flush()
        raise

    link = HotmartPaymentLinkRecord(
        workspace_id=workspace_id,
        order_id=order.id,
        created_by_user_id=order.buyer_user_id,
        internal_product_key=product_key,
        hotmart_payment_link_id=api_result.provider_ref,
        provider_ref=api_result.provider_ref,
        checkout_url=api_result.checkout_url,
        activation_status="pending_activation",
        gross_amount_cents=order.subtotal_cents,
        discount_amount_cents=max(0, order.subtotal_cents - order.total_cents),
        net_amount_cents=order.total_cents,
        currency=(mapping.currency or order.currency).upper(),
        internal_unit_amount_usd_cents=mapping.internal_unit_amount_usd_cents,
        discount_origin=str(order.metadata_payload.get("discount_origin") or "internal_upgrade_credit")
        if order.subtotal_cents != order.total_cents
        else "none",
        request_payload_redacted=redact_payload(request_payload),
        response_payload_redacted=api_result.payload_redacted,
    )
    session.add(link)
    order.checkout_url = api_result.checkout_url
    order.metadata_payload = {
        **order.metadata_payload,
        "hotmart_payment_link_id": api_result.provider_ref,
        "hotmart_payment_link_activation_status": "pending_activation",
        "hotmart_payment_link_http_status": api_result.http_status,
    }
    order.updated_at = utc_now()
    session.add(order)
    record_commercial_event(
        session,
        workspace_id=workspace_id,
        session_id=order.session_id,
        user_id=order.buyer_user_id,
        event_key="hotmart_payment_link_pending_activation",
        product_key=product_key,
        source="hotmart_payment_links",
        revenue_cents=order.total_cents,
        currency=link.currency,
        metadata={"order_id": str(order.id), "provider_ref": api_result.provider_ref},
        correlation_id=order.checkout_ref,
    )
    session.flush()
    return serialize_hotmart_payment_link(link)


def list_hotmart_payment_links(
    session: Session,
    *,
    workspace_id: UUID,
) -> list[HotmartPaymentLinkResponse]:
    rows = session.exec(
        select(HotmartPaymentLinkRecord)
        .where(HotmartPaymentLinkRecord.workspace_id == workspace_id)
        .order_by(HotmartPaymentLinkRecord.created_at.desc())
    ).all()
    return [serialize_hotmart_payment_link(row) for row in rows]


def refresh_hotmart_payment_link(
    session: Session,
    *,
    workspace_id: UUID,
    payment_link_id: UUID,
    environment: str = "sandbox",
    transport: httpx.BaseTransport | None = None,
) -> HotmartPaymentLinkResponse:
    env = normalize_hotmart_environment(environment)
    link = session.get(HotmartPaymentLinkRecord, payment_link_id)
    if link is None or link.workspace_id != workspace_id:
        raise ValueError("Hotmart payment link was not found in this workspace.")
    if link.activation_status == "active":
        return serialize_hotmart_payment_link(link)
    status = build_hotmart_status(session, workspace_id=workspace_id, environment=env)
    credentials = load_hotmart_credentials(session, workspace_id=workspace_id, environment=env)
    if credentials is None:
        raise ValueError("Hotmart OAuth credentials are required before refreshing payment links.")
    token = HotmartAuthClient(
        environment=env,
        auth_base_url=status.auth_base_url,
        timeout_seconds=get_settings().hotmart_request_timeout_seconds,
        transport=transport,
    ).fetch_access_token(credentials)
    api_client = HotmartPaymentLinkApiClient(
        api_base_url=status.api_base_url or default_hotmart_api_base_url(env),
        timeout_seconds=get_settings().hotmart_request_timeout_seconds,
        transport=transport,
    )
    payload = api_client.list_payment_links(access_token=token.access_token)
    if _payload_contains_link(payload, link.provider_ref):
        link.activation_status = "active"
        link.response_payload_redacted = payload
        link.updated_at = utc_now()
        session.add(link)
        order = session.get(CommercialOrderRecord, link.order_id)
        if order is not None:
            order.metadata_payload = {
                **order.metadata_payload,
                "hotmart_payment_link_activation_status": "active",
            }
            order.updated_at = utc_now()
            session.add(order)
        session.flush()
    return serialize_hotmart_payment_link(link)


def _payload_contains_link(payload: Any, provider_ref: str) -> bool:
    if not provider_ref:
        return False
    if isinstance(payload, dict):
        if provider_ref in {_extract_provider_ref(payload), _extract_checkout_url(payload)}:
            return True
        for key in ("items", "results", "data", "payment_links"):
            value = payload.get(key)
            if isinstance(value, list) and any(_payload_contains_link(item, provider_ref) for item in value):
                return True
    if isinstance(payload, list):
        return any(_payload_contains_link(item, provider_ref) for item in payload)
    return False

