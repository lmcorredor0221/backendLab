from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import httpx
from sqlmodel import Session, select

from app.core.config import get_settings
from app.models import (
    CommercialEntitlementRecord,
    CommercialEntitlementStatus,
    CommercialOrderRecord,
    CommercialOrderStatus,
    CommercialPaymentRecord,
    CommercialPaymentStatus,
    HotmartIntegrationConfigRecord,
    HotmartPaymentLinkRecord,
    HotmartProductMappingRecord,
    HotmartPromotionRecord,
    HotmartReconciliationIssueRecord,
    HotmartReconciliationIssueResponse,
    HotmartReconciliationResolveRequest,
    HotmartSyncCursorRecord,
    HotmartSyncCursorResponse,
    HotmartSyncRequest,
    HotmartSyncRunRecord,
    HotmartSyncRunResponse,
    HotmartWebhookEventRecord,
    HotmartWebhookReplayResponse,
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


SYNC_RESOURCES = {"products", "offers", "plans", "sales", "subscriptions", "coupons", "payment_links"}
APPROVED_SALE_STATUSES = {"APPROVED", "COMPLETE"}
REFUNDED_SALE_STATUSES = {"REFUNDED", "CHARGEBACK", "PARTIALLY_REFUNDED"}


class HotmartSyncError(RuntimeError):
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


@dataclass
class SyncIssueStats:
    created: int = 0
    updated: int = 0
    skipped: int = 0

    @property
    def total(self) -> int:
        return self.created + self.updated


@dataclass(frozen=True)
class HotmartResourcePage:
    payload_redacted: dict[str, Any]
    items: list[dict[str, Any]]
    next_page_token: str = ""
    last_transaction: str = ""


@dataclass(frozen=True)
class HotmartFetchedPayload:
    payload: dict[str, Any]
    payload_redacted: dict[str, Any]


class HotmartSyncApiClient:
    def __init__(
        self,
        *,
        api_base_url: str,
        timeout_seconds: int = 30,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_base_url = api_base_url.rstrip("/") or default_hotmart_api_base_url("sandbox")
        self.timeout_seconds = max(1, timeout_seconds)
        self.transport = transport

    def _url(self, path: str) -> str:
        normalized_path = path if path.startswith("/") else f"/{path}"
        return f"{self.api_base_url}{normalized_path}"

    def fetch_path(
        self,
        *,
        access_token: str,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> HotmartFetchedPayload:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}",
        }
        with httpx.Client(timeout=self.timeout_seconds, transport=self.transport) as client:
            response = client.get(self._url(path), headers=headers, params=_flatten_params(params or {}))
        payload = _read_response_payload(response)
        redacted = redact_payload(payload)
        if response.status_code == 429:
            raise HotmartSyncError(
                "rate_limited",
                "Hotmart rate limited the sync request.",
                http_status=response.status_code,
                payload=redacted,
            )
        if response.status_code >= 400:
            raise HotmartSyncError(
                "sync_resource_rejected",
                "Hotmart rejected the sync resource request.",
                http_status=response.status_code,
                payload=redacted,
            )
        return HotmartFetchedPayload(payload=payload, payload_redacted=redacted)


def _read_response_payload(response: httpx.Response) -> dict[str, Any]:
    if not response.text:
        return {}
    try:
        payload = response.json()
    except ValueError:
        return {"raw": response.text[:500]}
    return payload if isinstance(payload, dict) else {"payload": payload}


def _flatten_params(values: dict[str, Any]) -> list[tuple[str, str]]:
    params: list[tuple[str, str]] = []
    for key, value in values.items():
        if value is None or value == "":
            continue
        if isinstance(value, list):
            params.extend((key, str(item)) for item in value if str(item).strip())
            continue
        params.append((key, str(value)))
    return params


def _extract_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("items", "results", "data", "content", "subscriptions", "sales", "payment_links", "coupons"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return [payload] if payload else []


def _extract_next_page_token(payload: dict[str, Any]) -> str:
    page_info = payload.get("page_info")
    if isinstance(page_info, dict):
        return str(page_info.get("next_page_token") or "").strip()
    return str(payload.get("next_page_token") or "").strip()


def _get_path(payload: dict[str, Any], *path: str) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_text(payload: dict[str, Any], *paths: tuple[str, ...] | str) -> str:
    for path in paths:
        if isinstance(path, str):
            value = payload.get(path)
        else:
            value = _get_path(payload, *path)
        if str(value or "").strip():
            return str(value).strip()
    return ""


def _extract_product_ref(item: dict[str, Any]) -> str:
    return _first_text(item, "id", "ucode", ("product", "id"), ("product", "ucode"))


def _extract_product_name(item: dict[str, Any]) -> str:
    return _first_text(item, "name", ("product", "name")) or "Hotmart product"


def _extract_transaction(item: dict[str, Any]) -> str:
    return _first_text(
        item,
        "transaction",
        "transaction_code",
        ("purchase", "transaction"),
        ("purchase", "transaction_code"),
        ("order", "transaction"),
    )


def _extract_sale_status(item: dict[str, Any]) -> str:
    return _first_text(item, "transaction_status", "status", ("purchase", "status")).upper()


def _extract_coupon_ref(item: dict[str, Any]) -> str:
    return _first_text(item, "id", "coupon_id", "coupon_code", "code")


def _extract_payment_link_ref(item: dict[str, Any]) -> str:
    return _first_text(item, "ucode", "id", "code", "payment_link_id", "payment_link_ucode", "url", "checkout_url")


def _serialize_sync_run(record: HotmartSyncRunRecord) -> HotmartSyncRunResponse:
    return HotmartSyncRunResponse(
        id=record.id,
        workspace_id=record.workspace_id,
        environment=record.environment,  # type: ignore[arg-type]
        resource=record.resource,
        status=record.status,
        started_by_user_id=record.started_by_user_id,
        started_at=record.started_at,
        finished_at=record.finished_at,
        cursor_before=record.cursor_before,
        cursor_after=record.cursor_after,
        records_read=record.records_read,
        records_created=record.records_created,
        records_updated=record.records_updated,
        records_skipped=record.records_skipped,
        error_summary=record.error_summary,
        issue_count=int(record.metadata_payload.get("issue_count") or 0),
    )


def serialize_hotmart_sync_cursor(record: HotmartSyncCursorRecord) -> HotmartSyncCursorResponse:
    return HotmartSyncCursorResponse(
        id=record.id,
        workspace_id=record.workspace_id,
        environment=record.environment,  # type: ignore[arg-type]
        resource=record.resource,
        page_token=record.page_token,
        last_event_at=record.last_event_at,
        last_transaction=record.last_transaction,
        last_success_at=record.last_success_at,
        updated_at=record.updated_at,
    )


def serialize_hotmart_reconciliation_issue(
    record: HotmartReconciliationIssueRecord,
) -> HotmartReconciliationIssueResponse:
    return HotmartReconciliationIssueResponse(
        id=record.id,
        workspace_id=record.workspace_id,
        environment=record.environment,  # type: ignore[arg-type]
        issue_type=record.issue_type,
        severity=record.severity,
        status=record.status,
        provider_ref=record.provider_ref,
        internal_ref=record.internal_ref,
        summary=record.summary,
        suggested_action=record.suggested_action,
        resolution_action=record.resolution_action,
        resolution_note=record.resolution_note,
        resolved_by_user_id=record.resolved_by_user_id,
        resolved_at=record.resolved_at,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _cursor_record(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str,
    resource: str,
) -> HotmartSyncCursorRecord | None:
    return session.exec(
        select(HotmartSyncCursorRecord).where(
            HotmartSyncCursorRecord.workspace_id == workspace_id,
            HotmartSyncCursorRecord.environment == environment,
            HotmartSyncCursorRecord.resource == resource,
        )
    ).first()


def _active_mappings(session: Session, *, workspace_id: UUID, environment: str) -> list[HotmartProductMappingRecord]:
    return session.exec(
        select(HotmartProductMappingRecord).where(
            HotmartProductMappingRecord.workspace_id == workspace_id,
            HotmartProductMappingRecord.environment == environment,
            HotmartProductMappingRecord.is_active == True,  # noqa: E712
        )
    ).all()


def _resource_paths(
    *,
    resource: str,
    mappings: list[HotmartProductMappingRecord],
    product_id: str = "",
) -> list[str]:
    settings = get_settings()
    if resource == "products":
        return [settings.hotmart_products_list_path]
    if resource == "sales":
        return [settings.hotmart_sales_history_path]
    if resource == "subscriptions":
        return [settings.hotmart_subscriptions_list_path]
    if resource == "payment_links":
        return [settings.hotmart_payment_link_list_path]

    product_refs = [product_id.strip()] if product_id.strip() else []
    if not product_refs:
        for mapping in mappings:
            ref = mapping.hotmart_product_id.strip() or mapping.hotmart_product_ucode.strip()
            if ref and ref not in product_refs:
                product_refs.append(ref)
    if resource == "coupons":
        return [settings.hotmart_coupon_list_path_template.format(product_id=ref) for ref in product_refs]
    if resource == "offers":
        return [settings.hotmart_product_offers_path_template.format(product_id=ref) for ref in product_refs]
    if resource == "plans":
        return [settings.hotmart_product_plans_path_template.format(product_id=ref) for ref in product_refs]
    return []


def _fetch_resource_page(
    *,
    client: HotmartSyncApiClient,
    access_token: str,
    resource: str,
    mappings: list[HotmartProductMappingRecord],
    cursor_before: str,
    payload: HotmartSyncRequest,
) -> HotmartResourcePage:
    params = {
        **payload.filters,
        "max_results": max(1, min(payload.max_results, get_settings().hotmart_sync_page_size)),
    }
    if cursor_before:
        params["page_token"] = cursor_before
    paths = _resource_paths(resource=resource, mappings=mappings, product_id=payload.product_id)
    if not paths:
        return HotmartResourcePage(payload_redacted={"items": []}, items=[])

    if len(paths) == 1:
        response = client.fetch_path(access_token=access_token, path=paths[0], params=params)
        items = _extract_items(response.payload)
        return HotmartResourcePage(
            payload_redacted=response.payload_redacted,
            items=items,
            next_page_token=_extract_next_page_token(response.payload),
            last_transaction=_extract_transaction(items[-1]) if items else "",
        )

    aggregate_items: list[dict[str, Any]] = []
    aggregate_payloads: list[dict[str, Any]] = []
    for path in paths:
        response = client.fetch_path(access_token=access_token, path=path, params=params)
        aggregate_payloads.append({"path": path, "payload": response.payload_redacted})
        aggregate_items.extend(_extract_items(response.payload))
    return HotmartResourcePage(
        payload_redacted={"items": aggregate_items, "resource_payloads": aggregate_payloads},
        items=aggregate_items,
        last_transaction=_extract_transaction(aggregate_items[-1]) if aggregate_items else "",
    )


def _matching_mapping(
    mappings: list[HotmartProductMappingRecord],
    *,
    product_ref: str,
) -> HotmartProductMappingRecord | None:
    ref = product_ref.strip()
    if not ref:
        return None
    return next(
        (
            mapping
            for mapping in mappings
            if ref in {mapping.hotmart_product_id.strip(), mapping.hotmart_product_ucode.strip()}
        ),
        None,
    )


def _open_or_update_issue(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str,
    issue_type: str,
    provider_ref: str = "",
    internal_ref: str = "",
    severity: str = "medium",
    summary: str,
    suggested_action: str,
    metadata: dict[str, Any] | None = None,
    actor_user_id: UUID | None = None,
) -> str:
    issue = session.exec(
        select(HotmartReconciliationIssueRecord).where(
            HotmartReconciliationIssueRecord.workspace_id == workspace_id,
            HotmartReconciliationIssueRecord.environment == environment,
            HotmartReconciliationIssueRecord.issue_type == issue_type,
            HotmartReconciliationIssueRecord.provider_ref == provider_ref,
            HotmartReconciliationIssueRecord.internal_ref == internal_ref,
            HotmartReconciliationIssueRecord.status != "resolved",
            HotmartReconciliationIssueRecord.status != "ignored",
        )
    ).first()
    if issue is not None:
        issue.severity = severity
        issue.summary = summary
        issue.suggested_action = suggested_action
        issue.metadata_payload = {**issue.metadata_payload, **(metadata or {})}
        issue.updated_at = utc_now()
        session.add(issue)
        return "updated"

    issue = HotmartReconciliationIssueRecord(
        workspace_id=workspace_id,
        environment=environment,
        issue_type=issue_type,
        provider_ref=provider_ref,
        internal_ref=internal_ref,
        severity=severity,
        status="open",
        summary=summary,
        suggested_action=suggested_action,
        metadata_payload=metadata or {},
    )
    session.add(issue)
    session.flush()
    record_commercial_event(
        session,
        workspace_id=workspace_id,
        session_id=None,
        user_id=actor_user_id,
        event_key="hotmart_reconciliation_opened",
        product_key=issue_type,
        source="hotmart_reconciliation",
        metadata={"issue_id": str(issue.id), "provider_ref": provider_ref, "internal_ref": internal_ref},
        correlation_id=f"{issue_type}:{provider_ref or internal_ref}",
    )
    return "created"


def _count_issue_result(stats: SyncIssueStats, result: str) -> None:
    if result == "created":
        stats.created += 1
    elif result == "updated":
        stats.updated += 1
    else:
        stats.skipped += 1


def _reconcile_remote_products(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str,
    items: list[dict[str, Any]],
    mappings: list[HotmartProductMappingRecord],
    actor_user_id: UUID | None,
) -> SyncIssueStats:
    stats = SyncIssueStats()
    remote_refs: set[str] = set()
    for item in items:
        product_ref = _extract_product_ref(item)
        if not product_ref:
            stats.skipped += 1
            continue
        remote_refs.add(product_ref)
        if _matching_mapping(mappings, product_ref=product_ref) is None:
            result = _open_or_update_issue(
                session,
                workspace_id=workspace_id,
                environment=environment,
                issue_type="hotmart_product_without_mapping",
                provider_ref=product_ref,
                severity="medium",
                summary=f"Hotmart product {product_ref} is not mapped to an internal product.",
                suggested_action="Create or update a Hotmart product mapping.",
                metadata={"product": redact_payload(item), "product_name": _extract_product_name(item)},
                actor_user_id=actor_user_id,
            )
            _count_issue_result(stats, result)
        else:
            stats.skipped += 1

    for mapping in mappings:
        mapping_refs = {mapping.hotmart_product_id.strip(), mapping.hotmart_product_ucode.strip()} - {""}
        if remote_refs and mapping_refs.isdisjoint(remote_refs):
            result = _open_or_update_issue(
                session,
                workspace_id=workspace_id,
                environment=environment,
                issue_type="internal_mapping_product_not_found",
                provider_ref=next(iter(mapping_refs)),
                internal_ref=mapping.internal_product_key,
                severity="high",
                summary=f"Internal mapping {mapping.internal_product_key} points to a product not found in Hotmart sync.",
                suggested_action="Verify Hotmart product id/ucode or disable the stale mapping.",
                metadata={"mapping_id": str(mapping.id)},
                actor_user_id=actor_user_id,
            )
            _count_issue_result(stats, result)
    return stats


def _reconcile_remote_sales(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str,
    items: list[dict[str, Any]],
    mappings: list[HotmartProductMappingRecord],
    actor_user_id: UUID | None,
) -> SyncIssueStats:
    stats = SyncIssueStats()
    for item in items:
        transaction = _extract_transaction(item)
        status = _extract_sale_status(item)
        product_ref = _extract_product_ref(item)
        if not transaction:
            stats.skipped += 1
            continue
        payment = session.exec(
            select(CommercialPaymentRecord).where(
                CommercialPaymentRecord.workspace_id == workspace_id,
                CommercialPaymentRecord.provider == "hotmart",
                CommercialPaymentRecord.provider_payment_id == transaction,
            )
        ).first()
        order = session.exec(
            select(CommercialOrderRecord).where(
                CommercialOrderRecord.workspace_id == workspace_id,
                CommercialOrderRecord.provider == "hotmart",
                CommercialOrderRecord.checkout_ref == transaction,
            )
        ).first()
        if status in APPROVED_SALE_STATUSES and payment is None and order is None:
            result = _open_or_update_issue(
                session,
                workspace_id=workspace_id,
                environment=environment,
                issue_type="hotmart_payment_without_internal_order",
                provider_ref=transaction,
                severity="high",
                summary=f"Hotmart sale {transaction} is approved but no internal order/payment is linked.",
                suggested_action="Link manually to an order or create entitlement after validating buyer/product.",
                metadata={"sale": redact_payload(item)},
                actor_user_id=actor_user_id,
            )
            _count_issue_result(stats, result)
        elif status in REFUNDED_SALE_STATUSES and payment is not None:
            stats.skipped += 1
        else:
            stats.skipped += 1

        if product_ref and _matching_mapping(mappings, product_ref=product_ref) is None:
            result = _open_or_update_issue(
                session,
                workspace_id=workspace_id,
                environment=environment,
                issue_type="sale_product_without_mapping",
                provider_ref=product_ref,
                internal_ref=transaction,
                severity="medium",
                summary=f"Sale {transaction} references Hotmart product {product_ref} without internal mapping.",
                suggested_action="Create product mapping before resolving access.",
                metadata={"sale": redact_payload(item)},
                actor_user_id=actor_user_id,
            )
            _count_issue_result(stats, result)
    return stats


def _reconcile_remote_coupons(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str,
    items: list[dict[str, Any]],
    actor_user_id: UUID | None,
) -> SyncIssueStats:
    stats = SyncIssueStats()
    for item in items:
        coupon_ref = _extract_coupon_ref(item)
        coupon_code = _first_text(item, "coupon_code", "code")
        if not coupon_ref and not coupon_code:
            stats.skipped += 1
            continue
        promotion = session.exec(
            select(HotmartPromotionRecord).where(
                HotmartPromotionRecord.workspace_id == workspace_id,
                HotmartPromotionRecord.environment == environment,
                (HotmartPromotionRecord.coupon_id == coupon_ref)
                | (HotmartPromotionRecord.coupon_code == coupon_code.upper()),
            )
        ).first()
        if promotion is None:
            result = _open_or_update_issue(
                session,
                workspace_id=workspace_id,
                environment=environment,
                issue_type="hotmart_coupon_without_internal_record",
                provider_ref=coupon_ref or coupon_code,
                severity="medium",
                summary=f"Hotmart coupon {coupon_code or coupon_ref} exists without an internal promotion record.",
                suggested_action="Import, map, or delete the provider coupon after validation.",
                metadata={"coupon": redact_payload(item)},
                actor_user_id=actor_user_id,
            )
            _count_issue_result(stats, result)
        else:
            stats.skipped += 1
    return stats


def _reconcile_local_state(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str,
    resource: str,
    remote_items: list[dict[str, Any]],
    actor_user_id: UUID | None,
) -> SyncIssueStats:
    stats = SyncIssueStats()
    if resource == "payment_links":
        remote_refs = {_extract_payment_link_ref(item) for item in remote_items}
        links = session.exec(
            select(HotmartPaymentLinkRecord).where(
                HotmartPaymentLinkRecord.workspace_id == workspace_id,
                HotmartPaymentLinkRecord.activation_status != "failed",
            )
        ).all()
        for link in links:
            candidates = {link.provider_ref, link.hotmart_payment_link_id, link.checkout_url} - {""}
            if remote_refs and candidates.isdisjoint(remote_refs):
                result = _open_or_update_issue(
                    session,
                    workspace_id=workspace_id,
                    environment=environment,
                    issue_type="internal_payment_link_not_found",
                    provider_ref=link.provider_ref,
                    internal_ref=str(link.id),
                    severity="medium",
                    summary=f"Internal Hotmart payment link {link.provider_ref} was not found in provider sync.",
                    suggested_action="Refresh the link or regenerate it if Hotmart no longer exposes it.",
                    metadata={"payment_link_id": str(link.id), "order_id": str(link.order_id)},
                    actor_user_id=actor_user_id,
                )
                _count_issue_result(stats, result)

    orders = session.exec(
        select(CommercialOrderRecord).where(
            CommercialOrderRecord.workspace_id == workspace_id,
            CommercialOrderRecord.provider == "hotmart",
            CommercialOrderRecord.status == CommercialOrderStatus.pending,
        )
    ).all()
    for order in orders:
        payment = session.exec(
            select(CommercialPaymentRecord).where(
                CommercialPaymentRecord.order_id == order.id,
                CommercialPaymentRecord.status == CommercialPaymentStatus.succeeded,
            )
        ).first()
        if payment is None:
            result = _open_or_update_issue(
                session,
                workspace_id=workspace_id,
                environment=environment,
                issue_type="internal_order_without_hotmart_payment",
                provider_ref=order.checkout_ref,
                internal_ref=str(order.id),
                severity="low",
                summary=f"Internal Hotmart order {order.checkout_ref} has no succeeded payment yet.",
                suggested_action="Confirm provider status or wait if checkout is still pending.",
                metadata={"order_id": str(order.id)},
                actor_user_id=actor_user_id,
            )
            _count_issue_result(stats, result)

    refunded_payments = session.exec(
        select(CommercialPaymentRecord).where(
            CommercialPaymentRecord.workspace_id == workspace_id,
            CommercialPaymentRecord.provider == "hotmart",
            CommercialPaymentRecord.status == CommercialPaymentStatus.refunded,
        )
    ).all()
    for payment in refunded_payments:
        entitlement = session.exec(
            select(CommercialEntitlementRecord).where(
                CommercialEntitlementRecord.workspace_id == workspace_id,
                CommercialEntitlementRecord.status == CommercialEntitlementStatus.active,
                (CommercialEntitlementRecord.payment_id == payment.id)
                | (CommercialEntitlementRecord.order_id == payment.order_id),
            )
        ).first()
        if entitlement is not None:
            result = _open_or_update_issue(
                session,
                workspace_id=workspace_id,
                environment=environment,
                issue_type="entitlement_active_with_refunded_payment",
                provider_ref=payment.provider_payment_id,
                internal_ref=str(entitlement.id),
                severity="critical",
                summary="An active entitlement is linked to a refunded Hotmart payment.",
                suggested_action="Suspend or revoke entitlement after validating refund status.",
                metadata={"payment_id": str(payment.id), "entitlement_id": str(entitlement.id)},
                actor_user_id=actor_user_id,
            )
            _count_issue_result(stats, result)

    succeeded_payments = session.exec(
        select(CommercialPaymentRecord).where(
            CommercialPaymentRecord.workspace_id == workspace_id,
            CommercialPaymentRecord.provider == "hotmart",
            CommercialPaymentRecord.status == CommercialPaymentStatus.succeeded,
        )
    ).all()
    for payment in succeeded_payments:
        entitlement = session.exec(
            select(CommercialEntitlementRecord).where(
                CommercialEntitlementRecord.workspace_id == workspace_id,
                (CommercialEntitlementRecord.payment_id == payment.id)
                | (CommercialEntitlementRecord.order_id == payment.order_id),
            )
        ).first()
        if entitlement is None:
            result = _open_or_update_issue(
                session,
                workspace_id=workspace_id,
                environment=environment,
                issue_type="entitlement_missing_for_approved_purchase",
                provider_ref=payment.provider_payment_id,
                internal_ref=str(payment.order_id),
                severity="high",
                summary="A succeeded Hotmart payment has no internal entitlement.",
                suggested_action="Create entitlement or replay the purchase webhook.",
                metadata={"payment_id": str(payment.id), "order_id": str(payment.order_id)},
                actor_user_id=actor_user_id,
            )
            _count_issue_result(stats, result)

    promotions = session.exec(
        select(HotmartPromotionRecord).where(
            HotmartPromotionRecord.workspace_id == workspace_id,
            HotmartPromotionRecord.environment == environment,
            HotmartPromotionRecord.discount_origin == "provider_coupon",
        )
    ).all()
    for promotion in promotions:
        if promotion.status in {"draft", "sync_error"}:
            result = _open_or_update_issue(
                session,
                workspace_id=workspace_id,
                environment=environment,
                issue_type="internal_coupon_not_published",
                provider_ref=promotion.coupon_id or promotion.coupon_code,
                internal_ref=str(promotion.id),
                severity="medium" if promotion.status == "draft" else "high",
                summary=f"Internal promotion {promotion.coupon_code} is not published/synced in Hotmart.",
                suggested_action="Publish, retry sync, or mark as intentionally internal.",
                metadata={"promotion_id": str(promotion.id), "status": promotion.status},
                actor_user_id=actor_user_id,
            )
            _count_issue_result(stats, result)
    return stats


def _reconcile_resource(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str,
    resource: str,
    items: list[dict[str, Any]],
    actor_user_id: UUID | None,
) -> SyncIssueStats:
    mappings = _active_mappings(session, workspace_id=workspace_id, environment=environment)
    total = SyncIssueStats()
    if resource == "products":
        remote_stats = _reconcile_remote_products(
            session,
            workspace_id=workspace_id,
            environment=environment,
            items=items,
            mappings=mappings,
            actor_user_id=actor_user_id,
        )
        total.created += remote_stats.created
        total.updated += remote_stats.updated
        total.skipped += remote_stats.skipped
    if resource == "sales":
        remote_stats = _reconcile_remote_sales(
            session,
            workspace_id=workspace_id,
            environment=environment,
            items=items,
            mappings=mappings,
            actor_user_id=actor_user_id,
        )
        total.created += remote_stats.created
        total.updated += remote_stats.updated
        total.skipped += remote_stats.skipped
    if resource == "coupons":
        remote_stats = _reconcile_remote_coupons(
            session,
            workspace_id=workspace_id,
            environment=environment,
            items=items,
            actor_user_id=actor_user_id,
        )
        total.created += remote_stats.created
        total.updated += remote_stats.updated
        total.skipped += remote_stats.skipped

    local_stats = _reconcile_local_state(
        session,
        workspace_id=workspace_id,
        environment=environment,
        resource=resource,
        remote_items=items,
        actor_user_id=actor_user_id,
    )
    total.created += local_stats.created
    total.updated += local_stats.updated
    total.skipped += local_stats.skipped
    return total


def _update_cursor(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str,
    resource: str,
    cursor_after: str,
    last_transaction: str,
) -> HotmartSyncCursorRecord:
    cursor = _cursor_record(session, workspace_id=workspace_id, environment=environment, resource=resource)
    if cursor is None:
        cursor = HotmartSyncCursorRecord(
            workspace_id=workspace_id,
            environment=environment,
            resource=resource,
        )
    cursor.page_token = cursor_after
    cursor.last_transaction = last_transaction or cursor.last_transaction
    cursor.last_success_at = utc_now()
    cursor.updated_at = utc_now()
    session.add(cursor)
    return cursor


def _update_config_last_sync(session: Session, *, workspace_id: UUID, environment: str) -> None:
    config = session.exec(
        select(HotmartIntegrationConfigRecord).where(
            HotmartIntegrationConfigRecord.workspace_id == workspace_id,
            HotmartIntegrationConfigRecord.environment == environment,
        )
    ).first()
    if config is not None:
        config.last_sync_at = utc_now()
        config.updated_at = utc_now()
        session.add(config)


def run_hotmart_manual_sync(
    session: Session,
    *,
    workspace_id: UUID,
    payload: HotmartSyncRequest,
    actor_user_id: UUID | None = None,
    transport: httpx.BaseTransport | None = None,
) -> HotmartSyncRunResponse:
    env = normalize_hotmart_environment(payload.environment)
    resource = payload.resource.strip().lower()
    if resource not in SYNC_RESOURCES:
        raise ValueError(f"Unsupported Hotmart sync resource: {payload.resource}.")

    existing_cursor = _cursor_record(session, workspace_id=workspace_id, environment=env, resource=resource)
    cursor_before = ""
    if not payload.force_reset:
        cursor_before = payload.page_token.strip()
        if not cursor_before and existing_cursor is not None:
            cursor_before = existing_cursor.page_token
    run = HotmartSyncRunRecord(
        workspace_id=workspace_id,
        environment=env,
        resource=resource,
        status="running",
        started_by_user_id=actor_user_id,
        cursor_before=cursor_before,
    )
    session.add(run)
    session.flush()
    record_commercial_event(
        session,
        workspace_id=workspace_id,
        session_id=None,
        user_id=actor_user_id,
        event_key="hotmart_sync_started",
        product_key=resource,
        source="hotmart_sync",
        metadata={"run_id": str(run.id), "cursor_before": cursor_before},
        correlation_id=str(run.id),
    )

    status = build_hotmart_status(session, workspace_id=workspace_id, environment=env)
    credentials = load_hotmart_credentials(session, workspace_id=workspace_id, environment=env)
    if credentials is None:
        run.status = "failed"
        run.finished_at = utc_now()
        run.error_summary = "Hotmart OAuth credentials are required before syncing resources."
        session.add(run)
        record_commercial_event(
            session,
            workspace_id=workspace_id,
            session_id=None,
            user_id=actor_user_id,
            event_key="hotmart_sync_failed",
            product_key=resource,
            source="hotmart_sync",
            metadata={"run_id": str(run.id), "error_summary": run.error_summary},
            correlation_id=str(run.id),
        )
        session.flush()
        raise ValueError(run.error_summary)

    mappings = _active_mappings(session, workspace_id=workspace_id, environment=env)
    client = HotmartSyncApiClient(
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
        page = _fetch_resource_page(
            client=client,
            access_token=token.access_token,
            resource=resource,
            mappings=mappings,
            cursor_before=cursor_before,
            payload=payload,
        )
        issue_stats = _reconcile_resource(
            session,
            workspace_id=workspace_id,
            environment=env,
            resource=resource,
            items=page.items,
            actor_user_id=actor_user_id,
        )
    except HotmartAuthError as exc:
        run.status = "failed"
        run.finished_at = utc_now()
        run.error_summary = "Hotmart OAuth failed while syncing resources."
        run.metadata_payload = {"error_code": exc.code, "error_payload_redacted": exc.payload}
        session.add(run)
        record_commercial_event(
            session,
            workspace_id=workspace_id,
            session_id=None,
            user_id=actor_user_id,
            event_key="hotmart_sync_failed",
            product_key=resource,
            source="hotmart_sync",
            metadata={"run_id": str(run.id), "error_code": exc.code},
            correlation_id=str(run.id),
        )
        session.flush()
        raise HotmartSyncError("sync_auth_failed", run.error_summary, http_status=exc.http_status, payload=exc.payload) from exc
    except HotmartSyncError as exc:
        run.status = "rate_limited" if exc.code == "rate_limited" else "failed"
        run.finished_at = utc_now()
        run.error_summary = str(exc)
        run.metadata_payload = {"error_code": exc.code, "error_payload_redacted": exc.payload}
        session.add(run)
        record_commercial_event(
            session,
            workspace_id=workspace_id,
            session_id=None,
            user_id=actor_user_id,
            event_key="hotmart_sync_failed",
            product_key=resource,
            source="hotmart_sync",
            metadata={"run_id": str(run.id), "error_code": exc.code},
            correlation_id=str(run.id),
        )
        session.flush()
        raise

    cursor = _update_cursor(
        session,
        workspace_id=workspace_id,
        environment=env,
        resource=resource,
        cursor_after=page.next_page_token,
        last_transaction=page.last_transaction,
    )
    _update_config_last_sync(session, workspace_id=workspace_id, environment=env)
    run.status = "succeeded"
    run.finished_at = utc_now()
    run.cursor_after = cursor.page_token
    run.records_read = len(page.items)
    run.records_created = issue_stats.created
    run.records_updated = issue_stats.updated
    run.records_skipped = issue_stats.skipped
    run.metadata_payload = {
        "issue_count": issue_stats.total,
        "remote_sample_redacted": page.items[:5],
        "payload_page_info": redact_payload(page.payload_redacted.get("page_info", {}))
        if isinstance(page.payload_redacted, dict)
        else {},
    }
    session.add(run)
    record_commercial_event(
        session,
        workspace_id=workspace_id,
        session_id=None,
        user_id=actor_user_id,
        event_key="hotmart_sync_completed",
        product_key=resource,
        source="hotmart_sync",
        metadata={
            "run_id": str(run.id),
            "records_read": run.records_read,
            "issues_created": run.records_created,
            "issues_updated": run.records_updated,
        },
        correlation_id=str(run.id),
    )
    session.flush()
    return _serialize_sync_run(run)


def list_hotmart_sync_runs(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str = "sandbox",
    resource: str = "",
) -> list[HotmartSyncRunResponse]:
    env = normalize_hotmart_environment(environment)
    query = select(HotmartSyncRunRecord).where(
        HotmartSyncRunRecord.workspace_id == workspace_id,
        HotmartSyncRunRecord.environment == env,
    )
    if resource.strip():
        query = query.where(HotmartSyncRunRecord.resource == resource.strip().lower())
    rows = session.exec(query.order_by(HotmartSyncRunRecord.started_at.desc())).all()
    return [_serialize_sync_run(row) for row in rows]


def list_hotmart_sync_cursors(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str = "sandbox",
) -> list[HotmartSyncCursorResponse]:
    env = normalize_hotmart_environment(environment)
    rows = session.exec(
        select(HotmartSyncCursorRecord)
        .where(
            HotmartSyncCursorRecord.workspace_id == workspace_id,
            HotmartSyncCursorRecord.environment == env,
        )
        .order_by(HotmartSyncCursorRecord.resource)
    ).all()
    return [serialize_hotmart_sync_cursor(row) for row in rows]


def list_hotmart_reconciliation_issues(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str = "sandbox",
    status: str = "open",
) -> list[HotmartReconciliationIssueResponse]:
    env = normalize_hotmart_environment(environment)
    query = select(HotmartReconciliationIssueRecord).where(
        HotmartReconciliationIssueRecord.workspace_id == workspace_id,
        HotmartReconciliationIssueRecord.environment == env,
    )
    if status.strip() and status != "all":
        query = query.where(HotmartReconciliationIssueRecord.status == status.strip())
    rows = session.exec(query.order_by(HotmartReconciliationIssueRecord.updated_at.desc())).all()
    return [serialize_hotmart_reconciliation_issue(row) for row in rows]


def resolve_hotmart_reconciliation_issue(
    session: Session,
    *,
    workspace_id: UUID,
    issue_id: UUID,
    payload: HotmartReconciliationResolveRequest,
    actor_user_id: UUID | None = None,
) -> HotmartReconciliationIssueResponse:
    issue = session.get(HotmartReconciliationIssueRecord, issue_id)
    if issue is None or issue.workspace_id != workspace_id:
        raise ValueError("Hotmart reconciliation issue was not found in this workspace.")
    if not payload.resolution_action.strip():
        raise ValueError("resolution_action is required.")
    issue.status = payload.status
    issue.resolution_action = payload.resolution_action.strip()
    issue.resolution_note = payload.resolution_note.strip()
    issue.resolved_by_user_id = actor_user_id
    issue.resolved_at = utc_now() if payload.status in {"resolved", "ignored"} else None
    issue.updated_at = utc_now()
    session.add(issue)
    record_commercial_event(
        session,
        workspace_id=workspace_id,
        session_id=None,
        user_id=actor_user_id,
        event_key="hotmart_reconciliation_resolved",
        product_key=issue.issue_type,
        source="hotmart_reconciliation",
        metadata={"issue_id": str(issue.id), "resolution_action": issue.resolution_action, "status": issue.status},
        correlation_id=f"{issue.issue_type}:{issue.provider_ref or issue.internal_ref}",
    )
    session.flush()
    return serialize_hotmart_reconciliation_issue(issue)


def replay_hotmart_webhook_event(
    session: Session,
    *,
    workspace_id: UUID,
    event_ref: str,
    environment: str = "sandbox",
    actor_user_id: UUID | None = None,
) -> HotmartWebhookReplayResponse:
    env = normalize_hotmart_environment(environment)
    event = session.exec(
        select(HotmartWebhookEventRecord).where(
            HotmartWebhookEventRecord.event_id == event_ref,
        )
    ).first()
    if event is None:
        try:
            event = session.get(HotmartWebhookEventRecord, UUID(event_ref))
        except ValueError:
            event = None
    if event is None or event.workspace_id != workspace_id:
        raise ValueError("Hotmart webhook event was not found in this workspace.")

    previous_status = event.processing_status
    event.retries += 1
    event.processing_status = "replay_requested"
    session.add(event)
    issue_result = _open_or_update_issue(
        session,
        workspace_id=workspace_id,
        environment=env,
        issue_type="webhook_replay_requested",
        provider_ref=event.transaction or event.event_id,
        internal_ref=str(event.order_id or ""),
        severity="medium",
        summary=f"Webhook {event.event_id} was queued for manual replay review.",
        suggested_action="Validate payload and replay business effects from the admin reconciliation workflow.",
        metadata={"event_id": event.event_id, "previous_status": previous_status, "retries": event.retries},
        actor_user_id=actor_user_id,
    )
    issue = session.exec(
        select(HotmartReconciliationIssueRecord).where(
            HotmartReconciliationIssueRecord.workspace_id == workspace_id,
            HotmartReconciliationIssueRecord.environment == env,
            HotmartReconciliationIssueRecord.issue_type == "webhook_replay_requested",
            HotmartReconciliationIssueRecord.provider_ref == (event.transaction or event.event_id),
            HotmartReconciliationIssueRecord.status != "resolved",
            HotmartReconciliationIssueRecord.status != "ignored",
        )
    ).first()
    session.flush()
    return HotmartWebhookReplayResponse(
        event_id=event.event_id,
        processing_status=event.processing_status,
        retries=event.retries,
        issue_id=issue.id if issue is not None else None,
        message="Webhook replay review issue opened." if issue_result == "created" else "Webhook replay review issue updated.",
    )
