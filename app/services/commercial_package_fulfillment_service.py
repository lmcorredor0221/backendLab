from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from sqlmodel import Session, select

from app.models import (
    CommercialLegacyPackageResolutionCandidateResponse,
    CommercialLegacyPackageResolutionResponse,
    CommercialOrderLineRecord,
    CommercialOrderRecord,
    CommercialOrderStatus,
    CommercialPackageCatalogRecord,
    CommercialPackageType,
    CommercialPaymentRecord,
    CommercialQuotaSourceKind,
    utc_now,
)
from app.services.commercial_catalog_service import (
    get_package_catalog_entry,
    list_enabled_packages_for_product,
    package_units_for_product,
)
from app.services.commercial_quota_service import (
    grant_balance_units,
    initialize_workspace_commercial_quota,
    list_balance_buckets,
)


COMMERCIAL_CREDIT_PRODUCTS: tuple[str, ...] = ("blueprint_pro", "acp")
DEFAULT_SUBSCRIPTION_CYCLE_DAYS = 30
LEGACY_PACKAGE_RESOLUTION_KEY = "legacy_package_resolution"
LEGACY_PACKAGE_RESOLUTION_PENDING = "pending_manual_resolution"
LEGACY_PACKAGE_RESOLUTION_RESOLVED = "resolved"


@dataclass(frozen=True)
class PackageCreditGrant:
    product_key: str
    units: int
    source_kind: CommercialQuotaSourceKind
    bucket_key: str
    source_ref: str
    starts_at: datetime
    ends_at: datetime | None


def resolve_paid_order_package(
    session: Session,
    *,
    order: CommercialOrderRecord,
    order_line: CommercialOrderLineRecord | None = None,
) -> tuple[CommercialPackageCatalogRecord | None, str]:
    explicit_package_code = ""
    if isinstance(order.metadata_payload, dict):
        explicit_package_code = str(order.metadata_payload.get("package_code") or "").strip()
    if not explicit_package_code and order_line is not None and isinstance(order_line.metadata_payload, dict):
        explicit_package_code = str(order_line.metadata_payload.get("package_code") or "").strip()
    if explicit_package_code:
        package = get_package_catalog_entry(session, package_code=explicit_package_code)
        return package, "explicit" if package is not None else "explicit_missing"

    product_key = _order_product_key(order, order_line)
    if not product_key:
        return None, "missing_product_key"
    candidates = list_enabled_packages_for_product(session, product_key=product_key)
    if len(candidates) == 1:
        return candidates[0], "unique_product_fallback"
    if not candidates:
        return None, "no_candidate"
    return None, "ambiguous_product_candidates"


def apply_paid_order_package_credits(
    session: Session,
    *,
    order: CommercialOrderRecord,
    payment: CommercialPaymentRecord,
    actor_user_id: UUID | None = None,
    order_line: CommercialOrderLineRecord | None = None,
) -> dict[str, Any] | None:
    resolved_line = order_line or _order_line(session, order)
    package, resolution_strategy = resolve_paid_order_package(session, order=order, order_line=resolved_line)
    if package is None:
        return None

    paid_at = order.paid_at or payment.updated_at or payment.created_at or order.updated_at
    grants: list[PackageCreditGrant] = []
    for product_key in COMMERCIAL_CREDIT_PRODUCTS:
        units = package_units_for_product(package, product_key)
        if units <= 0:
            continue
        initialize_workspace_commercial_quota(
            session,
            workspace_id=order.workspace_id,
            actor_user_id=actor_user_id,
            at_time=paid_at,
        )
        grant = _build_credit_grant(
            session,
            order=order,
            package=package,
            product_key=product_key,
            units=units,
            paid_at=paid_at,
        )
        grant_balance_units(
            session,
            workspace_id=order.workspace_id,
            product_key=grant.product_key,
            source_kind=grant.source_kind,
            units=grant.units,
            bucket_key=grant.bucket_key,
            source_ref=grant.source_ref,
            actor_user_id=actor_user_id,
            order_id=order.id,
            payment_id=payment.id,
            starts_at=grant.starts_at,
            ends_at=grant.ends_at,
            overwrite_existing=True,
            metadata={
                "package_code": package.package_code,
                "package_type": package.package_type.value,
                "resolution_strategy": resolution_strategy,
                "order_product_key": _order_product_key(order, resolved_line),
            },
            at_time=paid_at,
        )
        grants.append(grant)

    if not grants:
        return None

    return {
        "package_code": package.package_code,
        "package_type": package.package_type.value,
        "resolution_strategy": resolution_strategy,
        "grants": [
            {
                "product_key": grant.product_key,
                "units": grant.units,
                "source_kind": grant.source_kind.value,
                "bucket_key": grant.bucket_key,
                "source_ref": grant.source_ref,
                "starts_at": grant.starts_at.isoformat(),
                "ends_at": grant.ends_at.isoformat() if grant.ends_at is not None else None,
            }
            for grant in grants
        ],
    }


def mark_pending_legacy_package_resolution(
    session: Session,
    *,
    order: CommercialOrderRecord,
    payment: CommercialPaymentRecord,
    order_line: CommercialOrderLineRecord | None = None,
) -> CommercialLegacyPackageResolutionResponse | None:
    resolved_line = order_line or _order_line(session, order)
    package, resolution_strategy = resolve_paid_order_package(session, order=order, order_line=resolved_line)
    if package is not None or resolution_strategy != "ambiguous_product_candidates":
        return None
    current_state = _legacy_package_resolution_state(order, payment=payment)
    detected_at = _parse_iso_datetime(current_state.get("detected_at")) if current_state else None
    if detected_at is None:
        detected_at = order.paid_at or payment.updated_at or payment.created_at or order.updated_at
    state = {
        "status": LEGACY_PACKAGE_RESOLUTION_PENDING,
        "reason": resolution_strategy,
        "detected_at": detected_at.isoformat(),
        "candidate_package_codes": [item.package_code for item in _candidate_packages_for_order(session, order, order_line=resolved_line)],
        "selected_package_code": str(current_state.get("selected_package_code") or "") if current_state else "",
        "resolution_note": str(current_state.get("resolution_note") or "") if current_state else "",
    }
    _persist_legacy_package_resolution_state(order=order, payment=payment, state=state)
    return build_legacy_package_resolution_response(
        session,
        order=order,
        payment=payment,
        order_line=resolved_line,
    )


def list_legacy_package_resolutions(
    session: Session,
    *,
    workspace_id: UUID,
    status_filter: str = "pending",
    product_key: str = "",
    limit: int = 100,
) -> list[CommercialLegacyPackageResolutionResponse]:
    normalized_status = status_filter.strip().lower() or "pending"
    rows = session.exec(
        select(CommercialOrderRecord)
        .where(
            CommercialOrderRecord.workspace_id == workspace_id,
            CommercialOrderRecord.status == CommercialOrderStatus.paid,
        )
        .order_by(CommercialOrderRecord.paid_at.desc(), CommercialOrderRecord.updated_at.desc())
    ).all()
    responses: list[CommercialLegacyPackageResolutionResponse] = []
    for order in rows:
        response = build_legacy_package_resolution_response(session, order=order)
        if response is None:
            continue
        if product_key.strip() and response.product_key != product_key.strip():
            continue
        if normalized_status == "pending" and response.status != LEGACY_PACKAGE_RESOLUTION_PENDING:
            continue
        if normalized_status == "resolved" and response.status != LEGACY_PACKAGE_RESOLUTION_RESOLVED:
            continue
        responses.append(response)
        if len(responses) >= max(1, min(limit, 200)):
            break
    return responses


def resolve_legacy_package_resolution(
    session: Session,
    *,
    workspace_id: UUID,
    order_id: UUID,
    package_code: str,
    resolution_note: str = "",
    actor_user_id: UUID | None = None,
) -> CommercialLegacyPackageResolutionResponse:
    order = session.get(CommercialOrderRecord, order_id)
    if order is None or order.workspace_id != workspace_id:
        raise ValueError("Commercial order not found.")
    payment = _latest_payment_for_order(session, order)
    if payment is None:
        raise ValueError("Legacy package resolution requires a paid payment record.")
    resolved_line = _order_line(session, order)
    current = build_legacy_package_resolution_response(
        session,
        order=order,
        payment=payment,
        order_line=resolved_line,
    )
    if current is None:
        raise ValueError("Order does not require legacy package resolution.")
    selected_package_code = package_code.strip()
    if not selected_package_code:
        raise ValueError("A package code is required.")
    if current.status == LEGACY_PACKAGE_RESOLUTION_RESOLVED:
        if current.selected_package_code == selected_package_code:
            return current
        raise ValueError("Legacy package resolution is already closed for this order.")
    if current.package_credit_applied:
        raise ValueError("Order already has package credit applied.")
    candidate_codes = {item.package_code for item in current.candidate_packages}
    if candidate_codes and selected_package_code not in candidate_codes:
        raise ValueError("Selected package is not an active candidate for this legacy order.")
    package = get_package_catalog_entry(session, package_code=selected_package_code)
    if package is None:
        raise ValueError(f"Commercial package {selected_package_code} is not active.")
    order_product_key = _order_product_key(order, resolved_line)
    if not order_product_key:
        raise ValueError("Legacy package resolution requires an order product key.")
    if package_units_for_product(package, order_product_key) <= 0:
        raise ValueError(f"Commercial package {selected_package_code} does not grant units for product {order_product_key}.")
    order.metadata_payload = {**dict(order.metadata_payload or {}), "package_code": selected_package_code}
    if resolved_line is not None:
        resolved_line.metadata_payload = {**dict(resolved_line.metadata_payload or {}), "package_code": selected_package_code}
        session.add(resolved_line)
    summary = apply_paid_order_package_credits(
        session,
        order=order,
        payment=payment,
        actor_user_id=actor_user_id,
        order_line=resolved_line,
    )
    if summary is None:
        raise ValueError("Unable to apply package credit for the selected package.")
    now = utc_now()
    order.metadata_payload = {**dict(order.metadata_payload or {}), "package_credit": summary}
    payment.metadata_payload = {**dict(payment.metadata_payload or {}), "package_credit": summary}
    state = {
        "status": LEGACY_PACKAGE_RESOLUTION_RESOLVED,
        "reason": current.reason or "ambiguous_product_candidates",
        "detected_at": current.detected_at.isoformat() if current.detected_at is not None else now.isoformat(),
        "candidate_package_codes": [item.package_code for item in current.candidate_packages],
        "selected_package_code": selected_package_code,
        "resolution_note": resolution_note.strip(),
        "resolved_at": now.isoformat(),
        "resolved_by_user_id": str(actor_user_id) if actor_user_id is not None else "",
    }
    _persist_legacy_package_resolution_state(order=order, payment=payment, state=state)
    order.updated_at = now
    payment.updated_at = now
    session.add(order)
    session.add(payment)
    session.flush()
    return build_legacy_package_resolution_response(
        session,
        order=order,
        payment=payment,
        order_line=resolved_line,
    ) or CommercialLegacyPackageResolutionResponse(
        order_id=order.id,
        workspace_id=order.workspace_id,
        session_id=order.session_id,
        payment_id=payment.id,
        product_key=order_product_key,
        status=LEGACY_PACKAGE_RESOLUTION_RESOLVED,
        reason="ambiguous_product_candidates",
        provider=order.provider,
        checkout_ref=order.checkout_ref,
        currency=order.currency,
        total_cents=order.total_cents,
        selected_package_code=selected_package_code,
        resolution_note=resolution_note.strip(),
        package_credit_applied=True,
        created_at=order.created_at,
        paid_at=order.paid_at,
        detected_at=current.detected_at,
        resolved_at=now,
        resolved_by_user_id=actor_user_id,
    )


def build_legacy_package_resolution_response(
    session: Session,
    *,
    order: CommercialOrderRecord,
    payment: CommercialPaymentRecord | None = None,
    order_line: CommercialOrderLineRecord | None = None,
) -> CommercialLegacyPackageResolutionResponse | None:
    resolved_line = order_line or _order_line(session, order)
    resolved_payment = payment or _latest_payment_for_order(session, order)
    state = _legacy_package_resolution_state(order, payment=resolved_payment)
    if not state:
        package, resolution_strategy = resolve_paid_order_package(session, order=order, order_line=resolved_line)
        if package is not None or resolution_strategy != "ambiguous_product_candidates":
            return None
        state = {
            "status": LEGACY_PACKAGE_RESOLUTION_PENDING,
            "reason": resolution_strategy,
            "detected_at": (
                order.paid_at or (resolved_payment.updated_at if resolved_payment is not None else None) or order.updated_at
            ).isoformat(),
            "candidate_package_codes": [item.package_code for item in _candidate_packages_for_order(session, order, order_line=resolved_line)],
        }
    product_key = _order_product_key(order, resolved_line)
    candidate_packages = [
        _serialize_legacy_package_candidate(item, order_product_key=product_key)
        for item in _candidate_packages_for_order(session, order, order_line=resolved_line)
    ]
    selected_package_code = str(state.get("selected_package_code") or order.metadata_payload.get("package_code") or "").strip()
    resolved_by_user_id = _parse_uuid(state.get("resolved_by_user_id"))
    return CommercialLegacyPackageResolutionResponse(
        order_id=order.id,
        workspace_id=order.workspace_id,
        session_id=order.session_id,
        payment_id=resolved_payment.id if resolved_payment is not None else None,
        product_key=product_key,
        status=_normalize_legacy_resolution_status(state.get("status")),
        reason=str(state.get("reason") or "ambiguous_product_candidates"),
        provider=order.provider,
        checkout_ref=order.checkout_ref,
        currency=order.currency,
        total_cents=order.total_cents,
        selected_package_code=selected_package_code,
        resolution_note=str(state.get("resolution_note") or ""),
        package_credit_applied=isinstance(order.metadata_payload.get("package_credit"), dict),
        candidate_packages=candidate_packages,
        created_at=order.created_at,
        paid_at=order.paid_at,
        detected_at=_parse_iso_datetime(state.get("detected_at")),
        resolved_at=_parse_iso_datetime(state.get("resolved_at")),
        resolved_by_user_id=resolved_by_user_id,
    )


def _build_credit_grant(
    session: Session,
    *,
    order: CommercialOrderRecord,
    package: CommercialPackageCatalogRecord,
    product_key: str,
    units: int,
    paid_at: datetime,
) -> PackageCreditGrant:
    if package.package_type == CommercialPackageType.one_time:
        ends_at = paid_at + timedelta(days=package.validity_days) if package.validity_days else None
        return PackageCreditGrant(
            product_key=product_key,
            units=units,
            source_kind=CommercialQuotaSourceKind.one_time,
            bucket_key=f"one-time:{package.package_code}:{product_key}:{order.id}",
            source_ref=f"one-time:{package.package_code}:{product_key}",
            starts_at=paid_at,
            ends_at=ends_at,
        )

    cycle_delta = _subscription_cycle_delta(package)
    source_ref = f"subscription:{package.package_code}:{product_key}"
    bucket_key = f"subscription:{package.package_code}:{product_key}:{order.id}"
    starts_at = paid_at
    existing_buckets = list_balance_buckets(
        session,
        workspace_id=order.workspace_id,
        product_key=product_key,
        at_time=paid_at,
    )
    related_ends = [
        bucket.ends_at
        for bucket in existing_buckets
        if bucket.source_kind == CommercialQuotaSourceKind.subscription
        and bucket.source_ref == source_ref
        and bucket.bucket_key != bucket_key
        and bucket.ends_at is not None
    ]
    if related_ends:
        latest_end = max(related_ends)
        if latest_end > starts_at:
            starts_at = latest_end
    return PackageCreditGrant(
        product_key=product_key,
        units=units,
        source_kind=CommercialQuotaSourceKind.subscription,
        bucket_key=bucket_key,
        source_ref=source_ref,
        starts_at=starts_at,
        ends_at=starts_at + cycle_delta,
    )


def _subscription_cycle_delta(package: CommercialPackageCatalogRecord) -> timedelta:
    if package.validity_days and package.validity_days > 0:
        return timedelta(days=package.validity_days)
    normalized_cycle = package.billing_cycle.strip().lower()
    known_cycles = {
        "daily": 1,
        "day": 1,
        "weekly": 7,
        "week": 7,
        "biweekly": 14,
        "fortnightly": 14,
        "monthly": 30,
        "month": 30,
        "quarterly": 90,
        "quarter": 90,
        "semiannual": 182,
        "semiannually": 182,
        "annual": 365,
        "annually": 365,
        "yearly": 365,
        "year": 365,
    }
    if normalized_cycle in known_cycles:
        return timedelta(days=known_cycles[normalized_cycle])
    return timedelta(days=DEFAULT_SUBSCRIPTION_CYCLE_DAYS)


def _order_line(session: Session, order: CommercialOrderRecord) -> CommercialOrderLineRecord | None:
    return session.exec(select(CommercialOrderLineRecord).where(CommercialOrderLineRecord.order_id == order.id)).first()


def _order_product_key(order: CommercialOrderRecord, order_line: CommercialOrderLineRecord | None = None) -> str:
    if order_line is not None and order_line.product_key:
        return order_line.product_key
    if isinstance(order.metadata_payload, dict):
        return str(order.metadata_payload.get("product_key") or "").strip()
    return ""


def _candidate_packages_for_order(
    session: Session,
    order: CommercialOrderRecord,
    *,
    order_line: CommercialOrderLineRecord | None = None,
) -> list[CommercialPackageCatalogRecord]:
    product_key = _order_product_key(order, order_line)
    if not product_key:
        return []
    return list_enabled_packages_for_product(session, product_key=product_key)


def _latest_payment_for_order(session: Session, order: CommercialOrderRecord) -> CommercialPaymentRecord | None:
    return session.exec(
        select(CommercialPaymentRecord)
        .where(CommercialPaymentRecord.order_id == order.id)
        .order_by(CommercialPaymentRecord.updated_at.desc(), CommercialPaymentRecord.created_at.desc())
    ).first()


def _legacy_package_resolution_state(
    order: CommercialOrderRecord,
    *,
    payment: CommercialPaymentRecord | None = None,
) -> dict[str, Any]:
    if isinstance(order.metadata_payload, dict):
        payload = order.metadata_payload.get(LEGACY_PACKAGE_RESOLUTION_KEY)
        if isinstance(payload, dict):
            return dict(payload)
    if payment is not None and isinstance(payment.metadata_payload, dict):
        payload = payment.metadata_payload.get(LEGACY_PACKAGE_RESOLUTION_KEY)
        if isinstance(payload, dict):
            return dict(payload)
    return {}


def _persist_legacy_package_resolution_state(
    *,
    order: CommercialOrderRecord,
    payment: CommercialPaymentRecord | None,
    state: dict[str, Any],
) -> None:
    normalized = dict(state)
    order.metadata_payload = {**dict(order.metadata_payload or {}), LEGACY_PACKAGE_RESOLUTION_KEY: normalized}
    if payment is not None:
        payment.metadata_payload = {**dict(payment.metadata_payload or {}), LEGACY_PACKAGE_RESOLUTION_KEY: normalized}


def _serialize_legacy_package_candidate(
    package: CommercialPackageCatalogRecord,
    *,
    order_product_key: str,
) -> CommercialLegacyPackageResolutionCandidateResponse:
    return CommercialLegacyPackageResolutionCandidateResponse(
        package_code=package.package_code,
        display_name=package.display_name,
        package_type=package.package_type,
        product_key=package.product_key,
        granted_units_for_order_product=package_units_for_product(package, order_product_key),
        granted_units_blueprint_pro=package_units_for_product(package, "blueprint_pro"),
        granted_units_acp=package_units_for_product(package, "acp"),
        offer_ref=package.offer_code.strip()
        or package.plan_code.strip()
        or package.hotmart_product_id.strip()
        or package.hotmart_product_ucode.strip(),
    )


def _normalize_legacy_resolution_status(raw: Any) -> str:
    normalized = str(raw or "").strip().lower()
    if normalized == LEGACY_PACKAGE_RESOLUTION_RESOLVED:
        return LEGACY_PACKAGE_RESOLUTION_RESOLVED
    return LEGACY_PACKAGE_RESOLUTION_PENDING


def _parse_iso_datetime(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _parse_uuid(raw: Any) -> UUID | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return UUID(text)
    except ValueError:
        return None
