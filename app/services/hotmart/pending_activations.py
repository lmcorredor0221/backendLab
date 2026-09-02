from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from sqlmodel import Session, select

from app.models import (
    CommercialEntitlementRecord,
    CommercialEntitlementSource,
    CommercialEntitlementStatus,
    CommercialOrderLineRecord,
    CommercialOrderRecord,
    CommercialOrderStatus,
    CommercialPackageCatalogRecord,
    CommercialPaymentRecord,
    CommercialPaymentStatus,
    HotmartPendingActivationClaimRequest,
    HotmartPendingActivationPublicResponse,
    HotmartPendingActivationRecord,
    HotmartPendingActivationResponse,
    HotmartPendingActivationStatus,
    HotmartProductMappingRecord,
    SessionRecord,
    UserRecord,
    utc_now,
)
from app.services.commerce_service import (
    apply_package_credits_from_paid_order,
    get_price,
    get_product,
    get_today_trm_data,
    record_commercial_event,
    settle_open_debts_from_paid_order,
    tier_rank,
)
from app.services.commercial_catalog_service import get_package_catalog_entry, package_units_for_product
from app.services.hotmart.auth import normalize_hotmart_environment
from app.services.runtime_access_control import is_platform_admin
from app.services.workspace_membership_service import get_effective_workspace_membership


@dataclass(frozen=True)
class _AdoptionContext:
    product_key: str
    price_code: str
    package: CommercialPackageCatalogRecord | None


def serialize_hotmart_pending_activation(record: HotmartPendingActivationRecord) -> HotmartPendingActivationResponse:
    return HotmartPendingActivationResponse(
        id=record.id,
        source_workspace_id=record.source_workspace_id,
        environment=record.environment,
        status=record.status,
        provider_ref=record.provider_ref,
        event_id=record.event_id,
        hotmart_product_id=record.hotmart_product_id,
        hotmart_product_ucode=record.hotmart_product_ucode,
        offer_code=record.offer_code,
        plan_code=record.plan_code,
        product_key=record.product_key,
        package_code=record.package_code,
        resolution_strategy=record.resolution_strategy,
        buyer_name=record.buyer_name,
        buyer_email=record.buyer_email,
        currency=record.currency,
        amount_cents=record.amount_cents,
        activation_token=record.activation_token,
        claimed_by_user_id=record.claimed_by_user_id,
        claimed_workspace_id=record.claimed_workspace_id,
        claimed_session_id=record.claimed_session_id,
        adopted_order_id=record.adopted_order_id,
        adopted_payment_id=record.adopted_payment_id,
        claimed_at=record.claimed_at,
        canceled_at=record.canceled_at,
        created_at=record.created_at,
        updated_at=record.updated_at,
        metadata=dict(record.metadata_payload or {}),
    )


def get_hotmart_pending_activation_record(
    session: Session,
    *,
    activation_token: str,
) -> HotmartPendingActivationRecord:
    token = activation_token.strip()
    if not token:
        raise ValueError("activation_token is required.")
    record = session.exec(
        select(HotmartPendingActivationRecord).where(HotmartPendingActivationRecord.activation_token == token)
    ).first()
    if record is None:
        raise ValueError("Hotmart pending activation was not found.")
    return record


def get_hotmart_pending_activation_public(
    session: Session,
    *,
    activation_token: str,
) -> HotmartPendingActivationPublicResponse:
    record = get_hotmart_pending_activation_record(session, activation_token=activation_token)
    resolved_product_key = ""
    claim_status_message = ""
    can_bootstrap = False

    if record.status == HotmartPendingActivationStatus.canceled:
        claim_status_message = "Esta activacion de Hotmart fue cancelada."
    elif record.status == HotmartPendingActivationStatus.claimed or record.claimed_session_id is not None:
        claim_status_message = "Esta compra ya fue activada en LAB."
    else:
        try:
            adoption = _resolve_adoption_context(session, record=record)
        except ValueError as exc:
            claim_status_message = str(exc)
        else:
            resolved_product_key = adoption.product_key
            can_bootstrap = True
            claim_status_message = "Tu compra esta lista para activarse en LAB."

    return HotmartPendingActivationPublicResponse(
        activation_token=record.activation_token,
        status=record.status,
        buyer_name=record.buyer_name,
        buyer_email=record.buyer_email,
        buyer_email_masked=_mask_email(record.buyer_email),
        product_key=record.product_key,
        resolved_product_key=resolved_product_key,
        package_code=record.package_code,
        display_name=_display_name_for_pending_activation(
            session,
            record=record,
            resolved_product_key=resolved_product_key,
        ),
        resolution_strategy=record.resolution_strategy,
        currency=record.currency,
        amount_cents=record.amount_cents,
        can_bootstrap=can_bootstrap,
        already_claimed=record.status == HotmartPendingActivationStatus.claimed or record.claimed_session_id is not None,
        claim_status_message=claim_status_message,
        claimed_session_id=record.claimed_session_id,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def register_pending_hotmart_activation(
    session: Session,
    *,
    payload: dict[str, Any],
    webhook_event_id: UUID,
    event_id: str,
    source_workspace_id: UUID,
    environment: str,
    transaction: str,
) -> HotmartPendingActivationRecord:
    env = normalize_hotmart_environment(environment)
    provider_ref = (transaction or event_id).strip()
    existing = session.exec(
        select(HotmartPendingActivationRecord).where(
            HotmartPendingActivationRecord.source_workspace_id == source_workspace_id,
            HotmartPendingActivationRecord.environment == env,
            HotmartPendingActivationRecord.provider_ref == provider_ref,
        )
    ).first()
    if existing is not None:
        if existing.webhook_event_id is None:
            existing.webhook_event_id = webhook_event_id
            existing.updated_at = utc_now()
            session.add(existing)
            session.flush()
        return existing

    hotmart_product_id = _first_string(
        payload,
        ("data", "product", "id"),
        ("data", "purchase", "product", "id"),
    )
    hotmart_product_ucode = _first_string(
        payload,
        ("data", "product", "ucode"),
        ("data", "purchase", "product", "ucode"),
    )
    offer_code = _first_string(
        payload,
        ("data", "offer", "code"),
        ("data", "purchase", "offer", "code"),
        ("data", "purchase", "offer_code"),
    )
    plan_code = _first_string(
        payload,
        ("data", "subscription", "plan", "code"),
        ("data", "purchase", "plan", "code"),
        ("data", "purchase", "plan_code"),
    )
    buyer_name = _first_string(
        payload,
        ("name",),
        ("data", "buyer", "name"),
        ("data", "purchase", "buyer", "name"),
        ("data", "purchase", "subscriber", "name"),
        ("data", "purchase", "customer", "name"),
    )
    buyer_email = _first_string(
        payload,
        ("email",),
        ("data", "buyer", "email"),
        ("data", "purchase", "buyer", "email"),
        ("data", "purchase", "subscriber", "email"),
        ("data", "purchase", "customer", "email"),
    ).lower()
    buyer_document = _first_string(
        payload,
        ("data", "buyer", "document"),
        ("data", "purchase", "buyer", "document"),
        ("data", "purchase", "subscriber", "document"),
        ("data", "purchase", "customer", "document"),
    )
    package = _resolve_package_catalog_match(
        session,
        environment=env,
        hotmart_product_id=hotmart_product_id,
        hotmart_product_ucode=hotmart_product_ucode,
        offer_code=offer_code,
        plan_code=plan_code,
    )
    mapping = _resolve_product_mapping_match(
        session,
        source_workspace_id=source_workspace_id,
        environment=env,
        hotmart_product_id=hotmart_product_id,
        hotmart_product_ucode=hotmart_product_ucode,
        offer_code=offer_code,
        plan_code=plan_code,
    )
    resolved_product_key = ""
    if mapping is not None:
        resolved_product_key = mapping.internal_product_key.strip()
    elif package is not None:
        resolved_product_key = package.product_key.strip()

    resolution_strategy = "manual_resolution_required"
    if package is not None and mapping is not None:
        resolution_strategy = "package_catalog_and_mapping"
    elif package is not None:
        resolution_strategy = "package_catalog"
    elif mapping is not None:
        resolution_strategy = "product_mapping"

    record = HotmartPendingActivationRecord(
        source_workspace_id=source_workspace_id,
        environment=env,
        status=HotmartPendingActivationStatus.pending_activation,
        provider_ref=provider_ref,
        event_id=event_id.strip(),
        webhook_event_id=webhook_event_id,
        hotmart_product_id=hotmart_product_id,
        hotmart_product_ucode=hotmart_product_ucode,
        offer_code=offer_code,
        plan_code=plan_code,
        product_key=resolved_product_key,
        package_code=package.package_code if package is not None else "",
        resolution_strategy=resolution_strategy,
        buyer_name=buyer_name,
        buyer_email=buyer_email,
        buyer_document=buyer_document,
        currency=_extract_currency(payload, fallback="USD"),
        amount_cents=_extract_amount_cents(payload),
        activation_token=uuid4().hex,
        metadata_payload={
            "source": "hotmart_webhook",
            "mapped_product_key": mapping.internal_product_key if mapping is not None else "",
            "resolved_package_code": package.package_code if package is not None else "",
            "workspace_origin": str(source_workspace_id),
        },
    )
    session.add(record)
    session.flush()
    record_commercial_event(
        session,
        workspace_id=source_workspace_id,
        session_id=None,
        user_id=None,
        event_key="hotmart_pending_activation_registered",
        product_key=record.product_key,
        source="hotmart_webhook",
        revenue_cents=record.amount_cents,
        currency=record.currency,
        metadata={
            "pending_activation_id": str(record.id),
            "provider_ref": record.provider_ref,
            "package_code": record.package_code,
            "resolution_strategy": record.resolution_strategy,
            "buyer_email": record.buyer_email,
        },
        correlation_id=record.event_id or record.provider_ref,
    )
    session.flush()
    return record


def list_user_pending_hotmart_activations(
    session: Session,
    *,
    current_user: UserRecord,
    limit: int = 50,
) -> list[HotmartPendingActivationResponse]:
    email = current_user.email.strip().lower()
    if not email:
        return []
    rows = session.exec(
        select(HotmartPendingActivationRecord)
        .where(
            HotmartPendingActivationRecord.buyer_email == email,
            HotmartPendingActivationRecord.status == HotmartPendingActivationStatus.pending_activation,
        )
        .order_by(HotmartPendingActivationRecord.created_at.desc())
        .limit(max(1, min(limit, 100)))
    ).all()
    return [serialize_hotmart_pending_activation(row) for row in rows]


def list_pending_hotmart_activations(
    session: Session,
    *,
    source_workspace_id: UUID | None = None,
    status_filter: str = "",
    limit: int = 100,
) -> list[HotmartPendingActivationResponse]:
    statement = select(HotmartPendingActivationRecord)
    if source_workspace_id is not None:
        statement = statement.where(HotmartPendingActivationRecord.source_workspace_id == source_workspace_id)
    normalized_status = status_filter.strip().lower()
    valid_statuses = {item.value for item in HotmartPendingActivationStatus}
    if normalized_status in valid_statuses:
        statement = statement.where(HotmartPendingActivationRecord.status == normalized_status)
    rows = session.exec(
        statement.order_by(HotmartPendingActivationRecord.created_at.desc(), HotmartPendingActivationRecord.provider_ref.asc())
        .limit(max(1, min(limit, 200)))
    ).all()
    return [serialize_hotmart_pending_activation(row) for row in rows]


def claim_hotmart_pending_activation(
    session: Session,
    *,
    activation_token: str,
    payload: HotmartPendingActivationClaimRequest,
    current_user: UserRecord,
) -> HotmartPendingActivationResponse:
    record = get_hotmart_pending_activation_record(session, activation_token=activation_token)
    if record.status == HotmartPendingActivationStatus.canceled:
        raise ValueError("Hotmart pending activation is canceled.")

    target_session = session.get(SessionRecord, payload.session_id)
    if target_session is None:
        raise ValueError("Target session was not found.")
    if target_session.workspace_id is None:
        raise ValueError("Target session is not attached to a workspace.")

    membership = get_effective_workspace_membership(
        session,
        workspace_id=target_session.workspace_id,
        user_id=current_user.id,
    )
    if membership is None:
        raise PermissionError("You do not have access to the target workspace.")

    platform_admin = is_platform_admin(session, current_user)
    buyer_email = record.buyer_email.strip().lower()
    if buyer_email and buyer_email != current_user.email.strip().lower() and not platform_admin:
        raise PermissionError("This Hotmart purchase belongs to another buyer email.")
    if record.claimed_session_id is not None and record.claimed_session_id != target_session.id:
        raise ValueError("Hotmart pending activation is already linked to another session.")
    if record.status == HotmartPendingActivationStatus.claimed and record.adopted_order_id is not None:
        return serialize_hotmart_pending_activation(record)

    existing_payment = session.exec(
        select(CommercialPaymentRecord).where(
            CommercialPaymentRecord.provider == "hotmart",
            CommercialPaymentRecord.provider_payment_id == record.provider_ref,
        )
    ).first()
    if existing_payment is not None:
        existing_order = session.get(CommercialOrderRecord, existing_payment.order_id)
        if existing_order is None:
            raise ValueError("Existing Hotmart payment is not attached to a valid commercial order.")
        if existing_order.session_id != target_session.id:
            raise ValueError("This Hotmart purchase was already adopted by another session.")
        _mark_claimed(
            session,
            record=record,
            current_user=current_user,
            target_session=target_session,
            order=existing_order,
            payment=existing_payment,
        )
        return serialize_hotmart_pending_activation(record)

    adoption = _resolve_adoption_context(session, record=record)
    product = get_product(session, adoption.product_key)
    price = get_price(session, adoption.product_key, adoption.price_code)
    amount_cents = max(
        0,
        record.amount_cents or price.unit_amount_cents or price.unit_amount_usd_cents,
    )
    currency = (record.currency or price.currency or "USD").strip().upper() or "USD"
    checkout_ref = f"hotmart-ext-{record.id.hex[:20]}"
    idempotency_key = _short_hash(f"hotmart-claim:{record.id}:{target_session.id}")
    commercial_snapshot = _build_external_sale_snapshot(
        product_key=product.product_key,
        product_version=product.version,
        price_code=price.price_code,
        price_version=price.version,
        provider="hotmart",
        subtotal_cents=amount_cents,
        total_cents=amount_cents,
        currency=currency,
        amount_usd_base_cents=price.unit_amount_usd_cents if price.unit_amount_usd_cents > 0 else price.unit_amount_cents,
    )

    order = CommercialOrderRecord(
        workspace_id=target_session.workspace_id,
        session_id=target_session.id,
        buyer_user_id=current_user.id,
        status=CommercialOrderStatus.paid,
        currency=currency,
        subtotal_cents=amount_cents,
        total_cents=amount_cents,
        provider="hotmart",
        checkout_ref=checkout_ref,
        checkout_url="",
        idempotency_key=idempotency_key,
        metadata_payload={
            "product_key": product.product_key,
            "price_code": price.price_code,
            "provider": "hotmart",
            "package_code": adoption.package.package_code if adoption.package is not None else record.package_code,
            "package_type": adoption.package.package_type.value if adoption.package is not None else "",
            "is_upgrade": False,
            "upgrade_discount_cents": 0,
            "base_product_cents": price.unit_amount_cents,
            "commercial_snapshot": commercial_snapshot,
            "external_origin": True,
            "hotmart_pending_activation_id": str(record.id),
            "hotmart_source_workspace_id": str(record.source_workspace_id),
            "hotmart_provider_ref": record.provider_ref,
            "hotmart_resolution_strategy": record.resolution_strategy,
            "buyer_email": record.buyer_email,
        },
        paid_at=utc_now(),
    )
    session.add(order)
    session.flush()

    line = CommercialOrderLineRecord(
        order_id=order.id,
        product_key=product.product_key,
        price_code=price.price_code,
        quantity=1,
        unit_amount_cents=amount_cents,
        total_amount_cents=amount_cents,
        metadata_payload={
            "product_version": product.version,
            "price_version": price.version,
            "package_code": adoption.package.package_code if adoption.package is not None else record.package_code,
            "provider_authoritative_amount_cents": amount_cents,
            "catalog_unit_amount_cents": price.unit_amount_cents,
        },
    )
    session.add(line)
    session.flush()

    payment = CommercialPaymentRecord(
        workspace_id=target_session.workspace_id,
        session_id=target_session.id,
        order_id=order.id,
        provider="hotmart",
        provider_payment_id=record.provider_ref,
        provider_checkout_ref=order.checkout_ref,
        status=CommercialPaymentStatus.succeeded,
        amount_cents=amount_cents,
        currency=currency,
        idempotency_key=_short_hash(f"hotmart-payment:{record.id}:{target_session.id}"),
        metadata_payload={
            "event_id": record.event_id,
            "pending_activation_id": str(record.id),
            "source_workspace_id": str(record.source_workspace_id),
        },
    )
    session.add(payment)
    session.flush()

    settle_open_debts_from_paid_order(
        session,
        order=order,
        payment=payment,
        actor_user_id=current_user.id,
    )
    apply_package_credits_from_paid_order(
        session,
        order=order,
        payment=payment,
        actor_user_id=current_user.id,
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
            metadata_payload={
                "provider": "hotmart",
                "pending_activation_id": str(record.id),
                "source_workspace_id": str(record.source_workspace_id),
            },
        )
    entitlement.status = CommercialEntitlementStatus.active
    entitlement.payment_id = payment.id
    entitlement.updated_at = utc_now()
    session.add(entitlement)

    if tier_rank(product.tier) > tier_rank(target_session.commercial_tier):
        target_session.commercial_tier = product.tier
    target_session.updated_at = utc_now()
    session.add(target_session)

    record_commercial_event(
        session,
        workspace_id=order.workspace_id,
        session_id=order.session_id,
        user_id=current_user.id,
        event_key="hotmart_external_sale_claimed",
        product_key=product.product_key,
        source="hotmart_pending_activation",
        revenue_cents=payment.amount_cents,
        currency=payment.currency,
        metadata={
            "order_id": str(order.id),
            "payment_id": str(payment.id),
            "pending_activation_id": str(record.id),
            "source_workspace_id": str(record.source_workspace_id),
            "package_code": adoption.package.package_code if adoption.package is not None else record.package_code,
        },
        correlation_id=record.provider_ref,
    )

    from app.services.product_processing.product_build_activation_service import activate_product_builds_for_paid_order

    activate_product_builds_for_paid_order(
        session,
        order=order,
        current_user=current_user,
        source="hotmart_pending_activation",
    )
    _mark_claimed(
        session,
        record=record,
        current_user=current_user,
        target_session=target_session,
        order=order,
        payment=payment,
    )
    return serialize_hotmart_pending_activation(record)


def _mark_claimed(
    session: Session,
    *,
    record: HotmartPendingActivationRecord,
    current_user: UserRecord,
    target_session: SessionRecord,
    order: CommercialOrderRecord,
    payment: CommercialPaymentRecord,
) -> None:
    now = utc_now()
    record.status = HotmartPendingActivationStatus.claimed
    record.claimed_by_user_id = current_user.id
    record.claimed_workspace_id = target_session.workspace_id
    record.claimed_session_id = target_session.id
    record.adopted_order_id = order.id
    record.adopted_payment_id = payment.id
    record.claimed_at = record.claimed_at or now
    record.updated_at = now
    session.add(record)
    session.flush()


def _resolve_adoption_context(
    session: Session,
    *,
    record: HotmartPendingActivationRecord,
) -> _AdoptionContext:
    package = get_package_catalog_entry(session, package_code=record.package_code) if record.package_code else None
    candidate_product_keys = [
        record.product_key.strip(),
        str(record.metadata_payload.get("mapped_product_key") or "").strip(),
        _best_product_key_from_package(package) if package is not None else "",
    ]
    for product_key in candidate_product_keys:
        if not product_key:
            continue
        try:
            price = get_price(session, product_key)
        except ValueError:
            continue
        return _AdoptionContext(product_key=product_key, price_code=price.price_code, package=package)
    raise ValueError(
        "The Hotmart sale is not fully parametrized. Configure the product or package mapping before claiming it."
    )


def _best_product_key_from_package(package: CommercialPackageCatalogRecord | None) -> str:
    if package is None:
        return ""
    if package_units_for_product(package, "acp") > 0:
        return "acp"
    if package_units_for_product(package, "blueprint_pro") > 0:
        return "blueprint_pro"
    return package.product_key.strip()


def _resolve_package_catalog_match(
    session: Session,
    *,
    environment: str,
    hotmart_product_id: str,
    hotmart_product_ucode: str,
    offer_code: str,
    plan_code: str,
) -> CommercialPackageCatalogRecord | None:
    rows = session.exec(
        select(CommercialPackageCatalogRecord).where(
            CommercialPackageCatalogRecord.enabled == True,  # noqa: E712
            CommercialPackageCatalogRecord.hotmart_environment == environment,
        )
    ).all()
    scored: list[tuple[int, CommercialPackageCatalogRecord]] = []
    for row in rows:
        score = _ref_match_score(
            hotmart_product_id=hotmart_product_id,
            configured_product_id=row.hotmart_product_id,
            hotmart_product_ucode=hotmart_product_ucode,
            configured_product_ucode=row.hotmart_product_ucode,
            offer_code=offer_code,
            configured_offer_code=row.offer_code,
            plan_code=plan_code,
            configured_plan_code=row.plan_code,
        )
        if score > 0:
            scored.append((score, row))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1].recommendation_priority, item[1].package_code))
    return scored[0][1]


def _resolve_product_mapping_match(
    session: Session,
    *,
    source_workspace_id: UUID,
    environment: str,
    hotmart_product_id: str,
    hotmart_product_ucode: str,
    offer_code: str,
    plan_code: str,
) -> HotmartProductMappingRecord | None:
    rows = session.exec(
        select(HotmartProductMappingRecord).where(
            HotmartProductMappingRecord.workspace_id == source_workspace_id,
            HotmartProductMappingRecord.environment == environment,
            HotmartProductMappingRecord.is_active == True,  # noqa: E712
        )
    ).all()
    scored: list[tuple[int, HotmartProductMappingRecord]] = []
    for row in rows:
        score = _ref_match_score(
            hotmart_product_id=hotmart_product_id,
            configured_product_id=row.hotmart_product_id,
            hotmart_product_ucode=hotmart_product_ucode,
            configured_product_ucode=row.hotmart_product_ucode,
            offer_code=offer_code,
            configured_offer_code=row.offer_code,
            plan_code=plan_code,
            configured_plan_code=row.plan_code,
        )
        if score > 0:
            scored.append((score, row))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1].internal_product_key))
    return scored[0][1]


def _ref_match_score(
    *,
    hotmart_product_id: str,
    configured_product_id: str,
    hotmart_product_ucode: str,
    configured_product_ucode: str,
    offer_code: str,
    configured_offer_code: str,
    plan_code: str,
    configured_plan_code: str,
) -> int:
    score = 0
    matched = False
    ref_pairs = (
        (plan_code, configured_plan_code, 16),
        (offer_code, configured_offer_code, 8),
        (hotmart_product_id, configured_product_id, 4),
        (hotmart_product_ucode, configured_product_ucode, 2),
    )
    for incoming, configured, weight in ref_pairs:
        normalized_incoming = incoming.strip()
        normalized_configured = configured.strip()
        if normalized_incoming and normalized_configured and normalized_incoming != normalized_configured:
            return 0
        if normalized_incoming and normalized_configured and normalized_incoming == normalized_configured:
            score += weight
            matched = True
    return score if matched else 0


def _extract_amount_cents(payload: dict[str, Any]) -> int:
    raw_value = (
        _first_value(payload, ("data", "purchase", "price", "value"))
        or _first_value(payload, ("data", "purchase", "offer", "price", "value"))
        or _first_value(payload, ("price", "value"))
        or 0
    )
    try:
        return max(0, int(round(float(raw_value) * 100)))
    except (TypeError, ValueError):
        return 0


def _extract_currency(payload: dict[str, Any], *, fallback: str) -> str:
    return (
        _first_string(
            payload,
            ("data", "purchase", "price", "currency_code"),
            ("data", "purchase", "offer", "price", "currency_code"),
            ("price", "currency_code"),
        ).upper()
        or fallback.strip().upper()
        or "USD"
    )


def _first_string(payload: dict[str, Any], *paths: tuple[str, ...]) -> str:
    for path in paths:
        value = _first_value(payload, path)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _first_value(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:64]


def _build_external_sale_snapshot(
    *,
    product_key: str,
    product_version: int,
    price_code: str,
    price_version: int,
    provider: str,
    subtotal_cents: int,
    total_cents: int,
    currency: str,
    amount_usd_base_cents: int,
) -> dict[str, object]:
    trm_info = get_today_trm_data()
    return {
        "contract_version": "commercial-order-snapshot.v1",
        "product_key": product_key,
        "product_version": product_version,
        "price_code": price_code,
        "price_version": price_version,
        "provider": provider,
        "subtotal_cents": subtotal_cents,
        "discount_cents": 0,
        "total_cents": total_cents,
        "checkout_amount_cents": total_cents,
        "checkout_currency": currency,
        "amount_usd_base_cents": amount_usd_base_cents,
        "discount_usd_cents": 0,
        "net_amount_usd_cents": amount_usd_base_cents,
        "trm_cop_frozen": trm_info["rate"],
        "trm_effective_date": trm_info["date"],
        "is_upgrade": False,
        "success_url": "",
        "cancel_url": "",
        "pricing_source": "hotmart_provider_authoritative",
    }


def _display_name_for_pending_activation(
    session: Session,
    *,
    record: HotmartPendingActivationRecord,
    resolved_product_key: str = "",
) -> str:
    if record.package_code:
        package = get_package_catalog_entry(session, package_code=record.package_code)
        if package is not None and package.display_name.strip():
            return package.display_name.strip()

    for candidate in (
        resolved_product_key.strip(),
        record.product_key.strip(),
        str(record.metadata_payload.get("mapped_product_key") or "").strip(),
    ):
        if not candidate:
            continue
        try:
            product = get_product(session, candidate)
        except ValueError:
            continue
        if product.name.strip():
            return product.name.strip()

    if record.package_code.strip():
        return record.package_code.strip()
    if record.product_key.strip():
        return record.product_key.strip().replace("_", " ").title()
    return "Compra Hotmart"


def _mask_email(email: str) -> str:
    normalized = email.strip()
    if not normalized or "@" not in normalized:
        return ""
    local_part, domain = normalized.split("@", 1)
    if len(local_part) <= 2:
        masked_local = f"{local_part[:1]}*"
    else:
        masked_local = f"{local_part[:2]}{'*' * max(1, len(local_part) - 2)}"
    return f"{masked_local}@{domain}"
