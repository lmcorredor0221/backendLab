from __future__ import annotations

from uuid import UUID

from sqlmodel import Session, select

from app.models import (
    CommerceProviderProductMappingRecord,
    CommerceProviderProductMappingResponse,
    CommerceProviderProductMappingUpsertRequest,
    utc_now,
)
from app.services.commerce_provider_registry import get_commerce_provider_registry
from app.services.commerce_provider_utils import (
    normalize_commerce_provider_environment,
    normalize_commerce_provider_key,
)


def serialize_commerce_provider_mapping(
    record: CommerceProviderProductMappingRecord,
) -> CommerceProviderProductMappingResponse:
    return CommerceProviderProductMappingResponse(
        id=record.id,
        workspace_id=record.workspace_id,
        provider_key=record.provider_key,
        environment=record.environment,  # type: ignore[arg-type]
        internal_product_key=record.internal_product_key,
        package_code=record.package_code,
        billing_mode=record.billing_mode,
        currency=record.currency,
        internal_unit_amount_usd_cents=record.internal_unit_amount_usd_cents,
        provider_product_id=record.provider_product_id,
        provider_plan_id=record.provider_plan_id,
        provider_price_id=record.provider_price_id,
        provider_payment_link_id=record.provider_payment_link_id,
        provider_offer_ref=record.provider_offer_ref,
        price_strategy=record.price_strategy,
        grants_tier=record.grants_tier,
        entitlement_scope=record.entitlement_scope,
        is_active=record.is_active,
        metadata=dict(record.metadata_payload or {}),
        updated_at=record.updated_at,
    )


def list_commerce_provider_mappings(
    session: Session,
    *,
    workspace_id: UUID,
    provider_key: str,
    environment: str = "sandbox",
) -> list[CommerceProviderProductMappingResponse]:
    provider = normalize_commerce_provider_key(provider_key)
    env = normalize_commerce_provider_environment(environment)
    get_commerce_provider_registry().require_definition(provider)
    records = session.exec(
        select(CommerceProviderProductMappingRecord)
        .where(
            CommerceProviderProductMappingRecord.workspace_id == workspace_id,
            CommerceProviderProductMappingRecord.provider_key == provider,
            CommerceProviderProductMappingRecord.environment == env,
        )
        .order_by(
            CommerceProviderProductMappingRecord.internal_product_key,
            CommerceProviderProductMappingRecord.package_code,
        )
    ).all()
    return [serialize_commerce_provider_mapping(record) for record in records]


def find_commerce_provider_mapping(
    session: Session,
    *,
    workspace_id: UUID,
    provider_key: str,
    environment: str,
    internal_product_key: str,
    package_code: str = "",
) -> CommerceProviderProductMappingRecord | None:
    provider = normalize_commerce_provider_key(provider_key)
    env = normalize_commerce_provider_environment(environment)
    product_key = internal_product_key.strip()
    package = package_code.strip()
    query = select(CommerceProviderProductMappingRecord).where(
        CommerceProviderProductMappingRecord.workspace_id == workspace_id,
        CommerceProviderProductMappingRecord.provider_key == provider,
        CommerceProviderProductMappingRecord.environment == env,
        CommerceProviderProductMappingRecord.internal_product_key == product_key,
        CommerceProviderProductMappingRecord.is_active == True,  # noqa: E712
    )
    if package:
        exact = session.exec(query.where(CommerceProviderProductMappingRecord.package_code == package)).first()
        if exact is not None:
            return exact
    return session.exec(query.where(CommerceProviderProductMappingRecord.package_code == "")).first()


def upsert_commerce_provider_mapping(
    session: Session,
    *,
    workspace_id: UUID,
    provider_key: str,
    payload: CommerceProviderProductMappingUpsertRequest,
) -> CommerceProviderProductMappingResponse:
    provider = normalize_commerce_provider_key(provider_key)
    env = normalize_commerce_provider_environment(payload.environment)
    get_commerce_provider_registry().require_definition(provider)
    product_key = payload.internal_product_key.strip()
    if not product_key:
        raise ValueError("internal_product_key is required.")
    package_code = payload.package_code.strip()
    billing_mode = payload.billing_mode.strip().lower() or "one_time"
    if billing_mode not in {"one_time", "subscription"}:
        raise ValueError("billing_mode must be one_time or subscription.")
    mapping = session.exec(
        select(CommerceProviderProductMappingRecord).where(
            CommerceProviderProductMappingRecord.workspace_id == workspace_id,
            CommerceProviderProductMappingRecord.provider_key == provider,
            CommerceProviderProductMappingRecord.environment == env,
            CommerceProviderProductMappingRecord.internal_product_key == product_key,
            CommerceProviderProductMappingRecord.package_code == package_code,
        )
    ).first()
    if mapping is None:
        mapping = CommerceProviderProductMappingRecord(
            workspace_id=workspace_id,
            provider_key=provider,
            environment=env,
            internal_product_key=product_key,
            package_code=package_code,
        )
    mapping.billing_mode = billing_mode
    mapping.currency = payload.currency.strip().upper() or "USD"
    mapping.internal_unit_amount_usd_cents = max(0, payload.internal_unit_amount_usd_cents)
    mapping.provider_product_id = payload.provider_product_id.strip()
    mapping.provider_plan_id = payload.provider_plan_id.strip()
    mapping.provider_price_id = payload.provider_price_id.strip()
    mapping.provider_payment_link_id = payload.provider_payment_link_id.strip()
    mapping.provider_offer_ref = payload.provider_offer_ref.strip()
    mapping.price_strategy = payload.price_strategy.strip() or "provider_authoritative"
    mapping.grants_tier = payload.grants_tier
    mapping.entitlement_scope = payload.entitlement_scope.strip() or "project"
    mapping.is_active = payload.is_active
    mapping.metadata_payload = dict(payload.metadata or {})
    mapping.updated_at = utc_now()
    session.add(mapping)
    session.flush()
    return serialize_commerce_provider_mapping(mapping)
