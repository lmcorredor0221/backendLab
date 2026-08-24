from __future__ import annotations

from uuid import UUID

from sqlmodel import Session, select

from app.models import (
    CommercialPackageCatalogRecord,
    CommercialPackageCatalogResponse,
    CommercialPackageCatalogUpsertRequest,
    CommercialPackageRecommendationResponse,
    utc_now,
)
from app.services.commercial_quota_service import resolve_effective_quota_config


def serialize_package_catalog_entry(record: CommercialPackageCatalogRecord) -> CommercialPackageCatalogResponse:
    return CommercialPackageCatalogResponse(
        id=record.id,
        package_code=record.package_code,
        display_name=record.display_name,
        product_key=record.product_key,
        package_type=record.package_type,
        enabled=record.enabled,
        granted_units=record.granted_units,
        granted_units_blueprint_pro=record.granted_units_blueprint_pro,
        granted_units_acp=record.granted_units_acp,
        validity_days=record.validity_days,
        billing_cycle=record.billing_cycle,
        renewal_policy=record.renewal_policy,
        recommendation_priority=record.recommendation_priority,
        hotmart_environment=record.hotmart_environment,
        hotmart_product_id=record.hotmart_product_id,
        hotmart_product_ucode=record.hotmart_product_ucode,
        offer_code=record.offer_code,
        plan_code=record.plan_code,
        checkout_currency_mode=record.checkout_currency_mode,
        hotmart_price_strategy=record.hotmart_price_strategy,
        updated_at=record.updated_at,
    )


def _package_units_for_product(record: CommercialPackageCatalogRecord, product_key: str) -> int:
    normalized_product_key = product_key.strip().lower()
    if normalized_product_key == "blueprint_pro":
        return max(record.granted_units_blueprint_pro, record.granted_units if record.product_key == "blueprint_pro" else 0)
    if normalized_product_key == "acp":
        return max(record.granted_units_acp, record.granted_units if record.product_key == "acp" else 0)
    if record.product_key == product_key:
        return record.granted_units
    return 0


def package_units_for_product(record: CommercialPackageCatalogRecord, product_key: str) -> int:
    return _package_units_for_product(record, product_key)


def get_package_catalog_entry(
    session: Session,
    *,
    package_code: str,
    include_disabled: bool = False,
) -> CommercialPackageCatalogRecord | None:
    normalized_package_code = package_code.strip()
    if not normalized_package_code:
        return None
    statement = select(CommercialPackageCatalogRecord).where(CommercialPackageCatalogRecord.package_code == normalized_package_code)
    if not include_disabled:
        statement = statement.where(CommercialPackageCatalogRecord.enabled == True)  # noqa: E712
    return session.exec(statement).first()


def list_enabled_packages_for_product(
    session: Session,
    *,
    product_key: str,
) -> list[CommercialPackageCatalogRecord]:
    normalized_product_key = product_key.strip()
    if not normalized_product_key:
        return []
    rows = session.exec(
        select(CommercialPackageCatalogRecord)
        .where(CommercialPackageCatalogRecord.enabled == True)  # noqa: E712
        .order_by(
            CommercialPackageCatalogRecord.recommendation_priority.asc(),
            CommercialPackageCatalogRecord.package_code.asc(),
        )
    ).all()
    return [row for row in rows if _package_units_for_product(row, normalized_product_key) > 0]


def list_package_catalog(
    session: Session,
    *,
    product_key: str = "",
    include_disabled: bool = True,
) -> list[CommercialPackageCatalogResponse]:
    statement = select(CommercialPackageCatalogRecord)
    if product_key.strip():
        statement = statement.where(CommercialPackageCatalogRecord.product_key == product_key.strip())
    if not include_disabled:
        statement = statement.where(CommercialPackageCatalogRecord.enabled == True)  # noqa: E712
    rows = session.exec(
        statement.order_by(
            CommercialPackageCatalogRecord.recommendation_priority.asc(),
            CommercialPackageCatalogRecord.package_code.asc(),
        )
    ).all()
    return [serialize_package_catalog_entry(row) for row in rows]


def upsert_package_catalog_entry(
    session: Session,
    *,
    payload: CommercialPackageCatalogUpsertRequest,
) -> CommercialPackageCatalogResponse:
    package_code = payload.package_code.strip()
    if not package_code:
        raise ValueError("package_code is required.")
    product_key = payload.product_key.strip()
    if not product_key:
        raise ValueError("product_key is required.")
    record = session.exec(
        select(CommercialPackageCatalogRecord).where(CommercialPackageCatalogRecord.package_code == package_code)
    ).first()
    if record is None:
        record = CommercialPackageCatalogRecord(package_code=package_code)
    record.display_name = payload.display_name.strip() or package_code
    record.product_key = product_key
    record.package_type = payload.package_type
    record.enabled = payload.enabled
    record.granted_units = max(0, payload.granted_units)
    record.granted_units_blueprint_pro = max(0, payload.granted_units_blueprint_pro)
    record.granted_units_acp = max(0, payload.granted_units_acp)
    record.validity_days = payload.validity_days if payload.validity_days is None else max(1, payload.validity_days)
    record.billing_cycle = payload.billing_cycle.strip()
    record.renewal_policy = payload.renewal_policy.strip()
    record.recommendation_priority = max(0, payload.recommendation_priority)
    record.hotmart_environment = payload.hotmart_environment.strip() or "sandbox"
    record.hotmart_product_id = payload.hotmart_product_id.strip()
    record.hotmart_product_ucode = payload.hotmart_product_ucode.strip()
    record.offer_code = payload.offer_code.strip()
    record.plan_code = payload.plan_code.strip()
    record.checkout_currency_mode = payload.checkout_currency_mode.strip() or "workspace_preferred"
    record.hotmart_price_strategy = payload.hotmart_price_strategy.strip() or "provider_authoritative"
    record.metadata_payload = dict(payload.metadata)
    record.updated_at = utc_now()
    session.add(record)
    session.flush()
    return serialize_package_catalog_entry(record)


def recommend_package_for_product(
    session: Session,
    *,
    product_key: str,
    required_units: int = 1,
    workspace_id: UUID | None = None,
) -> CommercialPackageRecommendationResponse:
    requested_product_key = product_key.strip()
    normalized_required_units = max(1, required_units)
    strategy = "minimum_sufficient"
    if workspace_id is not None:
        try:
            strategy = resolve_effective_quota_config(
                session,
                workspace_id=workspace_id,
                product_key=requested_product_key,
            ).catalog_priority_strategy
        except ValueError:
            strategy = "minimum_sufficient"

    rows = session.exec(
        select(CommercialPackageCatalogRecord).where(CommercialPackageCatalogRecord.enabled == True)  # noqa: E712
    ).all()
    candidates: list[tuple[CommercialPackageCatalogRecord, int]] = []
    for row in rows:
        granted_units = _package_units_for_product(row, requested_product_key)
        if granted_units <= 0:
            continue
        candidates.append((row, granted_units))

    if not candidates:
        return CommercialPackageRecommendationResponse(
            requested_product_key=requested_product_key,
            required_units=normalized_required_units,
            recommendation_reason="No hay paquetes activos parametrizados para este producto.",
        )

    if strategy == "priority_first":
        candidates.sort(
            key=lambda item: (
                item[0].recommendation_priority,
                0 if item[1] >= normalized_required_units else 1,
                item[1],
                item[0].package_code,
            )
        )
    else:
        candidates.sort(
            key=lambda item: (
                0 if item[1] >= normalized_required_units else 1,
                item[1] if item[1] >= normalized_required_units else -item[1],
                item[0].recommendation_priority,
                item[0].package_code,
            )
        )

    selected, granted_units = candidates[0]
    reason = (
        "Paquete minimo suficiente segun configuracion efectiva."
        if granted_units >= normalized_required_units
        else "No hay paquete suficiente; se sugiere el de mayor cobertura disponible."
    )
    return CommercialPackageRecommendationResponse(
        requested_product_key=requested_product_key,
        required_units=normalized_required_units,
        package_code=selected.package_code,
        display_name=selected.display_name,
        package_type=selected.package_type,
        granted_units_for_product=granted_units,
        recommendation_priority=selected.recommendation_priority,
        recommendation_reason=reason,
        hotmart_environment=selected.hotmart_environment,
        hotmart_product_id=selected.hotmart_product_id,
        hotmart_product_ucode=selected.hotmart_product_ucode,
        offer_code=selected.offer_code,
        plan_code=selected.plan_code,
    )
