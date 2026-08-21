from __future__ import annotations

from typing import Any

from sqlmodel import Session, select

from app.models import CommercialOrderLineRecord, CommercialOrderRecord, CommercialOrderStatus, SessionRecord, UserRecord
from app.services.product_processing.contracts import ProductBuildProductKey, ProductBuildStatus
from app.services.product_processing.product_build_orchestrator import (
    ProductBuildOrchestrationOptions,
    ensure_product_build_orchestration,
)
from app.services.product_processing.acp_product_orchestration_service import ensure_acp_product_orchestration


ORDER_PRODUCT_TO_BUILD_PRODUCT: dict[str, ProductBuildProductKey] = {
    "blueprint_pro": ProductBuildProductKey.blueprint_pro,
    "acp": ProductBuildProductKey.acp,
}


def activate_product_builds_for_paid_order(
    db: Session,
    *,
    order: CommercialOrderRecord,
    current_user: UserRecord | None = None,
    source: str = "commerce_checkout",
) -> list[ProductBuildStatus]:
    """Create or resume product build runs when a paid order grants a product.

    Pending, failed or canceled orders intentionally do nothing; this keeps checkout
    pending as an access state instead of starting paid-product processing early.
    """

    if order.status != CommercialOrderStatus.paid or order.session_id is None:
        return []
    record = db.get(SessionRecord, order.session_id)
    if record is None:
        return []

    statuses: list[ProductBuildStatus] = []
    for product_key in _ordered_build_products_for_order(db, order):
        activation_payload = _activation_payload(order=order, product_key=product_key, source=source)
        if product_key == ProductBuildProductKey.acp:
            statuses.append(
                ensure_acp_product_orchestration(
                    db,
                    record=record,
                    current_user=current_user,
                    activation_payload=activation_payload,
                )
            )
            continue
        statuses.append(
            ensure_product_build_orchestration(
                db,
                record=record,
                product_key=product_key,
                current_user=current_user,
                options=ProductBuildOrchestrationOptions(
                    current_stage=getattr(record.current_stage, "value", str(record.current_stage or "discover")),
                    activation_payload=activation_payload,
                ),
            )
        )
    return statuses


def _ordered_build_products_for_order(db: Session, order: CommercialOrderRecord) -> list[ProductBuildProductKey]:
    lines = db.exec(
        select(CommercialOrderLineRecord).where(CommercialOrderLineRecord.order_id == order.id)
    ).all()
    seen: set[ProductBuildProductKey] = set()
    products: list[ProductBuildProductKey] = []
    for line in lines:
        product = ORDER_PRODUCT_TO_BUILD_PRODUCT.get(line.product_key)
        if product is None or product in seen:
            continue
        seen.add(product)
        products.append(product)
    return products


def _activation_payload(
    *,
    order: CommercialOrderRecord,
    product_key: ProductBuildProductKey,
    source: str,
) -> dict[str, Any]:
    return {
        "source": source,
        "product_key": product_key.value,
        "checkout_ref": order.checkout_ref,
        "order_id": str(order.id),
        "provider": order.provider,
        "currency": order.currency,
        "total_cents": order.total_cents,
        "confirmed_at": order.paid_at.isoformat() if order.paid_at is not None else None,
    }
