from __future__ import annotations

from collections import defaultdict
from uuid import UUID

from sqlmodel import Session, select

from app.models import (
    CommercialAccessRequestRecord,
    CommercialAccessRequestStatus,
    CommercialCurrencyTotal,
    CommercialOrderRecord,
    CommercialQuotaSourceKind,
    WorkspaceCommercialOrderSummaryResponse,
    WorkspaceCommercialProductSummaryResponse,
    WorkspaceCommercialSummaryResponse,
)
from app.services.commerce_service import build_order_response, serialize_access_request
from app.services.commercial_catalog_service import recommend_package_for_product
from app.services.commercial_debt_service import list_commercial_debts
from app.services.commercial_quota_service import (
    get_balance_snapshot,
    list_quota_product_configs,
)


COMMERCIAL_WORKSPACE_PRODUCTS: tuple[str, ...] = ("blueprint_pro", "acp")


def build_workspace_commercial_summary(
    session: Session,
    *,
    workspace_id: UUID,
    session_id: UUID,
) -> WorkspaceCommercialSummaryResponse:
    quota_configs = {
        config.product_key: config
        for config in list_quota_product_configs(session)
        if config.product_key in COMMERCIAL_WORKSPACE_PRODUCTS
    }

    open_debts = list_commercial_debts(session, workspace_id=workspace_id, status="open")
    debt_totals_by_product: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    debt_count_by_product: dict[str, int] = defaultdict(int)
    for debt in open_debts:
        remaining = max(0, debt.amount_cents - debt.settled_amount_cents)
        if remaining <= 0:
            continue
        debt_totals_by_product[debt.product_key][debt.currency] += remaining
        debt_count_by_product[debt.product_key] += 1

    workspace_pending_requests = session.exec(
        select(CommercialAccessRequestRecord)
        .where(
            CommercialAccessRequestRecord.workspace_id == workspace_id,
            CommercialAccessRequestRecord.status == CommercialAccessRequestStatus.pending,
            CommercialAccessRequestRecord.product_key.in_(COMMERCIAL_WORKSPACE_PRODUCTS),
        )
        .order_by(CommercialAccessRequestRecord.created_at.desc(), CommercialAccessRequestRecord.id.desc())
    ).all()
    pending_counts_by_product: dict[str, int] = defaultdict(int)
    for request in workspace_pending_requests:
        pending_counts_by_product[request.product_key] += 1

    request_history_rows = session.exec(
        select(CommercialAccessRequestRecord)
        .where(
            CommercialAccessRequestRecord.workspace_id == workspace_id,
            CommercialAccessRequestRecord.session_id == session_id,
            CommercialAccessRequestRecord.product_key.in_(COMMERCIAL_WORKSPACE_PRODUCTS),
        )
        .order_by(CommercialAccessRequestRecord.created_at.desc(), CommercialAccessRequestRecord.id.desc())
    ).all()

    order_rows = session.exec(
        select(CommercialOrderRecord)
        .where(
            CommercialOrderRecord.workspace_id == workspace_id,
            CommercialOrderRecord.session_id == session_id,
        )
        .order_by(CommercialOrderRecord.created_at.desc(), CommercialOrderRecord.id.desc())
    ).all()

    products: list[WorkspaceCommercialProductSummaryResponse] = []
    for product_key in COMMERCIAL_WORKSPACE_PRODUCTS:
        config = quota_configs.get(product_key)
        snapshot = get_balance_snapshot(session, workspace_id=workspace_id, product_key=product_key)
        recommendation = recommend_package_for_product(
            session,
            product_key=product_key,
            required_units=1,
            workspace_id=workspace_id,
        )
        debt_totals = [
            CommercialCurrencyTotal(currency=currency, amount_cents=amount_cents)
            for currency, amount_cents in sorted(debt_totals_by_product.get(product_key, {}).items())
            if amount_cents > 0
        ]
        products.append(
            WorkspaceCommercialProductSummaryResponse(
                product_key=product_key,
                display_name=config.display_name if config is not None else product_key,
                available_units=snapshot.total_available_units,
                free_units=snapshot.by_source_kind.get(CommercialQuotaSourceKind.free, 0),
                subscription_units=snapshot.by_source_kind.get(CommercialQuotaSourceKind.subscription, 0),
                one_time_units=snapshot.by_source_kind.get(CommercialQuotaSourceKind.one_time, 0),
                adjustment_units=snapshot.by_source_kind.get(CommercialQuotaSourceKind.adjustment, 0),
                pending_request_count=pending_counts_by_product.get(product_key, 0),
                open_debt_count=debt_count_by_product.get(product_key, 0),
                debt_totals=debt_totals,
                recommendation=recommendation if recommendation.package_code else None,
            )
        )

    recent_orders = []
    for row in order_rows[:6]:
        response = build_order_response(session, row)
        product_key = response.lines[0].product_key if response.lines else ""
        recent_orders.append(
            WorkspaceCommercialOrderSummaryResponse(
                order_id=response.id,
                product_key=product_key,
                status=response.status,
                currency=response.currency,
                total_cents=response.total_cents,
                checkout_url=response.checkout_url,
                created_at=response.created_at,
                updated_at=response.updated_at,
            )
        )

    return WorkspaceCommercialSummaryResponse(
        products=products,
        open_debts=open_debts,
        recent_orders=recent_orders,
        request_history=[serialize_access_request(item) for item in request_history_rows[:8]],
    )
