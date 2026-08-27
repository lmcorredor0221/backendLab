from __future__ import annotations

from uuid import UUID

from sqlalchemy import func
from sqlmodel import Session, select

from app.models import (
    CommercialDebtRecord,
    CommercialDebtResponse,
    CommercialDebtSettlementRecord,
    CommercialDebtSettlementRequest,
    CommercialDebtStatus,
    utc_now,
)


def serialize_commercial_debt(record: CommercialDebtRecord) -> CommercialDebtResponse:
    return CommercialDebtResponse(
        id=record.id,
        workspace_id=record.workspace_id,
        product_key=record.product_key,
        access_request_id=record.access_request_id,
        order_id=record.order_id,
        status=record.status,
        reason_code=record.reason_code,
        reason_label=record.reason_label,
        summary=record.summary,
        amount_cents=record.amount_cents,
        settled_amount_cents=record.settled_amount_cents,
        currency=record.currency,
        opened_by_user_id=record.opened_by_user_id,
        resolved_by_user_id=record.resolved_by_user_id,
        due_at=record.due_at,
        resolved_at=record.resolved_at,
        updated_at=record.updated_at,
        created_at=record.created_at,
    )


def list_commercial_debts(
    session: Session,
    *,
    workspace_id: UUID,
    status: str = "open",
    product_key: str = "",
) -> list[CommercialDebtResponse]:
    statement = select(CommercialDebtRecord).where(CommercialDebtRecord.workspace_id == workspace_id)
    if status.strip() and status != "all":
        statement = statement.where(CommercialDebtRecord.status == CommercialDebtStatus(status.strip()))
    if product_key.strip():
        statement = statement.where(CommercialDebtRecord.product_key == product_key.strip())
    rows = session.exec(statement.order_by(CommercialDebtRecord.created_at.asc(), CommercialDebtRecord.id.asc())).all()
    return [serialize_commercial_debt(row) for row in rows]


def count_commercial_debts(
    session: Session,
    *,
    workspace_id: UUID,
    status: str = "open",
    product_key: str = "",
) -> int:
    statement = select(func.count()).select_from(CommercialDebtRecord).where(CommercialDebtRecord.workspace_id == workspace_id)
    if status.strip() and status != "all":
        statement = statement.where(CommercialDebtRecord.status == CommercialDebtStatus(status.strip()))
    if product_key.strip():
        statement = statement.where(CommercialDebtRecord.product_key == product_key.strip())
    return int(session.exec(statement).one())


def has_open_commercial_debt(
    session: Session,
    *,
    workspace_id: UUID,
    product_key: str | None = None,
) -> bool:
    statement = select(CommercialDebtRecord).where(
        CommercialDebtRecord.workspace_id == workspace_id,
        CommercialDebtRecord.status == CommercialDebtStatus.open,
    )
    if product_key is not None:
        statement = statement.where(CommercialDebtRecord.product_key == product_key)
    return session.exec(statement).first() is not None


def create_commercial_debt(
    session: Session,
    *,
    workspace_id: UUID,
    product_key: str,
    access_request_id: UUID | None,
    order_id: UUID | None = None,
    amount_cents: int,
    currency: str = "USD",
    actor_user_id: UUID | None = None,
    reason_code: str = "",
    reason_label: str = "",
    summary: str = "",
    metadata: dict[str, object] | None = None,
) -> CommercialDebtResponse:
    normalized_amount = max(0, amount_cents)
    if normalized_amount <= 0:
        raise ValueError("Debt amount must be greater than zero.")
    record = CommercialDebtRecord(
        workspace_id=workspace_id,
        product_key=product_key.strip(),
        access_request_id=access_request_id,
        order_id=order_id,
        status=CommercialDebtStatus.open,
        reason_code=reason_code.strip() or "debt_pending",
        reason_label=reason_label.strip() or "Deuda pendiente",
        summary=summary.strip() or "Aprobacion comercial con deuda pendiente.",
        amount_cents=normalized_amount,
        currency=(currency or "USD").strip().upper(),
        opened_by_user_id=actor_user_id,
        metadata_payload=dict(metadata or {}),
        updated_at=utc_now(),
    )
    session.add(record)
    session.flush()
    from app.services.commerce_service import record_commercial_event

    record_commercial_event(
        session,
        workspace_id=workspace_id,
        session_id=None,
        user_id=actor_user_id,
        event_key="commercial_debt_opened",
        product_key=record.product_key,
        source="commercial_debt",
        revenue_cents=normalized_amount,
        currency=record.currency,
        metadata={
            "debt_id": str(record.id),
            "access_request_id": str(access_request_id) if access_request_id is not None else "",
            "reason_code": record.reason_code,
        },
        correlation_id=f"debt:{record.id}",
    )
    return serialize_commercial_debt(record)


def settle_commercial_debt(
    session: Session,
    *,
    workspace_id: UUID,
    debt_id: UUID,
    payload: CommercialDebtSettlementRequest,
    actor_user_id: UUID | None = None,
    order_id: UUID | None = None,
    payment_id: UUID | None = None,
    metadata: dict[str, object] | None = None,
) -> CommercialDebtResponse:
    record = session.get(CommercialDebtRecord, debt_id)
    if record is None or record.workspace_id != workspace_id:
        raise ValueError("Commercial debt was not found in this workspace.")
    if record.status != CommercialDebtStatus.open:
        return serialize_commercial_debt(record)
    amount_cents = max(0, payload.amount_cents)
    if amount_cents <= 0:
        raise ValueError("Settlement amount must be greater than zero.")
    normalized_currency = (payload.currency or record.currency).strip().upper()
    if normalized_currency != record.currency:
        raise ValueError("Settlement currency must match the debt currency.")

    remaining = max(0, record.amount_cents - record.settled_amount_cents)
    applied = min(amount_cents, remaining)
    settlement = CommercialDebtSettlementRecord(
        debt_id=record.id,
        workspace_id=workspace_id,
        order_id=order_id,
        payment_id=payment_id,
        settled_amount_cents=applied,
        currency=record.currency,
        settlement_kind=payload.settlement_kind.strip() or "manual",
        actor_user_id=actor_user_id,
        metadata_payload={
            **dict(metadata or {}),
            "resolution_note": payload.resolution_note.strip(),
            "requested_amount_cents": amount_cents,
        },
    )
    session.add(settlement)
    record.settled_amount_cents += applied
    record.updated_at = utc_now()
    if record.settled_amount_cents >= record.amount_cents:
        record.status = CommercialDebtStatus.settled
        record.resolved_by_user_id = actor_user_id
        record.resolved_at = utc_now()
    session.add(record)
    session.flush()
    from app.services.commerce_service import record_commercial_event

    record_commercial_event(
        session,
        workspace_id=workspace_id,
        session_id=None,
        user_id=actor_user_id,
        event_key="commercial_debt_settled",
        product_key=record.product_key,
        source="commercial_debt",
        revenue_cents=applied,
        currency=record.currency,
        metadata={
            "debt_id": str(record.id),
            "settlement_id": str(settlement.id),
            "settlement_kind": settlement.settlement_kind,
            "status": record.status.value,
        },
        correlation_id=f"debt:{record.id}",
    )
    return serialize_commercial_debt(record)


def settle_open_commercial_debts(
    session: Session,
    *,
    workspace_id: UUID,
    amount_cents: int,
    currency: str,
    actor_user_id: UUID | None = None,
    order_id: UUID | None = None,
    payment_id: UUID | None = None,
    settlement_kind: str = "payment",
    metadata: dict[str, object] | None = None,
) -> int:
    remaining = max(0, amount_cents)
    if remaining <= 0:
        return 0
    normalized_currency = (currency or "USD").strip().upper()
    debts = session.exec(
        select(CommercialDebtRecord)
        .where(
            CommercialDebtRecord.workspace_id == workspace_id,
            CommercialDebtRecord.status == CommercialDebtStatus.open,
            CommercialDebtRecord.currency == normalized_currency,
        )
        .order_by(CommercialDebtRecord.created_at.asc(), CommercialDebtRecord.id.asc())
    ).all()
    for debt in debts:
        if remaining <= 0:
            break
        before = debt.settled_amount_cents
        settle_commercial_debt(
            session,
            workspace_id=workspace_id,
            debt_id=debt.id,
            payload=CommercialDebtSettlementRequest(
                amount_cents=remaining,
                currency=normalized_currency,
                settlement_kind=settlement_kind,
                resolution_note="Settlement applied from payment flow.",
            ),
            actor_user_id=actor_user_id,
            order_id=order_id,
            payment_id=payment_id,
            metadata=metadata,
        )
        session.refresh(debt)
        remaining -= max(0, debt.settled_amount_cents - before)
    return remaining
