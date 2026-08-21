from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.models import (
    CommercialCheckoutSessionRequest,
    CommercialEntitlementRecord,
    CommercialEntitlementStatus,
    CommercialOrderRecord,
    CommercialOrderStatus,
    CommercialPaymentRecord,
    CommercialPaymentStatus,
    CommercialTier,
    HotmartCredentialUpsertRequest,
    HotmartWebhookEventRecord,
    SessionRecord,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceRecord,
    WorkspaceRole,
)
from app.services.auth_service import hash_password
from app.services.commerce_service import create_checkout_session
from app.services.deliverable_catalog.persistence import DeliverableGenerationJobRecord
from app.services.diagram_center.persistence import DiagramGenerationJobRecord
from app.services.hotmart.secrets import upsert_hotmart_credentials
from app.services.hotmart.webhooks import process_hotmart_webhook
from app.services.product_processing.persistence import ProductBuildRunRecord, ProductBuildStepRecord


@pytest.fixture()
def db_session() -> Iterator[Session]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _seed_checkout_context(session: Session) -> tuple[UserRecord, WorkspaceRecord, SessionRecord]:
    user = UserRecord(
        email="hotmart-webhooks@leanbuilder.local",
        full_name="Hotmart Webhooks Tester",
        password_hash=hash_password("Secret123!"),
    )
    session.add(user)
    session.flush()
    workspace = WorkspaceRecord(
        name="Hotmart Webhooks Workspace",
        slug=f"hotmart-webhooks-{str(user.id)[:8]}",
        created_by_user_id=user.id,
    )
    session.add(workspace)
    session.flush()
    session.add(WorkspaceMembershipRecord(workspace_id=workspace.id, user_id=user.id, role=WorkspaceRole.owner))
    record = SessionRecord(user_id=user.id, workspace_id=workspace.id, title="Hotmart Webhooks Project")
    session.add(record)
    session.commit()
    session.refresh(user)
    session.refresh(workspace)
    session.refresh(record)
    return user, workspace, record


def _configure_hottok(session: Session, workspace: WorkspaceRecord) -> None:
    upsert_hotmart_credentials(
        session,
        workspace_id=workspace.id,
        payload=HotmartCredentialUpsertRequest(
            environment="sandbox",
            enabled=True,
            client_id="client-id-value",
            client_secret="client-secret-value",
            basic_token="basic-token-value",
            hottok="hottok-value",
        ),
    )
    session.commit()


def _create_hotmart_order(session: Session, record: SessionRecord, user: UserRecord, suffix: str) -> CommercialOrderRecord:
    checkout = create_checkout_session(
        session,
        payload=CommercialCheckoutSessionRequest(
            session_id=record.id,
            product_key="blueprint_pro",
            provider="hotmart",
            idempotency_key=f"{record.id}:hotmart-webhook-order:{suffix}",
        ),
        record=record,
        current_user=user,
        base_url="http://localhost:3200",
    )
    session.commit()
    return session.exec(select(CommercialOrderRecord).where(CommercialOrderRecord.id == checkout.order_id)).one()


def _approved_payload(order: CommercialOrderRecord, *, event_id: str = "evt-approved-1", transaction: str = "HP123"):
    return {
        "id": event_id,
        "event": "PURCHASE_APPROVED",
        "data": {
            "purchase": {
                "transaction": transaction,
                "metadata": {"order_id": str(order.id), "checkout_ref": order.checkout_ref},
                "price": {"value": order.total_cents / 100, "currency_code": order.currency},
            }
        },
    }


def test_hotmart_approved_webhook_grants_entitlement_and_is_idempotent(db_session: Session) -> None:
    user, workspace, record = _seed_checkout_context(db_session)
    _configure_hottok(db_session, workspace)
    order = _create_hotmart_order(db_session, record, user, "approved")

    response = process_hotmart_webhook(
        db_session,
        payload=_approved_payload(order),
        hottok_header="hottok-value",
        environment="sandbox",
    )
    db_session.commit()

    assert response.processing_status == "processed"
    assert response.duplicate is False
    refreshed_order = db_session.get(CommercialOrderRecord, order.id)
    refreshed_session = db_session.get(SessionRecord, record.id)
    payment = db_session.exec(select(CommercialPaymentRecord)).one()
    entitlement = db_session.exec(select(CommercialEntitlementRecord)).one()
    assert refreshed_order is not None
    assert refreshed_session is not None
    assert refreshed_order.status == CommercialOrderStatus.paid
    assert payment.status == CommercialPaymentStatus.succeeded
    assert entitlement.status == CommercialEntitlementStatus.active
    assert refreshed_session.commercial_tier == CommercialTier.blueprint_pro
    runs = db_session.exec(
        select(ProductBuildRunRecord).where(
            ProductBuildRunRecord.session_id == record.id,
            ProductBuildRunRecord.product_key == "blueprint_pro",
        )
    ).all()
    assert len(runs) == 1
    assert runs[0].checkpoint_payload["activation"]["checkout_ref"] == order.checkout_ref

    duplicate = process_hotmart_webhook(
        db_session,
        payload=_approved_payload(order),
        hottok_header="hottok-value",
        environment="sandbox",
    )
    db_session.commit()

    assert duplicate.duplicate is True
    assert len(db_session.exec(select(CommercialPaymentRecord)).all()) == 1
    assert (
        len(
            db_session.exec(
                select(ProductBuildRunRecord).where(
                    ProductBuildRunRecord.session_id == record.id,
                    ProductBuildRunRecord.product_key == "blueprint_pro",
                )
            ).all()
        )
        == 1
    )
    assert len(db_session.exec(select(CommercialEntitlementRecord)).all()) == 1


def test_hotmart_refund_webhook_marks_payment_and_entitlement_refunded(db_session: Session) -> None:
    user, workspace, record = _seed_checkout_context(db_session)
    _configure_hottok(db_session, workspace)
    order = _create_hotmart_order(db_session, record, user, "refund")
    process_hotmart_webhook(
        db_session,
        payload=_approved_payload(order, event_id="evt-approved-refund", transaction="HP124"),
        hottok_header="hottok-value",
        environment="sandbox",
    )
    db_session.commit()

    refund_payload = {
        "id": "evt-refund-1",
        "event": "PURCHASE_REFUNDED",
        "data": {
            "purchase": {
                "transaction": "HP124",
                "metadata": {"order_id": str(order.id), "checkout_ref": order.checkout_ref},
                "price": {"value": order.total_cents / 100, "currency_code": order.currency},
            }
        },
    }
    response = process_hotmart_webhook(
        db_session,
        payload=refund_payload,
        hottok_header="hottok-value",
        environment="sandbox",
    )
    db_session.commit()

    refreshed_order = db_session.get(CommercialOrderRecord, order.id)
    refreshed_session = db_session.get(SessionRecord, record.id)
    payment = db_session.exec(select(CommercialPaymentRecord)).one()
    entitlement = db_session.exec(select(CommercialEntitlementRecord)).one()
    assert response.processing_status == "processed"
    assert refreshed_order is not None
    assert refreshed_session is not None
    assert refreshed_order.status == CommercialOrderStatus.refunded
    assert payment.status == CommercialPaymentStatus.refunded
    assert entitlement.status == CommercialEntitlementStatus.refunded
    assert refreshed_session.commercial_tier == CommercialTier.blueprint


def test_hotmart_chargeback_webhook_revokes_entitlement(db_session: Session) -> None:
    user, workspace, record = _seed_checkout_context(db_session)
    _configure_hottok(db_session, workspace)
    order = _create_hotmart_order(db_session, record, user, "chargeback")
    process_hotmart_webhook(
        db_session,
        payload=_approved_payload(order, event_id="evt-approved-chargeback", transaction="HP125"),
        hottok_header="hottok-value",
        environment="sandbox",
    )
    db_session.commit()

    chargeback_payload = {
        "id": "evt-chargeback-1",
        "event": "PURCHASE_CHARGEBACK",
        "data": {
            "purchase": {
                "transaction": "HP125",
                "metadata": {"order_id": str(order.id), "checkout_ref": order.checkout_ref},
            }
        },
    }
    response = process_hotmart_webhook(
        db_session,
        payload=chargeback_payload,
        hottok_header="hottok-value",
        environment="sandbox",
    )
    db_session.commit()

    entitlement = db_session.exec(select(CommercialEntitlementRecord)).one()
    assert response.processing_status == "processed"
    assert entitlement.status == CommercialEntitlementStatus.revoked


def test_hotmart_webhook_rejects_invalid_hottok_and_records_redacted_event(db_session: Session) -> None:
    user, workspace, record = _seed_checkout_context(db_session)
    _configure_hottok(db_session, workspace)
    order = _create_hotmart_order(db_session, record, user, "invalid-token")
    payload = _approved_payload(order, event_id="evt-invalid-token", transaction="HP126")
    payload["data"]["purchase"]["client_secret"] = "must-redact"

    with pytest.raises(PermissionError):
        process_hotmart_webhook(
            db_session,
            payload=payload,
            hottok_header="wrong-token",
            environment="sandbox",
        )
    db_session.commit()

    event = db_session.exec(select(HotmartWebhookEventRecord)).one()
    assert event.processing_status == "rejected"
    assert event.hottok_validated is False
    assert event.payload_redacted["data"]["purchase"]["client_secret"] == "[redacted]"
    assert db_session.exec(select(CommercialPaymentRecord)).all() == []
    assert db_session.exec(select(CommercialEntitlementRecord)).all() == []
