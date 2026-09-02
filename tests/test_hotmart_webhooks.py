from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.models import (
    CommercialAccessRequestRecord,
    CommercialAccessRequestStatus,
    CommercialDebtRecord,
    CommercialDebtStatus,
    CommercialPackageCatalogUpsertRequest,
    CommercialCheckoutSessionRequest,
    CommercialEntitlementRecord,
    CommercialEntitlementStatus,
    CommercialOrderRecord,
    CommercialOrderStatus,
    CommercialPaymentRecord,
    CommercialPaymentStatus,
    CommercialTier,
    HotmartCredentialUpsertRequest,
    HotmartPendingActivationClaimRequest,
    HotmartPendingActivationRecord,
    HotmartPendingActivationStatus,
    HotmartReconciliationIssueRecord,
    HotmartWebhookEventRecord,
    SessionRecord,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceRecord,
    WorkspaceRole,
)
from app.services.auth_service import hash_password
from app.services.commerce_service import create_checkout_session
from app.services.commercial_catalog_service import upsert_package_catalog_entry
from app.services.commercial_package_fulfillment_service import resolve_legacy_package_resolution
from app.services.commercial_debt_service import create_commercial_debt
from app.services.commercial_quota_service import get_balance_snapshot, list_balance_buckets
from app.services.deliverable_catalog.persistence import DeliverableGenerationJobRecord
from app.services.diagram_center.persistence import DiagramGenerationJobRecord
from app.services.hotmart.pending_activations import claim_hotmart_pending_activation
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


def _create_hotmart_order(
    session: Session,
    record: SessionRecord,
    user: UserRecord,
    suffix: str,
    *,
    product_key: str = "blueprint_pro",
    package_code: str = "",
) -> CommercialOrderRecord:
    checkout = create_checkout_session(
        session,
        payload=CommercialCheckoutSessionRequest(
            session_id=record.id,
            product_key=product_key,
            package_code=package_code,
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


def _external_sale_payload(
    *,
    buyer_email: str,
    buyer_name: str = "External Hotmart Buyer",
    event_id: str = "evt-external-sale-1",
    transaction: str = "HP900",
    product_id: str = "hm-prod-bundle",
    product_ucode: str = "HMBUNDLE",
    offer_code: str = "offer-bundle",
    plan_code: str = "plan-bundle",
    amount_value: float = 149.0,
    currency_code: str = "USD",
) -> dict[str, object]:
    return {
        "id": event_id,
        "event": "PURCHASE_APPROVED",
        "data": {
            "buyer": {
                "name": buyer_name,
                "email": buyer_email,
                "document": "1234567890",
            },
            "purchase": {
                "transaction": transaction,
                "price": {"value": amount_value, "currency_code": currency_code},
                "product": {"id": product_id, "ucode": product_ucode},
                "offer": {"code": offer_code},
                "plan": {"code": plan_code},
            },
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
    event = db_session.exec(select(HotmartWebhookEventRecord)).one()
    assert event.retries == 1
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


def test_hotmart_approved_webhook_settles_open_workspace_debt(db_session: Session) -> None:
    user, workspace, record = _seed_checkout_context(db_session)
    _configure_hottok(db_session, workspace)
    order = _create_hotmart_order(db_session, record, user, "debt-settlement")
    create_commercial_debt(
        db_session,
        workspace_id=workspace.id,
        product_key="blueprint_pro",
        access_request_id=None,
        amount_cents=order.total_cents,
        currency="USD",
        actor_user_id=user.id,
        reason_code="manual_debt",
        reason_label="Deuda manual",
    )
    db_session.commit()

    response = process_hotmart_webhook(
        db_session,
        payload=_approved_payload(order, event_id="evt-approved-debt", transaction="HP128"),
        hottok_header="hottok-value",
        environment="sandbox",
    )
    db_session.commit()

    refreshed_order = db_session.get(CommercialOrderRecord, order.id)
    payment = db_session.exec(select(CommercialPaymentRecord).where(CommercialPaymentRecord.order_id == order.id)).one()
    debts = db_session.exec(select(CommercialDebtRecord).where(CommercialDebtRecord.workspace_id == workspace.id)).all()
    assert response.processing_status == "processed"
    assert refreshed_order is not None
    assert len(debts) == 1
    assert debts[0].status == CommercialDebtStatus.settled
    assert debts[0].settled_amount_cents == order.total_cents
    assert payment.metadata_payload["debt_settlement"]["strategy"] == "payment_currency"
    assert refreshed_order.metadata_payload["debt_settlement"]["settled_amount_cents"] == order.total_cents


def test_hotmart_approved_webhook_credits_bundle_subscription_across_products(db_session: Session) -> None:
    user, workspace, record = _seed_checkout_context(db_session)
    _configure_hottok(db_session, workspace)
    upsert_package_catalog_entry(
        db_session,
        payload=CommercialPackageCatalogUpsertRequest(
            package_code="bundle-monthly",
            display_name="Bundle mensual",
            product_key="bundle",
            package_type="bundle_subscription",
            granted_units_blueprint_pro=2,
            granted_units_acp=1,
            billing_cycle="monthly",
        ),
    )
    order = _create_hotmart_order(
        db_session,
        record,
        user,
        "bundle-credit",
        product_key="acp",
        package_code="bundle-monthly",
    )

    response = process_hotmart_webhook(
        db_session,
        payload=_approved_payload(order, event_id="evt-approved-bundle", transaction="HP129"),
        hottok_header="hottok-value",
        environment="sandbox",
    )
    db_session.commit()

    bp_snapshot = get_balance_snapshot(db_session, workspace_id=workspace.id, product_key="blueprint_pro")
    acp_snapshot = get_balance_snapshot(db_session, workspace_id=workspace.id, product_key="acp")
    acp_buckets = list_balance_buckets(db_session, workspace_id=workspace.id, product_key="acp")
    payment = db_session.exec(select(CommercialPaymentRecord).where(CommercialPaymentRecord.order_id == order.id)).one()
    assert response.processing_status == "processed"
    assert bp_snapshot.total_available_units == 2
    assert acp_snapshot.total_available_units == 1
    assert any(bucket.source_ref == "subscription:bundle-monthly:acp" for bucket in acp_buckets)
    assert len(payment.metadata_payload["package_credit"]["grants"]) == 2
    assert payment.metadata_payload["package_credit"]["package_type"] == "bundle_subscription"


def test_hotmart_approved_webhook_marks_ambiguous_legacy_order_for_manual_resolution(db_session: Session) -> None:
    user, workspace, record = _seed_checkout_context(db_session)
    _configure_hottok(db_session, workspace)
    upsert_package_catalog_entry(
        db_session,
        payload=CommercialPackageCatalogUpsertRequest(
            package_code="bp-pack-2",
            display_name="Blueprint 2",
            product_key="blueprint_pro",
            granted_units=2,
        ),
    )
    upsert_package_catalog_entry(
        db_session,
        payload=CommercialPackageCatalogUpsertRequest(
            package_code="bp-pack-4",
            display_name="Blueprint 4",
            product_key="blueprint_pro",
            granted_units=4,
        ),
    )
    order = _create_hotmart_order(db_session, record, user, "legacy-ambiguous")

    response = process_hotmart_webhook(
        db_session,
        payload=_approved_payload(order, event_id="evt-approved-legacy", transaction="HP130"),
        hottok_header="hottok-value",
        environment="sandbox",
    )
    db_session.commit()

    refreshed_order = db_session.get(CommercialOrderRecord, order.id)
    payment = db_session.exec(select(CommercialPaymentRecord).where(CommercialPaymentRecord.order_id == order.id)).one()
    snapshot = get_balance_snapshot(db_session, workspace_id=workspace.id, product_key="blueprint_pro")
    assert response.processing_status == "processed"
    assert refreshed_order is not None
    assert snapshot.total_available_units == 0
    assert "package_credit" not in payment.metadata_payload
    assert refreshed_order.metadata_payload["legacy_package_resolution"]["status"] == "pending_manual_resolution"
    assert set(refreshed_order.metadata_payload["legacy_package_resolution"]["candidate_package_codes"]) == {
        "bp-pack-2",
        "bp-pack-4",
    }


def test_manual_legacy_package_resolution_credits_workspace_and_autoapproves_pending_request(db_session: Session) -> None:
    user, workspace, record = _seed_checkout_context(db_session)
    _configure_hottok(db_session, workspace)
    pending_record = SessionRecord(user_id=user.id, workspace_id=workspace.id, title="Legacy Pending Project")
    db_session.add(pending_record)
    db_session.flush()
    access_request = CommercialAccessRequestRecord(
        workspace_id=workspace.id,
        session_id=pending_record.id,
        requester_user_id=user.id,
        capability="blueprint.build",
        product_key="blueprint_pro",
        target_tier=CommercialTier.blueprint_pro,
        reason="Esperando saldo legacy",
        status=CommercialAccessRequestStatus.pending,
    )
    db_session.add(access_request)
    upsert_package_catalog_entry(
        db_session,
        payload=CommercialPackageCatalogUpsertRequest(
            package_code="bp-pack-1",
            display_name="Blueprint 1",
            product_key="blueprint_pro",
            granted_units=1,
        ),
    )
    upsert_package_catalog_entry(
        db_session,
        payload=CommercialPackageCatalogUpsertRequest(
            package_code="bp-pack-3",
            display_name="Blueprint 3",
            product_key="blueprint_pro",
            granted_units=3,
        ),
    )
    db_session.commit()
    order = _create_hotmart_order(db_session, record, user, "legacy-resolve")

    process_hotmart_webhook(
        db_session,
        payload=_approved_payload(order, event_id="evt-approved-legacy-resolve", transaction="HP131"),
        hottok_header="hottok-value",
        environment="sandbox",
    )
    db_session.commit()

    unresolved_request = db_session.get(CommercialAccessRequestRecord, access_request.id)
    assert unresolved_request is not None
    assert unresolved_request.status == CommercialAccessRequestStatus.pending

    resolved = resolve_legacy_package_resolution(
        db_session,
        workspace_id=workspace.id,
        order_id=order.id,
        package_code="bp-pack-3",
        resolution_note="Seleccionado por platform admin.",
        actor_user_id=user.id,
    )
    db_session.commit()

    refreshed_request = db_session.get(CommercialAccessRequestRecord, access_request.id)
    refreshed_order = db_session.get(CommercialOrderRecord, order.id)
    payment = db_session.exec(select(CommercialPaymentRecord).where(CommercialPaymentRecord.order_id == order.id)).one()
    snapshot = get_balance_snapshot(db_session, workspace_id=workspace.id, product_key="blueprint_pro")
    assert refreshed_request is not None
    assert refreshed_order is not None
    assert resolved.status == "resolved"
    assert resolved.selected_package_code == "bp-pack-3"
    assert refreshed_request.status == CommercialAccessRequestStatus.approved
    assert payment.metadata_payload["package_credit"]["package_code"] == "bp-pack-3"
    assert refreshed_order.metadata_payload["legacy_package_resolution"]["status"] == "resolved"
    assert snapshot.total_available_units == 2


def test_hotmart_external_sale_webhook_creates_pending_activation_when_order_is_absent(db_session: Session) -> None:
    user, workspace, _record = _seed_checkout_context(db_session)
    _configure_hottok(db_session, workspace)
    upsert_package_catalog_entry(
        db_session,
        payload=CommercialPackageCatalogUpsertRequest(
            package_code="bundle-hotmart-monthly",
            display_name="Bundle Hotmart mensual",
            product_key="bundle",
            package_type="bundle_subscription",
            granted_units_blueprint_pro=3,
            granted_units_acp=1,
            billing_cycle="monthly",
            hotmart_environment="sandbox",
            hotmart_product_id="hm-prod-bundle",
            hotmart_product_ucode="HMBUNDLE",
            offer_code="offer-bundle",
            plan_code="plan-bundle",
        ),
    )
    db_session.commit()

    response = process_hotmart_webhook(
        db_session,
        payload=_external_sale_payload(
            buyer_email=user.email,
            event_id="evt-external-pending",
            transaction="HP901",
        ),
        hottok_header="hottok-value",
        environment="sandbox",
    )
    db_session.commit()

    pending = db_session.exec(select(HotmartPendingActivationRecord)).one()
    assert response.processing_status == "pending_activation"
    assert response.pending_activation_id == pending.id
    assert pending.status == HotmartPendingActivationStatus.pending_activation
    assert pending.source_workspace_id == workspace.id
    assert pending.package_code == "bundle-hotmart-monthly"
    assert pending.product_key == "bundle"
    assert pending.buyer_email == user.email
    assert pending.resolution_strategy == "package_catalog"
    assert db_session.exec(select(CommercialOrderRecord)).all() == []
    assert db_session.exec(select(CommercialPaymentRecord)).all() == []
    assert db_session.exec(select(CommercialEntitlementRecord)).all() == []


def test_claiming_pending_external_sale_adopts_order_and_grants_bundle_credits(db_session: Session) -> None:
    user, workspace, record = _seed_checkout_context(db_session)
    _configure_hottok(db_session, workspace)
    upsert_package_catalog_entry(
        db_session,
        payload=CommercialPackageCatalogUpsertRequest(
            package_code="bundle-hotmart-monthly",
            display_name="Bundle Hotmart mensual",
            product_key="bundle",
            package_type="bundle_subscription",
            granted_units_blueprint_pro=3,
            granted_units_acp=1,
            billing_cycle="monthly",
            hotmart_environment="sandbox",
            hotmart_product_id="hm-prod-bundle",
            hotmart_product_ucode="HMBUNDLE",
            offer_code="offer-bundle",
            plan_code="plan-bundle",
        ),
    )
    queued_record = SessionRecord(user_id=user.id, workspace_id=workspace.id, title="Queued Blueprint Project")
    db_session.add(queued_record)
    db_session.flush()
    queued_request = CommercialAccessRequestRecord(
        workspace_id=workspace.id,
        session_id=queued_record.id,
        requester_user_id=user.id,
        capability="blueprint.build",
        product_key="blueprint_pro",
        target_tier=CommercialTier.blueprint_pro,
        reason="Esperando saldo de bundle Hotmart",
        status=CommercialAccessRequestStatus.pending,
    )
    db_session.add(queued_request)
    db_session.commit()

    process_hotmart_webhook(
        db_session,
        payload=_external_sale_payload(
            buyer_email=user.email,
            event_id="evt-external-claim",
            transaction="HP902",
        ),
        hottok_header="hottok-value",
        environment="sandbox",
    )
    db_session.commit()

    pending = db_session.exec(select(HotmartPendingActivationRecord)).one()
    response = claim_hotmart_pending_activation(
        db_session,
        activation_token=pending.activation_token,
        payload=HotmartPendingActivationClaimRequest(session_id=record.id),
        current_user=user,
    )
    db_session.commit()

    refreshed_pending = db_session.get(HotmartPendingActivationRecord, pending.id)
    adopted_order = db_session.get(CommercialOrderRecord, response.adopted_order_id)
    adopted_payment = db_session.get(CommercialPaymentRecord, response.adopted_payment_id)
    refreshed_record = db_session.get(SessionRecord, record.id)
    refreshed_request = db_session.get(CommercialAccessRequestRecord, queued_request.id)
    bp_snapshot = get_balance_snapshot(db_session, workspace_id=workspace.id, product_key="blueprint_pro")
    acp_snapshot = get_balance_snapshot(db_session, workspace_id=workspace.id, product_key="acp")
    runs = db_session.exec(
        select(ProductBuildRunRecord).where(
            ProductBuildRunRecord.session_id == record.id,
            ProductBuildRunRecord.product_key == "acp",
        )
    ).all()

    assert refreshed_pending is not None
    assert adopted_order is not None
    assert adopted_payment is not None
    assert refreshed_record is not None
    assert refreshed_request is not None
    assert refreshed_pending.status == HotmartPendingActivationStatus.claimed
    assert refreshed_pending.claimed_session_id == record.id
    assert refreshed_pending.adopted_order_id == adopted_order.id
    assert adopted_order.status == CommercialOrderStatus.paid
    assert adopted_order.session_id == record.id
    assert adopted_order.metadata_payload["package_code"] == "bundle-hotmart-monthly"
    assert adopted_order.metadata_payload["external_origin"] is True
    assert adopted_payment.status == CommercialPaymentStatus.succeeded
    assert refreshed_record.commercial_tier == CommercialTier.acp
    assert refreshed_request.status == CommercialAccessRequestStatus.approved
    assert bp_snapshot.total_available_units == 2
    assert acp_snapshot.total_available_units == 1
    assert len(runs) == 1
    assert runs[0].checkpoint_payload["activation"]["order_id"] == str(adopted_order.id)


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
            request_headers={
                "X-HOTMART-HOTTOK": "wrong-token",
                "User-Agent": "Hotmart webhook test",
            },
            environment="sandbox",
        )
    db_session.commit()

    event = db_session.exec(select(HotmartWebhookEventRecord)).one()
    assert event.processing_status == "rejected"
    assert event.hottok_validated is False
    assert event.payload_redacted["data"]["purchase"]["client_secret"] == "[redacted]"
    diagnostics = event.payload_redacted["_lab_request_diagnostics"]
    assert diagnostics["headers_redacted"]["X-HOTMART-HOTTOK"] == "[redacted]"
    assert diagnostics["headers_redacted"]["User-Agent"] == "Hotmart webhook test"
    assert diagnostics["hottok_header_present"] is True
    assert diagnostics["hottok_header_length"] == len("wrong-token")
    assert diagnostics["hottok_header_sha256_prefix"]
    assert "wrong-token" not in str(event.payload_redacted)
    assert db_session.exec(select(CommercialPaymentRecord)).all() == []
    assert db_session.exec(select(CommercialEntitlementRecord)).all() == []


def test_hotmart_duplicate_webhook_revalidates_hottok_and_records_retry_diagnostics(db_session: Session) -> None:
    user, workspace, record = _seed_checkout_context(db_session)
    _configure_hottok(db_session, workspace)
    order = _create_hotmart_order(db_session, record, user, "duplicate-invalid-token")
    payload = _approved_payload(order, event_id="evt-duplicate-invalid-token", transaction="HP126-DUP")

    process_hotmart_webhook(
        db_session,
        payload=payload,
        hottok_header="hottok-value",
        environment="sandbox",
    )
    db_session.commit()

    with pytest.raises(PermissionError):
        process_hotmart_webhook(
            db_session,
            payload=payload,
            hottok_header="wrong-token",
            request_headers={
                "X-HOTMART-HOTTOK": "wrong-token",
                "User-Agent": "Hotmart retry test",
            },
            environment="sandbox",
        )
    db_session.commit()

    event = db_session.exec(select(HotmartWebhookEventRecord)).one()
    retry_diagnostics = event.payload_redacted["_lab_last_duplicate_request_diagnostics"]
    assert event.retries == 1
    assert event.processing_status == "rejected"
    assert event.error_code == "invalid_hottok"
    assert retry_diagnostics["headers_redacted"]["X-HOTMART-HOTTOK"] == "[redacted]"
    assert retry_diagnostics["headers_redacted"]["User-Agent"] == "Hotmart retry test"
    assert retry_diagnostics["hottok_header_present"] is True
    assert retry_diagnostics["hottok_header_length"] == len("wrong-token")
    assert "wrong-token" not in str(event.payload_redacted)
    assert len(db_session.exec(select(CommercialPaymentRecord)).all()) == 1
    assert len(db_session.exec(select(CommercialEntitlementRecord)).all()) == 1


def test_hotmart_webhook_conflict_opens_reconciliation_issue_without_duplicate_credit(db_session: Session) -> None:
    user, workspace, record = _seed_checkout_context(db_session)
    _configure_hottok(db_session, workspace)
    order = _create_hotmart_order(db_session, record, user, "conflict")

    process_hotmart_webhook(
        db_session,
        payload=_approved_payload(order, event_id="evt-conflict-1", transaction="HP127"),
        hottok_header="hottok-value",
        environment="sandbox",
    )
    db_session.commit()

    conflicting_payload = _approved_payload(order, event_id="evt-conflict-1", transaction="HP127")
    conflicting_payload["data"]["purchase"]["price"]["value"] = 9999
    conflicting = process_hotmart_webhook(
        db_session,
        payload=conflicting_payload,
        hottok_header="hottok-value",
        request_headers={"X-HOTMART-HOTTOK": "hottok-value"},
        environment="sandbox",
    )
    db_session.commit()

    issue = db_session.exec(select(HotmartReconciliationIssueRecord)).one()
    event = db_session.exec(select(HotmartWebhookEventRecord)).one()
    assert conflicting.processing_status == "observed"
    assert conflicting.duplicate is True
    assert issue.issue_type == "webhook_payload_conflict"
    assert event.retries == 1
    assert event.payload_redacted["_lab_last_duplicate_request_diagnostics"]["hottok_header_present"] is True
    assert len(db_session.exec(select(CommercialPaymentRecord)).all()) == 1
