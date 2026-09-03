from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Iterator

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.models import (
    CommerceProviderCheckoutRecord,
    CommerceProviderCredentialUpsertRequest,
    CommerceProviderProductMappingUpsertRequest,
    CommerceProviderWebhookEventRecord,
    CommercialCheckoutCompletionRequest,
    CommercialCheckoutSessionRequest,
    CommercialDebtRecord,
    CommercialDebtStatus,
    CommercialEntitlementRecord,
    CommercialPackageCatalogUpsertRequest,
    CommercialOrderRecord,
    CommercialOrderStatus,
    CommercialPaymentRecord,
    HotmartPaymentLinkRecord,
    SessionRecord,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceRecord,
    WorkspaceRole,
)
from app.services.auth_service import hash_password
from app.services.commerce_provider_router import (
    get_commerce_payment_provider,
    normalize_commerce_payment_provider,
)
from app.services.commerce_service import complete_checkout_session, create_checkout_session
from app.services.commerce_provider_mappings import upsert_commerce_provider_mapping
from app.services.commerce_provider_secrets import upsert_commerce_provider_credentials
from app.services.commercial_catalog_service import upsert_package_catalog_entry
from app.services.commercial_debt_service import create_commercial_debt
from app.services.commercial_quota_service import get_balance_snapshot
from app.services.payment_providers.rebill import RebillPaymentProvider
from app.services.rebill.client import RebillApiResult
from app.services.rebill.webhooks import process_rebill_webhook
from app.services.deliverable_catalog.persistence import DeliverableGenerationJobRecord  # noqa: F401
from app.services.diagram_center.persistence import DiagramGenerationJobRecord  # noqa: F401
from app.services.product_processing.persistence import ProductBuildRunRecord, ProductBuildStepRecord  # noqa: F401


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
        email="commerce-provider@leanbuilder.local",
        full_name="Commerce Provider Tester",
        password_hash=hash_password("Secret123!"),
    )
    session.add(user)
    session.flush()
    workspace = WorkspaceRecord(
        name="Commerce Provider Workspace",
        slug=f"commerce-provider-{str(user.id)[:8]}",
        created_by_user_id=user.id,
    )
    session.add(workspace)
    session.flush()
    session.add(WorkspaceMembershipRecord(workspace_id=workspace.id, user_id=user.id, role=WorkspaceRole.owner))
    record = SessionRecord(user_id=user.id, workspace_id=workspace.id, title="Commerce Provider Project")
    session.add(record)
    session.commit()
    session.refresh(user)
    session.refresh(workspace)
    session.refresh(record)
    return user, workspace, record


def test_commerce_provider_router_normalizes_supported_providers() -> None:
    assert normalize_commerce_payment_provider(None) == "sandbox"
    assert normalize_commerce_payment_provider("default") == "sandbox"
    assert normalize_commerce_payment_provider("HOTMART") == "hotmart"
    assert normalize_commerce_payment_provider("REBILL") == "rebill"
    assert get_commerce_payment_provider("sandbox").provider_key == "sandbox"
    assert get_commerce_payment_provider("hotmart").provider_key == "hotmart"
    assert get_commerce_payment_provider("rebill").provider_key == "rebill"

    with pytest.raises(ValueError, match="Unsupported commerce checkout provider"):
        normalize_commerce_payment_provider("stripe")


def test_sandbox_checkout_provider_preserves_existing_order_flow(db_session: Session) -> None:
    user, _, record = _seed_checkout_context(db_session)

    response = create_checkout_session(
        db_session,
        payload=CommercialCheckoutSessionRequest(
            session_id=record.id,
            product_key="blueprint_pro",
            idempotency_key=f"{record.id}:sandbox-provider",
        ),
        record=record,
        current_user=user,
        base_url="http://localhost:3200",
    )
    db_session.commit()

    order = db_session.exec(select(CommercialOrderRecord).where(CommercialOrderRecord.id == response.order_id)).one()
    assert response.provider == "sandbox"
    assert response.checkout_ref.startswith("sandbox_")
    assert response.checkout_url.endswith(f"/checkout/sandbox/{response.checkout_ref}")
    assert response.next_action == "open_checkout"
    assert order.provider == "sandbox"
    assert order.status == CommercialOrderStatus.pending
    assert order.metadata_payload["provider_stage"] == "sandbox_checkout"


def test_hotmart_checkout_provider_creates_pending_order_without_payment_link(db_session: Session) -> None:
    user, _, record = _seed_checkout_context(db_session)

    response = create_checkout_session(
        db_session,
        payload=CommercialCheckoutSessionRequest(
            session_id=record.id,
            product_key="blueprint_pro",
            provider="hotmart",
            idempotency_key=f"{record.id}:hotmart-provider",
            success_url="https://example.test/success",
            cancel_url="https://example.test/cancel",
        ),
        record=record,
        current_user=user,
        base_url="http://localhost:3200",
    )
    db_session.commit()

    order = db_session.exec(select(CommercialOrderRecord).where(CommercialOrderRecord.id == response.order_id)).one()
    payment_links = db_session.exec(select(HotmartPaymentLinkRecord)).all()
    assert response.provider == "hotmart"
    assert response.checkout_ref.startswith("hotmart_")
    assert response.checkout_url == ""
    assert response.status == CommercialOrderStatus.pending
    assert response.next_action == "await_payment_link"
    assert order.provider == "hotmart"
    assert order.status == CommercialOrderStatus.pending
    assert order.total_cents > 0
    assert order.metadata_payload["provider_stage"] == "hotmart_order_pending_payment_link"
    assert order.metadata_payload["requires_payment_link"] is True
    assert order.metadata_payload["payment_link_stage"] == "stage_3"
    assert payment_links == []


class FakeRebillClient:
    create_calls: list[dict[str, object]] = []
    payment_payload: dict[str, object] = {}

    def __init__(self, config) -> None:
        self.config = config

    def create_payment_link(
        self,
        *,
        secret_key: str,
        payload: dict[str, object],
        idempotency_key: str,
    ) -> RebillApiResult:
        self.__class__.create_calls.append(
            {
                "secret_key": secret_key,
                "payload": payload,
                "idempotency_key": idempotency_key,
                "api_base_url": self.config.api_base_url,
            }
        )
        response_payload = {"id": "plink_123", "url": "https://checkout.rebill.test/plink_123"}
        return RebillApiResult(
            provider_ref="plink_123",
            checkout_url="https://checkout.rebill.test/plink_123",
            http_status=201,
            payload=response_payload,
            payload_redacted=response_payload,
        )

    def get_payment(self, *, secret_key: str, payment_id: str) -> dict[str, object]:
        return self.__class__.payment_payload


def _configure_rebill(session: Session, workspace: WorkspaceRecord, user: UserRecord) -> None:
    upsert_commerce_provider_credentials(
        session,
        workspace_id=workspace.id,
        provider_key="rebill",
        payload=CommerceProviderCredentialUpsertRequest(
            environment="sandbox",
            enabled=True,
            api_base_url="https://api.rebill.test/v3",
            webhook_public_url="https://api.lean.test/api/v1/webhooks/rebill/url_secret/sandbox",
            secrets={
                "secret_key": "sk_rebill_test",
                "webhook_signing_secret": "whsec_rebill_test",
                "webhook_url_secret": "url_secret",
            },
        ),
        actor_user_id=user.id,
    )
    upsert_commerce_provider_mapping(
        session,
        workspace_id=workspace.id,
        provider_key="rebill",
        payload=CommerceProviderProductMappingUpsertRequest(
            environment="sandbox",
            internal_product_key="blueprint_pro",
            billing_mode="one_time",
            currency="USD",
            provider_product_id="prd_rebill_blueprint",
        ),
    )
    session.flush()


def test_rebill_checkout_provider_creates_hosted_checkout_with_provider_record(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user, workspace, record = _seed_checkout_context(db_session)
    _configure_rebill(db_session, workspace, user)
    FakeRebillClient.create_calls = []
    monkeypatch.setattr(RebillPaymentProvider, "client_factory", FakeRebillClient)

    response = create_checkout_session(
        db_session,
        payload=CommercialCheckoutSessionRequest(
            session_id=record.id,
            product_key="blueprint_pro",
            provider="rebill",
            idempotency_key=f"{record.id}:rebill-provider",
            success_url="https://example.test/success",
            cancel_url="https://example.test/cancel",
        ),
        record=record,
        current_user=user,
        base_url="http://localhost:3200",
    )
    db_session.commit()

    order = db_session.exec(select(CommercialOrderRecord).where(CommercialOrderRecord.id == response.order_id)).one()
    checkout_record = db_session.exec(select(CommerceProviderCheckoutRecord)).one()
    assert response.provider == "rebill"
    assert response.checkout_ref.startswith("rebill_")
    assert response.checkout_url == "https://checkout.rebill.test/plink_123"
    assert response.next_action == "open_checkout"
    assert order.metadata_payload["provider_stage"] == "rebill_payment_link_created"
    assert order.metadata_payload["rebill_payment_link_id"] == "plink_123"
    assert checkout_record.provider_key == "rebill"
    assert checkout_record.provider_payment_link_id == "plink_123"
    assert FakeRebillClient.create_calls[0]["secret_key"] == "sk_rebill_test"
    assert FakeRebillClient.create_calls[0]["idempotency_key"] == f"rebill:{order.id}:payment-link"
    payload = FakeRebillClient.create_calls[0]["payload"]
    assert isinstance(payload, dict)
    assert payload["metadata"]["lab_order_id"] == str(order.id)
    assert payload["metadata"]["lab_checkout_ref"] == response.checkout_ref
    assert payload["title"] == [{"language": "es", "text": "Blueprint Profesional"}]
    assert payload["paymentMethods"] == [{"currency": "USD", "methods": ["card"]}]
    assert payload["prefilledFields"] == {
        "customer": {"email": "commerce-provider@leanbuilder.local", "language": "es", "fullName": "Commerce Provider Tester"}
    }
    assert payload["redirectUrls"] == {
        "approved": "https://example.test/success",
        "rejected": "https://example.test/cancel",
    }
    assert payload["product"] == {
        "id": "prd_rebill_blueprint",
        "quantity": 1,
        "isQuantityEditable": False,
        "isRemovable": False,
    }
    assert "prices" not in payload
    assert "plan" not in payload


def test_rebill_webhook_approved_payment_uses_common_fulfillment_and_dedupes(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user, workspace, record = _seed_checkout_context(db_session)
    _configure_rebill(db_session, workspace, user)
    FakeRebillClient.create_calls = []
    monkeypatch.setattr(RebillPaymentProvider, "client_factory", FakeRebillClient)
    checkout = create_checkout_session(
        db_session,
        payload=CommercialCheckoutSessionRequest(
            session_id=record.id,
            product_key="blueprint_pro",
            provider="rebill",
            idempotency_key=f"{record.id}:rebill-webhook-provider",
        ),
        record=record,
        current_user=user,
        base_url="http://localhost:3200",
    )
    order = db_session.get(CommercialOrderRecord, checkout.order_id)
    assert order is not None
    payload = {
        "event": "payment.updated",
        "webhook": {"id": "evt_rebill_approved_1"},
        "data": {
            "id": "pay_rebill_123",
            "status": "approved",
            "amount": 49.0,
            "currency": "USD",
            "metadata": {
                "lab_order_id": str(order.id),
                "lab_checkout_ref": order.checkout_ref,
                "lab_workspace_id": str(workspace.id),
            },
        },
    }
    FakeRebillClient.payment_payload = {"data": payload["data"]}
    raw_body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signature = hmac.new(b"whsec_rebill_test", raw_body, hashlib.sha256).hexdigest()

    response = process_rebill_webhook(
        db_session,
        raw_body=raw_body,
        request_headers={"x-rebill-signature": signature},
        url_secret="url_secret",
        environment="sandbox",
        client_factory=FakeRebillClient,
    )
    duplicate = process_rebill_webhook(
        db_session,
        raw_body=raw_body,
        request_headers={"x-rebill-signature": signature},
        url_secret="url_secret",
        environment="sandbox",
        client_factory=FakeRebillClient,
    )
    db_session.commit()

    db_session.refresh(order)
    payments = db_session.exec(select(CommercialPaymentRecord).where(CommercialPaymentRecord.order_id == order.id)).all()
    entitlements = db_session.exec(
        select(CommercialEntitlementRecord).where(CommercialEntitlementRecord.order_id == order.id)
    ).all()
    webhook_event = db_session.exec(select(CommerceProviderWebhookEventRecord)).one()
    assert response.processing_status == "processed"
    assert duplicate.duplicate is True
    assert order.status == CommercialOrderStatus.paid
    assert len(payments) == 1
    assert payments[0].provider == "rebill"
    assert payments[0].provider_payment_id == "pay_rebill_123"
    assert len(entitlements) == 1
    assert webhook_event.signature_validated is True
    assert webhook_event.retries == 1


def test_rebill_webhook_rejects_invalid_signature(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user, workspace, record = _seed_checkout_context(db_session)
    _configure_rebill(db_session, workspace, user)
    monkeypatch.setattr(RebillPaymentProvider, "client_factory", FakeRebillClient)
    checkout = create_checkout_session(
        db_session,
        payload=CommercialCheckoutSessionRequest(
            session_id=record.id,
            product_key="blueprint_pro",
            provider="rebill",
            idempotency_key=f"{record.id}:rebill-invalid-signature",
        ),
        record=record,
        current_user=user,
        base_url="http://localhost:3200",
    )
    payload = {
        "event": "payment.updated",
        "webhook": {"id": "evt_rebill_invalid_sig"},
        "data": {
            "id": "pay_rebill_invalid",
            "status": "approved",
            "metadata": {
                "lab_order_id": str(checkout.order_id),
                "lab_workspace_id": str(workspace.id),
            },
        },
    }
    raw_body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    with pytest.raises(PermissionError, match="Invalid Rebill webhook signature"):
        process_rebill_webhook(
            db_session,
            raw_body=raw_body,
            request_headers={"x-rebill-signature": "bad-signature"},
            url_secret="url_secret",
            environment="sandbox",
            client_factory=FakeRebillClient,
        )

    order = db_session.get(CommercialOrderRecord, checkout.order_id)
    webhook_event = db_session.exec(select(CommerceProviderWebhookEventRecord)).one()
    assert order is not None
    assert order.status == CommercialOrderStatus.pending
    assert webhook_event.processing_status == "rejected"
    assert webhook_event.signature_validated is False


def test_checkout_rejects_open_redirect_urls(db_session: Session) -> None:
    user, _, record = _seed_checkout_context(db_session)

    with pytest.raises(ValueError, match="Redirect URL host is not permitted"):
        create_checkout_session(
            db_session,
            payload=CommercialCheckoutSessionRequest(
                session_id=record.id,
                product_key="blueprint_pro",
                success_url="https://phishing-site.example.org/steal",
            ),
            record=record,
            current_user=user,
            base_url="http://localhost:3200",
        )

    with pytest.raises(ValueError, match="Invalid redirect URL format"):
        create_checkout_session(
            db_session,
            payload=CommercialCheckoutSessionRequest(
                session_id=record.id,
                product_key="blueprint_pro",
                success_url="javascript:alert(1)",
            ),
            record=record,
            current_user=user,
            base_url="http://localhost:3200",
        )


def test_checkout_enforces_idempotency(db_session: Session) -> None:
    user, _, record = _seed_checkout_context(db_session)
    idempotency_key = f"{record.id}:idempotent-checkout-test"

    response1 = create_checkout_session(
        db_session,
        payload=CommercialCheckoutSessionRequest(
            session_id=record.id,
            product_key="blueprint_pro",
            idempotency_key=idempotency_key,
        ),
        record=record,
        current_user=user,
        base_url="http://localhost:3200",
    )
    db_session.commit()

    response2 = create_checkout_session(
        db_session,
        payload=CommercialCheckoutSessionRequest(
            session_id=record.id,
            product_key="blueprint_pro",
            idempotency_key=idempotency_key,
        ),
        record=record,
        current_user=user,
        base_url="http://localhost:3200",
    )

    assert response1.order_id == response2.order_id
    assert response1.checkout_ref == response2.checkout_ref
    orders = db_session.exec(
        select(CommercialOrderRecord).where(CommercialOrderRecord.idempotency_key == idempotency_key)
    ).all()
    assert len(orders) == 1


def test_checkout_enforces_buyer_permission(db_session: Session) -> None:
    _, workspace, record = _seed_checkout_context(db_session)
    viewer = UserRecord(
        email="viewer@leanbuilder.local",
        full_name="Viewer User",
        password_hash=hash_password("Secret123!"),
    )
    db_session.add(viewer)
    db_session.flush()
    db_session.add(
        WorkspaceMembershipRecord(
            workspace_id=workspace.id,
            user_id=viewer.id,
            role=WorkspaceRole.viewer,
        )
    )
    db_session.commit()

    with pytest.raises(PermissionError, match="Only workspace owners or admins can start checkout"):
        create_checkout_session(
            db_session,
            payload=CommercialCheckoutSessionRequest(
                session_id=record.id,
                product_key="blueprint_pro",
            ),
            record=record,
            current_user=viewer,
            base_url="http://localhost:3200",
        )


def test_sandbox_payment_settles_open_workspace_debt(db_session: Session) -> None:
    user, workspace, record = _seed_checkout_context(db_session)
    checkout = create_checkout_session(
        db_session,
        payload=CommercialCheckoutSessionRequest(
            session_id=record.id,
            product_key="blueprint_pro",
            idempotency_key=f"{record.id}:sandbox-debt-settlement",
        ),
        record=record,
        current_user=user,
        base_url="http://localhost:3200",
    )
    create_commercial_debt(
        db_session,
        workspace_id=workspace.id,
        product_key="blueprint_pro",
        access_request_id=None,
        amount_cents=checkout.total_cents,
        currency="USD",
        actor_user_id=user.id,
        reason_code="manual_debt",
        reason_label="Deuda manual",
    )
    db_session.commit()

    response = complete_checkout_session(
        db_session,
        checkout_ref=checkout.checkout_ref,
        request=CommercialCheckoutCompletionRequest(
            outcome="success",
            provider_payment_id=f"sandbox_pay_{checkout.checkout_ref}",
        ),
        current_user=user,
    )
    db_session.commit()

    order = db_session.exec(select(CommercialOrderRecord).where(CommercialOrderRecord.id == response.order_id)).one()
    payment = db_session.exec(select(CommercialPaymentRecord).where(CommercialPaymentRecord.order_id == order.id)).one()
    debts = db_session.exec(select(CommercialDebtRecord).where(CommercialDebtRecord.workspace_id == workspace.id)).all()
    assert len(debts) == 1
    assert debts[0].status == CommercialDebtStatus.settled
    assert debts[0].settled_amount_cents == checkout.total_cents
    assert payment.metadata_payload["debt_settlement"]["settled_amount_cents"] == checkout.total_cents
    assert order.metadata_payload["debt_settlement"]["currency"] == "USD"


def test_sandbox_payment_credits_workspace_balance_from_package_code(db_session: Session) -> None:
    user, workspace, record = _seed_checkout_context(db_session)
    upsert_package_catalog_entry(
        db_session,
        payload=CommercialPackageCatalogUpsertRequest(
            package_code="bp-pack-3",
            display_name="Blueprint Pack 3",
            product_key="blueprint_pro",
            granted_units=3,
            validity_days=30,
        ),
    )
    checkout = create_checkout_session(
        db_session,
        payload=CommercialCheckoutSessionRequest(
            session_id=record.id,
            product_key="blueprint_pro",
            package_code="bp-pack-3",
            idempotency_key=f"{record.id}:sandbox-package-credit",
        ),
        record=record,
        current_user=user,
        base_url="http://localhost:3200",
    )
    db_session.commit()

    response = complete_checkout_session(
        db_session,
        checkout_ref=checkout.checkout_ref,
        request=CommercialCheckoutCompletionRequest(
            outcome="success",
            provider_payment_id=f"sandbox_pay_{checkout.checkout_ref}",
        ),
        current_user=user,
    )
    db_session.commit()

    snapshot = get_balance_snapshot(db_session, workspace_id=workspace.id, product_key="blueprint_pro")
    order = db_session.exec(select(CommercialOrderRecord).where(CommercialOrderRecord.id == response.order_id)).one()
    payment = db_session.exec(select(CommercialPaymentRecord).where(CommercialPaymentRecord.order_id == order.id)).one()
    assert snapshot.total_available_units == 3
    assert payment.metadata_payload["package_credit"]["package_code"] == "bp-pack-3"
    assert payment.metadata_payload["package_credit"]["grants"][0]["units"] == 3
    assert order.metadata_payload["package_credit"]["grants"][0]["product_key"] == "blueprint_pro"
