from __future__ import annotations

from collections.abc import Iterator
import hashlib
import hmac
import json
from urllib.parse import urlencode

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
from app.services.payment_providers.rapyd import RapydPaymentProvider
from app.services.payu.checkout_redirect import render_payu_checkout_redirect, resolve_payu_response_redirect
from app.services.payu.signatures import format_payu_confirmation_value, sign_payu_payment_form
from app.services.payu.webhooks import process_payu_webhook
from app.services.rapyd.client import RapydApiResult
from app.services.rapyd.signatures import sign_rapyd_webhook
from app.services.rapyd.webhooks import process_rapyd_webhook
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
    assert normalize_commerce_payment_provider("PAYU") == "payu"
    assert normalize_commerce_payment_provider("RAPYD") == "rapyd"
    assert get_commerce_payment_provider("sandbox").provider_key == "sandbox"
    assert get_commerce_payment_provider("hotmart").provider_key == "hotmart"
    assert get_commerce_payment_provider("rebill").provider_key == "rebill"
    assert get_commerce_payment_provider("payu").provider_key == "payu"
    assert get_commerce_payment_provider("rapyd").provider_key == "rapyd"

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


class FakeRapydClient:
    create_calls: list[dict[str, object]] = []

    def __init__(self, config) -> None:
        self.config = config

    def create_checkout(
        self,
        *,
        access_key: str,
        secret_key: str,
        payload: dict[str, object],
        idempotency_key: str,
    ) -> RapydApiResult:
        self.__class__.create_calls.append(
            {
                "access_key": access_key,
                "secret_key": secret_key,
                "payload": payload,
                "idempotency_key": idempotency_key,
                "api_base_url": self.config.api_base_url,
            }
        )
        response_payload = {"data": {"id": "checkout_123", "redirect_url": "https://checkout.rapyd.test/checkout_123"}}
        return RapydApiResult(
            provider_ref="checkout_123",
            checkout_url="https://checkout.rapyd.test/checkout_123",
            http_status=200,
            payload=response_payload,
            payload_redacted=response_payload,
        )


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


def _configure_payu(session: Session, workspace: WorkspaceRecord, user: UserRecord) -> None:
    upsert_commerce_provider_credentials(
        session,
        workspace_id=workspace.id,
        provider_key="payu",
        payload=CommerceProviderCredentialUpsertRequest(
            environment="sandbox",
            enabled=True,
            api_base_url="https://sandbox.api.payulatam.test/payments-api/4.0/service.cgi",
            webhook_public_url="https://api.lean.test/api/v1/webhooks/payu/url_secret/sandbox",
            secrets={
                "secret_key": "4Vj8eK4rloUd272L48hsrarnUA",
                "public_key": "pRRXKOl8ikMmt9u",
                "merchant_id": "508029",
                "account_id": "512321",
                "webhook_url_secret": "url_secret",
            },
        ),
        actor_user_id=user.id,
    )
    upsert_commerce_provider_mapping(
        session,
        workspace_id=workspace.id,
        provider_key="payu",
        payload=CommerceProviderProductMappingUpsertRequest(
            environment="sandbox",
            internal_product_key="blueprint_pro",
            billing_mode="one_time",
            currency="USD",
            provider_product_id="512321",
            provider_plan_id="CO",
            provider_price_id="VISA,MASTERCARD",
        ),
    )
    session.flush()


def _configure_rapyd(session: Session, workspace: WorkspaceRecord, user: UserRecord) -> None:
    upsert_commerce_provider_credentials(
        session,
        workspace_id=workspace.id,
        provider_key="rapyd",
        payload=CommerceProviderCredentialUpsertRequest(
            environment="sandbox",
            enabled=True,
            api_base_url="https://sandboxapi.rapyd.test",
            webhook_public_url="https://api.lean.test/api/v1/webhooks/rapyd/url_secret/sandbox",
            secrets={
                "access_key": "access_rapyd_test",
                "secret_key": "secret_rapyd_test",
                "webhook_url_secret": "url_secret",
            },
        ),
        actor_user_id=user.id,
    )
    upsert_commerce_provider_mapping(
        session,
        workspace_id=workspace.id,
        provider_key="rapyd",
        payload=CommerceProviderProductMappingUpsertRequest(
            environment="sandbox",
            internal_product_key="blueprint_pro",
            billing_mode="one_time",
            currency="USD",
            provider_product_id="ewallet_lab",
            provider_plan_id="CO",
            provider_price_id="co_visa_card,co_pse_bank",
            provider_offer_ref="LAB Blueprint",
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


def test_payu_checkout_provider_creates_signed_webcheckout_redirect(
    db_session: Session,
) -> None:
    user, workspace, record = _seed_checkout_context(db_session)
    _configure_payu(db_session, workspace, user)

    response = create_checkout_session(
        db_session,
        payload=CommercialCheckoutSessionRequest(
            session_id=record.id,
            product_key="blueprint_pro",
            provider="payu",
            idempotency_key=f"{record.id}:payu-provider",
            success_url="https://example.test/success",
            cancel_url="https://example.test/cancel",
        ),
        record=record,
        current_user=user,
        base_url="http://localhost:3200",
    )
    db_session.commit()

    order = db_session.exec(select(CommercialOrderRecord).where(CommercialOrderRecord.id == response.order_id)).one()
    checkout_record = db_session.exec(select(CommerceProviderCheckoutRecord).where(CommerceProviderCheckoutRecord.provider_key == "payu")).one()
    fields = checkout_record.metadata_payload["payu_form_fields"]
    assert response.provider == "payu"
    assert response.checkout_ref.startswith("payu_")
    assert response.checkout_url.endswith(f"/api/v1/commerce/checkout-redirects/payu/{response.checkout_ref}")
    assert response.next_action == "open_checkout"
    assert order.metadata_payload["provider_stage"] == "payu_webcheckout_form_created"
    assert checkout_record.provider_checkout_id == fields["referenceCode"]
    assert checkout_record.checkout_url == response.checkout_url
    assert checkout_record.metadata_payload["payu_checkout_gateway_url"] == "https://sandbox.checkout.payulatam.com/ppp-web-gateway-payu/"
    assert fields["merchantId"] == "508029"
    assert fields["accountId"] == "512321"
    assert fields["amount"] == "49.00"
    assert fields["currency"] == "USD"
    assert fields["tax"] == "0"
    assert fields["taxReturnBase"] == "0"
    assert fields["buyerEmail"] == user.email
    assert fields["confirmationUrl"] == "https://api.lean.test/api/v1/webhooks/payu/url_secret/sandbox"
    assert fields["responseUrl"].endswith(f"/api/v1/commerce/checkout-responses/payu/{response.checkout_ref}")
    assert fields["paymentMethods"] == "VISA,MASTERCARD"
    assert fields["billingCountry"] == "CO"
    assert fields["extra1"] == order.checkout_ref
    assert fields["extra2"] == str(order.id)
    assert fields["extra3"] == str(workspace.id)
    assert fields["signature"] == sign_payu_payment_form(
        api_key="4Vj8eK4rloUd272L48hsrarnUA",
        merchant_id="508029",
        reference_code=fields["referenceCode"],
        amount="49.00",
        currency="USD",
        payment_methods="VISA,MASTERCARD",
    )

    html = render_payu_checkout_redirect(db_session, checkout_ref=response.checkout_ref)
    assert 'method="post"' in html
    assert 'action="https://sandbox.checkout.payulatam.com/ppp-web-gateway-payu/"' in html
    assert f'name="referenceCode" value="{fields["referenceCode"]}"' in html


def test_rapyd_checkout_provider_creates_hosted_checkout_with_provider_record(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user, workspace, record = _seed_checkout_context(db_session)
    _configure_rapyd(db_session, workspace, user)
    FakeRapydClient.create_calls = []
    monkeypatch.setattr(RapydPaymentProvider, "client_factory", FakeRapydClient)

    response = create_checkout_session(
        db_session,
        payload=CommercialCheckoutSessionRequest(
            session_id=record.id,
            product_key="blueprint_pro",
            provider="rapyd",
            idempotency_key=f"{record.id}:rapyd-provider",
            success_url="https://example.test/success",
            cancel_url="https://example.test/cancel",
        ),
        record=record,
        current_user=user,
        base_url="http://localhost:3200",
    )
    db_session.commit()

    order = db_session.exec(select(CommercialOrderRecord).where(CommercialOrderRecord.id == response.order_id)).one()
    checkout_record = db_session.exec(
        select(CommerceProviderCheckoutRecord).where(CommerceProviderCheckoutRecord.provider_key == "rapyd")
    ).one()
    assert response.provider == "rapyd"
    assert response.checkout_ref.startswith("rapyd_")
    assert response.checkout_url == "https://checkout.rapyd.test/checkout_123"
    assert response.next_action == "open_checkout"
    assert order.metadata_payload["provider_stage"] == "rapyd_checkout_created"
    assert order.metadata_payload["rapyd_checkout_id"] == "checkout_123"
    assert checkout_record.provider_checkout_id == "checkout_123"
    assert checkout_record.checkout_url == response.checkout_url
    assert FakeRapydClient.create_calls[0]["access_key"] == "access_rapyd_test"
    assert FakeRapydClient.create_calls[0]["secret_key"] == "secret_rapyd_test"
    assert FakeRapydClient.create_calls[0]["idempotency_key"] == f"rapyd:{order.id}:checkout"
    assert FakeRapydClient.create_calls[0]["api_base_url"] == "https://sandboxapi.rapyd.test"
    payload = FakeRapydClient.create_calls[0]["payload"]
    assert isinstance(payload, dict)
    assert payload["amount"] == 49.0
    assert payload["country"] == "CO"
    assert payload["currency"] == "USD"
    assert payload["merchant_reference_id"] == order.checkout_ref
    assert payload["complete_payment_url"] == "https://example.test/success"
    assert payload["error_payment_url"] == "https://example.test/cancel"
    assert payload["payment_method_types_include"] == ["co_visa_card", "co_pse_bank"]
    assert payload["merchant_ewallet"] == "ewallet_lab"
    assert payload["statement_descriptor"] == "LAB Blueprint"
    assert payload["metadata"]["lab_order_id"] == str(order.id)
    assert payload["metadata"]["lab_checkout_ref"] == response.checkout_ref
    assert checkout_record.request_payload_redacted["metadata"]["lab_provider"] == "rapyd"


def test_rapyd_checkout_uses_market_mapping_amount_and_currency(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user, workspace, record = _seed_checkout_context(db_session)
    _configure_rapyd(db_session, workspace, user)
    upsert_package_catalog_entry(
        db_session,
        payload=CommercialPackageCatalogUpsertRequest(
            package_code="blueprint_pro_co",
            display_name="Blueprint Pro - Colombia",
            product_key="blueprint_pro",
            granted_units=1,
            granted_units_blueprint_pro=1,
            offer_code="rapyd_blueprint_pro_co",
            plan_code="blueprint_pro_co",
        ),
    )
    upsert_commerce_provider_mapping(
        db_session,
        workspace_id=workspace.id,
        provider_key="rapyd",
        payload=CommerceProviderProductMappingUpsertRequest(
            environment="sandbox",
            internal_product_key="blueprint_pro",
            package_code="blueprint_pro_co",
            billing_mode="one_time",
            currency="COP",
            internal_unit_amount_usd_cents=19_900_000,
            provider_plan_id="CO",
            provider_offer_ref="LAB Blueprint",
            grants_tier="blueprint_pro",
        ),
    )
    FakeRapydClient.create_calls = []
    monkeypatch.setattr(RapydPaymentProvider, "client_factory", FakeRapydClient)

    response = create_checkout_session(
        db_session,
        payload=CommercialCheckoutSessionRequest(
            session_id=record.id,
            product_key="blueprint_pro",
            package_code="blueprint_pro_co",
            provider="rapyd",
            idempotency_key=f"{record.id}:rapyd-co-provider",
        ),
        record=record,
        current_user=user,
        base_url="http://localhost:3200",
    )
    db_session.commit()

    order = db_session.exec(select(CommercialOrderRecord).where(CommercialOrderRecord.id == response.order_id)).one()
    checkout_record = db_session.exec(
        select(CommerceProviderCheckoutRecord).where(CommerceProviderCheckoutRecord.provider_key == "rapyd")
    ).one()
    payload = FakeRapydClient.create_calls[0]["payload"]
    assert order.currency == "USD"
    assert order.total_cents == 4900
    assert payload["amount"] == 199000.0
    assert payload["country"] == "CO"
    assert payload["currency"] == "COP"
    assert payload["metadata"]["lab_package_code"] == "blueprint_pro_co"
    assert checkout_record.amount_cents == 19_900_000
    assert checkout_record.currency == "COP"


def test_payu_checkout_response_redirects_by_verified_browser_state(
    db_session: Session,
) -> None:
    user, workspace, record = _seed_checkout_context(db_session)
    _configure_payu(db_session, workspace, user)
    checkout = create_checkout_session(
        db_session,
        payload=CommercialCheckoutSessionRequest(
            session_id=record.id,
            product_key="blueprint_pro",
            provider="payu",
            idempotency_key=f"{record.id}:payu-response-provider",
            success_url="https://example.test/success",
            cancel_url="https://example.test/cancel",
        ),
        record=record,
        current_user=user,
        base_url="http://localhost:3200",
    )
    checkout_record = db_session.exec(select(CommerceProviderCheckoutRecord).where(CommerceProviderCheckoutRecord.provider_key == "payu")).one()
    query = {
        "merchantId": "508029",
        "referenceCode": checkout_record.provider_checkout_id,
        "TX_VALUE": "49.00",
        "currency": "USD",
        "transactionState": "4",
        "lapTransactionState": "APPROVED",
    }
    query["signature"] = hashlib.md5(
        f"4Vj8eK4rloUd272L48hsrarnUA~508029~{query['referenceCode']}~49.0~USD~4".encode("utf-8")
    ).hexdigest()

    approved_redirect = resolve_payu_response_redirect(db_session, checkout_ref=checkout.checkout_ref, query_params=query)
    rejected_redirect = resolve_payu_response_redirect(
        db_session,
        checkout_ref=checkout.checkout_ref,
        query_params={**query, "signature": "bad-signature"},
    )

    assert approved_redirect.startswith("https://example.test/success?")
    assert "payment_status=approved" in approved_redirect
    assert rejected_redirect.startswith("https://example.test/cancel?")
    assert "payment_status=unverified" in rejected_redirect


def test_payu_confirmation_approved_payment_uses_common_fulfillment_and_dedupes(
    db_session: Session,
) -> None:
    user, workspace, record = _seed_checkout_context(db_session)
    _configure_payu(db_session, workspace, user)
    checkout = create_checkout_session(
        db_session,
        payload=CommercialCheckoutSessionRequest(
            session_id=record.id,
            product_key="blueprint_pro",
            provider="payu",
            idempotency_key=f"{record.id}:payu-webhook-provider",
        ),
        record=record,
        current_user=user,
        base_url="http://localhost:3200",
    )
    order = db_session.get(CommercialOrderRecord, checkout.order_id)
    checkout_record = db_session.exec(select(CommerceProviderCheckoutRecord).where(CommerceProviderCheckoutRecord.provider_key == "payu")).one()
    assert order is not None
    payload = {
        "merchant_id": "508029",
        "reference_sale": checkout_record.provider_checkout_id,
        "value": "49.00",
        "currency": "USD",
        "state_pol": "4",
        "reference_pol": "7069375",
        "transaction_id": "payu_txn_123",
        "response_code_pol": "APPROVED",
        "extra1": order.checkout_ref,
        "extra2": str(order.id),
        "extra3": str(workspace.id),
    }
    payload["sign"] = hashlib.md5(
        f"4Vj8eK4rloUd272L48hsrarnUA~508029~{payload['reference_sale']}~49.0~USD~4".encode("utf-8")
    ).hexdigest()
    raw_body = urlencode(payload).encode("utf-8")

    response = process_payu_webhook(
        db_session,
        raw_body=raw_body,
        request_headers={"content-type": "application/x-www-form-urlencoded"},
        url_secret="url_secret",
        environment="sandbox",
    )
    duplicate = process_payu_webhook(
        db_session,
        raw_body=raw_body,
        request_headers={"content-type": "application/x-www-form-urlencoded"},
        url_secret="url_secret",
        environment="sandbox",
    )
    db_session.commit()

    db_session.refresh(order)
    payments = db_session.exec(select(CommercialPaymentRecord).where(CommercialPaymentRecord.order_id == order.id)).all()
    entitlements = db_session.exec(
        select(CommercialEntitlementRecord).where(CommercialEntitlementRecord.order_id == order.id)
    ).all()
    webhook_event = db_session.exec(select(CommerceProviderWebhookEventRecord).where(CommerceProviderWebhookEventRecord.provider_key == "payu")).one()
    assert response.processing_status == "processed"
    assert duplicate.duplicate is True
    assert order.status == CommercialOrderStatus.paid
    assert len(payments) == 1
    assert payments[0].provider == "payu"
    assert payments[0].provider_payment_id == "payu_txn_123"
    assert len(entitlements) == 1
    assert webhook_event.signature_validated is True
    assert webhook_event.retries == 1


def test_payu_confirmation_rejects_invalid_signature(
    db_session: Session,
) -> None:
    user, workspace, record = _seed_checkout_context(db_session)
    _configure_payu(db_session, workspace, user)
    checkout = create_checkout_session(
        db_session,
        payload=CommercialCheckoutSessionRequest(
            session_id=record.id,
            product_key="blueprint_pro",
            provider="payu",
            idempotency_key=f"{record.id}:payu-invalid-signature",
        ),
        record=record,
        current_user=user,
        base_url="http://localhost:3200",
    )
    order = db_session.get(CommercialOrderRecord, checkout.order_id)
    assert order is not None
    payload = {
        "merchant_id": "508029",
        "reference_sale": str(order.metadata_payload["payu_reference_code"]),
        "value": "49.00",
        "currency": "USD",
        "state_pol": "4",
        "transaction_id": "payu_txn_invalid",
        "sign": "bad-signature",
        "extra1": order.checkout_ref,
        "extra2": str(order.id),
        "extra3": str(workspace.id),
    }

    with pytest.raises(PermissionError, match="Invalid PayU confirmation signature"):
        process_payu_webhook(
            db_session,
            raw_body=urlencode(payload).encode("utf-8"),
            request_headers={"content-type": "application/x-www-form-urlencoded"},
            url_secret="url_secret",
            environment="sandbox",
        )

    webhook_event = db_session.exec(select(CommerceProviderWebhookEventRecord).where(CommerceProviderWebhookEventRecord.provider_key == "payu")).one()
    assert order.status == CommercialOrderStatus.pending
    assert webhook_event.processing_status == "rejected"
    assert webhook_event.signature_validated is False


def test_payu_confirmation_amount_formatting_matches_payu_rounding_rules() -> None:
    assert format_payu_confirmation_value("150.00") == "150.0"
    assert format_payu_confirmation_value("150.25") == "150.25"
    assert format_payu_confirmation_value("150") == "150.0"


def test_rapyd_webhook_payment_succeeded_uses_common_fulfillment_and_dedupes(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user, workspace, record = _seed_checkout_context(db_session)
    _configure_rapyd(db_session, workspace, user)
    FakeRapydClient.create_calls = []
    monkeypatch.setattr(RapydPaymentProvider, "client_factory", FakeRapydClient)
    checkout = create_checkout_session(
        db_session,
        payload=CommercialCheckoutSessionRequest(
            session_id=record.id,
            product_key="blueprint_pro",
            provider="rapyd",
            idempotency_key=f"{record.id}:rapyd-webhook-provider",
        ),
        record=record,
        current_user=user,
        base_url="http://localhost:3200",
    )
    order = db_session.get(CommercialOrderRecord, checkout.order_id)
    assert order is not None
    payload = {
        "id": "wh_rapyd_approved_1",
        "type": "PAYMENT_SUCCEEDED",
        "data": {
            "id": "payment_rapyd_123",
            "status": "CLO",
            "amount": 49.0,
            "currency_code": "USD",
            "merchant_reference_id": order.checkout_ref,
            "metadata": {
                "lab_order_id": str(order.id),
                "lab_checkout_ref": order.checkout_ref,
                "lab_workspace_id": str(workspace.id),
            },
        },
    }
    raw_body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = {
        "access_key": "access_rapyd_test",
        "salt": "salt12345678",
        "timestamp": "1700000000",
        "signature": sign_rapyd_webhook(
            webhook_url="https://api.lean.test/api/v1/webhooks/rapyd/url_secret/sandbox",
            salt="salt12345678",
            timestamp="1700000000",
            access_key="access_rapyd_test",
            secret_key="secret_rapyd_test",
            body_string=raw_body.decode("utf-8"),
        ),
    }

    response = process_rapyd_webhook(
        db_session,
        raw_body=raw_body,
        request_headers=headers,
        url_secret="url_secret",
        environment="sandbox",
    )
    duplicate = process_rapyd_webhook(
        db_session,
        raw_body=raw_body,
        request_headers=headers,
        url_secret="url_secret",
        environment="sandbox",
    )
    db_session.commit()

    db_session.refresh(order)
    payments = db_session.exec(select(CommercialPaymentRecord).where(CommercialPaymentRecord.order_id == order.id)).all()
    entitlements = db_session.exec(
        select(CommercialEntitlementRecord).where(CommercialEntitlementRecord.order_id == order.id)
    ).all()
    webhook_event = db_session.exec(select(CommerceProviderWebhookEventRecord).where(CommerceProviderWebhookEventRecord.provider_key == "rapyd")).one()
    assert response.processing_status == "processed"
    assert duplicate.duplicate is True
    assert order.status == CommercialOrderStatus.paid
    assert len(payments) == 1
    assert payments[0].provider == "rapyd"
    assert payments[0].provider_payment_id == "payment_rapyd_123"
    assert len(entitlements) == 1
    assert webhook_event.signature_validated is True
    assert webhook_event.retries == 1


def test_rapyd_webhook_rejects_invalid_signature(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user, workspace, record = _seed_checkout_context(db_session)
    _configure_rapyd(db_session, workspace, user)
    monkeypatch.setattr(RapydPaymentProvider, "client_factory", FakeRapydClient)
    checkout = create_checkout_session(
        db_session,
        payload=CommercialCheckoutSessionRequest(
            session_id=record.id,
            product_key="blueprint_pro",
            provider="rapyd",
            idempotency_key=f"{record.id}:rapyd-invalid-signature",
        ),
        record=record,
        current_user=user,
        base_url="http://localhost:3200",
    )
    payload = {
        "id": "wh_rapyd_invalid_sig",
        "type": "PAYMENT_SUCCEEDED",
        "data": {
            "id": "payment_rapyd_invalid",
            "status": "CLO",
            "metadata": {
                "lab_order_id": str(checkout.order_id),
                "lab_workspace_id": str(workspace.id),
            },
        },
    }
    raw_body = json.dumps(payload, separators=(",", ":")).encode("utf-8")

    with pytest.raises(PermissionError, match="Invalid Rapyd webhook signature"):
        process_rapyd_webhook(
            db_session,
            raw_body=raw_body,
            request_headers={
                "access_key": "access_rapyd_test",
                "salt": "salt12345678",
                "timestamp": "1700000000",
                "signature": "bad-signature",
            },
            url_secret="url_secret",
            environment="sandbox",
        )

    order = db_session.get(CommercialOrderRecord, checkout.order_id)
    webhook_event = db_session.exec(select(CommerceProviderWebhookEventRecord).where(CommerceProviderWebhookEventRecord.provider_key == "rapyd")).one()
    assert order is not None
    assert order.status == CommercialOrderStatus.pending
    assert webhook_event.processing_status == "rejected"
    assert webhook_event.signature_validated is False


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
