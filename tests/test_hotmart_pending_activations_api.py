from __future__ import annotations

from collections.abc import Generator
from uuid import UUID

from fastapi.testclient import TestClient
import pytest
from sqlmodel import Session, select

from app.db import get_session
from app.models import (
    CommercialOrderRecord,
    CommercialPackageCatalogUpsertRequest,
    CommercialPaymentRecord,
    HotmartCredentialUpsertRequest,
    HotmartPendingActivationRecord,
    SessionRecord,
    UserRecord,
)
from app.services.commercial_catalog_service import upsert_package_catalog_entry
from app.services.hotmart.secrets import upsert_hotmart_credentials
from app.services.hotmart.webhooks import process_hotmart_webhook
from app.services.product_processing.persistence import ProductBuildRunRecord
from tests.api_testkit import TEST_EMAIL, TEST_PASSWORD, build_test_client


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    with build_test_client(monkeypatch) as test_client:
        yield test_client


def _auth_headers(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _external_sale_payload(*, buyer_email: str) -> dict[str, object]:
    return {
        "id": "evt-api-external-claim",
        "event": "PURCHASE_APPROVED",
        "data": {
            "buyer": {
                "name": "API Hotmart Buyer",
                "email": buyer_email,
                "document": "1234567890",
            },
            "purchase": {
                "transaction": "HP-API-901",
                "price": {"value": 149.0, "currency_code": "USD"},
                "product": {"id": "hm-prod-bundle", "ucode": "HMBUNDLE"},
                "offer": {"code": "offer-bundle"},
                "plan": {"code": "plan-bundle"},
            },
        },
    }


def test_pending_activation_routes_list_and_claim_external_hotmart_sale(client: TestClient) -> None:
    headers = _auth_headers(client)
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = UUID(create_response.json()["id"])

    session_override = client.app.dependency_overrides[get_session]
    session_generator = session_override()
    db = next(session_generator)
    try:
        user = db.exec(select(UserRecord).where(UserRecord.email == TEST_EMAIL)).one()
        record = db.get(SessionRecord, session_id)
        assert record is not None
        assert record.workspace_id is not None
        upsert_hotmart_credentials(
            db,
            workspace_id=record.workspace_id,
            payload=HotmartCredentialUpsertRequest(
                environment="sandbox",
                enabled=True,
                client_id="client-id-value",
                client_secret="client-secret-value",
                basic_token="basic-token-value",
                hottok="hottok-value",
            ),
        )
        upsert_package_catalog_entry(
            db,
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
        process_hotmart_webhook(
            db,
            payload=_external_sale_payload(buyer_email=user.email),
            hottok_header="hottok-value",
            environment="sandbox",
        )
        db.commit()
        pending = db.exec(select(HotmartPendingActivationRecord)).one()
    finally:
        session_generator.close()

    list_response = client.get("/api/v1/commerce/hotmart/pending-activations", headers=headers)
    assert list_response.status_code == 200
    listed = list_response.json()
    assert len(listed) == 1
    assert listed[0]["id"] == str(pending.id)
    assert listed[0]["status"] == "pending_activation"

    claim_response = client.post(
        f"/api/v1/commerce/hotmart/pending-activations/{pending.activation_token}/claim",
        headers=headers,
        json={"session_id": str(session_id)},
    )
    assert claim_response.status_code == 200
    claimed = claim_response.json()
    assert claimed["status"] == "claimed"
    assert claimed["claimed_session_id"] == str(session_id)

    session_generator = session_override()
    db = next(session_generator)
    try:
        adopted_order = db.get(CommercialOrderRecord, UUID(claimed["adopted_order_id"]))
        adopted_payment = db.get(CommercialPaymentRecord, UUID(claimed["adopted_payment_id"]))
        runs = db.exec(
            select(ProductBuildRunRecord).where(
                ProductBuildRunRecord.session_id == session_id,
                ProductBuildRunRecord.product_key == "acp",
            )
        ).all()
    finally:
        session_generator.close()

    assert adopted_order is not None
    assert adopted_payment is not None
    assert len(runs) == 1


def test_pending_activation_public_and_bootstrap_routes_create_session_for_buyer(client: TestClient) -> None:
    headers = _auth_headers(client)

    session_override = client.app.dependency_overrides[get_session]
    session_generator = session_override()
    db = next(session_generator)
    try:
        user = db.exec(select(UserRecord).where(UserRecord.email == TEST_EMAIL)).one()
        user_id = user.id
        assert user.default_workspace_id is not None
        existing_session_count = len(
            db.exec(select(SessionRecord).where(SessionRecord.user_id == user_id)).all()
        )
        upsert_hotmart_credentials(
            db,
            workspace_id=user.default_workspace_id,
            payload=HotmartCredentialUpsertRequest(
                environment="sandbox",
                enabled=True,
                client_id="client-id-value",
                client_secret="client-secret-value",
                basic_token="basic-token-value",
                hottok="hottok-value",
            ),
        )
        upsert_package_catalog_entry(
            db,
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
        process_hotmart_webhook(
            db,
            payload=_external_sale_payload(
                buyer_email=user.email,
            ),
            hottok_header="hottok-value",
            environment="sandbox",
        )
        db.commit()
        pending = db.exec(select(HotmartPendingActivationRecord)).one()
    finally:
        session_generator.close()

    public_response = client.get(
        f"/api/v1/commerce/hotmart/pending-activations/{pending.activation_token}/public"
    )
    assert public_response.status_code == 200
    public_payload = public_response.json()
    assert public_payload["buyer_email"] == TEST_EMAIL
    assert public_payload["can_bootstrap"] is True
    assert public_payload["display_name"] == "Bundle Hotmart mensual"

    bootstrap_response = client.post(
        f"/api/v1/commerce/hotmart/pending-activations/{pending.activation_token}/bootstrap",
        headers=headers,
    )
    assert bootstrap_response.status_code == 200
    bootstrap_payload = bootstrap_response.json()
    assert bootstrap_payload["created_session"] is True
    assert bootstrap_payload["redirect_path"].endswith("/discover")
    assert bootstrap_payload["product_redirect_path"].endswith("/acp")
    assert bootstrap_payload["pending_activation"]["status"] == "claimed"

    session_generator = session_override()
    db = next(session_generator)
    try:
        updated_pending = db.get(HotmartPendingActivationRecord, pending.id)
        adopted_order = db.get(
            CommercialOrderRecord,
            UUID(bootstrap_payload["pending_activation"]["adopted_order_id"]),
        )
        adopted_payment = db.get(
            CommercialPaymentRecord,
            UUID(bootstrap_payload["pending_activation"]["adopted_payment_id"]),
        )
        created_session = db.get(SessionRecord, UUID(bootstrap_payload["session_id"]))
        runs = db.exec(
            select(ProductBuildRunRecord).where(
                ProductBuildRunRecord.session_id == UUID(bootstrap_payload["session_id"]),
                ProductBuildRunRecord.product_key == "acp",
            )
        ).all()
        updated_session_count = len(
            db.exec(select(SessionRecord).where(SessionRecord.user_id == user_id)).all()
        )
    finally:
        session_generator.close()

    assert updated_pending is not None
    assert adopted_order is not None
    assert adopted_payment is not None
    assert created_session is not None
    assert updated_pending.claimed_session_id == created_session.id
    assert updated_session_count == existing_session_count + 1
    assert len(runs) == 1
