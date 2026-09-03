from __future__ import annotations

from collections.abc import Iterator

from fastapi.testclient import TestClient
from cryptography.fernet import Fernet
import httpx
import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.models import (
    HotmartCredentialUpsertRequest,
    HotmartIntegrationConfigRecord,
    HotmartIntegrationSecretRecord,
    WorkspaceRecord,
)
from app.core.config import get_settings
from app.services.hotmart.secrets import (
    build_hotmart_status,
    load_hotmart_credentials,
    load_hotmart_hottok,
    test_hotmart_connection as run_hotmart_connection_test,
    upsert_hotmart_credentials,
)
from tests.api_testkit import TEST_EMAIL, TEST_PASSWORD, build_test_client


@pytest.fixture(autouse=True)
def _isolate_hotmart_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "hotmart_enabled", False)
    monkeypatch.setattr(settings, "hotmart_environment", "sandbox")
    monkeypatch.setattr(settings, "hotmart_api_base_url", "")
    monkeypatch.setattr(settings, "hotmart_auth_base_url", "")
    monkeypatch.setattr(settings, "hotmart_webhook_public_url", "")
    monkeypatch.setattr(settings, "hotmart_client_id", "")
    monkeypatch.setattr(settings, "hotmart_client_secret", "")
    monkeypatch.setattr(settings, "hotmart_basic_token", "")
    monkeypatch.setattr(settings, "hotmart_hottok", "")


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


def _workspace(session: Session) -> WorkspaceRecord:
    workspace = WorkspaceRecord(name="Hotmart Test Workspace", slug="hotmart-test-workspace")
    session.add(workspace)
    session.commit()
    session.refresh(workspace)
    return workspace


def test_hotmart_status_is_redacted_when_not_configured(db_session: Session) -> None:
    workspace = _workspace(db_session)

    status = build_hotmart_status(db_session, workspace_id=workspace.id, environment="sandbox")

    assert status.status == "not_configured"
    assert status.client_id_configured is False
    assert status.client_secret_configured is False
    assert status.basic_token_configured is False
    assert status.hottok_configured is False
    assert "[redacted]" not in status.model_dump_json()


def test_hotmart_credentials_are_encrypted_and_never_returned(db_session: Session) -> None:
    workspace = _workspace(db_session)
    payload = HotmartCredentialUpsertRequest(
        environment="sandbox",
        enabled=True,
        client_id="client-id-value",
        client_secret="client-secret-value",
        basic_token="basic-token-value",
        hottok="hottok-value",
    )

    status = upsert_hotmart_credentials(db_session, workspace_id=workspace.id, payload=payload)
    db_session.commit()

    assert status.status == "configured"
    assert status.client_id_configured is True
    assert status.client_secret_configured is True
    assert status.basic_token_configured is True
    assert status.hottok_configured is True
    serialized = status.model_dump_json()
    assert "client-id-value" not in serialized
    assert "client-secret-value" not in serialized
    assert "basic-token-value" not in serialized
    assert "hottok-value" not in serialized

    secret_records = db_session.exec(select(HotmartIntegrationSecretRecord)).all()
    assert len(secret_records) == 4
    stored = " ".join(record.secret_ciphertext for record in secret_records)
    assert "client-id-value" not in stored
    assert "client-secret-value" not in stored
    assert "basic-token-value" not in stored
    assert "hottok-value" not in stored


def test_hotmart_status_detects_environment_credentials_without_exposing_values(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(db_session)
    settings = get_settings()
    monkeypatch.setattr(settings, "hotmart_environment", "sandbox")
    monkeypatch.setattr(settings, "hotmart_client_id", "env-client-id")
    monkeypatch.setattr(settings, "hotmart_client_secret", "env-client-secret")
    monkeypatch.setattr(settings, "hotmart_basic_token", "env-basic-token")
    monkeypatch.setattr(settings, "hotmart_hottok", "env-hottok")

    status = build_hotmart_status(db_session, workspace_id=workspace.id, environment="sandbox")

    assert status.status == "configured"
    assert status.storage_mode == "environment"
    assert status.client_id_configured is True
    assert status.client_secret_configured is True
    assert status.basic_token_configured is True
    assert status.hottok_configured is True
    serialized = status.model_dump_json()
    assert "env-client-id" not in serialized
    assert "env-client-secret" not in serialized
    assert "env-basic-token" not in serialized
    assert "env-hottok" not in serialized


def test_hotmart_test_connection_uses_environment_credentials_without_db_config(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(db_session)
    settings = get_settings()
    monkeypatch.setattr(settings, "hotmart_environment", "sandbox")
    monkeypatch.setattr(settings, "hotmart_client_id", "env-client-id")
    monkeypatch.setattr(settings, "hotmart_client_secret", "env-client-secret")
    monkeypatch.setattr(settings, "hotmart_basic_token", "env-basic-token")
    monkeypatch.setattr(settings, "hotmart_hottok", "")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api-sec-vlc.hotmart.com"
        assert request.url.path == "/security/oauth/token"
        assert request.url.params["client_id"] == "env-client-id"
        assert request.url.params["client_secret"] == "env-client-secret"
        assert request.headers["Authorization"] == "Basic env-basic-token"
        return httpx.Response(
            200,
            json={
                "access_token": "env-hotmart-access-token-value",
                "token_type": "bearer",
                "expires_in": 1800,
            },
        )

    result = run_hotmart_connection_test(
        db_session,
        workspace_id=workspace.id,
        environment="sandbox",
        transport=httpx.MockTransport(handler),
    )
    db_session.commit()

    assert result.reachable is True
    assert result.status == "connected"
    assert result.token_expires_in == 1800
    serialized = result.model_dump_json()
    assert "env-hotmart-access-token-value" not in serialized
    assert "env-client-secret" not in serialized
    config = db_session.exec(select(HotmartIntegrationConfigRecord)).one()
    assert config.status == "connected"
    assert config.last_health_status == "connected"


def test_hotmart_loaders_fall_back_to_environment_when_db_secret_cannot_be_decrypted(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(db_session)
    settings = get_settings()
    monkeypatch.setattr(settings, "hotmart_environment", "production")
    monkeypatch.setattr(settings, "runtime_secrets_master_key", Fernet.generate_key().decode("utf-8"))
    upsert_hotmart_credentials(
        db_session,
        workspace_id=workspace.id,
        payload=HotmartCredentialUpsertRequest(
            environment="production",
            enabled=True,
            client_id="db-client-id",
            client_secret="db-client-secret",
            basic_token="db-basic-token",
            hottok="db-hottok-value",
        ),
    )
    db_session.commit()

    monkeypatch.setattr(settings, "runtime_secrets_master_key", Fernet.generate_key().decode("utf-8"))
    monkeypatch.setattr(settings, "hotmart_client_id", "env-client-id")
    monkeypatch.setattr(settings, "hotmart_client_secret", "env-client-secret")
    monkeypatch.setattr(settings, "hotmart_basic_token", "env-basic-token")
    monkeypatch.setattr(settings, "hotmart_hottok", "env-hottok-value")

    credentials = load_hotmart_credentials(db_session, workspace_id=workspace.id, environment="production")
    hottok = load_hotmart_hottok(db_session, workspace_id=workspace.id, environment="production")

    assert credentials is not None
    assert credentials.client_id == "env-client-id"
    assert credentials.client_secret == "env-client-secret"
    assert credentials.basic_token == "env-basic-token"
    assert hottok == "env-hottok-value"


def test_hotmart_effective_runtime_prefers_environment_over_database_config(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(db_session)
    settings = get_settings()
    monkeypatch.setattr(settings, "hotmart_environment", "production")
    monkeypatch.setattr(settings, "hotmart_enabled", True)
    monkeypatch.setattr(settings, "hotmart_api_base_url", "https://env-api.hotmart.test")
    monkeypatch.setattr(settings, "hotmart_auth_base_url", "https://env-auth.hotmart.test")
    monkeypatch.setattr(settings, "hotmart_webhook_public_url", "https://env-webhook.test/hotmart")
    monkeypatch.setattr(settings, "hotmart_client_id", "env-client-id")
    monkeypatch.setattr(settings, "hotmart_client_secret", "env-client-secret")
    monkeypatch.setattr(settings, "hotmart_basic_token", "env-basic-token")
    monkeypatch.setattr(settings, "hotmart_hottok", "env-hottok")
    upsert_hotmart_credentials(
        db_session,
        workspace_id=workspace.id,
        payload=HotmartCredentialUpsertRequest(
            environment="production",
            enabled=True,
            api_base_url="https://db-api.hotmart.test",
            auth_base_url="https://db-auth.hotmart.test",
            webhook_public_url="https://db-webhook.test/hotmart",
            client_id="db-client-id",
            client_secret="db-client-secret",
            basic_token="db-basic-token",
            hottok="db-hottok",
        ),
    )
    db_session.commit()

    status = build_hotmart_status(db_session, workspace_id=workspace.id, environment="production")
    credentials = load_hotmart_credentials(db_session, workspace_id=workspace.id, environment="production")
    hottok = load_hotmart_hottok(db_session, workspace_id=workspace.id, environment="production")

    assert status.api_base_url == "https://env-api.hotmart.test"
    assert status.auth_base_url == "https://env-auth.hotmart.test"
    assert status.webhook_public_url == "https://env-webhook.test/hotmart"
    assert credentials is not None
    assert credentials.client_id == "env-client-id"
    assert credentials.client_secret == "env-client-secret"
    assert credentials.basic_token == "env-basic-token"
    assert hottok == "env-hottok"


def test_hotmart_test_connection_uses_sandbox_oauth_and_redacts_token(db_session: Session) -> None:
    workspace = _workspace(db_session)
    upsert_hotmart_credentials(
        db_session,
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
    db_session.commit()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.scheme == "https"
        assert request.url.host == "api-sec-vlc.hotmart.com"
        assert request.url.path == "/security/oauth/token"
        assert request.url.params["grant_type"] == "client_credentials"
        assert request.url.params["client_id"] == "client-id-value"
        assert request.url.params["client_secret"] == "client-secret-value"
        assert request.headers["Authorization"] == "Basic basic-token-value"
        return httpx.Response(
            200,
            json={
                "access_token": "hotmart-access-token-value",
                "token_type": "bearer",
                "expires_in": 3600,
            },
        )

    result = run_hotmart_connection_test(
        db_session,
        workspace_id=workspace.id,
        environment="sandbox",
        transport=httpx.MockTransport(handler),
    )
    db_session.commit()

    assert result.reachable is True
    assert result.status == "connected"
    assert result.token_expires_in == 3600
    serialized = result.model_dump_json()
    assert "hotmart-access-token-value" not in serialized
    assert "client-secret-value" not in serialized
    config = db_session.exec(select(HotmartIntegrationConfigRecord)).one()
    assert config.status == "connected"
    assert config.last_health_status == "connected"


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch):
    with build_test_client(monkeypatch) as test_client:
        yield test_client


def _auth_headers(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_hotmart_admin_status_and_credentials_routes_are_admin_only_and_redacted(client: TestClient) -> None:
    headers = _auth_headers(client)

    initial_status = client.get("/api/v1/admin/integrations/hotmart/status", headers=headers)
    assert initial_status.status_code == 200
    assert initial_status.json()["status"] == "not_configured"

    credentials_response = client.post(
        "/api/v1/admin/integrations/hotmart/credentials",
        headers=headers,
        json={
            "environment": "sandbox",
            "enabled": True,
            "client_id": "client-id-value",
            "client_secret": "client-secret-value",
            "basic_token": "basic-token-value",
            "hottok": "hottok-value",
        },
    )

    assert credentials_response.status_code == 200
    payload = credentials_response.json()
    assert payload["status"] == "configured"
    assert payload["client_id_configured"] is True
    serialized = str(payload)
    assert "client-id-value" not in serialized
    assert "client-secret-value" not in serialized
    assert "basic-token-value" not in serialized
    assert "hottok-value" not in serialized


def test_hotmart_admin_credentials_use_platform_scope_when_active_workspace_changes(client: TestClient) -> None:
    headers = _auth_headers(client)
    admin_user = client.get("/api/v1/auth/me", headers=headers).json()
    admin_workspace_id = admin_user["active_workspace_id"]

    customer_response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "hotmart-customer-scope@example.com",
            "password": "StrongPass123!",
            "confirm_password": "StrongPass123!",
            "full_name": "Hotmart Customer Scope",
            "accept_terms": True,
            "accept_privacy": True,
            "accept_data_treatment": True,
        },
    )
    assert customer_response.status_code == 200
    customer_workspace_id = customer_response.json()["user"]["active_workspace_id"]
    assert customer_workspace_id != admin_workspace_id

    credentials_response = client.post(
        "/api/v1/admin/integrations/hotmart/credentials",
        headers={**headers, "x-workspace-id": customer_workspace_id},
        json={
            "environment": "sandbox",
            "enabled": True,
            "client_id": "global-client-id",
            "client_secret": "global-client-secret",
            "basic_token": "global-basic-token",
            "hottok": "global-hottok",
        },
    )

    assert credentials_response.status_code == 200
    payload = credentials_response.json()
    assert payload["workspace_id"] == admin_workspace_id
    assert payload["status"] == "configured"


def test_hotmart_commercial_admin_routes_expose_quota_and_effective_config_for_platform_admin(client: TestClient) -> None:
    headers = _auth_headers(client)

    list_response = client.get("/api/v1/admin/integrations/hotmart/commercial/quota-products", headers=headers)
    assert list_response.status_code == 200
    payload = list_response.json()
    assert {item["product_key"] for item in payload} >= {"blueprint_pro", "acp"}

    update_response = client.post(
        "/api/v1/admin/integrations/hotmart/commercial/quota-products",
        headers=headers,
        json={
          "product_key": "blueprint_pro",
          "display_name": "Blueprint Pro",
          "enabled": True,
          "initial_free_units": 2,
          "consumption_priority": ["free", "subscription", "one_time"],
          "checkout_required_on_zero_balance": True,
          "fifo_auto_approval_enabled": True,
          "default_blocked_request_ttl_hours": 72,
          "default_checkout_ttl_minutes": 30,
          "debt_enabled": True,
          "allow_manual_override_without_charge": True,
          "allow_courtesy": True,
          "allow_debt_pending": True,
          "catalog_priority_strategy": "minimum_sufficient",
          "sync_retry_limit": 5,
          "duplicate_conflict_visibility": "platform_admin_only",
          "metadata": {},
        },
    )
    assert update_response.status_code == 200
    assert update_response.json()["initial_free_units"] == 2

    effective_response = client.get(
        "/api/v1/admin/integrations/hotmart/commercial/effective-config?product_key=blueprint_pro",
        headers=headers,
    )
    assert effective_response.status_code == 200
    assert effective_response.json()["initial_free_units"] == 2

    bootstrap_response = client.get(
        "/api/v1/admin/integrations/hotmart/commercial/bootstrap?product_key=blueprint_pro",
        headers=headers,
    )
    assert bootstrap_response.status_code == 200
    bootstrap_payload = bootstrap_response.json()
    assert bootstrap_payload["product_key"] == "blueprint_pro"
    assert bootstrap_payload["effective_config"]["initial_free_units"] == 2
    assert bootstrap_payload["balance_snapshot"]["product_key"] == "blueprint_pro"
    assert bootstrap_payload["open_debt_count"] == 0

    legacy_response = client.get(
        "/api/v1/admin/integrations/hotmart/commercial/legacy-package-resolutions?status=pending&product_key=blueprint_pro",
        headers=headers,
    )
    assert legacy_response.status_code == 200
    assert legacy_response.json() == []
