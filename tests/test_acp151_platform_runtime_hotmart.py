from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.db import get_session
from app.models import (
    CommercialOrderRecord,
    HotmartPaymentLinkRecord,
    LLMRuntimeSettingsUpdateRequest,
    RuntimePropagationMode,
    RuntimePropagationRequest,
    SessionRecord,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceRecord,
    WorkspaceRole,
)
from app.core.config import get_settings
from app.services.auth_service import hash_password
from app.services.llm_runtime.runtime_settings_service import (
    load_effective_runtime_settings,
    persist_workspace_runtime_settings,
)
from app.services.llm_runtime.runtime_secrets_service import annotate_runtime_settings_with_workspace_secrets
from app.services.runtime_propagation_service import propagate_platform_runtime_settings
from tests.api_testkit import TEST_EMAIL, TEST_PASSWORD, build_test_client


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    with build_test_client(monkeypatch) as test_client:
        yield test_client


def _auth_headers(client: TestClient, *, email: str = TEST_EMAIL, password: str = TEST_PASSWORD) -> dict[str, str]:
    response = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@contextmanager
def _db_session(client: TestClient) -> Generator[Session, None, None]:
    session_override = client.app.dependency_overrides[get_session]
    session_generator = session_override()
    session = next(session_generator)
    try:
        yield session
    finally:
        session.close()
        session_generator.close()


def _seed_external_workspace(session: Session) -> tuple[WorkspaceRecord, UserRecord]:
    user = UserRecord(
        email="global-viewer@leanbuilder.local",
        full_name="Global Viewer",
        password_hash=hash_password("GlobalViewer123!"),
        email_verified=True,
    )
    session.add(user)
    session.flush()
    workspace = WorkspaceRecord(
        name="Global ACP151 Workspace",
        slug=f"global-acp151-{str(user.id)[:8]}",
        created_by_user_id=user.id,
    )
    session.add(workspace)
    session.flush()
    session.add(
        WorkspaceMembershipRecord(
            workspace_id=workspace.id,
            user_id=user.id,
            role=WorkspaceRole.viewer,
            is_active=True,
        )
    )
    session.add(SessionRecord(user_id=user.id, workspace_id=workspace.id, title="Proyecto externo ACP151"))
    session.commit()
    session.refresh(workspace)
    session.refresh(user)
    return workspace, user


def _runtime_payload(provider: str = "deepseek") -> LLMRuntimeSettingsUpdateRequest:
    return LLMRuntimeSettingsUpdateRequest.model_validate(
        {
            "active_provider": provider,
            "agent_execution_backend": "provider_native",
            "knowledge_access_backend": "workspace_staged",
            "uses_platform_credentials": True,
            "openai": {
                "fast_model": "gpt-5.4-mini",
                "reasoning_model": "gpt-5.5",
                "reasoning_effort": "low",
            },
            "deepseek": {
                "base_url": "https://api.deepseek.com",
                "fast_model": "deepseek-v4-flash",
                "reasoning_model": "deepseek-v4-pro",
                "reasoning_effort": "high",
            },
            "codex_local": {
                "command": "codex",
                "model": "gpt-5.5",
                "profile": "",
                "cost_policy": "hybrid",
                "timeout_ms": 150000,
                "max_concurrency": 1,
                "runner_id": "local",
                "auth_mode": "auto",
                "fallback_models": [],
                "primary_agents": [],
                "shadow_agents": [],
                "staged_agents": [],
            },
        }
    )


def test_platform_admin_users_route_lists_users_across_workspaces(client: TestClient) -> None:
    headers = _auth_headers(client)
    with _db_session(client) as session:
        _, external_user = _seed_external_workspace(session)

    response = client.get("/api/v1/platform/admin/users", headers=headers)

    assert response.status_code == 200
    payload = response.json()
    emails = {item["email"] for item in payload["users"]}
    assert TEST_EMAIL in emails
    assert external_user.email in emails
    external_summary = next(item for item in payload["users"] if item["email"] == external_user.email)
    assert external_summary["workspace_count"] == 1
    assert external_summary["project_count"] == 1


def test_platform_admin_routes_forbid_non_platform_workspace_owner(client: TestClient) -> None:
    with _db_session(client) as session:
        workspace = session.exec(select(WorkspaceRecord)).first()
        assert workspace is not None
        owner = UserRecord(
            email="plain-owner@leanbuilder.local",
            full_name="Plain Owner",
            password_hash=hash_password("PlainOwner123!"),
            email_verified=True,
        )
        session.add(owner)
        session.flush()
        session.add(WorkspaceMembershipRecord(workspace_id=workspace.id, user_id=owner.id, role=WorkspaceRole.owner))
        session.commit()
    owner_headers = _auth_headers(client, email="plain-owner@leanbuilder.local", password="PlainOwner123!")

    response = client.get("/api/v1/platform/admin/users", headers=owner_headers)

    assert response.status_code == 403
    assert response.json()["detail"] == "Solo un platform admin puede ejecutar esta accion."


def test_runtime_propagation_dry_run_preserves_workspace_override(client: TestClient) -> None:
    with _db_session(client) as session:
        actor = session.exec(select(UserRecord).where(UserRecord.email == TEST_EMAIL)).one()
        workspace, _ = _seed_external_workspace(session)
        persist_workspace_runtime_settings(
            session,
            workspace.id,
            _runtime_payload("openai"),
            actor_user_id=actor.id,
        )
        response = propagate_platform_runtime_settings(
            session,
            payload=RuntimePropagationRequest(
                mode=RuntimePropagationMode.fallback_only,
                payload=_runtime_payload("deepseek"),
                dry_run=True,
            ),
            actor_user_id=actor.id,
        )
        runtime = load_effective_runtime_settings(session, workspace.id)

    assert response.dry_run is True
    assert any(item.workspace_id == workspace.id and item.action == "preserve_workspace_override" for item in response.items)
    assert runtime.active_provider.value == "openai"


def test_runtime_propagation_force_selected_updates_target_workspace(client: TestClient) -> None:
    with _db_session(client) as session:
        actor = session.exec(select(UserRecord).where(UserRecord.email == TEST_EMAIL)).one()
        workspace, _ = _seed_external_workspace(session)
        response = propagate_platform_runtime_settings(
            session,
            payload=RuntimePropagationRequest(
                mode=RuntimePropagationMode.force_selected,
                payload=_runtime_payload("deepseek"),
                workspace_ids=[workspace.id],
                dry_run=False,
            ),
            actor_user_id=actor.id,
        )
        runtime = load_effective_runtime_settings(session, workspace.id)

    assert response.status.value == "applied"
    assert response.items[0].action == "write_workspace_runtime_settings"
    assert runtime.active_provider.value == "deepseek"


def test_hotmart_readiness_exposes_provider_and_quota_checks(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "commerce_checkout_provider", "sandbox")
    monkeypatch.setattr(settings, "hotmart_enabled", False)
    monkeypatch.setattr(settings, "hotmart_environment", "sandbox")
    headers = _auth_headers(client)

    response = client.get("/api/v1/admin/integrations/hotmart/release-readiness?environment=production", headers=headers)

    assert response.status_code == 200
    payload = response.json()
    checks = {item["key"]: item for item in payload["checklist"]}
    alerts = {item["key"] for item in payload["alerts"]}
    assert checks["commerce_provider_hotmart"]["status"] == "failed"
    assert checks["quota_product_configs"]["status"] == "passed"
    assert "commerce_provider_not_hotmart" in alerts


def test_hotmart_payment_links_route_applies_admin_list_limit(client: TestClient) -> None:
    headers = _auth_headers(client)
    with _db_session(client) as session:
        workspace = session.exec(select(WorkspaceRecord)).first()
        actor = session.exec(select(UserRecord).where(UserRecord.email == TEST_EMAIL)).one()
        assert workspace is not None
        order = CommercialOrderRecord(
            workspace_id=workspace.id,
            buyer_user_id=actor.id,
            status="pending",
            currency="USD",
            subtotal_cents=10000,
            total_cents=10000,
            provider="hotmart",
            checkout_ref="limit-check-order",
            checkout_url="https://pay.hotmart.test/limit-check-order",
            idempotency_key="limit-check-order",
        )
        session.add(order)
        session.flush()
        for index in range(3):
            session.add(
                HotmartPaymentLinkRecord(
                    workspace_id=workspace.id,
                    order_id=order.id,
                    environment="sandbox",
                    internal_product_key="blueprint_pro",
                    hotmart_payment_link_id=f"limit-check-{index}",
                    checkout_url=f"https://pay.hotmart.test/limit-check-{index}",
                    activation_status="pending_activation",
                    provider_ref=f"limit-check-{index}",
                    gross_amount_cents=10000,
                    net_amount_cents=10000,
                    currency="USD",
                    internal_unit_amount_usd_cents=10000,
                    discount_origin="none",
                )
            )
        session.commit()

    response = client.get("/api/v1/admin/integrations/hotmart/payment-links?limit=2", headers=headers)

    assert response.status_code == 200
    assert len(response.json()) == 2


def test_platform_provider_secret_applies_to_inherited_workspaces(client: TestClient) -> None:
    headers = _auth_headers(client)
    with _db_session(client) as session:
        workspace = session.exec(select(WorkspaceRecord)).first()
        assert workspace is not None

    secret_response = client.post(
        "/api/v1/platform/runtime/secrets/openai",
        headers=headers,
        json={
            "activate_for_runtime": True,
            "secret_kind": "api_key",
            "secret_ref": "",
            "secret_value": "sk-test-platform-secret",
        },
    )

    assert secret_response.status_code == 200
    secret_payload = secret_response.json()
    assert secret_payload["configured"] is True
    assert secret_payload["secret_source"] == "platform_managed"
    assert secret_payload["storage_mode"] == "ciphertext"

    runtime_response = client.get("/api/v1/runtime/llm", headers=headers)

    assert runtime_response.status_code == 200
    runtime_payload = runtime_response.json()
    assert runtime_payload["uses_platform_credentials"] is True
    assert runtime_payload["openai"]["api_key_configured"] is True
    assert runtime_payload["openai"]["secret_source"] == "platform_managed"
    assert runtime_payload["openai"]["health_status"] == "platform_ready"


def test_platform_provider_secret_routes_forbid_non_platform_admin(client: TestClient) -> None:
    with _db_session(client) as session:
        workspace = session.exec(select(WorkspaceRecord)).first()
        assert workspace is not None
        owner = UserRecord(
            email="secret-owner@leanbuilder.local",
            full_name="Secret Owner",
            password_hash=hash_password("SecretOwner123!"),
            email_verified=True,
        )
        session.add(owner)
        session.flush()
        session.add(WorkspaceMembershipRecord(workspace_id=workspace.id, user_id=owner.id, role=WorkspaceRole.owner))
        session.commit()
    owner_headers = _auth_headers(client, email="secret-owner@leanbuilder.local", password="SecretOwner123!")

    response = client.get("/api/v1/platform/runtime/secrets/openai", headers=owner_headers)

    assert response.status_code == 403


def test_registered_customer_workspace_inherits_platform_llm_secret(client: TestClient) -> None:
    headers = _auth_headers(client)
    secret_response = client.post(
        "/api/v1/platform/runtime/secrets/openai",
        headers=headers,
        json={
            "activate_for_runtime": True,
            "secret_kind": "api_key",
            "secret_ref": "",
            "secret_value": "sk-test-platform-customer-secret",
        },
    )
    assert secret_response.status_code == 200

    register_response = client.post(
        "/api/v1/auth/register",
        json={
            "accept_data_treatment": True,
            "accept_privacy": True,
            "accept_terms": True,
            "confirm_password": "ClienteLab123!",
            "email": "cliente-nuevo-acp151@leanbuilder.local",
            "full_name": "Cliente Nuevo ACP151",
            "password": "ClienteLab123!",
            "workspace_name": "Cliente Nuevo Workspace ACP151",
        },
    )

    assert register_response.status_code == 200
    register_payload = register_response.json()
    customer_token = register_payload["access_token"]
    customer_user = register_payload["user"]
    assert customer_user["active_workspace_id"]
    assert customer_user["active_workspace_name"] == "Cliente Nuevo Workspace ACP151"
    assert customer_user["workspaces"][0]["role"] == "owner"

    admin_runtime_response = client.get(
        "/api/v1/runtime/llm",
        headers={"Authorization": f"Bearer {customer_token}"},
    )

    assert admin_runtime_response.status_code == 403

    create_session_response = client.post(
        "/api/v1/sessions",
        headers={"Authorization": f"Bearer {customer_token}"},
    )

    assert create_session_response.status_code == 201
    assert create_session_response.json()["workspace_id"] == customer_user["active_workspace_id"]

    with _db_session(client) as session:
        workspace_id = UUID(customer_user["active_workspace_id"])
        runtime_payload = annotate_runtime_settings_with_workspace_secrets(
            session,
            workspace_id,
            load_effective_runtime_settings(session, workspace_id),
        ).model_dump(mode="json")

    assert runtime_payload["uses_platform_credentials"] is True
    assert runtime_payload["active_provider"] == "openai"
    assert runtime_payload["openai"]["api_key_configured"] is True
    assert runtime_payload["openai"]["secret_source"] == "platform_managed"
    assert runtime_payload["openai"]["health_status"] == "platform_ready"
