from __future__ import annotations

from collections.abc import Generator
from datetime import datetime
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlmodel import select

from app.db import get_session
from app.models import (
    LLMBudgetPeriodType,
    LLMBudgetPolicyRecord,
    LLMBudgetScopeType,
    LLMUsageLedgerRecord,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceRole,
)
from app.services.auth_service import hash_password
from app.services.llm_finops.budget_service import LLMBudgetService
from tests.api_testkit import TEST_EMAIL, TEST_PASSWORD, build_test_client


VIEWER_EMAIL = "viewer@leanbuilder.local"
VIEWER_PASSWORD = "Viewer123!"


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    with build_test_client(monkeypatch) as test_client:
        yield test_client


def auth_headers(client: TestClient, *, email: str = TEST_EMAIL, password: str = TEST_PASSWORD) -> dict[str, str]:
    response = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def active_workspace_id(client: TestClient, headers: dict[str, str]) -> UUID:
    response = client.get("/api/v1/auth/me", headers=headers)
    assert response.status_code == 200
    return UUID(response.json()["active_workspace_id"])


def session_from_client(client: TestClient):
    session_override = client.app.dependency_overrides[get_session]
    session_generator = session_override()
    return session_generator, next(session_generator)


def seed_usage_record(client: TestClient, *, workspace_id: UUID, cost_total: float) -> None:
    session_generator, session = session_from_client(client)
    try:
        user = session.exec(select(UserRecord).where(UserRecord.email == TEST_EMAIL)).first()
        assert user is not None
        session.add(
            LLMUsageLedgerRecord(
                workspace_id=workspace_id,
                user_id=user.id,
                session_id=uuid4(),
                project_id=uuid4(),
                initiative_id=uuid4(),
                stage="define",
                agent_key="builder",
                capability_key="define_requirements",
                provider_key="openai",
                model_name="gpt-5.5",
                request_id=f"req-{uuid4()}",
                status="succeeded",
                started_at=datetime(2026, 8, 13, 10, 0, 0),
                duration_ms=100,
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                cost_total=cost_total,
                currency="USD",
            )
        )
        session.commit()
    finally:
        session.close()
        session_generator.close()


def seed_viewer_user(client: TestClient, *, workspace_id: UUID) -> None:
    session_generator, session = session_from_client(client)
    try:
        viewer = UserRecord(
            email=VIEWER_EMAIL,
            full_name="Lean Builder Viewer",
            password_hash=hash_password(VIEWER_PASSWORD),
            default_workspace_id=workspace_id,
        )
        session.add(viewer)
        session.flush()
        session.add(
            WorkspaceMembershipRecord(
                workspace_id=workspace_id,
                user_id=viewer.id,
                role=WorkspaceRole.viewer,
            )
        )
        session.commit()
    finally:
        session.close()
        session_generator.close()


def seed_other_workspace_budget(client: TestClient) -> None:
    session_generator, session = session_from_client(client)
    try:
        LLMBudgetService().create_policy(
            session,
            LLMBudgetPolicyRecord(
                workspace_id=uuid4(),
                policy_key="other-workspace-budget",
                scope_type=LLMBudgetScopeType.workspace,
                period_type=LLMBudgetPeriodType.monthly,
                limit_amount=999,
            ),
        )
    finally:
        session.close()
        session_generator.close()


def test_llm_finops_budget_crud_and_list_are_workspace_scoped(client: TestClient) -> None:
    headers = auth_headers(client)
    workspace_id = active_workspace_id(client, headers)
    seed_other_workspace_budget(client)

    create_response = client.post(
        "/api/v1/finops/llm/budgets",
        headers=headers,
        json={
            "policy_key": "workspace-monthly",
            "scope_type": "workspace",
            "period_type": "monthly",
            "limit_amount": 10,
            "threshold_percentages": [50, 80, 100],
        },
    )

    assert create_response.status_code == 201
    created = create_response.json()
    assert created["workspace_id"] == str(workspace_id)
    assert created["scope_value"] == str(workspace_id)
    assert created["evaluation"]["status"] == "ok"

    patch_response = client.patch(
        f"/api/v1/finops/llm/budgets/{created['id']}",
        headers=headers,
        json={"name": "Monthly LLM budget", "limit_amount": 20},
    )
    list_response = client.get("/api/v1/finops/llm/budgets", headers=headers)

    assert patch_response.status_code == 200
    assert patch_response.json()["name"] == "Monthly LLM budget"
    assert patch_response.json()["limit_amount"] == 20
    assert list_response.status_code == 200
    assert list_response.json()["count"] == 1
    assert list_response.json()["items"][0]["policy_key"] == "workspace-monthly"


def test_llm_finops_budget_writes_require_workspace_admin(client: TestClient) -> None:
    admin_headers = auth_headers(client)
    workspace_id = active_workspace_id(client, admin_headers)
    seed_viewer_user(client, workspace_id=workspace_id)
    viewer_headers = auth_headers(client, email=VIEWER_EMAIL, password=VIEWER_PASSWORD)
    viewer_headers["x-workspace-id"] = str(workspace_id)

    read_response = client.get("/api/v1/finops/llm/budgets", headers=viewer_headers)
    write_response = client.post(
        "/api/v1/finops/llm/budgets",
        headers=viewer_headers,
        json={"policy_key": "viewer-budget", "limit_amount": 10},
    )

    assert read_response.status_code == 200
    assert write_response.status_code == 403


def test_llm_finops_alert_endpoint_syncs_budget_alerts_once(client: TestClient) -> None:
    headers = auth_headers(client)
    workspace_id = active_workspace_id(client, headers)
    create_response = client.post(
        "/api/v1/finops/llm/budgets",
        headers=headers,
        json={
            "policy_key": "workspace-alert-budget",
            "scope_type": "workspace",
            "period_type": "monthly",
            "limit_amount": 10,
            "threshold_percentages": [50, 80, 100],
        },
    )
    assert create_response.status_code == 201
    seed_usage_record(client, workspace_id=workspace_id, cost_total=6)

    first = client.get("/api/v1/finops/llm/alerts?as_of=2026-08-13T12:00:00", headers=headers)
    second = client.get("/api/v1/finops/llm/alerts?as_of=2026-08-13T12:00:00", headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["count"] == 1
    assert second.json()["count"] == 1
    alert = first.json()["items"][0]
    assert alert["alert_type"] == "budget_threshold"
    assert alert["threshold_percent"] == 50
    assert alert["workspace_id"] == str(workspace_id)
    assert alert["scope_value"] == str(workspace_id)
