from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlmodel import select

from app.db import get_session
from app.models import LLMUsageLedgerRecord, UserRecord
from app.services.auth_service import hash_password
from tests.api_testkit import TEST_EMAIL, TEST_PASSWORD, build_test_client


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    with build_test_client(monkeypatch) as test_client:
        yield test_client


def auth_headers(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def auth_headers_for(client: TestClient, *, email: str, password: str) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def active_workspace_id(client: TestClient, headers: dict[str, str]) -> str:
    response = client.get("/api/v1/auth/me", headers=headers)
    assert response.status_code == 200
    return response.json()["active_workspace_id"]


def seed_usage_record(
    client: TestClient,
    *,
    workspace_id: UUID,
    provider_key: str,
    model_name: str,
    stage: str,
    cost_total: float,
    total_tokens: int,
    status: str = "succeeded",
    started_at: datetime | None = None,
) -> None:
    session_override = client.app.dependency_overrides[get_session]
    session_generator = session_override()
    session = next(session_generator)
    try:
        user = session.exec(select(UserRecord).where(UserRecord.email == TEST_EMAIL)).first()
        assert user is not None
        session.add(
            LLMUsageLedgerRecord(
                workspace_id=workspace_id,
                user_id=user.id,
                session_id=uuid4(),
                stage=stage,
                agent_key="builder",
                capability_key="define_requirements",
                provider_key=provider_key,
                model_name=model_name,
                status=status,
                started_at=started_at or datetime(2026, 8, 13, 10, 0, 0, tzinfo=timezone.utc),
                duration_ms=100,
                input_tokens=total_tokens // 2,
                output_tokens=total_tokens - (total_tokens // 2),
                total_tokens=total_tokens,
                cost_total=cost_total,
            )
        )
        session.commit()
    finally:
        session.close()
        session_generator.close()


def seed_non_platform_user(client: TestClient, *, email: str, password: str) -> None:
    session_override = client.app.dependency_overrides[get_session]
    session_generator = session_override()
    session = next(session_generator)
    try:
        session.add(
            UserRecord(
                email=email,
                full_name="FinOps Viewer",
                password_hash=hash_password(password),
                email_verified=True,
            )
        )
        session.commit()
    finally:
        session.close()
        session_generator.close()


def test_llm_finops_summary_empty_state(client: TestClient) -> None:
    headers = auth_headers(client)

    response = client.get("/api/v1/finops/llm/summary", headers=headers)

    assert response.status_code == 200
    assert response.json()["call_count"] == 0
    assert response.json()["cost_total"] == 0


def test_llm_finops_routes_require_platform_admin(client: TestClient) -> None:
    seed_non_platform_user(
        client,
        email="finops-viewer@leanbuilder.local",
        password="FinopsViewer123!",
    )
    headers = auth_headers_for(
        client,
        email="finops-viewer@leanbuilder.local",
        password="FinopsViewer123!",
    )

    response = client.get("/api/v1/finops/llm/summary", headers=headers)

    assert response.status_code == 403
    assert response.json()["detail"] == "Solo un platform admin puede ejecutar esta accion."


def test_llm_finops_summary_is_global_for_platform_admin(client: TestClient) -> None:
    headers = auth_headers(client)
    workspace_id = UUID(active_workspace_id(client, headers))
    seed_usage_record(
        client,
        workspace_id=workspace_id,
        provider_key="openai",
        model_name="gpt-5.5",
        stage="define",
        cost_total=2.0,
        total_tokens=1000,
    )
    seed_usage_record(
        client,
        workspace_id=uuid4(),
        provider_key="deepseek",
        model_name="deepseek-v4-pro",
        stage="design",
        cost_total=99.0,
        total_tokens=9000,
    )

    response = client.get("/api/v1/finops/llm/summary", headers=headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["call_count"] == 2
    assert payload["cost_total"] == 101.0
    assert payload["total_tokens"] == 10000


def test_llm_finops_usage_filters_and_breakdowns(client: TestClient) -> None:
    headers = auth_headers(client)
    workspace_id = UUID(active_workspace_id(client, headers))
    seed_usage_record(
        client,
        workspace_id=workspace_id,
        provider_key="openai",
        model_name="gpt-5.5",
        stage="define",
        cost_total=3.0,
        total_tokens=1500,
    )
    seed_usage_record(
        client,
        workspace_id=workspace_id,
        provider_key="deepseek",
        model_name="deepseek-v4-pro",
        stage="design",
        cost_total=1.0,
        total_tokens=500,
        status="failed",
    )

    usage_response = client.get("/api/v1/finops/llm/usage?provider_key=openai", headers=headers)
    top_response = client.get("/api/v1/finops/llm/top-consumers?dimension=provider_key", headers=headers)
    provider_response = client.get("/api/v1/finops/llm/provider-breakdown", headers=headers)

    assert usage_response.status_code == 200
    assert usage_response.json()["count"] == 1
    assert usage_response.json()["items"][0]["provider_key"] == "openai"
    assert top_response.status_code == 200
    assert top_response.json()["items"][0]["key"] == "openai"
    assert provider_response.status_code == 200
    assert provider_response.json()["count"] == 2
    assert provider_response.json()["items"][0]["provider_key"] == "openai"


def test_llm_finops_timeseries_is_global_and_bucketed(client: TestClient) -> None:
    headers = auth_headers(client)
    workspace_id = UUID(active_workspace_id(client, headers))
    seed_usage_record(
        client,
        workspace_id=workspace_id,
        provider_key="openai",
        model_name="gpt-5.5",
        stage="define",
        cost_total=3.0,
        total_tokens=1500,
        started_at=datetime(2026, 8, 13, 10, 0, 0, tzinfo=timezone.utc),
    )
    seed_usage_record(
        client,
        workspace_id=workspace_id,
        provider_key="openai",
        model_name="gpt-5.5",
        stage="define",
        cost_total=2.0,
        total_tokens=1000,
        status="failed",
        started_at=datetime(2026, 8, 14, 10, 0, 0, tzinfo=timezone.utc),
    )
    seed_usage_record(
        client,
        workspace_id=uuid4(),
        provider_key="deepseek",
        model_name="deepseek-v4-pro",
        stage="design",
        cost_total=99.0,
        total_tokens=9000,
        started_at=datetime(2026, 8, 14, 10, 0, 0, tzinfo=timezone.utc),
    )

    response = client.get("/api/v1/finops/llm/timeseries?granularity=day", headers=headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert payload["items"][0]["cost_total"] == 3.0
    assert payload["items"][1]["cost_total"] == 101.0
    assert payload["items"][1]["error_count"] == 1
