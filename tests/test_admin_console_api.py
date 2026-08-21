from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.db import get_session
from app.models import (
    ArtifactStatus,
    LLMUsageLedgerRecord,
    RuntimeSettingsAuditRecord,
    SessionRecord,
    SessionStage,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceRole,
)
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


def active_workspace_id(client: TestClient, headers: dict[str, str]) -> UUID:
    response = client.get("/api/v1/auth/me", headers=headers)
    assert response.status_code == 200
    return UUID(response.json()["active_workspace_id"])


@contextmanager
def db_session_from_client(client: TestClient) -> Generator[Session, None, None]:
    session_override = client.app.dependency_overrides[get_session]
    session_generator = session_override()
    session = next(session_generator)
    try:
        yield session
    finally:
        session.close()
        session_generator.close()


def seed_admin_console_records(client: TestClient, workspace_id: UUID) -> UUID:
    with db_session_from_client(client) as session:
        admin = session.exec(select(UserRecord).where(UserRecord.email == TEST_EMAIL)).first()
        assert admin is not None
        project = SessionRecord(
            user_id=admin.id,
            workspace_id=workspace_id,
            title="Proyecto Alpha",
            current_stage=SessionStage.draft_capture,
            status=ArtifactStatus.draft,
            created_at=datetime(2026, 8, 10, 9, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2026, 8, 10, 9, 0, 0, tzinfo=timezone.utc),
        )
        finished = SessionRecord(
            user_id=admin.id,
            workspace_id=workspace_id,
            title="Proyecto listo",
            current_stage=SessionStage.ready_for_export,
            status=ArtifactStatus.ready,
            created_at=datetime(2026, 8, 11, 9, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2026, 8, 12, 9, 0, 0, tzinfo=timezone.utc),
        )
        out_of_scope = SessionRecord(
            user_id=admin.id,
            workspace_id=uuid4(),
            title="Proyecto externo",
            current_stage=SessionStage.build_blueprint,
            status=ArtifactStatus.ready,
            created_at=datetime(2026, 8, 12, 9, 0, 0, tzinfo=timezone.utc),
        )
        session.add(project)
        session.add(finished)
        session.add(out_of_scope)
        session.flush()
        session.add(
            LLMUsageLedgerRecord(
                workspace_id=workspace_id,
                user_id=admin.id,
                session_id=project.id,
                project_id=project.id,
                stage="define",
                agent_key="builder",
                capability_key="define_requirements",
                provider_key="openai",
                model_name="gpt-5.5",
                status="succeeded",
                started_at=datetime(2026, 8, 13, 10, 0, 0, tzinfo=timezone.utc),
                input_tokens=700,
                output_tokens=300,
                total_tokens=1000,
                cost_total=4.5,
                currency="USD",
            )
        )
        session.add(
            LLMUsageLedgerRecord(
                workspace_id=uuid4(),
                user_id=admin.id,
                session_id=uuid4(),
                stage="define",
                provider_key="deepseek",
                model_name="deepseek-v4-pro",
                status="succeeded",
                started_at=datetime(2026, 8, 13, 10, 0, 0, tzinfo=timezone.utc),
                total_tokens=9000,
                cost_total=99.0,
            )
        )
        session.commit()
        return project.id


def seed_workspace_user(client: TestClient, workspace_id: UUID) -> UUID:
    with db_session_from_client(client) as session:
        user = UserRecord(
            email="viewer@leanbuilder.local",
            full_name="Viewer Test",
            password_hash=hash_password("Viewer123!"),
            email_verified=True,
        )
        session.add(user)
        session.flush()
        session.add(
            WorkspaceMembershipRecord(
                workspace_id=workspace_id,
                user_id=user.id,
                role=WorkspaceRole.viewer,
                is_active=True,
            )
        )
        session.commit()
        return user.id


def test_admin_overview_uses_real_scoped_sources_and_exposes_uninstrumented_states(client: TestClient) -> None:
    headers = auth_headers(client)
    workspace_id = active_workspace_id(client, headers)
    seed_admin_console_records(client, workspace_id)

    response = client.get(
        "/api/v1/admin/overview"
        "?started_from=2026-08-01T00:00:00Z"
        "&started_to=2026-08-31T23:59:59Z",
        headers=headers,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["workspace"]["id"] == str(workspace_id)
    assert payload["llm"]["summary"]["call_count"] == 1
    assert payload["llm"]["summary"]["cost_total"] == 4.5
    assert payload["projects"]["total"] == 2
    assert payload["projects"]["finalized"] == 1
    assert payload["users"]["total"] >= 1
    assert payload["availability"]["connected_users"]["status"] == "not_instrumented"
    assert payload["availability"]["project_finalized_at"]["status"] == "not_instrumented"


def test_admin_projects_analytics_returns_global_distribution_without_page_bias(client: TestClient) -> None:
    headers = auth_headers(client)
    workspace_id = active_workspace_id(client, headers)
    seed_admin_console_records(client, workspace_id)

    response = client.get(
        "/api/v1/admin/projects/analytics"
        "?started_from=2026-08-01T00:00:00Z"
        "&started_to=2026-08-31T23:59:59Z",
        headers=headers,
    )

    assert response.status_code == 200
    payload = response.json()
    stages = {item["stage"]: item["count"] for item in payload["distribution_by_stage"]}
    assert payload["total"] == 2
    assert stages["draft_capture"] == 1
    assert stages["ready_for_export"] == 1
    assert payload["created_series"]["availability"]["status"] == "available"
    assert payload["finalized_series"]["availability"]["status"] == "not_instrumented"


def test_admin_users_roles_invitations_and_patch_are_guarded_and_audited(client: TestClient) -> None:
    headers = auth_headers(client)
    workspace_id = active_workspace_id(client, headers)
    target_user_id = seed_workspace_user(client, workspace_id)

    users_response = client.get("/api/v1/admin/users?role=viewer", headers=headers)
    roles_response = client.get("/api/v1/admin/roles", headers=headers)
    invite_response = client.post(
        "/api/v1/admin/users/invitations",
        headers=headers,
        json={
            "email": "new.admin@example.com",
            "full_name": "New Admin",
            "role": "admin",
            "message": "Bienvenido al workspace.",
        },
    )
    patch_response = client.patch(
        f"/api/v1/admin/users/{target_user_id}",
        headers=headers,
        json={"membership_role": "editor"},
    )

    assert users_response.status_code == 200
    assert users_response.json()["items"][0]["email"] == "viewer@leanbuilder.local"
    assert roles_response.status_code == 200
    assert {item["key"] for item in roles_response.json()["workspace_roles"]} >= {"owner", "admin", "editor", "viewer"}
    assert invite_response.status_code == 201
    assert invite_response.json()["delivery_status"] == "manual_delivery_required"
    assert patch_response.status_code == 200
    assert patch_response.json()["membership"]["role"] == "editor"

    with db_session_from_client(client) as session:
        audit = session.exec(
            select(RuntimeSettingsAuditRecord).where(RuntimeSettingsAuditRecord.change_type == "admin_user_update")
        ).first()
        assert audit is not None
        assert audit.scope_id == str(workspace_id)

    invitations_response = client.get("/api/v1/admin/users/invitations", headers=headers)
    assert invitations_response.status_code == 200
    assert invitations_response.json()["count"] == 1
