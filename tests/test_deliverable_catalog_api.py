from __future__ import annotations

from collections.abc import Generator
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlmodel import select

from app.db import get_session
from app.models import ArtifactRegistryRecord, SessionRecord, WorkspaceRole
from app.services.deliverable_catalog.persistence import DeliverableQualitySnapshotRecord
from app.services.product_processing import ProductBuildProductKey
from app.services.product_processing.persistence import ProductBuildRunRecord, ProductBuildStepRecord
from tests.api_testkit import TEST_EMAIL, TEST_PASSWORD, build_test_client
from tests.test_sessions_api import (
    assign_platform_role,
    auth_headers_for_credentials,
    create_workspace_for_user,
    seed_user,
)


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    with build_test_client(monkeypatch) as test_client:
        yield test_client


def _headers(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_deliverable_catalog_api_lists_entries_with_policy_decisions(client: TestClient) -> None:
    headers = _headers(client)

    response = client.get(
        "/api/v3/deliverables/catalog",
        params={"tier": "blueprint", "current_stage": "discover"},
        headers=headers,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["contract_version"] == "deliverable-catalog-response.v1"
    assert payload["entries"]
    keys = {item["key"] for item in payload["entries"]}
    assert "discovery.problem_context_brief" in keys
    assert "diagram.architecture_overview" in keys
    assert all("access" in item for item in payload["entries"])


def test_blueprint_commercial_result_generates_governed_artifacts(client: TestClient) -> None:
    headers = _headers(client)
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]
    session_uuid = UUID(session_id)

    response = client.post(f"/api/v1/sessions/{session_id}/blueprint/commercial-result", headers=headers)

    assert response.status_code == 200
    session_override = client.app.dependency_overrides[get_session]
    session_generator = session_override()
    db = next(session_generator)
    try:
        generated_artifacts = db.exec(
            select(ArtifactRegistryRecord).where(
                ArtifactRegistryRecord.session_id == session_uuid,
                ArtifactRegistryRecord.source_action == "deliverable_generation_agent",
            )
        ).all()
        product_run = db.exec(
            select(ProductBuildRunRecord).where(
                ProductBuildRunRecord.session_id == session_uuid,
                ProductBuildRunRecord.product_key == ProductBuildProductKey.blueprint_basic.value,
            )
        ).first()
        product_steps = (
            db.exec(select(ProductBuildStepRecord).where(ProductBuildStepRecord.run_id == product_run.id)).all()
            if product_run is not None
            else []
        )
    finally:
        session_generator.close()
    assert generated_artifacts
    assert product_run is not None
    assert product_steps
    assert any(
        artifact.artifact_metadata.get("deliverable_key") == "discovery.problem_context_brief"
        for artifact in generated_artifacts
    )


def test_deliverable_stage_and_detail_api_respect_tier_and_stage(client: TestClient) -> None:
    headers = _headers(client)

    stage_response = client.get(
        "/api/v3/deliverables/stage/discover",
        params={"tier": "blueprint", "current_stage": "discover"},
        headers=headers,
    )
    detail_response = client.get(
        "/api/v3/deliverables/diagram.architecture_overview",
        params={"tier": "acp", "current_stage": "discover"},
        headers=headers,
    )

    assert stage_response.status_code == 200
    assert {item["stage"] for item in stage_response.json()["entries"]} == {"discover"}
    assert detail_response.status_code == 200
    assert detail_response.json()["access"]["access_state"] == "stage_locked"


def test_deliverable_catalog_rollout_flags_gate_catalog_and_admin_surfaces(client: TestClient) -> None:
    headers = _headers(client)
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    disabled_catalog = client.patch(
        f"/api/v1/sessions/{session_id}/feature-flags/deliverable_catalog_enabled",
        headers=headers,
        json={"enabled": False},
    )
    assert disabled_catalog.status_code == 200

    blocked_catalog = client.get("/api/v3/deliverables/catalog", headers=headers)
    assert blocked_catalog.status_code == 409
    assert "Deliverable catalog feature flag is disabled" in blocked_catalog.json()["detail"]

    reenabled_catalog = client.patch(
        f"/api/v1/sessions/{session_id}/feature-flags/deliverable_catalog_enabled",
        headers=headers,
        json={"enabled": True},
    )
    assert reenabled_catalog.status_code == 200
    disabled_admin = client.patch(
        f"/api/v1/sessions/{session_id}/feature-flags/deliverable_governance_admin_enabled",
        headers=headers,
        json={"enabled": False},
    )
    assert disabled_admin.status_code == 200
    assign_platform_role(client, email=TEST_EMAIL)

    visible_catalog = client.get("/api/v3/deliverables/catalog", headers=headers)
    blocked_admin = client.get("/api/v3/admin/deliverable-governance/overview", headers=headers)

    assert visible_catalog.status_code == 200
    assert blocked_admin.status_code == 409
    assert "Deliverable governance admin feature flag is disabled" in blocked_admin.json()["detail"]


def test_deliverable_governance_admin_api_requires_platform_role_and_audits_changes(client: TestClient) -> None:
    viewer_email = "deliverable-viewer@leanbuilder.local"
    viewer_password = "LeanBuilderViewer123!"
    seed_user(client, email=viewer_email, password=viewer_password, full_name="Deliverable Viewer")
    viewer_headers = auth_headers_for_credentials(client, email=viewer_email, password=viewer_password)
    headers = _headers(client)

    forbidden = client.get("/api/v3/admin/deliverable-governance", headers=viewer_headers)
    assert forbidden.status_code == 403

    assign_platform_role(client, email=TEST_EMAIL)

    overview = client.get("/api/v3/admin/deliverable-governance/overview", headers=headers)
    assert overview.status_code == 200
    assert overview.json()["total_entries"] >= 44

    update = client.patch(
        "/api/v3/admin/deliverable-governance/discovery.problem_context_brief",
        headers=headers,
        json={
            "enabled": False,
            "generation_enabled": True,
            "required_tier_override": "",
            "preview_mode_override": "",
            "prompt_status": "active",
            "prompt_override": {},
            "notes": "BDG6 test pause",
        },
    )
    assert update.status_code == 200
    assert update.json()["enabled"] is False

    detail = client.get(
        "/api/v3/deliverables/discovery.problem_context_brief",
        params={"tier": "acp", "current_stage": "discover"},
        headers=headers,
    )
    assert detail.status_code == 200
    assert detail.json()["access"]["access_state"] == "disabled"

    updated_overview = client.get("/api/v3/admin/deliverable-governance/overview", headers=headers)
    assert updated_overview.status_code == 200
    assert updated_overview.json()["recent_audit"]


def test_deliverable_governance_overview_includes_quality_summary_filters(client: TestClient) -> None:
    headers = _headers(client)
    assign_platform_role(client, email=TEST_EMAIL)
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]
    session_uuid = UUID(session_id)

    session_override = client.app.dependency_overrides[get_session]
    session_generator = session_override()
    db = next(session_generator)
    try:
        session_record = db.exec(select(SessionRecord).where(SessionRecord.id == session_uuid)).one()
        workspace_id = str(session_record.workspace_id)
        db.add(
            DeliverableQualitySnapshotRecord(
                workspace_id=session_record.workspace_id,
                session_id=session_uuid,
                deliverable_key="discovery.problem_context_brief",
                version_ref="quality-test-v1",
                state="passed",
                score=96,
                warnings=["minor_copy_tone"],
            )
        )
        db.add(
            DeliverableQualitySnapshotRecord(
                workspace_id=session_record.workspace_id,
                session_id=session_uuid,
                deliverable_key="diagram.architecture_overview",
                version_ref="quality-test-v2",
                state="warning",
                score=82,
                warnings=["dense_layout"],
            )
        )
        db.commit()
    finally:
        session_generator.close()

    response = client.get(
        "/api/v3/admin/deliverable-governance/overview",
        params={
            "scope": "workspace",
            "product": "blueprint",
            "stage": "discover",
            "type": "artifact",
            "quality_state": "passed",
        },
        headers={**headers, "x-workspace-id": workspace_id},
    )

    assert response.status_code == 200
    quality_summary = response.json()["quality_summary"]
    assert quality_summary["total_snapshots"] == 1
    assert quality_summary["average_score"] == 96
    assert quality_summary["by_state"] == {"passed": 1}
    assert quality_summary["recent_snapshots"][0]["deliverable_key"] == "discovery.problem_context_brief"
    assert quality_summary["recent_snapshots"][0]["stage"] == "discover"
    assert quality_summary["recent_snapshots"][0]["warnings_count"] == 1


def test_deliverable_prompt_governance_versions_validates_and_scopes_overrides(client: TestClient) -> None:
    headers = _headers(client)
    assign_platform_role(client, email=TEST_EMAIL)
    workspace_b_id = create_workspace_for_user(
        client,
        email=TEST_EMAIL,
        name="Prompt Governance B",
        role=WorkspaceRole.admin,
    )
    workspace_b_headers = {**headers, "x-workspace-id": workspace_b_id}

    prompt = client.get(
        "/api/v3/admin/deliverable-governance/discovery.problem_context_brief/prompt",
        headers=headers,
    )
    assert prompt.status_code == 200
    assert prompt.json()["prompt_status"] == "active"

    invalid = client.post(
        "/api/v3/admin/deliverable-governance/discovery.problem_context_brief/prompt/validate",
        headers=headers,
        json={
            "prompt_body": "Genera un texto libre sin contrato.",
            "schema_contract": "wrong-schema.v1",
            "validator_key": "wrong-validator",
            "fallback_policy": "fallback",
        },
    )
    assert invalid.status_code == 200
    assert invalid.json()["valid"] is False
    assert "schema_contract_mismatch" in invalid.json()["errors"]

    updated = client.patch(
        "/api/v3/admin/deliverable-governance/discovery.problem_context_brief/prompt",
        params={"scope": "workspace"},
        headers=headers,
        json={
            "prompt_status": "paused",
            "prompt_body": (
                "Genera evidencia trazable para discovery.problem_context_brief "
                "usando el schema_contract deliverable-artifact.v1."
            ),
            "schema_contract": "deliverable-artifact.v1",
            "validator_key": "artifact.markdown_json.v1",
            "fallback_policy": "use_structured_stage_snapshot",
            "version": "bdg7-test",
            "change_reason": "Pause only current workspace",
            "metadata": {"source": "test"},
        },
    )
    assert updated.status_code == 200
    assert updated.json()["prompt_status"] == "paused"
    assert updated.json()["versions"][0]["version"] == "bdg7-test"

    scoped_detail = client.get(
        "/api/v3/deliverables/discovery.problem_context_brief",
        params={"tier": "acp", "current_stage": "discover"},
        headers=headers,
    )
    other_workspace_detail = client.get(
        "/api/v3/deliverables/discovery.problem_context_brief",
        params={"tier": "acp", "current_stage": "discover"},
        headers=workspace_b_headers,
    )

    assert scoped_detail.status_code == 200
    assert scoped_detail.json()["access"]["effective_prompt_status"] == "paused"
    assert scoped_detail.json()["access"]["can_generate"] is False
    assert other_workspace_detail.status_code == 200
    assert other_workspace_detail.json()["access"]["effective_prompt_status"] == "active"
