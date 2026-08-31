from __future__ import annotations

from uuid import UUID

from app.db import get_session
from app.main import app
from app.models import ACPBuildRunRecord, ExportJobRecord, JourneyStateRecord
from app.services import export_delivery_service
from fastapi.testclient import TestClient
import pytest
from sqlmodel import select

from tests.api_testkit import TEST_EMAIL, TEST_PASSWORD, build_test_client


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


def _buy_product(client: TestClient, headers: dict[str, str], session_id: str, product_key: str) -> None:
    checkout_response = client.post(
        "/api/v1/commerce/checkout-sessions",
        headers=headers,
        json={
            "idempotency_key": f"{session_id}:{product_key}:sp9-sp16",
            "product_key": product_key,
            "session_id": session_id,
        },
    )
    assert checkout_response.status_code == 200
    checkout = checkout_response.json()
    payment_response = client.post(
        f"/api/v1/commerce/checkout-sessions/{checkout['checkout_ref']}/sandbox-complete",
        headers=headers,
        json={"outcome": "success", "provider_payment_id": f"sandbox_{checkout['checkout_ref']}"},
    )
    assert payment_response.status_code == 200


def test_sp9_sp16_productization_surfaces_are_gated_and_operational(client: TestClient) -> None:
    headers = _auth_headers(client)
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    blocked_workspace = client.get(f"/api/v1/sessions/{session_id}/acp/workspace", headers=headers)
    assert blocked_workspace.status_code == 403

    attention_response = client.get(f"/api/v1/sessions/{session_id}/attention", headers=headers)
    assert attention_response.status_code == 200
    assert attention_response.json()["contract_version"] == "attention.v1"

    free_catalog = client.get(f"/api/v1/sessions/{session_id}/exports/catalog", headers=headers)
    assert free_catalog.status_code == 200
    assert any(item["access_state"] == "locked" for item in free_catalog.json()["items"])

    blocked_export = client.post(
        f"/api/v1/sessions/{session_id}/exports/jobs",
        headers=headers,
        json={"artifact_kind": "construction_pack", "idempotency_key": f"{session_id}:blocked-export"},
    )
    assert blocked_export.status_code == 403

    _buy_product(client, headers, session_id, "acp")

    session_gen = app.dependency_overrides[get_session]()
    db = next(session_gen)
    try:
        before_workspace_state = db.exec(
            select(JourneyStateRecord).where(JourneyStateRecord.session_id == UUID(session_id))
        ).one()
        before_workspace_revision = before_workspace_state.revision
        assert db.exec(
            select(ACPBuildRunRecord).where(ACPBuildRunRecord.session_id == UUID(session_id))
        ).all() == []
    finally:
        session_gen.close()

    workspace_response = client.get(f"/api/v1/sessions/{session_id}/acp/workspace", headers=headers)
    assert workspace_response.status_code == 200
    workspace = workspace_response.json()
    assert workspace["contract_version"] == "acp-workspace.v1"
    assert workspace["access"]["tier"] == "acp"
    assert workspace["run"]["id"] is None
    assert workspace["journey_state_machine"]["state_source"] == "canonical"
    assert workspace["journey_state_machine"]["current"]["state_key"] == "acp_prep"
    assert len(workspace["phases"]) == 6

    session_gen = app.dependency_overrides[get_session]()
    db = next(session_gen)
    try:
        assert db.exec(
            select(ACPBuildRunRecord).where(ACPBuildRunRecord.session_id == UUID(session_id))
        ).all() == []
        after_workspace_state = db.exec(
            select(JourneyStateRecord).where(JourneyStateRecord.session_id == UUID(session_id))
        ).one()
        assert after_workspace_state.revision == before_workspace_revision
    finally:
        session_gen.close()

    phase_response = client.post(
        f"/api/v1/sessions/{session_id}/acp/workspace/phases/blueprint_validation/run",
        headers=headers,
        json={"idempotency_key": f"{session_id}:blueprint_validation:test"},
    )
    assert phase_response.status_code == 200
    phase_payload = phase_response.json()
    first_phase = next(item for item in phase_payload["phases"] if item["phase_key"] == "blueprint_validation")
    assert first_phase["attempt_count"] == 1
    assert first_phase["input_refs"]
    assert first_phase["status"] in {"completed", "completed_with_observations", "blocked"}
    assert phase_payload["run"]["id"] is not None
    assert phase_payload["journey_state_machine"]["current"]["state_key"] == "validate"
    expected_substate = "completed" if first_phase["status"] == "completed_with_observations" else first_phase["status"]
    assert phase_payload["journey_state_machine"]["current"]["substate"] == expected_substate

    launcher_response = client.get(f"/api/v1/sessions/{session_id}/acp/launcher", headers=headers)
    assert launcher_response.status_code == 200
    launcher = launcher_response.json()
    assert launcher["contract_version"] == "acp-launcher-metadata.v1"
    assert launcher["requires_lean_backend"] is False
    assert any(item["platform"] == "windows_cmd" for item in launcher["scripts"])

    acp_catalog_response = client.get(f"/api/v1/sessions/{session_id}/exports/catalog", headers=headers)
    assert acp_catalog_response.status_code == 200
    assert any(item["key"] == "construction_pack" and item["access_state"] == "allowed" for item in acp_catalog_response.json()["items"])

    export_response = client.post(
        f"/api/v1/sessions/{session_id}/exports/jobs",
        headers=headers,
        json={"artifact_kind": "construction_pack", "idempotency_key": f"{session_id}:construction-pack:test"},
    )
    assert export_response.status_code == 200
    export_job = export_response.json()
    assert export_job["status"] == "ready"
    assert export_job["checksum_sha256"]

    download_response = client.get(
        f"/api/v1/sessions/{session_id}/exports/jobs/{export_job['id']}/download",
        headers=headers,
    )
    assert download_response.status_code == 200
    assert download_response.headers["x-export-checksum-sha256"] == export_job["checksum_sha256"]
    assert download_response.headers["cache-control"] == "private, no-store"
    assert len(download_response.content) == export_job["size_bytes"]

    cancel_ready_response = client.post(
        f"/api/v1/sessions/{session_id}/exports/jobs/{export_job['id']}/cancel",
        headers=headers,
    )
    assert cancel_ready_response.status_code == 409

    acp_zip_response = client.post(
        f"/api/v1/sessions/{session_id}/exports/jobs",
        headers=headers,
        json={"artifact_kind": "acp_portable_zip", "idempotency_key": f"{session_id}:acp-zip-not-ready:test"},
    )
    assert acp_zip_response.status_code == 200
    acp_zip_job = acp_zip_response.json()
    assert acp_zip_job["status"] == "failed"
    assert "conformance" in acp_zip_job["error_message"].lower()

    retry_response = client.post(
        f"/api/v1/sessions/{session_id}/exports/jobs/{acp_zip_job['id']}/retry",
        headers=headers,
    )
    assert retry_response.status_code == 200
    retry_job = retry_response.json()
    assert retry_job["status"] == "failed"
    assert retry_job["metadata"]["retry_count"] == 1

    cancel_failed_response = client.post(
        f"/api/v1/sessions/{session_id}/exports/jobs/{acp_zip_job['id']}/cancel",
        headers=headers,
    )
    assert cancel_failed_response.status_code == 200
    assert cancel_failed_response.json()["status"] == "canceled"

    launcher_report_response = client.post(
        f"/api/v1/sessions/{session_id}/acp/launcher/report",
        headers=headers,
        json={
            "launcher_version": launcher["launcher_version"] or "test-launcher",
            "detected_tool": "codex-cli",
            "detected_ide": "cursor",
            "status": "dry_run_completed",
            "summary": "Launcher dry-run completed outside Lean.",
            "report": {"tools": ["codex-cli"], "ide": "cursor", "mode": "dry_run"},
        },
    )
    assert launcher_report_response.status_code == 200
    assert launcher_report_response.json()["contract_version"] == "acp-launch-report.v1"

    activity_response = client.get(f"/api/v1/sessions/{session_id}/activity", headers=headers)
    assert activity_response.status_code == 200
    activity_payload = activity_response.json()
    timeline_titles = [item["title"] for item in activity_payload["timeline"]]
    assert "export_job_ready" in timeline_titles
    assert "export_job_retry_requested" in timeline_titles
    assert "export_job_canceled" in timeline_titles
    assert "acp_launcher_report_received" in timeline_titles
    export_entry = next(
        item
        for item in activity_payload["timeline"]
        if item["type"] == "export" and item["metadata"]["export_job_id"] == export_job["id"]
    )
    assert export_entry["metadata"]["export_job_id"] == export_job["id"]
    assert export_entry["detail"]
    assert export_entry["title"] in {"export_job_ready", "export_job_regenerated"}

    plan_response = client.get(f"/api/v1/sessions/{session_id}/plan-access", headers=headers)
    assert plan_response.status_code == 200
    assert plan_response.json()["access"]["tier"] == "acp"
    assert len(plan_response.json()["products"]) >= 3

    diagrams_response = client.get(f"/api/v1/sessions/{session_id}/diagrams/catalog-v2?limit=5", headers=headers)
    assert diagrams_response.status_code == 200
    diagrams = diagrams_response.json()
    assert diagrams["contract_version"] == "diagram-catalog.v2"
    assert len(diagrams["entries"]) <= 5
    assert diagrams["total_count"] == 24


def test_sp9_sp16_download_route_recovers_missing_ready_payload(client: TestClient) -> None:
    headers = _auth_headers(client)
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    _buy_product(client, headers, session_id, "blueprint_pro")

    export_response = client.post(
        f"/api/v1/sessions/{session_id}/exports/jobs",
        headers=headers,
        json={"artifact_kind": "blueprint_professional", "idempotency_key": f"{session_id}:blueprint-zip:missing-payload"},
    )
    assert export_response.status_code == 200
    export_job = export_response.json()
    assert export_job["status"] == "ready"

    session_gen = app.dependency_overrides[get_session]()
    db = next(session_gen)
    try:
        job_record = db.get(ExportJobRecord, UUID(export_job["id"]))
        assert job_record is not None
        export_delivery_service._storage_path(job_record).unlink()
    finally:
        session_gen.close()

    download_response = client.get(
        f"/api/v1/sessions/{session_id}/exports/jobs/{export_job['id']}/download",
        headers=headers,
    )
    assert download_response.status_code == 200
    assert download_response.headers["content-disposition"].endswith(".zip\"")
    assert download_response.headers["cache-control"] == "private, no-store"
    assert download_response.content[:2] == b"PK"

    session_gen = app.dependency_overrides[get_session]()
    db = next(session_gen)
    try:
        job_record = db.get(ExportJobRecord, UUID(export_job["id"]))
        assert job_record is not None
        assert job_record.status.value == "ready"
        assert int(job_record.metadata_payload.get("retry_count", 0) or 0) == 1
        assert export_delivery_service._storage_path(job_record).exists()
    finally:
        session_gen.close()
