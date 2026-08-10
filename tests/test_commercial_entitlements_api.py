from fastapi.testclient import TestClient
import pytest

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


def test_acp_routes_are_backend_blocked_without_acp_entitlement(client: TestClient) -> None:
    headers = _auth_headers(client)
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    blocked_requests = [
        ("get", f"/api/v1/sessions/{session_id}/acp/preview"),
        ("post", f"/api/v1/sessions/{session_id}/acp/generate"),
        ("get", f"/api/v1/sessions/{session_id}/acp/construction-readiness"),
        ("get", f"/api/v1/sessions/{session_id}/acp/questions"),
        ("get", f"/api/v1/sessions/{session_id}/acp/knowledge-graph"),
        ("get", f"/api/v1/sessions/{session_id}/acp/gaps/deployment_target_unknown"),
        ("get", f"/api/v1/sessions/{session_id}/acp/files/ACP/manifest.yaml"),
        ("get", f"/api/v1/sessions/{session_id}/acp/export.zip"),
        ("get", f"/api/v1/sessions/{session_id}/export/construction-pack"),
        ("get", f"/api/v1/sessions/{session_id}/export/prompt-pack"),
        ("get", f"/api/v1/sessions/{session_id}/export/test-pack"),
        ("get", f"/api/v1/sessions/{session_id}/acp/workspace"),
        ("post", f"/api/v1/sessions/{session_id}/acp/workspace/resume"),
        ("post", f"/api/v1/sessions/{session_id}/acp/workspace/phases/blueprint_validation/run"),
        ("get", f"/api/v1/sessions/{session_id}/acp/launcher"),
    ]

    for method, url in blocked_requests:
        response = getattr(client, method)(url, headers=headers)
        assert response.status_code == 403
        assert "ACP Premium" in response.json()["detail"]

    blocked_launcher_report = client.post(
        f"/api/v1/sessions/{session_id}/acp/launcher/report",
        headers=headers,
        json={
            "launcher_version": "test",
            "status": "dry_run_completed",
            "summary": "should be blocked before ACP purchase",
        },
    )
    assert blocked_launcher_report.status_code == 403

    blocked_job = client.post(
        f"/api/v1/sessions/{session_id}/exports/jobs",
        headers=headers,
        json={"artifact_kind": "acp_portable_zip", "idempotency_key": f"{session_id}:free-acp-zip"},
    )
    assert blocked_job.status_code == 403


def test_acp_invitation_event_is_recorded_before_acp_purchase(client: TestClient) -> None:
    headers = _auth_headers(client)
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    event_response = client.post(
        f"/api/v1/sessions/{session_id}/acp/invitation-events",
        headers=headers,
        json={
            "event_key": "invitation_viewed",
            "metadata": {"stage": "validate"},
            "product": "acp",
            "source": "test",
        },
    )

    assert event_response.status_code == 200
    assert any(item["message"] == "Evento comercial ACP registrado" for item in event_response.json()["activity"])


def test_generic_commercial_event_is_recorded_before_purchase(client: TestClient) -> None:
    headers = _auth_headers(client)
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    event_response = client.post(
        f"/api/v1/sessions/{session_id}/commercial-events",
        headers=headers,
        json={
            "event_key": "diagram_protected_action_blocked",
            "metadata": {"action": "copy", "diagram_key": "architecture_overview"},
            "product": "blueprint",
            "source": "diagram_browser",
        },
    )

    assert event_response.status_code == 200
    assert any(item["message"] == "Evento comercial registrado" for item in event_response.json()["activity"])

    activity_response = client.get(f"/api/v1/sessions/{session_id}/activity", headers=headers)
    assert activity_response.status_code == 200
    event = next(item for item in activity_response.json()["timeline"] if item["title"] == "diagram_protected_action_blocked")
    assert event["metadata"]["event_schema_version"] == "commercial-event.v1"
    assert event["metadata"]["event_category"] == "content_protection"


def test_free_blueprint_result_does_not_leak_acp_portable_paths_or_prompts(client: TestClient) -> None:
    headers = _auth_headers(client)
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    result_response = client.get(f"/api/v1/sessions/{session_id}/blueprint/result", headers=headers)

    assert result_response.status_code == 200
    payload = result_response.json()
    serialized = str(payload)
    assert payload["access"]["tier"] == "blueprint"
    assert "ACP/" not in serialized
    assert "builder-handoff" not in serialized
    assert "launch-manifest" not in serialized
    assert all(not item["source_paths"] for item in payload["diagrams"] if item["access_state"] != "unlocked")


def test_commercial_audit_report_summarizes_events_and_redacts_sensitive_metadata(client: TestClient) -> None:
    headers = _auth_headers(client)
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    client.post(
        f"/api/v1/sessions/{session_id}/commercial-events",
        headers=headers,
        json={
            "event_key": "blueprint_results_viewed",
            "metadata": {
                "content": "<svg>secret diagram</svg>",
                "confidence_score": 90,
                "nested": {"api_key": "sk-test-secret"},
            },
            "product": "blueprint",
            "source": "test",
        },
    )
    client.post(
        f"/api/v1/sessions/{session_id}/commercial-events",
        headers=headers,
        json={
            "event_key": "diagram_protected_action_blocked",
            "metadata": {"action": "copy", "diagram_key": "architecture_overview"},
            "product": "blueprint",
            "source": "diagram_browser",
        },
    )
    checkout_response = client.post(
        "/api/v1/commerce/checkout-sessions",
        headers=headers,
        json={
            "idempotency_key": f"{session_id}:blueprint-pro:audit-test",
            "product_key": "blueprint_pro",
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

    report_response = client.get(f"/api/v1/sessions/{session_id}/commercial-audit", headers=headers)

    assert report_response.status_code == 200
    report = report_response.json()
    assert report["contract_version"] == "commercial-audit.v1"
    assert report["current_tier"] == "blueprint_pro"
    assert report["workspace_id"]
    assert report["requested_by_user_id"]
    assert any(item["key"] == "total_events" and item["value"] >= 3 for item in report["metrics"])
    assert any(item["key"] == "blocked_events" and item["value"] >= 1 for item in report["metrics"])
    assert any(item["product"] == "blueprint_pro" and item["purchases"] == 1 for item in report["product_summary"])
    serialized = str(report)
    assert "sk-test-secret" not in serialized
    assert "secret diagram" not in serialized
    assert "[redacted]" in serialized
