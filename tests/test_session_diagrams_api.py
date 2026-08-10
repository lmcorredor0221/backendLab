from __future__ import annotations

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


def test_diagram_catalog_endpoint_lists_available_and_locked_diagrams(client: TestClient) -> None:
    headers = _auth_headers(client)
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    response = client.get(f"/api/v1/sessions/{session_id}/diagrams/catalog", headers=headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_count"] == 24
    assert len(payload["entries"]) == 24
    assert {item["diagram_key"] for item in payload["entries"]} >= {
        "architecture_overview",
        "tool_contract_sequence",
        "memory_rag_architecture",
        "rag_ingestion_pipeline",
        "commercial_value_flow",
    }
    assert any(item["generation_state"] == "not_generated" for item in payload["entries"])
    assert all("content" not in item for item in payload["entries"])


def test_diagram_content_endpoint_blocks_payload_until_policy_allows_access(client: TestClient) -> None:
    headers = _auth_headers(client)
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    response = client.get(
        f"/api/v1/sessions/{session_id}/diagrams/tool_contract_sequence?format=mermaid",
        headers=headers,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["diagram_key"] == "tool_contract_sequence"
    assert payload["content"] is None
    assert payload["access_state"] in {"stage_locked", "locked_acp", "not_generated"}
    assert payload["upsell"] is not None


def test_diagram_content_endpoint_returns_404_for_unknown_diagram(client: TestClient) -> None:
    headers = _auth_headers(client)
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    response = client.get(f"/api/v1/sessions/{session_id}/diagrams/missing-diagram", headers=headers)

    assert response.status_code == 404
    assert response.json()["detail"] == "Diagram not found"
