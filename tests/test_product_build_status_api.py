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


def test_product_build_status_api_lists_all_product_surfaces(client: TestClient) -> None:
    headers = _auth_headers(client)
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    response = client.get(f"/api/v1/sessions/{session_id}/product-builds", headers=headers)
    assert response.status_code == 200
    payload = response.json()

    assert [item["product_key"] for item in payload] == ["blueprint_basic", "blueprint_pro", "acp"]
    assert all(item["contract_version"] == "product-build-status.v1" for item in payload)
    assert payload[0]["entitlement"]["access_state"] == "allowed"


def test_product_build_status_api_returns_single_product_surface(client: TestClient) -> None:
    headers = _auth_headers(client)
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    response = client.get(f"/api/v1/sessions/{session_id}/product-builds/blueprint_basic", headers=headers)
    assert response.status_code == 200
    payload = response.json()

    assert payload["contract_version"] == "product-build-status.v1"
    assert payload["product_key"] == "blueprint_basic"
    assert payload["progress"]["calculation"] == "weighted_units"


def test_product_journey_overview_api_returns_canonical_v2(client: TestClient) -> None:
    headers = _auth_headers(client)
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    response = client.get(f"/api/v1/sessions/{session_id}/product-journey-overview", headers=headers)
    assert response.status_code == 200
    payload = response.json()

    assert payload["contract_version"] == "product-journey-overview.v2"
    assert payload["session_id"] == session_id
    assert [item["product_key"] for item in payload["products"]] == ["blueprint_basic", "blueprint_pro", "acp"]
    assert payload["recommended_next_action"]["product_key"] == "blueprint_basic"
    assert "product-build-status.v1" in payload["source_contracts"]


def test_legacy_product_overview_delegates_to_canonical_overview(client: TestClient) -> None:
    headers = _auth_headers(client)
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    canonical_response = client.get(f"/api/v1/sessions/{session_id}/product-journey-overview", headers=headers)
    assert canonical_response.status_code == 200
    canonical = canonical_response.json()

    legacy_response = client.get(f"/api/v1/sessions/{session_id}/product-overview", headers=headers)
    assert legacy_response.status_code == 200
    legacy = legacy_response.json()

    canonical_blueprint = next(item for item in canonical["products"] if item["product_key"] == "blueprint_basic")
    legacy_blueprint = next(item for item in legacy["products"] if item["key"] == "blueprint")

    assert legacy["canonical_overview_contract"] == "product-journey-overview.v2"
    assert legacy["recommended_next_action"]["action_key"] == canonical["recommended_next_action"]["action_key"]
    assert legacy["active_stage"] == canonical["current_stage"]["stage_key"]
    assert legacy_blueprint["progress_percent"] == canonical_blueprint["progress_percent"]


def test_product_build_telemetry_api_returns_admin_report(client: TestClient) -> None:
    headers = _auth_headers(client)
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    response = client.get(f"/api/v1/sessions/{session_id}/product-builds/telemetry", headers=headers)
    assert response.status_code == 200
    payload = response.json()

    assert payload["contract_version"] == "product-build-telemetry.v1"
    assert payload["session_id"] == session_id
    assert [item["product_key"] for item in payload["products"]] == ["blueprint_basic", "blueprint_pro", "acp"]
    assert payload["totals"]["product_count"] == 3
    assert "raw prompts" in payload["redaction_policy"].lower()
