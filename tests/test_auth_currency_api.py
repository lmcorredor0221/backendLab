from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from tests.api_testkit import TEST_PASSWORD, build_test_client, TEST_EMAIL


def _auth_headers(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_auth_currency_route_persists_preference_and_auth_me_exposes_it(monkeypatch: pytest.MonkeyPatch) -> None:
    with build_test_client(monkeypatch) as client:
        headers = _auth_headers(client)

        initial_me = client.get("/api/v1/auth/me", headers=headers)
        assert initial_me.status_code == 200
        assert initial_me.json()["preferred_currency"] == "COP"

        update_response = client.patch(
            "/api/v1/auth/currency",
            headers=headers,
            json={"preferred_currency": "USD"},
        )
        assert update_response.status_code == 200
        assert update_response.json()["preferred_currency"] == "USD"

        current_currency = client.get("/api/v1/auth/currency", headers=headers)
        assert current_currency.status_code == 200
        assert current_currency.json()["preferred_currency"] == "USD"

        refreshed_me = client.get("/api/v1/auth/me", headers=headers)
        assert refreshed_me.status_code == 200
        assert refreshed_me.json()["preferred_currency"] == "USD"


def test_auth_currency_route_rejects_unknown_currency(monkeypatch: pytest.MonkeyPatch) -> None:
    with build_test_client(monkeypatch) as client:
        headers = _auth_headers(client)

        response = client.patch(
            "/api/v1/auth/currency",
            headers=headers,
            json={"preferred_currency": "EUR"},
        )

        assert response.status_code == 400
        assert "Moneda no soportada" in response.json()["detail"]
