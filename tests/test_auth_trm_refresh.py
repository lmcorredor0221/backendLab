from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from tests.api_testkit import TEST_EMAIL, TEST_PASSWORD, build_test_client


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    with build_test_client(monkeypatch) as test_client:
        yield test_client


def test_successful_login_refreshes_trm(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    calls: list[str] = []

    def fake_refresh_trm_on_login() -> dict[str, object]:
        calls.append("refresh")
        return {"rate": 4000.0, "date": "2026-08-10", "fetched_at": 1.0}

    monkeypatch.setattr("app.api.routes.auth.refresh_trm_on_login", fake_refresh_trm_on_login)

    response = client.post(
        "/api/v1/auth/login",
        json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
    )

    assert response.status_code == 200
    assert calls == ["refresh"]


def test_failed_login_does_not_refresh_trm(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    calls: list[str] = []

    def fake_refresh_trm_on_login() -> dict[str, object]:
        calls.append("refresh")
        return {"rate": 4000.0, "date": "2026-08-10", "fetched_at": 1.0}

    monkeypatch.setattr("app.api.routes.auth.refresh_trm_on_login", fake_refresh_trm_on_login)

    response = client.post(
        "/api/v1/auth/login",
        json={"email": TEST_EMAIL, "password": "wrong-password"},
    )

    assert response.status_code == 401
    assert calls == []


def test_trm_force_refresh_bypasses_cached_value(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import commerce_service

    calls: list[str] = []
    commerce_service._TRM_CACHE.update({"rate": 1.0, "date": "2000-01-01", "fetched_at": 9_999_999_999.0})

    class FakeResponse:
        status = 200

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'[{"valor":"4001.25","vigenciadesde":"2026-08-10T00:00:00.000","vigenciahasta":"2026-08-10T00:00:00.000"}]'

    def fake_urlopen(*args: object, **kwargs: object) -> FakeResponse:
        calls.append("urlopen")
        return FakeResponse()

    monkeypatch.setattr(commerce_service.urllib.request, "urlopen", fake_urlopen)

    data = commerce_service.get_today_trm_data(force_refresh=True)

    assert calls == ["urlopen"]
    assert data["rate"] == 4001.25
    assert data["date"] == "2026-08-10"
