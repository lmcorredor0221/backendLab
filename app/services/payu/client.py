from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import get_settings
from app.services.commerce_provider_redaction import redact_payload
from app.services.commerce_provider_utils import normalize_commerce_provider_environment


DEFAULT_PAYU_API_URLS = {
    "sandbox": "https://sandbox.api.payulatam.com/payments-api/4.0/service.cgi",
    "production": "https://api.payulatam.com/payments-api/4.0/service.cgi",
}


@dataclass(frozen=True)
class PayUClientConfig:
    api_base_url: str = ""
    timeout_seconds: int = 30
    environment: str = "sandbox"


class PayUApiError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        http_status: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        self.payload = redact_payload(payload or {})


class PayUClient:
    def __init__(
        self,
        config: PayUClientConfig | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        settings = get_settings()
        resolved = config or PayUClientConfig(
            api_base_url=settings.payu_api_base_url,
            timeout_seconds=settings.payu_request_timeout_seconds,
            environment=settings.payu_environment,
        )
        env = normalize_commerce_provider_environment(resolved.environment)
        self.api_base_url = (resolved.api_base_url or DEFAULT_PAYU_API_URLS[env]).rstrip("/")
        self.timeout_seconds = max(1, resolved.timeout_seconds)
        self.environment = env
        self.transport = transport

    def test_connection(self, *, api_key: str, api_login: str, is_test: bool | None = None) -> tuple[bool, str, int | None]:
        payload = {
            "test": self.environment == "sandbox" if is_test is None else is_test,
            "language": "es",
            "command": "PING",
            "merchant": {
                "apiLogin": api_login,
                "apiKey": api_key,
            },
        }
        response_payload, status_code = self._request(payload)
        if status_code >= 400:
            return False, "PayU returned an HTTP error while validating the API credentials.", status_code
        if str(response_payload.get("code") or "").upper() == "SUCCESS":
            return True, "PayU API is reachable with the configured API key and API login.", status_code
        return False, str(response_payload.get("error") or "PayU did not accept the configured API credentials."), status_code

    def _request(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(timeout=self.timeout_seconds, transport=self.transport) as client:
                response = client.post(self.api_base_url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise PayUApiError("network_error", "Unable to reach PayU API.", payload={"error": str(exc)}) from exc
        try:
            response_payload = response.json() if response.text else {}
        except ValueError:
            response_payload = {"raw": response.text[:500]}
        if not isinstance(response_payload, dict):
            response_payload = {"payload": response_payload}
        return response_payload, response.status_code
