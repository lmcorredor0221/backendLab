from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import get_settings
from app.services.commerce_provider_redaction import redact_payload
from app.services.commerce_provider_utils import normalize_commerce_provider_environment
from app.services.rapyd.signatures import compact_json_body, generate_rapyd_salt, rapyd_unix_timestamp, sign_rapyd_request


DEFAULT_RAPYD_API_URLS = {
    "sandbox": "https://sandboxapi.rapyd.net",
    "production": "https://api.rapyd.net",
}


@dataclass(frozen=True)
class RapydClientConfig:
    api_base_url: str = ""
    timeout_seconds: int = 30
    environment: str = "sandbox"


@dataclass(frozen=True)
class RapydApiResult:
    provider_ref: str
    checkout_url: str
    http_status: int
    payload: dict[str, Any]
    payload_redacted: dict[str, Any]


class RapydApiError(RuntimeError):
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


class RapydClient:
    def __init__(
        self,
        config: RapydClientConfig | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        settings = get_settings()
        resolved = config or RapydClientConfig(
            api_base_url=settings.rapyd_api_base_url,
            timeout_seconds=settings.rapyd_request_timeout_seconds,
            environment=settings.rapyd_environment,
        )
        env = normalize_commerce_provider_environment(resolved.environment)
        self.api_base_url = (resolved.api_base_url or DEFAULT_RAPYD_API_URLS[env]).rstrip("/")
        self.timeout_seconds = max(1, resolved.timeout_seconds)
        self.environment = env
        self.transport = transport

    def create_checkout(
        self,
        *,
        access_key: str,
        secret_key: str,
        payload: dict[str, Any],
        idempotency_key: str,
    ) -> RapydApiResult:
        response_payload, status_code = self._request(
            "post",
            "/v1/checkout",
            access_key=access_key,
            secret_key=secret_key,
            json_payload=payload,
            idempotency_key=idempotency_key,
        )
        if status_code >= 400:
            raise RapydApiError(
                "checkout_rejected",
                "Rapyd rejected the checkout creation request.",
                http_status=status_code,
                payload=response_payload,
            )
        checkout_url = _extract_checkout_url(response_payload)
        provider_ref = _extract_provider_ref(response_payload) or checkout_url
        if not checkout_url or not provider_ref:
            raise RapydApiError(
                "invalid_checkout_response",
                "Rapyd checkout response did not include a redirect URL and identifier.",
                http_status=status_code,
                payload=response_payload,
            )
        return RapydApiResult(
            provider_ref=provider_ref,
            checkout_url=checkout_url,
            http_status=status_code,
            payload=response_payload,
            payload_redacted=redact_payload(response_payload),
        )

    def test_connection(self, *, access_key: str, secret_key: str) -> tuple[bool, str, int | None]:
        response_payload, status_code = self._request(
            "get",
            "/v1/data/countries",
            access_key=access_key,
            secret_key=secret_key,
        )
        if status_code >= 400:
            return False, "Rapyd returned an HTTP error while validating the access key and secret key.", status_code
        status_payload = response_payload.get("status")
        if isinstance(status_payload, dict) and str(status_payload.get("status") or "").upper() == "SUCCESS":
            return True, "Rapyd API is reachable with the configured access key and secret key.", status_code
        return False, "Rapyd did not accept the configured API credentials.", status_code

    def _request(
        self,
        method: str,
        path: str,
        *,
        access_key: str,
        secret_key: str,
        json_payload: dict[str, Any] | None = None,
        idempotency_key: str = "",
    ) -> tuple[dict[str, Any], int]:
        body_string = compact_json_body(json_payload)
        salt = generate_rapyd_salt()
        timestamp = rapyd_unix_timestamp()
        signature = sign_rapyd_request(
            http_method=method,
            url_path=path,
            salt=salt,
            timestamp=timestamp,
            access_key=access_key,
            secret_key=secret_key,
            body_string=body_string,
        )
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "access_key": access_key,
            "salt": salt,
            "timestamp": timestamp,
            "signature": signature,
        }
        if idempotency_key:
            headers["idempotency"] = idempotency_key
        url = f"{self.api_base_url}{path if path.startswith('/') else f'/{path}'}"
        try:
            with httpx.Client(timeout=self.timeout_seconds, transport=self.transport) as client:
                response = client.request(method.upper(), url, headers=headers, content=body_string.encode("utf-8"))
        except httpx.HTTPError as exc:
            raise RapydApiError("network_error", "Unable to reach Rapyd API.", payload={"error": str(exc)}) from exc
        try:
            payload = response.json() if response.text else {}
        except ValueError:
            payload = {"raw": response.text[:500]}
        if not isinstance(payload, dict):
            payload = {"payload": payload}
        return payload, response.status_code


def _extract_provider_ref(payload: Any) -> str:
    for value in _candidate_values(payload, ("data", "id"), ("id",), ("checkout", "id")):
        if value:
            return value
    return ""


def _extract_checkout_url(payload: Any) -> str:
    for value in _candidate_values(
        payload,
        ("data", "redirect_url"),
        ("data", "redirectUrl"),
        ("redirect_url",),
        ("redirectUrl",),
        ("checkout", "redirect_url"),
    ):
        if value:
            return value
    return ""


def _candidate_values(payload: Any, *paths: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for path in paths:
        value = payload
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if value is not None and str(value).strip():
            values.append(str(value).strip())
    return values
