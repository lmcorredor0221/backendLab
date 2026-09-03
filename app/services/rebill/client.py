from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import get_settings
from app.services.commerce_provider_redaction import redact_payload


@dataclass(frozen=True)
class RebillClientConfig:
    api_base_url: str = "https://api.rebill.com/v3"
    timeout_seconds: int = 30


@dataclass(frozen=True)
class RebillApiResult:
    provider_ref: str
    checkout_url: str
    http_status: int
    payload: dict[str, Any]
    payload_redacted: dict[str, Any]


class RebillApiError(RuntimeError):
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


class RebillClient:
    def __init__(
        self,
        config: RebillClientConfig | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        settings = get_settings()
        resolved = config or RebillClientConfig(
            api_base_url=settings.rebill_api_base_url,
            timeout_seconds=settings.rebill_request_timeout_seconds,
        )
        self.api_base_url = resolved.api_base_url.rstrip("/") or "https://api.rebill.com/v3"
        self.timeout_seconds = max(1, resolved.timeout_seconds)
        self.transport = transport

    def create_payment_link(
        self,
        *,
        secret_key: str,
        payload: dict[str, Any],
        idempotency_key: str,
    ) -> RebillApiResult:
        response_payload, status_code = self._request(
            "POST",
            "/payment-links",
            secret_key=secret_key,
            json_payload=payload,
            idempotency_key=idempotency_key,
        )
        if status_code >= 400:
            raise RebillApiError(
                "payment_link_rejected",
                "Rebill rejected the payment link creation request.",
                http_status=status_code,
                payload=response_payload,
            )
        checkout_url = _extract_checkout_url(response_payload)
        provider_ref = _extract_provider_ref(response_payload) or checkout_url
        if not checkout_url and not provider_ref:
            raise RebillApiError(
                "invalid_payment_link_response",
                "Rebill payment link response did not include a checkout URL or identifier.",
                http_status=status_code,
                payload=response_payload,
            )
        return RebillApiResult(
            provider_ref=provider_ref,
            checkout_url=checkout_url,
            http_status=status_code,
            payload=response_payload,
            payload_redacted=redact_payload(response_payload),
        )

    def get_payment(self, *, secret_key: str, payment_id: str) -> dict[str, Any]:
        response_payload, status_code = self._request(
            "GET",
            f"/payments/{payment_id}",
            secret_key=secret_key,
        )
        if status_code >= 400:
            raise RebillApiError(
                "payment_lookup_failed",
                "Rebill payment confirmation request failed.",
                http_status=status_code,
                payload=response_payload,
            )
        return response_payload

    def test_connection(self, *, secret_key: str) -> tuple[bool, str, int | None]:
        response_payload, status_code = self._request(
            "GET",
            "/payments",
            secret_key=secret_key,
            query={"limit": "1"},
        )
        if status_code >= 400:
            return False, "Rebill returned an error while validating the secret key.", status_code
        return True, "Rebill API is reachable with the configured secret key.", status_code

    def _request(
        self,
        method: str,
        path: str,
        *,
        secret_key: str,
        json_payload: dict[str, Any] | None = None,
        idempotency_key: str = "",
        query: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any], int]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-api-key": secret_key,
        }
        if idempotency_key:
            headers["x-idempotency-key"] = idempotency_key
        url = f"{self.api_base_url}{path if path.startswith('/') else f'/{path}'}"
        try:
            with httpx.Client(timeout=self.timeout_seconds, transport=self.transport) as client:
                response = client.request(method, url, headers=headers, json=json_payload, params=query)
        except httpx.HTTPError as exc:
            raise RebillApiError("network_error", "Unable to reach Rebill API.", payload={"error": str(exc)}) from exc
        try:
            payload = response.json() if response.text else {}
        except ValueError:
            payload = {"raw": response.text[:500]}
        if not isinstance(payload, dict):
            payload = {"payload": payload}
        return payload, response.status_code


def _extract_provider_ref(payload: Any) -> str:
    for value in _candidate_values(
        payload,
        ("id",),
        ("uuid",),
        ("payment_link_id",),
        ("paymentLinkId",),
        ("data", "id"),
        ("data", "uuid"),
        ("data", "payment_link_id"),
        ("data", "paymentLinkId"),
        ("payment_link", "id"),
        ("paymentLink", "id"),
    ):
        if value:
            return value
    return ""


def _extract_checkout_url(payload: Any) -> str:
    for value in _candidate_values(
        payload,
        ("url",),
        ("checkout_url",),
        ("checkoutUrl",),
        ("payment_url",),
        ("paymentUrl",),
        ("link",),
        ("data", "url"),
        ("data", "checkout_url"),
        ("data", "checkoutUrl"),
        ("data", "payment_url"),
        ("data", "paymentUrl"),
        ("payment_link", "url"),
        ("paymentLink", "url"),
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
