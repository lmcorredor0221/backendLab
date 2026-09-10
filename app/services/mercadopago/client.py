from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import get_settings
from app.services.commerce_provider_redaction import redact_payload
from app.services.commerce_provider_utils import normalize_commerce_provider_environment


DEFAULT_MERCADOPAGO_API_URL = "https://api.mercadopago.com"


@dataclass(frozen=True)
class MercadoPagoClientConfig:
    api_base_url: str = ""
    timeout_seconds: int = 30
    environment: str = "sandbox"


@dataclass(frozen=True)
class MercadoPagoApiResult:
    provider_ref: str
    checkout_url: str
    http_status: int
    payload: dict[str, Any]
    payload_redacted: dict[str, Any]


class MercadoPagoApiError(RuntimeError):
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


class MercadoPagoClient:
    def __init__(
        self,
        config: MercadoPagoClientConfig | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        settings = get_settings()
        resolved = config or MercadoPagoClientConfig(
            api_base_url=settings.mercadopago_api_base_url,
            timeout_seconds=settings.mercadopago_request_timeout_seconds,
            environment=settings.mercadopago_environment,
        )
        self.api_base_url = (resolved.api_base_url or DEFAULT_MERCADOPAGO_API_URL).rstrip("/")
        self.timeout_seconds = max(1, resolved.timeout_seconds)
        self.environment = normalize_commerce_provider_environment(resolved.environment)
        self.transport = transport

    def create_order(
        self,
        *,
        access_token: str,
        payload: dict[str, Any],
        idempotency_key: str,
    ) -> MercadoPagoApiResult:
        response_payload, status_code = self._request(
            "post",
            "/v1/orders",
            access_token=access_token,
            json_payload=payload,
            idempotency_key=idempotency_key,
        )
        if status_code >= 400:
            raise MercadoPagoApiError(
                "order_rejected",
                "Mercado Pago rejected the order creation request.",
                http_status=status_code,
                payload=response_payload,
            )
        checkout_url = _extract_checkout_url(response_payload, environment=self.environment)
        provider_ref = _extract_provider_ref(response_payload) or checkout_url
        if not checkout_url or not provider_ref:
            raise MercadoPagoApiError(
                "invalid_order_response",
                "Mercado Pago order response did not include a checkout URL and identifier.",
                http_status=status_code,
                payload=response_payload,
            )
        return MercadoPagoApiResult(
            provider_ref=provider_ref,
            checkout_url=checkout_url,
            http_status=status_code,
            payload=response_payload,
            payload_redacted=redact_payload(response_payload),
        )

    def get_order(self, *, access_token: str, order_id: str) -> dict[str, Any]:
        response_payload, status_code = self._request(
            "get",
            f"/v1/orders/{order_id}",
            access_token=access_token,
        )
        if status_code >= 400:
            raise MercadoPagoApiError(
                "order_lookup_failed",
                "Mercado Pago rejected the order lookup request.",
                http_status=status_code,
                payload=response_payload,
            )
        return response_payload

    def test_connection(self, *, access_token: str) -> tuple[bool, str, int | None]:
        response_payload, status_code = self._request("get", "/users/me", access_token=access_token)
        if status_code >= 400:
            return False, "Mercado Pago returned an HTTP error while validating the access token.", status_code
        if response_payload.get("id") or response_payload.get("nickname") or response_payload.get("email"):
            return True, "Mercado Pago API is reachable with the configured access token.", status_code
        return False, "Mercado Pago did not accept the configured access token.", status_code

    def _request(
        self,
        method: str,
        path: str,
        *,
        access_token: str,
        json_payload: dict[str, Any] | None = None,
        idempotency_key: str = "",
    ) -> tuple[dict[str, Any], int]:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        if idempotency_key:
            headers["X-Idempotency-Key"] = idempotency_key
        url = f"{self.api_base_url}{path if path.startswith('/') else f'/{path}'}"
        try:
            with httpx.Client(timeout=self.timeout_seconds, transport=self.transport) as client:
                response = client.request(method.upper(), url, headers=headers, json=json_payload)
        except httpx.HTTPError as exc:
            raise MercadoPagoApiError("network_error", "Unable to reach Mercado Pago API.", payload={"error": str(exc)}) from exc
        try:
            payload = response.json() if response.text else {}
        except ValueError:
            payload = {"raw": response.text[:500]}
        if not isinstance(payload, dict):
            payload = {"payload": payload}
        return payload, response.status_code


def _extract_provider_ref(payload: Any) -> str:
    for value in _candidate_values(payload, ("id",), ("order_id",), ("preference_id",)):
        if value:
            return value
    return ""


def _extract_checkout_url(payload: Any, *, environment: str) -> str:
    preferred_paths = (
        (("checkout_url",), ("sandbox_init_point",), ("init_point",))
        if normalize_commerce_provider_environment(environment) == "sandbox"
        else (("checkout_url",), ("init_point",), ("sandbox_init_point",))
    )
    for value in _candidate_values(payload, *preferred_paths):
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
