from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from app.services.hotmart.redaction import redact_payload


HOTMART_TOKEN_PATH = "/security/oauth/token"


def normalize_hotmart_environment(environment: str | None) -> str:
    candidate = (environment or "sandbox").strip().lower()
    if candidate not in {"sandbox", "production"}:
        return "sandbox"
    return candidate


def default_hotmart_auth_base_url(environment: str | None) -> str:
    return "https://api-sec-vlc.hotmart.com"


def default_hotmart_api_base_url(environment: str | None) -> str:
    resolved = normalize_hotmart_environment(environment)
    if resolved == "sandbox":
        return "https://sandbox.hotmart.com"
    return "https://api-hot-connect.hotmart.com"


@dataclass(frozen=True)
class HotmartCredentials:
    client_id: str
    client_secret: str
    basic_token: str

    def missing_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.client_id.strip():
            missing.append("client_id")
        if not self.client_secret.strip():
            missing.append("client_secret")
        if not self.basic_token.strip():
            missing.append("basic_token")
        return missing


@dataclass(frozen=True)
class HotmartAccessToken:
    access_token: str
    token_type: str = "Bearer"
    expires_in: int | None = None
    scope: str = ""
    raw_payload_redacted: dict[str, Any] = field(default_factory=dict)


class HotmartAuthError(RuntimeError):
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


class HotmartAuthClient:
    def __init__(
        self,
        *,
        environment: str = "sandbox",
        auth_base_url: str = "",
        timeout_seconds: int = 30,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.environment = normalize_hotmart_environment(environment)
        self.auth_base_url = (auth_base_url or default_hotmart_auth_base_url(self.environment)).rstrip("/")
        self.timeout_seconds = max(1, timeout_seconds)
        self.transport = transport

    @property
    def token_url(self) -> str:
        return f"{self.auth_base_url}{HOTMART_TOKEN_PATH}"

    def fetch_access_token(self, credentials: HotmartCredentials) -> HotmartAccessToken:
        missing = credentials.missing_fields()
        if missing:
            raise HotmartAuthError(
                "missing_credentials",
                f"Missing Hotmart credential fields: {', '.join(missing)}.",
            )

        basic_token = credentials.basic_token.strip()
        authorization_value = basic_token if basic_token.lower().startswith("basic ") else f"Basic {basic_token}"
        headers = {
            "Accept": "application/json",
            "Authorization": authorization_value,
        }
        params = {
            "grant_type": "client_credentials",
            "client_id": credentials.client_id.strip(),
            "client_secret": credentials.client_secret.strip(),
        }

        try:
            with httpx.Client(timeout=self.timeout_seconds, transport=self.transport) as client:
                response = client.post(self.token_url, headers=headers, params=params)
        except httpx.HTTPError as exc:
            raise HotmartAuthError("network_error", "Hotmart OAuth request failed before a response was received.") from exc

        try:
            payload = response.json()
        except ValueError:
            payload = {"raw": response.text[:500]}

        if response.status_code >= 400:
            raise HotmartAuthError(
                "oauth_rejected",
                "Hotmart rejected the OAuth credential exchange.",
                http_status=response.status_code,
                payload=payload if isinstance(payload, dict) else {"payload": payload},
            )

        if not isinstance(payload, dict) or not str(payload.get("access_token", "")).strip():
            raise HotmartAuthError(
                "invalid_oauth_response",
                "Hotmart OAuth response did not include an access token.",
                http_status=response.status_code,
                payload=payload if isinstance(payload, dict) else {"payload": payload},
            )

        expires_in_raw = payload.get("expires_in")
        try:
            expires_in = int(expires_in_raw) if expires_in_raw is not None else None
        except (TypeError, ValueError):
            expires_in = None

        return HotmartAccessToken(
            access_token=str(payload["access_token"]),
            token_type=str(payload.get("token_type") or "Bearer"),
            expires_in=expires_in,
            scope=str(payload.get("scope") or ""),
            raw_payload_redacted=redact_payload(payload),
        )
