from __future__ import annotations

from dataclasses import dataclass

import httpx

from app.services.hotmart.auth import (
    HotmartAuthClient,
    HotmartAuthError,
    HotmartCredentials,
    default_hotmart_api_base_url,
    default_hotmart_auth_base_url,
    normalize_hotmart_environment,
)


@dataclass(frozen=True)
class HotmartClientConfig:
    environment: str = "sandbox"
    api_base_url: str = ""
    auth_base_url: str = ""
    timeout_seconds: int = 30

    @property
    def normalized_environment(self) -> str:
        return normalize_hotmart_environment(self.environment)

    @property
    def resolved_api_base_url(self) -> str:
        return (self.api_base_url or default_hotmart_api_base_url(self.normalized_environment)).rstrip("/")

    @property
    def resolved_auth_base_url(self) -> str:
        return (self.auth_base_url or default_hotmart_auth_base_url(self.normalized_environment)).rstrip("/")


@dataclass(frozen=True)
class HotmartConnectionTestResult:
    reachable: bool
    status: str
    message: str
    token_expires_in: int | None = None
    http_status: int | None = None
    rate_limit_remaining: int | None = None


class HotmartClient:
    def __init__(
        self,
        config: HotmartClientConfig,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.config = config
        self.transport = transport

    def test_connection(self, credentials: HotmartCredentials) -> HotmartConnectionTestResult:
        auth_client = HotmartAuthClient(
            environment=self.config.normalized_environment,
            auth_base_url=self.config.resolved_auth_base_url,
            timeout_seconds=self.config.timeout_seconds,
            transport=self.transport,
        )
        try:
            token = auth_client.fetch_access_token(credentials)
        except HotmartAuthError as exc:
            return HotmartConnectionTestResult(
                reachable=False,
                status=exc.code,
                message=str(exc),
                http_status=exc.http_status,
            )

        return HotmartConnectionTestResult(
            reachable=True,
            status="connected",
            message="Hotmart OAuth token exchange succeeded.",
            token_expires_in=token.expires_in,
            http_status=200,
        )

