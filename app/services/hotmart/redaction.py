from __future__ import annotations

from typing import Any


SENSITIVE_TOKENS = (
    "authorization",
    "basic",
    "client_secret",
    "secret",
    "token",
    "access_token",
    "refresh_token",
    "hottok",
    "password",
    "credential",
)


def is_sensitive_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return any(token in normalized for token in SENSITIVE_TOKENS)


def redact_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[redacted]" if is_sensitive_key(str(key)) else redact_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_payload(item) for item in value]
    if isinstance(value, tuple):
        return [redact_payload(item) for item in value]
    return value


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    return {
        key: "[redacted]" if is_sensitive_key(key) else value
        for key, value in headers.items()
    }

