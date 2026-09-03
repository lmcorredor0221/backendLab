from __future__ import annotations

from typing import Any


SENSITIVE_KEY_PARTS = {
    "authorization",
    "card",
    "cvv",
    "hottok",
    "key",
    "password",
    "secret",
    "token",
    "x-api-key",
    "x-rebill-signature",
}


def is_sensitive_key(key: str) -> bool:
    lowered = key.strip().lower()
    return any(part in lowered for part in SENSITIVE_KEY_PARTS)


def redact_payload(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            redacted[key_text] = "[redacted]" if is_sensitive_key(key_text) else redact_payload(item)
        return redacted
    if isinstance(value, list):
        return [redact_payload(item) for item in value[:50]]
    if isinstance(value, str) and len(value) > 500:
        return f"{value[:240]}...[truncated]"
    return value


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    return {
        key: ("[redacted]" if is_sensitive_key(key) else value)
        for key, value in headers.items()
    }
