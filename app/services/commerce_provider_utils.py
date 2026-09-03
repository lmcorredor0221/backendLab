from __future__ import annotations


def normalize_commerce_provider_environment(environment: str | None = None) -> str:
    candidate = (environment or "sandbox").strip().lower()
    if candidate in {"", "test", "testing", "stage", "staging"}:
        return "sandbox"
    if candidate in {"prod", "live"}:
        return "production"
    if candidate not in {"sandbox", "production"}:
        raise ValueError("environment must be sandbox or production.")
    return candidate


def normalize_commerce_provider_key(provider_key: str) -> str:
    candidate = provider_key.strip().lower()
    if not candidate:
        raise ValueError("provider_key is required.")
    return candidate
