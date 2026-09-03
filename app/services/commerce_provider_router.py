from __future__ import annotations

from app.core.config import get_settings
from app.services.commerce_provider_registry import get_commerce_provider_registry
from app.services.payment_providers.base import CommercePaymentProvider


SUPPORTED_COMMERCE_PAYMENT_PROVIDERS = set(get_commerce_provider_registry().supported_provider_keys)


def normalize_commerce_payment_provider(provider_key: str | None = None) -> str:
    raw_value = provider_key if provider_key is not None else get_settings().commerce_checkout_provider
    candidate = (raw_value or "sandbox").strip().lower()
    if candidate in {"", "default"}:
        return "sandbox"
    return get_commerce_provider_registry().require_definition(candidate).provider_key


def get_commerce_payment_provider(provider_key: str | None = None) -> CommercePaymentProvider:
    resolved = normalize_commerce_payment_provider(provider_key)
    return get_commerce_provider_registry().create_provider(resolved)
