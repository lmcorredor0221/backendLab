from __future__ import annotations

from app.core.config import get_settings
from app.services.payment_providers.base import CommercePaymentProvider
from app.services.payment_providers.hotmart import HotmartPaymentProvider
from app.services.payment_providers.sandbox import SandboxPaymentProvider


SUPPORTED_COMMERCE_PAYMENT_PROVIDERS = {"sandbox", "hotmart"}


def normalize_commerce_payment_provider(provider_key: str | None = None) -> str:
    raw_value = provider_key if provider_key is not None else get_settings().commerce_checkout_provider
    candidate = (raw_value or "sandbox").strip().lower()
    if candidate in {"", "default"}:
        return "sandbox"
    if candidate not in SUPPORTED_COMMERCE_PAYMENT_PROVIDERS:
        raise ValueError(f"Unsupported commerce checkout provider: {candidate}")
    return candidate


def get_commerce_payment_provider(provider_key: str | None = None) -> CommercePaymentProvider:
    resolved = normalize_commerce_payment_provider(provider_key)
    if resolved == "hotmart":
        return HotmartPaymentProvider()
    return SandboxPaymentProvider()

