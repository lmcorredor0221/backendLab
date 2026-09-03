from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from app.services.payment_providers.base import CommercePaymentProvider
from app.services.payment_providers.hotmart import HotmartPaymentProvider
from app.services.payment_providers.sandbox import SandboxPaymentProvider


@dataclass(frozen=True)
class CommerceProviderDefinition:
    provider_key: str
    display_name: str
    capabilities: tuple[str, ...] = field(default_factory=tuple)
    default_environment: str = "sandbox"
    create_provider: Callable[[], CommercePaymentProvider] = SandboxPaymentProvider


class CommerceProviderRegistry:
    def __init__(self, definitions: list[CommerceProviderDefinition]) -> None:
        self._definitions = {definition.provider_key: definition for definition in definitions}

    @property
    def supported_provider_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._definitions))

    def list_definitions(self) -> list[CommerceProviderDefinition]:
        return [self._definitions[key] for key in self.supported_provider_keys]

    def require_definition(self, provider_key: str) -> CommerceProviderDefinition:
        candidate = provider_key.strip().lower()
        definition = self._definitions.get(candidate)
        if definition is None:
            raise ValueError(f"Unsupported commerce checkout provider: {candidate}")
        return definition

    def create_provider(self, provider_key: str) -> CommercePaymentProvider:
        return self.require_definition(provider_key).create_provider()


def _rebill_provider_factory() -> CommercePaymentProvider:
    from app.services.payment_providers.rebill import RebillPaymentProvider

    return RebillPaymentProvider()


def get_commerce_provider_registry() -> CommerceProviderRegistry:
    return CommerceProviderRegistry(
        [
            CommerceProviderDefinition(
                provider_key="sandbox",
                display_name="Sandbox",
                capabilities=("hosted_checkout", "test_mode"),
                create_provider=SandboxPaymentProvider,
            ),
            CommerceProviderDefinition(
                provider_key="hotmart",
                display_name="Hotmart",
                capabilities=("payment_links", "webhooks", "external_activation"),
                create_provider=HotmartPaymentProvider,
            ),
            CommerceProviderDefinition(
                provider_key="rebill",
                display_name="Rebill",
                capabilities=("hosted_checkout", "payment_links", "subscriptions", "webhooks"),
                create_provider=_rebill_provider_factory,
            ),
        ]
    )


def list_commerce_provider_definitions() -> list[CommerceProviderDefinition]:
    return get_commerce_provider_registry().list_definitions()
