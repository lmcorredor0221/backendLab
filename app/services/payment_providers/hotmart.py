from __future__ import annotations

from uuid import uuid4

from app.core.config import get_settings
from app.services.hotmart.auth import normalize_hotmart_environment
from app.services.payment_providers.base import CheckoutProviderContext, CheckoutProviderDraft


class HotmartPaymentProvider:
    provider_key = "hotmart"

    def create_checkout_draft(self, context: CheckoutProviderContext) -> CheckoutProviderDraft:
        environment = normalize_hotmart_environment(get_settings().hotmart_environment)
        checkout_ref = f"hotmart_{uuid4().hex}"
        return CheckoutProviderDraft(
            provider=self.provider_key,
            checkout_ref=checkout_ref,
            checkout_url="",
            metadata={
                "provider_stage": "hotmart_order_pending_payment_link",
                "hotmart_environment": environment,
                "requires_payment_link": True,
                "payment_link_stage": "stage_3",
                "success_url": context.success_url,
                "cancel_url": context.cancel_url,
            },
        )

