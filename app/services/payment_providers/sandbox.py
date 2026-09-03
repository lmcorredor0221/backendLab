from __future__ import annotations

from uuid import uuid4

from app.services.payment_providers.base import CheckoutProviderContext, CheckoutProviderDraft
from app.services.payment_providers.template import TemplateCommercePaymentProvider


class SandboxPaymentProvider(TemplateCommercePaymentProvider):
    provider_key = "sandbox"
    display_name = "Sandbox"

    def build_checkout_seed(self, context: CheckoutProviderContext) -> CheckoutProviderDraft:
        checkout_ref = f"sandbox_{uuid4().hex}"
        fallback_checkout_url = (
            f"{context.base_url.rstrip('/')}/checkout/sandbox/{checkout_ref}"
            if context.base_url
            else f"/checkout/sandbox/{checkout_ref}"
        )
        return CheckoutProviderDraft(
            provider=self.provider_key,
            checkout_ref=checkout_ref,
            checkout_url=context.success_url or fallback_checkout_url,
            metadata={
                "provider_stage": "sandbox_checkout",
                "checkout_url_strategy": "success_url" if context.success_url else "local_sandbox_fallback",
            },
        )
