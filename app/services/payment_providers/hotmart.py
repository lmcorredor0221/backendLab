from __future__ import annotations

from uuid import uuid4

from sqlmodel import Session

from app.core.config import get_settings
from app.models import CommercialOrderRecord, CommercialOrderStatus, HotmartPaymentLinkCreateRequest
from app.services.hotmart.auth import normalize_hotmart_environment
from app.services.payment_providers.base import CheckoutProviderContext, CheckoutProviderDraft
from app.services.payment_providers.template import CheckoutProviderFinalizeResult
from app.services.payment_providers.template import TemplateCommercePaymentProvider


class HotmartPaymentProvider(TemplateCommercePaymentProvider):
    provider_key = "hotmart"
    display_name = "Hotmart"

    def build_checkout_seed(self, context: CheckoutProviderContext) -> CheckoutProviderDraft:
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

    def finalize_checkout(
        self,
        session: Session,
        *,
        order: CommercialOrderRecord,
        context: CheckoutProviderContext,
    ) -> CheckoutProviderFinalizeResult:
        # Import lazily to avoid a module cycle with commerce_service.
        from app.services.commerce_provider_scope import resolve_commerce_provider_configuration_workspace_id
        from app.services.hotmart.payment_links import HotmartPaymentLinkError, create_hotmart_payment_link_for_order

        environment = normalize_hotmart_environment(get_settings().hotmart_environment)
        configuration_workspace_id = resolve_commerce_provider_configuration_workspace_id(
            session,
            workspace_id=order.workspace_id,
        )
        callback_url = _hotmart_callback_url(context.base_url)
        try:
            payment_link = create_hotmart_payment_link_for_order(
                session,
                workspace_id=order.workspace_id,
                integration_workspace_id=configuration_workspace_id,
                payload=HotmartPaymentLinkCreateRequest(
                    order_id=order.id,
                    environment=environment,  # type: ignore[arg-type]
                    callback_url=callback_url,
                ),
            )
        except ValueError as exc:
            return CheckoutProviderFinalizeResult(
                metadata={
                    "provider_stage": "hotmart_order_pending_payment_link",
                    "commerce_provider_configuration_workspace_id": str(configuration_workspace_id),
                    "hotmart_environment": environment,
                    "hotmart_payment_link_error": str(exc),
                },
            )
        except HotmartPaymentLinkError as exc:
            return CheckoutProviderFinalizeResult(
                metadata={
                    "provider_stage": "hotmart_payment_link_failed",
                    "commerce_provider_configuration_workspace_id": str(configuration_workspace_id),
                    "hotmart_environment": environment,
                    "hotmart_payment_link_error_code": exc.code,
                    "hotmart_payment_link_http_status": exc.http_status,
                },
            )
        return CheckoutProviderFinalizeResult(
            checkout_url=payment_link.checkout_url,
            status=CommercialOrderStatus.pending,
            provider_payment_link_id=payment_link.provider_ref or payment_link.hotmart_payment_link_id,
            metadata={
                "provider_stage": "hotmart_payment_link_created",
                "commerce_provider_configuration_workspace_id": str(configuration_workspace_id),
                "hotmart_environment": environment,
                "hotmart_payment_link_record_id": str(payment_link.id),
                "hotmart_payment_link_id": payment_link.hotmart_payment_link_id,
                "hotmart_provider_ref": payment_link.provider_ref,
                "hotmart_payment_link_activation_status": payment_link.activation_status,
            },
        )


def _hotmart_callback_url(base_url: str) -> str:
    normalized_base = base_url.strip().rstrip("/")
    if not normalized_base:
        return ""
    if normalized_base.startswith(("http://localhost", "http://127.0.0.1")):
        return ""
    return f"{normalized_base}/api/v1/webhooks/hotmart"
