from __future__ import annotations

from uuid import uuid4

from sqlmodel import Session, select

from app.core.config import get_settings
from app.models import (
    CommerceProviderCheckoutRecord,
    CommercialOrderLineRecord,
    CommercialOrderRecord,
    CommercialOrderStatus,
)
from app.services.commerce_provider_mappings import find_commerce_provider_mapping
from app.services.commerce_provider_redaction import redact_payload
from app.services.commerce_provider_secrets import build_commerce_provider_status, load_commerce_provider_secret
from app.services.commerce_provider_utils import normalize_commerce_provider_environment
from app.services.payment_providers.base import CheckoutProviderContext, CheckoutProviderDraft
from app.services.payment_providers.template import CheckoutProviderFinalizeResult, TemplateCommercePaymentProvider
from app.services.rebill.client import RebillClient, RebillClientConfig


class RebillPaymentProvider(TemplateCommercePaymentProvider):
    provider_key = "rebill"
    display_name = "Rebill"
    client_factory = RebillClient

    def build_checkout_seed(self, context: CheckoutProviderContext) -> CheckoutProviderDraft:
        checkout_ref = f"rebill_{uuid4().hex}"
        environment = normalize_commerce_provider_environment(get_settings().rebill_environment)
        return CheckoutProviderDraft(
            provider=self.provider_key,
            checkout_ref=checkout_ref,
            checkout_url="",
            status=CommercialOrderStatus.pending,
            metadata={
                "provider_stage": "rebill_order_seeded",
                "rebill_environment": environment,
                "requires_payment_link": False,
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
        settings = get_settings()
        environment = normalize_commerce_provider_environment(settings.rebill_environment)
        status = build_commerce_provider_status(
            session,
            workspace_id=order.workspace_id,
            provider_key=self.provider_key,
            environment=environment,
        )
        if not status.enabled:
            raise ValueError("Rebill provider is disabled for this workspace.")
        secret_key = load_commerce_provider_secret(
            session,
            workspace_id=order.workspace_id,
            provider_key=self.provider_key,
            environment=environment,
            secret_kind="secret_key",
        )
        if not secret_key:
            raise ValueError("Rebill secret key is not configured for this workspace.")
        package_code = str(order.metadata_payload.get("package_code") or "")
        mapping = find_commerce_provider_mapping(
            session,
            workspace_id=order.workspace_id,
            provider_key=self.provider_key,
            environment=environment,
            internal_product_key=context.product.product_key,
            package_code=package_code,
        )
        if mapping is None:
            raise ValueError(
                f"Rebill product mapping is not configured for product {context.product.product_key}"
                f"{f' and package {package_code}' if package_code else ''}."
            )

        lab_metadata = {
            "lab_provider": self.provider_key,
            "lab_order_id": str(order.id),
            "lab_checkout_ref": order.checkout_ref,
            "lab_workspace_id": str(order.workspace_id),
            "lab_session_id": str(order.session_id or ""),
            "lab_product_key": context.product.product_key,
            "lab_price_code": context.price.price_code,
            "lab_package_code": package_code,
            "lab_environment": environment,
        }
        payload = _build_rebill_payment_link_payload(
            context=context,
            order=order,
            mapping=mapping,
            metadata=lab_metadata,
        )
        idempotency_key = f"rebill:{order.id}:payment-link"
        client = self.client_factory(
            RebillClientConfig(
                api_base_url=status.api_base_url or settings.rebill_api_base_url,
                timeout_seconds=settings.rebill_request_timeout_seconds,
            )
        )
        result = client.create_payment_link(
            secret_key=secret_key,
            payload=payload,
            idempotency_key=idempotency_key,
        )
        checkout_record = session.exec(
            select(CommerceProviderCheckoutRecord).where(
                CommerceProviderCheckoutRecord.provider_key == self.provider_key,
                CommerceProviderCheckoutRecord.checkout_ref == order.checkout_ref,
            )
        ).first()
        if checkout_record is None:
            checkout_record = CommerceProviderCheckoutRecord(
                workspace_id=order.workspace_id,
                provider_key=self.provider_key,
                environment=environment,
                order_id=order.id,
                checkout_ref=order.checkout_ref,
            )
        checkout_record.provider_payment_link_id = result.provider_ref
        checkout_record.checkout_url = result.checkout_url
        checkout_record.status = "created"
        checkout_record.amount_cents = order.total_cents
        checkout_record.currency = order.currency
        checkout_record.request_payload_redacted = redact_payload(payload)
        checkout_record.response_payload_redacted = result.payload_redacted
        checkout_record.metadata_payload = {
            "idempotency_key": idempotency_key,
            "mapping_id": str(mapping.id),
            "provider_stage": "rebill_payment_link_created",
        }
        session.add(checkout_record)
        session.flush()
        return CheckoutProviderFinalizeResult(
            checkout_url=result.checkout_url,
            status=CommercialOrderStatus.pending,
            provider_payment_link_id=result.provider_ref,
            metadata={
                "provider_stage": "rebill_payment_link_created",
                "rebill_environment": environment,
                "rebill_payment_link_id": result.provider_ref,
                "commerce_provider_checkout_record_id": str(checkout_record.id),
                "rebill_request_idempotency_key": idempotency_key,
            },
        )

    def build_next_action(self, order: CommercialOrderRecord) -> str:
        if order.status == CommercialOrderStatus.paid:
            return "refresh_access"
        if order.status == CommercialOrderStatus.pending and not order.checkout_url:
            return "await_payment_link"
        return super().build_next_action(order)


def _build_rebill_payment_link_payload(
    *,
    context: CheckoutProviderContext,
    order: CommercialOrderRecord,
    mapping,
    metadata: dict[str, str],
) -> dict[str, object]:
    amount = round(max(0, order.total_cents) / 100, 2)
    currency = (mapping.currency or order.currency or "USD").upper()
    customer: dict[str, str] = {
        "email": context.current_user.email,
        "language": "es",
    }
    full_name = _valid_rebill_full_name(context.current_user.full_name)
    if full_name:
        customer["fullName"] = full_name

    payload: dict[str, object] = {
        "title": _localized_rebill_text(context.product.name or context.product.product_key),
        "metadata": metadata,
        "paymentMethods": [{"currency": currency, "methods": ["card"]}],
        "prefilledFields": {"customer": customer},
        "redirectUrls": {
            "approved": context.success_url,
            "rejected": context.cancel_url,
        },
        "showCoupon": False,
    }
    description = _localized_rebill_text(context.product.description)
    if description:
        payload["description"] = description
    if mapping.provider_plan_id:
        payload["plan"] = {"id": mapping.provider_plan_id, "quantity": 1}
    elif mapping.provider_product_id:
        payload["product"] = {
            "id": mapping.provider_product_id,
            "quantity": 1,
            "isQuantityEditable": False,
            "isRemovable": False,
        }
    else:
        payload["prices"] = [
            {
                "amount": amount,
                "currency": currency,
                "isPriceFixed": True,
                "isDefault": True,
            }
        ]
        payload["isSingleUse"] = True
    if mapping.provider_offer_ref:
        payload["offer_ref"] = mapping.provider_offer_ref
    return payload


def _localized_rebill_text(value: str) -> list[dict[str, str]]:
    text = (value or "").strip()
    if not text:
        return []
    return [{"language": "es", "text": text}]


def _valid_rebill_full_name(value: str | None) -> str:
    text = " ".join(str(value or "").split())
    parts = [part for part in text.split(" ") if len(part) >= 2]
    if len(parts) < 2:
        return ""
    return text
