from __future__ import annotations

import re
from urllib.parse import quote
from uuid import uuid4

from sqlmodel import Session, select

from app.core.config import get_settings
from app.models import (
    CommerceProviderCheckoutRecord,
    CommercialOrderRecord,
    CommercialOrderStatus,
)
from app.services.commerce_provider_mappings import find_commerce_provider_mapping
from app.services.commerce_provider_redaction import redact_payload
from app.services.commerce_provider_scope import resolve_commerce_provider_configuration_workspace_id
from app.services.commerce_provider_secrets import build_commerce_provider_status, load_commerce_provider_secret
from app.services.commerce_provider_utils import normalize_commerce_provider_environment
from app.services.payment_providers.base import CheckoutProviderContext, CheckoutProviderDraft
from app.services.payment_providers.template import CheckoutProviderFinalizeResult, TemplateCommercePaymentProvider
from app.services.payu.signatures import format_payu_amount_from_cents, sign_payu_payment_form


DEFAULT_PAYU_CHECKOUT_URLS = {
    "sandbox": "https://sandbox.checkout.payulatam.com/ppp-web-gateway-payu/",
    "production": "https://checkout.payulatam.com/ppp-web-gateway-payu/",
}


class PayUPaymentProvider(TemplateCommercePaymentProvider):
    provider_key = "payu"
    display_name = "PayU Latam"

    def build_checkout_seed(self, context: CheckoutProviderContext) -> CheckoutProviderDraft:
        token = uuid4().hex
        environment = normalize_commerce_provider_environment(get_settings().payu_environment)
        checkout_ref = f"payu_{token}"
        return CheckoutProviderDraft(
            provider=self.provider_key,
            checkout_ref=checkout_ref,
            checkout_url="",
            status=CommercialOrderStatus.pending,
            metadata={
                "provider_stage": "payu_order_seeded",
                "payu_environment": environment,
                "payu_reference_code": f"payu{token}",
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
        environment = normalize_commerce_provider_environment(settings.payu_environment)
        configuration_workspace_id = resolve_commerce_provider_configuration_workspace_id(
            session,
            workspace_id=order.workspace_id,
        )
        status = build_commerce_provider_status(
            session,
            workspace_id=configuration_workspace_id,
            provider_key=self.provider_key,
            environment=environment,
        )
        if not status.enabled:
            raise ValueError("PayU provider is disabled for this workspace.")
        api_key = load_commerce_provider_secret(
            session,
            workspace_id=configuration_workspace_id,
            provider_key=self.provider_key,
            environment=environment,
            secret_kind="secret_key",
        )
        merchant_id = load_commerce_provider_secret(
            session,
            workspace_id=configuration_workspace_id,
            provider_key=self.provider_key,
            environment=environment,
            secret_kind="merchant_id",
        )
        fallback_account_id = load_commerce_provider_secret(
            session,
            workspace_id=configuration_workspace_id,
            provider_key=self.provider_key,
            environment=environment,
            secret_kind="account_id",
        )
        if not api_key:
            raise ValueError("PayU API key is not configured for this workspace.")
        if not merchant_id:
            raise ValueError("PayU merchant ID is not configured for this workspace.")
        if not status.webhook_public_url:
            raise ValueError("PayU confirmation URL is not configured for this workspace.")
        package_code = str(order.metadata_payload.get("package_code") or "")
        mapping = find_commerce_provider_mapping(
            session,
            workspace_id=configuration_workspace_id,
            provider_key=self.provider_key,
            environment=environment,
            internal_product_key=context.product.product_key,
            package_code=package_code,
        )
        if mapping is None:
            raise ValueError(
                f"PayU product mapping is not configured for product {context.product.product_key}"
                f"{f' and package {package_code}' if package_code else ''}."
            )
        account_id = (mapping.provider_product_id or fallback_account_id).strip()
        if not account_id:
            raise ValueError("PayU account ID is required in credentials or in the product mapping.")

        checkout_gateway_url = _payu_checkout_gateway_url(environment)
        form_fields = _build_payu_form_fields(
            context=context,
            order=order,
            api_key=api_key,
            merchant_id=merchant_id,
            account_id=account_id,
            confirmation_url=status.webhook_public_url,
            mapping=mapping,
            environment=environment,
        )
        checkout_url = _build_checkout_redirect_url(context.base_url, order.checkout_ref)
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
        checkout_record.provider_checkout_id = str(form_fields["referenceCode"])
        checkout_record.checkout_url = checkout_url
        checkout_record.status = "created"
        checkout_record.amount_cents = order.total_cents
        checkout_record.currency = str(form_fields["currency"])
        checkout_record.request_payload_redacted = redact_payload(
            {"action": checkout_gateway_url, "method": "POST", "fields": form_fields}
        )
        checkout_record.response_payload_redacted = {"checkout_url": checkout_url}
        checkout_record.metadata_payload = {
            "mapping_id": str(mapping.id),
            "configuration_workspace_id": str(configuration_workspace_id),
            "provider_stage": "payu_webcheckout_form_created",
            "payu_reference_code": str(form_fields["referenceCode"]),
            "payu_checkout_gateway_url": checkout_gateway_url,
            "payu_form_fields": form_fields,
        }
        session.add(checkout_record)
        session.flush()
        return CheckoutProviderFinalizeResult(
            checkout_url=checkout_url,
            status=CommercialOrderStatus.pending,
            provider_checkout_id=str(form_fields["referenceCode"]),
            metadata={
                "provider_stage": "payu_webcheckout_form_created",
                "commerce_provider_configuration_workspace_id": str(configuration_workspace_id),
                "payu_environment": environment,
                "payu_reference_code": str(form_fields["referenceCode"]),
                "commerce_provider_checkout_record_id": str(checkout_record.id),
            },
        )


def _payu_checkout_gateway_url(environment: str) -> str:
    settings = get_settings()
    configured = settings.payu_checkout_base_url.strip().rstrip("/")
    if configured:
        return f"{configured}/"
    return DEFAULT_PAYU_CHECKOUT_URLS[normalize_commerce_provider_environment(environment)]


def _build_checkout_redirect_url(base_url: str, checkout_ref: str) -> str:
    path = f"/api/v1/commerce/checkout-redirects/payu/{quote(checkout_ref)}"
    if base_url:
        return f"{base_url.rstrip('/')}{path}"
    return path


def _build_checkout_response_url(base_url: str, checkout_ref: str) -> str:
    path = f"/api/v1/commerce/checkout-responses/payu/{quote(checkout_ref)}"
    if base_url:
        return f"{base_url.rstrip('/')}{path}"
    return path


def _build_payu_form_fields(
    *,
    context: CheckoutProviderContext,
    order: CommercialOrderRecord,
    api_key: str,
    merchant_id: str,
    account_id: str,
    confirmation_url: str,
    mapping,
    environment: str,
) -> dict[str, str]:
    amount = format_payu_amount_from_cents(order.total_cents)
    currency = (mapping.currency or order.currency or "USD").strip().upper()
    reference_code = _safe_reference_code(str(order.metadata_payload.get("payu_reference_code") or order.checkout_ref))
    payment_methods = _clean_payu_csv(mapping.provider_price_id)
    iin = ""
    pse_banks = ""
    signature = sign_payu_payment_form(
        api_key=api_key,
        merchant_id=merchant_id,
        reference_code=reference_code,
        amount=amount,
        currency=currency,
        algorithm="MD5",
        payment_methods=payment_methods,
        iin=iin,
        pse_banks=pse_banks,
    )
    response_url = _build_checkout_response_url(context.base_url, order.checkout_ref) or context.success_url
    fields: dict[str, str] = {
        "lng": "es",
        "merchantId": merchant_id.strip(),
        "accountId": account_id.strip(),
        "description": _safe_payu_text(context.product.name or context.product.product_key),
        "referenceCode": reference_code,
        "amount": amount,
        "tax": "0",
        "taxReturnBase": "0",
        "currency": currency,
        "signature": signature,
        "algorithmSignature": "MD5",
        "test": "1" if environment == "sandbox" else "0",
        "buyerEmail": context.current_user.email,
        "responseUrl": response_url,
        "confirmationUrl": confirmation_url.strip(),
        "displayShippingInformation": "0",
        "extra1": order.checkout_ref,
        "extra2": str(order.id),
        "extra3": str(order.workspace_id),
    }
    if payment_methods:
        fields["paymentMethods"] = payment_methods
    selected_payment_method = _safe_identifier(mapping.provider_payment_link_id)
    if selected_payment_method:
        fields["selectedPaymentMethod"] = selected_payment_method
    template = _safe_identifier(mapping.provider_offer_ref)
    if template:
        fields["template"] = template
    billing_country = _safe_country(mapping.provider_plan_id)
    if billing_country:
        fields["billingCountry"] = billing_country
    full_name = _safe_payu_text(context.current_user.full_name or "")
    if full_name:
        fields["payerFullName"] = full_name
        fields["buyerFullName"] = full_name
    return fields


def _safe_reference_code(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", value or "")
    return cleaned[:255] or f"payu{uuid4().hex}"


def _safe_identifier(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_,._-]", "", value or "")
    return cleaned[:255]


def _clean_payu_csv(value: str) -> str:
    parts = [_safe_identifier(part.strip()) for part in str(value or "").split(",")]
    return ",".join(part for part in parts if part)


def _safe_country(value: str) -> str:
    candidate = re.sub(r"[^A-Za-z]", "", value or "").upper()
    return candidate if len(candidate) == 2 else ""


def _safe_payu_text(value: str) -> str:
    text = " ".join(str(value or "").split())
    return re.sub(r"[~<>]", "", text)[:255]
