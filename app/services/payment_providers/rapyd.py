from __future__ import annotations

import re
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
from app.services.rapyd.client import RapydClient, RapydClientConfig


class RapydPaymentProvider(TemplateCommercePaymentProvider):
    provider_key = "rapyd"
    display_name = "Rapyd"
    client_factory = RapydClient

    def build_checkout_seed(self, context: CheckoutProviderContext) -> CheckoutProviderDraft:
        checkout_ref = f"rapyd_{uuid4().hex}"
        environment = normalize_commerce_provider_environment(get_settings().rapyd_environment)
        return CheckoutProviderDraft(
            provider=self.provider_key,
            checkout_ref=checkout_ref,
            checkout_url="",
            status=CommercialOrderStatus.pending,
            metadata={
                "provider_stage": "rapyd_order_seeded",
                "rapyd_environment": environment,
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
        environment = normalize_commerce_provider_environment(settings.rapyd_environment)
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
            raise ValueError("Rapyd provider is disabled for this workspace.")
        access_key = load_commerce_provider_secret(
            session,
            workspace_id=configuration_workspace_id,
            provider_key=self.provider_key,
            environment=environment,
            secret_kind="access_key",
        )
        secret_key = load_commerce_provider_secret(
            session,
            workspace_id=configuration_workspace_id,
            provider_key=self.provider_key,
            environment=environment,
            secret_kind="secret_key",
        )
        if not access_key:
            raise ValueError("Rapyd access key is not configured for this workspace.")
        if not secret_key:
            raise ValueError("Rapyd secret key is not configured for this workspace.")

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
                f"Rapyd product mapping is not configured for product {context.product.product_key}"
                f"{f' and package {package_code}' if package_code else ''}."
            )

        lab_metadata = {
            "lab_provider": self.provider_key,
            "lab_order_id": str(order.id),
            "lab_checkout_ref": order.checkout_ref,
            "lab_workspace_id": str(order.workspace_id),
            "lab_configuration_workspace_id": str(configuration_workspace_id),
            "lab_session_id": str(order.session_id or ""),
            "lab_product_key": context.product.product_key,
            "lab_price_code": context.price.price_code,
            "lab_package_code": package_code,
            "lab_environment": environment,
        }
        payload = _build_rapyd_checkout_payload(
            context=context,
            order=order,
            mapping=mapping,
            metadata=lab_metadata,
        )
        idempotency_key = f"rapyd:{order.id}:checkout"
        client = self.client_factory(
            RapydClientConfig(
                api_base_url=status.api_base_url or settings.rapyd_api_base_url,
                timeout_seconds=settings.rapyd_request_timeout_seconds,
                environment=environment,
            )
        )
        result = client.create_checkout(
            access_key=access_key,
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
        checkout_amount_cents = _rapyd_checkout_amount_cents(order=order, mapping=mapping)
        checkout_record.provider_checkout_id = result.provider_ref
        checkout_record.checkout_url = result.checkout_url
        checkout_record.status = "created"
        checkout_record.amount_cents = checkout_amount_cents
        checkout_record.currency = str(payload.get("currency") or order.currency)
        checkout_record.request_payload_redacted = redact_payload(payload)
        checkout_record.response_payload_redacted = result.payload_redacted
        checkout_record.metadata_payload = {
            "idempotency_key": idempotency_key,
            "mapping_id": str(mapping.id),
            "configuration_workspace_id": str(configuration_workspace_id),
            "provider_stage": "rapyd_checkout_created",
            "rapyd_checkout_id": result.provider_ref,
        }
        session.add(checkout_record)
        session.flush()
        return CheckoutProviderFinalizeResult(
            checkout_url=result.checkout_url,
            status=CommercialOrderStatus.pending,
            provider_checkout_id=result.provider_ref,
            metadata={
                "provider_stage": "rapyd_checkout_created",
                "commerce_provider_configuration_workspace_id": str(configuration_workspace_id),
                "rapyd_environment": environment,
                "rapyd_checkout_id": result.provider_ref,
                "commerce_provider_checkout_record_id": str(checkout_record.id),
                "rapyd_request_idempotency_key": idempotency_key,
            },
        )

    def build_next_action(self, order: CommercialOrderRecord) -> str:
        if order.status == CommercialOrderStatus.paid:
            return "refresh_access"
        if order.status == CommercialOrderStatus.pending and not order.checkout_url:
            return "await_payment_link"
        return super().build_next_action(order)


def _build_rapyd_checkout_payload(
    *,
    context: CheckoutProviderContext,
    order: CommercialOrderRecord,
    mapping,
    metadata: dict[str, str],
) -> dict[str, object]:
    country = _safe_country(mapping.provider_plan_id)
    if not country:
        raise ValueError("Rapyd country is required in the provider_plan_id mapping field, for example CO, MX or AR.")
    currency = (mapping.currency or order.currency or "USD").strip().upper()
    amount_cents = _rapyd_checkout_amount_cents(order=order, mapping=mapping)
    payload: dict[str, object] = {
        "amount": _rapyd_amount_from_cents(amount_cents),
        "country": country,
        "currency": currency,
        "description": _safe_rapyd_text(context.product.name or context.product.product_key),
        "merchant_reference_id": order.checkout_ref,
        "metadata": metadata,
        "language": "es",
        "complete_payment_url": _checkout_return_url(
            context.success_url,
            context.base_url,
            f"/checkout/rapyd/{order.checkout_ref}/success",
        ),
        "error_payment_url": _checkout_return_url(
            context.cancel_url,
            context.base_url,
            f"/checkout/rapyd/{order.checkout_ref}/cancel",
        ),
    }
    included_methods = _clean_rapyd_csv(mapping.provider_price_id)
    excluded_methods = _clean_rapyd_csv(mapping.provider_payment_link_id)
    if included_methods:
        payload["payment_method_types_include"] = included_methods
    if excluded_methods:
        payload["payment_method_types_exclude"] = excluded_methods
    merchant_ewallet = _safe_rapyd_identifier(mapping.provider_product_id)
    if merchant_ewallet:
        payload["merchant_ewallet"] = merchant_ewallet
    statement_descriptor = _safe_rapyd_text(mapping.provider_offer_ref)[:22]
    if statement_descriptor:
        payload["statement_descriptor"] = statement_descriptor
    return payload


def _rapyd_checkout_amount_cents(*, order: CommercialOrderRecord, mapping) -> int:
    mapping_amount_cents = int(getattr(mapping, "internal_unit_amount_usd_cents", 0) or 0)
    if mapping_amount_cents > 0:
        return mapping_amount_cents
    return max(0, order.total_cents)


def _rapyd_amount_from_cents(amount_cents: int) -> float:
    return round(max(0, amount_cents) / 100, 2)


def _checkout_return_url(primary_url: str, base_url: str, fallback_path: str) -> str:
    if primary_url.strip():
        return primary_url.strip()
    if base_url.strip():
        return f"{base_url.rstrip('/')}{fallback_path}"
    return "https://www.leanagentbuilder.com/"


def _safe_country(value: str) -> str:
    candidate = re.sub(r"[^A-Za-z]", "", value or "").upper()
    return candidate if len(candidate) == 2 else ""


def _safe_rapyd_identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_,._-]", "", str(value or "").strip())[:255]


def _clean_rapyd_csv(value: str) -> list[str]:
    parts = [_safe_rapyd_identifier(part.strip()) for part in str(value or "").split(",")]
    return [part for part in parts if part]


def _safe_rapyd_text(value: str) -> str:
    text = " ".join(str(value or "").split())
    return re.sub(r"[~<>]", "", text)[:255]
