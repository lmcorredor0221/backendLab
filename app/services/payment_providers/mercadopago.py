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
from app.services.mercadopago.client import MercadoPagoClient, MercadoPagoClientConfig
from app.services.payment_providers.base import CheckoutProviderContext, CheckoutProviderDraft
from app.services.payment_providers.template import CheckoutProviderFinalizeResult, TemplateCommercePaymentProvider


class MercadoPagoPaymentProvider(TemplateCommercePaymentProvider):
    provider_key = "mercadopago"
    display_name = "Mercado Pago"
    client_factory = MercadoPagoClient

    def build_checkout_seed(self, context: CheckoutProviderContext) -> CheckoutProviderDraft:
        checkout_ref = f"mp_{uuid4().hex}"
        environment = normalize_commerce_provider_environment(get_settings().mercadopago_environment)
        return CheckoutProviderDraft(
            provider=self.provider_key,
            checkout_ref=checkout_ref,
            checkout_url="",
            status=CommercialOrderStatus.pending,
            metadata={
                "provider_stage": "mercadopago_order_seeded",
                "mercadopago_environment": environment,
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
        environment = normalize_commerce_provider_environment(settings.mercadopago_environment)
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
            raise ValueError("Mercado Pago provider is disabled for this workspace.")
        access_token = load_commerce_provider_secret(
            session,
            workspace_id=configuration_workspace_id,
            provider_key=self.provider_key,
            environment=environment,
            secret_kind="secret_key",
        )
        if not access_token:
            raise ValueError("Mercado Pago access token is not configured for this workspace.")
        if not status.webhook_public_url:
            raise ValueError("Mercado Pago webhook URL is not configured for this workspace.")
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
                f"Mercado Pago product mapping is not configured for product {context.product.product_key}"
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
        payload = _build_mercadopago_order_payload(
            context=context,
            order=order,
            mapping=mapping,
        )
        idempotency_key = f"mercadopago:{order.id}:order"
        client = self.client_factory(
            MercadoPagoClientConfig(
                api_base_url=status.api_base_url or settings.mercadopago_api_base_url,
                timeout_seconds=settings.mercadopago_request_timeout_seconds,
                environment=environment,
            )
        )
        result = client.create_order(
            access_token=access_token,
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
        checkout_amount_cents = _mercadopago_checkout_amount_cents(order=order, mapping=mapping)
        checkout_record.provider_checkout_id = result.provider_ref
        checkout_record.checkout_url = result.checkout_url
        checkout_record.status = "created"
        checkout_record.amount_cents = checkout_amount_cents
        checkout_record.currency = (mapping.currency or order.currency or "COP").strip().upper()
        checkout_record.request_payload_redacted = redact_payload(payload)
        checkout_record.response_payload_redacted = result.payload_redacted
        checkout_record.metadata_payload = {
            "idempotency_key": idempotency_key,
            "mapping_id": str(mapping.id),
            "configuration_workspace_id": str(configuration_workspace_id),
            "provider_stage": "mercadopago_order_created",
            "mercadopago_order_id": result.provider_ref,
            "lab_metadata": lab_metadata,
        }
        session.add(checkout_record)
        session.flush()
        return CheckoutProviderFinalizeResult(
            checkout_url=result.checkout_url,
            status=CommercialOrderStatus.pending,
            provider_checkout_id=result.provider_ref,
            metadata={
                "provider_stage": "mercadopago_order_created",
                "commerce_provider_configuration_workspace_id": str(configuration_workspace_id),
                "mercadopago_environment": environment,
                "mercadopago_order_id": result.provider_ref,
                "commerce_provider_checkout_record_id": str(checkout_record.id),
                "mercadopago_request_idempotency_key": idempotency_key,
            },
        )

    def build_next_action(self, order: CommercialOrderRecord) -> str:
        if order.status == CommercialOrderStatus.paid:
            return "refresh_access"
        if order.status == CommercialOrderStatus.pending and not order.checkout_url:
            return "await_payment_link"
        return super().build_next_action(order)


def _build_mercadopago_order_payload(
    *,
    context: CheckoutProviderContext,
    order: CommercialOrderRecord,
    mapping,
) -> dict[str, object]:
    amount_cents = _mercadopago_checkout_amount_cents(order=order, mapping=mapping)
    amount = _amount_text_from_cents(amount_cents)
    success_url = _checkout_return_url(context.success_url, context.base_url, f"/checkout/mercadopago/{order.checkout_ref}/success")
    failure_url = _checkout_return_url(context.cancel_url, context.base_url, f"/checkout/mercadopago/{order.checkout_ref}/cancel")
    pending_url = _checkout_return_url(context.cancel_url or context.success_url, context.base_url, f"/checkout/mercadopago/{order.checkout_ref}/pending")
    item: dict[str, object] = {
        "external_code": context.product.product_key,
        "title": _safe_mercadopago_text(context.product.name or context.product.product_key),
        "description": _safe_mercadopago_text(context.product.description),
        "quantity": 1,
        "unit_price": amount,
    }
    category_id = _safe_mercadopago_identifier(mapping.provider_product_id)
    if category_id:
        item["category_id"] = category_id
    binary_mode = _metadata_bool(mapping.metadata_payload, "binary_mode")
    config: dict[str, object] = {
        "online": {
            "success_url": success_url,
            "failure_url": failure_url,
            "pending_url": pending_url,
            "auto_return": "approved",
        }
    }
    payment_method = _build_mercadopago_payment_method_config(mapping)
    if payment_method:
        config["payment_method"] = payment_method
    statement_descriptor = _safe_statement_descriptor(mapping.provider_offer_ref)
    if statement_descriptor:
        config["statement_descriptor"] = statement_descriptor
    payload: dict[str, object] = {
        "type": "online",
        "total_amount": amount,
        "external_reference": order.checkout_ref[:64],
        "processing_mode": "manual",
        "capture_mode": "automatic" if binary_mode else "automatic_async",
        "items": [item],
        "payer": {"email": context.current_user.email},
        "config": config,
    }
    first_name, surname = _split_full_name(context.current_user.full_name or "")
    if first_name:
        payer = {"email": context.current_user.email, "first_name": first_name}
        if surname:
            payer["last_name"] = surname
        payload["payer"] = payer
    return payload


def _mercadopago_checkout_amount_cents(*, order: CommercialOrderRecord, mapping) -> int:
    mapping_amount_cents = int(getattr(mapping, "internal_unit_amount_usd_cents", 0) or 0)
    if mapping_amount_cents > 0:
        return mapping_amount_cents
    return max(0, order.total_cents)


def _amount_text_from_cents(amount_cents: int) -> str:
    whole, cents = divmod(max(0, int(amount_cents)), 100)
    return f"{whole}.{cents:02d}"


def _checkout_return_url(primary_url: str, base_url: str, fallback_path: str) -> str:
    if primary_url.strip():
        return primary_url.strip()
    if base_url.strip():
        return f"{base_url.rstrip('/')}{fallback_path}"
    return "https://www.leanagentbuilder.com/"


def _build_mercadopago_payment_method_config(mapping) -> dict[str, object]:
    payment_method: dict[str, object] = {}
    excluded_methods = _clean_mercadopago_id_list(mapping.provider_payment_link_id)
    if excluded_methods:
        payment_method["not_allowed_ids"] = excluded_methods
    excluded_types = _clean_mercadopago_id_list(mapping.provider_plan_id)
    if excluded_types:
        payment_method["not_allowed_types"] = excluded_types
    default_payment_type = _safe_mercadopago_identifier(mapping.provider_price_id)
    if default_payment_type:
        payment_method["default_type"] = default_payment_type
    installments = _metadata_int(mapping.metadata_payload, "installments", minimum=1, maximum=36)
    if installments is not None:
        payment_method["max_installments"] = installments
    return payment_method


def _safe_statement_descriptor(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9 ]", "", " ".join(str(value or "").split()))
    return text[:22]


def _safe_mercadopago_identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "", str(value or "").strip())[:255]


def _clean_mercadopago_id_list(value: str) -> list[str]:
    parts = [_safe_mercadopago_identifier(part.strip()) for part in str(value or "").split(",")]
    return [part for part in parts if part]


def _metadata_int(metadata: object, key: str, *, minimum: int, maximum: int) -> int | None:
    if not isinstance(metadata, dict) or key not in metadata:
        return None
    try:
        value = int(str(metadata[key]).strip())
    except (TypeError, ValueError):
        return None
    if value < minimum or value > maximum:
        return None
    return value


def _metadata_bool(metadata: object, key: str) -> bool | None:
    if not isinstance(metadata, dict) or key not in metadata:
        return None
    value = metadata[key]
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _split_full_name(value: str) -> tuple[str, str]:
    text = _safe_mercadopago_text(value)
    if not text:
        return "", ""
    first_name, _, surname = text.partition(" ")
    return first_name[:255], surname[:255]


def _safe_mercadopago_text(value: str) -> str:
    text = " ".join(str(value or "").split())
    return re.sub(r"[<>]", "", text)[:255]
