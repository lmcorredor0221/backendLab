from __future__ import annotations

from collections.abc import Mapping
from html import escape
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from sqlmodel import Session, select

from app.models import CommerceProviderCheckoutRecord, CommercialOrderRecord
from app.services.commerce_provider_secrets import load_commerce_provider_secret
from app.services.commerce_provider_utils import normalize_commerce_provider_environment
from app.services.payu.signatures import verify_payu_confirmation_signature


def render_payu_checkout_redirect(session: Session, *, checkout_ref: str) -> str:
    order = session.exec(
        select(CommercialOrderRecord).where(
            CommercialOrderRecord.provider == "payu",
            CommercialOrderRecord.checkout_ref == checkout_ref,
        )
    ).first()
    if order is None:
        raise LookupError("PayU checkout was not found.")
    checkout_record = session.exec(
        select(CommerceProviderCheckoutRecord).where(
            CommerceProviderCheckoutRecord.provider_key == "payu",
            CommerceProviderCheckoutRecord.checkout_ref == checkout_ref,
        )
    ).first()
    if checkout_record is None:
        raise LookupError("PayU checkout record was not found.")
    metadata = dict(checkout_record.metadata_payload or {})
    gateway_url = str(metadata.get("payu_checkout_gateway_url") or "").strip()
    fields = metadata.get("payu_form_fields")
    if not gateway_url or not isinstance(fields, dict):
        raise ValueError("PayU checkout form is not ready.")
    return build_payu_checkout_redirect_html(gateway_url=gateway_url, form_fields=fields)


def build_payu_checkout_redirect_html(*, gateway_url: str, form_fields: dict[str, Any]) -> str:
    inputs = "\n".join(
        f'<input type="hidden" name="{escape(str(key), quote=True)}" value="{escape(str(value), quote=True)}">'
        for key, value in form_fields.items()
        if str(value or "").strip()
    )
    action = escape(gateway_url, quote=True)
    return f"""<!doctype html>
<html lang="es">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="robots" content="noindex,nofollow">
    <title>PayU checkout</title>
  </head>
  <body>
    <form id="payu-checkout-form" method="post" action="{action}">
      {inputs}
      <button type="submit">Continuar a PayU</button>
    </form>
    <script>
      document.getElementById("payu-checkout-form").submit();
    </script>
  </body>
</html>"""


def resolve_payu_response_redirect(
    session: Session,
    *,
    checkout_ref: str,
    query_params: Mapping[str, Any],
) -> str:
    order = session.exec(
        select(CommercialOrderRecord).where(
            CommercialOrderRecord.provider == "payu",
            CommercialOrderRecord.checkout_ref == checkout_ref,
        )
    ).first()
    if order is None:
        raise LookupError("PayU checkout response was not found.")

    metadata = dict(order.metadata_payload or {})
    success_url = str(metadata.get("success_url") or "").strip()
    cancel_url = str(metadata.get("cancel_url") or "").strip()
    fallback_url = success_url or cancel_url or "/"
    response_status = _response_status(query_params)
    verified = _verify_response_signature(session, order=order, query_params=query_params)
    target_url = success_url if verified and response_status == "approved" else cancel_url or fallback_url
    return _append_result_params(
        target_url or fallback_url,
        checkout_ref=checkout_ref,
        response_status=response_status if verified else "unverified",
    )


def _verify_response_signature(
    session: Session,
    *,
    order: CommercialOrderRecord,
    query_params: Mapping[str, Any],
) -> bool:
    environment = normalize_commerce_provider_environment(str(order.metadata_payload.get("payu_environment") or "sandbox"))
    api_key = load_commerce_provider_secret(
        session,
        workspace_id=order.workspace_id,
        provider_key="payu",
        environment=environment,
        secret_kind="secret_key",
    )
    hmac_secret = load_commerce_provider_secret(
        session,
        workspace_id=order.workspace_id,
        provider_key="payu",
        environment=environment,
        secret_kind="webhook_signing_secret",
    )
    return verify_payu_confirmation_signature(query_params, api_key=api_key, hmac_secret=hmac_secret)


def _response_status(query_params: Mapping[str, Any]) -> str:
    state = _query_value(query_params, "transactionState", "polTransactionState", "state_pol").upper()
    lap_state = _query_value(query_params, "lapTransactionState", "message").upper()
    if state == "4" or lap_state == "APPROVED":
        return "approved"
    if state in {"5", "6"} or lap_state in {"DECLINED", "EXPIRED", "ERROR", "REJECTED"}:
        return "rejected"
    return "pending"


def _query_value(query_params: Mapping[str, Any], *keys: str) -> str:
    lowered = {str(key).lower(): value for key, value in query_params.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _append_result_params(url: str, *, checkout_ref: str, response_status: str) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update(
        {
            "payment_provider": "payu",
            "payment_status": response_status,
            "checkout_ref": checkout_ref,
        }
    )
    return urlunparse(parsed._replace(query=urlencode(query)))
