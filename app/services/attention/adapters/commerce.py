from __future__ import annotations

from typing import Any

from app.models import AttentionItemV2
from app.services.attention.contract import create_attention_item_v2


def _value(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip()


def items_from_commercial_access(
    access: Any,
    pending_requests: list[Any] | None = None,
    approved_requests: list[Any] | None = None,
    *,
    base_href: str,
    return_href: str,
) -> list[AttentionItemV2]:
    del approved_requests
    items: list[AttentionItemV2] = []
    if _value(getattr(access, "checkout_state", "")) == "pending":
        items.append(
            create_attention_item_v2(
                item_type="confirmation",
                severity="warning",
                product="commercial",
                stage="commercial",
                source="checkout",
                source_ref={"artifact_id": "commercial_access", "field_path": "checkout_state"},
                title="Checkout pendiente",
                reason="Existe una orden pendiente antes de activar el producto.",
                impact="El acceso premium se habilitara al confirmar la orden; el modo basico sigue disponible.",
                consequence_if_unresolved="El usuario seguira viendo contenidos en modo basico hasta completar la orden.",
                action_kind="confirm",
                href=f"{base_href}/blueprint/pro",
                return_href=return_href,
                owner_role="workspace_owner",
            )
        )
    for request in pending_requests or []:
        request_id = _value(getattr(request, "id", ""))
        product_key = _value(getattr(request, "product_key", "")) or "producto premium"
        items.append(
            create_attention_item_v2(
                item_type="access_request",
                severity="warning",
                product="commercial",
                stage="commercial",
                source="access_request",
                source_ref={"artifact_id": "commercial_access", "entity_id": request_id, "field_path": "requests"},
                title=f"Solicitud de acceso a {product_key}",
                reason=_value(getattr(request, "reason", "")) or "Un usuario solicito habilitar una capacidad premium.",
                impact="Un owner o admin debe aprobar o rechazar la solicitud.",
                consequence_if_unresolved="La solicitud permanecera pendiente y el acceso no cambiara.",
                action_kind="approve",
                href=f"{base_href}/attention",
                return_href=return_href,
                owner_role="owner/admin",
            )
        )
    return items
