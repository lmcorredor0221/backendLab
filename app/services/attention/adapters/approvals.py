from __future__ import annotations

from typing import Any

from app.models import AttentionItemV2
from app.services.attention.contract import create_attention_item_v2


def _value(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip()


def items_from_approval_gates(
    approvals: list[Any],
    *,
    base_href: str,
    return_href: str,
) -> list[AttentionItemV2]:
    items: list[AttentionItemV2] = []
    for approval in approvals or []:
        if _value(getattr(approval, "status", "")) != "pending":
            continue
        approval_id = _value(getattr(approval, "id", ""))
        gate_key = _value(getattr(approval, "gate_key", "")) or approval_id or "approval"
        stage = _value(getattr(approval, "requested_in_stage", "")) or "discover"
        items.append(
            create_attention_item_v2(
                item_type="approval",
                severity="blocking",
                product="blueprint",
                stage=stage,
                source="approval_gate",
                source_ref={"artifact_id": "approval_gate", "entity_id": approval_id or gate_key, "field_path": f"approvals.{gate_key}"},
                title=_value(getattr(approval, "title", "")) or "Aprobacion pendiente",
                reason=_value(getattr(approval, "rationale", "")) or "La etapa requiere aprobacion antes de continuar.",
                impact=_value(getattr(approval, "instructions", "")),
                consequence_if_unresolved="La etapa no podra promoverse mientras la aprobacion siga pendiente.",
                action_kind="approve",
                href=f"{base_href}/{stage}",
                return_href=return_href,
                owner_role="business_owner",
                can_resolve_inline=False,
            )
        )
    return items
