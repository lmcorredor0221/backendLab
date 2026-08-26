from __future__ import annotations

from typing import Any

from app.models import AttentionItemV2
from app.services.attention.contract import create_attention_item_v2


def _value(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip()


def _policy_refs(policy: Any) -> list[str]:
    refs: list[str] = []
    policy_key = _value(getattr(policy, "policy_key", ""))
    if policy_key:
        refs.append(f"policy_key={policy_key}")
    for raw_ref in list(getattr(policy, "evidence", []) or []):
        ref = _value(raw_ref)
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def items_from_handoffs(
    handoffs: list[Any],
    *,
    base_href: str,
    return_href: str,
) -> list[AttentionItemV2]:
    items: list[AttentionItemV2] = []
    for handoff in handoffs or []:
        if _value(getattr(handoff, "status", "")) != "pending":
            continue
        handoff_id = _value(getattr(handoff, "id", "handoff"))
        stage = _value(getattr(handoff, "from_stage", "")) or "validate"
        items.append(
            create_attention_item_v2(
                item_type="hitl",
                severity="blocking",
                product="blueprint",
                stage=stage,
                source="governance_handoff",
                source_ref={"artifact_id": "handoff", "entity_id": handoff_id, "field_path": "status"},
                title=_value(getattr(handoff, "title", "")) or "Revisión de gobierno del blueprint",
                reason=_value(getattr(handoff, "summary", "")) or "Existe un handoff pendiente de decisión humana.",
                impact="Puede bloquear la promoción o exportación del entregable.",
                consequence_if_unresolved="El flujo seguirá esperando la decisión del responsable asignado.",
                action_kind="confirm",
                action_label="Confirmar gobernanza",
                can_resolve_inline=True,
                href=f"{base_href}/attention",
                return_href=return_href,
                owner_role=_value(getattr(handoff, "owner_role", "")) or "business_owner",
            )
        )
    return items


def items_from_governance_policies(
    policies: list[Any],
    *,
    base_href: str,
    return_href: str,
) -> list[AttentionItemV2]:
    items: list[AttentionItemV2] = []
    for policy in policies or []:
        status = _value(getattr(policy, "compliance_status", "unknown"))
        if status not in {"fail", "failed", "non_compliant", "blocked", "warning"}:
            continue
        policy_id = _value(getattr(policy, "id", "")) or _value(getattr(policy, "policy_key", "policy"))
        items.append(
            create_attention_item_v2(
                item_type="validation",
                severity="blocking" if status in {"fail", "failed", "non_compliant", "blocked"} else "warning",
                product="blueprint",
                stage=_value(getattr(policy, "scope", "")) or "validate",
                source="governance_policy",
                source_ref={"artifact_id": "governance_policy", "entity_id": policy_id, "field_path": "compliance_status"},
                title=_value(getattr(policy, "label", "")) or "Politica de gobierno requiere revision",
                reason=_value(getattr(policy, "summary", "")) or "Una politica de gobierno no cumple el estado esperado.",
                impact="Puede afectar aprobaciones, auditoria o autorizacion de exportacion.",
                consequence_if_unresolved="La evidencia de gobierno quedara incompleta o bloqueada.",
                action_kind="navigate",
                href=f"{base_href}/attention",
                return_href=return_href,
                owner_role="governance_owner",
                affected_artifact_refs=_policy_refs(policy),
            )
        )
    return items
