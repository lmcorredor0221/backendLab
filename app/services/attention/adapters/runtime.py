from __future__ import annotations

from typing import Any

from app.models import AttentionItemV2
from app.services.attention.contract import create_attention_item_v2


def _value(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip()


def _get(source: Any, key: str, default: Any = "") -> Any:
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def items_from_runtime_operation(
    operation: Any,
    *,
    href: str,
    return_href: str,
) -> list[AttentionItemV2]:
    state = _value(_get(operation, "state", _get(operation, "status", "")))
    if state not in {"waiting_for_user", "blocked", "error", "retry_available"}:
        return []
    operation_id = _value(_get(operation, "id", "")) or _value(_get(operation, "operation_id", "")) or "operation"
    stage = _value(_get(operation, "stage", "")) or "runtime"
    is_error = state in {"error", "retry_available"}
    return [
        create_attention_item_v2(
            item_type="runtime_error" if is_error else "hitl",
            severity="blocking",
            product=_value(_get(operation, "product", "")) or "blueprint",
            stage=stage,
            source="runtime_operation",
            source_ref={"artifact_id": "operation", "entity_id": operation_id, "field_path": "state"},
            title=_value(_get(operation, "title", "")) or "Operacion requiere intervencion",
            reason=_value(_get(operation, "message", "")) or "El runtime necesita una accion humana para continuar.",
            impact=_value(_get(operation, "impact", "")),
            consequence_if_unresolved="La operacion permanecera detenida hasta resolver la condicion.",
            action_kind="retry" if is_error else "navigate",
            href=href,
            return_href=return_href,
            owner_role=_value(_get(operation, "owner_role", "")) or "operator",
        )
    ]
