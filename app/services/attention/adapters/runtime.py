from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.models import AttentionItemV2
from app.services.attention.contract import create_attention_item_v2
from app.services.attention.runtime_issue_normalizer import runtime_operation_to_attention_item

RuntimeAttentionKind = str

_RUNTIME_RECOVERY_MARKERS = (
    "policy=",
    "timeout",
    "timed out",
    "codex local",
    "no pudo ejecutar",
    "no pudo generar",
    "failed",
    "failure",
    "error",
    "exception",
)

_PROVIDER_SCHEMA_FAILURE_MARKERS = (
    "provider failed",
    "provider failure",
    "provider error",
    "schema failed",
    "schema failure",
    "schema mismatch",
    "schema invalid",
    "invalid schema",
    "schema invalido",
    "schema inválido",
    "schema no valido",
    "schema no válido",
    "esquema invalido",
    "esquema inválido",
    "esquema no valido",
    "esquema no válido",
    "no entrego una salida valida",
    "no entregó una salida válida",
    "no entrego respuesta valida",
    "no entregó respuesta válida",
)

_HUMAN_ACTION_MARKERS = (
    "hitl",
    "human",
    "humana",
    "humano",
    "usuario",
    "user",
    "aprobacion",
    "aprobación",
    "approval",
    "handoff",
    "decision",
    "decisión",
    "intervencion",
    "intervención",
    "accion humana",
    "acción humana",
    "waiting_human",
    "waiting for user",
    "requiere revision humana",
    "requiere revisión humana",
    "requiere una decision",
    "requiere una decisión",
)

_BENIGN_RUNTIME_SUMMARY_MARKERS = (
    "definition consolidada",
    "definicion consolidada",
    "consolidada con",
    "consolidado con",
    "generacion completada",
    "generacion finalizada",
    "resultado sincronizado",
)

_TRUE_VALUES = {"1", "true", "yes", "si", "sí", "y", "on"}


def _value(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip()


def _get(source: Any, key: str, default: Any = "") -> Any:
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def _boolish(source: Any, *keys: str) -> bool:
    for key in keys:
        value = _get(source, key, None)
        if isinstance(value, bool):
            if value:
                return True
            continue
        if str(value or "").strip().lower() in _TRUE_VALUES:
            return True
    return False


def _raw_operation_text(operation: Any) -> str:
    return " ".join(
        part
        for part in (
            _value(_get(operation, "title", "")),
            _value(_get(operation, "message", "")),
            _value(_get(operation, "summary", "")),
            _value(_get(operation, "reason", "")),
            _value(_get(operation, "error_kind", "")),
            _value(_get(operation, "failure_code", "")),
        )
        if part
    )


def _has_runtime_error_signal(operation: Any, state: str, raw_text: str) -> bool:
    error_kind = _value(_get(operation, "error_kind", "")).lower()
    failure_code = _value(_get(operation, "failure_code", "")).lower()
    return (
        state in {"error", "failed", "retry_available"}
        or bool(error_kind or failure_code)
        or any(marker in raw_text for marker in _RUNTIME_RECOVERY_MARKERS)
        or any(marker in raw_text for marker in _PROVIDER_SCHEMA_FAILURE_MARKERS)
        or _boolish(operation, "recoverable_error", "runtime_error", "technical_error")
    )


@dataclass(frozen=True)
class RuntimeOperationClassification:
    kind: RuntimeAttentionKind
    state: str
    reason: str


def classify_runtime_operation(operation: Any) -> RuntimeOperationClassification:
    """Classify runtime events before they enter Attention.

    Attention must only surface actionable work. Internal skill names, summaries
    or checkpoints are trace evidence, not enough to block the user.
    """

    state = _value(_get(operation, "state", _get(operation, "status", ""))).lower()
    raw_text = _raw_operation_text(operation).lower()
    if state not in {"waiting_for_user", "waiting_human", "blocked", "error", "failed", "retry_available"}:
        return RuntimeOperationClassification(kind="none", state=state, reason="non_actionable_state")

    if _has_runtime_error_signal(operation, state, raw_text):
        return RuntimeOperationClassification(kind="runtime_error", state=state, reason="runtime_error_signal")

    if any(marker in raw_text for marker in _BENIGN_RUNTIME_SUMMARY_MARKERS):
        return RuntimeOperationClassification(kind="none", state=state, reason="benign_runtime_summary")

    has_human_action_signal = (
        state in {"waiting_for_user", "waiting_human"}
        or _boolish(operation, "requires_user_action", "human_action_required", "needs_human", "needs_attention")
        or any(marker in raw_text for marker in _HUMAN_ACTION_MARKERS)
    )
    if has_human_action_signal:
        return RuntimeOperationClassification(kind="hitl", state=state, reason="human_action_signal")

    return RuntimeOperationClassification(kind="none", state=state, reason="blocked_without_actionable_signal")


def items_from_runtime_operation(
    operation: Any,
    *,
    href: str,
    return_href: str,
) -> list[AttentionItemV2]:
    classification = classify_runtime_operation(operation)
    if classification.kind == "none":
        return []
    if classification.kind == "runtime_error":
        return [runtime_operation_to_attention_item(operation, href=href, return_href=return_href)]
    operation_id = _value(_get(operation, "id", "")) or _value(_get(operation, "operation_id", "")) or "operation"
    stage = _value(_get(operation, "stage", "")) or "runtime"
    return [
        create_attention_item_v2(
            item_type="hitl",
            severity="blocking",
            product=_value(_get(operation, "product", "")) or "blueprint",
            stage=stage,
            source="runtime_operation",
            source_ref={"artifact_id": "operation", "entity_id": operation_id, "field_path": "state"},
            title=_value(_get(operation, "title", "")) or "Operacion requiere intervencion",
            reason=_value(_get(operation, "message", "")) or "El runtime necesita una accion humana para continuar.",
            impact=_value(_get(operation, "impact", "")),
            consequence_if_unresolved="La operacion permanecera detenida hasta resolver la condicion.",
            action_kind="navigate",
            href=href,
            return_href=return_href,
            owner_role=_value(_get(operation, "owner_role", "")) or "operator",
        )
    ]
