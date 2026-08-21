from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from app.models import AttentionItemV2
from app.services.attention.contract import create_attention_item_v2


def _value(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip()


def _float_value(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return dict(model_dump(mode="json"))
    return {}


def _question_title(question: Any) -> str:
    data = _mapping(question)
    if data:
        for key in ("question", "question_text", "title", "summary", "detail"):
            title = _value(data.get(key))
            if title:
                return title
    return _value(question)


def _question_options(question: Any) -> list[dict[str, Any]]:
    data = _mapping(question)
    raw_options = data.get("answer_options") or data.get("options") or []
    if not isinstance(raw_options, list):
        raw_options = []
    options: list[dict[str, Any]] = []
    for index, raw_option in enumerate(raw_options, start=1):
        option = _mapping(raw_option)
        if not option and isinstance(raw_option, str):
            option = {"label": raw_option}
        label = _value(option.get("label")) or _value(option.get("value")) or _value(raw_option)
        if not label:
            continue
        key = _value(option.get("key")) or _value(option.get("value")) or f"option_{index}"
        options.append(
            {
                "key": key,
                "label": label,
                "description": _value(option.get("description")),
                "impact": _value(option.get("impact")),
                "example": _value(option.get("example")),
                "recommended": bool(option.get("recommended", False)),
                "confidence": _float_value(option.get("confidence")),
                "source_refs": list(option.get("source_refs") or []),
            }
        )
    return options or _fallback_question_options(question)


def _fallback_question_options(question: Any) -> list[dict[str, Any]]:
    data = _mapping(question)
    suggested_answer = _value(data.get("suggested_answer"))
    if not suggested_answer:
        return []
    source_refs = list(data.get("source_refs") or ["guided_question.suggested_answer"])
    return [
        {
            "key": "accept_suggested_answer",
            "label": "Usar respuesta sugerida",
            "description": suggested_answer[:220],
            "impact": _value(data.get("impact")) or "Permite cerrar la pregunta con la inferencia trazada por el sistema.",
            "example": suggested_answer[:220],
            "recommended": True,
            "confidence": _float_value(data.get("confidence")) or 0.7,
            "source_refs": source_refs,
        },
        {
            "key": "provide_custom_answer",
            "label": "Responder manualmente",
            "description": "Reemplazar la sugerencia con una respuesta propia del owner.",
            "impact": "Mantiene control humano cuando la inferencia no refleja la realidad del negocio.",
            "example": "Escribe la decision final y el motivo.",
            "recommended": False,
            "confidence": 0.55,
            "source_refs": source_refs,
        },
    ]


def _question_source_ref_entity(question: Any, index: int) -> str:
    data = _mapping(question)
    return _value(data.get("key")) or _value(data.get("question_key")) or f"question_{index}"


def _question_suggested_answer(question: Any) -> str:
    data = _mapping(question)
    return _value(data.get("suggested_answer"))


def _question_reason(question: Any) -> str:
    data = _mapping(question)
    return (
        _value(data.get("rationale"))
        or _value(data.get("reason"))
        or "La etapa genero una pregunta abierta que requiere criterio humano."
    )


def _question_impact(question: Any) -> str:
    data = _mapping(question)
    return _value(data.get("impact")) or _value(data.get("purpose")) or "Puede reducir la calidad del artefacto si no se resuelve."


def _is_non_actionable_runtime_warning(warning: str) -> bool:
    normalized = warning.strip().lower()
    if not normalized:
        return False
    runtime_markers = (
        "policy=",
        "fallback deterministico",
        "preflight heuristico",
        "no devolvio un discovery estructurado",
        "no devolvio un canvas estructurado",
    )
    return any(marker in normalized for marker in runtime_markers)


def items_from_stage_payload(
    *,
    product: str,
    stage: str,
    source: str,
    artifact_id: str,
    artifact_version: int | None,
    href: str,
    return_href: str,
    open_questions: Iterable[Any] | None = None,
    gaps: Iterable[str] | None = None,
    decisions: Iterable[Mapping[str, Any]] | None = None,
    warnings: Iterable[str] | None = None,
) -> list[AttentionItemV2]:
    items: list[AttentionItemV2] = []
    for index, question in enumerate(open_questions or [], start=1):
        title = _question_title(question)
        if not title:
            continue
        items.append(
            create_attention_item_v2(
                item_type="question",
                severity="warning",
                product=product,
                stage=stage,
                source=source,
                source_ref={
                    "artifact_id": artifact_id,
                    "artifact_version": artifact_version,
                    "entity_id": _question_source_ref_entity(question, index),
                    "field_path": "open_questions",
                },
                title=title,
                reason=_question_reason(question),
                impact=_question_impact(question),
                consequence_if_unresolved="La pregunta quedara visible como incertidumbre residual de la etapa.",
                action_kind="answer",
                href=href,
                return_href=return_href,
                owner_role="business_owner",
                options=_question_options(question),
                suggested_answer=_question_suggested_answer(question),
                can_resolve_inline=True,
            )
        )
    for index, gap in enumerate(gaps or [], start=1):
        title = _value(gap)
        if not title:
            continue
        items.append(
            create_attention_item_v2(
                item_type="gap",
                severity="blocking",
                product=product,
                stage=stage,
                source=source,
                source_ref={
                    "artifact_id": artifact_id,
                    "artifact_version": artifact_version,
                    "entity_id": f"gap_{index}",
                    "field_path": "gaps",
                },
                title=title,
                reason="Existe un GAP que impide cerrar la etapa con calidad suficiente.",
                impact="Bloquea la promocion o aprobacion hasta que se remedie.",
                consequence_if_unresolved="La etapa no deberia aprobarse sin cerrar o justificar el GAP.",
                action_kind="navigate",
                href=href,
                return_href=return_href,
                owner_role="business_owner",
            )
        )
    for index, decision in enumerate(decisions or [], start=1):
        title = _value(decision.get("title")) or f"Decision pendiente {index}"
        items.append(
            create_attention_item_v2(
                item_type="decision",
                severity=_value(decision.get("severity")) or "warning",
                product=product,
                stage=stage,
                source=source,
                source_ref={
                    "artifact_id": artifact_id,
                    "artifact_version": artifact_version,
                    "entity_id": _value(decision.get("key")) or f"decision_{index}",
                    "field_path": "decisions",
                },
                title=title,
                reason=_value(decision.get("reason")) or "La etapa requiere una decision humana.",
                impact=_value(decision.get("impact")),
                consequence_if_unresolved=_value(decision.get("consequence")) or "La decision quedara pendiente para seguimiento.",
                action_kind="confirm",
                href=href,
                return_href=return_href,
                owner_role=_value(decision.get("owner_role")) or "business_owner",
                options=decision.get("options") or [],
                can_resolve_inline=True,
            )
        )
    for index, warning in enumerate(warnings or [], start=1):
        title = _value(warning)
        if not title:
            continue
        if _is_non_actionable_runtime_warning(title):
            continue
        items.append(
            create_attention_item_v2(
                item_type="inconsistency",
                severity="warning",
                product=product,
                stage=stage,
                source=source,
                source_ref={
                    "artifact_id": artifact_id,
                    "artifact_version": artifact_version,
                    "entity_id": f"warning_{index}",
                    "field_path": "warnings",
                },
                title=title,
                reason="El sistema detecto una advertencia o inconsistencia accionable.",
                impact="Puede afectar consistencia, confianza o trazabilidad si no se revisa.",
                consequence_if_unresolved="La advertencia quedara registrada como observacion no bloqueante.",
                action_kind="navigate",
                href=href,
                return_href=return_href,
            )
        )
    return items
