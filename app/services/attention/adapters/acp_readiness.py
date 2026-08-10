from __future__ import annotations

from collections.abc import Iterable
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


def items_from_construction_readiness(
    readiness: Any,
    *,
    base_href: str,
    return_href: str,
    answered_question_keys: set[str] | None = None,
) -> list[AttentionItemV2]:
    items: list[AttentionItemV2] = []
    answered_keys = answered_question_keys or set()
    for gap in getattr(readiness, "gaps", []) or []:
        if _value(getattr(gap, "status", "open")) not in {"", "open"}:
            continue
        gap_key = _value(getattr(gap, "gap_key", ""))
        stage = _value(getattr(gap, "blocking_stage", "")) or "package"
        severity = "blocking" if _value(getattr(gap, "severity", "")) == "blocking" else "warning"
        evidence_paths = list(getattr(gap, "evidence_paths", []) or [])
        items.append(
            create_attention_item_v2(
                item_type="gap",
                severity=severity,
                product="acp",
                stage=stage,
                source="acp_readiness",
                source_ref={"artifact_id": "construction_readiness", "entity_id": gap_key, "field_path": "gaps"},
                title=_value(getattr(gap, "title", "")) or "GAP de implementacion",
                reason=_value(getattr(gap, "summary", "")) or "Existe un GAP pendiente para construir el ACP.",
                impact=_value(getattr(gap, "remediation", "")),
                consequence_if_unresolved="El ACP conservara una decision pendiente o no podra cerrar la preparacion tecnica.",
                action_kind="navigate",
                href=base_href,
                return_href=return_href,
                owner_role="implementation_owner",
                affected_artifact_refs=evidence_paths,
            )
        )
        for question in getattr(gap, "questions", []) or []:
            question_key = _value(getattr(question, "question_key", ""))
            if not question_key or question_key in answered_keys:
                continue
            blocking = bool(getattr(question, "blocking", False))
            items.append(
                create_attention_item_v2(
                    item_type="question",
                    severity="blocking" if blocking else "warning",
                    product="acp",
                    stage=stage,
                    source="acp_questions",
                    source_ref={
                        "artifact_id": "construction_readiness",
                        "entity_id": question_key,
                        "field_path": f"gaps.{gap_key}.questions",
                    },
                    title=_value(getattr(question, "question_text", "")) or "Pregunta de implementacion",
                    reason=_value(getattr(question, "rationale", "")) or _value(getattr(gap, "summary", "")),
                    impact=_value(getattr(question, "purpose", "")),
                    consequence_if_unresolved="La pregunta quedara documentada para resolverse durante la implementacion.",
                    action_kind="answer",
                    href=base_href,
                    return_href=return_href,
                    owner_role=_value(getattr(question, "target_owner", "")) or "implementation_owner",
                    options=_coerce_question_options(getattr(question, "options", []) or []),
                    affected_artifact_refs=evidence_paths,
                    can_resolve_inline=True,
                )
            )
    return items


def _coerce_question_options(options: Iterable[Any]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for index, option in enumerate(options, start=1):
        key = _value(getattr(option, "key", "")) or f"option_{index}"
        label = _value(getattr(option, "label", "")) or key
        normalized.append(
            {
                "key": key,
                "label": label,
                "description": _value(getattr(option, "description", "")),
                "impact": _value(getattr(option, "impact", "")),
                "example": _value(getattr(option, "example", "")),
                "recommended": bool(getattr(option, "recommended", False)),
                "confidence": _float_value(getattr(option, "confidence", 0.0)),
                "source_refs": list(getattr(option, "source_refs", []) or []),
            }
        )
    return normalized
