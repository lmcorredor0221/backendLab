from __future__ import annotations

from collections import Counter
from hashlib import sha256
import re
from collections.abc import Iterable, Mapping
from typing import Any
from uuid import UUID

from app.models import (
    AttentionActionV2,
    AttentionItemV2,
    AttentionOptionV2,
    AttentionResponseV2,
    AttentionSourceRefV2,
)

ATTENTION_CONTRACT_VERSION_V2 = "attention.v2"

ACTIONABLE_ATTENTION_STATUSES_V2 = frozenset({"open", "in_progress"})

_SEVERITY_ORDER = {"blocking": 0, "warning": 1, "info": 2}
_TYPE_ORDER = {
    "runtime_error": 0,
    "gap": 1,
    "question": 2,
    "decision": 3,
    "approval": 4,
    "hitl": 5,
    "confirmation": 6,
    "validation": 7,
    "inconsistency": 8,
    "stale": 9,
    "access_request": 10,
}
_PRODUCT_ORDER = {"blueprint": 0, "acp": 1, "commercial": 2}
_STAGE_ORDER = {
    "discover": 0,
    "draft_capture": 0,
    "define": 1,
    "definition": 1,
    "design": 2,
    "tools": 3,
    "memory": 4,
    "validate": 5,
    "validation": 5,
    "estimate": 6,
    "package": 7,
    "acp": 8,
    "commercial": 9,
    "runtime": 10,
}


def _value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").strip()


def _slug(value: Any) -> str:
    normalized = re.sub(r"[^a-z0-9._:-]+", "-", _value(value).lower()).strip("-")
    return normalized or "unknown"


def _source_ref_from_mapping(source_ref: AttentionSourceRefV2 | Mapping[str, Any] | None) -> AttentionSourceRefV2:
    if isinstance(source_ref, AttentionSourceRefV2):
        return source_ref
    if not source_ref:
        return AttentionSourceRefV2()
    return AttentionSourceRefV2(
        artifact_id=source_ref.get("artifact_id"),
        artifact_version=source_ref.get("artifact_version"),
        entity_id=source_ref.get("entity_id"),
        field_path=source_ref.get("field_path"),
    )


def build_attention_key(
    *,
    item_type: str,
    stage: str,
    source: str,
    source_ref: AttentionSourceRefV2 | Mapping[str, Any] | None = None,
    fallback: str = "",
) -> str:
    ref = _source_ref_from_mapping(source_ref)
    parts = [
        ATTENTION_CONTRACT_VERSION_V2,
        _slug(item_type),
        _slug(stage),
        _slug(source),
        _slug(ref.artifact_id),
        _slug(ref.artifact_version),
        _slug(ref.entity_id),
        _slug(ref.field_path),
        _slug(fallback),
    ]
    key = ":".join(parts)
    if len(key) <= 180:
        return key
    digest = sha256(key.encode("utf-8")).hexdigest()[:20]
    return ":".join(parts[:4] + [digest])


def _default_action_label(kind: str) -> str:
    return {
        "answer": "Responder",
        "approve": "Aprobar",
        "reject": "Rechazar",
        "confirm": "Confirmar",
        "regenerate": "Regenerar",
        "retry": "Reintentar",
    }.get(kind, "Abrir")


def create_attention_item_v2(
    *,
    item_type: str,
    severity: str,
    product: str,
    stage: str,
    source: str,
    title: str,
    reason: str,
    action_kind: str = "navigate",
    href: str,
    return_href: str = "",
    impact: str = "",
    consequence_if_unresolved: str = "",
    status: str = "open",
    owner_role: str = "",
    owner_user_id: str = "",
    options: Iterable[AttentionOptionV2 | Mapping[str, Any]] | None = None,
    suggested_answer: str = "",
    source_ref: AttentionSourceRefV2 | Mapping[str, Any] | None = None,
    affected_artifact_refs: Iterable[str] | None = None,
    action_label: str = "",
    can_resolve_inline: bool = False,
    key: str = "",
) -> AttentionItemV2:
    ref = _source_ref_from_mapping(source_ref)
    normalized_key = key or build_attention_key(
        item_type=item_type,
        stage=stage,
        source=source,
        source_ref=ref,
        fallback=title,
    )
    normalized_options = [
        option if isinstance(option, AttentionOptionV2) else AttentionOptionV2(**dict(option))
        for option in (options or [])
    ]
    normalized_return_href = return_href or href
    return AttentionItemV2(
        key=normalized_key,
        type=item_type,  # type: ignore[arg-type]
        severity=severity,  # type: ignore[arg-type]
        blocking=severity == "blocking",
        product=product,  # type: ignore[arg-type]
        stage=_value(stage),
        source=_value(source),
        source_ref=ref,
        title=title,
        reason=reason,
        impact=impact,
        consequence_if_unresolved=consequence_if_unresolved or impact or reason,
        status=status,  # type: ignore[arg-type]
        owner_role=owner_role,
        owner_user_id=owner_user_id,
        options=normalized_options,
        suggested_answer=suggested_answer,
        action=AttentionActionV2(
            kind=action_kind,  # type: ignore[arg-type]
            label=action_label or _default_action_label(action_kind),
            href=href,
            return_href=normalized_return_href,
            can_resolve_inline=can_resolve_inline,
        ),
        affected_artifact_refs=list(affected_artifact_refs or []),
    )


def sort_attention_items_v2(items: Iterable[AttentionItemV2], *, current_stage: str = "") -> list[AttentionItemV2]:
    current_stage_key = _slug(current_stage)

    def sort_key(item: AttentionItemV2) -> tuple[int, int, int, int, int, str]:
        item_stage = _slug(item.stage)
        is_current_stage = 0 if current_stage_key and item_stage == current_stage_key else 1
        status_penalty = 0 if item.status in ACTIONABLE_ATTENTION_STATUSES_V2 else 1
        return (
            status_penalty,
            _SEVERITY_ORDER.get(item.severity, 9),
            is_current_stage,
            _STAGE_ORDER.get(item_stage, 99),
            _TYPE_ORDER.get(item.type, 99),
            item.key,
        )

    return sorted(items, key=sort_key)


def dedupe_attention_items_v2(items: Iterable[AttentionItemV2], *, current_stage: str = "") -> list[AttentionItemV2]:
    winners: dict[str, AttentionItemV2] = {}
    for item in sort_attention_items_v2(items, current_stage=current_stage):
        winners.setdefault(item.key, item)
    return sort_attention_items_v2(winners.values(), current_stage=current_stage)


def _count_by(items: Iterable[AttentionItemV2], attribute: str) -> dict[str, int]:
    counter = Counter(_value(getattr(item, attribute, "")) for item in items)
    counter.pop("", None)
    return dict(sorted(counter.items()))


def build_attention_response_v2(
    *,
    session_id: UUID,
    workspace_id: UUID,
    items: Iterable[AttentionItemV2],
    count_items: Iterable[AttentionItemV2] | None = None,
    current_stage: str = "",
    cursor: str = "",
) -> AttentionResponseV2:
    ordered = dedupe_attention_items_v2(items, current_stage=current_stage)
    counted = ordered if count_items is None else dedupe_attention_items_v2(count_items, current_stage=current_stage)
    actionable = [item for item in counted if item.status in ACTIONABLE_ATTENTION_STATUSES_V2]
    primary_item = next((item for item in actionable if item.severity != "info"), None) or (actionable[0] if actionable else None)
    return AttentionResponseV2(
        session_id=session_id,
        workspace_id=workspace_id,
        current_stage=current_stage,
        total_count=len(counted),
        actionable_count=len(actionable),
        blocking_count=sum(1 for item in counted if item.severity == "blocking"),
        warning_count=sum(1 for item in counted if item.severity == "warning"),
        info_count=sum(1 for item in counted if item.severity == "info"),
        counts_by_stage=_count_by(counted, "stage"),
        counts_by_type=_count_by(counted, "type"),
        counts_by_product=_count_by(counted, "product"),
        primary_item=primary_item,
        items=ordered,
        cursor=cursor,
    )
