from __future__ import annotations

from typing import Literal

from pydantic import Field as PydanticField, field_validator, model_validator

from app.models import AttentionItemV2, AttentionOptionV2, ContractModel
from app.services.attention.contract import create_attention_item_v2

ATTENTION_DECISION_CONTRACT_VERSION_V3 = "attention.decision.v3"


class AttentionDecisionSourceV3(ContractModel):
    product: Literal["blueprint", "acp", "commercial"] = "blueprint"
    stage: str
    source: str
    artifact_id: str | None = None
    artifact_version: int | None = None
    entity_id: str | None = None
    field_path: str | None = None
    href: str
    return_href: str = ""
    owner_role: str = "business_owner"
    affected_artifact_refs: list[str] = PydanticField(default_factory=list)

    @field_validator("stage", "source", "href")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Decision source fields stage, source and href are required")
        return normalized


class AttentionDecisionOptionV3(ContractModel):
    key: str
    label: str
    description: str = ""
    impact: str = ""
    example: str = ""
    recommended: bool = False
    confidence: float = 0.0
    risk: str = ""
    tradeoff: str = ""
    source_refs: list[str] = PydanticField(default_factory=list)

    @field_validator("key", "label")
    @classmethod
    def validate_option_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Decision option key and label are required")
        return normalized

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        return min(max(float(value or 0.0), 0.0), 1.0)

    def to_attention_option_v2(self) -> AttentionOptionV2:
        return AttentionOptionV2(
            key=self.key,
            label=self.label,
            description=self.description,
            impact=self.impact,
            example=self.example,
            recommended=self.recommended,
            confidence=self.confidence,
            source_refs=self.source_refs,
        )


class AttentionDecisionActionV3(ContractModel):
    primary_kind: Literal["navigate", "answer", "approve", "reject", "confirm", "regenerate", "retry"] = "navigate"
    primary_label: str = ""
    can_resolve_inline: bool = False
    allowed_kinds: list[Literal["navigate", "answer", "approve", "reject", "confirm", "regenerate", "retry", "defer"]] = (
        PydanticField(default_factory=list)
    )


class AttentionDecisionV3(ContractModel):
    contract_version: Literal["attention.decision.v3"] = ATTENTION_DECISION_CONTRACT_VERSION_V3
    decision_key: str = ""
    item_type: Literal[
        "question",
        "gap",
        "decision",
        "approval",
        "confirmation",
        "validation",
        "hitl",
        "inconsistency",
        "stale",
        "runtime_error",
        "access_request",
    ] = "decision"
    severity: Literal["info", "warning", "blocking"] = "warning"
    status: Literal["open", "in_progress", "deferred", "resolved", "dismissed", "superseded"] = "open"
    title: str
    reason: str
    impact: str = ""
    consequence_if_unresolved: str
    required_decision: str = ""
    suggested_answer: str = ""
    source: AttentionDecisionSourceV3
    options: list[AttentionDecisionOptionV3] = PydanticField(default_factory=list)
    action: AttentionDecisionActionV3 = PydanticField(default_factory=AttentionDecisionActionV3)
    diagnostics: dict[str, object] = PydanticField(default_factory=dict)

    @field_validator("title", "reason", "consequence_if_unresolved")
    @classmethod
    def validate_required_decision_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Decision title, reason and consequence are required")
        return normalized

    @model_validator(mode="after")
    def validate_decision_guidance(self) -> "AttentionDecisionV3":
        if self.item_type == "question" and not self.suggested_answer and not self.options:
            raise ValueError("Questions must include a suggested answer or answer options")
        if self.severity == "blocking" and self.action.primary_kind == "navigate":
            raise ValueError("Blocking decisions must expose a resolving primary action")
        recommended_count = sum(1 for option in self.options if option.recommended)
        if recommended_count > 1:
            raise ValueError("Only one decision option can be recommended")
        return self


def decision_to_attention_item_v2(decision: AttentionDecisionV3) -> AttentionItemV2:
    source = decision.source
    return create_attention_item_v2(
        item_type=decision.item_type,
        severity=decision.severity,
        product=source.product,
        stage=source.stage,
        source=source.source,
        source_ref={
            "artifact_id": source.artifact_id,
            "artifact_version": source.artifact_version,
            "entity_id": source.entity_id or decision.decision_key,
            "field_path": source.field_path,
        },
        title=decision.title,
        reason=decision.reason,
        impact=decision.impact,
        consequence_if_unresolved=decision.consequence_if_unresolved,
        status=decision.status,
        action_kind=decision.action.primary_kind,
        action_label=decision.action.primary_label,
        href=source.href,
        return_href=source.return_href or source.href,
        owner_role=source.owner_role,
        options=[option.to_attention_option_v2() for option in decision.options],
        suggested_answer=decision.suggested_answer,
        affected_artifact_refs=source.affected_artifact_refs,
        can_resolve_inline=decision.action.can_resolve_inline,
        diagnostics=decision.diagnostics,
        key=decision.decision_key,
    )
