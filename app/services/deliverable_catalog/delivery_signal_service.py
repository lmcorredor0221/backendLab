from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from sqlmodel import Session, select

from app.models import JourneyArtifactState, JourneyStageArtifactRecord, SessionRecord
from app.services.deliverable_catalog.project_generation_context import ProjectGenerationContext


APPROVED_STATES = (JourneyArtifactState.approved, JourneyArtifactState.approved_legacy)


@dataclass(frozen=True)
class ConfirmedDeliverySignals:
    signals: tuple[str, ...]
    evidence: tuple[dict[str, object], ...]


def _contains_any(values: list[str], *tokens: str) -> bool:
    text = " ".join(values).lower()
    return any(token.lower() in text for token in tokens)


def _has_truthy_key(value: object, keys: set[str]) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) in keys and item is True:
                return True
            if _has_truthy_key(item, keys):
                return True
    elif isinstance(value, list):
        return any(_has_truthy_key(item, keys) for item in value)
    return False


def _payload_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, default=str).lower()


def derive_confirmed_delivery_signals(context: ProjectGenerationContext) -> ConfirmedDeliverySignals:
    evidence: list[dict[str, object]] = []
    signals: list[str] = []
    source_ref = context.source_refs[0].ref if context.source_refs else "approved_context"

    def _add(signal: str, reason: str, refs: list[str] | None = None) -> None:
        if signal in signals:
            return
        signals.append(signal)
        evidence.append({"signal": signal, "reason": reason, "source_refs": refs or [source_ref]})

    if context.tools:
        _add(
            "confirmed_external_systems",
            "approved_tools_present",
            [tool.source_ref for tool in context.tools if tool.source_ref],
        )

    if context.rag_required is True or context.knowledge_sources:
        _add(
            "confirmed_rag_scope",
            "approved_retrieval_or_knowledge_sources_present",
            [source.source_ref for source in context.knowledge_sources if source.source_ref],
        )
    elif _contains_any([context.memory_strategy or ""], "rag", "retrieval", "vector", "knowledge", "conocimiento"):
        _add("confirmed_rag_scope", "approved_memory_strategy_mentions_retrieval")

    if any(tool.has_side_effects or tool.requires_approval for tool in context.tools):
        _add(
            "confirmed_side_effects",
            "approved_tool_requires_approval_or_has_side_effects",
            [tool.source_ref for tool in context.tools if tool.source_ref],
        )
    elif context.nondelegable_decisions or _contains_any(
        [*context.guardrails, *context.constraints, *context.risks],
        "side effect",
        "efecto secundario",
        "write",
        "modificar",
        "actualizar",
        "aprobar",
    ):
        _add("confirmed_side_effects", "approved_human_gate_or_write_risk_present")

    if _contains_any(
        [*context.constraints, *context.guardrails, *context.risks],
        "pii",
        "datos personales",
        "personal data",
        "sensible",
        "sensitive",
    ):
        _add("confirmed_pii", "approved_pii_or_sensitive_data_risk_present")

    if context.tool_contracts or _contains_any(
        [*context.constraints, *context.guardrails, *context.acceptance_criteria],
        "schema",
        "payload",
        "contrato de datos",
        "json",
    ):
        _add("confirmed_structured_payloads", "approved_tool_contract_or_payload_schema_present")

    if _contains_any(
        [context.architecture or "", *context.constraints, *context.guardrails],
        "runtime",
        "stack",
        "deploy",
        "despliegue",
        "docker",
        "kubernetes",
        "render",
        "vercel",
    ):
        _add("confirmed_runtime_stack", "approved_runtime_stack_or_deployment_evidence_present")

    return ConfirmedDeliverySignals(signals=tuple(signals), evidence=tuple(evidence))


def build_confirmed_delivery_signals(db: Session, *, record: SessionRecord) -> ConfirmedDeliverySignals:
    approved_records = db.exec(
        select(JourneyStageArtifactRecord)
        .where(
            JourneyStageArtifactRecord.session_id == record.id,
            JourneyStageArtifactRecord.state.in_(APPROVED_STATES),
        )
        .order_by(JourneyStageArtifactRecord.stage_key.asc(), JourneyStageArtifactRecord.version_number.desc())
    ).all()
    latest_by_stage: dict[str, JourneyStageArtifactRecord] = {}
    for artifact in approved_records:
        if artifact.stage_key not in latest_by_stage:
            latest_by_stage[artifact.stage_key] = artifact

    refs = [f"journey:{artifact.id}:v{artifact.version_number}" for artifact in latest_by_stage.values()]
    payload: dict[str, Any] = {
        "project_title": record.title,
        "session_id": str(record.id),
        "workspace_id": str(record.workspace_id),
        "approved_context_refs": refs,
        "approved_context": {
            "stages": {stage: artifact.proposal_payload for stage, artifact in latest_by_stage.items()},
            "artifacts": {},
        },
    }
    context = ProjectGenerationContext.from_approved_payload(
        payload,
        deliverable_key="product.delivery_plan",
        policy=None,
    )
    derived = derive_confirmed_delivery_signals(context)
    signals = list(derived.signals)
    evidence = list(derived.evidence)
    raw_context = payload["approved_context"]["stages"]
    raw_text = _payload_text(raw_context)

    def _add(signal: str, reason: str) -> None:
        if signal in signals:
            return
        signals.append(signal)
        evidence.append({"signal": signal, "reason": reason, "source_refs": refs or ["approved_context"]})

    if _has_truthy_key(raw_context, {"has_side_effects", "requires_approval", "side_effects"}):
        _add("confirmed_side_effects", "approved_payload_marks_tool_side_effects_or_approval")
    if _has_truthy_key(raw_context, {"rag_required", "retrieval_required", "requires_rag"}):
        _add("confirmed_rag_scope", "approved_payload_marks_retrieval_required")
    if any(token in raw_text for token in ("schema", "payload", "contrato de datos", "json")):
        _add("confirmed_structured_payloads", "approved_payload_mentions_structured_contracts")
    if any(
        token in raw_text
        for token in ("runtime", "stack", "deploy", "despliegue", "docker", "kubernetes", "render", "vercel")
    ):
        _add("confirmed_runtime_stack", "approved_payload_mentions_runtime_stack")
    return ConfirmedDeliverySignals(signals=tuple(signals), evidence=tuple(evidence))
