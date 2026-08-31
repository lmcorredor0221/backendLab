from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from app.models import ContractModel, PydanticField, utc_now


REACT_TRACE_CONTRACT_VERSION = "builder.react.trace.v1"
REACT_RUN_CONTRACT_VERSION = "builder.react.run.v1"
QUALITY_GATE_CONTRACT_VERSION = "builder.quality_gate.v1"


class BuilderQualityGateResult(ContractModel):
    contract_version: str = QUALITY_GATE_CONTRACT_VERSION
    quality_gate_version: str = "quality-gate.v1"
    stage: str = ""
    capability: str = ""
    quality_confidence: float = 0.0
    evidence_confidence: float = 0.0
    pending_resolution: int = 0
    flow_readiness: bool = False
    issues: list[str] = PydanticField(default_factory=list)
    warnings: list[str] = PydanticField(default_factory=list)
    repair_policy: Literal[
        "none",
        "react_repair",
        "document_and_delegate",
        "attention_required",
        "schema_retry",
        "language_repair",
    ] = "none"
    language_status: Literal["not_checked", "ok", "mismatch"] = "not_checked"
    schema_status: Literal["not_checked", "valid", "invalid"] = "not_checked"
    reason_summary: str = ""
    should_repair: bool = False
    blocking: bool = False


class BuilderAgentRunRequest(ContractModel):
    contract_version: str = REACT_RUN_CONTRACT_VERSION
    run_id: UUID = PydanticField(default_factory=uuid4)
    session_id: UUID | None = None
    workspace_id: UUID | None = None
    stage: str
    capability: str
    mode: Literal["dry_run", "run", "resume"] = "run"
    checkpoint_id: str = ""
    feature_flag_key: str = "react_runtime_v1"
    context_refs: list[str] = PydanticField(default_factory=list)


class BuilderAgentState(ContractModel):
    contract_version: str = REACT_RUN_CONTRACT_VERSION
    run_id: UUID
    session_id: UUID | None = None
    workspace_id: UUID | None = None
    stage: str
    capability: str
    status: Literal["pending", "running", "waiting_human", "completed", "failed", "cancelled"] = "pending"
    iteration: int = 0
    llm_calls: int = 0
    token_usage: int = 0
    quality_repair_cycles: int = 0
    quality_gate: dict[str, Any] = PydanticField(default_factory=dict)
    last_action: str = ""
    last_observation: dict[str, Any] = PydanticField(default_factory=dict)
    last_evaluation: dict[str, Any] = PydanticField(default_factory=dict)
    context_refs: list[str] = PydanticField(default_factory=list)
    checkpoint_id: str = ""
    resume_action: str = ""
    resume_scope: str = "stage"
    started_at: datetime = PydanticField(default_factory=utc_now)
    updated_at: datetime = PydanticField(default_factory=utc_now)


class BuilderActionRequest(ContractModel):
    key: str
    stage: str
    capability: str = ""
    arguments: dict[str, Any] = PydanticField(default_factory=dict)
    idempotency_key: str = ""
    side_effect: bool = False


class BuilderActionResult(ContractModel):
    key: str
    status: Literal["success", "retryable", "failed", "waiting_human"] = "success"
    output: dict[str, Any] = PydanticField(default_factory=dict)
    summary: str = ""
    warnings: list[str] = PydanticField(default_factory=list)
    error_kind: str = ""
    token_usage: int = 0
    side_effect_applied: bool = False


class BuilderObservation(ContractModel):
    contract_version: str = REACT_TRACE_CONTRACT_VERSION
    action_key: str
    status: str
    summary: str = ""
    output_refs: list[str] = PydanticField(default_factory=list)
    warnings: list[str] = PydanticField(default_factory=list)
    error_kind: str = ""
    token_usage: int = 0


class BuilderEvaluation(ContractModel):
    contract_version: str = REACT_TRACE_CONTRACT_VERSION
    status: Literal["continue", "finish", "waiting_human", "fail"]
    reason_summary: str = ""
    confidence: float = 0.0
    issues: list[str] = PydanticField(default_factory=list)
    next_action: str = ""
    quality_gate: BuilderQualityGateResult | None = None


class BuilderIterationTrace(ContractModel):
    contract_version: str = REACT_TRACE_CONTRACT_VERSION
    iteration_id: str
    iteration: int
    reason_summary: str
    action: BuilderActionRequest
    observation: BuilderObservation
    evaluation: BuilderEvaluation
    duration_ms: int = 0
    created_at: datetime = PydanticField(default_factory=utc_now)


class BuilderAgentRunResult(ContractModel):
    contract_version: str = REACT_RUN_CONTRACT_VERSION
    run_id: UUID
    status: Literal["completed", "waiting_human", "failed", "cancelled"]
    state: BuilderAgentState
    traces: list[BuilderIterationTrace] = PydanticField(default_factory=list)
    output: dict[str, Any] = PydanticField(default_factory=dict)
    checkpoint_id: str = ""
    message: str = ""
