from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import UUID

from app.services.agentic_runtime.contracts import (
    BuilderActionRequest,
    BuilderActionResult,
    BuilderAgentRunRequest,
    BuilderAgentRunResult,
    BuilderAgentState,
)
from app.services.agentic_runtime.controller import BuilderReActController


@dataclass
class ReactCapabilityOutput:
    value: Any = None
    traces: list[Any] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: str = ""
    token_usage: int = 0


@dataclass
class ReactStageExecution:
    value: Any = None
    traces: list[Any] = field(default_factory=list)
    react_run: BuilderAgentRunResult | None = None
    warnings: list[str] = field(default_factory=list)


CapabilityRunner = Callable[[], ReactCapabilityOutput]
SecondaryRunner = Callable[[Any], ReactCapabilityOutput]
Validator = Callable[[Any], tuple[list[str], bool, str]]


def _model_payload(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    return {"value": value}


def run_react_stage(
    *,
    stage: str,
    capability: str,
    session_id: UUID,
    workspace_id: UUID,
    context_refs: list[str],
    primary_runner: CapabilityRunner,
    validator: Validator,
    secondary_capability: str = "",
    secondary_runner: SecondaryRunner | None = None,
    remediation_action: str = "",
    initial_state: BuilderAgentState | None = None,
) -> ReactStageExecution:
    """Run one bounded ReAct loop around an existing stage capability.

    The skill runtime remains the source of truth for generation. This adapter
    only governs sequencing, validation, HITL and checkpoint boundaries.
    """

    current_value: Any = None
    collected_traces: list[Any] = []
    warnings: list[str] = []

    def reasoner(request: BuilderAgentRunRequest, state: BuilderAgentState, previous: BuilderActionResult | None) -> BuilderActionRequest:
        previous_key = previous.key if previous else ""
        previous_output = previous.output if previous else {}
        if not previous_key:
            return BuilderActionRequest(key="retrieve_context", stage=stage)
        if previous is not None and previous.status == "retryable":
            if previous_key == "invoke_capability" and bool(previous_output.get("can_auto_retry", False)):
                return BuilderActionRequest(
                    key="invoke_capability",
                    stage=stage,
                    capability=capability,
                    arguments={"phase": "retry"},
                )
            if previous_key == "invoke_critique" and secondary_capability and bool(previous_output.get("can_auto_retry", False)):
                return BuilderActionRequest(
                    key="invoke_critique",
                    stage=stage,
                    capability=secondary_capability,
                    arguments={"phase": "retry"},
                )
            if bool(previous_output.get("repairable", False)):
                return BuilderActionRequest(key="repair_structured_output", stage=stage)
            return BuilderActionRequest(
                key=remediation_action or "create_attention_decision",
                stage=stage,
                arguments={
                    "issues": [str(item) for item in previous_output.get("issues", []) if str(item).strip()],
                    "blocking": bool(previous_output.get("blocking", True)),
                },
            )
        if previous_key == "retrieve_context":
            return BuilderActionRequest(
                key="invoke_capability",
                stage=stage,
                capability=capability,
                arguments={"phase": "propose"},
            )
        if previous_key == "invoke_capability" and secondary_capability and secondary_runner is not None:
            return BuilderActionRequest(
                key="invoke_critique",
                stage=stage,
                capability=secondary_capability,
                arguments={"phase": "critique"},
            )
        if previous_key in {"invoke_capability", "invoke_critique", "repair_structured_output"}:
            return BuilderActionRequest(key="run_validator", stage=stage)
        if previous_key == "run_validator":
            issues = [str(item) for item in (previous.output if previous else {}).get("issues", []) if str(item).strip()]
            blocking = bool((previous.output if previous else {}).get("blocking", False))
            if blocking or issues:
                return BuilderActionRequest(
                    key=remediation_action or "create_attention_decision",
                    stage=stage,
                    arguments={"issues": issues, "blocking": blocking},
                )
            return BuilderActionRequest(key="persist_stage_artifact", stage=stage)
        if previous_key == "repair_structured_output":
            return BuilderActionRequest(key="run_validator", stage=stage)
        if previous_key == "persist_stage_artifact":
            return BuilderActionRequest(key="finish_stage", stage=stage)
        if previous_key in {"create_attention_decision", "raise_cross_stage_remediation"}:
            return BuilderActionRequest(key="checkpoint", stage=stage)
        return BuilderActionRequest(key="finish_stage", stage=stage)

    def execute(action: BuilderActionRequest, _state: BuilderAgentState) -> BuilderActionResult:
        nonlocal current_value, collected_traces, warnings
        if action.key == "retrieve_context":
            return BuilderActionResult(
                key=action.key,
                output={"output_refs": list(context_refs)},
                summary="Contexto aprobado y memoria compacta recuperados.",
            )
        if action.key == "invoke_capability":
            try:
                result = primary_runner()
            except Exception as exc:  # noqa: BLE001
                can_auto_retry = _state.llm_calls < 1
                return BuilderActionResult(
                    key=action.key,
                    status="retryable",
                    output={
                        "issues": [f"No se pudo ejecutar {capability}: {type(exc).__name__}"],
                        "blocking": True,
                        "can_auto_retry": can_auto_retry,
                        "retry_attempt": _state.llm_calls + 1,
                    },
                    summary=(
                        f"La capability {capability} fallo; se intentara un reintento gobernado."
                        if can_auto_retry
                        else f"La capability {capability} fallo despues del reintento y requiere recuperacion guiada."
                    ),
                    error_kind="provider_or_schema_failure",
                )
            current_value = result.value
            collected_traces.extend(result.traces)
            warnings.extend(result.warnings)
            return BuilderActionResult(
                key=action.key,
                output={"artifact": _model_payload(current_value), "output_refs": list(context_refs)},
                summary=result.summary or f"Capability {capability} ejecutada.",
                warnings=list(result.warnings),
                token_usage=result.token_usage,
            )
        if action.key == "invoke_critique":
            if current_value is None:
                return BuilderActionResult(
                    key=action.key,
                    status="failed",
                    summary="No existe una propuesta para criticar.",
                    error_kind="missing_stage_output",
                )
            try:
                result = secondary_runner(current_value) if secondary_runner is not None else ReactCapabilityOutput(value=current_value)
            except Exception as exc:  # noqa: BLE001
                can_auto_retry = _state.llm_calls < 2
                return BuilderActionResult(
                    key=action.key,
                    status="retryable",
                    output={
                        "issues": [f"No se pudo ejecutar {secondary_capability}: {type(exc).__name__}"],
                        "blocking": True,
                        "can_auto_retry": can_auto_retry,
                        "retry_attempt": _state.llm_calls + 1,
                    },
                    summary=(
                        f"La critica {secondary_capability} fallo; se intentara un reintento gobernado."
                        if can_auto_retry
                        else f"La critica {secondary_capability} fallo despues del reintento y requiere recuperacion guiada."
                    ),
                    error_kind="provider_or_schema_failure",
                )
            current_value = result.value
            collected_traces.extend(result.traces)
            warnings.extend(result.warnings)
            return BuilderActionResult(
                key=action.key,
                output={"artifact": _model_payload(current_value), "output_refs": list(context_refs)},
                summary=result.summary or f"Critica {secondary_capability} completada.",
                warnings=list(result.warnings),
                token_usage=result.token_usage,
            )
        if action.key == "run_validator":
            issues, blocking, summary = validator(current_value)
            return BuilderActionResult(
                key=action.key,
                output={"issues": issues, "blocking": blocking, "output_refs": list(context_refs)},
                summary=summary,
            )
        if action.key == "repair_structured_output":
            if current_value is None:
                return BuilderActionResult(
                    key=action.key,
                    status="failed",
                    output={"issues": ["No existe una salida estructurada reparable."], "blocking": True},
                    summary="La salida requiere una decision humana.",
                    error_kind="missing_stage_output",
                )
            return BuilderActionResult(
                key=action.key,
                output={"artifact": _model_payload(current_value)},
                summary="La salida estructurada fue normalizada antes de revalidar.",
            )
        if action.key in {"create_attention_decision", "raise_cross_stage_remediation"}:
            issues, blocking, _summary = validator(current_value)
            return BuilderActionResult(
                key=action.key,
                output={"issues": issues, "blocking": blocking, "output_refs": [f"attention.{stage}"]},
                summary=(
                    "La incompatibilidad entre etapas fue derivada a una remediacion guiada."
                    if action.key == "raise_cross_stage_remediation"
                    else "La salida fue derivada a una decision guiada de Atencion."
                ),
            )
        if action.key in {"persist_stage_artifact", "finish_stage"}:
            return BuilderActionResult(
                key=action.key,
                summary="La persistencia transaccional queda a cargo del endpoint de la etapa.",
                side_effect_applied=False,
            )
        if action.key == "checkpoint":
            return BuilderActionResult(
                key=action.key,
                status="waiting_human",
                summary="Checkpoint ReAct preparado para resolver la decision de Atencion.",
                output={"issues": [f"La etapa {stage} requiere una decision humana."], "blocking": True},
            )
        return BuilderActionResult(key=action.key, status="failed", summary="Accion no soportada por el adaptador ReAct.", error_kind="unsupported_stage_action")

    request = BuilderAgentRunRequest(
        session_id=session_id,
        workspace_id=workspace_id,
        stage=stage,
        capability=capability,
        mode="resume" if initial_state is not None else "run",
        checkpoint_id=initial_state.checkpoint_id if initial_state is not None else "",
        context_refs=list(context_refs),
    )
    react_run = BuilderReActController().run(
        request,
        execute,
        reasoner=reasoner,
        initial_state=initial_state,
    )
    return ReactStageExecution(
        value=current_value,
        traces=collected_traces,
        react_run=react_run,
        warnings=list(dict.fromkeys(warnings)),
    )
