from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from app.models import (
    CanvasArtifact,
    DiscoveryArtifact,
    LLMRuntimeSettings,
)
from app.services.agentic_runtime.contracts import (
    BuilderAgentRunResult,
    BuilderAgentState,
)
from app.services.agentic_runtime.stages.pipeline import ReactCapabilityOutput, run_react_stage
from app.services.llm_runtime.builder_contracts import RequirementsDefinitionOutput
from app.services.llm_runtime.stage_context_types import StageContextBundle
from app.services.skill_runtime import SkillExecutionTrace, run_definition_stage, validate_definition_artifact


@dataclass
class DefineReactExecution:
    definition: RequirementsDefinitionOutput
    skill_traces: list[SkillExecutionTrace] = field(default_factory=list)
    react_run: BuilderAgentRunResult | None = None
    warnings: list[str] = field(default_factory=list)


def run_define_react(
    *,
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact,
    runtime_settings: LLMRuntimeSettings,
    stage_context: StageContextBundle,
    session_id: UUID,
    workspace_id: UUID,
    initial_state: BuilderAgentState | None = None,
    answer_inference_enabled: bool = False,
    product_mode: str = "basic_free",
) -> DefineReactExecution:
    """Runs Define through ReAct while delegating generation to the current skill runtime."""

    context_refs = [item.key for item in stage_context.approved_refs]
    context_refs.extend(item.key for item in stage_context.retrieved_hits)
    if not context_refs:
        context_refs = ["session.discovery", "session.canvas", "knowledge.requirements_definition"]

    def run_definition_capability() -> ReactCapabilityOutput:
        definition, traces = run_definition_stage(
            discovery,
            canvas,
            runtime_settings=runtime_settings,
            stage_context=stage_context,
        )
        token_usage = 0
        for trace in traces:
            llm_trace = trace.llm_trace
            usage = getattr(llm_trace, "context_stats", {}) if llm_trace is not None else {}
            token_usage += int(usage.get("total_tokens", 0) or 0) if isinstance(usage, dict) else 0
        return ReactCapabilityOutput(
            value=definition,
            traces=traces,
            warnings=list(dict.fromkeys(item for trace in traces for item in trace.warnings)),
            summary="Define genero una propuesta estructurada sobre el contexto aprobado.",
            token_usage=token_usage,
        )

    def validate_definition(value: object) -> tuple[list[str], bool, str]:
        if not isinstance(value, RequirementsDefinitionOutput):
            return ["Define no produjo una propuesta estructurada."], True, "Define requiere revision."
        validated = validate_definition_artifact(value)
        issues = list(validated.validation.blocking_issues)
        return (
            issues,
            bool(issues),
            "La propuesta Define no tiene bloqueos estructurales."
            if not issues
            else "La propuesta Define requiere una decision humana sobre informacion faltante.",
        )

    react_execution = run_react_stage(
        stage="define",
        capability="define_requirements",
        session_id=session_id,
        workspace_id=workspace_id,
        context_refs=context_refs,
        primary_runner=run_definition_capability,
        validator=validate_definition,
        initial_state=initial_state,
        effective_language=stage_context.effective_language,
        answer_inference_enabled=answer_inference_enabled,
        product_mode=product_mode,
    )
    definition = react_execution.value
    react_run = react_execution.react_run
    if definition is None and react_run is not None and react_run.status == "waiting_human":
        issues = [str(item) for item in react_run.output.get("issues", []) if str(item).strip()]
        definition = RequirementsDefinitionOutput(
            summary="Define requiere una decision humana antes de generar una propuesta completa.",
            validation={
                "blocking_issues": issues or ["El proveedor o schema no permitio generar Definition."],
            },
            canvas_projection=canvas,
        )
    if definition is None:
        raise RuntimeError("El piloto ReAct de Define no produjo una propuesta estructurada.")
    return DefineReactExecution(
        definition=definition,
        skill_traces=react_execution.traces,
        react_run=react_run,
        warnings=react_execution.warnings,
    )
