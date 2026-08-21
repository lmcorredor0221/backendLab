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
    BuilderActionRequest,
    BuilderActionResult,
    BuilderAgentRunRequest,
    BuilderAgentRunResult,
    BuilderAgentState,
)
from app.services.agentic_runtime.controller import BuilderReActController
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
) -> DefineReactExecution:
    """Runs Define through ReAct while delegating generation to the current skill runtime."""

    definition: RequirementsDefinitionOutput | None = None
    traces: list[SkillExecutionTrace] = []
    warnings: list[str] = []

    def execute(action: BuilderActionRequest, state: BuilderAgentState) -> BuilderActionResult:
        nonlocal definition, traces, warnings
        if action.key == "retrieve_context":
            refs = [item.key for item in stage_context.approved_refs]
            refs.extend(item.key for item in stage_context.retrieved_hits)
            if not refs:
                refs = ["session.discovery", "session.canvas", "knowledge.requirements_definition"]
            return BuilderActionResult(
                key=action.key,
                output={"output_refs": refs},
                summary="Contexto aprobado de Discover y Canvas recuperado.",
            )
        if action.key == "invoke_capability":
            try:
                definition, traces = run_definition_stage(
                    discovery,
                    canvas,
                    runtime_settings=runtime_settings,
                    stage_context=stage_context,
                )
            except Exception as exc:  # noqa: BLE001
                can_auto_retry = state.llm_calls < 1
                return BuilderActionResult(
                    key=action.key,
                    status="retryable",
                    output={
                        "issues": [f"No se pudo generar Define: {type(exc).__name__}"],
                        "blocking": True,
                        "can_auto_retry": can_auto_retry,
                        "retry_attempt": state.llm_calls + 1,
                    },
                    summary=(
                        "La capability Define fallo; el controlador intentara un reintento gobernado."
                        if can_auto_retry
                        else "La capability Define fallo despues del reintento y requiere recuperacion guiada."
                    ),
                    error_kind="provider_or_schema_failure",
                )
            warnings = list(dict.fromkeys(item for trace in traces for item in trace.warnings))
            token_usage = 0
            for trace in traces:
                llm_trace = trace.llm_trace
                usage = getattr(llm_trace, "context_stats", {}) if llm_trace is not None else {}
                token_usage += int(usage.get("total_tokens", 0) or 0) if isinstance(usage, dict) else 0
            return BuilderActionResult(
                key=action.key,
                output={"definition": definition.model_dump(mode="json")},
                summary="Define genero una propuesta estructurada sobre el contexto aprobado.",
                warnings=warnings,
                token_usage=token_usage,
            )
        if action.key == "run_validator":
            if definition is None:
                return BuilderActionResult(
                    key=action.key,
                    status="failed",
                    summary="No existe una propuesta Definition para validar.",
                    error_kind="missing_definition_output",
                )
            validated = validate_definition_artifact(definition)
            issues = list(validated.validation.blocking_issues)
            return BuilderActionResult(
                key=action.key,
                output={"issues": issues, "blocking": bool(issues)},
                summary=(
                    "La propuesta Define no tiene bloqueos estructurales."
                    if not issues
                    else "La propuesta Define requiere una decision humana sobre informacion faltante."
                ),
            )
        if action.key == "repair_structured_output":
            if definition is None:
                return BuilderActionResult(
                    key=action.key,
                    output={
                        "issues": ["No existe una propuesta Definition reparable."],
                        "blocking": True,
                    },
                    summary="No existe una propuesta Definition reparable; se requiere revision humana.",
                    error_kind="missing_definition_output",
                )
            definition = validate_definition_artifact(
                RequirementsDefinitionOutput.model_validate(definition.model_dump(mode="json"))
            )
            return BuilderActionResult(
                key=action.key,
                output={"definition": definition.model_dump(mode="json")},
                summary="La salida estructurada fue auto-remediada y normalizada por el agente ReAct.",
            )
        if action.key == "create_attention_decision":
            issues = list(definition.validation.blocking_issues) if definition is not None else ["definition_output_missing"]
            return BuilderActionResult(
                key=action.key,
                output={"issues": issues, "blocking": True, "output_refs": ["attention.define_requirements"]},
                summary="Se creo una decision HITL para resolver bloqueos de Definition.",
            )
        if action.key == "persist_stage_artifact":
            return BuilderActionResult(
                key=action.key,
                summary="La persistencia queda a cargo del endpoint transaccional existente.",
                side_effect_applied=False,
            )
        if action.key == "checkpoint":
            return BuilderActionResult(
                key=action.key,
                summary="Checkpoint ReAct preparado para persistencia transaccional.",
                side_effect_applied=False,
            )
        if action.key == "finish_stage":
            return BuilderActionResult(
                key=action.key,
                summary="La etapa Define puede cerrar o quedar pausada según sus validaciones.",
                side_effect_applied=False,
            )
        return BuilderActionResult(
            key=action.key,
            status="failed",
            summary="Accion no soportada por el piloto Define.",
            error_kind="unsupported_define_action",
        )

    controller = BuilderReActController()
    request = BuilderAgentRunRequest(
        session_id=session_id,
        workspace_id=workspace_id,
        stage="define",
        capability="define_requirements",
        mode="resume" if initial_state is not None else "run",
        checkpoint_id=initial_state.checkpoint_id if initial_state is not None else "",
        context_refs=["session.discovery", "session.canvas", "knowledge.requirements_definition"],
    )
    react_run = controller.run(request, execute, initial_state=initial_state)
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
        skill_traces=traces,
        react_run=react_run,
        warnings=warnings,
    )
