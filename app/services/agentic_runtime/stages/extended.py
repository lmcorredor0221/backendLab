from __future__ import annotations

from typing import Any, Callable
from uuid import UUID

from app.models import (
    ApprovedToolsDigest,
    BlueprintArtifact,
    CanvasArtifact,
    DiscoveryArtifact,
    DesignRecommendationArtifact,
    EvaluationDatasetArtifact,
    EvaluationRubricArtifact,
    LLMRuntimeSettings,
    MemoryRecommendationArtifact,
    SessionSnapshot,
    ToolRecommendationArtifact,
    SimulationSpecificationArtifact,
)
from app.services.llm_runtime.builder_contracts import RequirementsDefinitionOutput
from app.services.agentic_runtime.cross_stage_evaluator import evaluate_tools_memory_compatibility
from app.services.agentic_runtime.stages.pipeline import (
    CapabilityRunner,
    ReactCapabilityOutput,
    ReactStageExecution,
    Validator,
    run_react_stage,
)
from app.services.skill_runtime import (
    run_design_stage,
    run_evaluation_stage,
    run_memory_recommendation_stage,
    run_tool_recommendation_stage,
)
from app.services.validation_simulation_service import build_validation_simulation_specification


def _validate_design(value: Any) -> tuple[list[str], bool, str]:
    if not isinstance(value, DesignRecommendationArtifact):
        return ["Design no produjo un artefacto estructurado."], True, "La propuesta de Design no es valida."
    issues = [item.title for item in value.critic_findings if str(item.severity).lower() == "blocking" and item.title]
    if not value.alternatives or not value.recommended_alternative_key:
        issues.append("Design debe producir alternativas y una recomendacion explicita.")
    return list(dict.fromkeys(issues)), bool(issues), "Design tiene alternativas, critica y recomendacion trazable." if not issues else "Design requiere revision antes de promoverse."


def run_design_react(
    *,
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact,
    definition_artifact: RequirementsDefinitionOutput,
    instructions: str,
    runtime_settings: LLMRuntimeSettings,
    proposal_stage_context: Any,
    critique_stage_context: Any,
    session_id: UUID,
    workspace_id: UUID,
    initial_state: Any = None,
    progress_callback: Callable[[str, str], None] | None = None,
) -> ReactStageExecution:
    return run_react_stage(
        stage="design",
        capability="propose_agent_design",
        secondary_capability="critique_agent_design",
        session_id=session_id,
        workspace_id=workspace_id,
        context_refs=[
            "session.discovery",
            "session.canvas",
            "session.journey_latest_artifacts.define",
            "knowledge.agent_design",
        ],
        primary_runner=lambda: _design_run(
            discovery,
            canvas,
            definition_artifact,
            instructions,
            runtime_settings,
            proposal_stage_context,
            critique_stage_context,
            progress_callback,
        ),
        secondary_runner=lambda value: ReactCapabilityOutput(
            value=value,
            summary="La critica de Design ya fue ejecutada por el skill especializado y queda expuesta para validacion ReAct.",
        ),
        validator=_validate_design,
        initial_state=initial_state,
    )


def _design_run(discovery: DiscoveryArtifact, canvas: CanvasArtifact, definition: RequirementsDefinitionOutput, instructions: str, settings: LLMRuntimeSettings, proposal_context: Any, critique_context: Any, progress_callback: Callable[[str, str], None] | None = None) -> ReactCapabilityOutput:
    artifact, traces = run_design_stage(
        discovery,
        canvas,
        definition,
        instructions=instructions,
        runtime_settings=settings,
        proposal_stage_context=proposal_context,
        critique_stage_context=critique_context,
        progress_callback=progress_callback,
    )
    return ReactCapabilityOutput(
        value=artifact,
        traces=traces,
        warnings=[warning for trace in traces for warning in trace.warnings],
        summary="Design genero alternativas y ejecuto una critica de arquitectura y comportamiento.",
    )


def _validate_tools(value: Any) -> tuple[list[str], bool, str]:
    if not isinstance(value, ToolRecommendationArtifact):
        return ["Tools no produjo un artefacto estructurado."], True, "La recomendacion de Tools no es valida."
    from app.services.tool_recommendation_service import evaluate_tool_recommendation_artifact
    re_evaluated = evaluate_tool_recommendation_artifact(value)
    blocking_findings = [item.title for item in re_evaluated.evaluation.findings if item.severity == "blocking" and item.title]
    is_blocked = bool(blocking_findings)
    return list(dict.fromkeys(blocking_findings)), is_blocked, "Tools clasifico capacidades y valido minimalidad." if not is_blocked else "Tools requiere resolver una dependencia o decision."


def run_tools_react(*, session_id: UUID, workspace_id: UUID, discovery: DiscoveryArtifact, canvas: CanvasArtifact, blueprint: BlueprintArtifact, definition_artifact: RequirementsDefinitionOutput, design_artifact: DesignRecommendationArtifact, instructions: str, blueprint_version_number: int | None, runtime_settings: LLMRuntimeSettings, stage_context: Any, initial_state: Any = None) -> ReactStageExecution:
    return run_react_stage(
        stage="tools",
        capability="recommend_minimal_tools",
        session_id=session_id,
        workspace_id=workspace_id,
        context_refs=["session.discovery", "session.canvas", "session.journey_latest_artifacts.define", "session.journey_latest_artifacts.design", "knowledge.tool_catalog"],
        primary_runner=lambda: _tools_run(session_id, discovery, canvas, blueprint, definition_artifact, design_artifact, instructions, blueprint_version_number, runtime_settings, stage_context),
        validator=_validate_tools,
        remediation_action="raise_cross_stage_remediation",
        initial_state=initial_state,
    )


def _tools_run(session_id: UUID, discovery: DiscoveryArtifact, canvas: CanvasArtifact, blueprint: BlueprintArtifact, definition: RequirementsDefinitionOutput, design: DesignRecommendationArtifact, instructions: str, version: int | None, settings: LLMRuntimeSettings, context: Any) -> ReactCapabilityOutput:
    envelope, traces = run_tool_recommendation_stage(session_id, discovery, canvas, blueprint, definition_artifact=definition, design_artifact=design, instructions=instructions, blueprint_version_number=version, runtime_settings=settings, stage_context=context)
    return ReactCapabilityOutput(value=envelope.data, traces=traces, warnings=list(envelope.warnings), summary="Tools propuso el conjunto minimo y justifico cada capacidad.")


def _validate_memory(value: Any, tools: ToolRecommendationArtifact | None) -> tuple[list[str], bool, str]:
    if not isinstance(value, MemoryRecommendationArtifact):
        return ["Memory no produjo un artefacto estructurado."], True, "La recomendacion de Memory no es valida."
    from app.services.memory_recommendation_service import auto_reconcile_memory_artifact
    reconciled = auto_reconcile_memory_artifact(value)
    issues, blocking, summary = evaluate_tools_memory_compatibility(tools, reconciled)
    findings = [item.title for item in reconciled.critic_findings if item.severity == "blocking" and item.title]
    return list(dict.fromkeys([*issues, *findings])), bool(blocking or findings), summary if not findings else "Memory requiere resolver una dependencia de Tools."


def run_memory_react(*, session_id: UUID, workspace_id: UUID, discovery: DiscoveryArtifact, canvas: CanvasArtifact, blueprint: BlueprintArtifact, definition_artifact: RequirementsDefinitionOutput | None, design_artifact: DesignRecommendationArtifact | None, approved_tools_digest: ApprovedToolsDigest, tools_artifact: ToolRecommendationArtifact | None, session_snapshot: SessionSnapshot, instructions: str, blueprint_version_number: int | None, source_stage_versions: Any, runtime_settings: LLMRuntimeSettings, proposal_stage_context: Any, critique_stage_context: Any, initial_state: Any = None) -> ReactStageExecution:
    return run_react_stage(
        stage="memory",
        capability="recommend_memory_architecture",
        secondary_capability="critique_memory_architecture",
        session_id=session_id,
        workspace_id=workspace_id,
        context_refs=["session.discovery", "session.canvas", "session.journey_latest_artifacts.define", "session.journey_latest_artifacts.design", "session.journey_latest_artifacts.tools", "knowledge.memory_strategy"],
        primary_runner=lambda: _memory_run(session_id, discovery, canvas, blueprint, definition_artifact, design_artifact, approved_tools_digest, session_snapshot, instructions, blueprint_version_number, source_stage_versions, runtime_settings, proposal_stage_context, critique_stage_context),
        secondary_runner=lambda value: ReactCapabilityOutput(value=value, summary="La critica de Memory fue ejecutada por el skill especializado y queda gobernada por la evaluacion cruzada Tools/Memory."),
        validator=lambda value: _validate_memory(value, tools_artifact),
        remediation_action="raise_cross_stage_remediation",
        initial_state=initial_state,
    )


def _memory_run(session_id: UUID, discovery: DiscoveryArtifact, canvas: CanvasArtifact, blueprint: BlueprintArtifact, definition: RequirementsDefinitionOutput | None, design: DesignRecommendationArtifact | None, digest: ApprovedToolsDigest, snapshot: SessionSnapshot, instructions: str, version: int | None, versions: Any, settings: LLMRuntimeSettings, proposal_context: Any, critique_context: Any) -> ReactCapabilityOutput:
    artifact, traces = run_memory_recommendation_stage(session_id=session_id, discovery=discovery, canvas=canvas, blueprint=blueprint, definition_artifact=definition, design_artifact=design, approved_tools_digest=digest, session_snapshot=snapshot, instructions=instructions, blueprint_version_number=version, source_stage_versions=versions, runtime_settings=settings, proposal_stage_context=proposal_context, critique_stage_context=critique_context)
    return ReactCapabilityOutput(value=artifact, traces=traces, warnings=[warning for trace in traces for warning in trace.warnings], summary="Memory genero propuesta de corto, largo plazo y conocimiento con dependencia Tools visible.")


def run_evaluation_react(*, session_id: UUID, workspace_id: UUID, discovery: DiscoveryArtifact | None, canvas: CanvasArtifact | None, blueprint: BlueprintArtifact | None, dataset: EvaluationDatasetArtifact | None, rubric: EvaluationRubricArtifact | None, runtime_settings: LLMRuntimeSettings, initial_state: Any = None) -> ReactStageExecution:
    return run_react_stage(
        stage="validate",
        capability="generate_validation_scenarios",
        session_id=session_id,
        workspace_id=workspace_id,
        context_refs=["session.journey_latest_artifacts.memory", "session.blueprint", "knowledge.evaluation_rubric"],
        primary_runner=lambda: _evaluation_run(discovery, canvas, blueprint, dataset, rubric, runtime_settings),
        validator=lambda value: _validate_evaluation(value),
        initial_state=initial_state,
    )


def _evaluation_run(discovery: Any, canvas: Any, blueprint: Any, dataset: Any, rubric: Any, settings: Any) -> ReactCapabilityOutput:
    envelope, traces = run_evaluation_stage(discovery, canvas, blueprint, dataset, rubric, runtime_settings=settings)
    return ReactCapabilityOutput(value=envelope, traces=traces, warnings=list(envelope.warnings), summary="Validate genero casos y evaluo cobertura del blueprint.")


def _validate_evaluation(value: Any) -> tuple[list[str], bool, str]:
    if value is None:
        return ["Validate no produjo un resultado."], True, "Validate requiere revision."
    data = getattr(value, "data", value)
    gaps = list(getattr(data, "gaps", []) or [])
    return [str(item) for item in gaps], bool(gaps), "Validate genero escenarios trazables." if not gaps else "Validate tiene gaps que requieren atencion."


def run_validation_spec_react(*, session_id: UUID, workspace_id: UUID, discovery: DiscoveryArtifact | None, canvas: CanvasArtifact | None, blueprint: BlueprintArtifact, definition_artifact: RequirementsDefinitionOutput | None, session_snapshot: SessionSnapshot, blueprint_version_number: int | None, source_stage_versions: dict[str, int | None], instructions: str, runtime_settings: LLMRuntimeSettings, stage_context: Any, initial_state: Any = None) -> ReactStageExecution:
    def run() -> ReactCapabilityOutput:
        artifact, traces = build_validation_simulation_specification(
            discovery=discovery,
            canvas=canvas,
            blueprint=blueprint,
            definition_artifact=definition_artifact,
            session_snapshot=session_snapshot,
            blueprint_version_number=blueprint_version_number,
            source_stage_versions=source_stage_versions,
            instructions=instructions,
            runtime_settings=runtime_settings,
            stage_context=stage_context,
        )
        return ReactCapabilityOutput(value=artifact, traces=traces, warnings=list(artifact.warnings), summary="Validate genero escenarios, fallos representativos y criterios de simulacion.")

    def validate(value: Any) -> tuple[list[str], bool, str]:
        if not isinstance(value, SimulationSpecificationArtifact):
            return ["Validate no produjo una especificacion de escenarios."], True, "La especificacion requiere revision."
        issues = list(value.coverage_gaps) + list(value.missing_information)
        return issues, bool(issues), "Validate produjo escenarios trazables." if not issues else "Validate tiene cobertura pendiente."

    return run_react_stage(
        stage="validate",
        capability="generate_validation_scenarios",
        session_id=session_id,
        workspace_id=workspace_id,
        context_refs=["session.journey_latest_artifacts.memory", "session.blueprint", "knowledge.evaluation_rubric"],
        primary_runner=run,
        validator=validate,
        initial_state=initial_state,
    )


def run_callable_react(*, stage: str, capability: str, session_id: UUID, workspace_id: UUID, context_refs: list[str], runner: CapabilityRunner, validator: Validator, initial_state: Any = None) -> ReactStageExecution:
    return run_react_stage(
        stage=stage,
        capability=capability,
        session_id=session_id,
        workspace_id=workspace_id,
        context_refs=context_refs,
        primary_runner=runner,
        validator=validator,
        initial_state=initial_state,
    )
