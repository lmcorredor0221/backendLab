from __future__ import annotations

from dataclasses import dataclass


CANONICAL_ARTIFACT_LIFECYCLE_STATES = (
    "generated",
    "reviewed",
    "approved",
    "stale",
)


@dataclass(frozen=True)
class JourneyStageBoundary:
    stage_key: str
    label: str
    required_predecessors: tuple[str, ...]
    current_source_actions: tuple[str, ...]
    future_source_actions: tuple[str, ...]
    owns_blueprint_sections: tuple[str, ...]
    transition_contract: str
    notes: tuple[str, ...] = ()
    governing_feature_flags: tuple[str, ...] = ()


DISCOVER_STAGE_BOUNDARY = JourneyStageBoundary(
    stage_key="discover",
    label="Discover",
    required_predecessors=(),
    current_source_actions=(
        "create_session",
        "load_session_snapshot",
        "update_commercial_tier",
        "draft_capture",
        "input_validation",
        "normalize_discovery",
    ),
    future_source_actions=("analyze_discovery",),
    owns_blueprint_sections=(),
    transition_contract="Produce y versiona discovery normalizado como base del journey.",
    notes=(
        "Las acciones operativas de inicio de sesion se anclan a Discover para mantener el primer corte del journey.",
    ),
)

DEFINE_STAGE_BOUNDARY = JourneyStageBoundary(
    stage_key="define",
    label="Define",
    required_predecessors=("discover",),
    current_source_actions=("build_canvas",),
    future_source_actions=("define_requirements",),
    owns_blueprint_sections=(),
    transition_contract="Consolida canvas, alcance MVP, objetivos y riesgos primarios antes de diseno.",
)

DESIGN_STAGE_BOUNDARY = JourneyStageBoundary(
    stage_key="design",
    label="Design",
    required_predecessors=("discover", "define"),
    current_source_actions=("propose_design", "build_blueprint", "enrich_blueprint", "manual_patch"),
    future_source_actions=("propose_agent_design", "critique_agent_design"),
    owns_blueprint_sections=("architecture", "reasoning_pattern", "safety_checks", "guardrails", "narrative"),
    transition_contract=(
        "Fija arquitectura, patron cognitivo, seguridad y narrativa base. "
        "Design no es propietario canonico de Tools ni Memory."
    ),
    notes=(
        "propose_design genera el artefacto comparativo aprobable de CI7 antes de proyectar al blueprint.",
        "build_blueprint puede sembrar informacion transitoria para Tools y Memory, pero esa informacion no es canonica hasta pasar por sus etapas propietarias.",
        "manual_patch sigue existiendo como ajuste transversal, pero no cambia el ownership de secciones.",
    ),
)

TOOLS_STAGE_BOUNDARY = JourneyStageBoundary(
    stage_key="tools",
    label="Tools",
    required_predecessors=("discover", "define", "design"),
    current_source_actions=("recommend_tools", "approve_tools_selection"),
    future_source_actions=("recommend_tools", "approve_tools_selection"),
    owns_blueprint_sections=("tools",),
    transition_contract=(
        "Se invoca despues de Design sobre contexto aprobado. "
        "Produce recomendacion minima de herramientas, decision report versionado y seleccion aprobada para blueprint.tools."
    ),
    notes=(
        "No debe depender de reglas genericas embebidas en build_blueprint.",
        "Solo se promueven tools aprobadas por el usuario o por una politica visible de gobierno.",
    ),
    governing_feature_flags=("tool_recommendation_llm_v1",),
)

MEMORY_STAGE_BOUNDARY = JourneyStageBoundary(
    stage_key="memory",
    label="Memory",
    required_predecessors=("discover", "define", "design", "tools"),
    current_source_actions=(
        "recommend_memory",
        "approve_memory_profile",
        "load_short_term_memory",
        "reload_short_term_memory",
        "rollback_short_term_memory",
    ),
    future_source_actions=(
        "recommend_memory_architecture",
        "critique_memory_architecture",
        "approve_memory_profile",
    ),
    owns_blueprint_sections=("memory_strategy", "memory_profile", "knowledge_profile"),
    transition_contract=(
        "Se invoca despues de Tools y consume solo el set aprobado de herramientas. "
        "Produce memoria operativa y conocimiento sin reenviar contexto redundante."
    ),
    notes=(
        "La entrada canonica de Memory debe derivarse del blueprint aprobado y del digest de tools.",
        "Las acciones de short-term memory pertenecen a la memoria del builder, no a la memoria del agente objetivo; se asignan aqui para no dejarlas fuera del mapa del journey.",
    ),
    governing_feature_flags=(
        "memory_hybrid_define_design_v1",
        "memory_hybrid_extended_journey_v1",
    ),
)

VALIDATE_STAGE_BOUNDARY = JourneyStageBoundary(
    stage_key="validate",
    label="Validate",
    required_predecessors=("discover", "define", "design", "tools", "memory"),
    current_source_actions=(
        "apply_workflow_template",
        "bootstrap_evaluation_workbench",
        "bootstrap_dataset",
        "bootstrap_rubric",
        "update_evaluation_dataset",
        "manual_dataset_edit",
        "update_evaluation_rubric",
        "manual_rubric_edit",
        "evaluate_blueprint",
        "generate_validation_scenarios",
        "approve_validation_scenarios",
        "run_validation_simulation",
        "inject_validation_event",
        "judge_validation_run",
        "resolve_approval",
        "resolve_handoff",
        "run_subagent:{run_kind}",
        "rerun:{skill_key}",
        "load_monitoring",
        "load_integrations",
        "check_integrations",
        "update_feature_flag",
    ),
    future_source_actions=(
        "generate_validation_scenarios",
        "simulate_validation_scenario",
        "judge_validation_run",
    ),
    owns_blueprint_sections=(),
    transition_contract="Evalua cobertura, readiness y gobierno del agente antes de estimar o exportar.",
    notes=(
        "Validate concentra evaluacion, governance, handoffs y simulaciones antes de Estimate.",
        "Las corridas de subagentes y reruns quedan ancladas a Validate como plano de control del review final.",
    ),
    governing_feature_flags=(
        "workflow_templates_v1",
        "governance_console_v1",
        "specialized_subagents_v1",
        "multi_agent_runtime_v1",
    ),
)

ESTIMATE_STAGE_BOUNDARY = JourneyStageBoundary(
    stage_key="estimate",
    label="Estimate",
    required_predecessors=("discover", "define", "design", "tools", "memory", "validate"),
    current_source_actions=(
        "generate_estimation_report",
        "analyze_estimation_risks",
        "apply_estimation_analysis_decision",
        "upsert_estimation_actuals",
    ),
    future_source_actions=(),
    owns_blueprint_sections=(),
    transition_contract="Calcula esfuerzo, costo, tiempo, confianza y ROI sobre artefactos aprobados y vigentes.",
    notes=(
        "Estimate produce un artefacto canonico independiente y no reclama ownership de secciones del blueprint.",
    ),
    governing_feature_flags=("estimation_comparative_v1",),
)

BUILD_STAGE_BOUNDARY = JourneyStageBoundary(
    stage_key="build",
    label="Build",
    required_predecessors=("discover", "define", "design", "tools", "memory", "validate", "estimate"),
    current_source_actions=(
        "generate_acp_preview",
        "export_acp_zip",
        "export_markdown",
        "export_json",
        "export_{contract_key}",
    ),
    future_source_actions=(),
    owns_blueprint_sections=(),
    transition_contract="Empaqueta exportes, ACP y artefactos de construccion listos para handoff usando solo artefactos aprobados.",
    notes=(
        "El UI usa el nombre comercial Package; el key interno build se mantiene por compatibilidad del runtime actual.",
    ),
)

_STAGE_BOUNDARIES = {
    item.stage_key: item
    for item in (
        DISCOVER_STAGE_BOUNDARY,
        DEFINE_STAGE_BOUNDARY,
        DESIGN_STAGE_BOUNDARY,
        TOOLS_STAGE_BOUNDARY,
        MEMORY_STAGE_BOUNDARY,
        VALIDATE_STAGE_BOUNDARY,
        ESTIMATE_STAGE_BOUNDARY,
        BUILD_STAGE_BOUNDARY,
    )
}


def list_journey_stage_boundaries() -> tuple[JourneyStageBoundary, ...]:
    return tuple(_STAGE_BOUNDARIES.values())


def get_journey_stage_boundary(stage_key: str) -> JourneyStageBoundary:
    normalized = stage_key.strip().lower()
    if normalized not in _STAGE_BOUNDARIES:
        raise KeyError(f"Unknown journey stage boundary: {stage_key}")
    return _STAGE_BOUNDARIES[normalized]


def journey_stage_for_source_action(source_action: str) -> tuple[str, str] | None:
    normalized = source_action.strip().lower()

    if normalized.startswith("rerun:"):
        rerun_target = normalized.split(":", 1)[1].strip()
        if rerun_target and "{" not in rerun_target:
            mapped = journey_stage_for_source_action(rerun_target)
            if mapped is not None:
                return mapped
        return VALIDATE_STAGE_BOUNDARY.stage_key, VALIDATE_STAGE_BOUNDARY.label

    if normalized.startswith("run_subagent:"):
        return VALIDATE_STAGE_BOUNDARY.stage_key, VALIDATE_STAGE_BOUNDARY.label

    if normalized.startswith(("journey_create:", "journey_patch:", "journey_approve:", "journey_reject:")):
        raw_stage_key = normalized.split(":", 1)[1].strip()
        if raw_stage_key in _STAGE_BOUNDARIES:
            boundary = get_journey_stage_boundary(raw_stage_key)
            return boundary.stage_key, boundary.label
        if raw_stage_key.startswith("{") and raw_stage_key.endswith("}"):
            return VALIDATE_STAGE_BOUNDARY.stage_key, VALIDATE_STAGE_BOUNDARY.label

    for boundary in list_journey_stage_boundaries():
        if normalized in boundary.current_source_actions or normalized in boundary.future_source_actions:
            return boundary.stage_key, boundary.label

    if normalized.startswith("generate_acp") or normalized.startswith("export_"):
        return BUILD_STAGE_BOUNDARY.stage_key, BUILD_STAGE_BOUNDARY.label

    return None
