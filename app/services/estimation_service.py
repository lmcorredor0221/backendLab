from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlmodel import Session, select

from app.models import (
    ACPPreview,
    ArtifactStatus,
    AgenticEstimate,
    AutomationMatrixProfile,
    BlueprintTool,
    EstimationComplexityLevel,
    EstimationConstructionScenario,
    EstimationMaturityStage,
    EstimationPackagePolicyState,
    EstimationReportArtifact,
    EvaluationRunEntry,
    LLMProviderKey,
    LLMPricingProfile,
    RoleRateCatalogEntry,
    ReviewState,
    RuntimeCatalogEntryRecord,
    SessionSnapshot,
    TraditionalEstimate,
    WorkstreamEstimate,
    WorkstreamEffortBand,
    WorkstreamEffortProfile,
    LLMRuntimeSettings,
    utc_now,
)
from app.services.automation_matrix import WORKSTREAM_ORDER, build_automation_matrix
from app.services.estimation_confidence import build_confidence_breakdown
from app.services.llm_cost_engine import estimate_agentic_costs
from app.services.llm_runtime.runtime_settings_service import (
    load_effective_runtime_settings,
    load_platform_runtime_defaults,
)


ARCHETYPE_DISTRIBUTIONS: dict[str, dict[str, int]] = {
    "app_web_transaccional": {
        "backend": 30,
        "frontend": 26,
        "integrations": 12,
        "data": 8,
        "qa": 14,
        "devops": 10,
    },
    "enterprise_integrations": {
        "backend": 22,
        "frontend": 18,
        "integrations": 24,
        "data": 10,
        "qa": 16,
        "devops": 10,
    },
    "agentic_platform": {
        "backend": 20,
        "frontend": 16,
        "integrations": 18,
        "data": 24,
        "qa": 12,
        "devops": 10,
    },
}
COMPLEXITY_BASELINE_POINTS: dict[EstimationComplexityLevel, int] = {
    EstimationComplexityLevel.simple: 50,
    EstimationComplexityLevel.moderate: 140,
    EstimationComplexityLevel.complex: 260,
    EstimationComplexityLevel.critical: 380,
}
MATURITY_FACTORS: dict[EstimationMaturityStage, float] = {
    EstimationMaturityStage.canvas: 0.62,
    EstimationMaturityStage.blueprint: 0.82,
    EstimationMaturityStage.ready_to_build: 1.0,
}
ACTIVE_FTE_BY_COMPLEXITY: dict[EstimationComplexityLevel, float] = {
    EstimationComplexityLevel.simple: 2.2,
    EstimationComplexityLevel.moderate: 2.8,
    EstimationComplexityLevel.complex: 3.4,
    EstimationComplexityLevel.critical: 4.0,
}
INTERNAL_AGENT_TOOL_NAMES = {
    "approval_gate",
    "document_ingestion",
    "human_handoff",
    "knowledge_retrieval",
    "memory_lookup",
    "memory_write",
    "read_system_of_record",
    "transactional_write",
}
INTERNAL_AGENT_ARCHETYPES = {
    "approval_gate",
    "document_ingestion",
    "governance_gate",
    "human_handoff",
    "knowledge_retrieval",
    "read_only_lookup",
}
INTERNAL_AGENT_INTEGRATION_KINDS = {"governed_handoff", "pipeline", "retrieval"}
INTERNAL_ENDPOINT_PREFIXES = ("internal://", "knowledge://", "memory://", "workflow://")
DESIGN_PLACEHOLDER_ENDPOINT_PREFIXES = ("integration://system-of-record/",)
EXTERNAL_IMPLEMENTATION_HINTS = (
    "api externa",
    "crm",
    "erp",
    "middleware",
    "plataforma externa",
    "salesforce",
    "sap",
    "servicenow",
    "tercero",
    "third-party",
    "webhook",
    "zendesk",
)
IMPLEMENTATION_UNCERTAINTY_DOMAINS = {"deployment", "runtime", "integrations"}
IMPLEMENTATION_UNCERTAINTY_KEY_HINTS = (
    "api",
    "credential",
    "database",
    "deployment",
    "environment",
    "external",
    "infra",
    "permission",
    "provider",
    "runtime",
    "sandbox",
    "secret",
)


@dataclass(frozen=True)
class ConstructionUncertaintyCounts:
    design_gap_count: int = 0
    implementation_gap_count: int = 0
    design_open_questions: int = 0
    implementation_open_questions: int = 0


@dataclass(frozen=True)
class EstimationSignals:
    maturity_stage: EstimationMaturityStage
    complexity: EstimationComplexityLevel
    project_archetype: str
    active_provider: LLMProviderKey
    source_artifacts: list[str]
    scope_points: int
    blocking_gaps: int
    open_questions: int
    design_gap_count: int
    implementation_gap_count: int
    design_open_questions: int
    implementation_open_questions: int
    assumptions_count: int
    assumptions: list[str]
    risk_drivers: list[str]
    positive_signals: list[str]
    negative_signals: list[str]
    tool_count: int
    side_effect_tools: int
    implementation_side_effect_tools: int
    external_tool_count: int
    internal_tool_count: int
    approval_tools: int
    workflow_steps: int
    safety_checks: int
    guardrails: int
    memory_layers: int
    observability_signals: int
    evaluation_cases: int
    deliverables: int
    schema_validated_tools: int
    readiness_state: ReviewState
    api_contract_maturity: str
    deployment_maturity: str
    knowledge_maturity: str
    acp_ready_to_build: bool
    evaluation_complete: bool
    blueprint_design_coverage_percent: int
    acp_package_readiness_percent: int


def build_estimation_report(
    session: Session,
    *,
    snapshot: SessionSnapshot,
    acp_preview: ACPPreview | None = None,
) -> EstimationReportArtifact:
    role_rates = _load_role_rates(session)
    workstream_profiles = _load_workstream_profiles(session)
    automation_profiles = _load_automation_profiles(session)
    pricing_profiles = _load_pricing_profiles(session)
    confidence_bands = _load_confidence_bands(session)
    confidence_weights = _load_confidence_weights(session)
    runtime_settings = (
        load_effective_runtime_settings(session, snapshot.session.workspace_id)
        if snapshot.session.workspace_id is not None
        else load_platform_runtime_defaults(session)
    )

    signals = _build_signals(snapshot=snapshot, acp_preview=acp_preview, runtime_settings=runtime_settings)
    traditional = _build_traditional_estimate(signals, role_rates, workstream_profiles)
    agentic = _build_agentic_estimate(
        signals,
        traditional,
        runtime_settings=runtime_settings,
        role_rates=role_rates,
        workstream_profiles=workstream_profiles,
        automation_profiles=automation_profiles,
        pricing_profiles=pricing_profiles,
    )
    construction_scenarios = _build_construction_scenarios(
        traditional=traditional,
        agentic=agentic,
        signals=signals,
    )
    confidence = build_confidence_breakdown(
        signals=signals,
        snapshot=snapshot,
        confidence_bands=confidence_bands,
        confidence_weights=confidence_weights,
    )

    notes = [
        "Costos humanos expresados en COP con tarifa cargada Colombia 2026.",
        f"Proveedor activo considerado para el escenario agentic: {signals.active_provider.value}.",
        f"La matriz agentic evaluo {len(agentic.automation_assessments)} familias de entregables con reglas deterministicas.",
        f"Pricing snapshot aplicado: {agentic.pricing_policy or 'sin perfil'} | modelo {agentic.provider_model or 'n/d'}.",
    ]
    if agentic.pricing_snapshot is not None and agentic.pricing_snapshot.effective_from:
        notes.append(f"Tarifas del proveedor snapshot tomadas con vigencia {agentic.pricing_snapshot.effective_from}.")
    if agentic.provider_runtime_cost_total_usd == 0:
        notes.append("El costo variable del proveedor es cero en este corte porque el perfil vigente no trae cargos variables o falta pricing.")

    package_policy = _build_package_policy_state(signals)

    return EstimationReportArtifact(
        maturity_stage=signals.maturity_stage,
        generated_at=utc_now(),
        source_artifacts=signals.source_artifacts,
        assumptions=signals.assumptions,
        risk_drivers=signals.risk_drivers,
        traditional=traditional,
        agentic=agentic,
        confidence=confidence,
        construction_scenarios=construction_scenarios,
        package_policy=package_policy,
        notes=notes,
    )


def _build_package_policy_state(signals: EstimationSignals) -> EstimationPackagePolicyState:
    reasons: list[str] = []
    if signals.blocking_gaps > 0:
        reasons.append("Persisten blocking gaps en la continuidad constructiva.")
    if signals.maturity_stage == EstimationMaturityStage.canvas:
        reasons.append("La definición del proyecto aún se encuentra en nivel Canvas.")

    can_continue = len(reasons) == 0
    return EstimationPackagePolicyState(
        preliminary=signals.maturity_stage != EstimationMaturityStage.ready_to_build,
        can_continue_to_package=can_continue,
        package_block_reasons=reasons,
        commercial_blocked=signals.readiness_state == ReviewState.blocked,
    )


def _build_signals(
    *,
    snapshot: SessionSnapshot,
    acp_preview: ACPPreview | None,
    runtime_settings: LLMRuntimeSettings,
) -> EstimationSignals:
    maturity_stage = _infer_maturity_stage(snapshot, acp_preview)
    active_provider = runtime_settings.active_provider

    discovery = snapshot.discovery
    canvas = snapshot.canvas
    blueprint = snapshot.blueprint
    evaluation = snapshot.evaluation
    evaluation_dataset = snapshot.evaluation_dataset
    latest_run = snapshot.evaluation_runs[0] if snapshot.evaluation_runs else None

    v1_scope_count = len(discovery.mvp_definition.v1_scope) if discovery is not None else 0
    constraints_count = len(discovery.constraints) if discovery is not None else 0
    automation_opportunities = len(discovery.operational_baseline.automation_opportunities) if discovery is not None else 0
    non_delegable_count = len(discovery.mvp_definition.non_delegable_decisions) if discovery is not None else 0
    approval_count = len(canvas.agent_profile.human_approvals) if canvas is not None else 0
    tool_count = len(blueprint.tools) if blueprint is not None else 0
    tools = list(blueprint.tools if blueprint is not None else [])
    schema_validated_tools = sum(1 for item in tools if _tool_has_schema_validation(item))
    side_effect_tools = sum(1 for item in tools if item.has_side_effects)
    external_tool_count = sum(1 for item in tools if _is_external_implementation_tool(item))
    internal_tool_count = max(0, tool_count - external_tool_count)
    implementation_side_effect_tools = sum(1 for item in tools if item.has_side_effects and _is_external_implementation_tool(item))
    approval_tools = sum(1 for item in tools if item.requires_approval)
    workflow_steps = len(blueprint.delivery_package.workflow_profile.steps) if blueprint is not None else 0
    safety_checks = len(blueprint.safety_checks) if blueprint is not None else 0
    guardrails = len(blueprint.guardrails) if blueprint is not None else 0
    memory_layers = len(blueprint.memory_profile.storage_layers) if blueprint is not None else 0
    observability_signals = len(blueprint.delivery_package.observability_plan.captured_signals) if blueprint is not None else 0
    deliverables = len(blueprint.delivery_package.deliverables) if blueprint is not None else 0
    evaluation_cases = (
        len(evaluation_dataset.cases)
        if evaluation_dataset is not None and evaluation_dataset.cases
        else len(evaluation.cases)
        if evaluation is not None
        else 0
    )

    blocking_gaps = acp_preview.construction_readiness.blocking_gaps if acp_preview is not None else 0
    open_questions = acp_preview.construction_readiness.open_questions if acp_preview is not None else 0
    assumptions_count = acp_preview.construction_readiness.assumptions_count if acp_preview is not None else 0
    uncertainty_counts = _classify_construction_uncertainty(acp_preview)
    acp_ready_to_build = (
        acp_preview is not None
        and acp_preview.construction_readiness.can_start_build
        and acp_preview.construction_readiness.overall_status == "ready_to_build"
        and blocking_gaps == 0
        and open_questions == 0
    )
    evaluation_complete = (
        latest_run is not None
        and latest_run.status == ArtifactStatus.ready
        and not latest_run.blocking_issues
        and latest_run.overall_score >= 80
    )

    # 1. Puntos de Negocio y Alcance Funcional
    p_business = 12
    p_business += v1_scope_count * 4
    p_business += constraints_count * 2
    p_business += non_delegable_count * 4
    p_business += automation_opportunities * 2

    # 2. Puntos de Herramientas (Ponderación por tipo, mutación, APIs externas y HITL)
    p_tools = 0
    for item in tools:
        is_ext = _is_external_implementation_tool(item) or any(
            ext in (item.name + " " + item.purpose).lower()
            for ext in ["api", "externa", "webhook", "outbound", "crm", "erp", "salesforce", "sap", "zendesk"]
        )
        has_se = item.has_side_effects or item.execution_mode == "async"
        if has_se and is_ext:
            weight = 12
        elif is_ext:
            weight = 8
        elif has_se:
            weight = 7
        else:
            weight = 4
        if item.requires_approval:
            weight += 3
        p_tools += weight

    # 3. Puntos de Memoria y Estado
    mem_strategy = (blueprint.memory_strategy if blueprint else "").lower()
    grounding = blueprint.memory_profile.grounding_policy if blueprint else None
    if "hybrid" in mem_strategy or "hibrid" in mem_strategy:
        p_memory = 24 + (memory_layers * 3)
    elif "persistent" in mem_strategy or "checkpoints" in mem_strategy:
        p_memory = 16 + (memory_layers * 2)
    elif "vector" in mem_strategy or "semantic" in mem_strategy:
        p_memory = 14 + (memory_layers * 2)
    elif memory_layers > 0:
        p_memory = 8 + (memory_layers * 2)
    else:
        p_memory = 0
    if grounding and getattr(grounding, "citations_policy", ""):
        p_memory += 4

    # 4. Puntos de Guardrails, Safety y Aprobaciones Humanas
    human_approval_count = approval_count
    if not human_approval_count and snapshot.approvals:
        human_approval_count = len(snapshot.approvals)
    p_guardrails = (safety_checks * 4) + (guardrails * 3) + (human_approval_count * 4)

    # 5. Puntos de Ejecución, Observabilidad y Evaluación
    p_execution = (workflow_steps * 3) + observability_signals + min(15, evaluation_cases * 2) + len(snapshot.evaluation_runs)
    if maturity_stage == EstimationMaturityStage.ready_to_build:
        p_execution += 6

    # Multiplicador de Arquitectura y Patrón de Razonamiento
    arch = (blueprint.architecture if blueprint else "single_agent").lower()
    pattern = (blueprint.reasoning_pattern if blueprint else "react").lower()
    arch_map = {
        "single_agent": 1.00,
        "single_agent_with_skills": 1.10,
        "plan_and_execute": 1.25,
        "supervisor_with_subagents": 1.50,
        "handoffs": 1.65,
        "hierarchical_multi_agent": 1.75,
    }
    m_arch = arch_map.get(arch, 1.15)
    if "plan-and-execute" in pattern or "plan and execute" in pattern:
        m_arch *= 1.08
    elif "reflection" in pattern or "tree" in pattern:
        m_arch *= 1.15

    raw_points = p_business + p_tools + p_memory + p_guardrails + p_execution
    scope_points = int(round(raw_points * m_arch))

    complexity = _infer_complexity(scope_points)
    project_archetype = _infer_project_archetype(snapshot, tool_count, implementation_side_effect_tools, memory_layers)
    readiness_state = _infer_readiness_state(snapshot=snapshot, acp_preview=acp_preview, acp_ready_to_build=acp_ready_to_build)
    api_contract_maturity = _infer_api_contract_maturity(
        acp_preview=acp_preview,
        project_archetype=project_archetype,
        external_tool_count=external_tool_count,
        tool_count=tool_count,
    )
    deployment_maturity = _infer_deployment_maturity(
        snapshot=snapshot,
        acp_preview=acp_preview,
        acp_ready_to_build=acp_ready_to_build,
    )
    knowledge_maturity = _infer_knowledge_maturity(
        snapshot=snapshot,
        acp_preview=acp_preview,
        project_archetype=project_archetype,
        memory_layers=memory_layers,
        acp_ready_to_build=acp_ready_to_build,
    )

    assumptions = [
        "El costo de esfuerzo humano usa tarifa cargada Colombia 2026 y no tarifa de staffing internacional.",
        "La duracion asume trabajo paralelo entre 2 y 4 perfiles activos segun complejidad.",
        "El alcance del Blueprint y del ACP se limita al diseño, especificación de contratos (required-api-contracts) y construcción del agente; la implementación de servicios/APIs externos no forma parte del alcance.",
    ]
    if maturity_stage != EstimationMaturityStage.ready_to_build:
        assumptions.append("Deployment, secretos, contratos API y detalles de ambiente aun pueden mover el esfuerzo final.")
    if evaluation_cases == 0:
        assumptions.append("La capa QA se estima con piso minimo porque todavia no existe dataset o rubrica formal.")
    if active_provider != LLMProviderKey.codex_local:
        assumptions.append("El costo LLM del proveedor activo se calcula con el pricing profile vigente del workspace y volumen estimado por etapa.")
    else:
        assumptions.append("Codex local se calcula con politica de costo configurable y snapshot de compute local, no por tokenizacion directa.")

    risk_drivers: list[str] = []
    negative_signals: list[str] = []
    positive_signals: list[str] = []

    if snapshot.canvas is not None:
        positive_signals.append("Canvas Lean disponible como base minima de negocio y alcance.")
    if snapshot.blueprint is not None:
        positive_signals.append("Blueprint estructurado disponible con tools, memoria y workflow.")
    if evaluation_cases > 0:
        positive_signals.append("Existe material de evaluacion para dimensionar QA desde temprano.")
    if acp_preview is not None and maturity_stage == EstimationMaturityStage.ready_to_build:
        positive_signals.append("ACP con construction readiness disponible para aterrizar build y despliegue.")
    if readiness_state == ReviewState.complete:
        positive_signals.append("El blueprint ya declara readiness suficiente para sostener una propuesta comercial mas firme.")
    if api_contract_maturity in {"complete", "not_applicable"}:
        positive_signals.append("La capa contractual de APIs ya no es una fuente mayor de incertidumbre para este corte.")
    if deployment_maturity == "complete":
        positive_signals.append("Deployment y runtime tienen suficiente definicion para estrechar la banda comercial.")
    if knowledge_maturity in {"complete", "not_applicable"}:
        positive_signals.append("Knowledge y retrieval ya no introducen ambiguedad material para la estimacion.")

    if side_effect_tools > 0:
        risk_drivers.append("Hay tools con side effects que requieren mas validacion y aprobaciones.")
    if api_contract_maturity in {"blocked", "missing"}:
        risk_drivers.append("Los contratos API externos aun no estan cerrados y siguen moviendo esfuerzo de backend e integraciones.")
        negative_signals.append("Contratos API y sandbox de terceros siguen abiertos o incompletos.")
    elif api_contract_maturity == "partial":
        risk_drivers.append("La capa de integraciones externas todavia esta parcial y puede mover esfuerzo de hardening.")
        negative_signals.append("Integraciones externas definidas solo de forma parcial para este corte.")
    if knowledge_maturity in {"blocked", "missing"}:
        risk_drivers.append("La estrategia de knowledge y retrieval sigue poco definida y mantiene incertidumbre en datos y runtime.")
        negative_signals.append("Memoria y knowledge aun no aterrizan owner, fuentes o refresh policy.")
    elif knowledge_maturity == "partial":
        risk_drivers.append("Knowledge y retrieval ya existen, pero todavia no cierran ownership o costo operativo.")
        negative_signals.append("La capa semantica sigue parcial y puede mover costo de datos y evaluacion.")
    if deployment_maturity in {"blocked", "missing"}:
        risk_drivers.append("Deployment target, runtime o politica de secretos aun no estan completamente definidos.")
        negative_signals.append("Falta cierre operativo de runtime, deployment o governance final.")
    elif deployment_maturity == "partial":
        risk_drivers.append("Deployment y runtime ya estan modelados, pero aun no bajan a un entorno final sin ambiguedad.")
        negative_signals.append("Persisten decisiones parciales de deployment, observabilidad o secretos.")
    if blocking_gaps > 0:
        risk_drivers.append(f"Persisten {blocking_gaps} gaps blocking en construction readiness.")
        negative_signals.append("Persisten gaps de diseno o implementacion clasificados con ponderacion residual controlada.")
    if open_questions > 0:
        risk_drivers.append(f"Persisten {open_questions} preguntas abiertas con impacto en build.")
        negative_signals.append("Existen preguntas abiertas documentadas para cerrarse durante la ejecucion del ACP.")
    if latest_run is not None and latest_run.blocking_issues:
        risk_drivers.append("La evaluacion ya detecto issues bloqueantes que pueden ensanchar el retrabajo.")
        negative_signals.append("La capa de evaluacion reporta issues bloqueantes.")

    source_artifacts = ["discovery", "canvas"]
    if snapshot.blueprint is not None:
        source_artifacts.append("blueprint")
    if snapshot.evaluation is not None or snapshot.evaluation_dataset is not None:
        source_artifacts.append("evaluation")
    if acp_preview is not None:
        source_artifacts.append("acp_preview")

    blueprint_design_coverage_percent = _build_blueprint_design_coverage_percent(
        snapshot=snapshot,
        schema_validated_tools=schema_validated_tools,
        external_tool_count=external_tool_count,
    )
    acp_package_readiness_percent = _build_acp_package_readiness_percent(
        acp_preview,
        uncertainty_counts=uncertainty_counts,
    )

    return EstimationSignals(
        maturity_stage=maturity_stage,
        complexity=complexity,
        project_archetype=project_archetype,
        active_provider=active_provider,
        source_artifacts=source_artifacts,
        scope_points=scope_points,
        blocking_gaps=blocking_gaps,
        open_questions=open_questions,
        design_gap_count=uncertainty_counts.design_gap_count,
        implementation_gap_count=uncertainty_counts.implementation_gap_count,
        design_open_questions=uncertainty_counts.design_open_questions,
        implementation_open_questions=uncertainty_counts.implementation_open_questions,
        assumptions_count=assumptions_count,
        assumptions=assumptions,
        risk_drivers=_dedupe(risk_drivers),
        positive_signals=_dedupe(positive_signals),
        negative_signals=_dedupe(negative_signals),
        tool_count=tool_count,
        side_effect_tools=side_effect_tools,
        implementation_side_effect_tools=implementation_side_effect_tools,
        external_tool_count=external_tool_count,
        internal_tool_count=internal_tool_count,
        approval_tools=approval_tools,
        workflow_steps=workflow_steps,
        safety_checks=safety_checks,
        guardrails=guardrails,
        memory_layers=memory_layers,
        observability_signals=observability_signals,
        evaluation_cases=evaluation_cases,
        deliverables=deliverables,
        schema_validated_tools=schema_validated_tools,
        readiness_state=readiness_state,
        api_contract_maturity=api_contract_maturity,
        deployment_maturity=deployment_maturity,
        knowledge_maturity=knowledge_maturity,
        acp_ready_to_build=acp_ready_to_build,
        evaluation_complete=evaluation_complete,
        blueprint_design_coverage_percent=blueprint_design_coverage_percent,
        acp_package_readiness_percent=acp_package_readiness_percent,
    )


def _build_traditional_estimate(
    signals: EstimationSignals,
    role_rates: dict[str, RoleRateCatalogEntry],
    workstream_profiles: dict[str, WorkstreamEffortProfile],
) -> TraditionalEstimate:
    shares = _build_workstream_shares(signals)
    total_hours = _estimate_total_traditional_hours(signals, workstream_profiles)
    breakdown = _build_workstream_breakdown(
        total_hours=total_hours,
        shares=shares,
        role_rates=role_rates,
        workstream_profiles=workstream_profiles,
        automation_by_workstream={key: 0 for key in WORKSTREAM_ORDER},
        note_builder=lambda key: _build_traditional_notes(key, signals),
    )
    estimated_cost = round(sum(item.estimated_cost for item in breakdown), 2)
    duration_weeks = _estimate_duration_weeks(signals.maturity_stage, signals.complexity, total_hours)

    return TraditionalEstimate(
        estimated_hours_total=round(total_hours, 2),
        estimated_duration_weeks=round(duration_weeks, 1),
        estimated_cost=estimated_cost,
        team_shape=_build_team_shape(breakdown, workstream_profiles, role_rates),
        workstream_breakdown=breakdown,
        assumptions=signals.assumptions,
        warnings=_build_traditional_warnings(signals),
    )


def _build_agentic_estimate(
    signals: EstimationSignals,
    traditional: TraditionalEstimate,
    *,
    runtime_settings: LLMRuntimeSettings,
    role_rates: dict[str, RoleRateCatalogEntry],
    workstream_profiles: dict[str, WorkstreamEffortProfile],
    automation_profiles: dict[str, AutomationMatrixProfile],
    pricing_profiles: dict[LLMProviderKey, list[LLMPricingProfile]],
) -> AgenticEstimate:
    automation_matrix = build_automation_matrix(signals, automation_profiles)
    automation_by_family = automation_matrix.automation_by_artifact_family
    automation_by_workstream = automation_matrix.automation_by_workstream

    weighted_automation = 0.0
    for item in traditional.workstream_breakdown:
        weighted_automation += item.estimated_hours * automation_by_workstream.get(item.workstream_key, 0)
    automation_percent = int(round(weighted_automation / max(traditional.estimated_hours_total, 1)))

    # Heuristic dynamic calculation of human supervision effort
    complexity_map = {
        EstimationComplexityLevel.simple: 8.0,
        EstimationComplexityLevel.moderate: 16.0,
        EstimationComplexityLevel.complex: 26.0,
        EstimationComplexityLevel.critical: 42.0,
    }
    complexity_base = complexity_map.get(signals.complexity, 16.0)
    
    # Robust test suite relief (up to 30% reduction in manual verification)
    eval_relief = 1.0
    if signals.evaluation_cases > 0:
        eval_relief = max(0.70, 1.0 - (signals.evaluation_cases * 0.03))
        
    tool_factor = (signals.tool_count * 1.5) + (signals.implementation_side_effect_tools * 4.0) + (signals.approval_tools * 2.0)
    guardrail_factor = (signals.safety_checks * 1.2) + (signals.guardrails * 0.8)
    ambiguity_factor = (signals.blocking_gaps * 3.5) + (signals.open_questions * 1.0)
    
    maturity_map = {
        EstimationMaturityStage.canvas: 1.30,
        EstimationMaturityStage.blueprint: 1.00,
        EstimationMaturityStage.ready_to_build: 0.70,
    }
    maturity_multiplier = maturity_map.get(signals.maturity_stage, 1.00)
    
    supervision_hours = (complexity_base * eval_relief + tool_factor + guardrail_factor + ambiguity_factor) * maturity_multiplier
    supervision_hours = round(max(8.0, supervision_hours), 2)

    shares = _build_workstream_shares(signals)
    breakdown: list[WorkstreamEstimate] = []
    for item in traditional.workstream_breakdown:
        auto_percent = automation_by_workstream.get(item.workstream_key, 0)
        human_build_hours = item.estimated_hours * (1 - (auto_percent / 100))
        supervision_alloc = supervision_hours * shares.get(item.workstream_key, 0)
        hours = human_build_hours + supervision_alloc
        profile = workstream_profiles[item.workstream_key]
        role_key = profile.default_role_keys[0] if profile.default_role_keys else ""
        rate = role_rates.get(role_key, RoleRateCatalogEntry()).hourly_rate
        breakdown.append(
            WorkstreamEstimate(
                workstream_key=item.workstream_key,
                label=item.label,
                estimated_hours=round(hours, 2),
                estimated_cost=round(hours * rate, 2),
                duration_days=round(hours / 6.5, 2),
                automation_percent=auto_percent,
                notes=_build_agentic_notes(item.workstream_key, signals, auto_percent),
            )
        )

    estimated_hours_total = round(sum(item.estimated_hours for item in breakdown), 2)
    human_delivery_cost = round(sum(item.estimated_cost for item in breakdown), 2)
    blended_supervision_rate = human_delivery_cost / max(estimated_hours_total, 1)
    provider_costs = estimate_agentic_costs(
        signals=signals,
        runtime_settings=runtime_settings,
        pricing_profiles=pricing_profiles,
        human_supervision_hours=supervision_hours,
        human_supervision_rate_cop=blended_supervision_rate,
    )
    estimated_cost = round(human_delivery_cost + provider_costs.provider_runtime_cost_total_cop, 2)
    warnings = list(provider_costs.warnings)
    if provider_costs.pricing_snapshot is None:
        warnings.append("No existe snapshot de pricing vigente para el proveedor activo; el costo variable permanece incompleto.")
    if signals.blocking_gaps > 0 or signals.open_questions > 0:
        warnings.append("La cobertura agentic cae mientras existan gaps blocking o preguntas abiertas en continuidad constructiva.")

    return AgenticEstimate(
        active_provider=signals.active_provider,
        pricing_policy=provider_costs.pricing_snapshot.profile_key if provider_costs.pricing_snapshot is not None else f"{signals.active_provider.value}_missing_profile",
        provider_model=provider_costs.provider_model,
        economic_model=provider_costs.economic_model,
        estimated_hours_total=estimated_hours_total,
        estimated_duration_weeks=round(_estimate_duration_weeks(signals.maturity_stage, signals.complexity, estimated_hours_total), 1),
        estimated_cost=estimated_cost,
        team_shape=[*traditional.team_shape[:4], "Orquestacion agentic supervisada"],
        workstream_breakdown=breakdown,
        assumptions=[
            *signals.assumptions,
            "El escenario agentic combina automatizacion de artefactos con supervision humana deliberada.",
        ],
        warnings=warnings,
        blueprint_design_coverage_percent=signals.blueprint_design_coverage_percent,
        acp_package_readiness_percent=signals.acp_package_readiness_percent,
        implementation_scope_coverage_percent=automation_percent,
        automation_coverage_percent=automation_percent,
        human_supervision_hours=round(supervision_hours, 2),
        human_delivery_cost=human_delivery_cost,
        human_supervision_cost=provider_costs.human_supervision_cost,
        llm_runtime_cost_usd=provider_costs.llm_runtime_cost_usd,
        tool_runtime_cost_usd=provider_costs.tool_runtime_cost_usd,
        platform_overhead_cost_usd=provider_costs.platform_overhead_cost_usd,
        provider_runtime_cost_total_usd=provider_costs.provider_runtime_cost_total_usd,
        provider_runtime_cost_total_cop=provider_costs.provider_runtime_cost_total_cop,
        tooling_cost_usd=provider_costs.tool_runtime_cost_usd,
        platform_cost_usd=provider_costs.platform_overhead_cost_usd,
        net_savings_vs_traditional=round(traditional.estimated_cost - estimated_cost, 2),
        automation_coverage_by_workstream=automation_by_workstream,
        automation_coverage_by_artifact_family=automation_by_family,
        automation_assessments=automation_matrix.family_assessments,
        pricing_assumptions=provider_costs.pricing_assumptions,
        pricing_snapshot=provider_costs.pricing_snapshot,
    )


def _build_construction_scenarios(
    *,
    traditional: TraditionalEstimate,
    agentic: AgenticEstimate,
    signals: EstimationSignals,
) -> list[EstimationConstructionScenario]:
    baseline_hours = max(traditional.estimated_hours_total, 1)
    baseline_cost = max(traditional.estimated_cost, 0)
    baseline_duration = max(traditional.estimated_duration_weeks, 0)
    blended_rate = baseline_cost / baseline_hours if baseline_hours else 0
    provider_runtime_cost = max(agentic.provider_runtime_cost_total_cop, 0)

    design_uncertainty_penalty = min(0.08, signals.design_gap_count * 0.04 + signals.design_open_questions * 0.01)
    implementation_uncertainty_penalty = min(
        0.06,
        signals.implementation_gap_count * 0.02 + signals.implementation_open_questions * 0.005,
    )
    residual_uncertainty_note = (
        "Incertidumbre residual separada: "
        f"diseno={signals.design_gap_count} gap(s)/{signals.design_open_questions} pregunta(s), "
        f"implementacion={signals.implementation_gap_count} gap(s)/{signals.implementation_open_questions} pregunta(s). "
        "Los gaps de implementacion se gestionan en ACP sin penalizar el Blueprint como si fueran diseno abierto."
    )

    blueprint_basic_reduction = _clamp(
        signals.blueprint_design_coverage_percent * 0.0032 - design_uncertainty_penalty * 0.45,
        0.12,
        0.30,
    )
    blueprint_premium_reduction = _clamp(
        signals.blueprint_design_coverage_percent * 0.0042 - design_uncertainty_penalty * 0.35,
        0.22,
        0.40,
    )
    acp_manual_reduction = _clamp(signals.acp_package_readiness_percent * 0.0035, 0.22, 0.36)
    blueprint_basic_hours = baseline_hours * (1 - blueprint_basic_reduction)
    blueprint_premium_hours = baseline_hours * (1 - blueprint_premium_reduction)

    acp_agentic_hours = agentic.estimated_hours_total
    agentic_bp_min_hours = acp_agentic_hours * 1.20
    agentic_blueprint_hours = _clamp(
        max(agentic_bp_min_hours, baseline_hours * 0.58),
        acp_agentic_hours * 1.18,
        blueprint_premium_hours * 0.94,
    )

    blueprint_premium_cost = round(blueprint_premium_hours * blended_rate + provider_runtime_cost * 0.45, 2)
    agentic_blueprint_cost = round(agentic_blueprint_hours * blended_rate + provider_runtime_cost * 0.75, 2)
    if agentic_blueprint_cost >= blueprint_premium_cost and blended_rate > 0:
        target_hours = (blueprint_premium_cost * 0.96 - provider_runtime_cost * 0.75) / blended_rate
        agentic_blueprint_hours = max(acp_agentic_hours * 1.18, min(agentic_blueprint_hours, target_hours))
        agentic_blueprint_cost = round(agentic_blueprint_hours * blended_rate + provider_runtime_cost * 0.75, 2)
    acp_manual_hours = max(acp_agentic_hours * 1.12, baseline_hours * (1 - acp_manual_reduction))
    done_for_you_factory_cost = round(agentic.estimated_cost * 0.9, 2)

    scenarios = [
        _construction_scenario(
            scenario_key="traditional_blueprint",
            label="Desarrollo tradicional",
            description="Equipo humano construye de forma tradicional; sirve como linea base para comparar el valor del Blueprint, ACP y tooling agentico.",
            hours=traditional.estimated_hours_total,
            duration=traditional.estimated_duration_weeks,
            cost=traditional.estimated_cost,
            baseline_hours=baseline_hours,
            baseline_cost=baseline_cost,
            human_intervention_percent=100,
            automation_leverage_percent=0,
            notes=[
                "Maximiza trazabilidad del diseno, pero mantiene ejecucion manual.",
                "Util cuando el cliente no usara tooling agentico durante construccion.",
                residual_uncertainty_note,
            ],
        ),
        _construction_scenario(
            scenario_key="blueprint_basic",
            label="Blueprint Basico",
            description="Producto de entrada: genera el diseno inicial, infiere, registra supuestos y continua con costo computacional controlado.",
            hours=round(blueprint_basic_hours, 2),
            duration=round(_scale_duration(baseline_duration, blueprint_basic_hours, baseline_hours, 0.96), 1),
            cost=round(blueprint_basic_hours * blended_rate + provider_runtime_cost * 0.25, 2),
            baseline_hours=baseline_hours,
            baseline_cost=baseline_cost,
            human_intervention_percent=82,
            automation_leverage_percent=int(_clamp(round(signals.blueprint_design_coverage_percent * 0.38), 15, 45)),
            notes=[
                "Inferir + registrar + continuar: las preguntas no bloquean salvo imposibilidad tecnica.",
                "Las dudas se preservan como oportunidades de enriquecimiento para convertir a Premium.",
                residual_uncertainty_note,
            ],
        ),
        _construction_scenario(
            scenario_key="blueprint_premium",
            label="Blueprint Premium",
            description="Blueprint enriquecido: preguntas relevantes se resuelven y solo se reprocesan los entregables afectados.",
            hours=round(blueprint_premium_hours, 2),
            duration=round(_scale_duration(baseline_duration, blueprint_premium_hours, baseline_hours, 0.92), 1),
            cost=blueprint_premium_cost,
            baseline_hours=baseline_hours,
            baseline_cost=baseline_cost,
            human_intervention_percent=68,
            automation_leverage_percent=int(_clamp(round(signals.blueprint_design_coverage_percent * 0.55), 32, 68)),
            notes=[
                "Pregunta + resolver + enriquecer: reduce incertidumbre de diseno antes de construir.",
                "El reprocesamiento selectivo se basa en dependencias de entregables y respuestas.",
                residual_uncertainty_note,
            ],
        ),
        _construction_scenario(
            scenario_key="agentic_blueprint",
            label="Agentic + Blueprint",
            description="Herramientas agenticas asisten la construccion tomando el Blueprint como insumo principal.",
            hours=round(agentic_blueprint_hours, 2),
            duration=round(_scale_duration(baseline_duration, agentic_blueprint_hours, baseline_hours, 0.9), 1),
            cost=agentic_blueprint_cost,
            baseline_hours=baseline_hours,
            baseline_cost=baseline_cost,
            human_intervention_percent=55,
            automation_leverage_percent=45,
            notes=[
                "Acelera construccion, pero aun requiere interpretar y convertir artefactos a estructura ejecutable.",
                "Debe mejorar el esfuerzo frente a Blueprint Premium porque adiciona herramientas agenticas de construccion.",
                "No reemplaza el ACP cuando se busca handoff premium y repetible.",
                residual_uncertainty_note,
            ],
        ),
        _construction_scenario(
            scenario_key="acp_manual",
            label="ACP + equipo humano",
            description="Equipo humano implementa desde el ACP sin redescubrir ni redisenar la solucion.",
            hours=round(acp_manual_hours, 2),
            duration=round(_scale_duration(baseline_duration, acp_manual_hours, baseline_hours, 0.95), 1),
            cost=round(acp_manual_hours * blended_rate, 2),
            baseline_hours=baseline_hours,
            baseline_cost=baseline_cost,
            human_intervention_percent=70,
            automation_leverage_percent=int(_clamp(round(signals.acp_package_readiness_percent * 0.52), 30, 72)),
            notes=[
                "El valor proviene de tener prompts, contratos, memoria, flujos y preguntas de implementacion ya empaquetadas.",
                "Reduce retrabajo aunque la ejecucion siga siendo principalmente humana.",
                "Incluye preguntas tecnicas y gaps de implementacion en el momento correcto del ACP.",
            ],
        ),
        _construction_scenario(
            scenario_key="acp_agentic",
            label="ACP + herramientas agenticas",
            description="Construccion acelerada ejecutando el ACP con tooling agentico y supervision humana.",
            hours=agentic.estimated_hours_total,
            duration=agentic.estimated_duration_weeks,
            cost=agentic.estimated_cost,
            baseline_hours=baseline_hours,
            baseline_cost=baseline_cost,
            human_intervention_percent=28,
            automation_leverage_percent=72,
            notes=[
                "Es el escenario de mayor apalancamiento porque parte del paquete tecnico completo.",
                "Las preguntas pendientes se resuelven durante implementacion, no reducen el valor del Blueprint ni del ACP.",
                f"Penalizacion residual controlada por implementacion: {implementation_uncertainty_penalty:.0%} maximo local, sin mezclarla con gaps de diseno.",
            ],
        ),
        _construction_scenario(
            scenario_key="done_for_you_factory",
            label="Hagalo con nosotros (Fabrica de Desarrollo)",
            description=(
                "Nosotros podemos desarrollar el agente por ustedes. El mayor tiempo considera la alineacion con "
                "la infraestructura del cliente, sincronizacion operativa y entendimiento de las herramientas "
                "entregadas por el cliente. Si tambien desean que desarrollemos herramientas externas, APIs, MCP "
                "o integraciones legacy, se cotiza por separado."
            ),
            hours=round(acp_manual_hours, 2),
            duration=traditional.estimated_duration_weeks,
            cost=done_for_you_factory_cost,
            baseline_hours=baseline_hours,
            baseline_cost=baseline_cost,
            human_intervention_percent=35,
            automation_leverage_percent=int(_clamp(round(signals.acp_package_readiness_percent * 0.72), 45, 86)),
            notes=[
                "Servicio de fabrica: el cliente delega la construccion del agente manteniendo aprobaciones y accesos bajo su control.",
                "El tiempo adicional cubre alineacion con infraestructura, coordinacion con equipos internos y transferencia de conocimiento de herramientas existentes.",
                "No incluye construir o modernizar APIs externas, MCP, conectores legacy, credenciales ni aprobaciones de seguridad; esos frentes requieren cotizacion separada.",
            ],
        ),
    ]
    return scenarios


def _construction_scenario(
    *,
    scenario_key: str,
    label: str,
    description: str,
    hours: float,
    duration: float,
    cost: float,
    baseline_hours: float,
    baseline_cost: float,
    human_intervention_percent: int,
    automation_leverage_percent: int,
    notes: list[str],
) -> EstimationConstructionScenario:
    effort_reduction = int(_clamp(round((1 - hours / max(baseline_hours, 1)) * 100), 0, 100))
    return EstimationConstructionScenario(
        scenario_key=scenario_key,
        label=label,
        description=description,
        estimated_hours_total=round(max(hours, 0), 2),
        estimated_duration_weeks=round(max(duration, 0), 1),
        estimated_cost=round(max(cost, 0), 2),
        human_intervention_percent=human_intervention_percent,
        automation_leverage_percent=automation_leverage_percent,
        effort_reduction_vs_traditional_percent=effort_reduction,
        cost_savings_vs_traditional=round(max(0, baseline_cost - cost), 2),
        notes=notes,
    )


def _scale_duration(baseline_duration: float, scenario_hours: float, baseline_hours: float, parallel_factor: float) -> float:
    if baseline_duration <= 0 or baseline_hours <= 0:
        return 0
    return max(0.5, baseline_duration * (scenario_hours / baseline_hours) * parallel_factor)


def _build_workstream_breakdown(
    *,
    total_hours: float,
    shares: dict[str, float],
    role_rates: dict[str, RoleRateCatalogEntry],
    workstream_profiles: dict[str, WorkstreamEffortProfile],
    automation_by_workstream: dict[str, int],
    note_builder,
) -> list[WorkstreamEstimate]:
    breakdown: list[WorkstreamEstimate] = []
    for workstream_key in WORKSTREAM_ORDER:
        profile = workstream_profiles[workstream_key]
        role_key = profile.default_role_keys[0] if profile.default_role_keys else ""
        rate = role_rates.get(role_key, RoleRateCatalogEntry()).hourly_rate
        hours = round(total_hours * shares.get(workstream_key, 0), 2)
        breakdown.append(
            WorkstreamEstimate(
                workstream_key=workstream_key,
                label=profile.label or workstream_key.title(),
                estimated_hours=hours,
                estimated_cost=round(hours * rate, 2),
                duration_days=round(hours / 6.5, 2),
                automation_percent=automation_by_workstream.get(workstream_key, 0),
                notes=note_builder(workstream_key),
            )
        )
    return breakdown


def _build_workstream_shares(signals: EstimationSignals) -> dict[str, float]:
    base_distribution = ARCHETYPE_DISTRIBUTIONS.get(signals.project_archetype, ARCHETYPE_DISTRIBUTIONS["app_web_transaccional"])
    weights: dict[str, float] = {}

    for workstream_key in WORKSTREAM_ORDER:
        weight = base_distribution[workstream_key] / 100
        if workstream_key == "backend":
            weight *= 1 + (0.12 if signals.workflow_steps >= 4 else 0) + (0.08 if signals.tool_count >= 4 else 0)
        elif workstream_key == "frontend":
            weight *= 1 + (0.08 if signals.maturity_stage != EstimationMaturityStage.canvas else 0) + (0.06 if signals.deliverables >= 8 else 0)
        elif workstream_key == "integrations":
            weight *= 1 + (0.25 if signals.implementation_side_effect_tools > 0 else 0) + (0.12 if signals.external_tool_count > 0 else 0)
        elif workstream_key == "data":
            weight *= 1 + (0.24 if signals.memory_layers > 0 else 0) + (0.12 if signals.project_archetype == "agentic_platform" else 0)
        elif workstream_key == "qa":
            weight *= 1 + (0.14 if signals.evaluation_cases >= 4 else 0) + (0.08 if signals.safety_checks >= 4 else 0)
        elif workstream_key == "devops":
            weight *= 1 + (0.12 if signals.observability_signals >= 4 else 0) + (0.1 if signals.maturity_stage == EstimationMaturityStage.ready_to_build else 0)
        weights[workstream_key] = weight

    shares = _normalize_weights(weights)
    return _apply_share_floors(shares, {"qa": 0.12, "devops": 0.10})


def _estimate_total_traditional_hours(
    signals: EstimationSignals,
    workstream_profiles: dict[str, WorkstreamEffortProfile],
) -> float:
    reference_total = 0.0
    for workstream_key in WORKSTREAM_ORDER:
        band = _get_workstream_band(workstream_profiles[workstream_key], signals.complexity)
        if band is None:
            continue
        reference_total += _band_midpoint(band)

    scope_baseline = COMPLEXITY_BASELINE_POINTS[signals.complexity]
    scope_factor = _clamp(signals.scope_points / max(scope_baseline, 1), 0.65, 2.20)

    risk_multiplier = 1.0
    if signals.implementation_side_effect_tools > 0:
        risk_multiplier += min(0.12, 0.04 + signals.implementation_side_effect_tools * 0.02)
    if signals.blocking_gaps > 0:
        risk_multiplier += min(0.14, signals.blocking_gaps * 0.04)
    if signals.open_questions > 0:
        risk_multiplier += min(0.08, signals.open_questions * 0.01)
    if signals.project_archetype == "enterprise_integrations" and signals.maturity_stage != EstimationMaturityStage.ready_to_build:
        risk_multiplier += 0.08
    if signals.project_archetype == "agentic_platform" and signals.memory_layers == 0:
        risk_multiplier += 0.05
    if signals.evaluation_cases == 0 and signals.maturity_stage != EstimationMaturityStage.canvas:
        risk_multiplier += 0.04

    # Cap risk multiplier to avoid compounded escalation
    risk_multiplier = min(1.25, risk_multiplier)

    # Calculate base hours for legacy rebuild
    raw_legacy_hours = reference_total * MATURITY_FACTORS[signals.maturity_stage] * scope_factor * risk_multiplier

    # Apply the agent-only scope factor (0.22) to exclude legacy backend/infrastructure build hours
    agent_only_hours = raw_legacy_hours * 0.22

    return round(agent_only_hours, 2)


def _build_traditional_notes(workstream_key: str, signals: EstimationSignals) -> list[str]:
    notes = ["Escenario tradicional con construccion humana asistida por tooling general, pero sin automatizacion agentic intensiva."]
    if workstream_key == "integrations":
        notes.append("El alcance se limita al contrato API (required-api-contracts) y wrapper cliente del agente; la construccion/backend de sistemas externos no forma parte del alcance del ACP.")
        if signals.implementation_side_effect_tools > 0:
            notes.append("Incluye mas tiempo de hardening por side effects y aprobaciones.")
    if workstream_key == "data" and signals.project_archetype == "agentic_platform":
        notes.append("Knowledge, retrieval y refresh policy pesan mas en este tipo de solucion.")
    if workstream_key == "devops" and signals.maturity_stage != EstimationMaturityStage.ready_to_build:
        notes.append("Aun existe incertidumbre de ambiente objetivo y politica de secretos.")
    return notes


def _build_agentic_notes(workstream_key: str, signals: EstimationSignals, auto_percent: int) -> list[str]:
    notes = [f"Cobertura agentic v1 estimada en {auto_percent}% para este workstream."]
    if workstream_key == "integrations":
        notes.append("Especificacion de esquemas/contratos y cliente del agente. La construccion de servicios externos queda bajo la responsabilidad de sus respectivos equipos/owners.")
        if signals.implementation_side_effect_tools > 0:
            notes.append("Las integraciones con side effects limitan la autonomia segura del sistema agentic.")
    if workstream_key == "qa":
        notes.append("La supervision humana sigue siendo critica para validar hallazgos y regresion.")
    if workstream_key == "devops" and signals.maturity_stage != EstimationMaturityStage.ready_to_build:
        notes.append("Deployment sin ambiente cerrado no debe automatizarse de manera ciega.")
    return notes


def _build_traditional_warnings(signals: EstimationSignals) -> list[str]:
    warnings: list[str] = []
    if signals.blocking_gaps > 0:
        warnings.append("La estimacion todavia absorbe gaps blocking no cerrados en el ACP.")
    if signals.open_questions > 0:
        warnings.append("Persisten preguntas abiertas que pueden mover secuencia, integraciones o deployment.")
    if signals.maturity_stage == EstimationMaturityStage.canvas:
        warnings.append("La confianza en Canvas es comercialmente util pero todavia no contractual.")
    return warnings


def _build_team_shape(
    breakdown: list[WorkstreamEstimate],
    workstream_profiles: dict[str, WorkstreamEffortProfile],
    role_rates: dict[str, RoleRateCatalogEntry],
) -> list[str]:
    labels: list[str] = []
    for item in breakdown:
        if item.estimated_hours < 35:
            continue
        profile = workstream_profiles[item.workstream_key]
        role_key = profile.default_role_keys[0] if profile.default_role_keys else ""
        role = role_rates.get(role_key)
        if role is not None:
            labels.append(role.label)
        else:
            labels.append(profile.label)
    return _dedupe(labels)


def _infer_maturity_stage(snapshot: SessionSnapshot, acp_preview: ACPPreview | None) -> EstimationMaturityStage:
    if (
        acp_preview is not None
        and acp_preview.construction_readiness.overall_status == "ready_to_build"
        and acp_preview.construction_readiness.can_start_build
        and acp_preview.construction_readiness.open_questions == 0
        and acp_preview.construction_readiness.blocking_gaps == 0
    ):
        return EstimationMaturityStage.ready_to_build
    if snapshot.blueprint is not None:
        return EstimationMaturityStage.blueprint
    return EstimationMaturityStage.canvas


def _infer_readiness_state(
    *,
    snapshot: SessionSnapshot,
    acp_preview: ACPPreview | None,
    acp_ready_to_build: bool,
) -> ReviewState:
    if acp_ready_to_build:
        return ReviewState.complete
    if snapshot.blueprint is not None:
        return snapshot.blueprint.readiness_state
    if acp_preview is not None and acp_preview.construction_readiness.blocking_gaps > 0:
        return ReviewState.blocked
    return ReviewState.partial


def _infer_api_contract_maturity(
    *,
    acp_preview: ACPPreview | None,
    project_archetype: str,
    external_tool_count: int,
    tool_count: int,
) -> str:
    relevant = project_archetype == "enterprise_integrations" or external_tool_count > 0
    if not relevant and tool_count == 0:
        return "not_applicable"
    if not relevant:
        return "not_applicable"

    gap = _find_active_gap(acp_preview, {"integrations"})
    if gap is not None:
        return "blocked" if gap.severity == "blocking" else "partial"
    if acp_preview is not None:
        return "complete" if acp_preview.construction_readiness.can_start_build else "partial"
    if relevant or tool_count > 0:
        return "partial"
    return "not_applicable"


def _infer_deployment_maturity(
    *,
    snapshot: SessionSnapshot,
    acp_preview: ACPPreview | None,
    acp_ready_to_build: bool,
) -> str:
    gap = _find_active_gap(acp_preview, {"deployment", "runtime"})
    if gap is not None:
        return "blocked" if gap.severity == "blocking" else "partial"
    if acp_ready_to_build:
        return "complete"
    if acp_preview is not None or snapshot.blueprint is not None:
        return "partial"
    return "missing"


def _infer_knowledge_maturity(
    *,
    snapshot: SessionSnapshot,
    acp_preview: ACPPreview | None,
    project_archetype: str,
    memory_layers: int,
    acp_ready_to_build: bool,
) -> str:
    relevant = project_archetype == "agentic_platform" or memory_layers > 0
    if not relevant:
        return "not_applicable"

    gap = _find_active_gap(acp_preview, {"knowledge"})
    if gap is not None:
        return "blocked" if gap.severity == "blocking" else "partial"
    if acp_ready_to_build and memory_layers > 0:
        return "complete"
    if snapshot.blueprint is not None or memory_layers > 0:
        return "partial"
    return "missing"


def _find_active_gap(acp_preview: ACPPreview | None, domains: set[str]):
    if acp_preview is None:
        return None
    for gap in acp_preview.construction_readiness.gaps:
        if gap.domain not in domains:
            continue
        if gap.status in {"resolved", "waived"}:
            continue
        return gap
    return None


def _build_blueprint_design_coverage_percent(
    *,
    snapshot: SessionSnapshot,
    schema_validated_tools: int,
    external_tool_count: int,
) -> int:
    discovery = snapshot.discovery
    canvas = snapshot.canvas
    blueprint = snapshot.blueprint
    evaluation = snapshot.evaluation
    evaluation_dataset = snapshot.evaluation_dataset
    latest_run = snapshot.evaluation_runs[0] if snapshot.evaluation_runs else None

    score = 0.0
    if discovery is not None:
        score += _scaled_count(
            [
                discovery.problem_statement,
                discovery.current_process,
                discovery.desired_outcome,
            ],
            6,
        )
        score += 3 if discovery.mvp_definition.v1_scope and discovery.mvp_definition.north_star_metric else 0
    if canvas is not None:
        score += _scaled_count([canvas.user_goal, canvas.success_metric, canvas.primary_risk], 3)

    if blueprint is None:
        return int(_clamp(round(score), 0, 100))

    workflow = blueprint.delivery_package.workflow_profile
    observability = blueprint.delivery_package.observability_plan
    knowledge = blueprint.knowledge_profile
    memory = blueprint.memory_profile

    score += _scaled_count([blueprint.architecture, blueprint.reasoning_pattern], 7)
    score += 3 if workflow.execution_pattern else 0
    score += min(4, len(workflow.steps))
    score += 2 if blueprint.delivery_package.pattern_catalog or blueprint.delivery_package.decision_trace else 0

    tools = blueprint.tools
    if tools:
        score += 4
        score += min(6, round((schema_validated_tools / max(len(tools), 1)) * 6))
        score += 2 if all(item.inputs and item.outputs for item in tools) else 0
        side_effect_tools = [item for item in tools if item.has_side_effects]
        score += 3 if not side_effect_tools or all(item.requires_approval for item in side_effect_tools) else 1
        score += 1 if external_tool_count == 0 or any(item.registered_api_ref or item.endpoint_reference for item in tools) else 0

    score += _scaled_count([blueprint.memory_strategy, memory.strategy], 4)
    score += min(3, len(memory.storage_layers))
    score += _scaled_count([memory.write_policy, memory.retrieval_policy, memory.retention_policy, memory.ttl_policy], 4)
    if knowledge.mode != "none":
        score += 2
        score += min(2, len(knowledge.sources))
        score += 1 if knowledge.retrieval_policy.search_mode or knowledge.retrieval_policy.top_k else 0
        score += 1 if knowledge.refresh_policy.frequency or knowledge.refresh_policy.triggers else 0
    score += _scaled_count(
        [
            memory.grounding_policy.citations_policy,
            memory.grounding_policy.confidence_policy,
            memory.grounding_policy.no_evidence_behavior,
            memory.grounding_policy.contradictory_evidence_behavior,
        ],
        2,
    )

    score += min(4, len(blueprint.safety_checks))
    score += min(3, len(blueprint.guardrails))
    score += _scaled_count([blueprint.llm_policy.provider, blueprint.llm_policy.reasoning_model or blueprint.llm_policy.fast_model, blueprint.llm_policy.context_policy], 2)
    score += 3 if blueprint.readiness_state == ReviewState.complete else 1 if blueprint.readiness_state == ReviewState.partial else 0

    evaluation_cases = (
        len(evaluation_dataset.cases)
        if evaluation_dataset is not None and evaluation_dataset.cases
        else len(evaluation.cases)
        if evaluation is not None
        else 0
    )
    score += min(5, evaluation_cases)
    score += min(4, len(observability.captured_signals))
    score += 3 if latest_run is not None and latest_run.status == ArtifactStatus.ready and latest_run.overall_score >= 80 else 0

    score += min(5, len(blueprint.delivery_package.deliverables))
    coverage = blueprint.delivery_package.blueprint_coverage
    if coverage.total_sections > 0:
        score += min(5, round((coverage.covered_sections / coverage.total_sections) * 5))

    return int(_clamp(round(score), 0, 100))


def _build_acp_package_readiness_percent(
    acp_preview: ACPPreview | None,
    *,
    uncertainty_counts: ConstructionUncertaintyCounts | None = None,
) -> int:
    if acp_preview is None:
        return 0
    counts = uncertainty_counts or _classify_construction_uncertainty(acp_preview)

    score = acp_preview.validation.completeness_percent
    if score == 0 and acp_preview.files:
        score = min(75, 30 + len(acp_preview.files) * 2)

    file_paths = {item.path for item in acp_preview.files}
    readiness = acp_preview.construction_readiness
    if acp_preview.validation.can_export_zip:
        score += 8
    if acp_preview.manifest_path in file_paths:
        score += 4
    if any(path.startswith("ACP/diagrams/") for path in file_paths):
        score += 4
    if any(path == "ACP/construction-readiness/open-questions.yaml" for path in file_paths):
        score += 4

    residual_gap_penalty = counts.design_gap_count * 4 + counts.implementation_gap_count * 2
    residual_question_penalty = min(6, counts.design_open_questions * 2 + counts.implementation_open_questions)
    score -= residual_gap_penalty + residual_question_penalty
    score -= min(3, readiness.assumptions_count)
    return int(_clamp(round(score), 0, 100))


def _classify_construction_uncertainty(acp_preview: ACPPreview | None) -> ConstructionUncertaintyCounts:
    if acp_preview is None:
        return ConstructionUncertaintyCounts()

    design_gap_count = 0
    implementation_gap_count = 0
    design_open_questions = 0
    implementation_open_questions = 0
    has_implementation_gap = False
    for gap in acp_preview.construction_readiness.gaps:
        if gap.status in {"resolved", "waived", "answered"}:
            continue
        is_implementation = _is_implementation_uncertainty(gap.domain, gap.gap_key, gap.blocking_stage)
        has_implementation_gap = has_implementation_gap or is_implementation
        open_question_count = sum(1 for question in gap.questions if getattr(question, "status", "open") == "open")
        if is_implementation:
            implementation_open_questions += open_question_count
            if gap.severity == "blocking":
                implementation_gap_count += 1
            continue
        design_open_questions += open_question_count
        if gap.severity == "blocking":
            design_gap_count += 1

    classified_open_questions = design_open_questions + implementation_open_questions
    remaining_open_questions = max(0, acp_preview.construction_readiness.open_questions - classified_open_questions)
    if remaining_open_questions:
        if has_implementation_gap:
            implementation_open_questions += remaining_open_questions
        else:
            design_open_questions += remaining_open_questions

    return ConstructionUncertaintyCounts(
        design_gap_count=design_gap_count,
        implementation_gap_count=implementation_gap_count,
        design_open_questions=design_open_questions,
        implementation_open_questions=implementation_open_questions,
    )


def _is_implementation_uncertainty(domain: str, gap_key: str, blocking_stage: str) -> bool:
    normalized_domain = _normalize_token(domain)
    if normalized_domain in IMPLEMENTATION_UNCERTAINTY_DOMAINS:
        return True
    haystack = " ".join([gap_key, domain, blocking_stage]).lower()
    return any(hint in haystack for hint in IMPLEMENTATION_UNCERTAINTY_KEY_HINTS)


def _scaled_count(values: list[str], max_score: int) -> int:
    if not values:
        return 0
    present = sum(1 for value in values if value.strip())
    return round((present / len(values)) * max_score)


def _tool_has_schema_validation(tool: BlueprintTool) -> bool:
    validation_tokens = {_normalize_token(value) for value in tool.validations}
    if any("schema" in token for token in validation_tokens):
        return True
    return bool(tool.request_schema or tool.response_schema or (tool.inputs and tool.outputs))


def _is_external_implementation_tool(tool: BlueprintTool) -> bool:
    if _is_internal_agent_tool(tool):
        return False
    endpoint = tool.endpoint_reference.strip().lower()
    if endpoint.startswith(DESIGN_PLACEHOLDER_ENDPOINT_PREFIXES):
        return False
    if tool.registered_api_ref.strip():
        return True
    text = " ".join(
        [
            tool.name,
            tool.purpose,
            tool.archetype,
            tool.integration_kind,
            tool.endpoint_reference,
            tool.when_to_use,
        ]
    ).lower()
    return tool.tool_type == "external" and (
        tool.integration_kind.lower() in {"api", "webhook", "external_api"}
        or any(hint in text for hint in EXTERNAL_IMPLEMENTATION_HINTS)
    )


def _is_internal_agent_tool(tool: BlueprintTool) -> bool:
    name = _normalize_token(tool.name)
    archetype = _normalize_token(tool.archetype)
    integration_kind = _normalize_token(tool.integration_kind)
    endpoint = tool.endpoint_reference.strip().lower()
    if endpoint.startswith(INTERNAL_ENDPOINT_PREFIXES):
        return True
    if name in INTERNAL_AGENT_TOOL_NAMES:
        return True
    if archetype in INTERNAL_AGENT_ARCHETYPES and integration_kind in INTERNAL_AGENT_INTEGRATION_KINDS:
        return True
    return tool.tool_type == "internal"


def _normalize_token(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _infer_project_archetype(
    snapshot: SessionSnapshot,
    tool_count: int,
    side_effect_tools: int,
    memory_layers: int,
) -> str:
    text = _collect_context_text(snapshot)
    if memory_layers > 0 or any(token in text for token in ("agent", "llm", "prompt", "retrieval", "vector", "knowledge", "memory", "evaluation", "observability")):
        return "agentic_platform"
    if side_effect_tools > 0 or tool_count >= 4 or any(token in text for token in ("api", "erp", "crm", "webhook", "integration", "middleware", "ticket", "email")):
        return "enterprise_integrations"
    return "app_web_transaccional"


def _collect_context_text(snapshot: SessionSnapshot) -> str:
    parts: list[str] = []
    if snapshot.discovery is not None:
        parts.extend(
            [
                snapshot.discovery.problem_statement,
                snapshot.discovery.current_process,
                snapshot.discovery.desired_outcome,
                " ".join(snapshot.discovery.constraints),
                " ".join(snapshot.discovery.mvp_definition.v1_scope),
            ]
        )
    if snapshot.canvas is not None:
        parts.extend(
            [
                snapshot.canvas.user_goal,
                snapshot.canvas.primary_risk,
                " ".join(snapshot.canvas.mvp_scope),
            ]
        )
    if snapshot.blueprint is not None:
        parts.extend(
            [
                snapshot.blueprint.architecture,
                snapshot.blueprint.reasoning_pattern,
                snapshot.blueprint.memory_strategy,
                snapshot.blueprint.narrative,
                " ".join(item.name for item in snapshot.blueprint.tools),
                " ".join(item.purpose for item in snapshot.blueprint.tools),
                " ".join(snapshot.blueprint.guardrails),
                " ".join(snapshot.blueprint.delivery_package.observability_plan.captured_signals),
            ]
        )
    return " ".join(item.lower() for item in parts if item)


def _infer_complexity(scope_points: int) -> EstimationComplexityLevel:
    if scope_points < 90:
        return EstimationComplexityLevel.simple
    if scope_points < 200:
        return EstimationComplexityLevel.moderate
    if scope_points < 320:
        return EstimationComplexityLevel.complex
    return EstimationComplexityLevel.critical


def _load_role_rates(session: Session) -> dict[str, RoleRateCatalogEntry]:
    rows = _load_catalog_rows(session, "estimation_role_rates")
    return {item.role_key: item for item in (_validate_catalog_payload(RoleRateCatalogEntry, row.payload) for row in rows)}


def _load_workstream_profiles(session: Session) -> dict[str, WorkstreamEffortProfile]:
    rows = _load_catalog_rows(session, "estimation_workstream_effort")
    return {
        item.workstream_key: item
        for item in (_validate_catalog_payload(WorkstreamEffortProfile, row.payload) for row in rows)
    }


def _load_automation_profiles(session: Session) -> dict[str, AutomationMatrixProfile]:
    rows = _load_catalog_rows(session, "estimation_automation_matrix")
    return {
        item.family_key: item
        for item in (_validate_catalog_payload(AutomationMatrixProfile, {**row.payload, "label": row.label}) for row in rows)
    }


def _load_pricing_profiles(session: Session) -> dict[LLMProviderKey, list[LLMPricingProfile]]:
    rows = _load_catalog_rows(session, "estimation_pricing_profiles")
    profiles = [_validate_catalog_payload(LLMPricingProfile, row.payload) for row in rows]
    grouped: dict[LLMProviderKey, list[LLMPricingProfile]] = {}
    for item in profiles:
        grouped.setdefault(item.provider, []).append(item)
    return grouped


def _load_confidence_bands(session: Session) -> list[dict[str, Any]]:
    rows = _load_catalog_rows(session, "estimation_confidence_bands")
    return [row.payload for row in rows]


def _load_confidence_weights(session: Session) -> dict[str, float]:
    rows = _load_catalog_rows(session, "estimation_confidence_weights")
    weights: dict[str, float] = {}
    for row in rows:
        metric_key = row.payload.get("metric_key")
        amount = row.payload.get("amount")
        if isinstance(metric_key, str) and isinstance(amount, (int, float)):
            weights[metric_key] = float(amount)
    return weights


def _load_catalog_rows(session: Session, catalog_key: str) -> list[RuntimeCatalogEntryRecord]:
    return session.exec(
        select(RuntimeCatalogEntryRecord)
        .where(
            RuntimeCatalogEntryRecord.catalog_key == catalog_key,
            RuntimeCatalogEntryRecord.is_active == True,  # noqa: E712
        )
        .order_by(RuntimeCatalogEntryRecord.order_index.asc())
    ).all()


def _validate_catalog_payload(model_cls, payload: dict[str, Any]):
    filtered_payload = {key: value for key, value in payload.items() if key in model_cls.model_fields}
    return model_cls.model_validate(filtered_payload)


def _get_workstream_band(
    profile: WorkstreamEffortProfile,
    complexity: EstimationComplexityLevel,
) -> WorkstreamEffortBand | None:
    for band in profile.bands:
        if band.complexity == complexity:
            return band
    return None


def _band_midpoint(band: WorkstreamEffortBand) -> float:
    return (band.base_hours_min + band.base_hours_max) / 2


def _estimate_duration_weeks(
    maturity_stage: EstimationMaturityStage,
    complexity: EstimationComplexityLevel,
    total_hours: float,
) -> float:
    active_fte = ACTIVE_FTE_BY_COMPLEXITY[complexity]
    if maturity_stage == EstimationMaturityStage.canvas:
        active_fte -= 0.2
    elif maturity_stage == EstimationMaturityStage.ready_to_build:
        active_fte += 0.2
    weekly_capacity = max(62.0, active_fte * 28.0)
    return total_hours / weekly_capacity


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values()) or 1.0
    return {key: value / total for key, value in weights.items()}


def _apply_share_floors(shares: dict[str, float], floor_map: dict[str, float]) -> dict[str, float]:
    adjusted = dict(shares)
    deficit = 0.0
    for key, floor in floor_map.items():
        current = adjusted.get(key, 0.0)
        if current < floor:
            deficit += floor - current
            adjusted[key] = floor

    if deficit <= 0:
        return _normalize_weights(adjusted)

    donor_keys = [key for key in adjusted.keys() if key not in floor_map]
    donor_total = sum(adjusted[key] for key in donor_keys) or 1.0
    for key in donor_keys:
        adjusted[key] = max(0.01, adjusted[key] - (adjusted[key] / donor_total) * deficit)
    return _normalize_weights(adjusted)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result
