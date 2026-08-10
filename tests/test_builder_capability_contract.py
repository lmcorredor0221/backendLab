from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.models import (
    AgentExecutionBackend,
    BlueprintArtifact,
    CanvasArtifact,
    CodexLocalProviderConfig,
    DeepSeekProviderConfig,
    DiscoveryArtifact,
    DiscoveryInput,
    EstimationBenchmarkRef,
    EstimationComplexityDriver,
    EstimationConfidenceAdjustmentProposal,
    EstimationReportArtifact,
    EstimationRiskRegisterEntry,
    EstimationScenarioAdjustment,
    EstimationSavingsOpportunity,
    EstimationUncertaintyFactor,
    KnowledgeAccessBackend,
    LLMProviderKey,
    LLMRuntimeSettings,
    MemoryProfile,
    OpenAIProviderConfig,
    ReviewState,
    ToolRecommendationAllowedToolKey,
    ToolRecommendationConfidence,
    ToolRecommendationLLMDecision,
    ToolRecommendationLLMOutput,
    ToolRecommendationPromptInput,
    ToolRecommendationPromptToolOption,
)
from app.services.llm_runtime.builder_contracts import (
    AcceptanceCriterion,
    AgentDesignCritiqueInput,
    AgentDesignInput,
    AgentDesignProposalOutput,
    BusinessRule,
    BlueprintNarrativeOutput,
    CritiqueFinding,
    Dependency,
    DesignCritiqueOutput,
    DiscoveryAnalysisInput,
    DiscoveryAnalysisOutput,
    EstimationRiskAnalysisInput,
    EstimationRiskAnalysisOutput,
    FunctionalRequirement,
    LLMArtifactResult,
    MemoryArchitectureCritiqueInput,
    MemoryArchitectureCritiqueOutput,
    MemoryArchitectureInput,
    MemoryArchitectureRecommendationOutput,
    NonFunctionalRequirement,
    OpenQuestion,
    PrioritizedQuestion,
    RequirementTraceEntry,
    RequirementsDefinitionInput,
    RequirementsDefinitionOutput,
    StructuredInsight,
    ValidationRunJudgmentInput,
    ValidationRunJudgmentOutput,
    ValidationScenarioGenerationInput,
    ValidationScenarioGenerationOutput,
    ValidationScenarioItem,
    ValidationScenarioSimulationInput,
    ValidationSimulationOutput,
)
from app.services.llm_runtime.capability_registry import BuilderCapability, CAPABILITY_SPECS, get_builder_capability_spec
from app.services.llm_runtime.codex_cli.provider_facade import CodexLocalBuilderService
from app.services.llm_runtime.provider_router import BuilderExecutionMode, BuilderProviderFacade
from app.services.openai_builder import DeepSeekBuilderService, OpenAIBuilderService


def build_runtime_settings(
    active_provider: LLMProviderKey,
    *,
    backend: AgentExecutionBackend = AgentExecutionBackend.provider_native,
) -> LLMRuntimeSettings:
    return LLMRuntimeSettings(
        active_provider=active_provider,
        agent_execution_backend=backend,
        knowledge_access_backend=KnowledgeAccessBackend.inline_context,
        openai=OpenAIProviderConfig(
            fast_model="gpt-5.4-mini",
            reasoning_model="gpt-5.5",
            reasoning_effort="low",
            api_key_configured=True,
            available=True,
            status_note="ready",
        ),
        deepseek=DeepSeekProviderConfig(
            base_url="https://api.deepseek.com",
            fast_model="deepseek-v4-flash",
            reasoning_model="deepseek-v4-pro",
            reasoning_effort="high",
            api_key_configured=True,
            available=True,
            status_note="ready",
        ),
        codex_local=CodexLocalProviderConfig(
            command="codex",
            model="gpt-5.5",
            profile="ci4-capabilities",
            executable_found=True,
            available=True,
        ),
    )


def sample_discovery_input() -> DiscoveryInput:
    return DiscoveryInput(
        problem_statement="Estandarizar discovery y blueprint en una sola plataforma Lean.",
        current_user="Arquitecto de soluciones",
        current_process="Analiza discovery en documentos y luego redacta entregables manualmente.",
        desired_outcome="Construir recomendaciones trazables para diseno, tools, memoria y estimacion.",
        autonomy_level="high",
        constraints=["Sin side effects irreversibles", "Mantener aprobaciones humanas visibles"],
        operational_baseline={
            "current_time_spent": "6 horas",
            "current_cost": "Retrabajo tecnico y drift del contexto",
            "frequent_errors": ["Se pierde continuidad entre etapas"],
            "automation_opportunities": ["Analisis", "Diseno", "Memoria"],
        },
        mvp_definition={
            "v1_scope": ["Discovery", "Canvas", "Design", "Tools", "Memory"],
            "out_of_scope": ["Provisioning automatico"],
            "north_star_metric": "Blueprint util en una sesion",
            "non_delegable_decisions": ["Aprobar cambios con side effects"],
        },
    )


def sample_discovery_artifact() -> DiscoveryArtifact:
    payload = sample_discovery_input().model_dump(mode="json")
    payload.update(
        {
            "case_type": "automatizacion",
            "value_statement": "Reducir retrabajo y perdida de trazabilidad entre etapas.",
        }
    )
    return DiscoveryArtifact.model_validate(payload)


def sample_canvas_artifact() -> CanvasArtifact:
    return CanvasArtifact(
        user_goal="Construir un blueprint trazable con memoria hibrida y herramientas minimas.",
        mvp_scope=["Discovery", "Canvas", "Design", "Tools", "Memory"],
        out_of_scope=["Provisioning automatico"],
        success_metric="Blueprint aprobado sin retrabajo mayor",
        primary_risk="Drift del contexto aprobado",
        agent_profile={
            "mission": "Orquestar etapas Lean sin perder trazabilidad.",
            "primary_user": "Arquitecto de soluciones",
            "agent_task": "Analizar, definir, disenar y validar un agente de IA.",
            "allowed_decisions": ["Proponer arquitectura", "Compactar contexto"],
            "prohibited_decisions": ["Ejecutar side effects sin aprobacion"],
            "key_inputs": ["Discovery", "Canvas", "Knowledge base"],
            "expected_outputs": ["Blueprint", "Tools", "Memory"],
            "human_approvals": ["Promocion a implementacion"],
            "success_metrics": ["Blueprint aprobado sin drift"],
        },
    )


def sample_blueprint_artifact() -> BlueprintArtifact:
    return BlueprintArtifact(
        architecture="single_agent_with_skills",
        reasoning_pattern="Plan-and-Execute",
        memory_strategy="session_memory_with_checkpoints",
        tools=[],
        memory_profile=MemoryProfile(
            strategy="session_memory_with_checkpoints",
            storage_layers=["session_state", "vector_store"],
            write_policy="Persist validated checkpoints",
            retrieval_policy="Recover only approved references",
            review_trigger="Missing evidence",
            goal_drift_guard="Compare each proposal against approved discovery and canvas",
        ),
        guardrails=["No inventar datos", "No promover cambios sin aprobacion"],
        readiness_state=ReviewState.partial,
        narrative="Blueprint base con memoria hibrida y tooling minimo.",
    )


def sample_estimation_report() -> EstimationReportArtifact:
    return EstimationReportArtifact(
        assumptions=["Equipo de 2 desarrolladores", "Integraciones acotadas"],
        risk_drivers=["Dependencia del contexto aprobado", "Cobertura de conocimiento"],
        notes=["Base determinista previa al analisis LLM."],
    )


def sample_tool_prompt_input() -> ToolRecommendationPromptInput:
    return ToolRecommendationPromptInput(
        source_session_id=uuid4(),
        source_blueprint_version=2,
        case_classification="automatizacion",
        agent_goal="Construir un blueprint trazable y con minima superficie operativa.",
        primary_user="Arquitecto de soluciones",
        workflow_summary="Discovery -> Define -> Design -> Tools -> Memory -> Validate -> Estimate",
        constraints_summary="Sin side effects irreversibles y con aprobaciones visibles.",
        mandatory_tool_keys=[ToolRecommendationAllowedToolKey.read_system_of_record],
        candidate_tools=[
            ToolRecommendationPromptToolOption(
                tool_key=ToolRecommendationAllowedToolKey.approval_gate,
                tool_label="Approval gate",
                family_key="approval_control",
                capability_covered="Aprobacion humana visible",
                reason="Hay decisiones no delegables.",
            )
        ],
    )


def sample_discovery_analysis_input() -> DiscoveryAnalysisInput:
    return DiscoveryAnalysisInput(
        discovery_capture=sample_discovery_input(),
        analysis_goal="Encontrar riesgos, supuestos y preguntas que impacten Design y Memory.",
    )


def sample_discovery_analysis_output() -> DiscoveryAnalysisOutput:
    return DiscoveryAnalysisOutput(
        summary="El discovery muestra una necesidad clara de continuidad de contexto con aprobacion humana.",
        facts=[
            StructuredInsight(
                key="avoid_stage_drift",
                statement="El usuario quiere evitar drift entre etapas.",
                source_refs=["discovery.problem_statement"],
                confidence=0.94,
            )
        ],
        inferred_needs=[
            StructuredInsight(
                key="governed_memory",
                statement="Se requiere memoria corta y larga gobernada.",
                source_refs=["discovery.desired_outcome", "discovery.current_process"],
                confidence=0.81,
            )
        ],
        assumptions=[
            StructuredInsight(
                key="manageable_document_volume",
                statement="El volumen documental inicial es manejable.",
                source_refs=["discovery.constraints"],
                confidence=0.62,
            )
        ],
        ambiguities=[
            StructuredInsight(
                key="approved_sources",
                statement="No se definieron fuentes documentales aprobadas por workspace.",
                source_refs=["discovery.constraints"],
                confidence=0.58,
            )
        ],
        open_questions=[
            PrioritizedQuestion(
                key="approved_sources",
                question="Que fuentes documentales son aprobadas para retrieval?",
                rationale="Impacta Memory y Tools.",
                priority="high",
                blocking_stages=["memory", "tools"],
                suggested_answer="Listar repositorios, owners y politicas de acceso por workspace.",
            )
        ],
        domain_signals=[
            StructuredInsight(
                key="agent_builder",
                statement="El caso corresponde a un builder de agentes con gates de aprobacion humana.",
                source_refs=["discovery.current_process"],
                confidence=0.77,
            )
        ],
        risk_signals=[
            StructuredInsight(
                key="knowledge_owner",
                statement="Knowledge sin owner aprobado.",
                source_refs=["discovery.constraints"],
                confidence=0.88,
            )
        ],
        sensitive_data_signals=[],
        missing_information=["Fuentes privadas por workspace"],
        evidence_refs=["session.discovery", "knowledge.taxonomy"],
        confidence=0.79,
    )


def sample_requirements_input() -> RequirementsDefinitionInput:
    return RequirementsDefinitionInput(
        discovery=sample_discovery_artifact(),
        canvas=sample_canvas_artifact(),
        known_constraints=["No exponer documentos privados de otros workspaces."],
    )


def sample_requirements_output() -> RequirementsDefinitionOutput:
    return RequirementsDefinitionOutput(
        summary="Se consolidan requisitos funcionales, NFR y reglas de gobierno.",
        measurable_objectives=[
            "Generar una definicion aprobable antes de pasar a Design.",
            "Reducir drift entre etapas con trazabilidad visible.",
        ],
        functional_requirements=[
            FunctionalRequirement(
                key="fr-traceability",
                title="Uso exclusivo de artefactos aprobados",
                requirement="Cada etapa debe usar solo artefactos aprobados.",
                rationale="Evitar drift entre propuestas y aprobaciones.",
                source_refs=["discovery.desired_outcome", "canvas.agent_profile.human_approvals"],
                priority="high",
                acceptance=["El runtime rechaza insumos stale o no aprobados."],
                actor="builder",
                trigger="Promocion entre etapas",
                happy_path="La siguiente etapa consume solo la version aprobada mas reciente.",
                exceptions=["Si el artefacto cambia, el dependiente queda stale."],
            )
        ],
        non_functional_requirements=[
            NonFunctionalRequirement(
                key="nfr-governance",
                title="Trazabilidad medible",
                requirement="La definicion debe conservar provider, prompt version y context fingerprint.",
                rationale="Los artefactos deben ser auditables.",
                source_refs=["session.skill_runs"],
                priority="high",
                acceptance=["Cada corrida registra provider, prompt version y context fingerprint."],
                category="governance",
                metric="trace_fields_coverage",
                target="100%",
            )
        ],
        business_rules=[
            BusinessRule(
                key="rule-human-approval",
                title="Aprobacion humana visible",
                rule="No promover a implementacion sin aprobacion humana.",
                rationale="Hay decisiones no delegables.",
                source_refs=["canvas.agent_profile.human_approvals"],
                priority="high",
                acceptance=["Debe existir evidencia de aprobacion antes de promotion."],
                owner="business_owner",
            )
        ],
        acceptance_criteria=[
            AcceptanceCriterion(
                key="ac-runtime-trace",
                title="Runtime trazable",
                criterion="El LLM devuelve trazas con provider, prompt version y context fingerprint.",
                rationale="La definicion debe poder auditarse.",
                source_refs=["session.skill_runs"],
                priority="high",
                acceptance=["La evidencia de skill runs contiene metadata de corrida."],
                requirement_keys=["fr-traceability", "nfr-governance"],
            )
        ],
        dependencies=[
            Dependency(
                key="dep-knowledge-memory",
                title="Knowledge memory indexado",
                dependency="Knowledge memory indexado",
                rationale="La etapa Design reutiliza conocimiento gobernado.",
                source_refs=["memory.knowledge_profile"],
                priority="medium",
                acceptance=["La fuente de conocimiento existe y tiene owner."],
                dependency_type="knowledge",
                owner="platform_owner",
            )
        ],
        assumptions=[],
        open_questions=[
            OpenQuestion(
                key="question-private-sources",
                title="Fuentes privadas por workspace",
                question="Como se gestionan las fuentes privadas por workspace?",
                rationale="Impacta diseno, memoria y seguridad.",
                source_refs=["discovery.constraints"],
                priority="medium",
                acceptance=["La respuesta debe definir owner y politica de acceso."],
                blocking=False,
                impacted_sections=["dependencies", "design", "memory"],
                suggested_answer="Documentar owner, sensitivity y filtro por workspace.",
            )
        ],
        traceability=[
            RequirementTraceEntry(
                key="trace-fr-traceability",
                requirement_key="fr-traceability",
                source_ref="discovery.desired_outcome",
                rationale="El requisito nace del objetivo de continuidad entre etapas.",
                coverage_status="covered",
            )
        ],
        evidence_refs=["session.discovery", "session.canvas", "knowledge.requirements"],
        confidence=0.83,
        canvas_projection=sample_canvas_artifact(),
    )


def sample_agent_design_input() -> AgentDesignInput:
    return AgentDesignInput(
        discovery=sample_discovery_artifact(),
        canvas=sample_canvas_artifact(),
        current_blueprint=sample_blueprint_artifact(),
        requirement_digest=["Usar solo artefactos aprobados", "Mantener aprobaciones visibles"],
    )


def sample_agent_design_output() -> AgentDesignProposalOutput:
    return AgentDesignProposalOutput(
        architecture="single_agent_with_skills",
        reasoning_pattern="Plan-and-Execute",
        memory_strategy="session_memory_with_checkpoints",
        coordination_model="single_supervisor",
        tooling_principles=["Minimizar herramientas", "Mantener gates humanos"],
        design_decisions=[],
        open_questions=["Definir contrato final de memoria larga por workspace"],
        narrative="Se propone un supervisor unico con retrieval gobernado y checkpoints por etapa.",
    )


def sample_design_critique_input() -> AgentDesignCritiqueInput:
    return AgentDesignCritiqueInput(
        discovery=sample_discovery_artifact(),
        canvas=sample_canvas_artifact(),
        proposal=sample_agent_design_output(),
    )


def sample_design_critique_output() -> DesignCritiqueOutput:
    return DesignCritiqueOutput(
        overall_status="needs_revision",
        summary="La propuesta es consistente pero necesita cerrar ownership de fuentes privadas.",
        findings=[
            CritiqueFinding(
                finding_key="workspace-knowledge",
                title="Workspace knowledge pendiente",
                severity="warning",
                detail="Falta explicitar como se aislan fuentes privadas por tenant.",
                suggested_action="Agregar politica de aislamiento y lineage por workspace.",
                source_refs=["memory.open_questions"],
            )
        ],
        missing_evidence=["Contrato final de visibility por workspace"],
    )


def sample_memory_input() -> MemoryArchitectureInput:
    return MemoryArchitectureInput(
        blueprint=sample_blueprint_artifact(),
        discovery=sample_discovery_artifact(),
        canvas=sample_canvas_artifact(),
        approved_tool_names=["knowledge_retrieval", "approval_gate"],
    )


def sample_memory_output() -> MemoryArchitectureRecommendationOutput:
    return MemoryArchitectureRecommendationOutput(
        memory_strategy="hybrid_context_memory",
        short_term_strategy="checkpoint_summary_cache",
        long_term_strategy="governed_docs_plus_session_artifacts",
        retrieval_strategy="stage_affinity_hybrid_search",
        storage_layers=["session_state", "vector_store", "artifact_registry"],
        write_policy="Persistir solo decisiones aprobadas y artefactos canonicos.",
        pruning_policy="Compactar por checkpoint y recuperar por referencia.",
        security_notes=["Filtrar por workspace_id en todo acceso a conocimiento privado."],
        rationale="La estrategia separa memoria del journey y memoria del agente objetivo.",
    )


def sample_memory_critique_input() -> MemoryArchitectureCritiqueInput:
    return MemoryArchitectureCritiqueInput(
        blueprint=sample_blueprint_artifact(),
        proposal=sample_memory_output(),
        approved_tool_names=["knowledge_retrieval", "approval_gate"],
    )


def sample_memory_critique_output() -> MemoryArchitectureCritiqueOutput:
    return MemoryArchitectureCritiqueOutput(
        overall_status="accepted",
        summary="La propuesta de memoria cubre budgets, retrieval y aislamiento.",
        findings=[],
    )


def sample_validation_scenario_generation_input() -> ValidationScenarioGenerationInput:
    return ValidationScenarioGenerationInput(
        blueprint=sample_blueprint_artifact(),
        discovery=sample_discovery_artifact(),
        canvas=sample_canvas_artifact(),
        focus_areas=["retrieval", "approvals", "traceability"],
    )


def sample_validation_scenario_item() -> ValidationScenarioItem:
    return ValidationScenarioItem(
        scenario_key="validate-retrieval-approval",
        title="Escenario de retrieval con aprobacion humana",
        objective="Comprobar que el agente recupera evidencia y respeta approval gates.",
        steps=["Cargar discovery aprobado", "Recuperar conocimiento", "Solicitar aprobacion"],
        expected_outcomes=["Evidencia citada", "Gate visible"],
        failure_signals=["Sin citations", "Escritura sin aprobacion"],
        priority="high",
    )


def sample_validation_scenario_generation_output() -> ValidationScenarioGenerationOutput:
    return ValidationScenarioGenerationOutput(
        summary="Se generan escenarios de retrieval, aprobaciones y drift.",
        scenarios=[sample_validation_scenario_item()],
    )


def sample_validation_simulation_input() -> ValidationScenarioSimulationInput:
    return ValidationScenarioSimulationInput(
        blueprint=sample_blueprint_artifact(),
        scenario=sample_validation_scenario_item(),
    )


def sample_validation_simulation_output() -> ValidationSimulationOutput:
    return ValidationSimulationOutput(
        scenario_key="validate-retrieval-approval",
        result_status="needs_revision",
        simulated_transcript=["Usuario solicita blueprint", "Agente recupera evidencia", "Agente pide aprobacion"],
        observed_decisions=["Se activa approval_gate"],
        tool_interactions=["knowledge_retrieval", "approval_gate"],
        issues=["Falta explicar la politica de fallback ante ausencia de evidencia."],
    )


def sample_validation_judgment_input() -> ValidationRunJudgmentInput:
    return ValidationRunJudgmentInput(
        simulation=sample_validation_simulation_output(),
        blueprint=sample_blueprint_artifact(),
    )


def sample_validation_judgment_output() -> ValidationRunJudgmentOutput:
    return ValidationRunJudgmentOutput(
        scenario_key="validate-retrieval-approval",
        judgment="needs_revision",
        summary="La corrida conserva aprobaciones pero debe reforzar la explicacion de ausencia de evidencia.",
        findings=[
            CritiqueFinding(
                finding_key="fallback-explanation",
                title="Fallback no explicado",
                severity="warning",
                detail="El transcript no deja claro como se comunica la ausencia de evidencia.",
                suggested_action="Agregar mensaje estandar de needs-resolution.",
            )
        ],
        score=78,
    )


def sample_estimation_risk_input() -> EstimationRiskAnalysisInput:
    return EstimationRiskAnalysisInput(
        blueprint=sample_blueprint_artifact(),
        estimation_report=sample_estimation_report(),
        pricing_summary=["Profile=openai_standard model=gpt-5-mini"],
        validation_summary=["Validate artifact approved with simulation coverage."],
        workspace_calibration_summary=["Coverage=66% band_hit_rate=60%."],
        benchmark_hints=["Benchmarks internos de proyectos similares"],
    )


def sample_estimation_risk_output() -> EstimationRiskAnalysisOutput:
    return EstimationRiskAnalysisOutput(
        summary="La estimacion es viable, pero depende de cerrar retrieval y ownership de conocimiento.",
        complexity_drivers=[
            EstimationComplexityDriver(
                driver_key="data",
                title="Knowledge y retrieval",
                workstream_key="data",
                impact_level="high",
                summary="La capa semantica y el ownership del corpus siguen siendo el principal driver de esfuerzo.",
                evidence_refs=["knowledge.visibility_policy"],
            )
        ],
        risk_register=[
            EstimationRiskRegisterEntry(
                risk_key="knowledge-private",
                title="Knowledge privado no clasificado",
                severity="high",
                likelihood="medium",
                impact="Puede mover governance, retrieval y validacion de acceso.",
                mitigation="Aplicar filtros por workspace y source lineage.",
                evidence_refs=["knowledge.visibility_policy"],
            )
        ],
        uncertainty_factors=[
            EstimationUncertaintyFactor(
                factor_key="retrieval-owner",
                title="Owner de retrieval por definir",
                category="knowledge",
                impact_area="confidence",
                summary="Mientras no exista owner y refresh policy la banda debe ampliarse.",
                evidence_refs=["memory.refresh_policy"],
            )
        ],
        benchmark_refs=[
            EstimationBenchmarkRef(
                benchmark_key="workspace-actual-1",
                title="Proyecto calibrado",
                source_kind="workspace_actuals",
                source_ref="workspace://estimation-runs/1",
                sample_size=1,
                captured_at="2026-07-01T10:00:00",
                freshness="reciente",
                summary="Proyecto similar con MAPE de costo controlado.",
                workspace_scoped=True,
            )
        ],
        scenario_adjustments=[
            EstimationScenarioAdjustment(
                scenario_key="optimistic",
                hours_multiplier=0.96,
                duration_multiplier=0.96,
                cost_multiplier=0.95,
                rationale="Menor retrabajo.",
            ),
            EstimationScenarioAdjustment(
                scenario_key="base",
                hours_multiplier=1.0,
                duration_multiplier=1.0,
                cost_multiplier=1.0,
                rationale="Base deterministica.",
            ),
            EstimationScenarioAdjustment(
                scenario_key="conservative",
                hours_multiplier=1.1,
                duration_multiplier=1.08,
                cost_multiplier=1.1,
                rationale="Mayor hardening.",
            ),
        ],
        savings_opportunities=[
            EstimationSavingsOpportunity(
                opportunity_key="close-governance",
                title="Cerrar governance temprano",
                summary="Baja retrabajo en datos y validacion.",
                expected_impact="Reduce supervision correctiva.",
                prerequisites=["Owner de corpus", "Reglas de visibilidad"],
                evidence_refs=["validate.simulation"],
            )
        ],
        assumptions=["El corpus de Docs ya esta indexado y gobernado."],
        questions=[],
        evidence_refs=["knowledge.visibility_policy"],
        confidence_adjustment_proposal=EstimationConfidenceAdjustmentProposal(
            proposed_score_delta=-4,
            proposed_uncertainty_band_delta=6,
            rationale="Persisten decisiones abiertas de knowledge.",
            evidence_refs=["memory.refresh_policy"],
        ),
    )


@dataclass
class FakeCapabilityService:
    provider_key: str
    results: dict[str, LLMArtifactResult]
    available: bool = True
    calls: dict[str, int] = field(default_factory=dict)

    def can_attempt(self) -> bool:
        return True

    def is_available(self) -> bool:
        return self.available

    def provider_summary(self) -> dict[str, str | bool]:
        return {
            "provider": self.provider_key,
            "mode": "fake",
            "configured": True,
            "sdk_ready": self.available,
            "fast_model": f"{self.provider_key}-fast",
            "reasoning_model": f"{self.provider_key}-reasoning",
        }

    def _result(self, capability_key: str) -> LLMArtifactResult:
        self.calls[capability_key] = self.calls.get(capability_key, 0) + 1
        return self.results[capability_key]

    def normalize_discovery(self, payload: DiscoveryInput, *, context_bundle=None) -> LLMArtifactResult:
        del payload, context_bundle
        return self._result("normalize_discovery")

    def analyze_discovery(self, payload: DiscoveryAnalysisInput, *, context_bundle=None) -> LLMArtifactResult:
        del payload, context_bundle
        return self._result("analyze_discovery")

    def build_canvas(self, discovery: DiscoveryArtifact, *, context_bundle=None) -> LLMArtifactResult:
        del discovery, context_bundle
        return self._result("build_canvas")

    def define_requirements(self, payload: RequirementsDefinitionInput, *, context_bundle=None) -> LLMArtifactResult:
        del payload, context_bundle
        return self._result("define_requirements")

    def synthesize_blueprint_narrative(
        self,
        discovery: DiscoveryArtifact,
        canvas: CanvasArtifact,
        blueprint: BlueprintArtifact,
        *,
        context_bundle=None,
    ) -> LLMArtifactResult:
        del discovery, canvas, blueprint, context_bundle
        return self._result("synthesize_blueprint_narrative")

    def propose_agent_design(self, payload: AgentDesignInput, *, context_bundle=None) -> LLMArtifactResult:
        del payload, context_bundle
        return self._result("propose_agent_design")

    def critique_agent_design(self, payload: AgentDesignCritiqueInput, *, context_bundle=None) -> LLMArtifactResult:
        del payload, context_bundle
        return self._result("critique_agent_design")

    def recommend_minimal_tools(self, payload: ToolRecommendationPromptInput, *, context_bundle=None) -> LLMArtifactResult:
        del payload, context_bundle
        return self._result("recommend_minimal_tools")

    def recommend_memory_architecture(self, payload: MemoryArchitectureInput, *, context_bundle=None) -> LLMArtifactResult:
        del payload, context_bundle
        return self._result("recommend_memory_architecture")

    def critique_memory_architecture(
        self,
        payload: MemoryArchitectureCritiqueInput,
        *,
        context_bundle=None,
    ) -> LLMArtifactResult:
        del payload, context_bundle
        return self._result("critique_memory_architecture")

    def generate_validation_scenarios(
        self,
        payload: ValidationScenarioGenerationInput,
        *,
        context_bundle=None,
    ) -> LLMArtifactResult:
        del payload, context_bundle
        return self._result("generate_validation_scenarios")

    def simulate_validation_scenario(
        self,
        payload: ValidationScenarioSimulationInput,
        *,
        context_bundle=None,
    ) -> LLMArtifactResult:
        del payload, context_bundle
        return self._result("simulate_validation_scenario")

    def judge_validation_run(self, payload: ValidationRunJudgmentInput, *, context_bundle=None) -> LLMArtifactResult:
        del payload, context_bundle
        return self._result("judge_validation_run")

    def analyze_estimation_risks(self, payload: EstimationRiskAnalysisInput, *, context_bundle=None) -> LLMArtifactResult:
        del payload, context_bundle
        return self._result("analyze_estimation_risks")


def _fake_results() -> dict[str, LLMArtifactResult]:
    return {
        "normalize_discovery": LLMArtifactResult(artifact=sample_discovery_artifact()),
        "analyze_discovery": LLMArtifactResult(artifact=sample_discovery_analysis_output()),
        "build_canvas": LLMArtifactResult(artifact=sample_canvas_artifact()),
        "define_requirements": LLMArtifactResult(artifact=sample_requirements_output()),
        "synthesize_blueprint_narrative": LLMArtifactResult(
            artifact=BlueprintNarrativeOutput(narrative="Narrativa sintetizada")
        ),
        "propose_agent_design": LLMArtifactResult(artifact=sample_agent_design_output()),
        "critique_agent_design": LLMArtifactResult(artifact=sample_design_critique_output()),
        "recommend_minimal_tools": LLMArtifactResult(
            artifact=ToolRecommendationLLMOutput(
                summary="Set minimo de tools aprobado.",
                confidence=ToolRecommendationConfidence(overall=0.74, band="medium", rationale="Cobertura suficiente."),
                tool_decisions=[
                    ToolRecommendationLLMDecision(
                        tool_key=ToolRecommendationAllowedToolKey.read_system_of_record,
                        classification="mandatory",
                        decision_reason="Es la fuente principal del workflow.",
                        confidence=0.88,
                    )
                ],
            )
        ),
        "recommend_memory_architecture": LLMArtifactResult(artifact=sample_memory_output()),
        "critique_memory_architecture": LLMArtifactResult(artifact=sample_memory_critique_output()),
        "generate_validation_scenarios": LLMArtifactResult(artifact=sample_validation_scenario_generation_output()),
        "simulate_validation_scenario": LLMArtifactResult(artifact=sample_validation_simulation_output()),
        "judge_validation_run": LLMArtifactResult(artifact=sample_validation_judgment_output()),
        "analyze_estimation_risks": LLMArtifactResult(artifact=sample_estimation_risk_output()),
    }


def _capability_invocations():
    return [
        ("analyze_discovery", lambda facade: facade.analyze_discovery(sample_discovery_analysis_input()), DiscoveryAnalysisOutput),
        ("define_requirements", lambda facade: facade.define_requirements(sample_requirements_input()), RequirementsDefinitionOutput),
        ("propose_agent_design", lambda facade: facade.propose_agent_design(sample_agent_design_input()), AgentDesignProposalOutput),
        ("critique_agent_design", lambda facade: facade.critique_agent_design(sample_design_critique_input()), DesignCritiqueOutput),
        ("recommend_memory_architecture", lambda facade: facade.recommend_memory_architecture(sample_memory_input()), MemoryArchitectureRecommendationOutput),
        ("critique_memory_architecture", lambda facade: facade.critique_memory_architecture(sample_memory_critique_input()), MemoryArchitectureCritiqueOutput),
        ("generate_validation_scenarios", lambda facade: facade.generate_validation_scenarios(sample_validation_scenario_generation_input()), ValidationScenarioGenerationOutput),
        ("simulate_validation_scenario", lambda facade: facade.simulate_validation_scenario(sample_validation_simulation_input()), ValidationSimulationOutput),
        ("judge_validation_run", lambda facade: facade.judge_validation_run(sample_validation_judgment_input()), ValidationRunJudgmentOutput),
        ("analyze_estimation_risks", lambda facade: facade.analyze_estimation_risks(sample_estimation_risk_input()), EstimationRiskAnalysisOutput),
    ]


def test_capability_registry_defines_policy_and_prompt_version_for_ci4_capabilities() -> None:
    expected = {
        BuilderCapability.analyze_discovery,
        BuilderCapability.define_requirements,
        BuilderCapability.propose_agent_design,
        BuilderCapability.critique_agent_design,
        BuilderCapability.recommend_minimal_tools,
        BuilderCapability.recommend_memory_architecture,
        BuilderCapability.critique_memory_architecture,
        BuilderCapability.generate_validation_scenarios,
        BuilderCapability.simulate_validation_scenario,
        BuilderCapability.judge_validation_run,
        BuilderCapability.analyze_estimation_risks,
    }

    assert expected.issubset(set(CAPABILITY_SPECS))
    for capability in expected:
        spec = get_builder_capability_spec(capability)
        assert spec.prompt_version.endswith(".v1")
        assert spec.timeout_ms > 0
        assert spec.max_retries >= 0
        assert spec.fallback_policy
        assert spec.llm_required is True


def test_capability_registry_instructs_llms_to_generate_guided_questions() -> None:
    guided_capabilities = [
        BuilderCapability.analyze_discovery,
        BuilderCapability.define_requirements,
        BuilderCapability.propose_agent_design,
        BuilderCapability.recommend_minimal_tools,
        BuilderCapability.recommend_memory_architecture,
        BuilderCapability.generate_validation_scenarios,
        BuilderCapability.analyze_estimation_risks,
    ]

    for capability in guided_capabilities:
        instruction = get_builder_capability_spec(capability).task_instruction
        assert "suggested_answer" in instruction
        assert "answer_options" in instruction
        assert "ACP" in instruction


def test_define_requirements_timeout_budget_covers_long_running_codex_reasoning() -> None:
    spec = get_builder_capability_spec(BuilderCapability.define_requirements)

    assert spec.task_kind == "requirements_definition"
    assert spec.preferred_model == "reasoning"
    assert spec.timeout_ms >= 240000


@pytest.mark.parametrize("active_provider", [LLMProviderKey.openai, LLMProviderKey.deepseek, LLMProviderKey.codex_local])
def test_facade_executes_ci4_capabilities_with_typed_outputs_for_all_active_providers(
    active_provider: LLMProviderKey,
) -> None:
    runtime_settings = build_runtime_settings(active_provider)
    facade = BuilderProviderFacade(
        runtime_settings,
        openai_service=FakeCapabilityService("openai", _fake_results()),
        deepseek_service=FakeCapabilityService("deepseek", _fake_results()),
        codex_service=FakeCapabilityService("codex_local", _fake_results()),
    )

    for capability_key, invoke, output_model in _capability_invocations():
        result = invoke(facade)
        assert isinstance(result.artifact, output_model)
        assert result.capability_key == capability_key
        assert result.prompt_version == get_builder_capability_spec(BuilderCapability(capability_key)).prompt_version
        assert result.capability_policy["fallback_policy"]
        assert result.execution_mode == BuilderExecutionMode.primary


def test_facade_keeps_primary_result_when_shadow_detects_semantic_divergence() -> None:
    runtime_settings = build_runtime_settings(
        LLMProviderKey.openai,
        backend=AgentExecutionBackend.shadow_codex_cli,
    )
    primary_results = _fake_results()
    shadow_results = _fake_results()
    shadow_results["analyze_discovery"] = LLMArtifactResult(
        artifact=DiscoveryAnalysisOutput(
            summary="La interpretacion shadow detecta una estrategia distinta.",
            facts=[
                StructuredInsight(
                    key="retrieval_cost",
                    statement="El principal riesgo seria el costo de retrieval.",
                    source_refs=["memory.sources"],
                    confidence=0.7,
                )
            ],
        )
    )
    facade = BuilderProviderFacade(
        runtime_settings,
        openai_service=FakeCapabilityService("openai", primary_results),
        deepseek_service=FakeCapabilityService("deepseek", primary_results),
        codex_service=FakeCapabilityService("codex_local", shadow_results),
    )

    result = facade.analyze_discovery(sample_discovery_analysis_input())

    assert isinstance(result.artifact, DiscoveryAnalysisOutput)
    assert result.provider_key == "openai"
    assert result.execution_mode == "shadow"
    assert result.rollout_comparison["semantic_divergence"] is True
    assert "shadow comparison" in (result.warning or "").lower()


def test_openai_builder_analyze_discovery_returns_typed_output_with_metadata() -> None:
    runtime_settings = build_runtime_settings(LLMProviderKey.openai)
    service = OpenAIBuilderService(runtime_settings)

    class FakeResponses:
        def parse(self, **kwargs):
            del kwargs
            return SimpleNamespace(
                id="resp-openai-1",
                status="completed",
                usage={"input_tokens": 120, "output_tokens": 48, "total_tokens": 168},
                output_parsed=sample_discovery_analysis_output(),
            )

    service._client = SimpleNamespace(responses=FakeResponses())
    result = service.analyze_discovery(sample_discovery_analysis_input())

    assert isinstance(result.artifact, DiscoveryAnalysisOutput)
    assert result.capability_key == "analyze_discovery"
    assert result.prompt_version == "analyze_discovery.v1"
    assert result.request_id == "resp-openai-1"
    assert result.finish_reason == "completed"
    assert result.schema_validation_status == "valid"
    assert result.token_usage["total_tokens"] == 168


def test_deepseek_builder_define_requirements_repairs_wrapped_payload() -> None:
    runtime_settings = build_runtime_settings(LLMProviderKey.deepseek)
    service = DeepSeekBuilderService(runtime_settings)
    service._client = object()

    def fake_request_structured_completion_payload(**kwargs):
        del kwargs
        return (
            {"result": sample_requirements_output().model_dump(mode="json")},
            {
                "request_id": "deepseek-req-1",
                "finish_reason": "stop",
                "token_usage": {"prompt_tokens": 90, "completion_tokens": 35, "total_tokens": 125},
            },
        )

    service._request_structured_completion_payload = fake_request_structured_completion_payload  # type: ignore[method-assign]
    result = service.define_requirements(sample_requirements_input())

    assert isinstance(result.artifact, RequirementsDefinitionOutput)
    assert result.request_id == "deepseek-req-1"
    assert result.schema_validation_status.startswith("repaired_")
    assert result.token_usage["total_tokens"] == 125


def test_codex_builder_generate_validation_scenarios_reads_runtime_audit_metadata() -> None:
    runtime_settings = build_runtime_settings(LLMProviderKey.codex_local, backend=AgentExecutionBackend.codex_cli)
    service = CodexLocalBuilderService(runtime_settings)

    service.execution_service.execute_structured_prompt = (  # type: ignore[method-assign]
        lambda **kwargs: sample_validation_scenario_generation_output()
    )
    service.execution_service.read_last_known_result = (  # type: ignore[method-assign]
        lambda: {"run_id": "codex-run-1", "status": "succeeded", "selected_model": "gpt-5.5"}
    )

    result = service.generate_validation_scenarios(sample_validation_scenario_generation_input())

    assert isinstance(result.artifact, ValidationScenarioGenerationOutput)
    assert result.capability_key == "generate_validation_scenarios"
    assert result.request_id == "codex-run-1"
    assert result.finish_reason == "succeeded"
    assert result.model_name == "gpt-5.5"
    assert result.prompt_version == "generate_validation_scenarios.v1"
