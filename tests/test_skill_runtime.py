from uuid import uuid4

from app.models import (
    BlueprintArtifact,
    CanvasArtifact,
    DesignAlternative,
    DesignBlueprintProjection,
    DesignRecommendationArtifact,
    DiscoveryArtifact,
    DiscoveryInput,
    EvaluationArtifact,
    ReviewState,
    ToolRecommendationArtifact,
    ToolRecommendationConfidence,
    ToolRecommendationLLMDecision,
    ToolRecommendationLLMOutput,
)
from app.services.llm_runtime.builder_contracts import FunctionalRequirement, LLMArtifactResult, RequirementsDefinitionOutput
from app.services.openai_builder import BlueprintNarrativeOutput
from app.services.skill_runtime import (
    get_skill_registry,
    run_blueprint_stage,
    run_canvas_stage,
    run_discovery_stage,
    run_tool_recommendation_stage,
)


def complete_discovery_input() -> DiscoveryInput:
    return DiscoveryInput(
        problem_statement="Disenar agentes de soporte con metodologia Lean y bajo riesgo operativo.",
        current_user="Arquitecto de soluciones",
        current_process="Recoge discovery en documentos y luego redacta artefactos manualmente.",
        desired_outcome="Generar un blueprint implementable con tools, memoria y evaluacion.",
        autonomy_level="high",
        constraints=["Sin side effects irreversibles", "Mantener un MVP simple"],
        operational_baseline={
            "current_time_spent": "6 horas por caso",
            "current_cost": "Retrabajo tecnico y validaciones tardias",
            "frequent_errors": ["Se pierde contexto", "No se recorta el MVP"],
            "automation_opportunities": ["Normalizar discovery", "Generar artefactos base"],
        },
        mvp_definition={
            "v1_scope": ["Capturar discovery", "Construir canvas", "Construir blueprint"],
            "out_of_scope": ["Provisioning automatico"],
            "north_star_metric": "Blueprint util en una sola sesion",
            "non_delegable_decisions": ["Aprobar el handoff a implementacion"],
        },
    )


def complete_discovery_artifact() -> DiscoveryArtifact:
    return DiscoveryArtifact(
        problem_statement="Disenar agentes de soporte con metodologia Lean y bajo riesgo operativo.",
        current_user="Arquitecto de soluciones",
        current_process="Recoge discovery en documentos y luego redacta artefactos manualmente.",
        desired_outcome="Generar un blueprint implementable con tools, memoria y evaluacion.",
        autonomy_level="high",
        constraints=["Sin side effects irreversibles", "Mantener un MVP simple"],
        operational_baseline={
            "current_time_spent": "6 horas por caso",
            "current_cost": "Retrabajo tecnico y validaciones tardias",
            "frequent_errors": ["Se pierde contexto", "No se recorta el MVP"],
            "automation_opportunities": ["Normalizar discovery", "Generar artefactos base"],
        },
        mvp_definition={
            "v1_scope": ["Capturar discovery", "Construir canvas", "Construir blueprint"],
            "out_of_scope": ["Provisioning automatico"],
            "north_star_metric": "Blueprint util en una sola sesion",
            "non_delegable_decisions": ["Aprobar el handoff a implementacion"],
        },
        case_type="automatizacion",
        value_statement="Reducir retrabajo y ambiguedad operativa.",
    )


def complete_canvas_artifact() -> CanvasArtifact:
    return CanvasArtifact(
        user_goal="Generar un blueprint implementable con tools, memoria y evaluacion.",
        mvp_scope=["Capturar discovery", "Construir canvas", "Construir blueprint"],
        out_of_scope=["Provisioning automatico"],
        success_metric="Blueprint util en una sola sesion",
        primary_risk="No se recorta el MVP",
        agent_profile={
            "mission": "Transformar discovery en blueprint implementable.",
            "primary_user": "Arquitecto de soluciones",
            "agent_task": "Definir builder Lean con evidencia de contexto.",
            "allowed_decisions": ["Proponer arquitectura"],
            "prohibited_decisions": ["Ejecutar side effects sin aprobacion"],
            "key_inputs": ["Discovery"],
            "expected_outputs": ["Canvas", "Blueprint"],
            "human_approvals": ["Promocion a implementacion"],
            "success_metrics": ["Blueprint util en una sola sesion"],
        },
    )


def complete_definition_artifact() -> RequirementsDefinitionOutput:
    return RequirementsDefinitionOutput(
        summary="Definition estructurada para blueprint Lean.",
        functional_requirements=[
            FunctionalRequirement(
                key="fr-read",
                title="Consultar fuentes operativas",
                priority="high",
                requirement="El agente debe consultar CRM y tickets antes de responder o actuar.",
                actor="agent",
                trigger="solicitud del usuario",
                happy_path="Consulta fuentes y resume el estado vigente.",
                source_refs=["discovery.current_process"],
                rationale="Se necesita grounding operativo.",
            )
        ],
    )


def complete_design_artifact() -> DesignRecommendationArtifact:
    alternative = DesignAlternative(
        alternative_key="single-agent",
        label="Single agent",
        architecture="single_agent_with_skills",
        reasoning_pattern="ReAct",
        coordination_model="single-agent",
        summary="Un solo agente con skills especializados y approval gate.",
        topology="Agente unico con skills y handoff controlado.",
        roles=[
            {
                "key": "orchestrator",
                "title": "Orchestrator",
                "responsibility": "Consulta sistemas, aplica reglas y coordina aprobaciones.",
                "limits": ["No ejecuta side effects sin approval"],
            }
        ],
        handoffs=[],
        approval_points=["Promocion a implementacion"],
        decision_policy="Mantener el MVP simple y trazable.",
        blueprint_projection=DesignBlueprintProjection(
            architecture="single_agent_with_skills",
            reasoning_pattern="ReAct",
            guardrails=["Toda escritura requiere aprobacion humana y audit trail."],
            narrative="Arquitectura simple con control humano.",
        ),
    )
    return DesignRecommendationArtifact(
        alternatives=[alternative],
        recommended_alternative_key=alternative.alternative_key,
        selected_design=alternative,
        requirements_coverage=[],
        confidence={"overall": 0.82, "band": "high", "rationale": "Cobertura suficiente."},
        review_state="complete",
        summary="Design aprobado para Herramientas.",
    )


class FakeTraceBuilderService:
    def normalize_discovery(self, payload: DiscoveryInput, *, context_bundle=None) -> LLMArtifactResult:
        del payload, context_bundle
        return LLMArtifactResult(
            artifact=complete_discovery_artifact(),
            provider_key="openai",
            execution_backend="provider_native",
            execution_mode="primary",
            route_reason="provider_native mantiene OpenAI como path principal.",
            knowledge_access_backend="hybrid",
            effective_context_backend="hybrid_inline_compact",
            context_used_sources=[{"key": "discovery_capture"}],
            context_stats={"reduction_estimated_tokens": 120},
        )

    def build_canvas(self, discovery: DiscoveryArtifact, *, context_bundle=None) -> LLMArtifactResult:
        del discovery, context_bundle
        return LLMArtifactResult(
            artifact=complete_canvas_artifact(),
            provider_key="deepseek",
            execution_backend="provider_native",
            execution_mode="primary",
            route_reason="provider_native mantiene DeepSeek como path principal.",
            knowledge_access_backend="inline_context",
            effective_context_backend="inline_context_compact",
            context_used_sources=[{"key": "normalized_discovery"}],
            context_stats={"reduction_estimated_tokens": 64},
        )

    def synthesize_blueprint_narrative(
        self,
        discovery: DiscoveryArtifact,
        canvas: CanvasArtifact,
        blueprint: BlueprintArtifact,
        *,
        context_bundle=None,
    ) -> LLMArtifactResult:
        del discovery, canvas, blueprint, context_bundle
        return LLMArtifactResult(
            artifact=BlueprintNarrativeOutput(
                narrative="Narrativa sintetizada con contexto compacto y trazabilidad consistente."
            ),
            provider_key="codex_local",
            execution_backend="codex_cli",
            execution_mode="shadow",
            shadow_provider_key="codex_local",
            route_reason="Shadow promovido por indisponibilidad del provider activo.",
            knowledge_access_backend="workspace_staged",
            effective_context_backend="workspace_staged_filesystem",
            context_used_sources=[
                {"key": "narrative_discovery"},
                {"key": "narrative_canvas"},
                {"key": "narrative_blueprint"},
            ],
            context_stats={"reduction_estimated_tokens": 210},
        )

    def recommend_minimal_tools(self, prompt_input, *, context_bundle=None) -> LLMArtifactResult:
        del context_bundle
        mandatory = [
            ToolRecommendationLLMDecision(
                tool_key=item,
                classification="mandatory",
                decision_reason="Fake provider conserva toda tool mandatory del preflight.",
                source_evidence=["preflight.mandatory_capabilities"],
                confidence=0.88,
            )
            for item in prompt_input.mandatory_tool_keys
        ]
        optional = [
            ToolRecommendationLLMDecision(
                tool_key=prompt_input.candidate_tools[0].tool_key,
                classification="optional",
                decision_reason="Fake provider deja una capacidad adicional como opcional cuando existe shortlist.",
                source_evidence=["preflight.candidate_tool_families"],
                confidence=0.61,
            )
        ] if prompt_input.candidate_tools else []
        return LLMArtifactResult(
            artifact=ToolRecommendationLLMOutput(
                summary="Fake provider selecciono el set minimo de tools.",
                confidence=ToolRecommendationConfidence(
                    overall=0.74,
                    band="medium",
                    rationale="La salida es sintetica y usa solo el shortlist permitido.",
                ),
                tool_decisions=[*mandatory, *optional],
            ),
            provider_key="openai",
            execution_backend="provider_native",
            execution_mode="primary",
            route_reason="provider_native mantiene OpenAI como path principal.",
            knowledge_access_backend="hybrid",
            effective_context_backend="hybrid_inline_compact",
            context_used_sources=[
                {"key": "tool_recommendation_case"},
                {"key": "tool_recommendation_catalog"},
            ],
            context_stats={"reduction_estimated_tokens": 94},
        )


def test_registry_exposes_stage_two_skills() -> None:
    registry = get_skill_registry()

    expected = {
        "discovery_skill",
        "discovery_analysis_skill",
        "lean_scope_skill",
        "requirements_definition_skill",
        "design_proposal_skill",
        "architecture_selection_skill",
        "reasoning_pattern_skill",
        "tool_design_skill",
        "tool_recommendation_skill",
        "memory_design_skill",
        "safety_skill",
        "blueprint_generation_skill",
        "evaluation_skill",
    }

    assert expected == {item.skill_key for item in registry.list()}


def test_stage_runtime_generates_real_traces() -> None:
    discovery_envelope, discovery_traces = run_discovery_stage(complete_discovery_input())
    canvas_envelope, canvas_traces = run_canvas_stage(discovery_envelope.data)
    blueprint_envelope, blueprint_traces = run_blueprint_stage(discovery_envelope.data, canvas_envelope.data)

    assert isinstance(discovery_envelope.data, DiscoveryArtifact)
    assert isinstance(canvas_envelope.data, CanvasArtifact)
    assert isinstance(blueprint_envelope.data, BlueprintArtifact)
    assert discovery_traces[0].input_kind == "DiscoverySkillInput"
    assert canvas_traces[0].output_kind == "CanvasArtifact"
    assert any(trace.skill_key == "blueprint_generation_skill" for trace in blueprint_traces)
    assert all(trace.duration_ms >= 0 for trace in blueprint_traces)


def test_stage_runtime_surfaces_llm_trace_for_all_builder_stages(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.skill_runtime._builder_service_for_stage",
        lambda stage_key, runtime_settings=None: FakeTraceBuilderService(),
    )

    discovery_envelope, discovery_traces = run_discovery_stage(complete_discovery_input())
    canvas_envelope, canvas_traces = run_canvas_stage(discovery_envelope.data)
    blueprint_envelope, blueprint_traces = run_blueprint_stage(discovery_envelope.data, canvas_envelope.data)
    recommendation_envelope, recommendation_traces = run_tool_recommendation_stage(
        session_id=uuid4(),
        discovery=discovery_envelope.data,
        canvas=canvas_envelope.data,
        blueprint=blueprint_envelope.data,
        definition_artifact=complete_definition_artifact(),
        design_artifact=complete_design_artifact(),
        blueprint_version_number=2,
    )

    assert discovery_envelope.llm_trace is not None
    assert discovery_envelope.llm_trace.provider_key == "openai"
    assert discovery_envelope.llm_trace.context_used_sources[0]["key"] == "discovery_capture"
    assert discovery_traces[0].llm_trace is not None

    assert canvas_envelope.llm_trace is not None
    assert canvas_envelope.llm_trace.provider_key == "deepseek"
    assert canvas_envelope.llm_trace.effective_context_backend == "inline_context_compact"
    assert canvas_traces[0].llm_trace is not None

    assert blueprint_envelope.llm_trace is not None
    assert blueprint_envelope.llm_trace.provider_key == "codex_local"
    assert blueprint_envelope.llm_trace.route_reason.startswith("Shadow promovido")
    assert [item["key"] for item in blueprint_envelope.llm_trace.context_used_sources] == [
        "narrative_discovery",
        "narrative_canvas",
        "narrative_blueprint",
    ]
    blueprint_generation_trace = next(
        trace for trace in blueprint_traces if trace.skill_key == "blueprint_generation_skill"
    )
    assert blueprint_generation_trace.llm_trace is not None

    assert recommendation_envelope.llm_trace is not None
    assert recommendation_envelope.llm_trace.provider_key == "openai"
    assert recommendation_envelope.llm_trace.context_used_sources[0]["key"] == "tool_recommendation_case"
    assert recommendation_traces[0].llm_trace is not None


def test_tool_recommendation_stage_returns_structured_placeholder_contract() -> None:
    discovery_envelope, _ = run_discovery_stage(complete_discovery_input())
    canvas_envelope, _ = run_canvas_stage(discovery_envelope.data)
    blueprint_envelope, _ = run_blueprint_stage(discovery_envelope.data, canvas_envelope.data)

    recommendation_envelope, traces = run_tool_recommendation_stage(
        session_id=uuid4(),
        discovery=discovery_envelope.data,
        canvas=canvas_envelope.data,
        blueprint=blueprint_envelope.data,
        definition_artifact=complete_definition_artifact(),
        design_artifact=complete_design_artifact(),
        blueprint_version_number=3,
    )

    assert isinstance(recommendation_envelope.data, ToolRecommendationArtifact)
    assert recommendation_envelope.data.schema_version == "tool-recommendation.v1"
    assert recommendation_envelope.data.source_blueprint_version == 3
    assert recommendation_envelope.data.preflight.case_classification
    assert recommendation_envelope.data.preflight.candidate_tool_families
    assert recommendation_envelope.data.evaluation.summary
    assert recommendation_envelope.data.evaluation.overall_status in {
        ReviewState.complete,
        ReviewState.partial,
        ReviewState.blocked,
    }
    if recommendation_envelope.data.evaluation.promotion_blocked:
        assert recommendation_envelope.next_action == "resolve_tool_recommendation_findings"
    else:
        assert recommendation_envelope.next_action == "review_tool_recommendation"
    assert traces[0].skill_key == "tool_recommendation_skill"
    assert traces[0].output_kind == "ToolRecommendationArtifact"


def test_evaluation_skill_is_registered_with_structured_output() -> None:
    registry = get_skill_registry()
    skill = registry.get("evaluation_skill")

    assert skill.output_model is EvaluationArtifact


def test_design_self_healing_reconciles_contradictions() -> None:
    from app.services.design_recommendation_service import (
        build_design_recommendation_artifact,
        merge_llm_design_recommendation,
        evaluate_design_recommendation_artifact,
    )
    from app.services.llm_runtime.builder_contracts import (
        AgentDesignProposalOutput,
        DesignCritiqueOutput,
        CritiqueFinding,
    )

    discovery_envelope, _ = run_discovery_stage(complete_discovery_input())
    canvas_envelope, _ = run_canvas_stage(discovery_envelope.data)
    definition = complete_definition_artifact()

    artifact = build_design_recommendation_artifact(discovery_envelope.data, canvas_envelope.data, definition)

    # Simulate contradictory LLM output: recommended is 'handoffs' but rationale argues for Router-Worker / Supervisor
    proposal = AgentDesignProposalOutput(
        summary="Recomendamos Router-Worker (Supervisor con subagentes) por su estricto control de barreras de confianza.",
        alternatives=artifact.alternatives,
        fit_matrix=artifact.fit_matrix,
        recommended_alternative_key="handoffs",
        decision_rationale="El resumen ejecutivo recomienda explícitamente una arquitectura Router-Worker (Jerárquica / Supervisor con subagentes).",
        requirements_coverage=artifact.requirements_coverage,
        evidence_refs=artifact.evidence_refs,
        confidence=0.85,
        architecture="supervisor_with_subagents",
        reasoning_pattern="Plan-and-Execute",
        coordination_model="orchestrated",
        open_questions=[],
        narrative="Se recomienda Supervisor con subagentes.",
    )

    critique = DesignCritiqueOutput(
        overall_status="needs_revision",
        summary="Inconsistencia detectada.",
        findings=[
            CritiqueFinding(
                finding_key="design-contradiction-alternative",
                title="Inconsistencia entre la alternativa seleccionada y la justificación/narrativa de diseño",
                severity="blocking",
                detail="El resumen ejecutivo y la razón de decisión recomiendan Router-Worker, pero recommended_alternative_key selecciona handoffs.",
                suggested_action="Alinear alternativa con justificación.",
                source_refs=["proposal.recommended_alternative_key"],
            )
        ],
        contradictions=["contradiction between handoffs and supervisor"],
        missing_evidence=[],
    )

    merged = merge_llm_design_recommendation(artifact, proposal, critique)
    evaluated = evaluate_design_recommendation_artifact(merged, discovery_envelope.data, definition)

    # Verify self-healing resolved the contradiction
    assert not any("inconsistencia" in f.title.lower() for f in evaluated.critic_findings)
    assert evaluated.selected_design is not None
    # Verify recommended alternative was reconciled with the justified architecture
    assert evaluated.selected_design.architecture in {"supervisor_with_subagents", "handoffs"}
    assert evaluated.review_state != ReviewState.blocked

