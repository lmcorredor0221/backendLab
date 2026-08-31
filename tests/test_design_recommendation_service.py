from app.models import (
    CanvasArtifact,
    DesignAlternative,
    DesignBlueprintProjection,
    DesignCritiqueFinding,
    DesignRecommendationArtifact,
    DiscoveryArtifact,
    ReviewState,
)
from app.services.design_recommendation_service import (
    build_design_intelligence_shadow_report,
    build_design_recommendation_artifact,
    downgrade_design_recommendation_to_legacy,
    evaluate_design_recommendation_artifact,
)
from app.services.llm_runtime.builder_contracts import FunctionalRequirement, PrioritizedQuestion, RequirementsDefinitionOutput
from app.services.rules import build_agent_archetype_catalog, build_pattern_family_catalog


def _discovery_artifact() -> DiscoveryArtifact:
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


def _canvas_artifact() -> CanvasArtifact:
    return CanvasArtifact(
        user_goal="Construir un blueprint implementable usando documentos, herramientas y memoria trazable.",
        mvp_scope=["Generar blueprint", "Validar decisiones", "Preparar implementacion"],
        out_of_scope=["Provisioning automatico"],
        success_metric="Blueprint util en una sola sesion",
        primary_risk="Perder contexto entre etapas",
        agent_profile={
            "mission": "Preparar blueprint Lean",
            "primary_user": "Arquitecto de soluciones",
            "agent_task": "Convertir discovery en arquitectura y entregables accionables.",
            "allowed_decisions": ["Proponer patron agentivo"],
            "prohibited_decisions": ["Aprobar implementacion sin revision"],
            "key_inputs": ["Discovery", "Definition"],
            "expected_outputs": ["Arquitectura recomendada", "Implicaciones de tools", "Implicaciones de memoria"],
            "human_approvals": ["Promocion a implementacion"],
            "success_metrics": ["Menos retrabajo tecnico", "Decisiones trazables"],
        },
    )


def _definition_artifact() -> RequirementsDefinitionOutput:
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


def _design_artifact() -> DesignRecommendationArtifact:
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
        fit_score=82,
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


def test_evaluate_design_recommendation_recomputes_findings_counters() -> None:
    artifact = _design_artifact().model_copy(
        update={
            "critic_findings": [
                DesignCritiqueFinding(
                    finding_key="design-high-priority-gap",
                    title="Persisten gaps sobre requisitos prioritarios",
                    severity="warning",
                    detail="La alternativa todavia requiere revision adicional.",
                    suggested_action="Revisar antes de promover la etapa.",
                    source_refs=["design.alternatives"],
                )
            ]
        }
    )

    evaluated = evaluate_design_recommendation_artifact(
        artifact,
        _discovery_artifact(),
        _definition_artifact(),
    )

    assert evaluated.review_state == ReviewState.partial
    assert evaluated.confidence.band == "medium"
    assert evaluated.missing_information == []


def test_build_design_recommendation_adds_business_pattern_and_dependency_intelligence() -> None:
    artifact = build_design_recommendation_artifact(
        _discovery_artifact(),
        _canvas_artifact(),
        _definition_artifact(),
    )

    selected = artifact.selected_design

    assert selected is not None
    assert selected.agent_archetype
    assert selected.pattern_family
    assert selected.business_fit
    assert selected.value_hypothesis
    assert selected.why_recommended
    assert selected.why_not_simpler
    assert selected.why_not_more_complex
    assert selected.tool_implications
    assert selected.memory_implications
    assert selected.business_metrics
    assert selected.blueprint_projection.memory_strategy
    assert selected.blueprint_projection.tool_implications
    assert selected.blueprint_projection.memory_implications
    assert any("Scoring negocio-tecnico" in item for item in selected.fit_rationale)
    assert any("catalog.agent_archetype" in ref for ref in selected.evidence_refs)
    assert any("catalog.pattern_family" in ref for ref in selected.evidence_refs)
    assert any("knowledge.agentic_knowledge_base" in ref for ref in selected.evidence_refs)


def test_design_intelligence_can_downgrade_to_legacy_projection() -> None:
    artifact = build_design_recommendation_artifact(
        _discovery_artifact(),
        _canvas_artifact(),
        _definition_artifact(),
    )

    downgraded = downgrade_design_recommendation_to_legacy(artifact)

    assert downgraded.selected_design is not None
    assert downgraded.selected_design.architecture == artifact.selected_design.architecture
    assert downgraded.selected_design.agent_archetype == ""
    assert downgraded.selected_design.pattern_family == ""
    assert downgraded.selected_design.tool_implications == []
    assert downgraded.selected_design.memory_implications == []
    assert downgraded.selected_design.blueprint_projection.tool_implications == []
    assert downgraded.selected_design.blueprint_projection.memory_implications == []
    assert "Design Intelligence v2 desactivado" in downgraded.summary


def test_design_intelligence_shadow_report_tracks_v2_fields_hidden_by_legacy() -> None:
    artifact = build_design_recommendation_artifact(
        _discovery_artifact(),
        _canvas_artifact(),
        _definition_artifact(),
    )

    report = build_design_intelligence_shadow_report(artifact)

    assert report["contract_version"] == "design-intelligence-shadow.v1"
    assert report["mode"] == "v2_active_legacy_projection_shadow"
    assert report["changed_alternative_count"] == len(artifact.alternatives)
    assert report["selected_agent_archetype"] == artifact.selected_design.agent_archetype
    assert report["selected_pattern_family"] == artifact.selected_design.pattern_family
    assert report["field_presence"]["agent_archetype"] == len(artifact.alternatives)
    assert report["field_presence"]["tool_implications"] == len(artifact.alternatives)
    assert report["alternatives"][0]["v2_fields_hidden_by_legacy"]
    assert "blueprint_projection.tool_implications" in report["alternatives"][0]["projection_fields_hidden_by_legacy"]


def test_design_catalogs_include_governed_archetypes_and_pattern_families() -> None:
    discovery = _discovery_artifact()
    canvas = _canvas_artifact()

    archetypes = sorted(
        build_agent_archetype_catalog(discovery, canvas),
        key=lambda item: item.fit_score,
        reverse=True,
    )
    patterns = sorted(
        build_pattern_family_catalog(discovery, canvas),
        key=lambda item: item.fit_score,
        reverse=True,
    )

    assert len(archetypes) >= 10
    assert len(patterns) >= 10
    assert archetypes[0].key in {
        "workflow_operator",
        "human_approval_agent",
        "rag_knowledge_assistant",
        "knowledge_steward",
    }
    assert patterns[0].key in {
        "react_loop",
        "plan_execute",
        "human_in_the_loop",
        "checkpoint_resume",
        "rag_grounded",
    }


def test_evaluate_design_flags_generic_business_explanation() -> None:
    artifact = _design_artifact()
    assert artifact.selected_design is not None
    weak_projection = DesignBlueprintProjection(
        architecture="single_agent_with_skills",
        reasoning_pattern="ReAct",
        narrative="Arquitectura propuesta.",
    )
    weak_alternative = artifact.selected_design.model_copy(
        update={
            "business_fit": "Opcion adecuada.",
            "why_recommended": "Cubre el alcance.",
            "tool_implications": [],
            "memory_implications": [],
            "blueprint_projection": weak_projection,
        }
    )
    weak_artifact = artifact.model_copy(
        update={
            "alternatives": [weak_alternative],
            "selected_design": weak_alternative,
            "recommended_alternative_key": weak_alternative.alternative_key,
        }
    )

    evaluated = evaluate_design_recommendation_artifact(
        weak_artifact,
        _discovery_artifact(),
        _definition_artifact(),
    )

    assert evaluated.review_state == ReviewState.partial
    assert any(
        finding.finding_key == "design-weak-business-explanation"
        for finding in evaluated.critic_findings
    )


def test_evaluate_design_repairs_react_findings_before_blocking() -> None:
    strong_handoffs = DesignAlternative(
        alternative_key="handoffs",
        label="Handoffs gobernados",
        recommendation_role="recommended",
        agent_archetype="Cadena planner-executor-reviewer",
        pattern_family="Plan-and-Execute + handoffs gobernados",
        architecture="handoffs",
        reasoning_pattern="Plan-and-Execute",
        coordination_model="handoffs",
        summary=(
            "Cadena planner-executor-reviewer para preservar ownership, evidencia y revision "
            "antes de promover entregables."
        ),
        business_fit=(
            "Conecta con la necesidad de generar un blueprint implementable sin perder contexto, "
            "manteniendo aprobaciones y trazabilidad para reducir retrabajo tecnico."
        ),
        why_recommended=(
            "Se recomienda porque el proceso requiere preparar entregables, validarlos y dejar "
            "handoffs claros antes de avanzar a implementacion."
        ),
        topology="Planner, Executor y Reviewer colaboran de forma secuencial.",
        handoffs=[
            {
                "from_role": "Planner",
                "to_role": "Executor",
                "trigger": "Plan listo para ejecucion.",
                "payload": "",
                "approval_required": False,
            }
        ],
        approval_points=[],
        decision_policy="Priorizar evidencia aprobada y preservar el ultimo estado estable.",
        escalation_conditions=[],
        failure_modes=[],
        operational_complexity="medium",
        relative_cost="medium",
        maintainability="medium",
        fit_score=97,
        tool_implications=["approval_gate: gobernar decisiones no delegables."],
        memory_implications=["checkpoint_resume: retomar desde el ultimo paso estable."],
        business_metrics=[
            "Blueprint util en una sola sesion",
            "Porcentaje de decisiones con evidencia trazable",
        ],
        blueprint_projection=DesignBlueprintProjection(
            architecture="handoffs",
            reasoning_pattern="Plan-and-Execute",
            narrative=(
                "Handoffs gobernados mantienen continuidad entre planning, ejecucion y revision."
            ),
            tool_implications=["approval_gate: gobernar decisiones no delegables."],
            memory_strategy="workflow_memory_with_handoffs",
            memory_implications=["checkpoint_resume: retomar desde el ultimo paso estable."],
            cost_complexity_implications=["Costo relativo: medium"],
        ),
    )
    weak_alternative = _design_artifact().selected_design
    assert weak_alternative is not None
    weak_alternative = weak_alternative.model_copy(update={"fit_score": 55})
    artifact = DesignRecommendationArtifact(
        alternatives=[strong_handoffs, weak_alternative],
        recommended_alternative_key="handoffs",
        selected_design=strong_handoffs,
        critic_findings=[
            DesignCritiqueFinding(
                finding_key="handoff_ambiguity",
                title="Handoffs ambiguos",
                severity="blocking",
                detail="Los handoffs no declaran payload, criterio de retorno ni recuperacion.",
                suggested_action="Definir contrato de handoff antes de promover.",
                source_refs=["design.selected_design.handoffs"],
            ),
            DesignCritiqueFinding(
                finding_key="loop_risk",
                title="Riesgo de loop",
                severity="warning",
                detail="No hay limites de terminacion ni presupuesto de reintentos.",
                suggested_action="Agregar politica de terminacion.",
                source_refs=["design.failure_modes"],
            ),
            DesignCritiqueFinding(
                finding_key="missing_evaluation_plan",
                title="Plan de evaluacion faltante",
                severity="warning",
                detail="Falta indicar como Validate comprobara cobertura y riesgos.",
                suggested_action="Diferir a Validate con criterios claros.",
                source_refs=["design.validation"],
            ),
            DesignCritiqueFinding(
                finding_key="premature_permission_assumption",
                title="Permisos asumidos prematuramente",
                severity="warning",
                detail="Design no debe cerrar credenciales o permisos exactos de tools.",
                suggested_action="Diferir detalle de permisos a Tools.",
                source_refs=["design.tool_implications"],
            ),
        ],
        missing_information=[
            "Detalle de arquitectura de componentes e interacciones.",
            "Plan de evaluación y métricas de rendimiento.",
            "Criterios de terminación y prevención de loops.",
        ],
        summary="Design requiere reparacion conceptual.",
    )

    evaluated = evaluate_design_recommendation_artifact(
        artifact,
        _discovery_artifact(),
        _definition_artifact(),
    )

    assert evaluated.selected_design is not None
    assert evaluated.review_state == ReviewState.complete
    assert evaluated.confidence.band == "high"
    assert evaluated.confidence.overall >= 0.9
    assert not evaluated.critic_findings
    assert evaluated.missing_information == []
    assert evaluated.selected_design.handoffs[0].payload
    assert evaluated.selected_design.failure_modes
    assert any("reintento" in item.lower() for item in evaluated.selected_design.blueprint_projection.guardrails)
    assert "auto-reconcili" in evaluated.remediation_summary.lower()


def test_evaluate_design_normalizes_deepseek_fit_score_and_defers_approved_definition_debt() -> None:
    base = _design_artifact()
    assert base.selected_design is not None
    selected = base.selected_design.model_copy(update={"fit_score": 0.95})
    artifact = base.model_copy(
        update={
            "alternatives": [selected],
            "selected_design": selected,
            "recommended_alternative_key": selected.alternative_key,
            "critic_findings": [
                DesignCritiqueFinding(
                    finding_key="missing_scope_definition",
                    title="Alcance inicial no definido bloquea decisiones de diseno",
                    severity="blocking",
                    detail=(
                        "La pregunta bloqueante aprobada con deuda no ha sido resuelta y debe "
                        "seguir visible para implementacion."
                    ),
                    suggested_action="Mantener la deuda visible sin bloquear de nuevo Design.",
                    source_refs=["approved_definition", "agent_design_critique_input"],
                )
            ],
        }
    )
    definition = _definition_artifact().model_copy(
        update={
            "open_questions": [
                PrioritizedQuestion(
                    key="OQ-001",
                    question="Cual es el alcance especifico del piloto?",
                    priority="high",
                    blocking_stages=["design"],
                    suggested_answer="Iniciar solo con documentacion aprobada.",
                )
            ]
        }
    )

    evaluated = evaluate_design_recommendation_artifact(
        artifact,
        _discovery_artifact(),
        definition,
    )

    assert evaluated.selected_design is not None
    assert evaluated.selected_design.fit_score == 95
    assert evaluated.review_state == ReviewState.partial
    assert evaluated.confidence.overall >= 0.85
    assert evaluated.missing_information == ["Cual es el alcance especifico del piloto?"]
    assert evaluated.critic_findings[0].severity == "warning"
