from app.models import (
    DesignAlternative,
    DesignBlueprintProjection,
    DesignCritiqueFinding,
    DesignRecommendationArtifact,
    DiscoveryArtifact,
    ReviewState,
)
from app.services.design_recommendation_service import evaluate_design_recommendation_artifact
from app.services.llm_runtime.builder_contracts import FunctionalRequirement, RequirementsDefinitionOutput


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
