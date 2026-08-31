from uuid import uuid4

from app.models import (
    BlueprintArtifact,
    CanvasArtifact,
    DesignAlternative,
    DesignBlueprintProjection,
    DesignRecommendationArtifact,
    DiscoveryArtifact,
    MemoryDryCompileStatus,
    MemoryProfile,
    MemoryRecommendationArtifact,
    MemoryRecommendationConfidence,
    MemoryRecommendationFinding,
    ReviewState,
)
from app.services.llm_runtime.builder_contracts import MemoryArchitectureRecommendationOutput
from app.services.memory_recommendation_service import auto_reconcile_memory_artifact, build_memory_recommendation_artifact


def _discovery() -> DiscoveryArtifact:
    return DiscoveryArtifact(
        problem_statement="Gestionar solicitudes operativas con trazabilidad.",
        current_user="Operations Lead",
        current_process="Hoy se revisa manualmente y se escala por correo.",
        desired_outcome="Responder con estado claro y escalar solo cuando haga falta.",
        constraints=["Mantener control humano ante incertidumbre"],
        mvp_definition={
            "v1_scope": ["Responder solicitud", "Escalar excepciones"],
            "out_of_scope": ["Automatizacion total"],
            "north_star_metric": "Tiempo de resolucion",
            "non_delegable_decisions": [],
        },
    )


def _canvas() -> CanvasArtifact:
    return CanvasArtifact(
        user_goal="Resolver solicitudes con continuidad entre pasos.",
        success_metric="Menos retrabajo por caso",
        primary_risk="Perder contexto en escalamiento",
        agent_profile={
            "mission": "Coordinar el caso",
            "primary_user": "Operations Lead",
            "agent_task": "Resolver y escalar solicitudes",
            "expected_outputs": ["Respuesta trazable"],
            "human_approvals": [],
        },
    )


def _blueprint() -> BlueprintArtifact:
    return BlueprintArtifact(
        architecture="single_agent_with_skills",
        reasoning_pattern="ReAct",
        memory_strategy="session_memory",
        guardrails=["No ejecutar side effects sin autorizacion"],
    )


def test_memory_preserves_design_implications_without_turning_soft_handoff_into_blocker() -> None:
    design = DesignRecommendationArtifact(
        alternatives=[
            DesignAlternative(
                alternative_key="handoff_flow",
                label="Handoff gobernado",
                architecture="handoffs",
                reasoning_pattern="Plan-and-Execute",
                tool_implications=[
                    "human_handoff: escalar a un owner cuando la evidencia sea insuficiente.",
                ],
                memory_implications=[
                    "handoff_resume_context: retomar desde el owner y payload correcto.",
                ],
                blueprint_projection=DesignBlueprintProjection(
                    architecture="handoffs",
                    reasoning_pattern="Plan-and-Execute",
                    memory_implications=[
                        "checkpoint_resume: retomar desde el ultimo paso estable.",
                    ],
                ),
            )
        ],
        recommended_alternative_key="handoff_flow",
    )

    artifact = build_memory_recommendation_artifact(
        discovery=_discovery(),
        canvas=_canvas(),
        blueprint=_blueprint(),
        approved_tools_digest=None,
        source_session_id=uuid4(),
        source_blueprint_version=1,
        current_blueprint_version=1,
        design_artifact=design,
    )
    dependencies = {item.tool_key: item for item in artifact.tool_dependencies}

    assert dependencies["human_handoff"].status == "design_signal"
    assert dependencies["human_handoff"].required is False
    assert any(item.task_kind == "design_memory_implications" for item in artifact.context_budget_plan)
    assert all(item.finding_key != "missing-tool:human_handoff" for item in artifact.critic_findings)


def test_memory_dependency_request_creates_governed_gap_without_inventing_tool_key() -> None:
    artifact = build_memory_recommendation_artifact(
        discovery=_discovery(),
        canvas=_canvas(),
        blueprint=_blueprint(),
        approved_tools_digest=None,
        source_session_id=uuid4(),
        source_blueprint_version=2,
        current_blueprint_version=2,
        proposal=MemoryArchitectureRecommendationOutput(
            memory_strategy="session_memory",
            short_term_strategy="Mantener resumen activo por caso.",
            long_term_strategy="Persistir decisiones aprobadas.",
            retrieval_strategy="Recuperar contexto bajo demanda.",
            tool_dependency_requests=["notification_sender"],
            rationale="Memory necesita avisar cuando una decision quede pendiente.",
        ),
    )
    dependencies = {item.tool_key: item for item in artifact.tool_dependencies}
    gaps = {item.capability_key: item for item in artifact.dependency_gaps}

    assert "notification_sender" not in dependencies
    assert dependencies["outbound_notification"].required is True
    assert dependencies["outbound_notification"].status == "missing"
    assert gaps["outbound_notification"].remediation_policy == "human_review"
    assert gaps["outbound_notification"].candidate_pattern_id == "candidate_tool_pattern:outbound_notification"
    assert artifact.review_state == "blocked"


def test_memory_reconciliation_removes_stale_empty_strategy_blocker_after_strategy_is_present() -> None:
    artifact = MemoryRecommendationArtifact(
        summary="La arquitectura de memoria propuesta esta incompleta: memory_strategy vacio y readiness_state bloqueado.",
        proposed_memory_profile=MemoryProfile(
            strategy="workflow_memory_with_handoffs",
            storage_layers=["session_state", "short_term_checkpoints"],
            write_policy="Persistir checkpoints validados, decisiones aprobadas y owners de handoff.",
            retrieval_policy="Recuperar por session_id, etapa y ultimo checkpoint consistente.",
        ),
        dry_compile_status=MemoryDryCompileStatus(
            status="ready",
            summary="Stage4 consumio correctamente los contratos de memoria.",
            generated_contracts=["memory-policy.v1", "short-term-memory.v1", "knowledge-contract.v1"],
        ),
        confidence=MemoryRecommendationConfidence(overall=0.48, band="low", rationale="Critica inicial obsoleta."),
        missing_information=[
            "memory_strategy vacio en approved_blueprint",
            "Definir retencion operativa por capa",
        ],
        critic_findings=[
            MemoryRecommendationFinding(
                finding_key="memory_strategy_empty",
                title="memory_strategy vacio",
                detail="El approved_blueprint conserva memory_strategy vacio y readiness_state bloqueado.",
                severity="blocking",
                category="critic",
                suggested_action="Generar estrategia de memoria.",
            )
        ],
        review_state=ReviewState.blocked,
    )

    reconciled = auto_reconcile_memory_artifact(artifact)

    assert reconciled.review_state != ReviewState.blocked
    assert all("memory_strategy" not in item.lower() for item in reconciled.missing_information)
    assert all(item.finding_key != "memory_strategy_empty" for item in reconciled.critic_findings)
    assert "memory_strategy vacio" not in reconciled.summary.lower()
    assert "readiness_state bloqueado" not in reconciled.summary.lower()
    assert reconciled.confidence.overall > artifact.confidence.overall
