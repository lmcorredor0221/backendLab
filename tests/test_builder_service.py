from app.models import (
    BlueprintArtifact,
    BlueprintPatchRequest,
    BlueprintTool,
    CanvasArtifact,
    DiscoveryArtifact,
    MemoryProfile,
    ReviewState,
    SafetyCheck,
)
from app.services.builder_service import build_blueprint, enrich_blueprint, evaluate_blueprint, patch_blueprint


def sample_discovery() -> DiscoveryArtifact:
    return DiscoveryArtifact(
        problem_statement="Automatizar la definicion de agentes",
        current_user="Arquitecto de soluciones",
        current_process="Recoge inputs en documentos y reuniones",
        desired_outcome="Generar un blueprint tecnico consistente",
        autonomy_level="high",
        constraints=["Sin microservicios en MVP"],
        operational_baseline={
            "current_time_spent": "6 horas por iniciativa",
            "current_cost": "Alto retrabajo del equipo tecnico",
            "frequent_errors": ["Faltan decisiones de alcance", "Se omiten riesgos de aprobacion"],
            "automation_opportunities": ["Normalizar discovery", "Generar artefactos tecnicos base"],
        },
        mvp_definition={
            "v1_scope": ["Discovery", "Canvas", "Blueprint"],
            "out_of_scope": ["Temporal", "Subagentes"],
            "north_star_metric": "Blueprint util en una sola sesion",
            "non_delegable_decisions": ["Aprobar promocion a implementacion"],
        },
        case_type="automatizacion",
        value_statement="Reducir ambiguedad y retrabajo",
    )


def sample_canvas() -> CanvasArtifact:
    return CanvasArtifact(
        user_goal="Generar un blueprint tecnico consistente",
        mvp_scope=["Discovery", "Canvas", "Blueprint"],
        out_of_scope=["Temporal", "Subagentes"],
        success_metric="Blueprint util en una sola sesion",
        primary_risk="Sobrealcance funcional",
        agent_profile={
            "mission": "Transformar un problema en un agente implementable.",
            "primary_user": "Arquitecto de soluciones",
            "agent_task": "Definir un builder de agentes",
            "allowed_decisions": ["Recomendar arquitectura"],
            "prohibited_decisions": ["Ejecutar side effects sin aprobacion"],
            "key_inputs": ["Problema", "Proceso actual"],
            "expected_outputs": ["Blueprint", "Backlog"],
            "human_approvals": ["Aprobar promocion a implementacion"],
            "success_metrics": ["Blueprint util en una sola sesion"],
        },
    )


def sample_blueprint() -> BlueprintArtifact:
    return BlueprintArtifact(
        architecture="single_agent_with_skills",
        reasoning_pattern="Plan-and-Execute",
        memory_strategy="session_memory_with_checkpoints",
        tools=[
            BlueprintTool(
                name="build_blueprint",
                purpose="Genera el blueprint base",
                risk_level="medium",
                requires_approval=True,
                inputs=["normalized_discovery", "canvas"],
                outputs=["blueprint"],
                validations=["Arquitectura explicita"],
                has_side_effects=True,
                execution_mode="collect_intent_then_workflow_execute",
                retry_strategy="Reenfile manual",
                compensation_strategy="Revertir promocion",
                approval_reason="La promocion del blueprint requiere validacion humana",
                failure_mode="Inconsistencias en reglas",
            )
        ],
        memory_profile=MemoryProfile(
            strategy="session_memory_with_checkpoints",
            storage_layers=["session_state"],
            write_policy="Persistir estado validado",
            retrieval_policy="Recuperar por session_id",
            review_trigger="Campos faltantes",
            goal_drift_guard="Comparar con desired_outcome",
        ),
        safety_checks=[
            SafetyCheck(
                category="hallucination_control",
                risk="Campos inventados",
                severity="high",
                mitigation="Unknown y validacion posterior",
                status="required",
            )
        ],
        guardrails=["No inventar datos"],
        readiness_state=ReviewState.partial,
        narrative="Blueprint base para un builder de agentes.",
    )


def test_enrich_blueprint_marks_complete_when_sections_exist() -> None:
    envelope = enrich_blueprint(sample_blueprint(), sample_discovery(), sample_canvas())
    assert envelope.data.memory_profile.storage_layers
    assert envelope.data.safety_checks
    assert envelope.data.delivery_package.deliverables
    assert envelope.data.readiness_state == ReviewState.partial


def test_build_blueprint_exposes_rule_first_decision_report() -> None:
    envelope = build_blueprint(sample_discovery(), sample_canvas())
    assert envelope.data.delivery_package.decision_summary
    assert len(envelope.data.delivery_package.decision_trace) == 3
    assert any(item.dimension == "architecture" for item in envelope.data.delivery_package.decision_trace)
    assert any(item.family == "architecture" for item in envelope.data.delivery_package.pattern_catalog)
    assert any(item.key == "ToT" for item in envelope.data.delivery_package.pattern_catalog)
    assert any(item.key == "decision_trace" for item in envelope.data.delivery_package.deliverables)
    assert any(item.key == "evolution_roadmap" for item in envelope.data.delivery_package.deliverables)
    assert envelope.data.delivery_package.roadmap_evolution.milestones
    assert envelope.data.delivery_package.blueprint_coverage.total_sections >= 14
    assert envelope.data.delivery_package.blueprint_coverage.sections


def test_patch_blueprint_revalidates_nested_tools_before_rebuilding_delivery_package() -> None:
    patched_tool = BlueprintTool(
        name="approve_release",
        purpose="Solicita aprobacion antes de promover el blueprint.",
        risk_level="high",
        requires_approval=True,
        inputs=["blueprint"],
        outputs=["approval_gate"],
        validations=["approval_reason requerido"],
        has_side_effects=True,
        execution_mode="approval_gate",
        retry_strategy="Reintento manual",
        compensation_strategy="Mantener en needs_review",
        approval_reason="Evita promocion prematura.",
        failure_mode="Aprobacion no resuelta",
    )

    envelope = patch_blueprint(
        sample_blueprint(),
        BlueprintPatchRequest(tools=[patched_tool]),
        sample_discovery(),
        sample_canvas(),
    )

    assert envelope.data.tools[0].requires_approval is True
    assert envelope.data.delivery_package.workflow_profile.steps[-1].requires_approval is True


def test_evaluate_blueprint_reports_complete_for_consistent_artifacts() -> None:
    enriched = enrich_blueprint(sample_blueprint(), sample_discovery(), sample_canvas()).data
    envelope = evaluate_blueprint(sample_discovery(), sample_canvas(), enriched)
    assert envelope.data.completeness_status == ReviewState.complete
    assert envelope.data.coherence_status == ReviewState.complete
    assert len(envelope.data.cases) >= 5
