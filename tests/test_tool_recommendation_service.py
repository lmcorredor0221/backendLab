from uuid import uuid4

from app.models import BlueprintArtifact, CanvasArtifact, DiscoveryArtifact, ReviewState
from app.services.llm_runtime.builder_contracts import NonFunctionalRequirement, RequirementsDefinitionOutput
from app.services.knowledge_tool_policy import build_memory_tool_dependencies
from app.services.rules import derive_memory_profile
from app.services.tool_recommendation_service import (
    annotate_tool_recommendation_status,
    build_approved_tools_digest_from_blueprint_tools,
    build_placeholder_tool_recommendation,
    evaluate_tool_recommendation_artifact,
    promote_tool_recommendation_to_blueprint_tools,
)


def build_discovery(
    *,
    problem_statement: str,
    current_process: str,
    desired_outcome: str,
    autonomy_level: str = "medium",
    constraints: list[str] | None = None,
    non_delegable_decisions: list[str] | None = None,
) -> DiscoveryArtifact:
    return DiscoveryArtifact(
        problem_statement=problem_statement,
        current_user="Operations Lead",
        current_process=current_process,
        desired_outcome=desired_outcome,
        autonomy_level=autonomy_level,
        constraints=constraints or ["Mantener MVP simple"],
        operational_baseline={
            "current_time_spent": "3 horas",
            "current_cost": "Retrabajo operativo",
            "frequent_errors": ["Falta de trazabilidad"],
            "automation_opportunities": ["Reducir pasos manuales"],
        },
        mvp_definition={
            "v1_scope": ["Resolver el workflow principal"],
            "out_of_scope": ["Automatizacion total del backoffice"],
            "north_star_metric": "Workflow consistente",
            "non_delegable_decisions": non_delegable_decisions or [],
        },
        case_type="automatizacion",
        value_statement="Reducir friccion operativa",
    )


def build_canvas(
    *,
    user_goal: str,
    expected_outputs: list[str] | None = None,
    human_approvals: list[str] | None = None,
) -> CanvasArtifact:
    return CanvasArtifact(
        user_goal=user_goal,
        mvp_scope=["Resolver el workflow principal"],
        out_of_scope=["Automatizacion total del backoffice"],
        success_metric="Workflow consistente",
        primary_risk="Sobrealcance",
        agent_profile={
            "mission": user_goal,
            "primary_user": "Operations Lead",
            "agent_task": user_goal,
            "allowed_decisions": ["Proponer pasos siguientes"],
            "prohibited_decisions": ["Tomar decisiones irreversibles sin aprobacion"],
            "key_inputs": ["Contexto de negocio"],
            "expected_outputs": expected_outputs or ["Resumen accionable"],
            "human_approvals": human_approvals or [],
            "success_metrics": ["Workflow consistente"],
        },
    )


def build_blueprint(
    *,
    architecture: str = "single_agent_with_skills",
    guardrails: list[str] | None = None,
    knowledge_mode: str = "none",
    knowledge_sources: list[dict[str, str]] | None = None,
    knowledge_refresh_frequency: str = "",
    workflow_steps: list[dict[str, object]] | None = None,
) -> BlueprintArtifact:
    return BlueprintArtifact(
        architecture=architecture,
        reasoning_pattern="ReAct",
        memory_strategy="session_memory",
        guardrails=guardrails or ["Mantener trazabilidad"],
        knowledge_profile={
            "mode": knowledge_mode,
            "sources": knowledge_sources or [],
            "refresh_policy": {
                "frequency": knowledge_refresh_frequency,
            },
        },
        delivery_package={
            "workflow_profile": {
                "steps": workflow_steps
                or [
                    {
                        "name": "Analizar caso",
                        "objective": "Entender el flujo principal",
                        "actor": "agent",
                        "outputs": ["recomendacion"],
                        "fallback": "escalar",
                        "requires_approval": False,
                    }
                ]
            }
        },
    )


def test_enterprise_copilot_shortlists_lookup_without_extra_tools() -> None:
    artifact = build_placeholder_tool_recommendation(
        session_id=uuid4(),
        discovery=build_discovery(
            problem_statement="Ayudar a soporte a responder usando datos operativos reales.",
            current_process="Consultar CRM y tickets abiertos del cliente antes de responder.",
            desired_outcome="Responder estado del caso sin actualizar registros.",
            constraints=[
                "Mantener MVP simple",
                "No ejecutar side effects irreversibles sin aprobacion humana",
            ],
        ),
        canvas=build_canvas(user_goal="Consultar CRM y tickets para responder con grounding."),
        blueprint=build_blueprint(
            guardrails=["Solo lectura sobre sistemas operativos", "Mantener trazabilidad"]
        ),
        blueprint_version_number=1,
    )

    candidate_map = {item.family_key: item for item in artifact.preflight.candidate_tool_families}

    assert artifact.preflight.case_classification == "enterprise_copilot"
    assert [item.tool_key for item in artifact.recommended_tools] == ["read_system_of_record"]
    assert artifact.optional_tools == []
    assert artifact.needs_information == []
    assert candidate_map["read_only_lookup"].status == "required"
    assert candidate_map["transactional_write"].status == "excluded"


def test_knowledge_assistant_requires_retrieval_and_ingestion_for_rag_sources() -> None:
    artifact = build_placeholder_tool_recommendation(
        session_id=uuid4(),
        discovery=build_discovery(
            problem_statement="Responder preguntas sobre politicas y manuales internos.",
            current_process="Hoy el equipo busca manuales y FAQ antes de responder.",
            desired_outcome="Citar procedimientos vigentes con grounding documental.",
        ),
        canvas=build_canvas(
            user_goal="Responder con citas y referencias institucionales.",
            expected_outputs=["Respuesta con citas", "Referencia documental"],
        ),
        blueprint=build_blueprint(
            knowledge_mode="rag",
            knowledge_sources=[
                {
                    "key": "manual-rh",
                    "title": "Manual RH",
                    "description": "Politicas internas",
                    "source_type": "document",
                    "uri": "kb://manual-rh",
                    "owner": "People Ops",
                    "license": "internal",
                    "sensitivity": "internal",
                    "source_version": "2026-07",
                }
            ],
        ),
        blueprint_version_number=2,
    )

    candidate_map = {item.family_key: item for item in artifact.preflight.candidate_tool_families}

    assert artifact.preflight.case_classification == "knowledge_assistant"
    assert [item.tool_key for item in artifact.recommended_tools] == ["knowledge_retrieval", "document_ingestion"]
    assert artifact.optional_tools == []
    assert candidate_map["retrieval"].status == "required"
    assert candidate_map["document_ingestion"].status == "required"
    assert artifact.needs_information == []


def test_scheduled_rag_refresh_requires_scheduler() -> None:
    artifact = build_placeholder_tool_recommendation(
        session_id=uuid4(),
        discovery=build_discovery(
            problem_statement="Responder preguntas sobre procedimientos vigentes.",
            current_process="Busca procedimientos y refresca el corpus cada dia antes de responder.",
            desired_outcome="Responder con citas frescas y consistentes.",
        ),
        canvas=build_canvas(
            user_goal="Responder con grounding documental y corpus actualizado.",
            expected_outputs=["Respuesta con citas"],
        ),
        blueprint=build_blueprint(
            knowledge_mode="rag",
            knowledge_sources=[
                {
                    "key": "ops-playbook",
                    "title": "Ops Playbook",
                    "description": "Playbook operativo",
                    "source_type": "document",
                    "uri": "kb://ops-playbook",
                    "owner": "Ops",
                    "license": "internal",
                    "sensitivity": "internal",
                    "source_version": "2026-07",
                }
            ],
            knowledge_refresh_frequency="daily",
        ),
        blueprint_version_number=2,
    )

    candidate_map = {item.family_key: item for item in artifact.preflight.candidate_tool_families}

    assert [item.tool_key for item in artifact.recommended_tools] == [
        "knowledge_retrieval",
        "document_ingestion",
        "scheduler",
    ]
    assert candidate_map["scheduler"].status == "required"


def test_approval_gated_operator_requires_lookup_write_and_gate() -> None:
    artifact = build_placeholder_tool_recommendation(
        session_id=uuid4(),
        discovery=build_discovery(
            problem_statement="Operar solicitudes con aprobacion controlada.",
            current_process="Revisa solicitudes en portal HR y luego actualiza el estado aprobado o rechazado.",
            desired_outcome="Aprobar o actualizar la solicitud con trazabilidad.",
            autonomy_level="high",
            constraints=["No ejecutar side effects irreversibles sin aprobacion humana"],
            non_delegable_decisions=["Aprobar la solicitud final"],
        ),
        canvas=build_canvas(
            user_goal="Consultar el portal HR y ejecutar la actualizacion aprobada.",
            human_approvals=["Lider RRHH aprueba antes de escribir."],
        ),
        blueprint=build_blueprint(
            guardrails=["Toda escritura requiere aprobacion humana y audit trail"],
            workflow_steps=[
                {
                    "name": "Consultar solicitud",
                    "objective": "Leer estado actual desde portal HR",
                    "actor": "agent",
                    "outputs": ["estado actual"],
                    "fallback": "escalar",
                    "requires_approval": False,
                },
                {
                    "name": "Actualizar estado",
                    "objective": "Escribir el resultado aprobado",
                    "actor": "agent",
                    "outputs": ["estado actualizado"],
                    "fallback": "detener",
                    "requires_approval": True,
                },
            ],
        ),
        blueprint_version_number=3,
    )

    recommended_map = {item.tool_key: item for item in artifact.recommended_tools}
    candidate_map = {item.family_key: item for item in artifact.preflight.candidate_tool_families}

    assert artifact.preflight.case_classification == "approval_gated_operator"
    assert set(recommended_map) == {"read_system_of_record", "approval_gate", "transactional_write"}
    assert "approval_gate" in recommended_map["transactional_write"].dependencies
    assert "read_system_of_record" in recommended_map["transactional_write"].dependencies
    assert candidate_map["transactional_write"].status == "required"
    assert artifact.needs_information == []


def test_notification_coordinator_keeps_short_shortlist() -> None:
    artifact = build_placeholder_tool_recommendation(
        session_id=uuid4(),
        discovery=build_discovery(
            problem_statement="Alertar a owners cuando un SLA entra en riesgo.",
            current_process="Detecta SLA en riesgo y avisa por Slack al owner.",
            desired_outcome="Enviar alertas y seguimiento sin tocar sistemas core.",
            autonomy_level="low",
        ),
        canvas=build_canvas(
            user_goal="Enviar alertas por Slack a tiempo.",
            expected_outputs=["Alerta enviada"],
        ),
        blueprint=build_blueprint(),
        blueprint_version_number=4,
    )

    assert artifact.preflight.case_classification == "notification_coordinator"
    assert [item.tool_key for item in artifact.recommended_tools] == ["outbound_notification"]
    assert artifact.optional_tools == []
    assert len(artifact.preflight.candidate_tool_families) == 1
    assert artifact.preflight.candidate_tool_families[0].family_key == "notification"
    assert artifact.needs_information == []


def test_handoff_signal_does_not_force_notification_gap() -> None:
    artifact = build_placeholder_tool_recommendation(
        session_id=uuid4(),
        discovery=build_discovery(
            problem_statement="Escalar al owner correcto cuando el caso salga del playbook.",
            current_process="Analiza el caso y escala al owner humano cuando falte contexto.",
            desired_outcome="Escalar con trazabilidad sin inventar canales adicionales.",
        ),
        canvas=build_canvas(user_goal="Escalar al owner correcto cuando se requiera juicio humano."),
        blueprint=build_blueprint(),
        blueprint_version_number=41,
    )

    candidate_keys = [item.family_key for item in artifact.preflight.candidate_tool_families]

    assert "human_handoff" in candidate_keys
    assert "notification" not in candidate_keys
    assert all(item.gap_key != "notification_channel_unspecified" for item in artifact.needs_information)


def test_inline_first_case_does_not_invent_tools() -> None:
    artifact = build_placeholder_tool_recommendation(
        session_id=uuid4(),
        discovery=build_discovery(
            problem_statement="Resumir briefs pegados por el usuario.",
            current_process="El usuario pega texto libre en la conversacion y espera una sintesis.",
            desired_outcome="Devolver un resumen y proximos pasos.",
            autonomy_level="low",
        ),
        canvas=build_canvas(user_goal="Resumir texto inline sin integraciones externas."),
        blueprint=build_blueprint(),
        blueprint_version_number=5,
    )

    assert artifact.preflight.case_classification == "lean_blueprint_builder"
    assert artifact.recommended_tools == []
    assert artifact.optional_tools == []
    assert artifact.coverage_gaps == []
    assert artifact.needs_information == []
    assert artifact.preflight.candidate_tool_families == []
    assert artifact.preflight.forbidden_capabilities == ["transactional_write"]


def test_missing_operational_source_creates_gap_instead_of_transactional_tool() -> None:
    artifact = build_placeholder_tool_recommendation(
        session_id=uuid4(),
        discovery=build_discovery(
            problem_statement="Resolver solicitudes operativas con un flujo controlado.",
            current_process="Analiza la solicitud y actualiza su estado final.",
            desired_outcome="Actualizar la solicitud aprobada.",
            autonomy_level="medium",
        ),
        canvas=build_canvas(user_goal="Actualizar la solicitud con el minimo de herramientas."),
        blueprint=build_blueprint(),
        blueprint_version_number=6,
    )

    candidate_map = {item.family_key: item for item in artifact.preflight.candidate_tool_families}
    gap_keys = {item.gap_key for item in artifact.needs_information}

    assert "transactional_write" not in {item.tool_key for item in artifact.recommended_tools}
    assert "approval_gate" in {item.tool_key for item in artifact.recommended_tools}
    assert "system_of_record_unspecified" in gap_keys
    assert "approval_boundary_unspecified" in gap_keys
    assert candidate_map["transactional_write"].status == "excluded"


def test_evaluator_blocks_duplicate_tool_keys_across_sections() -> None:
    artifact = build_placeholder_tool_recommendation(
        session_id=uuid4(),
        discovery=build_discovery(
            problem_statement="Ayudar a soporte a responder usando datos operativos reales.",
            current_process="Consultar CRM antes de responder.",
            desired_outcome="Responder estado del caso sin escribir.",
        ),
        canvas=build_canvas(user_goal="Consultar CRM para responder con grounding."),
        blueprint=build_blueprint(),
        blueprint_version_number=7,
    )
    duplicated = artifact.model_copy(
        update={
            "optional_tools": [*artifact.optional_tools, artifact.recommended_tools[0]],
        }
    )

    evaluated = evaluate_tool_recommendation_artifact(duplicated)

    assert evaluated.evaluation.promotion_blocked is True
    assert evaluated.review_state == ReviewState.blocked
    assert any(item.finding_key == "duplicate-tool:read_system_of_record" for item in evaluated.evaluation.findings)


def test_evaluator_blocks_transactional_write_without_approval_gate() -> None:
    artifact = build_placeholder_tool_recommendation(
        session_id=uuid4(),
        discovery=build_discovery(
            problem_statement="Operar solicitudes con aprobacion controlada.",
            current_process="Revisa solicitudes y actualiza el estado aprobado.",
            desired_outcome="Actualizar la solicitud aprobada.",
            autonomy_level="high",
            constraints=["No ejecutar side effects irreversibles sin aprobacion humana"],
            non_delegable_decisions=["Aprobar la solicitud final"],
        ),
        canvas=build_canvas(
            user_goal="Consultar el portal HR y ejecutar la actualizacion aprobada.",
            human_approvals=["Lider RRHH aprueba antes de escribir."],
        ),
        blueprint=build_blueprint(
            guardrails=["Toda escritura requiere aprobacion humana y audit trail"],
            workflow_steps=[
                {
                    "name": "Consultar solicitud",
                    "objective": "Leer estado actual",
                    "actor": "agent",
                    "outputs": ["estado actual"],
                    "fallback": "escalar",
                    "requires_approval": False,
                },
                {
                    "name": "Actualizar estado",
                    "objective": "Escribir el resultado aprobado",
                    "actor": "agent",
                    "outputs": ["estado actualizado"],
                    "fallback": "detener",
                    "requires_approval": True,
                },
            ],
        ),
        blueprint_version_number=8,
    )
    without_gate = artifact.model_copy(
        update={
            "recommended_tools": [item for item in artifact.recommended_tools if item.tool_key != "approval_gate"],
        }
    )

    evaluated = evaluate_tool_recommendation_artifact(without_gate)

    assert evaluated.evaluation.promotion_blocked is True
    assert any(item.finding_key == "write-without-approval-gate" for item in evaluated.evaluation.findings)
    assert evaluated.evaluation.governance_status == ReviewState.blocked


def test_evaluator_drops_confidence_when_high_gaps_persist() -> None:
    artifact = build_placeholder_tool_recommendation(
        session_id=uuid4(),
        discovery=build_discovery(
            problem_statement="Resolver solicitudes operativas con un flujo controlado.",
            current_process="Analiza la solicitud y actualiza su estado final.",
            desired_outcome="Actualizar la solicitud aprobada.",
            autonomy_level="medium",
        ),
        canvas=build_canvas(user_goal="Actualizar la solicitud con el minimo de herramientas."),
        blueprint=build_blueprint(),
        blueprint_version_number=9,
    )

    evaluated = evaluate_tool_recommendation_artifact(artifact)

    assert evaluated.evaluation.promotion_blocked is True
    assert evaluated.confidence.overall < artifact.confidence.overall
    assert evaluated.confidence.band == "low"
    assert any(item.category == "coverage" for item in evaluated.evaluation.findings)


def test_operational_efficiency_nfr_is_covered_by_core_automation_tools() -> None:
    artifact = build_placeholder_tool_recommendation(
        session_id=uuid4(),
        discovery=build_discovery(
            problem_statement="Operar solicitudes con aprobacion controlada.",
            current_process="Revisa solicitudes en portal HR y luego actualiza el estado aprobado o rechazado.",
            desired_outcome="Reducir tiempo operativo manteniendo trazabilidad.",
            autonomy_level="high",
            constraints=["No ejecutar side effects irreversibles sin aprobacion humana"],
            non_delegable_decisions=["Aprobar la solicitud final"],
        ),
        canvas=build_canvas(
            user_goal="Consultar el portal HR y ejecutar la actualizacion aprobada.",
            human_approvals=["Lider RRHH aprueba antes de escribir."],
        ),
        blueprint=build_blueprint(
            guardrails=["Toda escritura requiere aprobacion humana y audit trail"],
            workflow_steps=[
                {
                    "name": "Consultar solicitud",
                    "objective": "Leer estado actual desde portal HR",
                    "actor": "agent",
                    "outputs": ["estado actual"],
                    "fallback": "escalar",
                    "requires_approval": False,
                },
                {
                    "name": "Actualizar estado",
                    "objective": "Escribir el resultado aprobado",
                    "actor": "agent",
                    "outputs": ["estado actualizado"],
                    "fallback": "detener",
                    "requires_approval": True,
                },
            ],
        ),
        definition_artifact=RequirementsDefinitionOutput(
            summary="Definition con foco en eficiencia operativa.",
            non_functional_requirements=[
                NonFunctionalRequirement(
                    key="nfr-latency",
                    title="Reducir tiempo operativo",
                    priority="high",
                    requirement="Reducir tiempo operativo del flujo principal con menos pasos manuales.",
                    category="latency",
                    metric="turnaround_time",
                    target="<10 minutos",
                    source_refs=["discovery.operational_baseline.current_time_spent"],
                )
            ],
        ),
        blueprint_version_number=42,
    )

    coverage = next(item for item in artifact.requirements_coverage if item.requirement_key == "nfr-latency")
    evaluated = evaluate_tool_recommendation_artifact(artifact)

    assert coverage.coverage_status == "covered"
    assert {"read_system_of_record", "transactional_write"}.issubset(set(coverage.covered_by_tool_keys))
    assert all(item.finding_key != "requirement-coverage:nfr-latency" for item in evaluated.evaluation.findings)


def test_evaluator_clears_happy_path_for_approval_gated_operator() -> None:
    artifact = build_placeholder_tool_recommendation(
        session_id=uuid4(),
        discovery=build_discovery(
            problem_statement="Operar solicitudes con aprobacion controlada.",
            current_process="Revisa solicitudes en portal HR y luego actualiza el estado aprobado o rechazado.",
            desired_outcome="Aprobar o actualizar la solicitud con trazabilidad.",
            autonomy_level="high",
            constraints=["No ejecutar side effects irreversibles sin aprobacion humana"],
            non_delegable_decisions=["Aprobar la solicitud final"],
        ),
        canvas=build_canvas(
            user_goal="Consultar el portal HR y ejecutar la actualizacion aprobada.",
            human_approvals=["Lider RRHH aprueba antes de escribir."],
        ),
        blueprint=build_blueprint(
            guardrails=["Toda escritura requiere aprobacion humana y audit trail"],
            workflow_steps=[
                {
                    "name": "Consultar solicitud",
                    "objective": "Leer estado actual desde portal HR",
                    "actor": "agent",
                    "outputs": ["estado actual"],
                    "fallback": "escalar",
                    "requires_approval": False,
                },
                {
                    "name": "Actualizar estado",
                    "objective": "Escribir el resultado aprobado",
                    "actor": "agent",
                    "outputs": ["estado actualizado"],
                    "fallback": "detener",
                    "requires_approval": True,
                },
            ],
        ),
        blueprint_version_number=10,
    )

    evaluated = evaluate_tool_recommendation_artifact(artifact)

    assert evaluated.evaluation.promotion_blocked is False
    assert evaluated.evaluation.findings == []
    assert evaluated.evaluation.overall_status == ReviewState.complete
    assert evaluated.review_state == ReviewState.complete


def test_promote_tool_recommendation_builds_blueprint_tools_and_digest() -> None:
    artifact = build_placeholder_tool_recommendation(
        session_id=uuid4(),
        discovery=build_discovery(
            problem_statement="Responder preguntas sobre politicas y manuales internos.",
            current_process="Hoy el equipo busca manuales y FAQ antes de responder.",
            desired_outcome="Citar procedimientos vigentes con grounding documental.",
        ),
        canvas=build_canvas(
            user_goal="Responder con citas y referencias institucionales.",
            expected_outputs=["Respuesta con citas", "Referencia documental"],
        ),
        blueprint=build_blueprint(
            knowledge_mode="rag",
            knowledge_sources=[
                {
                    "key": "manual-rh",
                    "title": "Manual RH",
                    "description": "Politicas internas",
                    "source_type": "document",
                    "uri": "kb://manual-rh",
                    "owner": "People Ops",
                    "license": "internal",
                    "sensitivity": "internal",
                    "source_version": "2026-07",
                }
            ],
        ),
        blueprint_version_number=11,
    )
    evaluated = evaluate_tool_recommendation_artifact(artifact)

    approved_tools, review_decisions, digest = promote_tool_recommendation_to_blueprint_tools(evaluated)

    assert [item.name for item in approved_tools] == ["knowledge_retrieval", "document_ingestion"]
    assert digest.tool_count == 2
    assert digest.knowledge_tool_keys == ["knowledge_retrieval", "document_ingestion"]
    assert digest.recommended_memory_strategy == "persistent_memory"
    assert "knowledge_grounding_required" in digest.memory_hints
    assert {item.tool_key: item.decision for item in review_decisions} == {
        "knowledge_retrieval": "approved",
        "document_ingestion": "approved",
        "broad_write_backoffice": "rejected",
    }


def test_memory_dependency_policy_requires_document_ingestion_for_rag_sources() -> None:
    blueprint = build_blueprint(
        knowledge_mode="rag",
        knowledge_sources=[
            {
                "key": "manual-rh",
                "title": "Manual RH",
                "description": "Politicas internas",
                "source_type": "document",
                "uri": "kb://manual-rh",
                "owner": "People Ops",
                "license": "internal",
                "sensitivity": "internal",
                "source_version": "2026-07",
            }
        ],
    )

    dependencies = build_memory_tool_dependencies(
        approved_tools_digest=None,
        knowledge_profile=blueprint.knowledge_profile,
        memory_profile=blueprint.memory_profile,
    )
    dependency_map = {item.tool_key: item for item in dependencies}

    assert dependency_map["knowledge_retrieval"].required is True
    assert dependency_map["knowledge_retrieval"].status == "missing"
    assert dependency_map["document_ingestion"].required is True
    assert dependency_map["document_ingestion"].status == "missing"


def test_memory_profile_consumes_only_approved_tools_digest() -> None:
    artifact = build_placeholder_tool_recommendation(
        session_id=uuid4(),
        discovery=build_discovery(
            problem_statement="Operar solicitudes con aprobacion controlada.",
            current_process="Revisa solicitudes en portal HR y luego actualiza el estado aprobado o rechazado.",
            desired_outcome="Aprobar o actualizar la solicitud con trazabilidad.",
            autonomy_level="high",
            constraints=["No ejecutar side effects irreversibles sin aprobacion humana"],
            non_delegable_decisions=["Aprobar la solicitud final"],
        ),
        canvas=build_canvas(
            user_goal="Consultar el portal HR y ejecutar la actualizacion aprobada.",
            human_approvals=["Lider RRHH aprueba antes de escribir."],
        ),
        blueprint=build_blueprint(
            guardrails=["Toda escritura requiere aprobacion humana y audit trail"],
            workflow_steps=[
                {
                    "name": "Consultar solicitud",
                    "objective": "Leer estado actual desde portal HR",
                    "actor": "agent",
                    "outputs": ["estado actual"],
                    "fallback": "escalar",
                    "requires_approval": False,
                },
                {
                    "name": "Actualizar estado",
                    "objective": "Escribir el resultado aprobado",
                    "actor": "agent",
                    "outputs": ["estado actualizado"],
                    "fallback": "detener",
                    "requires_approval": True,
                },
            ],
        ),
        blueprint_version_number=12,
    )
    evaluated = evaluate_tool_recommendation_artifact(artifact)
    approved_tools, _, digest = promote_tool_recommendation_to_blueprint_tools(evaluated)

    rebuilt_digest = build_approved_tools_digest_from_blueprint_tools(
        approved_tools,
        source_session_id=evaluated.source_session_id,
        source_blueprint_version=evaluated.source_blueprint_version,
        mandatory_tool_keys=digest.mandatory_tool_keys,
        optional_tool_keys=digest.optional_tool_keys,
    )
    memory_profile = derive_memory_profile(
        build_discovery(
            problem_statement="Operar solicitudes con aprobacion controlada.",
            current_process="Revisa solicitudes y actualiza el estado aprobado.",
            desired_outcome="Actualizar la solicitud aprobada.",
            autonomy_level="high",
            constraints=["No ejecutar side effects irreversibles sin aprobacion humana"],
            non_delegable_decisions=["Aprobar la solicitud final"],
        ),
        build_canvas(
            user_goal="Consultar el portal HR y ejecutar la actualizacion aprobada.",
            human_approvals=["Lider RRHH aprueba antes de escribir."],
        ),
        approved_tools_digest=rebuilt_digest,
    )

    assert "checkpoint_summary" in memory_profile.storage_layers
    assert "approved_tools_digest" in memory_profile.retrieval_policy
    assert "tools aprobadas" in memory_profile.write_policy


def test_tool_recommendation_is_marked_stale_when_context_changes() -> None:
    discovery = build_discovery(
        problem_statement="Coordinar solicitudes con aprobacion controlada.",
        current_process="Revisa solicitudes y actualiza su estado.",
        desired_outcome="Actualizar solo decisiones aprobadas.",
        autonomy_level="high",
        constraints=["No ejecutar side effects irreversibles sin aprobacion humana"],
        non_delegable_decisions=["Aprobar la solicitud final"],
    )
    canvas = build_canvas(
        user_goal="Consultar el portal HR y ejecutar la actualizacion aprobada.",
        human_approvals=["Lider RRHH aprueba antes de escribir."],
    )
    blueprint = build_blueprint(
        guardrails=["Toda escritura requiere aprobacion humana y audit trail"],
        workflow_steps=[
            {
                "name": "Consultar solicitud",
                "objective": "Leer estado actual desde portal HR",
                "actor": "agent",
                "outputs": ["estado actual"],
                "fallback": "escalar",
                "requires_approval": False,
            },
            {
                "name": "Actualizar estado",
                "objective": "Escribir el resultado aprobado",
                "actor": "agent",
                "outputs": ["estado actualizado"],
                "fallback": "detener",
                "requires_approval": True,
            },
        ],
    )
    artifact = build_placeholder_tool_recommendation(
        session_id=uuid4(),
        discovery=discovery,
        canvas=canvas,
        blueprint=blueprint,
        blueprint_version_number=13,
    )
    evaluated = evaluate_tool_recommendation_artifact(artifact)
    approved_tools, _, digest = promote_tool_recommendation_to_blueprint_tools(evaluated)
    promoted_artifact = evaluated.model_copy(update={"approved_tools_digest": digest, "review_state": ReviewState.complete})

    mutated_canvas = build_canvas(
        user_goal="Consultar el portal HR, aprobar y notificar el resultado al owner.",
        expected_outputs=["estado actualizado", "alerta enviada"],
        human_approvals=["Lider RRHH aprueba antes de escribir."],
    )
    annotated = annotate_tool_recommendation_status(
        promoted_artifact,
        discovery=discovery,
        canvas=mutated_canvas,
        blueprint=blueprint.model_copy(update={"tools": approved_tools}),
        current_blueprint_version=14,
    )

    assert annotated.context_digest.digest_sha256
    assert annotated.current_blueprint_version == 14
    assert annotated.is_stale is True
    assert "tool_recommendation_context_changed" in annotated.stale_reasons
    assert "approved_tools_digest_outdated" in annotated.stale_reasons
