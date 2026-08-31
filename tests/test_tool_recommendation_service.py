from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from sqlmodel import SQLModel, Session, create_engine, select

from app.models import (
    BlueprintArtifact,
    CanvasArtifact,
    DesignAlternative,
    DesignBlueprintProjection,
    DesignRecommendationArtifact,
    DesignRole,
    DiscoveryArtifact,
    JourneyStageArtifactRecord,
    ReviewState,
    SessionRecord,
    ToolPatternLearningCandidateRecord,
    ToolRecommendationArtifact,
    WorkspaceRecord,
)
from app.services.llm_runtime.builder_contracts import NonFunctionalRequirement, RequirementsDefinitionOutput
from app.services.knowledge_tool_policy import build_memory_tool_dependencies
from app.services.rules import derive_memory_profile
from app.services.stage_proposal_service import StageProposalService
from app.services.tool_recommendation_service import (
    annotate_tool_recommendation_status,
    build_approved_tools_digest_from_blueprint_tools,
    build_placeholder_tool_recommendation,
    build_tool_recommendation_prompt_input,
    ensure_document_ingestion_for_knowledge_retrieval,
    ensure_memory_tool_dependencies,
    evaluate_tool_recommendation_artifact,
    promote_tool_recommendation_to_blueprint_tools,
)
from app.services.tool_pattern_learning_service import persist_tool_pattern_learning_candidates


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
        ),
        canvas=build_canvas(user_goal="Consultar CRM y tickets para responder con grounding."),
        blueprint=build_blueprint(guardrails=["Solo lectura sobre sistemas operativos", "Mantener trazabilidad"]),
        blueprint_version_number=1,
    )

    candidate_map = {item.family_key: item for item in artifact.preflight.candidate_tool_families}

    assert artifact.preflight.case_classification == "enterprise_copilot"
    assert [item.tool_key for item in artifact.recommended_tools] == ["read_system_of_record"]
    assert artifact.optional_tools == []
    assert artifact.needs_information == []
    assert candidate_map["read_only_lookup"].status == "required"
    assert candidate_map["transactional_write"].status == "excluded"
    assert artifact.learning_report.global_write_allowed is False
    assert artifact.learning_report.candidate_count == len(artifact.candidate_tool_patterns)


def test_tool_learning_report_prepares_safe_patterns_without_writing_global_knowledge() -> None:
    artifact = build_placeholder_tool_recommendation(
        session_id=uuid4(),
        discovery=build_discovery(
            problem_statement="Ayudar a soporte a responder usando datos operativos reales.",
            current_process="Consultar CRM y tickets abiertos del cliente antes de responder.",
            desired_outcome="Responder estado del caso sin actualizar registros.",
        ),
        canvas=build_canvas(user_goal="Consultar CRM y tickets para responder con grounding."),
        blueprint=build_blueprint(guardrails=["Solo lectura sobre sistemas operativos", "Mantener trazabilidad"]),
        blueprint_version_number=31,
    )

    report = artifact.learning_report
    candidates = {item.capability_key: item for item in report.candidates}

    assert report.schema_version == "tool-pattern-learning.v1"
    assert report.global_write_allowed is False
    assert report.ready_for_global_review_count == 1
    assert "knowledge_documents:tooling_pattern_catalog" in report.catalog_refs
    assert candidates["read_system_of_record"].promotion_status == "ready_for_global_review"
    assert candidates["read_system_of_record"].global_promotion_allowed is True
    assert candidates["read_system_of_record"].contract_quality == "complete"
    assert candidates["read_system_of_record"].source_level == "project_tool"
    assert candidates["read_system_of_record"].dedupe_signature


def test_tool_learning_candidates_persist_and_dedupe_by_session_signature() -> None:
    with TemporaryDirectory(prefix="lean-builder-tool-learning-") as tmp_dir:
        engine = create_engine(
            f"sqlite:///{(Path(tmp_dir) / 'tool-learning.db').as_posix()}",
            connect_args={"check_same_thread": False},
        )
        try:
            SQLModel.metadata.create_all(engine)
            workspace_id = uuid4()
            session_id = uuid4()
            with Session(engine) as session:
                session.add(WorkspaceRecord(id=workspace_id, name="Learning Workspace", slug="learning-workspace"))
                session.add(SessionRecord(id=session_id, user_id=uuid4(), workspace_id=workspace_id))
                session.commit()

                artifact = build_placeholder_tool_recommendation(
                    session_id=session_id,
                    discovery=build_discovery(
                        problem_statement="Ayudar a soporte a responder usando datos operativos reales.",
                        current_process="Consultar CRM y tickets abiertos del cliente antes de responder.",
                        desired_outcome="Responder estado del caso sin actualizar registros.",
                    ),
                    canvas=build_canvas(user_goal="Consultar CRM y tickets para responder con grounding."),
                    blueprint=build_blueprint(
                        guardrails=["Solo lectura sobre sistemas operativos", "Mantener trazabilidad"]
                    ),
                    blueprint_version_number=32,
                )

                first = persist_tool_pattern_learning_candidates(
                    session,
                    workspace_id=workspace_id,
                    session_id=session_id,
                    recommendation=artifact,
                )
                second = persist_tool_pattern_learning_candidates(
                    session,
                    workspace_id=workspace_id,
                    session_id=session_id,
                    recommendation=artifact,
                )
                rows = session.exec(
                    select(ToolPatternLearningCandidateRecord).where(
                        ToolPatternLearningCandidateRecord.workspace_id == workspace_id,
                        ToolPatternLearningCandidateRecord.session_id == session_id,
                    )
                ).all()
        finally:
            engine.dispose()

    assert first.inserted_count == 1
    assert first.updated_count == 0
    assert second.inserted_count == 0
    assert second.updated_count == 1
    assert len(rows) == 1
    assert rows[0].observation_count == 2
    assert rows[0].promotion_status == "ready_for_global_review"
    assert rows[0].global_promotion_allowed is True
    assert rows[0].contract_seed_payload


def test_tool_recommendation_legacy_payload_defaults_learning_report_safely() -> None:
    artifact = ToolRecommendationArtifact.model_validate(
        {
            "source_session_id": str(uuid4()),
            "schema_version": "tool-recommendation.v1",
            "recommended_tools": [],
            "optional_tools": [],
            "rejected_tools": [],
        }
    )

    assert artifact.learning_report.schema_version == "tool-pattern-learning.v1"
    assert artifact.learning_report.global_write_allowed is False
    assert artifact.learning_report.candidates == []


def test_tools_preflight_consumes_design_implications_as_architecture_evidence() -> None:
    design = DesignRecommendationArtifact(
        alternatives=[
            DesignAlternative(
                alternative_key="skill_orchestrator",
                label="Orquestador con retrieval gobernado",
                architecture="single_agent_with_skills",
                reasoning_pattern="ReAct",
                tool_implications=[
                    "knowledge_retrieval: recuperar politicas aprobadas antes de responder.",
                ],
                memory_implications=[
                    "source_ref_grounding: conservar referencias y versiones de fuente.",
                ],
                blueprint_projection=DesignBlueprintProjection(
                    architecture="single_agent_with_skills",
                    reasoning_pattern="ReAct",
                    tool_implications=[
                        "approval_gate: revisar decisiones sensibles antes de promover.",
                    ],
                    memory_implications=[
                        "decision_traceability: conservar decision, evidencia y owner.",
                    ],
                ),
            )
        ],
        recommended_alternative_key="skill_orchestrator",
    )

    artifact = build_placeholder_tool_recommendation(
        session_id=uuid4(),
        discovery=build_discovery(
            problem_statement="Responder preguntas usando lineamientos internos.",
            current_process="El equipo interpreta reglas manualmente.",
            desired_outcome="Responder con evidencia y control de aprobacion.",
        ),
        canvas=build_canvas(user_goal="Resolver solicitudes con evidencia."),
        blueprint=build_blueprint(),
        design_artifact=design,
        blueprint_version_number=1,
    )
    mandatory = {item.capability_key for item in artifact.preflight.mandatory_capabilities}
    families = {item.family_key: item for item in artifact.preflight.candidate_tool_families}

    assert "knowledge_retrieval" in mandatory
    assert "approval_gate" in mandatory
    assert artifact.preflight.design_tool_implications
    assert artifact.preflight.design_memory_implications
    assert families["retrieval"].status == "required"
    assert families["approval_control"].status == "required"
    resolutions = {item.capability_key: item for item in artifact.capability_resolutions}
    patterns = {item.capability_key: item for item in artifact.candidate_tool_patterns}
    assert resolutions["knowledge_retrieval"].necessity == "required"
    assert resolutions["knowledge_retrieval"].promotion_policy == "auto"
    assert resolutions["knowledge_retrieval"].available is False
    assert resolutions["approval_gate"].necessity == "required"
    assert patterns["knowledge_retrieval"].status == "ready_for_project"
    assert patterns["approval_gate"].status == "ready_for_project"

    prompt_input = build_tool_recommendation_prompt_input(artifact)
    assert prompt_input.design_tool_implications == artifact.preflight.design_tool_implications
    assert prompt_input.design_memory_implications == artifact.preflight.design_memory_implications


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
    assert all(item.contract_seed is not None for item in artifact.recommended_tools)
    assert {item.tool_key: item.contract_seed.tool_type for item in artifact.recommended_tools if item.contract_seed} == {
        "knowledge_retrieval": "internal",
        "document_ingestion": "external",
    }
    assert artifact.optional_tools == []
    assert candidate_map["retrieval"].status == "required"
    assert candidate_map["document_ingestion"].status == "required"
    assert artifact.needs_information == []


def test_rag_without_final_sources_still_requires_ingestion_capability() -> None:
    artifact = build_placeholder_tool_recommendation(
        session_id=uuid4(),
        discovery=build_discovery(
            problem_statement="Responder preguntas usando politicas, manuales y FAQ aprobadas.",
            current_process="Hoy el equipo consulta politicas internas antes de responder.",
            desired_outcome="Responder con evidencia y citas cuando exista soporte documental.",
        ),
        canvas=build_canvas(
            user_goal="Responder con grounding documental y trazabilidad.",
            expected_outputs=["Respuesta con citas", "Evidencia consultada"],
        ),
        blueprint=build_blueprint(knowledge_mode="rag"),
        blueprint_version_number=21,
    )

    candidate_map = {item.family_key: item for item in artifact.preflight.candidate_tool_families}

    assert [item.tool_key for item in artifact.recommended_tools] == [
        "knowledge_retrieval",
        "document_ingestion",
    ]
    assert candidate_map["document_ingestion"].status == "required"
    assert artifact.needs_information == []


def test_auto_remediation_adds_ingestion_for_legacy_retrieval_digest() -> None:
    base_artifact = build_placeholder_tool_recommendation(
        session_id=uuid4(),
        discovery=build_discovery(
            problem_statement="Responder preguntas con conocimiento aprobado.",
            current_process="Consulta politicas internas.",
            desired_outcome="Responder con evidencia.",
        ),
        canvas=build_canvas(user_goal="Responder con conocimiento aprobado."),
        blueprint=build_blueprint(knowledge_mode="rag"),
        blueprint_version_number=22,
    )
    legacy_artifact = base_artifact.model_copy(
        update={
            "recommended_tools": [
                item for item in base_artifact.recommended_tools if item.tool_key == "knowledge_retrieval"
            ],
            "optional_tools": [],
            "rejected_tools": [],
        },
        deep=True,
    )
    retrieval_only_tool = legacy_artifact.recommended_tools[0].contract_seed
    assert retrieval_only_tool is not None
    legacy_digest = build_approved_tools_digest_from_blueprint_tools(
        [retrieval_only_tool],
        source_session_id=legacy_artifact.source_session_id,
        source_blueprint_version=legacy_artifact.source_blueprint_version,
        mandatory_tool_keys=["knowledge_retrieval"],
    )
    legacy_artifact = legacy_artifact.model_copy(update={"approved_tools_digest": legacy_digest})

    remediated, changed = ensure_document_ingestion_for_knowledge_retrieval(
        artifact=legacy_artifact,
        blueprint=build_blueprint(knowledge_mode="rag"),
    )

    assert changed is True
    assert [item.tool_key for item in remediated.recommended_tools] == [
        "knowledge_retrieval",
        "document_ingestion",
    ]
    assert any(item.capability_key == "document_ingestion" for item in remediated.preflight.mandatory_capabilities)
    assert remediated.evaluation.promotion_blocked is False
    approved_tools, _, digest = promote_tool_recommendation_to_blueprint_tools(remediated)
    assert [item.name for item in approved_tools] == ["knowledge_retrieval", "document_ingestion"]
    assert digest.knowledge_tool_keys == ["knowledge_retrieval", "document_ingestion"]


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


def test_memory_dependency_remediation_promotes_missing_scheduler() -> None:
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
    legacy_tools = [item for item in artifact.recommended_tools if item.tool_key != "scheduler"]
    approved_tools = [
        item.contract_seed
        for item in legacy_tools
        if item.contract_seed is not None
    ]
    legacy_digest = build_approved_tools_digest_from_blueprint_tools(
        approved_tools,
        source_session_id=artifact.source_session_id,
        source_blueprint_version=artifact.source_blueprint_version,
        mandatory_tool_keys=[item.tool_key for item in legacy_tools],
    )
    legacy_artifact = artifact.model_copy(
        update={
            "recommended_tools": legacy_tools,
            "approved_tools_digest": legacy_digest,
        },
        deep=True,
    )

    remediated, added_tool_keys = ensure_memory_tool_dependencies(
        artifact=legacy_artifact,
        blueprint=build_blueprint(knowledge_mode="rag", knowledge_refresh_frequency="daily"),
        required_tool_keys=["scheduler"],
        source_reason="Memoria requiere refresh programado.",
    )

    assert added_tool_keys == ["scheduler"]
    assert [item.tool_key for item in remediated.recommended_tools] == [
        "knowledge_retrieval",
        "document_ingestion",
        "scheduler",
    ]
    assert any(item.capability_key == "scheduler" for item in remediated.preflight.mandatory_capabilities)
    scheduler_resolution = {
        item.capability_key: item for item in remediated.capability_resolutions
    }["scheduler"]
    assert scheduler_resolution.necessity == "required"
    assert scheduler_resolution.promotion_policy == "human_review"
    assert "tools.memory_dependency_remediation" in scheduler_resolution.source_evidence
    approved_after_remediation, _, digest = promote_tool_recommendation_to_blueprint_tools(remediated)
    assert [item.name for item in approved_after_remediation] == [
        "knowledge_retrieval",
        "document_ingestion",
        "scheduler",
    ]
    assert "scheduler" in digest.approved_tool_keys


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
    assert artifact.capability_resolutions[0].capability_key == "outbound_notification"
    assert artifact.capability_resolutions[0].side_effect_level == "medium"
    assert artifact.capability_resolutions[0].promotion_policy == "human_review"
    assert artifact.candidate_tool_patterns[0].status == "human_review"
    learning_candidate = artifact.learning_report.candidates[0]
    assert learning_candidate.promotion_status == "needs_human_review"
    assert learning_candidate.global_promotion_allowed is False
    assert "side_effect_level:medium" in learning_candidate.risk_flags


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


def test_internal_design_roles_do_not_block_tool_promotion() -> None:
    selected_design = DesignAlternative(
        alternative_key="supervisor_with_subagents",
        label="Supervisor con especialistas internos",
        architecture="supervisor_with_subagents",
        reasoning_pattern="ReAct",
        coordination_model="Supervisor enruta a especialistas de dominio internos.",
        roles=[
            DesignRole(
                key="domain_specialists",
                title="Especialistas de dominio",
                responsibility="Resolver criterios internos por dominio usando routing del supervisor.",
            )
        ],
        fit_score=80,
    )
    design_artifact = DesignRecommendationArtifact(
        alternatives=[selected_design],
        recommended_alternative_key=selected_design.alternative_key,
        selected_design=selected_design,
        review_state=ReviewState.complete,
        summary="Diseno aprobado con roles internos.",
    )
    artifact = build_placeholder_tool_recommendation(
        session_id=uuid4(),
        discovery=build_discovery(
            problem_statement="Atender solicitudes de empleados con politicas internas y datos del portal HR.",
            current_process="Consulta portal HR y manuales antes de actualizar el estado aprobado.",
            desired_outcome="Responder y actualizar solicitudes aprobadas con trazabilidad.",
            autonomy_level="high",
            constraints=["No ejecutar side effects irreversibles sin aprobacion humana"],
            non_delegable_decisions=["Aprobar la solicitud final"],
        ),
        canvas=build_canvas(
            user_goal="Consultar portal HR, recuperar politicas y ejecutar la actualizacion aprobada.",
            expected_outputs=["Respuesta trazable", "Estado actualizado"],
            human_approvals=["Lider RH aprueba antes de escribir."],
        ),
        blueprint=build_blueprint(
            architecture="supervisor_with_subagents",
            guardrails=["Toda escritura requiere aprobacion humana y audit trail"],
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
        design_artifact=design_artifact,
        blueprint_version_number=101,
    )

    evaluated = evaluate_tool_recommendation_artifact(artifact)
    finding = next(
        item
        for item in evaluated.evaluation.findings
        if item.finding_key == "design-role-coverage:domain_specialists"
    )

    assert finding.severity == "warning"
    assert evaluated.evaluation.promotion_blocked is False
    assert all(item.contract_seed is not None for item in evaluated.recommended_tools)


def test_stage_proposal_approval_reevaluates_persisted_tools_payload_before_promotion() -> None:
    artifact = build_placeholder_tool_recommendation(
        session_id=uuid4(),
        discovery=build_discovery(
            problem_statement="Ayudar a soporte a responder usando datos operativos reales.",
            current_process="Consultar CRM y tickets abiertos del cliente antes de responder.",
            desired_outcome="Responder estado del caso sin actualizar registros.",
        ),
        canvas=build_canvas(user_goal="Consultar CRM y tickets para responder con grounding."),
        blueprint=build_blueprint(
            guardrails=["Solo lectura sobre sistemas operativos", "Mantener trazabilidad"]
        ),
        blueprint_version_number=102,
    )
    evaluated = evaluate_tool_recommendation_artifact(artifact)
    persisted_stale_blocked = evaluated.model_copy(
        update={
            "evaluation": evaluated.evaluation.model_copy(update={"promotion_blocked": True}),
            "review_state": ReviewState.blocked,
        }
    )
    workspace_id = uuid4()
    session_id = uuid4()
    artifact_record = JourneyStageArtifactRecord(
        workspace_id=workspace_id,
        session_id=session_id,
        artifact_kind="tool_recommendation_artifact",
        stage_key="tools",
        schema_version="tool-recommendation.v1",
        proposal_payload=persisted_stale_blocked.model_dump(mode="json"),
    )
    session_record = SessionRecord(
        id=session_id,
        user_id=uuid4(),
        workspace_id=workspace_id,
    )

    approved_tools, digest, promoted_payload, recommendation = StageProposalService()._resolve_tools_projection_payload(
        artifact_record=artifact_record,
        decision_payload={"include_optional_tool_keys": []},
        session_record=session_record,
    )

    assert approved_tools
    assert digest.tool_count == len(approved_tools)
    assert recommendation is not None
    assert promoted_payload["evaluation"]["promotion_blocked"] is False
    assert promoted_payload["review_state"] == ReviewState.complete


def test_stage_proposal_artifact_entry_uses_payload_schema_version_for_legacy_tools() -> None:
    artifact = build_placeholder_tool_recommendation(
        session_id=uuid4(),
        discovery=build_discovery(
            problem_statement="Responder con grounding documental aprobado.",
            current_process="Consultar manuales internos antes de responder.",
            desired_outcome="Entregar respuestas trazables sin depender de memoria humana.",
        ),
        canvas=build_canvas(user_goal="Consultar conocimiento aprobado y responder con citas."),
        blueprint=build_blueprint(guardrails=["Mantener trazabilidad"]),
        blueprint_version_number=7,
    )
    artifact_record = JourneyStageArtifactRecord(
        workspace_id=uuid4(),
        session_id=uuid4(),
        artifact_kind="tool_recommendation_artifact",
        stage_key="tools",
        schema_version="",
        proposal_payload=artifact.model_dump(mode="json"),
    )

    entry = StageProposalService()._build_artifact_entry(None, artifact_record, decisions=[])

    assert entry is not None
    assert entry.schema_version == "tool-recommendation.v1"


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


def test_tool_recommendation_ignores_downstream_memory_changes_after_promotion() -> None:
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
    downstream_blueprint = blueprint.model_copy(
        update={
            "tools": approved_tools,
            "memory_strategy": "semantic_rag_with_short_term_checkpoints",
            "narrative": "Memoria aprobo una estrategia downstream sin cambiar Discover, Define ni Design.",
        }
    )

    annotated = annotate_tool_recommendation_status(
        promoted_artifact,
        discovery=discovery,
        canvas=canvas,
        blueprint=downstream_blueprint,
        current_blueprint_version=14,
    )

    assert annotated.current_blueprint_version == 14
    assert annotated.is_stale is False
    assert annotated.stale_reasons == []
