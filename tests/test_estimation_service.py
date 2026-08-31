from uuid import uuid4

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from app.models import (
    ACPPreview,
    ACPValidationReport,
    AgentCanvasProfile,
    BlueprintArtifact,
    BlueprintTool,
    CanvasArtifact,
    ConstructionReadinessReport,
    DeliveryPackage,
    DesignAlternative,
    DesignBlueprintProjection,
    DesignRecommendationArtifact,
    DiscoveryArtifact,
    EstimationMaturityStage,
    EvaluationDatasetArtifact,
    EvaluationDatasetCase,
    EvaluationRunEntry,
    MvpDefinition,
    ObservabilityPlan,
    OperationalBaseline,
    JourneyArtifactState,
    JourneyStageArtifactEntry,
    SessionCreateResponse,
    SessionSnapshot,
    SessionStage,
    SkillDefinition,
    WorkflowProfile,
    WorkflowStep,
    MemoryProfile,
    SafetyCheck,
    GeneratedDeliverable,
    ArtifactStatus,
    utc_now,
)
from app.services.estimation_service import build_estimation_report
from app.services.workspace_bootstrap import apply_workspace_bootstrap


def test_build_estimation_report_returns_deterministic_comparative_projection() -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        apply_workspace_bootstrap(session, uuid4())

        snapshot = SessionSnapshot(
            session=SessionCreateResponse(
                id=uuid4(),
                title="Asistente de soporte",
                status=ArtifactStatus.ready,
                current_stage=SessionStage.post_validation,
                created_at=utc_now(),
                updated_at=utc_now(),
            ),
            discovery=DiscoveryArtifact(
                problem_statement="Construir un builder Lean para crear agentes de soporte en produccion.",
                current_user="Arquitecto de soluciones",
                current_process="Descubre, documenta y arma artefactos manualmente.",
                desired_outcome="Generar blueprints, evaluacion y ACP con menos retrabajo.",
                autonomy_level="high",
                constraints=["Sin microservicios en MVP", "Aprobaciones humanas para side effects"],
                operational_baseline=OperationalBaseline(
                    current_time_spent="6 horas",
                    current_cost="Retrabajo operativo",
                    frequent_errors=["Perdida de contexto"],
                    automation_opportunities=["Normalizar discovery", "Generar artefactos base"],
                ),
                mvp_definition=MvpDefinition(
                    v1_scope=["Canvas", "Blueprint", "ACP Preview"],
                    out_of_scope=["Provisioning full"],
                    north_star_metric="Paquete listo para construir",
                    non_delegable_decisions=["Aprobar handoff a build"],
                ),
                case_type="automatizacion",
                value_statement="Reducir tiempo de arquitectura.",
            ),
            canvas=CanvasArtifact(
                user_goal="Aterrizar un agente builder con metodologia Lean.",
                mvp_scope=["Captura", "Blueprint", "Estimacion"],
                out_of_scope=["Microservicios"],
                success_metric="ACP usable",
                primary_risk="Falta de contratos externos",
                agent_profile=AgentCanvasProfile(
                    mission="Guiar el diseno del agente.",
                    primary_user="Arquitecto de soluciones",
                    agent_task="Convertir discovery en blueprint y ACP",
                    human_approvals=["Handoff final"],
                    success_metrics=["Tiempo de entrega", "Cobertura de artefactos"],
                ),
            ),
            blueprint=BlueprintArtifact(
                architecture="single_agent_with_skills",
                reasoning_pattern="Plan-and-Execute",
                memory_strategy="session_memory_with_checkpoints",
                tools=[
                    BlueprintTool(
                        name="Buscar conocimiento",
                        purpose="Consultar base documental interna",
                        risk_level="medium",
                        inputs=["query"],
                        outputs=["answer"],
                        validations=["schema"],
                    ),
                    BlueprintTool(
                        name="Crear ticket",
                        purpose="Abrir ticket en plataforma externa",
                        risk_level="high",
                        requires_approval=True,
                        has_side_effects=True,
                        inputs=["payload"],
                        outputs=["ticket_id"],
                        validations=["schema", "approval"],
                    ),
                ],
                memory_profile=MemoryProfile(
                    strategy="session_memory_with_checkpoints",
                    storage_layers=["session_state", "vector_index"],
                    write_policy="Solo estado validado",
                    retrieval_policy="Priorizar artefactos del workspace",
                    review_trigger="Cambio de etapa",
                    goal_drift_guard="Comparar contra desired_outcome",
                ),
                safety_checks=[
                    SafetyCheck(category="hallucination", risk="Campos inventados", severity="high", mitigation="Schema"),
                    SafetyCheck(category="security", risk="Side effects", severity="high", mitigation="Approval gate"),
                ],
                guardrails=["No inventar contratos API", "No desplegar sin target definido"],
                delivery_package=DeliveryPackage(
                    workflow_profile=WorkflowProfile(
                        execution_pattern="durable_linear",
                        steps=[
                            WorkflowStep(name="discover", objective="Capturar discovery", actor="agent"),
                            WorkflowStep(name="design", objective="Generar blueprint", actor="agent"),
                            WorkflowStep(name="build", objective="Empaquetar ACP", actor="agent", requires_approval=True),
                        ],
                    ),
                    observability_plan=ObservabilityPlan(
                        captured_signals=["inputs", "outputs", "cost", "duration", "alerts"],
                        cost_tracking="persisted",
                        duration_tracking="persisted",
                    ),
                    deliverables=[
                        GeneratedDeliverable(key="prd", title="PRD", summary="PRD base"),
                        GeneratedDeliverable(key="tool_schema", title="Tools", summary="Contratos"),
                        GeneratedDeliverable(key="test_cases", title="Deploy", summary="Guia"),
                    ],
                ),
                readiness_state="complete",
                narrative="Blueprint con memoria, tools y evaluacion.",
            ),
            evaluation_dataset=EvaluationDatasetArtifact(
                cases=[
                    EvaluationDatasetCase(case_key="core_1", title="Caso 1", category="core"),
                    EvaluationDatasetCase(case_key="core_2", title="Caso 2", category="core"),
                    EvaluationDatasetCase(case_key="risk_1", title="Caso 3", category="risk"),
                ]
            ),
            evaluation_runs=[
                EvaluationRunEntry(
                    id=uuid4(),
                    created_at=utc_now(),
                    status=ArtifactStatus.ready,
                    overall_score=88,
                    summary="Corrida base estable",
                )
            ],
            journey_latest_artifacts={
                "design": JourneyStageArtifactEntry(
                    id=uuid4(),
                    workspace_id=uuid4(),
                    session_id=uuid4(),
                    artifact_kind="stage_proposal",
                    stage_key="design",
                    version_number=1,
                    state=JourneyArtifactState.approved,
                    source_action="propose_design",
                    created_at=utc_now(),
                    updated_at=utc_now(),
                    proposal_payload=DesignRecommendationArtifact(
                        alternatives=[
                            DesignAlternative(
                                alternative_key="single-agent-with-skills",
                                label="Agente con skills",
                                architecture="single_agent_with_skills",
                                reasoning_pattern="Plan-and-Execute",
                                pattern_family="Plan-and-Execute",
                                blueprint_projection=DesignBlueprintProjection(
                                    architecture="single_agent_with_skills",
                                    reasoning_pattern="Plan-and-Execute",
                                    cost_complexity_implications=[
                                        "Costo relativo: medium",
                                        "Complejidad operacional: medium",
                                        "Mantenibilidad: high",
                                    ],
                                ),
                            )
                        ],
                        recommended_alternative_key="single-agent-with-skills",
                        selected_design=DesignAlternative(
                            alternative_key="single-agent-with-skills",
                            label="Agente con skills",
                            architecture="single_agent_with_skills",
                            reasoning_pattern="Plan-and-Execute",
                            pattern_family="Plan-and-Execute",
                            blueprint_projection=DesignBlueprintProjection(
                                architecture="single_agent_with_skills",
                                reasoning_pattern="Plan-and-Execute",
                                cost_complexity_implications=[
                                    "Costo relativo: medium",
                                    "Complejidad operacional: medium",
                                    "Mantenibilidad: high",
                                ],
                            ),
                        ),
                    ).model_dump(mode="json"),
                )
            },
            skill_catalog=[SkillDefinition(skill_key="blueprint_generation_skill", label="Blueprint", stage_hint="build")],
        )
        acp_preview = ACPPreview(
            session_id=snapshot.session.id,
            blueprint_version_number=1,
            validation=ACPValidationReport(
                overall_status="complete",
                completeness_percent=100,
                can_export_zip=True,
            ),
            construction_readiness=ConstructionReadinessReport(
                overall_status="ready_to_build",
                can_start_build=True,
                blocking_gaps=0,
                open_questions=0,
                assumptions_count=1,
            ),
        )

        artifact = build_estimation_report(session, snapshot=snapshot, acp_preview=acp_preview)

    assert artifact.maturity_stage == EstimationMaturityStage.ready_to_build
    assert artifact.traditional.estimated_hours_total > 0
    assert artifact.traditional.estimated_cost > 0
    assert artifact.agentic.estimated_hours_total > 0
    assert artifact.agentic.estimated_hours_total < artifact.traditional.estimated_hours_total
    assert artifact.agentic.estimated_cost < artifact.traditional.estimated_cost
    assert artifact.agentic.blueprint_design_coverage_percent >= 65
    assert artifact.agentic.acp_package_readiness_percent >= 85
    assert artifact.agentic.implementation_scope_coverage_percent == artifact.agentic.automation_coverage_percent
    assert artifact.agentic.automation_coverage_percent >= 15
    assert [item.scenario_key for item in artifact.construction_scenarios] == [
        "traditional_blueprint",
        "blueprint_basic",
        "blueprint_premium",
        "agentic_blueprint",
        "acp_manual",
        "acp_agentic",
        "done_for_you_factory",
    ]
    blueprint_basic = next(item for item in artifact.construction_scenarios if item.scenario_key == "blueprint_basic")
    blueprint_premium = next(item for item in artifact.construction_scenarios if item.scenario_key == "blueprint_premium")
    agentic_blueprint = next(item for item in artifact.construction_scenarios if item.scenario_key == "agentic_blueprint")
    acp_manual = next(item for item in artifact.construction_scenarios if item.scenario_key == "acp_manual")
    acp_agentic = next(item for item in artifact.construction_scenarios if item.scenario_key == "acp_agentic")
    done_for_you_factory = next(item for item in artifact.construction_scenarios if item.scenario_key == "done_for_you_factory")
    assert blueprint_basic.estimated_hours_total < artifact.traditional.estimated_hours_total
    assert blueprint_premium.estimated_hours_total < blueprint_basic.estimated_hours_total
    assert agentic_blueprint.estimated_hours_total < blueprint_premium.estimated_hours_total
    assert agentic_blueprint.estimated_cost < blueprint_premium.estimated_cost
    assert done_for_you_factory.estimated_hours_total == acp_manual.estimated_hours_total
    assert done_for_you_factory.estimated_duration_weeks == artifact.traditional.estimated_duration_weeks
    assert done_for_you_factory.estimated_cost == round(acp_agentic.estimated_cost * 0.9, 2)
    assert blueprint_basic.automation_leverage_percent < blueprint_premium.automation_leverage_percent
    assert any("Inferir + registrar + continuar" in note for note in blueprint_basic.notes)
    assert any("Pregunta + resolver + enriquecer" in note for note in blueprint_premium.notes)
    assert any("mejorar el esfuerzo frente a Blueprint Premium" in note for note in agentic_blueprint.notes)
    assert any("cotizacion separada" in note for note in done_for_you_factory.notes)
    assert acp_agentic.estimated_hours_total == artifact.agentic.estimated_hours_total
    assert acp_agentic.cost_savings_vs_traditional == artifact.agentic.net_savings_vs_traditional
    assert artifact.agentic.automation_assessments
    assert artifact.agentic.pricing_snapshot is not None
    assert artifact.agentic.pricing_snapshot.provider.value in {"openai", "codex_local"}
    assert artifact.agentic.provider_runtime_cost_total_usd > 0
    assert artifact.agentic.estimated_cost > artifact.agentic.human_delivery_cost
    assert any("Design aporto implicaciones" in note for note in artifact.notes)
    assert any("Estimate incorpora implicaciones" in assumption for assumption in artifact.assumptions)
    assert any(item.family_key == "deployment_infra" for item in artifact.agentic.automation_assessments)
    assert any(item.family_key == "implementation_code" for item in artifact.agentic.automation_assessments)
    assert any(item.bonuses_applied for item in artifact.agentic.automation_assessments)
    assert artifact.agentic.net_savings_vs_traditional > 0
    assert artifact.confidence.score >= 60
    assert artifact.confidence.label in {"medium", "medium_high", "high"}

    tool_schema = next(item for item in artifact.agentic.automation_assessments if item.family_key == "tool_schemas")
    assert tool_schema.label == "Tool schemas / contratos internos"
    assert "Hay side effects todavia no gobernados" in tool_schema.penalties_applied
    assert "Todos los tools tienen esquema validable" in tool_schema.bonuses_applied

    deployment = next(item for item in artifact.agentic.automation_assessments if item.family_key == "deployment_infra")
    assert "ACP con readiness operativo cerrado" in deployment.bonuses_applied
    assert deployment.mandatory_human_review is True
    assert deployment.coverage_percent >= 15


def test_agentic_platform_internal_rag_tools_are_not_external_api_implementation_scope() -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        apply_workspace_bootstrap(session, uuid4())

        snapshot = SessionSnapshot(
            session=SessionCreateResponse(
                id=uuid4(),
                title="Agente de soporte con RAG",
                status=ArtifactStatus.ready,
                current_stage=SessionStage.post_validation,
                created_at=utc_now(),
                updated_at=utc_now(),
            ),
            discovery=DiscoveryArtifact(
                problem_statement="Automatizar atencion de soporte con retrieval documental y aprobaciones humanas.",
                current_process="Los analistas consultan documentos y responden solicitudes repetitivas.",
                desired_outcome="Blueprint y ACP para construir un agente de soporte con conocimiento gobernado.",
                autonomy_level="medium",
                constraints=["La implementacion de APIs externas queda fuera del alcance"],
                mvp_definition=MvpDefinition(
                    v1_scope=["Blueprint", "ACP", "RAG"],
                    out_of_scope=["Implementacion de servicios externos", "Deployment productivo"],
                    north_star_metric="ACP accionable",
                ),
            ),
            canvas=CanvasArtifact(
                user_goal="Disenar un agente de soporte con RAG interno.",
                mvp_scope=["Blueprint", "ACP"],
                out_of_scope=["Construccion de APIs externas"],
                success_metric="Paquete listo para handoff",
                primary_risk="Fuentes de conocimiento incompletas",
            ),
            blueprint=BlueprintArtifact(
                architecture="single_agent_with_skills",
                reasoning_pattern="Plan-and-Execute",
                memory_strategy="rag_with_session_checkpoints",
                tools=[
                    BlueprintTool(
                        name="knowledge_retrieval",
                        purpose="Consultar conocimiento aprobado",
                        tool_type="internal",
                        integration_kind="retrieval",
                        archetype="knowledge_retrieval",
                        endpoint_reference="knowledge://approved/retrieve",
                        inputs=["query"],
                        outputs=["answer"],
                        validations=["request_schema_validation", "response_schema_validation"],
                    ),
                    BlueprintTool(
                        name="document_ingestion",
                        purpose="Actualizar indice documental aprobado",
                        integration_kind="pipeline",
                        archetype="document_ingestion",
                        endpoint_reference="knowledge://approved-ingestion/refresh",
                        has_side_effects=True,
                        requires_approval=True,
                        inputs=["source_id"],
                        outputs=["ingestion_run_id"],
                        validations=["payload_schema_validation"],
                    ),
                    BlueprintTool(
                        name="approval_gate",
                        purpose="Solicitar aprobacion humana",
                        tool_type="internal",
                        integration_kind="governed_handoff",
                        archetype="approval_gate",
                        inputs=["decision"],
                        outputs=["approval_token"],
                        validations=["request_schema_validation"],
                    ),
                    BlueprintTool(
                        name="read_system_of_record",
                        purpose="Leer datos por contrato placeholder del sistema de registro",
                        integration_kind="api",
                        archetype="read_only_lookup",
                        endpoint_reference="integration://system-of-record/read",
                        inputs=["record_id"],
                        outputs=["record"],
                        validations=["request_schema_validation", "response_schema_validation"],
                    ),
                    BlueprintTool(
                        name="transactional_write",
                        purpose="Contrato placeholder para escritura gobernada durante implementacion ACP",
                        integration_kind="api",
                        archetype="transactional_write",
                        endpoint_reference="integration://system-of-record/write",
                        has_side_effects=True,
                        requires_approval=True,
                        inputs=["payload"],
                        outputs=["operation_id"],
                        validations=["payload_schema_validation", "approval_token_validation"],
                    ),
                ],
                memory_profile=MemoryProfile(
                    strategy="rag_with_session_checkpoints",
                    storage_layers=["session_state", "vector_index", "artifact_store"],
                    write_policy="Persistir solo artefactos aprobados",
                    retrieval_policy="Recuperar por workspace, etapa y artefacto",
                    review_trigger="Cambio de etapa",
                    goal_drift_guard="Comparar contra el objetivo del canvas",
                ),
                guardrails=["No responder sin evidencia", "Escalar contradicciones"],
                safety_checks=[
                    SafetyCheck(category="knowledge", risk="Respuesta sin fuentes", severity="high", mitigation="Citas obligatorias"),
                ],
                delivery_package=DeliveryPackage(
                    workflow_profile=WorkflowProfile(
                        execution_pattern="durable_linear",
                        steps=[
                            WorkflowStep(name="retrieve", objective="Buscar evidencia", actor="agent"),
                            WorkflowStep(name="answer", objective="Proponer respuesta", actor="agent"),
                            WorkflowStep(name="approve", objective="Aprobar side effects", actor="human", requires_approval=True),
                        ],
                    ),
                    observability_plan=ObservabilityPlan(captured_signals=["retrieval_hits", "citations", "approvals"]),
                    deliverables=[GeneratedDeliverable(key="tool_schema", title="Tools", summary="Contratos internos")],
                ),
                readiness_state="complete",
                narrative="Blueprint agentico con RAG y herramientas internas gobernadas.",
            ),
        )

        artifact = build_estimation_report(session, snapshot=snapshot, acp_preview=None)

    family_keys = {item.family_key for item in artifact.agentic.automation_assessments}
    assert "external_api_contracts" not in family_keys
    assert artifact.agentic.blueprint_design_coverage_percent >= 60
    tool_schema = next(item for item in artifact.agentic.automation_assessments if item.family_key == "tool_schemas")
    assert "Todos los tools tienen esquema validable" in tool_schema.bonuses_applied
    assert "Hay side effects todavia no gobernados" not in tool_schema.penalties_applied


def test_build_estimation_report_penalizes_incomplete_integrations() -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        apply_workspace_bootstrap(session, uuid4())

        snapshot = SessionSnapshot(
            session=SessionCreateResponse(
                id=uuid4(),
                title="Integraciones enterprise",
                status=ArtifactStatus.ready,
                current_stage=SessionStage.post_validation,
                created_at=utc_now(),
                updated_at=utc_now(),
            ),
            discovery=DiscoveryArtifact(
                problem_statement="Disenar un flujo que opera sobre CRM y ERP con side effects.",
                current_user="Lider de integraciones",
                current_process="Se documentan APIs de forma parcial.",
                desired_outcome="Tener un blueprint inicial para automatizar integraciones de ventas y soporte.",
                autonomy_level="medium",
                constraints=["No ejecutar cambios sin aprobacion"],
                operational_baseline=OperationalBaseline(
                    current_time_spent="8 horas",
                    current_cost="Retrabajo por contratos incompletos",
                    frequent_errors=["Faltan payloads definitivos"],
                    automation_opportunities=["Generar artefactos base"],
                ),
                mvp_definition=MvpDefinition(
                    v1_scope=["Canvas", "Blueprint"],
                    out_of_scope=["Despliegue productivo"],
                    north_star_metric="Flujo base aprobado",
                    non_delegable_decisions=["Aprobacion de side effects"],
                ),
                case_type="automatizacion",
                value_statement="Reducir tiempo de definicion tecnica.",
            ),
            canvas=CanvasArtifact(
                user_goal="Automatizar un flujo con integraciones externas.",
                mvp_scope=["Discovery", "Blueprint"],
                out_of_scope=["Go live"],
                success_metric="Blueprint aprobado",
                primary_risk="Contratos externos abiertos",
                agent_profile=AgentCanvasProfile(
                    mission="Traducir discovery a blueprint enterprise de integraciones.",
                    primary_user="Lider de integraciones",
                    agent_task="Mapear herramientas y riesgos",
                    human_approvals=["Aprobacion de integraciones"],
                    success_metrics=["Cobertura base"],
                ),
            ),
            blueprint=BlueprintArtifact(
                architecture="workflow_with_tools",
                reasoning_pattern="Plan-and-Execute",
                memory_strategy="state_store",
                tools=[
                    BlueprintTool(
                        name="Actualizar CRM",
                        purpose="Crea o modifica registros externos",
                        risk_level="high",
                        requires_approval=True,
                        has_side_effects=True,
                        inputs=["payload"],
                        outputs=["result"],
                        validations=["schema"],
                        )
                    ],
                memory_profile=MemoryProfile(strategy="state_store"),
                safety_checks=[
                    SafetyCheck(category="operational", risk="Side effects", severity="high", mitigation="Approval gate"),
                ],
                delivery_package=DeliveryPackage(
                    workflow_profile=WorkflowProfile(
                        execution_pattern="durable_linear",
                        steps=[WorkflowStep(name="design", objective="Disenar", actor="system")],
                    ),
                        observability_plan=ObservabilityPlan(captured_signals=["inputs", "outputs"]),
                        deliverables=[GeneratedDeliverable(key="tool_schema", title="Contracts", summary="Draft")],
                    ),
                    readiness_state="partial",
                    narrative="Blueprint con una integracion enterprise aun incompleta.",
                ),
            )

        artifact = build_estimation_report(session, snapshot=snapshot, acp_preview=None)

    assert artifact.maturity_stage == EstimationMaturityStage.blueprint
    assert artifact.agentic.pricing_snapshot is not None
    assert artifact.agentic.pricing_snapshot.provider.value in {"openai", "codex_local"}
    integration_family = next(item for item in artifact.agentic.automation_assessments if item.family_key == "external_api_contracts")
    assert "Integraciones con side effects productivos" in integration_family.penalties_applied
    assert integration_family.non_automatable_reasons
    assert integration_family.coverage_percent <= 35
    assert artifact.confidence.implementation_open_questions >= 0
    assert artifact.confidence.design_gap_count <= artifact.confidence.blocking_gaps
