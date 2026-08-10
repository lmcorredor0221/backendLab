from uuid import uuid4

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from app.models import (
    ACPPreview,
    AgentCanvasProfile,
    BlueprintArtifact,
    BlueprintTool,
    CanvasArtifact,
    ConstructionGapEntry,
    ConstructionReadinessReport,
    DeliveryPackage,
    DiscoveryArtifact,
    EvaluationDatasetArtifact,
    EvaluationDatasetCase,
    EvaluationRunEntry,
    GeneratedDeliverable,
    MemoryProfile,
    MvpDefinition,
    ObservabilityPlan,
    OperationalBaseline,
    ReviewState,
    SessionCreateResponse,
    SessionSnapshot,
    SessionStage,
    SafetyCheck,
    ArtifactStatus,
    WorkflowProfile,
    WorkflowStep,
    utc_now,
)
from app.services.estimation_service import build_estimation_report
from app.services.workspace_bootstrap import apply_workspace_bootstrap


def test_estimation_confidence_progresses_from_canvas_to_ready_to_build() -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        apply_workspace_bootstrap(session, uuid4())

        canvas_snapshot = build_snapshot(title="Canvas temprano", include_blueprint=False)
        blueprint_snapshot = build_snapshot(title="Blueprint parcial", include_blueprint=True, include_dataset=False, include_run=False)
        ready_snapshot = build_snapshot(title="ACP listo", include_blueprint=True, include_dataset=True, include_run=True)

        canvas_report = build_estimation_report(session, snapshot=canvas_snapshot, acp_preview=None)
        blueprint_report = build_estimation_report(session, snapshot=blueprint_snapshot, acp_preview=build_partial_acp_preview(blueprint_snapshot))
        ready_report = build_estimation_report(session, snapshot=ready_snapshot, acp_preview=build_ready_acp_preview(ready_snapshot))

    assert canvas_report.maturity_stage == "canvas"
    assert blueprint_report.maturity_stage == "blueprint"
    assert ready_report.maturity_stage == "ready_to_build"

    assert ready_report.confidence.score > blueprint_report.confidence.score > canvas_report.confidence.score
    assert canvas_report.confidence.uncertainty_band_percent >= blueprint_report.confidence.uncertainty_band_percent
    assert blueprint_report.confidence.uncertainty_band_percent >= ready_report.confidence.uncertainty_band_percent

    assert blueprint_report.confidence.subscores["api_contract_maturity"] == 12
    assert blueprint_report.confidence.subscores["deployment_maturity"] == 12
    assert blueprint_report.confidence.subscores["knowledge_maturity"] == 12
    assert ready_report.confidence.subscores["api_contract_maturity"] == 96
    assert ready_report.confidence.subscores["deployment_maturity"] == 96
    assert ready_report.confidence.subscores["knowledge_maturity"] == 96
    assert ready_report.confidence.subscores["build_readiness"] == 96
    assert any("deployment target" in item.lower() for item in canvas_report.confidence.recommended_next_actions)
    assert any("contratos api" in item.lower() for item in blueprint_report.confidence.recommended_next_actions)


def test_estimation_confidence_band_widens_when_gaps_increase() -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        apply_workspace_bootstrap(session, uuid4())

        snapshot = build_snapshot(title="Blueprint con ACP", include_blueprint=True, include_dataset=True, include_run=False)
        ready_report = build_estimation_report(session, snapshot=snapshot, acp_preview=build_ready_acp_preview(snapshot))
        blocked_report = build_estimation_report(session, snapshot=snapshot, acp_preview=build_partial_acp_preview(snapshot))

    assert blocked_report.maturity_stage == "blueprint"
    assert ready_report.confidence.score > blocked_report.confidence.score
    assert ready_report.confidence.uncertainty_band_percent < blocked_report.confidence.uncertainty_band_percent
    assert blocked_report.confidence.blocking_gaps == 3
    assert blocked_report.confidence.open_questions == 4
    assert blocked_report.confidence.design_gap_count == 1
    assert blocked_report.confidence.implementation_gap_count == 2
    assert blocked_report.confidence.implementation_open_questions == 4
    assert blocked_report.confidence.assumptions_count == 2
    assert any("preguntas residuales" in item.lower() for item in blocked_report.confidence.recommended_next_actions)


def build_snapshot(
    *,
    title: str,
    include_blueprint: bool,
    include_dataset: bool = False,
    include_run: bool = False,
) -> SessionSnapshot:
    snapshot = SessionSnapshot(
        session=SessionCreateResponse(
            id=uuid4(),
            title=title,
            status=ArtifactStatus.ready,
            current_stage=SessionStage.post_validation,
            created_at=utc_now(),
            updated_at=utc_now(),
        ),
        discovery=DiscoveryArtifact(
            problem_statement="Construir un agente de soporte con knowledge retrieval y acciones sobre CRM externo.",
            current_user="Lider de producto",
            current_process="Documenta alcance, contratos API y despliegue de forma parcial.",
            desired_outcome="Tener un paquete tecnico listo para construir con costo comercial temprano.",
            autonomy_level="medium",
            constraints=["No desplegar sin target definido", "Side effects siempre requieren aprobacion"],
            operational_baseline=OperationalBaseline(
                current_time_spent="5 horas",
                current_cost="Retrabajo por gaps de integracion",
                frequent_errors=["Contratos externos incompletos"],
                automation_opportunities=["Discovery estructurado", "ACP incremental"],
            ),
            mvp_definition=MvpDefinition(
                v1_scope=["Canvas", "Blueprint", "Estimacion"],
                out_of_scope=["Go live productivo"],
                north_star_metric="ACP util para construir",
                non_delegable_decisions=["Aprobar side effects y despliegue"],
            ),
            case_type="automatizacion",
            value_statement="Reducir retrabajo antes de construir.",
        ),
        canvas=CanvasArtifact(
            user_goal="Definir un agente util con retrieval y pasos de ejecucion gobernados.",
            mvp_scope=["Discovery", "Blueprint", "ACP"],
            out_of_scope=["Operacion productiva automatica"],
            success_metric="Paquete listo para build",
            primary_risk="Contratos externos, knowledge y deployment no cerrados",
            agent_profile=AgentCanvasProfile(
                mission="Traducir discovery a una propuesta construible.",
                primary_user="Arquitecto de soluciones",
                agent_task="Definir flujo, tools, memoria y artefactos de continuidad.",
                human_approvals=["Side effects", "Handoff a build"],
                success_metrics=["Tiempo total", "Confianza comercial"],
            ),
        ),
    )

    if include_blueprint:
        snapshot.blueprint = BlueprintArtifact(
            architecture="single_agent_with_skills",
            reasoning_pattern="Plan-and-Execute",
            memory_strategy="session_memory_with_retrieval",
            tools=[
                BlueprintTool(
                    name="Consultar knowledge",
                    purpose="Recupera informacion documental del workspace",
                    risk_level="medium",
                    inputs=["query"],
                    outputs=["answer"],
                    validations=["schema"],
                ),
                BlueprintTool(
                    name="Actualizar CRM",
                    purpose="Ejecuta side effects sobre una API externa",
                    risk_level="high",
                    requires_approval=True,
                    has_side_effects=True,
                    inputs=["payload"],
                    outputs=["result"],
                    validations=["schema", "approval"],
                ),
            ],
            memory_profile=MemoryProfile(
                strategy="session_memory_with_retrieval",
                storage_layers=["session_state", "vector_index"],
                write_policy="Persistir solo decisiones validadas",
                retrieval_policy="Priorizar fuentes del workspace",
                review_trigger="Cambio de etapa",
                goal_drift_guard="Comparar contra desired_outcome",
            ),
            safety_checks=[
                SafetyCheck(category="hallucination", risk="Campos inventados", severity="high", mitigation="Schema"),
                SafetyCheck(category="operations", risk="Side effects", severity="high", mitigation="Approval gate"),
            ],
            guardrails=["No inventar contratos API", "No desplegar sin ambiente objetivo"],
            delivery_package=DeliveryPackage(
                workflow_profile=WorkflowProfile(
                    execution_pattern="durable_linear",
                    steps=[
                        WorkflowStep(name="discover", objective="Capturar discovery", actor="agent"),
                        WorkflowStep(name="design", objective="Generar blueprint", actor="agent"),
                        WorkflowStep(name="package", objective="Exportar ACP", actor="agent", requires_approval=True),
                    ],
                ),
                observability_plan=ObservabilityPlan(
                    captured_signals=["inputs", "outputs", "cost", "duration"],
                    cost_tracking="persisted",
                    duration_tracking="persisted",
                ),
                deliverables=[
                    GeneratedDeliverable(key="prd", title="PRD", summary="Discovery consolidado"),
                    GeneratedDeliverable(key="tool_schema", title="Contracts", summary="Contratos y validaciones"),
                    GeneratedDeliverable(key="technical_spec", title="Technical spec", summary="Paquete para construir"),
                ],
            ),
            readiness_state=ReviewState.partial,
            narrative="Blueprint con API externa, retrieval y empaquetado ACP.",
        )

    if include_dataset:
        snapshot.evaluation_dataset = EvaluationDatasetArtifact(
            cases=[
                EvaluationDatasetCase(case_key="core_1", title="Caso core", category="core"),
                EvaluationDatasetCase(case_key="risk_1", title="Caso riesgo", category="risk"),
            ]
        )

    if include_run:
        snapshot.evaluation_runs = [
            EvaluationRunEntry(
                id=uuid4(),
                created_at=utc_now(),
                status=ArtifactStatus.ready,
                overall_score=89,
                summary="Corrida estable",
            )
        ]

    return snapshot


def build_partial_acp_preview(snapshot: SessionSnapshot) -> ACPPreview:
    return ACPPreview(
        session_id=snapshot.session.id,
        blueprint_version_number=1,
        construction_readiness=ConstructionReadinessReport(
            overall_status="blocked",
            can_start_build=False,
            blocking_gaps=3,
            open_questions=4,
            assumptions_count=2,
            gaps=[
                ConstructionGapEntry(
                    gap_key="external_api_contracts_missing",
                    title="Contratos API abiertos",
                    domain="integrations",
                    severity="blocking",
                    status="open",
                    summary="Faltan payloads definitivos y sandbox.",
                ),
                ConstructionGapEntry(
                    gap_key="deployment_target_unknown",
                    title="Deployment sin target",
                    domain="deployment",
                    severity="blocking",
                    status="open",
                    summary="No esta definido el entorno final.",
                ),
                ConstructionGapEntry(
                    gap_key="knowledge_sources_missing",
                    title="Knowledge incompleto",
                    domain="knowledge",
                    severity="blocking",
                    status="open",
                    summary="Faltan owner, fuentes y refresh policy.",
                ),
            ],
        ),
    )


def build_ready_acp_preview(snapshot: SessionSnapshot) -> ACPPreview:
    return ACPPreview(
        session_id=snapshot.session.id,
        blueprint_version_number=1,
        construction_readiness=ConstructionReadinessReport(
            overall_status="ready_to_build",
            can_start_build=True,
            blocking_gaps=0,
            open_questions=0,
            assumptions_count=0,
            gaps=[],
        ),
    )
