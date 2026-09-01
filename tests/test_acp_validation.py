from uuid import uuid4

from app.models import (
    ACPFileEntry,
    ArtifactStatus,
    BlueprintArtifact,
    BlueprintTool,
    BlueprintVersionEntry,
    CanvasArtifact,
    DiscoveryArtifact,
    EvaluationDatasetArtifact,
    EvaluationRubricArtifact,
    IntegrationStatusEntry,
    MemoryProfile,
    ReviewState,
    SafetyCheck,
    SessionCreateResponse,
    SessionSnapshot,
    SessionStage,
    utc_now,
)
from app.services.acp_validation import (
    build_acp_file_entry,
    build_acp_preview,
    build_acp_validation_report,
    derive_acp_export_status,
)
from app.services.builder_service import enrich_blueprint
from app.services.evaluation_workbench import build_default_evaluation_dataset, build_default_evaluation_rubric


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


def build_ready_snapshot() -> SessionSnapshot:
    discovery = sample_discovery()
    canvas = sample_canvas()
    enriched = enrich_blueprint(sample_blueprint(), discovery, canvas).data.model_copy(
        update={"readiness_state": ReviewState.complete}
    )
    dataset: EvaluationDatasetArtifact = build_default_evaluation_dataset(
        discovery,
        canvas,
        enriched,
        blueprint_version_number=3,
    )
    rubric: EvaluationRubricArtifact = build_default_evaluation_rubric(blueprint_version_number=3)
    now = utc_now()
    return SessionSnapshot(
        session=SessionCreateResponse(
            id=uuid4(),
            title="Customer Support Agent",
            status=ArtifactStatus.ready,
            current_stage=SessionStage.ready_for_export,
            created_at=now,
            updated_at=now,
        ),
        discovery=discovery,
        canvas=canvas,
        blueprint=enriched,
        evaluation_dataset=dataset,
        evaluation_rubric=rubric,
        blueprint_versions=[
            BlueprintVersionEntry(
                version_number=3,
                source_action="enrich_blueprint",
                status=ArtifactStatus.ready,
                readiness_state=ReviewState.complete,
                architecture=enriched.architecture,
                reasoning_pattern=enriched.reasoning_pattern,
                created_at=now,
            )
        ],
        integration_statuses=[
            IntegrationStatusEntry(
                id=uuid4(),
                integration_key="openai",
                label="OpenAI",
                status="healthy",
                configured=True,
                reachable=True,
                detail="provider=openai",
                checked_at=now,
            ),
            IntegrationStatusEntry(
                id=uuid4(),
                integration_key="postgresql",
                label="PostgreSQL",
                status="healthy",
                configured=True,
                reachable=True,
                detail="local database ready",
                checked_at=now,
            ),
        ],
    )


def test_validation_report_blocks_missing_core_artifacts() -> None:
    now = utc_now()
    snapshot = SessionSnapshot(
        session=SessionCreateResponse(
            id=uuid4(),
            title="",
            status=ArtifactStatus.draft,
            current_stage=SessionStage.draft_capture,
            created_at=now,
            updated_at=now,
        )
    )

    report = build_acp_validation_report(snapshot)

    assert report.can_export_zip is False
    assert report.overall_status == "incomplete"
    codes = {item.code for item in report.issues}
    assert {"missing_agent_name", "missing_discovery", "missing_blueprint", "missing_evaluation_base"}.issubset(codes)
    assert all(item.remediation for item in report.issues)
    assert derive_acp_export_status(report) == ArtifactStatus.failed


def test_validation_report_allows_export_for_consistent_snapshot() -> None:
    snapshot = build_ready_snapshot()

    report = build_acp_validation_report(snapshot)

    assert report.can_export_zip is True
    assert report.overall_status == "complete"
    assert report.completeness_percent == 100
    assert not any(item.blocking for item in report.issues)
    assert derive_acp_export_status(report) == ArtifactStatus.ready


def test_validation_report_tool_issues_use_generated_contract_paths() -> None:
    snapshot = build_ready_snapshot()
    blueprint = snapshot.blueprint
    assert blueprint is not None

    broken_tool = blueprint.tools[0].model_copy(update={"outputs": []})
    snapshot = snapshot.model_copy(
        update={
            "blueprint": blueprint.model_copy(update={"tools": [broken_tool]}),
        }
    )

    report = build_acp_validation_report(snapshot)

    issue = next(item for item in report.issues if item.code == "tool_missing_outputs")
    assert issue.path == "ACP/tools/external/tool-build-blueprint.yaml"
    assert issue.severity == "warning"
    assert issue.blocking is False
    assert issue.remediation
    assert report.can_export_zip is True


def test_acp_preview_uses_file_statuses_for_completeness_and_hashes() -> None:
    snapshot = build_ready_snapshot()
    files: list[ACPFileEntry] = [
        build_acp_file_entry(
            path="ACP/manifest.yaml",
            domain="manifest",
            title="Manifest",
            format="yaml",
            source_sections=["session.title", "discovery", "blueprint"],
            content_text="metadata:\r\n  name: Customer Support Agent\r\n",
        ),
        build_acp_file_entry(
            path="ACP/deployment/kubernetes/README.md",
            domain="deployment",
            title="Kubernetes placeholder",
            format="markdown",
            source_sections=["integration_statuses"],
            content_text="# Kubernetes\nPendiente\r\n",
            warnings=["Falta configuracion cloud-agnostic detallada."],
        ),
    ]

    preview = build_acp_preview(snapshot, files)

    assert preview.blueprint_version_number == 3
    assert preview.validation.can_export_zip is True
    assert preview.validation.overall_status == "needs_review"
    assert preview.validation.completeness_percent == 50
    assert preview.files[0].status == "complete"
    assert preview.files[0].content_hash
    assert preview.files[1].status == "needs_review"
    assert preview.construction_readiness.overall_status == "needs_questions"
    assert preview.construction_readiness.can_start_build is False
    assert preview.construction_readiness.blocking_gaps == 0
    assert preview.construction_readiness.open_questions >= 1
    assert preview.construction_readiness.next_recommended_action == "answer_open_questions"
