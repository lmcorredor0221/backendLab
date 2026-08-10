from io import BytesIO
import json
from uuid import UUID, uuid4
from zipfile import ZipFile

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.models import (
    ArtifactRegistryRecord,
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
    SessionRecord,
    SessionSnapshot,
    SessionStage,
    UserRecord,
    utc_now,
)
from app.services.acp_generator import generate_acp_preview
from app.services.acp_export_profiles import apply_acp_export_profile
from app.services.acp_zip_export import build_acp_zip
from app.services.builder_service import enrich_blueprint
from app.services.estimation_service import build_estimation_report
from app.services.evaluation_workbench import build_default_evaluation_dataset, build_default_evaluation_rubric
from app.services.operations_service import record_acp_preview_artifacts
from app.services.workspace_bootstrap import apply_workspace_bootstrap


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


def build_ready_snapshot(*, session_id: UUID | None = None) -> SessionSnapshot:
    discovery = sample_discovery()
    canvas = sample_canvas()
    enriched = enrich_blueprint(sample_blueprint(), discovery, canvas).data
    dataset: EvaluationDatasetArtifact = build_default_evaluation_dataset(
        discovery,
        canvas,
        enriched,
        blueprint_version_number=3,
    )
    rubric: EvaluationRubricArtifact = build_default_evaluation_rubric(blueprint_version_number=3)
    now = utc_now()
    snapshot_session_id = uuid4() if session_id is None else session_id
    return SessionSnapshot(
        session=SessionCreateResponse(
            id=snapshot_session_id,
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
                detail="provider=openai mode=responses fast=gpt-5.4-mini reasoning=gpt-5.5",
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
            IntegrationStatusEntry(
                id=uuid4(),
                integration_key="local_auth",
                label="Auth local",
                status="healthy",
                configured=True,
                reachable=True,
                detail="active_users=1",
                checked_at=now,
            ),
        ],
    )


def test_generate_acp_preview_builds_cross_domain_files() -> None:
    snapshot = build_ready_snapshot()

    preview = generate_acp_preview(snapshot)

    paths = [item.path for item in preview.files]
    assert paths == sorted(paths)
    assert "ACP/manifest.yaml" in paths
    assert "ACP/tools/external/tool-build-blueprint.yaml" in paths
    assert "ACP/memory/strategy.yaml" in paths
    assert "ACP/evaluation/rubrics.yaml" in paths
    assert "ACP/observability/alerts.yaml" in paths
    assert "ACP/deployment/env.template" in paths
    assert "ACP/runtime/env.template" not in paths
    assert "ACP/construction-readiness/overview.yaml" in paths
    assert "ACP/construction-readiness/blocking-gaps.yaml" in paths
    assert "ACP/construction-readiness/open-questions.yaml" in paths
    assert "ACP/construction-readiness/assumptions.yaml" in paths
    assert "ACP/construction-readiness/external-dependencies.yaml" in paths
    assert "ACP/construction-readiness/required-api-contracts.yaml" in paths
    assert "ACP/construction-readiness/deployment-decisions-needed.yaml" in paths
    assert "ACP/construction-readiness/resolution-workflow.yaml" in paths
    assert "ACP/prompts/builder-handoff.md" in paths
    assert "ACP/prompts/gap-closure.md" in paths
    assert "ACP/diagrams/Architecture.md" in paths
    assert "ACP/diagrams/KnowledgeGraph.md" in paths
    assert "ACP/mermaid/Architecture.mmd" in paths
    assert "ACP/plantuml/Architecture.puml" in paths
    assert "ACP/d2/Architecture.d2" in paths
    assert "ACP/svg/Architecture.svg" in paths
    assert "ACP/png/README.md" in paths
    assert "ACP/blueprint.graph.json" in paths
    assert "ACP/blueprint.graphml" in paths
    assert "ACP/blueprint.cypher" in paths
    assert "ACP/blueprint.manifest.json" in paths
    assert "ACP/conformance/file-index.json" in paths
    assert "ACP/conformance/checksums.sha256" in paths
    assert "ACP/conformance/portability-report.json" in paths
    assert "ACP/conformance/portability-report.md" in paths
    manifest = next(item for item in preview.files if item.path == "ACP/manifest.yaml")
    assert "generated_by: Lean Agent Builder" in manifest.content_text
    tool_contract = next(item for item in preview.files if item.path == "ACP/tools/external/tool-build-blueprint.yaml")
    assert "name: build_blueprint" in tool_contract.content_text
    readiness_overview = next(
        item for item in preview.files if item.path == "ACP/construction-readiness/overview.yaml"
    )
    assert "construction_readiness:" in readiness_overview.content_text
    assert "next_recommended_action: answer_open_questions" in readiness_overview.content_text
    architecture_diagram = next(item for item in preview.files if item.path == "ACP/diagrams/Architecture.md")
    assert "## Mermaid" in architecture_diagram.content_text
    architecture_svg = next(item for item in preview.files if item.path == "ACP/svg/Architecture.svg")
    assert "<svg" in architecture_svg.content_text
    assert "Architecture" in architecture_svg.content_text
    graph_export = next(item for item in preview.files if item.path == "ACP/blueprint.graph.json")
    assert '"graph_version": "blueprint-graph.v1"' in graph_export.content_text
    visualization_manifest = next(item for item in preview.files if item.path == "ACP/blueprint.manifest.json")
    assert '"svg"' in visualization_manifest.content_text
    assert '"pending": [' in visualization_manifest.content_text
    assert '"png"' in visualization_manifest.content_text
    builder_handoff = next(item for item in preview.files if item.path == "ACP/prompts/builder-handoff.md")
    assert "Lee primero `ACP/construction-readiness/overview.yaml`." in builder_handoff.content_text
    assert "`ACP/blueprint.graph.json`" in builder_handoff.content_text
    assert preview.validation.can_export_zip is True
    assert preview.validation.overall_status == "needs_review"
    assert preview.validation.completeness_percent > 0
    assert preview.construction_readiness.overall_status == "needs_questions"
    assert preview.construction_readiness.can_start_build is False
    assert preview.construction_readiness.blocking_gaps == 0
    assert preview.construction_readiness.open_questions >= 1


def test_generate_acp_preview_produces_deterministic_construction_readiness() -> None:
    snapshot = build_ready_snapshot()

    first_preview = generate_acp_preview(snapshot)
    second_preview = generate_acp_preview(snapshot)

    assert first_preview.construction_readiness.model_dump(mode="json") == second_preview.construction_readiness.model_dump(
        mode="json"
    )
    assert [item.gap_key for item in first_preview.construction_readiness.gaps] == [
        item.gap_key for item in second_preview.construction_readiness.gaps
    ]
    assert any(item.gap_key == "deployment_target_unknown" for item in first_preview.construction_readiness.gaps)


def test_build_acp_zip_contains_construction_readiness_block() -> None:
    snapshot = build_ready_snapshot()

    preview = generate_acp_preview(snapshot)
    zip_bytes = build_acp_zip(preview)

    with ZipFile(BytesIO(zip_bytes)) as archive:
        names = sorted(archive.namelist())

    assert "ACP/construction-readiness/overview.yaml" in names
    assert "ACP/construction-readiness/blocking-gaps.yaml" in names
    assert "ACP/prompts/builder-handoff.md" in names
    assert "ACP/diagrams/KnowledgeGraph.md" in names
    assert "ACP/svg/KnowledgeGraph.svg" in names
    assert "ACP/blueprint.graph.json" in names
    assert names.count("ACP/deployment/env.template") == 1


def test_design_only_profile_filters_deployment_and_observability_from_preview_and_zip() -> None:
    snapshot = build_ready_snapshot()

    preview = generate_acp_preview(snapshot)
    profiled_preview = apply_acp_export_profile(preview, "design-only")
    zip_bytes = build_acp_zip(profiled_preview)

    paths = [item.path for item in profiled_preview.files]
    assert all(not path.startswith("ACP/deployment/") for path in paths)
    assert all(not path.startswith("ACP/observability/") for path in paths)
    assert "ACP/runtime/config.yaml" in paths
    assert "ACP/launcher/acp-launcher.py" in paths
    assert "ACP/adapters/adapter-registry.json" in paths
    assert "ACP/conformance/portability-report.json" in paths

    with ZipFile(BytesIO(zip_bytes)) as archive:
        names = sorted(archive.namelist())

    assert all(not name.startswith("ACP/deployment/") for name in names)
    assert all(not name.startswith("ACP/observability/") for name in names)
    assert "ACP/runtime/providers.yaml" in names


def test_commercial_export_profiles_split_blueprint_portable_and_full() -> None:
    snapshot = build_ready_snapshot()

    preview = generate_acp_preview(snapshot)
    blueprint_profile = apply_acp_export_profile(preview, "blueprint-professional")
    portable_profile = apply_acp_export_profile(preview, "acp-portable")
    full_profile = apply_acp_export_profile(preview, "acp-full")
    design_only_alias = apply_acp_export_profile(preview, "design-only")
    extended_alias = apply_acp_export_profile(preview, "extended")

    blueprint_paths = {item.path for item in blueprint_profile.files}
    portable_paths = {item.path for item in portable_profile.files}
    full_paths = {item.path for item in full_profile.files}

    assert "ACP/architecture/topology.yaml" in blueprint_paths
    assert "ACP/memory/strategy.yaml" in blueprint_paths
    assert "ACP/tools/permissions.yaml" in blueprint_paths
    assert "ACP/conformance/portability-report.json" in blueprint_paths
    assert all(not path.startswith("ACP/launcher/") for path in blueprint_paths)
    assert all(not path.startswith("ACP/adapters/") for path in blueprint_paths)
    assert all(not path.startswith("ACP/construction-readiness/") for path in blueprint_paths)
    assert all(not path.startswith("ACP/prompts/") for path in blueprint_paths)
    assert all(not path.startswith("ACP/runtime/") for path in blueprint_paths)
    assert all(not path.startswith("ACP/evaluation/") for path in blueprint_paths)

    assert "ACP/launcher/acp-launcher.py" in portable_paths
    assert "ACP/adapters/adapter-registry.json" in portable_paths
    assert "ACP/construction-readiness/overview.yaml" in portable_paths
    assert "ACP/evaluation/benchmarks.yaml" in portable_paths
    assert all(not path.startswith("ACP/deployment/") for path in portable_paths)
    assert all(not path.startswith("ACP/observability/") for path in portable_paths)

    assert "ACP/deployment/env.template" in full_paths
    assert "ACP/observability/telemetry.yaml" in full_paths
    assert {item.path for item in design_only_alias.files} == portable_paths
    assert {item.path for item in extended_alias.files} == full_paths

    blueprint_report = json.loads(
        next(item for item in blueprint_profile.files if item.path == "ACP/conformance/portability-report.json").content_text
    )
    portable_report = json.loads(
        next(item for item in portable_profile.files if item.path == "ACP/conformance/portability-report.json").content_text
    )
    full_report = json.loads(
        next(item for item in full_profile.files if item.path == "ACP/conformance/portability-report.json").content_text
    )
    assert blueprint_report["profile"] == "blueprint-professional"
    assert portable_report["profile"] == "acp-portable"
    assert full_report["profile"] == "acp-full"
    assert blueprint_report["requires_lean_backend"] is False
    assert portable_report["reference_integrity"]["broken_references"] == []
    assert full_report["signals"]["evaluation"]["present"] is True


def test_generate_acp_preview_includes_estimation_package_when_report_exists() -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    snapshot = build_ready_snapshot()

    with Session(engine) as session:
        ws_id = uuid4()
        apply_workspace_bootstrap(session, ws_id)
        base_preview = generate_acp_preview(snapshot)
        snapshot.estimation_report = build_estimation_report(session, snapshot=snapshot, acp_preview=base_preview)

    preview = generate_acp_preview(snapshot)
    paths = [item.path for item in preview.files]

    assert "ACP/estimation/estimation-report.json" in paths
    assert "ACP/estimation/estimation-report.md" in paths
    assert "ACP/estimation/assumptions.yaml" in paths
    assert "ACP/estimation/sensitivity-drivers.yaml" in paths

    estimation_json = next(item for item in preview.files if item.path == "ACP/estimation/estimation-report.json")
    assert '"contract_version": "estimation-report.v1"' in estimation_json.content_text
    estimation_md = next(item for item in preview.files if item.path == "ACP/estimation/estimation-report.md")
    assert "# Estimacion comparativa" in estimation_md.content_text
    sensitivity = next(item for item in preview.files if item.path == "ACP/estimation/sensitivity-drivers.yaml")
    assert "workstream_deltas:" in sensitivity.content_text
    portability_report = next(item for item in preview.files if item.path == "ACP/conformance/portability-report.json")
    assert json.loads(portability_report.content_text)["signals"]["estimation"]["present"] is True


def test_record_acp_preview_artifacts_persists_preview_and_files() -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    now = utc_now()
    user_id = uuid4()
    session_id = uuid4()

    snapshot = build_ready_snapshot(session_id=session_id)
    preview = generate_acp_preview(snapshot)

    with Session(engine) as session:
        ws_id = uuid4()
        ws = apply_workspace_bootstrap(session, ws_id)
        session.add(
            UserRecord(
                id=user_id,
                email="admin@leanbuilder.local",
                full_name="Lean Builder Test",
                password_hash="hash",
                is_active=True,
                created_at=now,
                updated_at=now,
            )
        )
        session.add(
            SessionRecord(
                id=session_id,
                user_id=user_id,
                workspace_id=ws_id,
                title="Customer Support Agent",
                status=ArtifactStatus.ready,
                current_stage=SessionStage.ready_for_export,
                created_at=now,
                updated_at=now,
            )
        )
        session.commit()

        records = record_acp_preview_artifacts(
            session,
            session_id=session_id,
            preview=preview,
            source_action="generate_acp_preview",
        )
        session.commit()

        persisted = session.exec(
            select(ArtifactRegistryRecord).where(ArtifactRegistryRecord.session_id == session_id)
        ).all()

    assert len(records) == len(preview.files) + 1
    assert len(persisted) == len(preview.files) + 1
    assert any(item.artifact_kind == "acp_manifest" for item in persisted)
    assert any(item.artifact_kind == "acp_file" for item in persisted)
    preview_record = next(item for item in persisted if item.artifact_kind == "acp_preview")
    assert preview_record.export_format == "json"
    assert preview_record.artifact_metadata["completeness_percent"] == preview.validation.completeness_percent
    assert preview_record.artifact_metadata["construction_readiness_status"] == "needs_questions"
    assert preview_record.artifact_metadata["can_start_build"] is False
    assert preview_record.artifact_metadata["blocking_gaps"] == 0
    assert preview_record.artifact_metadata["open_questions"] >= 1
    manifest_record = next(item for item in persisted if item.artifact_kind == "acp_manifest")
    assert manifest_record.artifact_metadata["acp_path"] == "ACP/manifest.yaml"
    assert manifest_record.content_hash
    continuity_record = next(
        item
        for item in persisted
        if item.artifact_metadata["acp_path"] == "ACP/construction-readiness/overview.yaml"
    )
    assert continuity_record.artifact_metadata["acp_domain"] == "construction-readiness"
    assert continuity_record.artifact_metadata["is_construction_readiness_artifact"] is True
    assert continuity_record.artifact_metadata["lineage_scope"] == "construction_readiness"
    assert continuity_record.artifact_metadata["related_gap_keys"]
    prompt_record = next(
        item for item in persisted if item.artifact_metadata["acp_path"] == "ACP/prompts/builder-handoff.md"
    )
    assert prompt_record.artifact_metadata["acp_domain"] == "prompts"
    assert prompt_record.artifact_metadata["is_prompt_artifact"] is True
    assert "deployment_target_unknown" in prompt_record.artifact_metadata["related_gap_keys"]
    diagram_record = next(item for item in persisted if item.artifact_metadata["acp_path"] == "ACP/diagrams/Architecture.md")
    assert diagram_record.artifact_metadata["acp_domain"] == "diagrams"
