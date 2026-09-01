from uuid import uuid4

from app.models import (
    ACPFileEntry,
    ACPValidationIssue,
    ACPValidationReport,
    ArtifactStatus,
    BlueprintArtifact,
    BlueprintConsistencyIssue,
    BlueprintConsistencyReport,
    BlueprintTool,
    MemoryProfile,
    ReviewState,
    SessionCreateResponse,
    SessionSnapshot,
    SessionStage,
    utc_now,
)
from app.services.acp_construction_readiness import build_initial_construction_readiness
from app.services.acp_paths import ACP_CANONICAL_ENV_TEMPLATE_PATH, build_tool_contract_path


def build_snapshot(
    *,
    memory_strategy: str = "session_memory_with_checkpoints",
    storage_layers: list[str] | None = None,
    tools: list[BlueprintTool] | None = None,
) -> SessionSnapshot:
    now = utc_now()
    resolved_storage_layers = storage_layers or ["session_state"]
    return SessionSnapshot(
        session=SessionCreateResponse(
            id=uuid4(),
            title="Lean Agent Builder",
            status=ArtifactStatus.ready,
            current_stage=SessionStage.ready_for_export,
            created_at=now,
            updated_at=now,
        ),
        blueprint=BlueprintArtifact(
            architecture="single_agent_with_skills",
            reasoning_pattern="Plan-and-Execute",
            memory_strategy=memory_strategy,
            tools=tools or [BlueprintTool(name="build_blueprint")],
            memory_profile=MemoryProfile(
                strategy=memory_strategy,
                storage_layers=resolved_storage_layers,
                write_policy="Persist validated state",
                retrieval_policy="Recover by session_id",
                review_trigger="Missing construction fields",
                goal_drift_guard="Stay aligned with the canvas",
            ),
            guardrails=["No inventar datos"],
            narrative="ACP listo para validar continuidad constructiva.",
        ),
    )


def build_file(
    path: str,
    domain: str,
    *,
    format: str = "yaml",
    status: str = "complete",
    content_text: str = "resolved: true",
    warnings: list[str] | None = None,
    source_sections: list[str] | None = None,
) -> ACPFileEntry:
    return ACPFileEntry(
        path=path,
        domain=domain,
        title=path.split("/")[-1],
        format=format,
        status=status,
        source_sections=source_sections or [domain],
        warnings=warnings or [],
        content_text=content_text,
    )


def build_complete_acp_files() -> list[ACPFileEntry]:
    return [
        build_file("ACP/knowledge/sources.yaml", "knowledge"),
        build_file("ACP/knowledge/ingestion.yaml", "knowledge"),
        build_file("ACP/knowledge/embeddings.yaml", "knowledge"),
        build_file("ACP/runtime/models.yaml", "runtime"),
        build_file("ACP/runtime/providers.yaml", "runtime"),
        build_file("ACP/runtime/config.yaml", "runtime"),
        build_file(ACP_CANONICAL_ENV_TEMPLATE_PATH, "deployment", content_text="OPENAI_API_KEY=\nDATABASE_URL=\n"),
        build_file("ACP/deployment/docker-compose.yaml", "deployment"),
        build_file("ACP/deployment/kubernetes/README.md", "deployment", format="markdown"),
        build_file("ACP/deployment/cicd/README.md", "deployment", format="markdown"),
    ]


def build_valid_validation_report() -> ACPValidationReport:
    return ACPValidationReport(
        overall_status="complete",
        completeness_percent=100,
        can_export_zip=True,
        issues=[],
    )


def test_build_initial_construction_readiness_returns_not_started_without_files() -> None:
    readiness = build_initial_construction_readiness(
        build_snapshot(),
        [],
        build_valid_validation_report(),
    )

    assert readiness.overall_status == "not_started"
    assert readiness.can_start_build is False
    assert readiness.blocking_gaps == 0
    assert readiness.open_questions == 0
    assert readiness.next_recommended_action == "generate_acp_preview"


def test_build_initial_construction_readiness_returns_ready_to_build_when_contracts_are_closed() -> None:
    readiness = build_initial_construction_readiness(
        build_snapshot(),
        build_complete_acp_files(),
        build_valid_validation_report(),
    )

    assert readiness.overall_status == "ready_to_build"
    assert readiness.can_start_build is True
    assert readiness.blocking_gaps == 0
    assert readiness.open_questions == 0
    assert readiness.gaps == []
    assert readiness.next_recommended_action == "start_agentic_build"


def test_blueprint_handoff_process_debt_does_not_become_acp_blocker() -> None:
    snapshot = build_snapshot().model_copy(
        update={
            "blueprint_consistency": BlueprintConsistencyReport(
                overall_status=ReviewState.blocked,
                summary="Stale operativo del Blueprint antes de aprobar ACP.",
                issues=[
                    BlueprintConsistencyIssue(
                        issue_key="tools_recommendation_stale",
                        severity="blocking",
                        category="design_to_tools",
                        title="Tools stale",
                        detail="La recomendacion de herramientas esta desactualizada.",
                        affected_stage_keys=["tools"],
                    ),
                    BlueprintConsistencyIssue(
                        issue_key="validate_source_stage_drift:memory",
                        severity="blocking",
                        category="memory_to_validate",
                        title="Validate stale",
                        detail="Validate referencia una version previa de Memory.",
                        affected_stage_keys=["memory", "validate"],
                    ),
                ],
                blocking_issues=[
                    "La recomendacion de herramientas esta desactualizada.",
                    "Validate referencia una version previa de Memory.",
                ],
            )
        }
    )

    readiness = build_initial_construction_readiness(
        snapshot,
        build_complete_acp_files(),
        build_valid_validation_report(),
    )

    assert readiness.overall_status == "ready_to_build"
    assert readiness.blocking_gaps == 0
    assert not any(item.gap_key == "cross_stage_consistency_drift" for item in readiness.gaps)


def test_build_initial_construction_readiness_requires_answers_when_only_warning_gaps_remain() -> None:
    files = build_complete_acp_files()
    files[0] = build_file(
        "ACP/knowledge/sources.yaml",
        "knowledge",
        status="needs_review",
        content_text="needs_review: true",
    )
    files[5] = build_file(
        "ACP/runtime/config.yaml",
        "runtime",
        status="needs_review",
        content_text="pendiente: secret_source",
    )

    readiness = build_initial_construction_readiness(
        build_snapshot(memory_strategy="session_memory_with_checkpoints", storage_layers=["session_state"]),
        files,
        build_valid_validation_report(),
    )

    assert readiness.overall_status == "needs_questions"
    assert readiness.can_start_build is False
    assert readiness.blocking_gaps == 0
    assert readiness.open_questions == 6
    assert readiness.next_recommended_action == "answer_open_questions"
    assert {item.gap_key for item in readiness.gaps} == {
        "knowledge_sources_missing",
        "runtime_contract_incomplete",
    }


def test_build_initial_construction_readiness_blocks_on_validation_deployment_runtime_and_external_contracts() -> None:
    external_tool = BlueprintTool(name="consult_ticket")
    files = build_complete_acp_files()
    files[4] = build_file(
        "ACP/runtime/providers.yaml",
        "runtime",
        status="needs_review",
        content_text="needs_review: vector_store_provider",
    )
    files[7] = build_file(
        "ACP/deployment/docker-compose.yaml",
        "deployment",
        status="needs_review",
        content_text="pendiente: deployment_target",
    )
    validation = ACPValidationReport(
        overall_status="needs_review",
        completeness_percent=92,
        can_export_zip=True,
        issues=[
            ACPValidationIssue(
                code="tool_missing_outputs",
                severity="error",
                path="ACP/tools/contracts/tool-build-blueprint.yaml",
                message="Tool outputs are required.",
                remediation="Documentar outputs tipados y volver a regenerar el contrato.",
                source_sections=["blueprint.tools"],
                blocking=True,
            )
        ],
    )

    readiness = build_initial_construction_readiness(
        build_snapshot(
            memory_strategy="persistent_memory",
            storage_layers=["session_state", "vector_store"],
            tools=[BlueprintTool(name="build_blueprint"), external_tool],
        ),
        files,
        validation,
    )

    gap_map = {item.gap_key: item for item in readiness.gaps}

    assert readiness.overall_status == "blocked"
    assert readiness.can_start_build is False
    assert readiness.blocking_gaps == 1
    assert readiness.open_questions == 7
    assert readiness.next_recommended_action == "resolve_blocking_construction_gaps"
    assert set(gap_map) == {
        "acp_package_validation_blocked",
        "runtime_contract_incomplete",
        "deployment_target_unknown",
        "external_api_contracts_missing",
    }
    assert gap_map["acp_package_validation_blocked"].remediation
    assert gap_map["runtime_contract_incomplete"].severity == "warning"
    assert gap_map["runtime_contract_incomplete"].remediation
    assert gap_map["deployment_target_unknown"].severity == "warning"
    assert gap_map["deployment_target_unknown"].remediation
    assert gap_map["external_api_contracts_missing"].evidence_paths == [
        build_tool_contract_path("consult_ticket", 2)
    ]
    assert gap_map["external_api_contracts_missing"].remediation


def test_acp_questions_are_non_blocking_and_rich_in_options() -> None:
    external_tool = BlueprintTool(name="consult_ticket")
    files = build_complete_acp_files()
    files[0] = build_file("ACP/knowledge/sources.yaml", "knowledge", status="needs_review", content_text="needs_review: true")
    files[4] = build_file("ACP/runtime/providers.yaml", "runtime", status="needs_review", content_text="needs_review: true")
    files[7] = build_file("ACP/deployment/docker-compose.yaml", "deployment", status="needs_review", content_text="needs_review: true")

    readiness = build_initial_construction_readiness(
        build_snapshot(tools=[BlueprintTool(name="build_blueprint"), external_tool]),
        files,
        build_valid_validation_report(),
    )

    all_questions = [q for gap in readiness.gaps for q in gap.questions]
    assert len(all_questions) > 0
    for q in all_questions:
        assert q.blocking is False
        assert q.purpose != ""
        assert len(q.options) >= 2
        for opt in q.options:
            assert opt.key != ""
            assert opt.label != ""
            assert opt.description != ""
            assert opt.impact != ""
            assert opt.example != ""
