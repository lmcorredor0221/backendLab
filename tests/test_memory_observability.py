from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.models import (
    ArtifactStatus,
    LLMContextTrace,
    SessionStage,
    ShortTermMemoryRuntimeState,
    ShortTermMemoryState,
    SkillRunEntry,
    utc_now,
)
from app.services.memory_observability import build_memory_observability_report
from app.services.memory_traceability import build_repo_document_lineage


def _skill_run(
    *,
    source: dict,
    source_action: str = "memory_observability_test",
    stage: SessionStage = SessionStage.build_blueprint,
    skill_key: str = "blueprint_generation_skill",
) -> SkillRunEntry:
    return SkillRunEntry(
        id=uuid4(),
        skill_key=skill_key,
        label="Blueprint generation",
        stage=stage,
        source_action=source_action,
        status=ArtifactStatus.ready,
        duration_ms=1800,
        result_summary="ok",
        llm_trace=LLMContextTrace(
            provider_key="codex_local",
            execution_backend="shadow_codex_cli",
            execution_mode="shadow",
            effective_context_backend="workspace_staged_filesystem",
            context_used_sources=[source],
            context_stats={
                "budget_tokens": 1000,
                "assembled_estimated_tokens": 240,
                "reduction_estimated_tokens": 760,
                "baseline_estimated_tokens": 1000,
            },
        ),
        created_at=utc_now(),
    )


def _short_term_memory(*, rollback_available: bool = True) -> ShortTermMemoryRuntimeState:
    return ShortTermMemoryRuntimeState(
        session_id=uuid4(),
        source_action="memory_observability_test",
        active_branch_key="main",
        active_checkpoint_key="main:cp2",
        last_consistent_checkpoint_key="main:cp1",
        rollback_available=rollback_available,
        branch_count=1,
        checkpoint_count=2,
        memory=ShortTermMemoryState(
            active_stage=SessionStage.build_blueprint.value,
            active_goal="cerrar blueprint",
            current_focus="validar memoria",
        ),
        updated_at=utc_now(),
    )


def test_memory_observability_report_flags_stale_repo_sources(tmp_path: Path) -> None:
    docs_path = tmp_path / "Docs" / "knowledge.md"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text("# Knowledge\n\ncontenido vigente", encoding="utf-8")

    stale_source = {
        "key": "reingenieria_core_canonical",
        "title": "Core canonical",
        "source_type": "repo_rule",
        "uri": "repo://Docs/knowledge.md",
        "authority_level": "canonical",
        "required": True,
        "relative_path": "knowledge/required/01-core.md",
        "source_refs": ["Docs/knowledge.md"],
        "source_lineage": ["Docs/knowledge.md::doc::0000000000000000"],
    }

    report = build_memory_observability_report(
        skill_runs=[_skill_run(source=stale_source)],
        short_term_memory=_short_term_memory(),
        repo_root=tmp_path,
    )

    metrics = {item.key: item for item in report.metrics}
    validations = {item.check_key: item for item in report.validations}

    assert report.llm_run_count == 1
    assert report.grounded_hit_runs == 1
    assert report.stale_source_count == 1
    assert metrics["hit_rate"].value == 100.0
    assert metrics["citation_coverage"].value == 100.0
    assert metrics["stale_rate"].value == 100.0
    assert metrics["recoverability"].value == 100.0
    assert validations["stale_source_invalidation"].status == "fail"
    assert validations["short_term_recoverability"].status == "pass"


def test_memory_observability_report_catches_contaminated_sources_and_preserves_long_context_recovery(tmp_path: Path) -> None:
    docs_path = tmp_path / "Docs" / "strategy.md"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text("# Strategy\n\nneedle evidence preserved after compression", encoding="utf-8")
    current_lineage = build_repo_document_lineage(tmp_path, "Docs/strategy.md")

    grounded_source = {
        "key": "system_analysis_operational",
        "title": "System analysis",
        "source_type": "repo_rule",
        "uri": "repo://Docs/strategy.md",
        "authority_level": "operational",
        "required": True,
        "relative_path": "knowledge/candidate/01-strategy.md",
        "source_refs": ["Docs/strategy.md"],
        "source_lineage": [current_lineage],
    }
    contaminated_source = {
        "key": "external_shadow_hint",
        "title": "Shadow hint",
        "source_type": "inline_artifact",
        "uri": "inline://shadow-hint",
        "authority_level": "untrusted",
        "required": False,
        "relative_path": "knowledge/candidate/02-shadow.md",
        "source_refs": [],
        "source_lineage": [],
    }

    report = build_memory_observability_report(
        skill_runs=[
            _skill_run(source=grounded_source, skill_key="planner_skill"),
            _skill_run(source=contaminated_source, skill_key="planner_skill"),
        ],
        short_term_memory=_short_term_memory(rollback_available=False),
        repo_root=tmp_path,
    )

    validations = {item.check_key: item for item in report.validations}
    by_agent = {item.scope_key: item for item in report.by_agent}
    by_stage = {item.scope_key: item for item in report.by_stage}

    assert report.llm_run_count == 2
    assert validations["needle_in_the_haystack_recovery"].status == "pass"
    assert validations["long_context_recovery"].status == "pass"
    assert validations["contaminated_memory_guard"].status == "fail"
    assert by_agent["planner_skill"].llm_runs == 2
    assert by_stage[SessionStage.build_blueprint.value].average_compression_gain >= 70


def test_memory_observability_report_includes_expected_rollout_stages(tmp_path: Path) -> None:
    docs_path = tmp_path / "Docs" / "canvas.md"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text("# Canvas\n\ncontexto resumido", encoding="utf-8")
    current_lineage = build_repo_document_lineage(tmp_path, "Docs/canvas.md")

    grounded_source = {
        "key": "normalized_discovery",
        "title": "Discovery",
        "source_type": "repo_rule",
        "uri": "repo://Docs/canvas.md",
        "authority_level": "canonical",
        "required": True,
        "relative_path": "knowledge/required/define.md",
        "source_refs": ["Docs/canvas.md"],
        "source_lineage": [current_lineage],
    }

    report = build_memory_observability_report(
        skill_runs=[_skill_run(source=grounded_source, source_action="build_canvas")],
        short_term_memory=_short_term_memory(),
        repo_root=tmp_path,
        expected_stages=[
            ("define", "Define"),
            ("design", "Design"),
            ("tools", "Tools"),
            ("memory", "Memory"),
            ("evaluate", "Evaluate"),
            ("build", "Build"),
        ],
    )

    by_stage = {item.scope_key: item for item in report.by_stage}

    assert by_stage["define"].llm_runs == 1
    assert by_stage["define"].average_compression_gain > 0
    assert by_stage["tools"].llm_runs == 0
    assert by_stage["memory"].llm_runs == 0
    assert by_stage["build"].llm_runs == 0
