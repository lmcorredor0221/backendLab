from __future__ import annotations

import pytest

from app.contracts.canonical_v1 import MemoryContextBudgetV1
from app.services.llm_runtime.codex_cli.context_assembler import (
    CodexContextAssembler,
    CodexContextInlineSource,
    CodexContextRequest,
)
from tests.api_testkit import build_test_client
from tests.canonical_fixture_builder import REPO_ROOT, build_full_session_snapshot


def test_context_assembler_builds_budgeted_repo_sources_from_session_snapshot() -> None:
    with pytest.MonkeyPatch.context() as monkeypatch:
        with build_test_client(monkeypatch) as client:
            snapshot = build_full_session_snapshot(client, "03-agent-with-knowledge-rag", "Caso agente con knowledge rag")

    assembler = CodexContextAssembler(repo_root=REPO_ROOT)
    assembly = assembler.assemble(
        task_kind="retrieval_runtime",
        request=CodexContextRequest(
            role="retrieval",
            knowledge_access_backend="workspace_staged",
            session_snapshot=snapshot,
        ),
    )

    required_keys = {item.key for item in assembly.required_sources}
    candidate_keys = {item.key for item in assembly.candidate_sources}

    assert "short_term_memory" in required_keys
    assert "session_snapshot_focus" in required_keys
    assert "reingenieria_core_canonical" in required_keys
    assert candidate_keys & {"reingenieria_stage_operational", "system_analysis_operational"}
    assert any(item.key == "reingenieria_core_canonical" and item.source_refs for item in assembly.required_sources)
    assert any(item.key == "reingenieria_core_canonical" and item.source_lineage for item in assembly.required_sources)
    assert all(item.source_version for item in assembly.required_sources)
    assert assembly.stats.assembled_estimated_tokens > 0
    assert assembly.stats.reduction_estimated_tokens > 0
    assert assembly.stats.used_full_documents is False
    assert "knowledge/required/" in assembly.required_sources[0].relative_path
    assert "knowledge/candidate/" in assembly.candidate_sources[0].relative_path
    assert "input/knowledge_manifest.json" in assembly.prompt_preamble


def test_context_assembler_filters_short_term_namespaces_for_planner_role() -> None:
    with pytest.MonkeyPatch.context() as monkeypatch:
        with build_test_client(monkeypatch) as client:
            snapshot = build_full_session_snapshot(client, "03-agent-with-knowledge-rag", "Caso agente con knowledge rag")

    assembler = CodexContextAssembler(repo_root=REPO_ROOT)
    assembly = assembler.assemble(
        task_kind="planning_runtime",
        request=CodexContextRequest(
            role="planner",
            knowledge_access_backend="workspace_staged",
            session_snapshot=snapshot,
        ),
    )

    short_term_memory = next(item for item in assembly.required_sources if item.key == "short_term_memory")

    assert "session.short_term.discovery" in short_term_memory.content
    assert "session.short_term.plan" in short_term_memory.content
    assert "session.short_term.execution" not in short_term_memory.content
    assert "session.short_term.retrieval" not in short_term_memory.content


def test_context_assembler_preserves_required_payload_as_full_staged_file() -> None:
    marker = "ARCHITECTURE_HANDOFF_MEMORY_TOOLS_MARKER"
    long_payload = "\n".join(
        [
            "current_user:",
            "  approved_context: classify intent and retrieve corporate knowledge",
            *(f"  evidence_{index}: {'x' * 120}" for index in range(120)),
            f"  marker: {marker}",
        ]
    )
    assembler = CodexContextAssembler(repo_root=REPO_ROOT)

    assembly = assembler.assemble(
        task_kind="agent_design_critique",
        request=CodexContextRequest(
            role="builder",
            knowledge_access_backend="workspace_staged",
            inline_sources=[
                CodexContextInlineSource(
                    key="agent_design_critique_input",
                    title="Agent design critique input",
                    content=long_payload,
                    required=True,
                    summary="Propuesta de diseno del agente y contexto aprobado para revision critica.",
                )
            ],
            strict_budget=MemoryContextBudgetV1(
                role="builder",
                max_tokens=300,
                max_items=1,
                max_chars=900,
                compaction_trigger="test_budget",
                overflow_policy="compact_by_priority_then_trim_candidates",
            ),
        ),
    )

    source = assembly.required_sources[0]
    payload = source.to_payload()

    assert source.key == "agent_design_critique_input"
    assert source.prompt_truncated is True
    assert source.truncated is False
    assert marker not in source.content
    assert marker in source.workspace_content
    assert payload["prompt_truncated"] is True
    assert payload["truncated"] is False
    assert payload["staged_file_truncated"] is False
    assert payload["delivery_mode"] == "filesystem_full_required"
    assert assembly.stats.prompt_truncated_source_count == 1
    assert assembly.stats.truncated_source_count == 0
