from __future__ import annotations

import pytest

from app.services.llm_runtime.codex_cli.context_assembler import CodexContextAssembler, CodexContextRequest
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
