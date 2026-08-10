from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.services.canonical_exports import (
    build_knowledge_contract,
    build_memory_policy,
    build_tool_contracts,
)
from app.services.stage4_compiler import compile_stage4_artifacts, derive_success_criteria
from tests.api_testkit import build_test_client
from tests.canonical_fixture_builder import build_full_session_snapshot

FIXED_GENERATED_AT = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)


def compile_case(case_key: str, title: str, *, compiler_key_override: str | None = None):
    with pytest.MonkeyPatch.context() as monkeypatch:
        with build_test_client(monkeypatch) as client:
            snapshot = build_full_session_snapshot(client, case_key, title)
            tool_contracts = build_tool_contracts(snapshot, generated_at=FIXED_GENERATED_AT)
            memory_policy = build_memory_policy(snapshot, generated_at=FIXED_GENERATED_AT)
            knowledge_contract = build_knowledge_contract(snapshot, generated_at=FIXED_GENERATED_AT)
            return snapshot, compile_stage4_artifacts(
                snapshot,
                generated_at=FIXED_GENERATED_AT,
                tool_contracts=tool_contracts,
                memory_policy=memory_policy,
                knowledge_contract=knowledge_contract,
                success_criteria=derive_success_criteria(snapshot),
                compiler_key_override=compiler_key_override,
            )


def test_stage4_compiler_generates_observable_differences_between_phase4a_patterns() -> None:
    snapshot, _ = compile_case("02-agent-with-tools", "Caso agente con tools")
    tool_contracts = build_tool_contracts(snapshot, generated_at=FIXED_GENERATED_AT)
    memory_policy = build_memory_policy(snapshot, generated_at=FIXED_GENERATED_AT)
    knowledge_contract = build_knowledge_contract(snapshot, generated_at=FIXED_GENERATED_AT)
    success_criteria = derive_success_criteria(snapshot)

    compiled = {
        key: compile_stage4_artifacts(
            snapshot,
            generated_at=FIXED_GENERATED_AT,
            tool_contracts=tool_contracts,
            memory_policy=memory_policy,
            knowledge_contract=knowledge_contract,
            success_criteria=success_criteria,
            compiler_key_override=key,
        )
        for key in ("deterministic", "tool-calling", "plan-and-execute", "react")
    }

    assert len({item.behavior_spec.execution_pattern for item in compiled.values()}) == 4
    assert compiled["plan-and-execute"].prompt_pack.planner_prompt.content != compiled["react"].prompt_pack.planner_prompt.content
    assert (
        compiled["deterministic"].prompt_pack.planner_prompt.output_schema
        != compiled["react"].prompt_pack.planner_prompt.output_schema
    )
    assert compiled["tool-calling"].prompt_pack.tool_use_prompt is not None
    assert compiled["tool-calling"].heuristic_decision.recommended_prompts.count("tool_use") == 1


def test_stage4_compiler_is_idempotent_for_same_input() -> None:
    snapshot, first = compile_case("01-copilot-simple", "Caso copiloto simple")
    tool_contracts = build_tool_contracts(snapshot, generated_at=FIXED_GENERATED_AT)
    memory_policy = build_memory_policy(snapshot, generated_at=FIXED_GENERATED_AT)
    knowledge_contract = build_knowledge_contract(snapshot, generated_at=FIXED_GENERATED_AT)
    second = compile_stage4_artifacts(
        snapshot,
        generated_at=FIXED_GENERATED_AT,
        tool_contracts=tool_contracts,
        memory_policy=memory_policy,
        knowledge_contract=knowledge_contract,
        success_criteria=derive_success_criteria(snapshot),
    )

    assert first.behavior_spec.model_dump(mode="json") == second.behavior_spec.model_dump(mode="json")
    assert first.heuristic_decision.model_dump(mode="json") == second.heuristic_decision.model_dump(mode="json")
    assert first.llm_policy.model_dump(mode="json") == second.llm_policy.model_dump(mode="json")
    assert first.prompt_pack.model_dump(mode="json") == second.prompt_pack.model_dump(mode="json")


def test_stage4_compiler_blocks_unsupported_reasoning_without_silent_fallback() -> None:
    snapshot, _ = compile_case("01-copilot-simple", "Caso copiloto simple")
    mutated_snapshot = snapshot.model_copy(deep=True)
    mutated_snapshot.blueprint.reasoning_pattern = "HTN"

    tool_contracts = build_tool_contracts(mutated_snapshot, generated_at=FIXED_GENERATED_AT)
    memory_policy = build_memory_policy(mutated_snapshot, generated_at=FIXED_GENERATED_AT)
    knowledge_contract = build_knowledge_contract(mutated_snapshot, generated_at=FIXED_GENERATED_AT)

    result = compile_stage4_artifacts(
        mutated_snapshot,
        generated_at=FIXED_GENERATED_AT,
        tool_contracts=tool_contracts,
        memory_policy=memory_policy,
        knowledge_contract=knowledge_contract,
        success_criteria=derive_success_criteria(mutated_snapshot),
    )

    assert result.behavior_spec.reasoning_pattern == "HTN"
    assert result.behavior_spec.execution_pattern.startswith("unsupported::")
    assert any("no soporta" in note for note in result.heuristic_decision.review_notes)
    assert result.prompt_pack.recovery_prompt is not None
    assert "needs-resolution" in result.prompt_pack.executor_prompt.content


def test_stage4_prompt_dependencies_only_change_roles_that_depend_on_tools() -> None:
    snapshot, baseline = compile_case("02-agent-with-tools", "Caso agente con tools")
    mutated_snapshot = snapshot.model_copy(deep=True)
    renamed_tool = mutated_snapshot.blueprint.tools[0].model_copy(
        update={"name": f"{mutated_snapshot.blueprint.tools[0].name}_v2"}
    )
    mutated_snapshot.blueprint.tools = [renamed_tool, *mutated_snapshot.blueprint.tools[1:]]

    tool_contracts = build_tool_contracts(mutated_snapshot, generated_at=FIXED_GENERATED_AT)
    memory_policy = build_memory_policy(mutated_snapshot, generated_at=FIXED_GENERATED_AT)
    knowledge_contract = build_knowledge_contract(mutated_snapshot, generated_at=FIXED_GENERATED_AT)
    changed = compile_stage4_artifacts(
        mutated_snapshot,
        generated_at=FIXED_GENERATED_AT,
        tool_contracts=tool_contracts,
        memory_policy=memory_policy,
        knowledge_contract=knowledge_contract,
        success_criteria=derive_success_criteria(mutated_snapshot),
    )

    assert baseline.prompt_pack.origin.input_hash != changed.prompt_pack.origin.input_hash
    assert baseline.prompt_pack.system_prompt.content == changed.prompt_pack.system_prompt.content
    assert baseline.prompt_pack.planner_prompt.content == changed.prompt_pack.planner_prompt.content
    assert baseline.prompt_pack.evaluator_prompt.content == changed.prompt_pack.evaluator_prompt.content
    assert baseline.prompt_pack.tool_use_prompt is not None
    assert changed.prompt_pack.tool_use_prompt is not None
    assert baseline.prompt_pack.tool_use_prompt.content != changed.prompt_pack.tool_use_prompt.content


def test_stage4_skips_retrieval_prompt_when_case_does_not_use_rag() -> None:
    _, compiled = compile_case("02-agent-with-tools", "Caso agente con tools")

    assert compiled.prompt_pack.memory_prompt is not None
    assert compiled.prompt_pack.retrieval_prompt is None


def test_stage4_memory_and_retrieval_prompts_reflect_s6_policies() -> None:
    _, compiled = compile_case("03-agent-with-knowledge-rag", "Caso agente con knowledge rag")

    assert compiled.prompt_pack.memory_prompt is not None
    assert compiled.prompt_pack.retrieval_prompt is not None

    memory_content = compiled.prompt_pack.memory_prompt.content
    retrieval_content = compiled.prompt_pack.retrieval_prompt.content
    llm_functions = {item.role: item for item in compiled.llm_policy.functions}

    assert "Retention policy: Retener checkpoints aprobados y notas de retrieval por 30 dias" in memory_content
    assert "TTL policy: Expirar cache de retrieval en 24 horas" in memory_content
    assert "Retrieval scopes: session.short_term.summary_cache" in memory_content
    assert "Summary policy:" in memory_content
    assert "Invalidation policy:" in memory_content
    assert "Context budgets: planner=" in memory_content
    assert "No persistir chunks completos de documentos privados" in memory_content
    assert "No evidence behavior: No reutilizar memoria dudosa; volver a retrieval o pedir aclaracion." in memory_content
    assert "Assembled context:" in memory_content
    assert "required staged sources: short_term_memory" in memory_content
    assert "knowledge-manifest.v1" in compiled.prompt_pack.memory_prompt.context_sources
    assert "short-term-memory.v1" in compiled.prompt_pack.memory_prompt.context_sources

    assert "Knowledge mode: rag" in retrieval_content
    assert "Source lineage: approved_runbooks::2026-07-12, service_kb::2026-07-14" in retrieval_content
    assert (
        "Fallback: Devolver needs-resolution con las fuentes consultadas cuando no exista evidencia suficiente."
        in retrieval_content
    )
    assert "No exportar documentos privados ni credenciales" in retrieval_content
    assert "Assembled context:" in retrieval_content
    assert "required staged sources: short_term_memory" in retrieval_content
    assert "reingenieria_core_canonical" in compiled.prompt_pack.retrieval_prompt.context_sources
    assert "knowledge-manifest.v1" in compiled.prompt_pack.retrieval_prompt.context_sources
    assert "short-term-memory.v1" in compiled.prompt_pack.retrieval_prompt.context_sources
    assert "memory" in llm_functions
    assert "retrieval" in llm_functions
    assert "knowledge-manifest.v1" in llm_functions["memory"].context_sources
    assert "short-term-memory.v1" in llm_functions["retrieval"].context_sources
    assert "reingenieria_core_canonical" in llm_functions["retrieval"].context_sources


def test_stage4_recovery_prompt_uses_assembled_memory_context_when_enabled() -> None:
    snapshot, _ = compile_case("01-copilot-simple", "Caso copiloto simple")
    mutated_snapshot = snapshot.model_copy(deep=True)
    mutated_snapshot.blueprint.reasoning_pattern = "HTN"

    tool_contracts = build_tool_contracts(mutated_snapshot, generated_at=FIXED_GENERATED_AT)
    memory_policy = build_memory_policy(mutated_snapshot, generated_at=FIXED_GENERATED_AT)
    knowledge_contract = build_knowledge_contract(mutated_snapshot, generated_at=FIXED_GENERATED_AT)
    compiled = compile_stage4_artifacts(
        mutated_snapshot,
        generated_at=FIXED_GENERATED_AT,
        tool_contracts=tool_contracts,
        memory_policy=memory_policy,
        knowledge_contract=knowledge_contract,
        success_criteria=derive_success_criteria(mutated_snapshot),
    )

    assert compiled.prompt_pack.recovery_prompt is not None
    assert "Assembled context:" in compiled.prompt_pack.recovery_prompt.content
    assert "required staged sources: short_term_memory" in compiled.prompt_pack.recovery_prompt.content
    assert "short-term-memory.v1" in compiled.prompt_pack.recovery_prompt.context_sources
    llm_functions = {item.role: item for item in compiled.llm_policy.functions}
    assert "recovery" in llm_functions
    assert "short-term-memory.v1" in llm_functions["recovery"].context_sources


def test_stage4_llm_policy_changes_invalidate_prompt_origin_and_override_models() -> None:
    snapshot, baseline = compile_case("02-agent-with-tools", "Caso agente con tools")
    mutated_snapshot = snapshot.model_copy(deep=True)
    mutated_snapshot.blueprint.llm_policy.provider = "deepseek"
    mutated_snapshot.blueprint.llm_policy.fast_model = "deepseek-chat"
    mutated_snapshot.blueprint.llm_policy.reasoning_model = "deepseek-reasoner"
    mutated_snapshot.blueprint.llm_policy.fallback_model = "manual_review_gate"
    mutated_snapshot.blueprint.llm_policy.functions[0].provider = "deepseek"
    mutated_snapshot.blueprint.llm_policy.functions[0].model = "deepseek-reasoner"
    mutated_snapshot.blueprint.llm_policy.functions[1].provider = "deepseek"
    mutated_snapshot.blueprint.llm_policy.functions[1].model = "deepseek-chat"

    tool_contracts = build_tool_contracts(mutated_snapshot, generated_at=FIXED_GENERATED_AT)
    memory_policy = build_memory_policy(mutated_snapshot, generated_at=FIXED_GENERATED_AT)
    knowledge_contract = build_knowledge_contract(mutated_snapshot, generated_at=FIXED_GENERATED_AT)
    changed = compile_stage4_artifacts(
        mutated_snapshot,
        generated_at=FIXED_GENERATED_AT,
        tool_contracts=tool_contracts,
        memory_policy=memory_policy,
        knowledge_contract=knowledge_contract,
        success_criteria=derive_success_criteria(mutated_snapshot),
    )

    assert baseline.prompt_pack.origin.input_hash != changed.prompt_pack.origin.input_hash
    assert changed.llm_policy.provider == "deepseek"
    assert changed.llm_policy.fast_model == "deepseek-chat"
    assert changed.llm_policy.reasoning_model == "deepseek-reasoner"
    assert changed.llm_policy.functions[0].provider == "deepseek"
    assert changed.llm_policy.functions[0].model == "deepseek-reasoner"
