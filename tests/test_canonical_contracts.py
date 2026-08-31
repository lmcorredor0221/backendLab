from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.contracts import (
    CANONICAL_CONTRACT_MODELS,
    BlueprintCoreV1,
    ConstructionPackV1,
    PromptPackV1,
    ToolContractV1,
    collect_validation_issues,
)
from app.services.canonical_exports import build_contract_bundle, build_knowledge_contract
from app.services.shared_specs import resolve_shared_specs_dir
from tests.api_testkit import build_test_client
from tests.canonical_fixture_builder import FIXTURE_CASES, build_full_session_snapshot, sanitize_dynamic_contract_value

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMAS_ROOT = resolve_shared_specs_dir() / "schemas"
STAGE1_GOLDEN_ROOT = REPO_ROOT / "Docs" / "reingenieria-plataforma-2026-07-15" / "stage-1" / "golden"


@pytest.fixture(scope="module")
def canonical_bundles() -> dict[str, dict[str, object]]:
    bundles: dict[str, dict[str, object]] = {}
    with pytest.MonkeyPatch.context() as monkeypatch:
        with build_test_client(monkeypatch) as client:
            for case in FIXTURE_CASES:
                snapshot = build_full_session_snapshot(client, case["key"], case["title"])
                bundles[case["key"]] = build_contract_bundle(snapshot)
    return bundles


def test_blueprint_core_excludes_operational_fields(canonical_bundles: dict[str, dict[str, object]]) -> None:
    for bundle in canonical_bundles.values():
        payload = bundle["blueprint-core.v1"].model_dump(mode="json")
        assert "integration_statuses" not in payload
        assert "feature_flags" not in payload
        assert "alert_events" not in payload
        assert "metric_snapshots" not in payload
        serialized = json.dumps(payload, ensure_ascii=True)
        assert "access_token" not in serialized
        assert "password" not in serialized


def test_round_trip_is_semantically_equivalent(canonical_bundles: dict[str, dict[str, object]]) -> None:
    for bundle in canonical_bundles.values():
        for contract_key, contract in bundle.items():
            if isinstance(contract, list):
                for item in contract:
                    model_cls = CANONICAL_CONTRACT_MODELS[contract_key]
                    payload = item.model_dump(mode="json")
                    round_trip = model_cls.model_validate(payload).model_dump(mode="json")
                    assert round_trip == payload
                continue

            model_cls = CANONICAL_CONTRACT_MODELS[contract_key]
            payload = contract.model_dump(mode="json")
            round_trip = model_cls.model_validate(payload).model_dump(mode="json")
            assert round_trip == payload


def test_prompt_pack_origin_versions_are_explicit(canonical_bundles: dict[str, dict[str, object]]) -> None:
    for bundle in canonical_bundles.values():
        prompt_pack = bundle["prompt-pack.v1"]
        assert isinstance(prompt_pack, PromptPackV1)
        assert prompt_pack.origin.blueprint_core_version == "blueprint-core.v1"
        assert prompt_pack.origin.behavior_spec_version == "behavior-spec.v1"
        assert prompt_pack.origin.llm_policy_version == "llm-policy.v1"
        assert prompt_pack.origin.heuristic_decision_version == "heuristic-decision.v1"
        assert len(prompt_pack.origin.input_hash) == 64


def test_supervisor_architecture_adds_multi_agent_contracts_benchmark_and_prompts() -> None:
    with pytest.MonkeyPatch.context() as monkeypatch:
        with build_test_client(monkeypatch) as client:
            snapshot = build_full_session_snapshot(client, "02-agent-with-tools", "Caso supervisor con especialistas")

    mutated_snapshot = snapshot.model_copy(deep=True)
    mutated_snapshot.blueprint.architecture = "supervisor_with_subagents"
    mutated_snapshot.selected_workflow_template_key = "subagent_escalation_workflow"
    bundle = build_contract_bundle(mutated_snapshot)

    construction_pack = bundle["construction-pack.v1"]
    prompt_pack = bundle["prompt-pack.v1"]

    assert construction_pack.multi_agent_benchmark is not None
    assert construction_pack.multi_agent_benchmark.go_decision == "go"
    assert construction_pack.behavior_spec.multi_agent_topology is not None
    assert construction_pack.behavior_spec.multi_agent_topology.declared_pattern == "supervisor_with_subagents"
    assert construction_pack.behavior_spec.multi_agent_topology.runtime_pattern == "supervisor_specialist_runtime"
    assert construction_pack.behavior_spec.multi_agent_topology.support_state == "supported"
    assert construction_pack.behavior_spec.multi_agent_topology.agent_contracts
    assert construction_pack.behavior_spec.multi_agent_topology.message_contracts
    assert construction_pack.behavior_spec.multi_agent_topology.handoff_contracts
    assert construction_pack.behavior_spec.multi_agent_topology.shared_state_contracts
    assert prompt_pack.agent_role_prompts
    assert prompt_pack.handoff_prompts
    assert any(item.prompt_key == "agent_role_supervisor" for item in prompt_pack.agent_role_prompts)
    assert any(item.prompt_key == "handoff_supervisor_to_risk_review" for item in prompt_pack.handoff_prompts)


def test_reference_consumers_load_construction_and_prompt_packs_without_snapshot_dependencies(
    canonical_bundles: dict[str, dict[str, object]]
) -> None:
    for bundle in canonical_bundles.values():
        construction_payload = bundle["construction-pack.v1"].model_dump(mode="json")
        prompt_payload = bundle["prompt-pack.v1"].model_dump(mode="json")

        construction_pack = ConstructionPackV1.model_validate(construction_payload)
        prompt_pack = PromptPackV1.model_validate(prompt_payload)

        serialized_construction = json.dumps(construction_payload, ensure_ascii=True)
        serialized_prompt = json.dumps(prompt_payload, ensure_ascii=True)

        assert "SessionSnapshot" not in serialized_construction
        assert "SessionSnapshot" not in serialized_prompt
        assert construction_pack.prompt_pack.schema_version == "prompt-pack.v1"
        assert prompt_pack.system_prompt.context_sources
        assert all(isinstance(item.output_schema, dict) for item in [
            prompt_pack.system_prompt,
            prompt_pack.planner_prompt,
            prompt_pack.executor_prompt,
            prompt_pack.evaluator_prompt,
        ])


def test_test_pack_exposes_mutation_prompt_recovery_and_stable_issue_catalog(
    canonical_bundles: dict[str, dict[str, object]]
) -> None:
    for bundle in canonical_bundles.values():
        test_pack = bundle["test-pack.v1"]

        assert test_pack.framework_target == "python-stdlib-external-consumer"
        assert any(item.contract_key == "evaluation-pack.v1" for item in test_pack.fixtures)
        assert any(item.contract_key == "evaluation-pack.v1" for item in test_pack.mutation_cases)
        assert test_pack.prompt_evaluation_cases
        assert test_pack.recovery_cases
        assert test_pack.acceptance_journeys
        assert test_pack.stable_issue_catalog
        assert all(item.remediation for item in test_pack.stable_issue_catalog)
        assert any(item.kind == "validation_issue" for item in test_pack.stable_issue_catalog)
        assert any(item.kind == "construction_gap" for item in test_pack.stable_issue_catalog)
        assert test_pack.external_consumer.relative_path == "consumers/python/reference_consumer.py"
        assert "reference_consumer.py" in test_pack.external_consumer.entry_command


def test_knowledge_contract_distinguishes_no_rag_and_rag_cases(
    canonical_bundles: dict[str, dict[str, object]]
) -> None:
    no_rag = canonical_bundles["02-agent-with-tools"]["knowledge-contract.v1"]
    rag = canonical_bundles["03-agent-with-knowledge-rag"]["knowledge-contract.v1"]

    assert no_rag.enabled is False
    assert no_rag.mode == "none"
    assert no_rag.sources == []
    assert no_rag.source_lineage == []
    assert no_rag.ingestion_policy is None
    assert no_rag.embedding_policy is None
    assert no_rag.retrieval_policy is None
    assert no_rag.refresh_policy is None
    assert no_rag.open_questions == []

    assert rag.enabled is True
    assert rag.mode == "rag"
    assert [item.key for item in rag.sources] == ["approved_runbooks", "service_kb"]
    assert rag.source_lineage == [
        "approved_runbooks::2026-07-12",
        "service_kb::2026-07-14",
    ]
    assert rag.ingestion_policy is not None
    assert rag.embedding_policy is not None
    assert rag.retrieval_policy is not None
    assert rag.refresh_policy is not None
    assert rag.open_questions == []


def test_knowledge_contract_lineage_changes_when_source_version_changes() -> None:
    with pytest.MonkeyPatch.context() as monkeypatch:
        with build_test_client(monkeypatch) as client:
            snapshot = build_full_session_snapshot(client, "03-agent-with-knowledge-rag", "Caso agente con knowledge rag")
            baseline = build_knowledge_contract(snapshot)

    mutated_snapshot = snapshot.model_copy(deep=True)
    mutated_snapshot.blueprint.knowledge_profile.sources[0].source_version = "2026-07-16"
    changed = build_knowledge_contract(mutated_snapshot)

    assert baseline.source_lineage[0] == "approved_runbooks::2026-07-12"
    assert changed.source_lineage[0] == "approved_runbooks::2026-07-16"
    assert baseline.source_lineage != changed.source_lineage


def test_memory_policy_publishes_context_budgets_summary_and_retrieval_scopes(
    canonical_bundles: dict[str, dict[str, object]]
) -> None:
    memory_policy = canonical_bundles["03-agent-with-knowledge-rag"]["memory-policy.v1"]

    assert memory_policy.context_budgets
    assert any(item.role == "planner" and item.max_tokens > 0 for item in memory_policy.context_budgets)
    assert memory_policy.summary_policy
    assert memory_policy.invalidation_policy
    assert "session.short_term.summary_cache" in memory_policy.retrieval_scopes
    assert "knowledge.approved_sources" in memory_policy.retrieval_scopes


def test_short_term_memory_contract_tracks_stage_checkpoints_and_operational_refs(
    canonical_bundles: dict[str, dict[str, object]]
) -> None:
    short_term_memory = canonical_bundles["03-agent-with-knowledge-rag"]["short-term-memory.v1"]

    assert short_term_memory.active_stage
    assert short_term_memory.active_goal
    assert any(item.key == "stage_checkpoint:discover" for item in short_term_memory.checkpoint_refs)
    assert any(item.namespace == "session.short_term.summary_cache" for item in short_term_memory.namespaces)
    assert any(item.namespace == "session.branch_board" for item in short_term_memory.namespaces)
    assert short_term_memory.compaction.summary_policy
    assert short_term_memory.compaction.invalidation_policy


def test_knowledge_manifest_merges_blueprint_sources_with_repo_taxonomy(
    canonical_bundles: dict[str, dict[str, object]]
) -> None:
    manifest = canonical_bundles["03-agent-with-knowledge-rag"]["knowledge-manifest.v1"]

    assert manifest.knowledge_backend_mode == "hybrid_docs_session_and_rag"
    assert {item.key for item in manifest.required_sources} >= {
        "approved_runbooks",
        "service_kb",
        "reingenieria_core_canonical",
    }
    assert any(item.key == "system_analysis_operational" for item in manifest.candidate_sources)
    assert all(item.key != "ux_visual_reference" for item in manifest.candidate_sources)
    assert "knowledge.required_sources" in manifest.retrieval_scopes


def test_invalid_contracts_surface_stable_field_codes() -> None:
    with pytest.raises(Exception) as blueprint_exc:
        BlueprintCoreV1.model_validate({})
    blueprint_issues = collect_validation_issues(blueprint_exc.value)
    assert any(issue.code == "missing" and issue.path == "source_session_id" for issue in blueprint_issues)
    assert any(issue.code == "missing" and issue.path == "identity" for issue in blueprint_issues)

    with pytest.raises(Exception) as tool_exc:
        ToolContractV1.model_validate(
            {
                "schema_version": "tool-contract.v1",
                "source_session_id": "10e6636d-8610-4ba7-bce8-8d2bd7490f27",
                "generated_at": "2026-07-16T00:00:00",
                "source_blueprint_version": 1,
                "provenance": [],
                "name": "",
                "purpose": "",
                "risk_level": "low",
                "execution_mode": "sync",
            }
        )
    tool_issues = collect_validation_issues(tool_exc.value)
    assert any(issue.code == "value_error" and issue.path == "name" for issue in tool_issues)
    assert any(issue.code == "value_error" and issue.path == "purpose" for issue in tool_issues)


def test_schema_files_match_current_models() -> None:
    for schema_version, model_cls in CANONICAL_CONTRACT_MODELS.items():
        expected_path = SCHEMAS_ROOT / f"{schema_version}.schema.json"
        expected_payload = json.loads(expected_path.read_text(encoding="utf-8"))
        assert expected_payload == model_cls.model_json_schema()


def test_stage1_golden_contracts_match_generated_bundles(canonical_bundles: dict[str, dict[str, object]]) -> None:
    for case in FIXTURE_CASES:
        bundle = canonical_bundles[case["key"]]
        case_root = STAGE1_GOLDEN_ROOT / case["key"]
        uuid_registry: dict[str, str] = {}
        for contract_key in (
            "blueprint-core.v1",
            "construction-pack.v1",
            "prompt-pack.v1",
            "estimation-pack.v1",
            "evaluation-pack.v1",
            "test-pack.v1",
        ):
            expected_payload = json.loads((case_root / f"{contract_key}.json").read_text(encoding="utf-8"))
            payload = sanitize_dynamic_contract_value(bundle[contract_key].model_dump(mode="json"), uuid_registry)
            assert payload == expected_payload
