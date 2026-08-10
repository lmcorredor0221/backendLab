from __future__ import annotations

import json
import sys
from uuid import UUID
from pathlib import Path
from typing import Any

import pytest

from app.contracts import CANONICAL_CONTRACT_MODELS, AgentConstructionPackageV2
from app.services.canonical_exports import build_agent_construction_package_v2
from tests.api_testkit import build_test_client
from tests.canonical_fixture_builder import build_full_session_snapshot

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ROOT = REPO_ROOT / "shared_specs" / "schemas"
REFERENCE_CONSUMER_ROOT = REPO_ROOT / "shared_specs" / "reference_consumers" / "python"
SAMPLE_PATH = REPO_ROOT / "shared_specs" / "examples" / "agent-construction-package.v2.sample.json"

if str(REFERENCE_CONSUMER_ROOT) not in sys.path:
    sys.path.insert(0, str(REFERENCE_CONSUMER_ROOT))

from acp_v2_reference_consumer import validate_acp_v2  # noqa: E402

LEAN_STAGE_KEYS = {
    "discover",
    "define",
    "design",
    "tools",
    "memory",
    "validate",
    "estimate",
    "package",
}


def _is_uuid(value: str) -> bool:
    try:
        UUID(value)
    except ValueError:
        return False
    return True


def _looks_like_local_or_internal_path(value: str) -> bool:
    normalized = value.lower().replace("\\", "/")
    return any(
        marker in normalized
        for marker in ("c:/", "/users/", "/home/", "/api/v1/sessions", "session_id", "project_id")
    )


@pytest.fixture(scope="module")
def acp_v2_payload() -> dict[str, Any]:
    with pytest.MonkeyPatch.context() as monkeypatch:
        with build_test_client(monkeypatch) as client:
            snapshot = build_full_session_snapshot(
                client,
                "03-agent-with-knowledge-rag",
                "Caso ACP v2 portable con RAG",
            )
            return build_agent_construction_package_v2(snapshot).model_dump(mode="json")


def test_agent_construction_package_v2_is_registered_and_round_trips(acp_v2_payload: dict[str, Any]) -> None:
    assert CANONICAL_CONTRACT_MODELS["agent-construction-package.v2"] is AgentConstructionPackageV2
    round_trip = AgentConstructionPackageV2.model_validate(acp_v2_payload).model_dump(mode="json")

    assert round_trip == acp_v2_payload
    assert round_trip["portable_manifest"]["manifest_version"] == "acp-portable-manifest.v1"
    assert len(round_trip["portable_manifest"]["contracts"]) >= 6
    assert all(len(entry["checksum_sha256"]) == 64 for entry in round_trip["portable_manifest"]["contracts"])
    assert round_trip["migration"]["from_schema_version"] == "construction-pack.v1"
    assert len(round_trip["migration"]["source_checksum_sha256"]) == 64


def test_agent_construction_package_v2_keeps_runtime_sections_portable(acp_v2_payload: dict[str, Any]) -> None:
    runtime_sections = {
        "agent_runtime": acp_v2_payload["agent_runtime"],
        "build_plan": acp_v2_payload["build_plan"],
        "implementation_decisions": acp_v2_payload["implementation_decisions"],
        "memory_knowledge_plan": acp_v2_payload["memory_knowledge_plan"],
        "memory_strategy": acp_v2_payload["memory_strategy"],
        "tool_contracts": acp_v2_payload["tool_contracts"],
    }
    serialized = json.dumps(runtime_sections, ensure_ascii=True, sort_keys=True)

    assert "/api/v1/sessions" not in serialized
    assert "SessionSnapshot" not in serialized
    assert "journey_stage_artifact_id" not in serialized
    assert "skill_run_id" not in serialized
    assert acp_v2_payload["build_plan"]["steps"]
    assert acp_v2_payload["agent_runtime"]["agents"]


def test_agent_construction_package_v2_exports_three_portable_workflows(acp_v2_payload: dict[str, Any]) -> None:
    workflow_types = {workflow["workflow_type"] for workflow in acp_v2_payload["workflows"]}

    assert workflow_types == {"construction", "runtime_operational", "human_decision_resolution"}
    for workflow in acp_v2_payload["workflows"]:
        assert workflow["entry_node"]
        assert workflow["terminal_nodes"]
        assert workflow["nodes"]
        for node in workflow["nodes"]:
            assert node["node_key"].lower() not in LEAN_STAGE_KEYS
            assert not _is_uuid(node["node_key"])
            assert node["portable_state"]
            assert node["workflow_role"] in {"construction", "runtime", "human_decision"}


def test_agent_construction_package_v2_exports_portable_checkpoints(acp_v2_payload: dict[str, Any]) -> None:
    checkpoints = acp_v2_payload["checkpoints"]
    scopes = {checkpoint["scope"] for checkpoint in checkpoints}

    assert {"construction", "runtime", "human_decision"} <= scopes
    for checkpoint in checkpoints:
        assert checkpoint["checkpoint_key"]
        assert checkpoint["checkpoint_key"].lower() not in LEAN_STAGE_KEYS
        assert not _is_uuid(checkpoint["checkpoint_key"])
        assert checkpoint["resume_strategy"]
        assert checkpoint["portable_ref"]


def test_agent_construction_package_v2_classifies_human_decisions_without_blocking_package(
    acp_v2_payload: dict[str, Any],
) -> None:
    decisions = acp_v2_payload["decision_registry"]

    assert decisions
    assert all(decision["blocking_scope"] != "package" for decision in decisions)
    assert any(decision["classification"] in {"mandatory", "environment_dependent"} for decision in decisions)
    for decision in decisions:
        assert decision["question"]
        assert decision["context"]
        assert decision["owner"]
        assert decision["recommended_moment"]
        assert decision["impact"]
        assert decision["examples"]
        assert decision["options"]
        assert all(option["description"] and option["tradeoffs"] for option in decision["options"])


def test_agent_construction_package_v2_guides_runtime_without_requiring_one(acp_v2_payload: dict[str, Any]) -> None:
    policy = acp_v2_payload["runtime_target_policy"]
    runtime_targets = acp_v2_payload["runtime_targets"]

    assert policy["recommended_runtime"]
    assert policy["required_runtime"] == []
    assert runtime_targets
    assert all(target["required"] is False for target in runtime_targets)
    assert any(target["recommendation_level"] == "recommended" for target in runtime_targets)
    for target in runtime_targets:
        assert target["rationale"]
        assert target["selection_criteria"]
        assert target["prerequisites"]
        assert target["tradeoffs"]


def test_agent_construction_package_v2_exposes_stack_as_guided_decisions(
    acp_v2_payload: dict[str, Any],
) -> None:
    decisions = acp_v2_payload["technology_decisions"]
    categories = {decision["category"] for decision in decisions}

    assert categories == {"language", "framework", "database", "vector_store", "hosting", "ci_cd", "observability"}
    assert all(decision["required_for_package"] is False for decision in decisions)
    assert {decision["decision_key"] for decision in decisions} <= {
        decision["decision_key"]
        for decision in acp_v2_payload["decision_registry"]
    }
    for decision in decisions:
        assert decision["question"]
        assert decision["selection_criteria"]
        assert decision["default_guidance"]
        assert decision["options"]
        for option in decision["options"]:
            assert option["rationale"]
            assert option["prerequisites"]
            assert option["tradeoffs"]
            assert option["examples"]


def test_agent_construction_package_v2_deployment_is_guidance_only(acp_v2_payload: dict[str, Any]) -> None:
    guide = acp_v2_payload["deployment_guide"]

    assert guide["mode"] == "guidance_only"
    assert guide["required_script"] is False
    assert guide["deployment_decision_refs"]
    assert guide["environment_prerequisites"]
    assert guide["steps"]
    assert guide["rollback_guidance"]
    assert guide["security_considerations"]
    assert guide["observability_considerations"]


def test_agent_construction_package_v2_normalizes_tools_as_abstract_capabilities(
    acp_v2_payload: dict[str, Any],
) -> None:
    capabilities = acp_v2_payload["capability_catalog"]
    tool_bindings = acp_v2_payload["tool_bindings"]
    tool_contracts = acp_v2_payload["tool_contracts"]

    assert capabilities
    assert tool_bindings
    assert {tool["tool_key"] for tool in tool_contracts} <= {binding["tool_key"] for binding in tool_bindings}
    assert {binding["capability_key"] for binding in tool_bindings} <= {
        capability["capability_key"]
        for capability in capabilities
    }
    for capability in capabilities:
        assert capability["description"]
        assert capability["abstract_inputs"]
        assert capability["abstract_outputs"]
        assert capability["replacement_options"]
        assert capability["requirement_level"] in {"required", "optional", "replaceable", "not_recommended"}


def test_agent_construction_package_v2_tool_bindings_declare_boundaries_risks_and_fallbacks(
    acp_v2_payload: dict[str, Any],
) -> None:
    for binding in acp_v2_payload["tool_bindings"]:
        assert binding["binding_type"] in {
            "producer_internal_tool",
            "external_api",
            "abstract_contract",
            "runtime_adapter",
        }
        assert binding["provider_boundary"] in {
            "producer_internal",
            "customer_external",
            "framework_runtime",
            "abstract",
        }
        assert binding["credentials_policy"]
        assert binding["cost_profile"]
        assert binding["risk_profile"]
        assert binding["fallback_strategy"]
        assert binding["replacement_strategy"]
        if binding["provider_boundary"] == "producer_internal":
            assert binding["requirement_level"] != "required"
            assert binding["replaceable"] is True


def test_agent_construction_package_v2_tool_analysis_surfaces_minimal_tooling_policy(
    acp_v2_payload: dict[str, Any],
) -> None:
    analysis = acp_v2_payload["tool_analysis"]

    assert analysis["summary"]
    assert analysis["overprovisioning_policy"]
    assert analysis["minimal_tooling_policy"]
    assert isinstance(analysis["redundancy_findings"], list)
    assert isinstance(analysis["incompatibility_findings"], list)
    assert isinstance(analysis["not_recommended_tools"], list)


def test_agent_construction_package_v2_memory_namespaces_are_portable(
    acp_v2_payload: dict[str, Any],
) -> None:
    plan = acp_v2_payload["memory_knowledge_plan"]
    namespaces = plan["namespaces"]
    namespace_types = {namespace["memory_type"] for namespace in namespaces}

    assert {"short_term", "long_term", "documentary_knowledge", "rag_index", "audit"} <= namespace_types
    for namespace in namespaces:
        assert namespace["namespace_key"]
        assert not _is_uuid(namespace["namespace_key"])
        assert "session_id" not in namespace["namespace_key"].lower()
        assert "project_id" not in namespace["namespace_key"].lower()
        assert "lean" not in namespace["namespace_key"].lower()
        assert namespace["portable_ref"].startswith("memory://")
        assert namespace["read_roles"]
        assert namespace["write_roles"]
        assert namespace["retention_policy"]
        assert namespace["compaction_policy"]


def test_agent_construction_package_v2_rag_dependencies_are_abstract_capabilities(
    acp_v2_payload: dict[str, Any],
) -> None:
    expected = {
        "capability_document_ingestion",
        "capability_embedding",
        "capability_vector_search",
        "capability_knowledge_retrieval",
    }
    plan = acp_v2_payload["memory_knowledge_plan"]
    capability_keys = {capability["capability_key"] for capability in acp_v2_payload["capability_catalog"]}
    dependency_keys = {
        dependency["capability_key"]
        for dependency in plan["rag_pipeline"]["capability_dependencies"]
        if dependency["required"]
    }

    assert plan["rag_pipeline"]["enabled"] is True
    assert expected <= capability_keys
    assert expected <= dependency_keys
    assert expected <= set(plan["capability_dependencies"])
    for dependency in plan["rag_pipeline"]["capability_dependencies"]:
        assert dependency["reason"]
        assert dependency["fallback"]


def test_agent_construction_package_v2_knowledge_artifacts_are_traceable_and_portable(
    acp_v2_payload: dict[str, Any],
) -> None:
    artifacts = acp_v2_payload["memory_knowledge_plan"]["knowledge_artifacts"]

    assert artifacts
    for artifact in artifacts:
        assert artifact["artifact_key"].startswith("knowledge_artifact_")
        assert artifact["location_hint"]
        assert not _looks_like_local_or_internal_path(artifact["location_hint"])
        assert artifact["owner"]
        assert artifact["sensitivity"]
        assert artifact["source_version"]
        assert artifact["reason_to_index"]
        assert artifact["ingestion_capability_ref"] == "capability_document_ingestion"
        assert artifact["retrieval_capability_ref"] == "capability_knowledge_retrieval"
        assert artifact["permissions"]
        assert artifact["refresh_triggers"]
        assert artifact["expiration_policy"]


def test_agent_construction_package_v2_context_window_policy_prevents_redundancy(
    acp_v2_payload: dict[str, Any],
) -> None:
    policy = acp_v2_payload["memory_knowledge_plan"]["context_window_policy"]

    assert policy["max_context_utilization_percent"] <= 85
    assert policy["short_term_budget_refs"]
    assert policy["compaction_trigger"]
    assert policy["anti_redundancy_rules"]
    assert any("artifact_ref" in rule for rule in policy["anti_redundancy_rules"])
    assert policy["retrieval_context_policy"]
    assert policy["pagination_policy"]
    assert policy["artifact_reference_policy"]


def test_agent_construction_package_v2_schema_file_matches_model() -> None:
    expected = json.loads((SCHEMA_ROOT / "agent-construction-package.v2.schema.json").read_text(encoding="utf-8"))

    assert expected == AgentConstructionPackageV2.model_json_schema()


def test_agent_construction_package_v2_sample_passes_external_consumer() -> None:
    payload = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))

    assert validate_acp_v2(payload) == []
