from __future__ import annotations

import hashlib
import json
import sys
from uuid import UUID
from pathlib import Path
from typing import Any


REQUIRED_SECTIONS = {
    "agent_runtime",
    "build_plan",
    "capability_catalog",
    "compatibility",
    "conformance",
    "checkpoints",
    "decision_registry",
    "deployment_guide",
    "implementation_decisions",
    "knowledge_sources",
    "memory_knowledge_plan",
    "memory_strategy",
    "migration",
    "portable_manifest",
    "producer_metadata",
    "prompts",
    "runtime_target_policy",
    "runtime_targets",
    "system_specification",
    "technology_decisions",
    "tests",
    "tool_analysis",
    "tool_bindings",
    "tool_contracts",
    "workflows",
}

INTERNAL_RUNTIME_MARKERS = (
    "/api/v1/sessions",
    "SessionSnapshot",
    "journey_stage_artifact_id",
    "skill_run_id",
    "workspace_internal_id",
)

INTERNAL_STAGE_KEYS = {
    "discover",
    "define",
    "design",
    "tools",
    "memory",
    "validate",
    "estimate",
    "package",
}

LOCAL_OR_INTERNAL_LOCATION_MARKERS = (
    "c:\\",
    "c:/",
    "/users/",
    "/home/",
    "/mnt/",
    "/var/",
    "/api/v1/sessions",
    "sessionid",
    "session_id",
    "projectid",
    "project_id",
)


def _looks_like_uuid(value: str) -> bool:
    try:
        UUID(value)
    except ValueError:
        return False
    return True


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_checksum(payload: Any) -> str:
    serialized = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _looks_like_internal_location(value: str) -> bool:
    normalized = value.lower().replace("\\", "/")
    return any(marker.replace("\\", "/") in normalized for marker in LOCAL_OR_INTERNAL_LOCATION_MARKERS)


def validate_acp_v2(payload: dict[str, Any]) -> list[str]:
    issues: list[str] = []

    if payload.get("schema_version") != "agent-construction-package.v2":
        issues.append("schema_version must be agent-construction-package.v2")

    missing_sections = sorted(section for section in REQUIRED_SECTIONS if section not in payload)
    issues.extend(f"missing section: {section}" for section in missing_sections)

    manifest = payload.get("portable_manifest") if isinstance(payload.get("portable_manifest"), dict) else {}
    manifest_entries = manifest.get("contracts") if isinstance(manifest.get("contracts"), list) else []
    if not manifest_entries:
        issues.append("portable_manifest.contracts must include source contract entries")
    for index, entry in enumerate(manifest_entries):
        if not isinstance(entry, dict):
            issues.append(f"portable_manifest.contracts[{index}] must be an object")
            continue
        checksum = str(entry.get("checksum_sha256") or "")
        if len(checksum) != 64:
            issues.append(f"portable_manifest.contracts[{index}].checksum_sha256 must be a sha256 hex digest")

    runtime_text = json.dumps(
        {
            "agent_runtime": payload.get("agent_runtime"),
            "build_plan": payload.get("build_plan"),
            "capability_catalog": payload.get("capability_catalog"),
            "checkpoints": payload.get("checkpoints"),
            "decision_registry": payload.get("decision_registry"),
            "deployment_guide": payload.get("deployment_guide"),
            "implementation_decisions": payload.get("implementation_decisions"),
            "memory_knowledge_plan": payload.get("memory_knowledge_plan"),
            "memory_strategy": payload.get("memory_strategy"),
            "runtime_target_policy": payload.get("runtime_target_policy"),
            "runtime_targets": payload.get("runtime_targets"),
            "technology_decisions": payload.get("technology_decisions"),
            "tool_analysis": payload.get("tool_analysis"),
            "tool_bindings": payload.get("tool_bindings"),
            "tool_contracts": payload.get("tool_contracts"),
            "workflows": payload.get("workflows"),
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    for marker in INTERNAL_RUNTIME_MARKERS:
        if marker in runtime_text:
            issues.append(f"internal runtime marker present: {marker}")

    if not payload.get("build_plan", {}).get("steps"):
        issues.append("build_plan.steps must not be empty")
    workflows = payload.get("workflows") if isinstance(payload.get("workflows"), list) else []
    workflow_types = {
        workflow.get("workflow_type")
        for workflow in workflows
        if isinstance(workflow, dict)
    }
    expected_workflow_types = {"construction", "runtime_operational", "human_decision_resolution"}
    if workflow_types != expected_workflow_types:
        issues.append("workflows must include construction, runtime_operational and human_decision_resolution")
    for workflow in workflows:
        if not isinstance(workflow, dict):
            issues.append("workflow entries must be objects")
            continue
        if not workflow.get("entry_node"):
            issues.append(f"workflow {workflow.get('workflow_key')} must include entry_node")
        nodes = workflow.get("nodes") if isinstance(workflow.get("nodes"), list) else []
        if not nodes:
            issues.append(f"workflow {workflow.get('workflow_key')} must include nodes")
        for node in nodes:
            if not isinstance(node, dict):
                issues.append("workflow nodes must be objects")
                continue
            node_key = str(node.get("node_key") or "")
            if node_key.lower() in INTERNAL_STAGE_KEYS:
                issues.append(f"workflow node_key must be portable, not Lean stage key: {node_key}")

    checkpoints = payload.get("checkpoints") if isinstance(payload.get("checkpoints"), list) else []
    if not checkpoints:
        issues.append("checkpoints must not be empty")
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict):
            issues.append("checkpoint entries must be objects")
            continue
        checkpoint_key = str(checkpoint.get("checkpoint_key") or "")
        if not checkpoint_key:
            issues.append("checkpoint_key must not be empty")
        if _looks_like_uuid(checkpoint_key):
            issues.append(f"checkpoint_key must be portable, not UUID: {checkpoint_key}")
        if checkpoint_key.lower() in INTERNAL_STAGE_KEYS:
            issues.append(f"checkpoint_key must be portable, not Lean stage key: {checkpoint_key}")
        if not checkpoint.get("resume_strategy"):
            issues.append(f"checkpoint {checkpoint_key} must include resume_strategy")

    decisions = payload.get("decision_registry") if isinstance(payload.get("decision_registry"), list) else []
    for decision in decisions:
        if not isinstance(decision, dict):
            issues.append("decision_registry entries must be objects")
            continue
        decision_key = str(decision.get("decision_key") or "")
        if decision.get("classification") == "deferable" and decision.get("blocking_scope") == "package":
            issues.append(f"deferable decision must not block package: {decision_key}")
        for field in ("question", "context", "owner", "recommended_moment", "impact", "examples", "options"):
            if not decision.get(field):
                issues.append(f"decision {decision_key} missing {field}")

    runtime_policy = payload.get("runtime_target_policy") if isinstance(payload.get("runtime_target_policy"), dict) else {}
    if runtime_policy.get("required_runtime"):
        issues.append("runtime_target_policy.required_runtime must be empty unless explicitly selected during ACP execution")
    if not runtime_policy.get("recommended_runtime"):
        issues.append("runtime_target_policy.recommended_runtime must include at least one recommendation")

    runtime_targets = payload.get("runtime_targets") if isinstance(payload.get("runtime_targets"), list) else []
    if not runtime_targets:
        issues.append("runtime_targets must not be empty")
    for target in runtime_targets:
        if not isinstance(target, dict):
            issues.append("runtime_targets entries must be objects")
            continue
        if target.get("required") is True:
            issues.append(f"runtime target must not be required by default: {target.get('target_key')}")
        for field in ("selection_criteria", "prerequisites", "tradeoffs", "rationale"):
            if not target.get(field):
                issues.append(f"runtime target {target.get('target_key')} missing {field}")

    technology_decisions = (
        payload.get("technology_decisions")
        if isinstance(payload.get("technology_decisions"), list)
        else []
    )
    expected_categories = {"language", "framework", "database", "vector_store", "hosting", "ci_cd", "observability"}
    categories = {
        decision.get("category")
        for decision in technology_decisions
        if isinstance(decision, dict)
    }
    if categories != expected_categories:
        issues.append("technology_decisions must cover language, framework, database, vector_store, hosting, ci_cd and observability")
    for decision in technology_decisions:
        if not isinstance(decision, dict):
            issues.append("technology_decisions entries must be objects")
            continue
        decision_key = str(decision.get("decision_key") or "")
        if decision.get("required_for_package") is True:
            issues.append(f"technology decision must not be required for package: {decision_key}")
        for field in ("question", "selection_criteria", "options", "default_guidance"):
            if not decision.get(field):
                issues.append(f"technology decision {decision_key} missing {field}")
        for option in decision.get("options") or []:
            if not isinstance(option, dict):
                issues.append(f"technology decision {decision_key} options must be objects")
                continue
            for field in ("rationale", "prerequisites", "tradeoffs", "examples"):
                if not option.get(field):
                    issues.append(f"technology option {option.get('option_key')} missing {field}")

    deployment_guide = payload.get("deployment_guide") if isinstance(payload.get("deployment_guide"), dict) else {}
    if deployment_guide.get("mode") != "guidance_only":
        issues.append("deployment_guide.mode must be guidance_only")
    if deployment_guide.get("required_script") is True:
        issues.append("deployment_guide.required_script must be false")
    for field in ("deployment_decision_refs", "environment_prerequisites", "steps", "rollback_guidance"):
        if not deployment_guide.get(field):
            issues.append(f"deployment_guide missing {field}")

    capability_catalog = payload.get("capability_catalog") if isinstance(payload.get("capability_catalog"), list) else []
    tool_bindings = payload.get("tool_bindings") if isinstance(payload.get("tool_bindings"), list) else []
    tool_contracts = payload.get("tool_contracts") if isinstance(payload.get("tool_contracts"), list) else []
    if tool_contracts and not capability_catalog:
        issues.append("capability_catalog must not be empty when tool_contracts exist")
    if tool_contracts and not tool_bindings:
        issues.append("tool_bindings must not be empty when tool_contracts exist")

    capability_keys = {
        capability.get("capability_key")
        for capability in capability_catalog
        if isinstance(capability, dict)
    }
    for capability in capability_catalog:
        if not isinstance(capability, dict):
            issues.append("capability_catalog entries must be objects")
            continue
        capability_key = str(capability.get("capability_key") or "")
        for field in ("description", "rationale", "abstract_inputs", "abstract_outputs", "replacement_options", "source_refs"):
            if not capability.get(field):
                issues.append(f"capability {capability_key} missing {field}")
        if capability.get("requirement_level") not in {"required", "optional", "replaceable", "not_recommended"}:
            issues.append(f"capability {capability_key} has invalid requirement_level")

    tool_contract_keys = {
        tool.get("tool_key") or tool.get("name")
        for tool in tool_contracts
        if isinstance(tool, dict)
    }
    binding_tool_keys = set()
    for binding in tool_bindings:
        if not isinstance(binding, dict):
            issues.append("tool_bindings entries must be objects")
            continue
        binding_key = str(binding.get("binding_key") or "")
        binding_tool_keys.add(binding.get("tool_key"))
        if binding.get("capability_key") not in capability_keys:
            issues.append(f"tool binding {binding_key} references unknown capability")
        if binding.get("provider_boundary") == "producer_internal" and binding.get("requirement_level") == "required":
            issues.append(f"producer internal binding must not be required: {binding_key}")
        for field in ("credentials_policy", "cost_profile", "risk_profile", "fallback_strategy", "replacement_strategy"):
            if not binding.get(field):
                issues.append(f"tool binding {binding_key} missing {field}")
        if binding.get("requirement_level") not in {"required", "optional", "replaceable", "not_recommended"}:
            issues.append(f"tool binding {binding_key} has invalid requirement_level")
    missing_bindings = sorted(str(key) for key in tool_contract_keys if key and key not in binding_tool_keys)
    if missing_bindings:
        issues.append(f"tool_contracts without bindings: {', '.join(missing_bindings)}")

    tool_analysis = payload.get("tool_analysis") if isinstance(payload.get("tool_analysis"), dict) else {}
    for field in ("summary", "overprovisioning_policy", "minimal_tooling_policy"):
        if not tool_analysis.get(field):
            issues.append(f"tool_analysis missing {field}")

    memory_knowledge_plan = (
        payload.get("memory_knowledge_plan")
        if isinstance(payload.get("memory_knowledge_plan"), dict)
        else {}
    )
    namespaces = (
        memory_knowledge_plan.get("namespaces")
        if isinstance(memory_knowledge_plan.get("namespaces"), list)
        else []
    )
    if not namespaces:
        issues.append("memory_knowledge_plan.namespaces must not be empty")
    namespace_types = {
        namespace.get("memory_type")
        for namespace in namespaces
        if isinstance(namespace, dict)
    }
    required_namespace_types = {"short_term", "long_term", "documentary_knowledge", "audit"}
    missing_namespace_types = sorted(required_namespace_types - namespace_types)
    if missing_namespace_types:
        issues.append(f"memory_knowledge_plan missing namespace types: {', '.join(missing_namespace_types)}")
    for namespace in namespaces:
        if not isinstance(namespace, dict):
            issues.append("memory_knowledge_plan.namespaces entries must be objects")
            continue
        namespace_key = str(namespace.get("namespace_key") or "")
        if not namespace_key:
            issues.append("memory namespace_key must not be empty")
        if _looks_like_uuid(namespace_key):
            issues.append(f"memory namespace_key must be portable, not UUID: {namespace_key}")
        forbidden_tokens = ("sessionid", "session_id", "projectid", "project_id", "lean")
        if any(token in namespace_key.lower() for token in forbidden_tokens):
            issues.append(f"memory namespace_key must not expose producer internals: {namespace_key}")
        for field in ("purpose", "read_roles", "write_roles", "retention_policy", "compaction_policy", "portable_ref"):
            if not namespace.get(field):
                issues.append(f"memory namespace {namespace_key} missing {field}")

    rag_pipeline = (
        memory_knowledge_plan.get("rag_pipeline")
        if isinstance(memory_knowledge_plan.get("rag_pipeline"), dict)
        else {}
    )
    rag_enabled = rag_pipeline.get("enabled") is True
    rag_dependencies = (
        rag_pipeline.get("capability_dependencies")
        if isinstance(rag_pipeline.get("capability_dependencies"), list)
        else []
    )
    rag_dependency_keys = {
        dependency.get("capability_key")
        for dependency in rag_dependencies
        if isinstance(dependency, dict)
    }
    expected_rag_dependencies = {
        "capability_document_ingestion",
        "capability_embedding",
        "capability_vector_search",
        "capability_knowledge_retrieval",
    }
    if rag_enabled:
        if "rag_index" not in namespace_types:
            issues.append("memory_knowledge_plan must include rag_index namespace when RAG is enabled")
        missing_rag_dependencies = sorted(expected_rag_dependencies - rag_dependency_keys)
        if missing_rag_dependencies:
            issues.append(f"rag_pipeline missing capability dependencies: {', '.join(missing_rag_dependencies)}")
        missing_rag_capabilities = sorted(expected_rag_dependencies - capability_keys)
        if missing_rag_capabilities:
            issues.append(f"capability_catalog missing RAG capabilities: {', '.join(missing_rag_capabilities)}")
        for dependency in rag_dependencies:
            if isinstance(dependency, dict) and dependency.get("capability_key") in expected_rag_dependencies:
                if dependency.get("required") is not True:
                    issues.append(f"RAG dependency must be required when enabled: {dependency.get('capability_key')}")
                for field in ("reason", "fallback"):
                    if not dependency.get(field):
                        issues.append(f"RAG dependency {dependency.get('capability_key')} missing {field}")
    for field in ("vector_store_decision_ref", "citation_policy", "deletion_policy", "fallback_policy"):
        if not rag_pipeline.get(field):
            issues.append(f"rag_pipeline missing {field}")

    knowledge_artifacts = (
        memory_knowledge_plan.get("knowledge_artifacts")
        if isinstance(memory_knowledge_plan.get("knowledge_artifacts"), list)
        else []
    )
    if rag_enabled and not knowledge_artifacts:
        issues.append("knowledge_artifacts must not be empty when RAG is enabled")
    for artifact in knowledge_artifacts:
        if not isinstance(artifact, dict):
            issues.append("knowledge_artifacts entries must be objects")
            continue
        artifact_key = str(artifact.get("artifact_key") or "")
        location_hint = str(artifact.get("location_hint") or "")
        for field in ("title", "owner", "sensitivity", "source_version", "reason_to_index", "permissions", "refresh_triggers", "expiration_policy"):
            if not artifact.get(field):
                issues.append(f"knowledge artifact {artifact_key} missing {field}")
        if _looks_like_internal_location(location_hint):
            issues.append(f"knowledge artifact {artifact_key} location_hint must be portable")
        if artifact.get("indexing_required") is True:
            if artifact.get("ingestion_capability_ref") != "capability_document_ingestion":
                issues.append(f"knowledge artifact {artifact_key} must reference capability_document_ingestion")
            if artifact.get("retrieval_capability_ref") != "capability_knowledge_retrieval":
                issues.append(f"knowledge artifact {artifact_key} must reference capability_knowledge_retrieval")

    context_policy = (
        memory_knowledge_plan.get("context_window_policy")
        if isinstance(memory_knowledge_plan.get("context_window_policy"), dict)
        else {}
    )
    if int(context_policy.get("max_context_utilization_percent") or 0) > 85:
        issues.append("context_window_policy.max_context_utilization_percent must be <= 85")
    for field in ("short_term_budget_refs", "compaction_trigger", "anti_redundancy_rules", "retrieval_context_policy", "pagination_policy", "artifact_reference_policy"):
        if not context_policy.get(field):
            issues.append(f"context_window_policy missing {field}")

    if not payload.get("compatibility"):
        issues.append("compatibility targets must not be empty")
    if not payload.get("conformance"):
        issues.append("conformance rules must not be empty")

    migration = payload.get("migration") if isinstance(payload.get("migration"), dict) else {}
    source_checksum = str(migration.get("source_checksum_sha256") or "")
    if len(source_checksum) != 64:
        issues.append("migration.source_checksum_sha256 must be a sha256 hex digest")

    # Full document checksum is useful for independent consumers that do not trust transport headers.
    if len(_stable_checksum(payload)) != 64:
        issues.append("document checksum could not be calculated")

    return issues


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python acp_v2_reference_consumer.py path/to/agent-construction-package.v2.json", file=sys.stderr)
        return 2

    payload = _load_json(Path(argv[1]))
    issues = validate_acp_v2(payload)
    if issues:
        for issue in issues:
            print(f"ERROR: {issue}", file=sys.stderr)
        return 1

    print("ACP v2 portable contract accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
