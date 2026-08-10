from __future__ import annotations

from uuid import uuid4

from app.models import ShortTermMemoryRuntimeState, ShortTermMemoryState, ShortTermMemoryStateNamespace
from app.services.agent_memory_policy import AgentMemoryPolicyService, branch_namespace_for_key


def test_short_term_memory_view_filters_namespaces_by_role() -> None:
    service = AgentMemoryPolicyService()
    memory_state = ShortTermMemoryState(
        active_stage="design",
        active_goal="Construir el blueprint con memoria gobernada.",
        namespaces=[
            ShortTermMemoryStateNamespace(
                namespace="session.short_term.discovery",
                read_roles=["planner", "memory", "retrieval"],
                write_roles=["discovery_skill", "memory"],
            ),
            ShortTermMemoryStateNamespace(
                namespace="session.short_term.plan",
                read_roles=["planner", "executor", "evaluator", "memory"],
                write_roles=["blueprint_generation_skill", "memory"],
            ),
            ShortTermMemoryStateNamespace(
                namespace="session.short_term.execution",
                read_roles=["executor", "recovery", "memory"],
                write_roles=["executor", "tool_use", "memory"],
            ),
            ShortTermMemoryStateNamespace(
                namespace="session.short_term.retrieval",
                read_roles=["retrieval", "executor", "memory"],
                write_roles=["retrieval", "memory"],
            ),
            ShortTermMemoryStateNamespace(
                namespace="session.short_term.summary_cache",
                read_roles=["planner", "executor", "memory", "retrieval"],
                write_roles=["memory"],
            ),
        ],
    )

    view = service.short_term_memory_view(memory_state, role="planner")

    assert [item.namespace for item in view.visible_namespaces] == [
        "session.short_term.discovery",
        "session.short_term.plan",
        "session.short_term.summary_cache",
    ]
    assert view.hidden_namespace_keys == (
        "session.short_term.execution",
        "session.short_term.retrieval",
    )


def test_write_access_restricts_specialists_to_branch_slot_and_blocks_knowledge_namespaces() -> None:
    service = AgentMemoryPolicyService()
    active_branch_key = "subagent_run:artifact-1"
    active_namespace = branch_namespace_for_key(active_branch_key)
    other_namespace = branch_namespace_for_key("subagent_run:artifact-2")
    runtime_state = ShortTermMemoryRuntimeState(
        session_id=uuid4(),
        active_branch_key=active_branch_key,
        memory=ShortTermMemoryState(
            active_stage="design",
            active_goal="Resolver findings del especialista.",
            namespaces=[
                ShortTermMemoryStateNamespace(
                    namespace=active_namespace,
                    read_roles=["supervisor", "specialist", "recovery", "memory"],
                    write_roles=["supervisor", "specialist", "memory"],
                ),
                ShortTermMemoryStateNamespace(
                    namespace=other_namespace,
                    read_roles=["supervisor", "specialist", "recovery", "memory"],
                    write_roles=["supervisor", "specialist", "memory"],
                ),
            ],
        ),
    )

    allowed = service.evaluate_write_access(
        runtime_state,
        role="artifact_specialist",
        namespace=active_namespace,
        branch_key=active_branch_key,
    )
    denied_branch = service.evaluate_write_access(
        runtime_state,
        role="artifact_specialist",
        namespace=other_namespace,
        branch_key=active_branch_key,
    )
    denied_knowledge = service.evaluate_write_access(
        runtime_state,
        role="planner",
        namespace="knowledge.canonical_docs",
    )

    assert allowed.allowed is True
    assert allowed.reason == "role_authorized_by_namespace_contract"
    assert denied_branch.allowed is False
    assert denied_branch.reason == "specialist_branch_slot_mismatch"
    assert denied_knowledge.allowed is False
    assert denied_knowledge.reason == "knowledge_namespaces_are_ingestion_only"
