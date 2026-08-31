from __future__ import annotations

from app.models import ContractModel, PydanticField


class StageAgentPolicy(ContractModel):
    stage: str
    allowed_action_keys: list[str] = PydanticField(default_factory=list)
    max_iterations: int = 6
    max_llm_calls: int = 4
    max_total_ms: int = 120_000
    deferred_domains: list[str] = PydanticField(default_factory=list)
    quality_gate_enabled: bool = True
    quality_threshold: float = 0.85
    min_repair_cycles: int = 2
    max_quality_repair_cycles: int = 2
    language_gate: bool = True
    schema_gate: bool = True
    allow_free_delegation_without_quality_penalty: bool = True

    def allows(self, action_key: str) -> bool:
        return action_key in self.allowed_action_keys


_COMMON_ACTIONS = [
    "retrieve_context",
    "invoke_capability",
    "invoke_critique",
    "infer_gap_resolutions",
    "run_validator",
    "repair_structured_output",
    "create_attention_decision",
    "persist_stage_artifact",
    "finish_stage",
    "checkpoint",
]


STAGE_AGENT_POLICIES: dict[str, StageAgentPolicy] = {
    "discover": StageAgentPolicy(
        stage="discover",
        allowed_action_keys=[*_COMMON_ACTIONS],
        max_iterations=10,
        max_llm_calls=3,
        deferred_domains=["tools", "memory", "framework", "database", "deployment", "credentials", "acp"],
    ),
    "define": StageAgentPolicy(
        stage="define",
        allowed_action_keys=[*_COMMON_ACTIONS, "resume_from_checkpoint"],
        max_iterations=10,
        max_llm_calls=4,
        max_total_ms=300_000,
        deferred_domains=["infrastructure", "deployment", "credentials", "physical_tool_contracts", "acp"],
    ),
    "design": StageAgentPolicy(
        stage="design",
        allowed_action_keys=[*_COMMON_ACTIONS, "resume_from_checkpoint"],
        max_iterations=14,
        max_llm_calls=6,
        max_total_ms=420_000,
        deferred_domains=["deployment", "credentials", "environment_configuration", "physical_tool_contracts"],
    ),
    "tools": StageAgentPolicy(
        stage="tools",
        allowed_action_keys=[*_COMMON_ACTIONS, "raise_cross_stage_remediation", "resume_from_checkpoint"],
        max_iterations=10,
        max_llm_calls=5,
        max_total_ms=240_000,
        deferred_domains=["credentials", "deployment_environment", "external_side_effect_activation"],
    ),
    "memory": StageAgentPolicy(
        stage="memory",
        allowed_action_keys=[*_COMMON_ACTIONS, "raise_cross_stage_remediation", "resume_from_checkpoint"],
        max_iterations=14,
        max_llm_calls=6,
        max_total_ms=300_000,
        deferred_domains=["vector_store_provisioning", "credentials", "deployment_environment"],
    ),
    "validate": StageAgentPolicy(stage="validate", allowed_action_keys=[*_COMMON_ACTIONS, "resume_from_checkpoint"], max_iterations=10),
    "estimate": StageAgentPolicy(stage="estimate", allowed_action_keys=[*_COMMON_ACTIONS, "resume_from_checkpoint"], max_iterations=10),
    "package": StageAgentPolicy(stage="package", allowed_action_keys=[*_COMMON_ACTIONS, "resume_from_checkpoint"], max_iterations=10),
}


def get_stage_agent_policy(stage: str) -> StageAgentPolicy:
    normalized = (stage or "").strip().lower()
    return STAGE_AGENT_POLICIES.get(normalized, StageAgentPolicy(stage=normalized, allowed_action_keys=[]))
