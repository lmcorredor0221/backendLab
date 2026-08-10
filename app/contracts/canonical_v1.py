from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import Field as PydanticField, ValidationError, field_validator

from app.models import (
    AgenticEstimate,
    ApprovalStatus,
    ArtifactStatus,
    ConfidenceBreakdown,
    ContractModel,
    DecisionTraceEntry,
    EstimationAnalysisArtifact,
    EstimationDeterministicInputs,
    EstimationRunEntry,
    PatternCatalogEntry,
    ReviewState,
    TraditionalEstimate,
)


class CanonicalValidationIssue(ContractModel):
    code: str
    message: str
    path: str


class CanonicalProvenanceEntry(ContractModel):
    target_path: str
    source_paths: list[str] = PydanticField(default_factory=list)
    note: str = ""


class CanonicalContractBase(ContractModel):
    schema_version: str
    source_session_id: UUID
    generated_at: datetime
    source_blueprint_version: int | None = None
    provenance: list[CanonicalProvenanceEntry] = PydanticField(default_factory=list)


class BlueprintIdentity(ContractModel):
    title: str
    case_type: str
    current_stage: str
    blueprint_version_number: int | None = None

    @field_validator("title", "case_type", "current_stage")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Field cannot be empty")
        return normalized


class BlueprintPurpose(ContractModel):
    problem_statement: str
    primary_user: str
    desired_outcome: str
    value_statement: str

    @field_validator("problem_statement", "primary_user", "desired_outcome")
    @classmethod
    def validate_core_purpose_fields(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Field cannot be empty")
        return normalized


class BlueprintScope(ContractModel):
    in_scope: list[str] = PydanticField(default_factory=list)
    out_of_scope: list[str] = PydanticField(default_factory=list)
    constraints: list[str] = PydanticField(default_factory=list)
    non_delegable_decisions: list[str] = PydanticField(default_factory=list)


class CanonicalDependency(ContractModel):
    key: str
    kind: str
    description: str


class CanonicalOpenQuestion(ContractModel):
    key: str
    question: str
    owner: str = ""


class ApprovalGateSummary(ContractModel):
    gate_key: str
    title: str
    status: ApprovalStatus
    requested_in_stage: str
    rationale: str = ""


class RiskEntry(ContractModel):
    category: str
    severity: str
    summary: str
    mitigation: str = ""
    status: str = ""


class SuccessCriterion(ContractModel):
    key: str
    description: str
    source: str = ""


class BehaviorState(ContractModel):
    name: str
    actor: str
    objective: str
    outputs: list[str] = PydanticField(default_factory=list)
    fallback: str = ""
    requires_approval: bool = False


class MultiAgentPermissionBoundaryV1(ContractModel):
    allowed_tools: list[str] = PydanticField(default_factory=list)
    required_approvals: list[str] = PydanticField(default_factory=list)
    side_effect_policy: str = ""
    escalation_policy: str = ""


class MultiAgentRoleContractV1(ContractModel):
    agent_key: str
    role: str
    purpose: str
    runtime_mode: str = ""
    permissions: MultiAgentPermissionBoundaryV1 = PydanticField(default_factory=MultiAgentPermissionBoundaryV1)
    input_contracts: list[str] = PydanticField(default_factory=list)
    output_contracts: list[str] = PydanticField(default_factory=list)
    success_signals: list[str] = PydanticField(default_factory=list)
    failure_mode: str = ""
    retry_strategy: str = ""
    timeout_policy: str = ""
    isolation_boundary: str = ""


class MultiAgentMessageContractV1(ContractModel):
    message_key: str
    from_agent: str
    to_agent: str
    purpose: str
    payload_schema: dict[str, Any] = PydanticField(default_factory=dict)
    required_fields: list[str] = PydanticField(default_factory=list)
    idempotency_strategy: str = ""
    timeout_policy: str = ""
    retry_strategy: str = ""
    failure_behavior: str = ""


class MultiAgentHandoffContractV1(ContractModel):
    handoff_key: str
    from_agent: str
    to_agent: str
    trigger: str
    ownership_transfer: str = ""
    required_artifacts: list[str] = PydanticField(default_factory=list)
    success_criteria: list[str] = PydanticField(default_factory=list)
    failure_behavior: str = ""
    audit_trail: list[str] = PydanticField(default_factory=list)


class MultiAgentSharedStateContractV1(ContractModel):
    state_key: str
    purpose: str
    owner_agent: str
    readers: list[str] = PydanticField(default_factory=list)
    writers: list[str] = PydanticField(default_factory=list)
    payload_schema: dict[str, Any] = PydanticField(default_factory=dict)
    update_policy: str = ""
    consistency_policy: str = ""
    rollback_strategy: str = ""


class MultiAgentExecutionBudgetV1(ContractModel):
    latency_budget_ms: int = 0
    max_parallel_agents: int = 0
    max_retries_per_handoff: int = 0
    max_tool_calls_per_agent: int = 0
    cost_budget: str = ""


class MultiAgentBenchmarkMetricV1(ContractModel):
    metric_key: str
    label: str
    direction: Literal["higher_is_better", "lower_is_better"]
    unit: str = ""
    baseline_single_agent: float = 0
    projected_multi_agent: float = 0
    improvement_delta: float = 0
    rationale: str = ""


class MultiAgentBenchmarkV1(ContractModel):
    go_decision: Literal["hold", "go"] = "hold"
    explicit_go_reason: str = ""
    measurable_single_agent_limitation: str = ""
    limitation_signals: list[str] = PydanticField(default_factory=list)
    metrics: list[MultiAgentBenchmarkMetricV1] = PydanticField(default_factory=list)
    success_gate: str = ""
    latency_budget: str = ""
    cost_budget: str = ""


class MultiAgentTopologyV1(ContractModel):
    declared_pattern: str
    runtime_pattern: str
    support_state: Literal["planned_only", "supported"]
    activation_mode: str = ""
    benchmark: MultiAgentBenchmarkV1 | None = None
    agent_contracts: list[MultiAgentRoleContractV1] = PydanticField(default_factory=list)
    message_contracts: list[MultiAgentMessageContractV1] = PydanticField(default_factory=list)
    handoff_contracts: list[MultiAgentHandoffContractV1] = PydanticField(default_factory=list)
    shared_state_contracts: list[MultiAgentSharedStateContractV1] = PydanticField(default_factory=list)
    execution_budget: MultiAgentExecutionBudgetV1 | None = None
    failure_isolation_rules: list[str] = PydanticField(default_factory=list)


class BehaviorSpecV1(CanonicalContractBase):
    schema_version: Literal["behavior-spec.v1"] = "behavior-spec.v1"
    execution_pattern: str
    reasoning_pattern: str
    selected_workflow_template_key: str = ""
    checkpoint_policy: str = ""
    retry_strategy: str = ""
    compensation_strategy: str = ""
    approval_pause: str = ""
    timeout_policy: str = ""
    states: list[BehaviorState] = PydanticField(default_factory=list)
    termination_criteria: list[str] = PydanticField(default_factory=list)
    outputs: list[str] = PydanticField(default_factory=list)
    required_approvals: list[str] = PydanticField(default_factory=list)
    multi_agent_topology: MultiAgentTopologyV1 | None = None

    @field_validator("execution_pattern", "reasoning_pattern")
    @classmethod
    def validate_behavior_fields(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Field cannot be empty")
        return normalized


class ToolSchemaField(ContractModel):
    type: str = "string"
    description: str = ""


class ToolSchemaShape(ContractModel):
    type: Literal["object"] = "object"
    properties: dict[str, ToolSchemaField] = PydanticField(default_factory=dict)
    required: list[str] = PydanticField(default_factory=list)


class ToolContractV1(CanonicalContractBase):
    schema_version: Literal["tool-contract.v1"] = "tool-contract.v1"
    name: str
    purpose: str
    owner: str
    archetype: str
    integration_kind: str
    endpoint_reference: str
    auth_reference: str
    risk_level: str
    execution_mode: str
    requires_approval: bool = False
    approval_reason: str = ""
    approval_policy: str = ""
    side_effects: bool = False
    idempotent: bool = False
    idempotency_strategy: str = ""
    input_schema: ToolSchemaShape = PydanticField(default_factory=ToolSchemaShape)
    output_schema: ToolSchemaShape = PydanticField(default_factory=ToolSchemaShape)
    validations: list[str] = PydanticField(default_factory=list)
    typed_errors: list[str] = PydanticField(default_factory=list)
    retry_strategy: str = ""
    compensation_strategy: str = ""
    failure_mode: str = ""
    permissions: list[str] = PydanticField(default_factory=list)
    scopes: list[str] = PydanticField(default_factory=list)
    sensitive_data: list[str] = PydanticField(default_factory=list)
    audit_rules: list[str] = PydanticField(default_factory=list)
    rate_limit_policy: str = ""
    timeout_policy: str = ""
    contract_review_state: str = ""

    @field_validator(
        "name",
        "purpose",
        "owner",
        "archetype",
        "integration_kind",
        "endpoint_reference",
        "auth_reference",
        "risk_level",
        "execution_mode",
        "approval_policy",
        "idempotency_strategy",
        "rate_limit_policy",
        "timeout_policy",
        "contract_review_state",
    )
    @classmethod
    def validate_tool_fields(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Field cannot be empty")
        return normalized


class MemoryContextBudgetV1(ContractModel):
    role: str
    max_tokens: int = 0
    max_items: int = 0
    max_chars: int = 0
    compaction_trigger: str = ""
    overflow_policy: str = ""

    @field_validator("role")
    @classmethod
    def validate_budget_role(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Field cannot be empty")
        return normalized

    @field_validator("max_tokens", "max_items", "max_chars")
    @classmethod
    def validate_non_negative_budget(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Budget values must be non-negative")
        return value


class ShortTermMemoryRefV1(ContractModel):
    key: str
    kind: str
    stage: str = ""
    source: str = ""
    summary: str = ""
    status: str = ""
    created_at: str = ""
    blueprint_version_number: int | None = None
    evidence_paths: list[str] = PydanticField(default_factory=list)

    @field_validator("key", "kind")
    @classmethod
    def validate_short_term_ref_fields(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Field cannot be empty")
        return normalized


class ShortTermMemoryNamespaceV1(ContractModel):
    namespace: str
    summary: str = ""
    ref_keys: list[str] = PydanticField(default_factory=list)
    freshness: str = ""
    read_roles: list[str] = PydanticField(default_factory=list)
    write_roles: list[str] = PydanticField(default_factory=list)

    @field_validator("namespace")
    @classmethod
    def validate_namespace(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Field cannot be empty")
        return normalized


class ShortTermMemoryCompactionV1(ContractModel):
    summary_policy: str = ""
    invalidation_policy: str = ""
    eviction_policy: str = ""
    last_compacted_at: str = ""


class MemoryPolicyV1(CanonicalContractBase):
    schema_version: Literal["memory-policy.v1"] = "memory-policy.v1"
    strategy: str
    storage_layers: list[str] = PydanticField(default_factory=list)
    context_budgets: list[MemoryContextBudgetV1] = PydanticField(default_factory=list)
    write_policy: str = ""
    retrieval_policy: str = ""
    retrieval_scopes: list[str] = PydanticField(default_factory=list)
    summary_policy: str = ""
    invalidation_policy: str = ""
    review_trigger: str = ""
    goal_drift_guard: str = ""
    retention_policy: str = ""
    ttl_policy: str = ""
    workspace_scope: str = ""
    agent_scope: str = ""
    grounding_policy: dict[str, str] = PydanticField(default_factory=dict)
    sensitivity_rules: list[str] = PydanticField(default_factory=list)
    checkpoints_required: bool = False

    @field_validator("strategy")
    @classmethod
    def validate_memory_strategy(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Field cannot be empty")
        return normalized


class ShortTermMemoryV1(CanonicalContractBase):
    schema_version: Literal["short-term-memory.v1"] = "short-term-memory.v1"
    active_stage: str
    active_goal: str
    current_focus: str = ""
    pending_approvals: list[str] = PydanticField(default_factory=list)
    open_handoffs: list[str] = PydanticField(default_factory=list)
    recent_decisions: list[str] = PydanticField(default_factory=list)
    namespaces: list[ShortTermMemoryNamespaceV1] = PydanticField(default_factory=list)
    checkpoint_refs: list[ShortTermMemoryRefV1] = PydanticField(default_factory=list)
    artifact_refs: list[ShortTermMemoryRefV1] = PydanticField(default_factory=list)
    skill_run_refs: list[ShortTermMemoryRefV1] = PydanticField(default_factory=list)
    branch_refs: list[ShortTermMemoryRefV1] = PydanticField(default_factory=list)
    compaction: ShortTermMemoryCompactionV1 = PydanticField(default_factory=ShortTermMemoryCompactionV1)

    @field_validator("active_stage", "active_goal")
    @classmethod
    def validate_short_term_memory_fields(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Field cannot be empty")
        return normalized


class KnowledgeSourceRef(ContractModel):
    key: str
    title: str
    source_type: str
    uri: str = ""
    owner: str = ""
    sensitivity: str = ""
    license: str = ""
    description: str = ""
    source_version: str = ""
    lineage_key: str = ""


class KnowledgeIngestionPolicyV1(ContractModel):
    parser: str = ""
    chunking_policy: str = ""
    metadata_fields: list[str] = PydanticField(default_factory=list)
    include_filters: list[str] = PydanticField(default_factory=list)
    exclude_filters: list[str] = PydanticField(default_factory=list)


class KnowledgeEmbeddingPolicyV1(ContractModel):
    provider: str = ""
    model: str = ""
    dimensions: int = 0
    version: str = ""


class KnowledgeRetrievalPolicyV1(ContractModel):
    top_k: int = 0
    filters: list[str] = PydanticField(default_factory=list)
    search_mode: str = ""
    reranking_policy: str = ""
    fallback_behavior: str = ""


class KnowledgeRefreshPolicyV1(ContractModel):
    frequency: str = ""
    triggers: list[str] = PydanticField(default_factory=list)
    expiration_policy: str = ""
    deletion_policy: str = ""


class KnowledgeContractV1(CanonicalContractBase):
    schema_version: Literal["knowledge-contract.v1"] = "knowledge-contract.v1"
    enabled: bool = False
    mode: str
    sources: list[KnowledgeSourceRef] = PydanticField(default_factory=list)
    source_lineage: list[str] = PydanticField(default_factory=list)
    ingestion_policy: KnowledgeIngestionPolicyV1 | None = None
    embedding_policy: KnowledgeEmbeddingPolicyV1 | None = None
    retrieval_policy: KnowledgeRetrievalPolicyV1 | None = None
    refresh_policy: KnowledgeRefreshPolicyV1 | None = None
    grounding_policy: dict[str, str] = PydanticField(default_factory=dict)
    sensitivity_rules: list[str] = PydanticField(default_factory=list)
    open_questions: list[str] = PydanticField(default_factory=list)

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Field cannot be empty")
        return normalized


class KnowledgeManifestSourceV1(ContractModel):
    key: str
    title: str
    uri: str
    source_type: str
    authority_level: str = ""
    memory_usage: str = ""
    stage_affinity: list[str] = PydanticField(default_factory=list)
    agent_affinity: list[str] = PydanticField(default_factory=list)
    owner: str = ""
    source_version: str = ""
    required: bool = False
    summary: str = ""

    @field_validator("key", "title", "uri", "source_type")
    @classmethod
    def validate_manifest_source_fields(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Field cannot be empty")
        return normalized


class KnowledgeManifestV1(CanonicalContractBase):
    schema_version: Literal["knowledge-manifest.v1"] = "knowledge-manifest.v1"
    knowledge_backend_mode: str
    operating_summary: str = ""
    retrieval_scopes: list[str] = PydanticField(default_factory=list)
    required_sources: list[KnowledgeManifestSourceV1] = PydanticField(default_factory=list)
    candidate_sources: list[KnowledgeManifestSourceV1] = PydanticField(default_factory=list)
    selection_policy: str = ""
    fallback_policy: str = ""

    @field_validator("knowledge_backend_mode")
    @classmethod
    def validate_knowledge_backend_mode(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Field cannot be empty")
        return normalized


class LLMFunctionPolicy(ContractModel):
    function_key: str
    role: str
    provider: str
    model: str
    intent: str
    reasoning_effort: str = "medium"
    context_sources: list[str] = PydanticField(default_factory=list)
    tool_availability: list[str] = PydanticField(default_factory=list)
    fallback_model: str = ""
    max_tokens: int = 0


class LLMPolicyV1(CanonicalContractBase):
    schema_version: Literal["llm-policy.v1"] = "llm-policy.v1"
    provider: str
    fast_model: str
    reasoning_model: str
    fallback_model: str
    functions: list[LLMFunctionPolicy] = PydanticField(default_factory=list)
    context_policy: str
    sampling_policy: str
    fallback_policy: str
    circuit_breaker_policy: str
    budget_policy: str
    output_validation_policy: str
    log_redaction_policy: str

    @field_validator(
        "provider",
        "fast_model",
        "reasoning_model",
        "fallback_model",
        "context_policy",
        "sampling_policy",
        "fallback_policy",
        "circuit_breaker_policy",
        "budget_policy",
        "output_validation_policy",
        "log_redaction_policy",
    )
    @classmethod
    def validate_policy_fields(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Field cannot be empty")
        return normalized


class HeuristicDecisionFact(ContractModel):
    key: str
    value: str
    source: str


class HeuristicDecisionV1(CanonicalContractBase):
    schema_version: Literal["heuristic-decision.v1"] = "heuristic-decision.v1"
    decision_summary: str
    decision_trace: list[DecisionTraceEntry] = PydanticField(default_factory=list)
    candidate_catalog: list[PatternCatalogEntry] = PydanticField(default_factory=list)
    facts: list[HeuristicDecisionFact] = PydanticField(default_factory=list)
    recommended_prompts: list[str] = PydanticField(default_factory=list)
    review_notes: list[str] = PydanticField(default_factory=list)

    @field_validator("decision_summary")
    @classmethod
    def validate_decision_summary(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Field cannot be empty")
        return normalized


class EvaluationCaseV1(ContractModel):
    key: str
    title: str
    category: str
    scenario: str
    expected_result: str


class EvaluationPackV1(CanonicalContractBase):
    schema_version: Literal["evaluation-pack.v1"] = "evaluation-pack.v1"
    readiness_state: ReviewState
    scores: dict[str, int] = PydanticField(default_factory=dict)
    blocking_issues: list[str] = PydanticField(default_factory=list)
    recommendations: list[str] = PydanticField(default_factory=list)
    cases: list[EvaluationCaseV1] = PydanticField(default_factory=list)
    dataset_version_number: int | None = None
    rubric_version_number: int | None = None
    latest_run_status: ArtifactStatus | None = None
    acceptance_cases: list[EvaluationCaseV1] = PydanticField(default_factory=list)


class PromptVariable(ContractModel):
    name: str
    description: str
    source_paths: list[str] = PydanticField(default_factory=list)


class PromptArtifactV1(ContractModel):
    prompt_key: str
    role: str
    title: str
    content: str
    variables: list[PromptVariable] = PydanticField(default_factory=list)
    context_sources: list[str] = PydanticField(default_factory=list)
    output_schema: dict[str, Any] = PydanticField(default_factory=dict)
    guardrails: list[str] = PydanticField(default_factory=list)
    stop_conditions: list[str] = PydanticField(default_factory=list)
    fallback: str = ""
    evaluation_case_keys: list[str] = PydanticField(default_factory=list)
    input_contracts: list[str] = PydanticField(default_factory=list)
    provenance: list[CanonicalProvenanceEntry] = PydanticField(default_factory=list)

    @field_validator("prompt_key", "role", "title", "content")
    @classmethod
    def validate_prompt_fields(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Field cannot be empty")
        return normalized


class PromptPackOrigin(ContractModel):
    blueprint_core_version: str
    behavior_spec_version: str
    llm_policy_version: str
    heuristic_decision_version: str
    input_hash: str


class PromptPackV1(CanonicalContractBase):
    schema_version: Literal["prompt-pack.v1"] = "prompt-pack.v1"
    origin: PromptPackOrigin
    system_prompt: PromptArtifactV1
    planner_prompt: PromptArtifactV1
    executor_prompt: PromptArtifactV1
    evaluator_prompt: PromptArtifactV1
    tool_use_prompt: PromptArtifactV1 | None = None
    memory_prompt: PromptArtifactV1 | None = None
    retrieval_prompt: PromptArtifactV1 | None = None
    recovery_prompt: PromptArtifactV1 | None = None
    agent_role_prompts: list[PromptArtifactV1] = PydanticField(default_factory=list)
    handoff_prompts: list[PromptArtifactV1] = PydanticField(default_factory=list)


class ConstructionComponent(ContractModel):
    key: str
    label: str
    role: str
    status: ReviewState
    summary: str


class ConstructionFileManifestEntry(ContractModel):
    path: str
    kind: str
    summary: str
    source_contract: str
    generated_from: list[str] = PydanticField(default_factory=list)


class ReadinessGapEntry(ContractModel):
    code: str
    severity: str
    summary: str
    remediation: str = ""


class ConstructionReadinessV1(ContractModel):
    status: ReviewState
    can_build: bool
    blocking_issues: list[str] = PydanticField(default_factory=list)
    warnings: list[str] = PydanticField(default_factory=list)
    remediation_notes: list[str] = PydanticField(default_factory=list)


class ContractReference(ContractModel):
    contract_kind: str
    schema_version: str
    source_blueprint_version: int | None = None


class ConstructionPackV1(CanonicalContractBase):
    schema_version: Literal["construction-pack.v1"] = "construction-pack.v1"
    blueprint_ref: ContractReference
    components: list[ConstructionComponent] = PydanticField(default_factory=list)
    topology: dict[str, Any] = PydanticField(default_factory=dict)
    multi_agent_benchmark: MultiAgentBenchmarkV1 | None = None
    behavior_spec: BehaviorSpecV1
    heuristic_decision: HeuristicDecisionV1
    prompt_pack: PromptPackV1
    llm_policy: LLMPolicyV1
    tool_contracts: list[ToolContractV1] = PydanticField(default_factory=list)
    memory_policy: MemoryPolicyV1
    knowledge_contract: KnowledgeContractV1
    evaluation_pack: EvaluationPackV1
    acceptance_cases: list[EvaluationCaseV1] = PydanticField(default_factory=list)
    file_manifest: list[ConstructionFileManifestEntry] = PydanticField(default_factory=list)
    readiness: ConstructionReadinessV1
    gaps: list[ReadinessGapEntry] = PydanticField(default_factory=list)
    remediation_notes: list[str] = PydanticField(default_factory=list)


class AcpV2ProducerMetadata(ContractModel):
    producer_name: str
    producer_contract_version: Literal["agent-construction-package.v2"] = "agent-construction-package.v2"
    generated_from_contracts: list[str]
    separated_from_system_spec: bool = True
    notes: list[str]


class AcpV2ManifestContractEntry(ContractModel):
    contract_key: str
    schema_version: str
    relative_path: str
    checksum_sha256: str
    required: bool = True


class AcpV2PortableManifest(ContractModel):
    package_id: str
    contract_version: Literal["agent-construction-package.v2"] = "agent-construction-package.v2"
    manifest_version: Literal["acp-portable-manifest.v1"] = "acp-portable-manifest.v1"
    created_at: datetime
    checksum_algorithm: Literal["sha256"] = "sha256"
    compatibility_targets: list[str]
    contracts: list[AcpV2ManifestContractEntry]


class AcpV2MigrationInfo(ContractModel):
    from_schema_version: str
    source_checksum_sha256: str
    migration_strategy: str
    breaking_changes: list[str]
    compatibility_notes: list[str]


class AcpV2BuildStep(ContractModel):
    step_key: str
    title: str
    objective: str
    depends_on: list[str]
    inputs: list[str]
    outputs: list[str]
    actions: list[str]
    validation: list[str]


class AcpV2BuildPlan(ContractModel):
    entrypoint: str
    steps: list[AcpV2BuildStep]
    completion_criteria: list[str]


class AcpV2RuntimeAgent(ContractModel):
    agent_key: str
    role: str
    goal: str
    runtime_mode: str = ""
    inputs: list[str]
    outputs: list[str]
    tools: list[str]
    memory_refs: list[str]
    handoff_targets: list[str]
    success_signals: list[str]
    failure_mode: str = ""


class AcpV2AgentRuntime(ContractModel):
    runtime_model: str
    orchestration_pattern: str
    state_machine: list[dict[str, Any]]
    routing_rules: list[dict[str, Any]]
    agents: list[AcpV2RuntimeAgent]
    failure_modes: list[str]
    execution_budget: dict[str, Any] = PydanticField(default_factory=dict)


class AcpV2WorkflowNode(ContractModel):
    node_key: str
    title: str
    workflow_role: Literal["construction", "runtime", "human_decision"]
    objective: str
    actor: str
    inputs: list[str]
    outputs: list[str]
    portable_state: str
    checkpoint_ref: str = ""
    decision_refs: list[str] = PydanticField(default_factory=list)
    timeout_policy: str = ""
    retry_policy: str = ""
    context_refs: list[str] = PydanticField(default_factory=list)


class AcpV2WorkflowTransition(ContractModel):
    transition_key: str
    from_node: str
    to_node: str
    condition: str
    routing_rule: str = ""
    requires_decision: bool = False
    checkpoint_ref: str = ""
    failure_behavior: str = ""


class AcpV2WorkflowSpec(ContractModel):
    workflow_key: str
    workflow_type: Literal["construction", "runtime_operational", "human_decision_resolution"]
    topology: Literal["sequential", "hierarchical", "event_driven", "consensus", "mixed"]
    entry_node: str
    terminal_nodes: list[str]
    nodes: list[AcpV2WorkflowNode]
    transitions: list[AcpV2WorkflowTransition]
    portable_state_policy: str
    handoff_contract_refs: list[str] = PydanticField(default_factory=list)


class AcpV2CheckpointSpec(ContractModel):
    checkpoint_key: str
    title: str
    scope: Literal["construction", "runtime", "human_decision"]
    trigger: str
    required_artifacts: list[str]
    validation: list[str]
    resume_strategy: str
    storage_hint: str
    portable_ref: str


class AcpV2DecisionOption(ContractModel):
    option_key: str
    label: str
    description: str
    tradeoffs: list[str]
    recommended: bool = False


class AcpV2DecisionRegistryEntry(ContractModel):
    decision_key: str
    classification: Literal["mandatory", "optional", "deferable", "environment_dependent"]
    blocking_scope: Literal["package", "implementation", "none"]
    question: str
    context: str
    owner: str
    recommended_moment: str
    impact: str
    options: list[AcpV2DecisionOption]
    examples: list[str]
    source_ref: str


class AcpV2ImplementationDecision(ContractModel):
    decision_key: str
    decision_type: str
    question: str
    owner: str
    timing: str
    required: bool
    options: list[str]
    impact: str
    default_option: str = ""
    source_ref: str


class AcpV2ToolContractRef(ContractModel):
    tool_key: str
    display_name: str
    purpose: str
    requirement_level: Literal["required", "optional", "conditional", "replaceable", "not_recommended"]
    capability: str
    integration_kind: str
    auth_requirements: list[str]
    side_effects: bool
    idempotent: bool
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    validations: list[str]
    retry_strategy: str = ""
    compensation_strategy: str = ""
    source_ref: str


class AcpV2CapabilityContract(ContractModel):
    capability_key: str
    title: str
    description: str
    requirement_level: Literal["required", "optional", "replaceable", "not_recommended"]
    rationale: str
    consumers: list[str]
    abstract_inputs: dict[str, Any]
    abstract_outputs: dict[str, Any]
    required_permissions: list[str]
    side_effect_profile: str
    memory_refs: list[str]
    tool_refs: list[str]
    replacement_options: list[str]
    source_refs: list[str]


class AcpV2ToolBinding(ContractModel):
    binding_key: str
    capability_key: str
    tool_key: str
    binding_type: Literal["producer_internal_tool", "external_api", "abstract_contract", "runtime_adapter"]
    provider_boundary: Literal["producer_internal", "customer_external", "framework_runtime", "abstract"]
    requirement_level: Literal["required", "optional", "replaceable", "not_recommended"]
    replaceable: bool
    replacement_strategy: str
    external_contract_hint: str
    credentials_policy: str
    permissions: list[str]
    side_effects: bool
    idempotent: bool
    cost_profile: str
    risk_profile: str
    fallback_strategy: str
    source_ref: str


class AcpV2ToolRedundancy(ContractModel):
    redundancy_key: str
    capability_key: str
    tool_keys: list[str]
    severity: Literal["info", "warning", "blocking"]
    rationale: str
    recommendation: str


class AcpV2ToolIncompatibility(ContractModel):
    incompatibility_key: str
    tool_keys: list[str]
    severity: Literal["info", "warning", "blocking"]
    reason: str
    mitigation: str


class AcpV2ToolAnalysis(ContractModel):
    summary: str
    overprovisioning_policy: str
    minimal_tooling_policy: str
    redundancy_findings: list[AcpV2ToolRedundancy] = PydanticField(default_factory=list)
    incompatibility_findings: list[AcpV2ToolIncompatibility] = PydanticField(default_factory=list)
    not_recommended_tools: list[str] = PydanticField(default_factory=list)


class AcpV2MemoryStrategy(ContractModel):
    short_term: dict[str, Any]
    long_term: dict[str, Any]
    retrieval: dict[str, Any]
    context_budget: list[dict[str, Any]]
    persistence: dict[str, Any]
    source_refs: list[str]


class AcpV2MemoryNamespace(ContractModel):
    namespace_key: str
    memory_type: Literal["short_term", "long_term", "documentary_knowledge", "rag_index", "audit"]
    purpose: str
    scope: Literal["agent", "tenant", "workspace", "session_portable"]
    read_roles: list[str]
    write_roles: list[str]
    retention_policy: str
    compaction_policy: str
    privacy_policy: str
    freshness_policy: str
    portable_ref: str


class AcpV2KnowledgeArtifactRef(ContractModel):
    artifact_key: str
    title: str
    source_type: str
    location_hint: str
    owner: str
    sensitivity: str
    license: str
    source_version: str
    indexing_required: bool
    reason_to_index: str
    ingestion_capability_ref: str
    retrieval_capability_ref: str
    permissions: list[str]
    refresh_triggers: list[str]
    expiration_policy: str
    source_ref: str


class AcpV2RagCapabilityDependency(ContractModel):
    capability_key: str
    required: bool
    reason: str
    fallback: str


class AcpV2RagPipelineSpec(ContractModel):
    enabled: bool
    mode: str
    capability_dependencies: list[AcpV2RagCapabilityDependency]
    vector_store_decision_ref: str
    ingestion_policy: dict[str, Any]
    embedding_policy: dict[str, Any]
    retrieval_policy: dict[str, Any]
    refresh_policy: dict[str, Any]
    grounding_policy: dict[str, Any]
    citation_policy: str
    deletion_policy: str
    fallback_policy: str
    source_refs: list[str]


class AcpV2ContextWindowPolicy(ContractModel):
    max_context_utilization_percent: int = 85
    short_term_budget_refs: list[str]
    compaction_trigger: str
    anti_redundancy_rules: list[str]
    retrieval_context_policy: str
    pagination_policy: str
    artifact_reference_policy: str


class AcpV2MemoryKnowledgePlan(ContractModel):
    namespaces: list[AcpV2MemoryNamespace]
    knowledge_artifacts: list[AcpV2KnowledgeArtifactRef]
    rag_pipeline: AcpV2RagPipelineSpec
    context_window_policy: AcpV2ContextWindowPolicy
    capability_dependencies: list[str]
    source_refs: list[str]


class AcpV2KnowledgeSource(ContractModel):
    source_key: str
    title: str
    kind: str
    location_hint: str
    ingestion_required: bool
    freshness: str
    owner: str
    source_ref: str


class AcpV2PromptRef(ContractModel):
    prompt_key: str
    role: str
    title: str
    content: str
    required: bool
    usage: str
    context_sources: list[str]
    input_contracts: list[str]
    output_schema: dict[str, Any]
    guardrails: list[str]
    source_ref: str


class AcpV2TestAsset(ContractModel):
    test_key: str
    kind: str
    title: str
    scenario: str
    expected_result: str
    required: bool
    acceptance_criteria: list[str]
    source_ref: str


class AcpV2ConformanceRule(ContractModel):
    rule_key: str
    severity: Literal["info", "warning", "blocking"]
    requirement: str
    validation_method: str


class AcpV2CompatibilityRule(ContractModel):
    target: str
    support_level: str
    adapter_notes: list[str]
    unsupported_features: list[str]


class AcpV2RuntimeTarget(ContractModel):
    target_key: str
    label: str
    category: Literal["agentic_ide", "agent_framework", "orchestration_runtime", "custom_runtime"]
    recommendation_level: Literal["recommended", "compatible", "optional", "not_recommended"]
    required: bool = False
    rationale: str
    selection_criteria: list[str]
    prerequisites: list[str]
    tradeoffs: list[str]
    adapter_notes: list[str]
    source_ref: str


class AcpV2RuntimeTargetPolicy(ContractModel):
    recommended_runtime: list[str]
    required_runtime: list[str]
    selection_policy: str
    override_policy: str


class AcpV2TechnologyOption(ContractModel):
    option_key: str
    label: str
    recommendation_level: Literal["recommended", "compatible", "optional", "not_recommended"]
    rationale: str
    prerequisites: list[str]
    tradeoffs: list[str]
    examples: list[str]


class AcpV2TechnologyDecision(ContractModel):
    decision_key: str
    category: Literal[
        "language",
        "framework",
        "database",
        "vector_store",
        "hosting",
        "ci_cd",
        "observability",
    ]
    question: str
    required_for_package: bool = False
    required_for_implementation: bool = True
    selection_criteria: list[str]
    options: list[AcpV2TechnologyOption]
    default_guidance: str
    source_ref: str


class AcpV2DeploymentGuideStep(ContractModel):
    step_key: str
    title: str
    objective: str
    prerequisites: list[str]
    actions: list[str]
    validation: list[str]
    optional: bool = False


class AcpV2DeploymentGuide(ContractModel):
    guide_key: str
    mode: Literal["guidance_only"] = "guidance_only"
    required_script: bool = False
    deployment_decision_refs: list[str]
    environment_prerequisites: list[str]
    steps: list[AcpV2DeploymentGuideStep]
    rollback_guidance: list[str]
    security_considerations: list[str]
    observability_considerations: list[str]


class AgentConstructionPackageV2(CanonicalContractBase):
    schema_version: Literal["agent-construction-package.v2"] = "agent-construction-package.v2"
    producer_metadata: AcpV2ProducerMetadata
    portable_manifest: AcpV2PortableManifest
    migration: AcpV2MigrationInfo
    system_specification: dict[str, Any]
    build_plan: AcpV2BuildPlan
    agent_runtime: AcpV2AgentRuntime
    implementation_decisions: list[AcpV2ImplementationDecision]
    workflows: list[AcpV2WorkflowSpec]
    checkpoints: list[AcpV2CheckpointSpec]
    decision_registry: list[AcpV2DecisionRegistryEntry]
    runtime_target_policy: AcpV2RuntimeTargetPolicy
    runtime_targets: list[AcpV2RuntimeTarget]
    technology_decisions: list[AcpV2TechnologyDecision]
    deployment_guide: AcpV2DeploymentGuide
    capability_catalog: list[AcpV2CapabilityContract]
    tool_contracts: list[AcpV2ToolContractRef]
    tool_bindings: list[AcpV2ToolBinding]
    tool_analysis: AcpV2ToolAnalysis
    memory_strategy: AcpV2MemoryStrategy
    memory_knowledge_plan: AcpV2MemoryKnowledgePlan
    knowledge_sources: list[AcpV2KnowledgeSource]
    prompts: list[AcpV2PromptRef]
    tests: list[AcpV2TestAsset]
    conformance: list[AcpV2ConformanceRule]
    compatibility: list[AcpV2CompatibilityRule]


class EstimationSensitivityDriver(ContractModel):
    key: str
    summary: str
    impact: str


class EstimationPackV1(CanonicalContractBase):
    schema_version: Literal["estimation-pack.v1"] = "estimation-pack.v1"
    blueprint_ref: ContractReference
    maturity_stage: str
    traditional: TraditionalEstimate
    agentic: AgenticEstimate
    confidence: ConfidenceBreakdown
    base_confidence: ConfidenceBreakdown | None = None
    analysis: EstimationAnalysisArtifact | None = None
    deterministic_inputs: EstimationDeterministicInputs | None = None
    assumptions: list[str] = PydanticField(default_factory=list)
    risk_drivers: list[str] = PydanticField(default_factory=list)
    sensitivity_drivers: list[EstimationSensitivityDriver] = PydanticField(default_factory=list)
    roi_summary: str
    estimation_runs: list[EstimationRunEntry] = PydanticField(default_factory=list)
    actuals_count: int = 0

    @field_validator("maturity_stage", "roi_summary")
    @classmethod
    def validate_estimation_fields(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Field cannot be empty")
        return normalized


class TestPackFixtureRef(ContractModel):
    key: str
    contract_key: str
    relative_path: str
    valid: bool = True
    summary: str = ""


class TestPackCommandV1(ContractModel):
    key: str
    title: str
    kind: str
    command: str
    workdir: str = "."
    expected_exit_code: int = 0


class TestPackMutationCaseV1(ContractModel):
    key: str
    contract_key: str
    path: str
    mutation: str
    expected_issue_code: str
    expected_issue_path: str
    blocks_readiness: bool = True


class TestPackPromptEvaluationCaseV1(ContractModel):
    key: str
    prompt_key: str
    mode: str
    failure_mode: str = ""
    expected_substrings: list[str] = PydanticField(default_factory=list)
    forbidden_substrings: list[str] = PydanticField(default_factory=list)
    measurable_criterion: str = ""
    blocking: bool = False


class TestPackRecoveryCaseV1(ContractModel):
    key: str
    trigger: str
    expected_prompt_key: str
    expected_behavior: str
    measurable_criterion: str


class TestPackAcceptanceJourneyV1(ContractModel):
    key: str
    title: str
    input_reference: str
    expected_behavior: str
    measurable_criterion: str


class StableIssueCatalogEntryV1(ContractModel):
    code: str
    kind: str
    severity: str
    remediation: str


class TestPackExternalConsumerV1(ContractModel):
    relative_path: str
    entry_command: str
    constraints: list[str] = PydanticField(default_factory=list)


class TestPackV1(CanonicalContractBase):
    schema_version: Literal["test-pack.v1"] = "test-pack.v1"
    blueprint_ref: ContractReference
    framework_target: str
    fixtures: list[TestPackFixtureRef] = PydanticField(default_factory=list)
    invalid_fixtures: list[TestPackFixtureRef] = PydanticField(default_factory=list)
    commands: list[TestPackCommandV1] = PydanticField(default_factory=list)
    mutation_cases: list[TestPackMutationCaseV1] = PydanticField(default_factory=list)
    prompt_evaluation_cases: list[TestPackPromptEvaluationCaseV1] = PydanticField(default_factory=list)
    recovery_cases: list[TestPackRecoveryCaseV1] = PydanticField(default_factory=list)
    acceptance_journeys: list[TestPackAcceptanceJourneyV1] = PydanticField(default_factory=list)
    stable_issue_catalog: list[StableIssueCatalogEntryV1] = PydanticField(default_factory=list)
    external_consumer: TestPackExternalConsumerV1

    @field_validator("framework_target")
    @classmethod
    def validate_framework_target(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Field cannot be empty")
        return normalized


class BlueprintCoreV1(CanonicalContractBase):
    schema_version: Literal["blueprint-core.v1"] = "blueprint-core.v1"
    identity: BlueprintIdentity
    purpose: BlueprintPurpose
    scope: BlueprintScope
    behavior_spec: BehaviorSpecV1
    heuristic_decision: HeuristicDecisionV1
    tool_contracts: list[ToolContractV1] = PydanticField(default_factory=list)
    memory_policy: MemoryPolicyV1
    knowledge_contract: KnowledgeContractV1
    llm_policy: LLMPolicyV1
    guardrails: list[str] = PydanticField(default_factory=list)
    approvals: list[ApprovalGateSummary] = PydanticField(default_factory=list)
    risks: list[RiskEntry] = PydanticField(default_factory=list)
    success_criteria: list[SuccessCriterion] = PydanticField(default_factory=list)
    completion_criteria: list[str] = PydanticField(default_factory=list)
    dependencies: list[CanonicalDependency] = PydanticField(default_factory=list)
    assumptions: list[str] = PydanticField(default_factory=list)
    open_questions: list[CanonicalOpenQuestion] = PydanticField(default_factory=list)


CANONICAL_CONTRACT_ORDER = [
    "blueprint-core.v1",
    "construction-pack.v1",
    "agent-construction-package.v2",
    "prompt-pack.v1",
    "estimation-pack.v1",
    "tool-contract.v1",
    "heuristic-decision.v1",
    "llm-policy.v1",
    "memory-policy.v1",
    "short-term-memory.v1",
    "knowledge-contract.v1",
    "knowledge-manifest.v1",
    "behavior-spec.v1",
    "evaluation-pack.v1",
    "test-pack.v1",
]

CANONICAL_CONTRACT_MODELS: dict[str, type[CanonicalContractBase]] = {
    "blueprint-core.v1": BlueprintCoreV1,
    "construction-pack.v1": ConstructionPackV1,
    "agent-construction-package.v2": AgentConstructionPackageV2,
    "prompt-pack.v1": PromptPackV1,
    "estimation-pack.v1": EstimationPackV1,
    "tool-contract.v1": ToolContractV1,
    "heuristic-decision.v1": HeuristicDecisionV1,
    "llm-policy.v1": LLMPolicyV1,
    "memory-policy.v1": MemoryPolicyV1,
    "short-term-memory.v1": ShortTermMemoryV1,
    "knowledge-contract.v1": KnowledgeContractV1,
    "knowledge-manifest.v1": KnowledgeManifestV1,
    "behavior-spec.v1": BehaviorSpecV1,
    "evaluation-pack.v1": EvaluationPackV1,
    "test-pack.v1": TestPackV1,
}


def format_error_path(location: tuple[Any, ...]) -> str:
    return ".".join(str(part) for part in location)


def collect_validation_issues(exc: ValidationError) -> list[CanonicalValidationIssue]:
    issues: list[CanonicalValidationIssue] = []
    for error in exc.errors():
        issues.append(
            CanonicalValidationIssue(
                code=str(error["type"]),
                message=str(error["msg"]),
                path=format_error_path(tuple(error["loc"])),
            )
        )
    return issues


def get_schema_file_name(schema_version: str) -> str:
    return f"{schema_version}.schema.json"
