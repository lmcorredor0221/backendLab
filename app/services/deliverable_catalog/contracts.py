from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID
from typing import Literal

from pydantic import field_validator, model_validator

from app.models import CommercialTier, ContractModel, PydanticField, WorkspaceRole


LEAN_STAGE_ORDER: tuple[str, ...] = (
    "discover",
    "define",
    "design",
    "tools",
    "memory",
    "estimate",
    "validate",
    "package",
)

DIAGRAM_SCHEMA_CONTRACTS: frozenset[str] = frozenset(
    {
        "diagram-model.v1",
        "diagram-presentation.v1",
        "plantuml-source.v1",
        "bpmn-source.v1",
        "c4-source.v1",
        "mermaid-source.v1",
        "data-lineage-source.v1",
    }
)


class DeliverableType(StrEnum):
    artifact = "artifact"
    diagram = "diagram"
    document = "document"
    contract = "contract"
    prompt = "prompt"
    test = "test"
    package = "package"
    lineage = "lineage"


class DeliverableGenerationMode(StrEnum):
    deterministic = "deterministic"
    llm_supported = "llm_supported"
    llm_required = "llm_required"
    llm_with_deterministic_fallback = "llm_with_deterministic_fallback"
    manual_review_required = "manual_review_required"


class DeliverableContentProtection(ContractModel):
    disable_copy: bool = True
    disable_download: bool = True
    disable_context_menu: bool = True


class DeliverableFormats(ContractModel):
    preferred: str
    available: list[str] = PydanticField(default_factory=list)

    @model_validator(mode="after")
    def validate_preferred_format(self) -> "DeliverableFormats":
        if self.preferred not in self.available:
            raise ValueError("preferred format must be included in available formats")
        return self


class DeliverablePromptPolicy(ContractModel):
    prompt_template_key: str = ""
    prompt_status: Literal["active", "draft", "deprecated", "paused", "needs_review"] = "active"
    prompt_version: str = "1.0.0"
    schema_contract: str = ""
    validator_key: str = ""
    fallback_policy: str = ""
    max_iterations: int = 1

    @field_validator("max_iterations")
    @classmethod
    def validate_iterations(cls, value: int) -> int:
        return min(max(int(value or 1), 1), 12)


class DeliverableContextPolicy(ContractModel):
    short_term_refs: list[str] = PydanticField(default_factory=list)
    long_term_collections: list[str] = PydanticField(default_factory=list)
    max_context_tokens: int = 0
    retrieval_strategy: str = "approved_stage_snapshot_only"


class DeliverableQualityPolicy(ContractModel):
    schema_contract: str
    validator_key: str
    minimum_score: int = 80
    checks: list[str] = PydanticField(default_factory=list)

    @field_validator("minimum_score")
    @classmethod
    def validate_score(cls, value: int) -> int:
        return min(max(int(value or 0), 0), 100)


class DeliverableDependencyPolicy(ContractModel):
    depends_on: list[str] = PydanticField(default_factory=list)
    invalidates_on_change: list[str] = PydanticField(default_factory=list)


class DeliverableAccessPolicy(ContractModel):
    preview_mode: Literal["full", "limited", "none"] = "limited"
    sample_enabled: bool = False
    content_protection: DeliverableContentProtection = PydanticField(default_factory=DeliverableContentProtection)


class DeliverableRegistryEntry(ContractModel):
    deliverable_key: str
    title: str
    description: str
    deliverable_type: DeliverableType
    category: str
    stage: str
    enabled_from_stage: str
    product_scope: list[Literal["blueprint", "blueprint_pro", "acp"]] = PydanticField(default_factory=list)
    required_tier: CommercialTier = CommercialTier.blueprint
    access_level: Literal["sample", "view_only", "downloadable", "premium", "restricted"] = "view_only"
    formats: DeliverableFormats
    generation_mode: DeliverableGenerationMode = DeliverableGenerationMode.deterministic
    prompt_policy: DeliverablePromptPolicy = PydanticField(default_factory=DeliverablePromptPolicy)
    context_policy: DeliverableContextPolicy = PydanticField(default_factory=DeliverableContextPolicy)
    quality_policy: DeliverableQualityPolicy
    dependency_policy: DeliverableDependencyPolicy = PydanticField(default_factory=DeliverableDependencyPolicy)
    access_policy: DeliverableAccessPolicy = PydanticField(default_factory=DeliverableAccessPolicy)
    canonical_paths: list[str] = PydanticField(default_factory=list)
    portable_paths: list[str] = PydanticField(default_factory=list)
    exportable: bool = False
    blueprint_download: bool = False
    acp_download: bool = False
    sort_order: int = 0
    active: bool = True

    @field_validator("deliverable_key", "title", "description", "category", "stage", "enabled_from_stage")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("deliverable key, title, description, category and stages are required")
        return normalized

    @model_validator(mode="after")
    def validate_governed_entry(self) -> "DeliverableRegistryEntry":
        if self.stage not in LEAN_STAGE_ORDER:
            raise ValueError(f"invalid stage: {self.stage}")
        if self.enabled_from_stage not in LEAN_STAGE_ORDER:
            raise ValueError(f"invalid enabled_from_stage: {self.enabled_from_stage}")
        if not self.product_scope:
            raise ValueError("product_scope cannot be empty")
        if self.exportable and not self.canonical_paths and not self.portable_paths:
            raise ValueError("exportable deliverables require canonical_paths or portable_paths")
        if self.deliverable_type == DeliverableType.diagram and self.quality_policy.schema_contract not in DIAGRAM_SCHEMA_CONTRACTS:
            raise ValueError("diagram deliverables must use a supported diagram source or presentation schema")
        if self.generation_mode in {
            DeliverableGenerationMode.llm_supported,
            DeliverableGenerationMode.llm_required,
            DeliverableGenerationMode.llm_with_deterministic_fallback,
        }:
            required = {
                "prompt_template_key": self.prompt_policy.prompt_template_key,
                "schema_contract": self.prompt_policy.schema_contract,
                "validator_key": self.prompt_policy.validator_key,
                "fallback_policy": self.prompt_policy.fallback_policy,
            }
            missing = [key for key, value in required.items() if not str(value or "").strip()]
            if missing:
                raise ValueError(f"LLM generation requires prompt policy fields: {missing}")
        if "acp" not in self.product_scope and self.acp_download:
            raise ValueError("acp_download requires acp product scope")
        if self.required_tier == CommercialTier.acp and self.blueprint_download:
            raise ValueError("ACP-only deliverables cannot be in blueprint_download")
        return self


class DeliverableCatalog(ContractModel):
    schema_version: Literal["deliverable-catalog.v1"] = "deliverable-catalog.v1"
    generated_at: str
    lean_stage_order: list[str]
    products: list[Literal["blueprint", "blueprint_pro", "acp"]]
    deliverable_types: list[DeliverableType]
    generation_modes: list[DeliverableGenerationMode]
    entries: list[DeliverableRegistryEntry]
    validation_rules: list[str] = PydanticField(default_factory=list)

    @model_validator(mode="after")
    def validate_catalog(self) -> "DeliverableCatalog":
        if tuple(self.lean_stage_order) != LEAN_STAGE_ORDER:
            raise ValueError("lean_stage_order must match canonical LEAN order")
        keys = [entry.deliverable_key for entry in self.entries]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(f"duplicate deliverable keys: {duplicates}")
        return self


class DeliverablePolicyContext(ContractModel):
    workspace_id: UUID | None = None
    user_id: UUID | None = None
    role: WorkspaceRole | None = None
    tier: CommercialTier = CommercialTier.blueprint
    current_stage: str = "discover"
    has_current_version: bool = False
    generation_state: Literal["pending", "queued", "generating", "available", "error", "updating"] = "pending"
    quality_state: Literal["unknown", "passed", "warning", "failed", "stale"] = "unknown"


class DeliverablePolicyDecision(ContractModel):
    contract_version: Literal["deliverable-policy-decision.v1"] = "deliverable-policy-decision.v1"
    visible: bool = True
    access_state: Literal[
        "available",
        "preview",
        "locked",
        "stage_locked",
        "disabled",
        "not_generated",
        "stale",
        "quality_failed",
    ] = "locked"
    can_view: bool = False
    can_generate: bool = False
    can_regenerate: bool = False
    can_download: bool = False
    can_compare: bool = False
    can_edit_prompt: bool = False
    reason_code: str = ""
    reason: str = ""
    cta_label: str = ""
    required_tier: CommercialTier = CommercialTier.blueprint
    effective_prompt_status: str = "active"
    preview_mode: str = "limited"


class DeliverableGovernanceUpdate(ContractModel):
    enabled: bool = True
    generation_enabled: bool = True
    required_tier_override: str = ""
    preview_mode_override: str = ""
    prompt_status: Literal["active", "draft", "deprecated", "paused", "needs_review"] = "active"
    prompt_override: dict[str, object] = PydanticField(default_factory=dict)
    notes: str = ""


class DeliverableGovernanceEntry(ContractModel):
    deliverable_key: str
    title: str
    description: str = ""
    deliverable_type: DeliverableType
    category: str = ""
    stage: str = ""
    enabled_from_stage: str = ""
    product_scope: list[str] = PydanticField(default_factory=list)
    access_level: str = ""
    formats: DeliverableFormats
    generation_mode: DeliverableGenerationMode
    prompt_policy: DeliverablePromptPolicy = PydanticField(default_factory=DeliverablePromptPolicy)
    context_policy: DeliverableContextPolicy = PydanticField(default_factory=DeliverableContextPolicy)
    quality_policy: DeliverableQualityPolicy
    dependency_policy: DeliverableDependencyPolicy = PydanticField(default_factory=DeliverableDependencyPolicy)
    access_policy: DeliverableAccessPolicy = PydanticField(default_factory=DeliverableAccessPolicy)
    canonical_paths: list[str] = PydanticField(default_factory=list)
    portable_paths: list[str] = PydanticField(default_factory=list)
    exportable: bool = False
    blueprint_download: bool = False
    acp_download: bool = False
    active: bool = True
    scope_key: str = "platform"
    workspace_id: UUID | None = None
    enabled: bool
    generation_enabled: bool
    required_tier: CommercialTier
    preview_mode: str
    prompt_status: str
    prompt_override: dict[str, object] = PydanticField(default_factory=dict)
    notes: str = ""
    updated_at: datetime | None = None


class DeliverableGovernanceAuditEntry(ContractModel):
    id: UUID
    deliverable_key: str
    scope_key: str
    action: str
    changed_fields: list[str] = PydanticField(default_factory=list)
    actor_user_id: UUID | None = None
    reason: str = ""
    created_at: datetime


class DeliverableQualitySnapshotSummaryEntry(ContractModel):
    id: UUID
    workspace_id: UUID
    session_id: UUID
    deliverable_key: str
    title: str = ""
    deliverable_type: str = ""
    stage: str = ""
    product_scope: list[str] = PydanticField(default_factory=list)
    version_ref: str = ""
    state: str = "unknown"
    score: int = 0
    errors_count: int = 0
    warnings_count: int = 0
    created_at: datetime


class DeliverableQualitySummary(ContractModel):
    total_snapshots: int = 0
    average_score: float = 0.0
    by_state: dict[str, int] = PydanticField(default_factory=dict)
    recent_snapshots: list[DeliverableQualitySnapshotSummaryEntry] = PydanticField(default_factory=list)


class DeliverableGovernanceResponse(ContractModel):
    contract_version: Literal["deliverable-governance.v1"] = "deliverable-governance.v1"
    entries: list[DeliverableGovernanceEntry]


class DeliverableCatalogItem(ContractModel):
    key: str
    title: str
    description: str
    deliverable_type: DeliverableType
    category: str
    stage: str
    enabled_from_stage: str
    product_scope: list[str]
    required_tier: CommercialTier
    access_level: str
    generation_mode: DeliverableGenerationMode
    formats: DeliverableFormats
    exportable: bool
    blueprint_download: bool
    acp_download: bool
    sort_order: int
    access: DeliverablePolicyDecision


class DeliverableCatalogResponse(ContractModel):
    contract_version: Literal["deliverable-catalog-response.v1"] = "deliverable-catalog-response.v1"
    registry_version: str = "deliverable-catalog.v1"
    current_stage: str
    tier: CommercialTier
    entries: list[DeliverableCatalogItem]


class DeliverableDetailResponse(ContractModel):
    contract_version: Literal["deliverable-detail.v1"] = "deliverable-detail.v1"
    entry: DeliverableRegistryEntry
    access: DeliverablePolicyDecision
    governance: DeliverableGovernanceEntry


class DeliverableGovernanceOverview(ContractModel):
    contract_version: Literal["deliverable-governance-overview.v1"] = "deliverable-governance-overview.v1"
    registry_version: str = "deliverable-catalog.v1"
    total_entries: int
    active_entries: int
    governed_entries: int
    by_type: dict[str, int] = PydanticField(default_factory=dict)
    by_stage: dict[str, int] = PydanticField(default_factory=dict)
    by_access_state: dict[str, int] = PydanticField(default_factory=dict)
    by_prompt_status: dict[str, int] = PydanticField(default_factory=dict)
    quality_summary: DeliverableQualitySummary = PydanticField(default_factory=DeliverableQualitySummary)
    recent_audit: list[DeliverableGovernanceAuditEntry] = PydanticField(default_factory=list)


class DeliverablePromptVersionEntry(ContractModel):
    id: UUID | None = None
    version: str
    status: str
    prompt_template_key: str
    schema_contract: str
    validator_key: str
    fallback_policy: str
    created_by_user_id: UUID | None = None
    created_at: datetime | None = None


class DeliverablePromptResponse(ContractModel):
    contract_version: Literal["deliverable-prompt.v1"] = "deliverable-prompt.v1"
    deliverable_key: str
    scope_key: str = "platform"
    workspace_id: UUID | None = None
    prompt_template_key: str
    prompt_status: Literal["active", "draft", "deprecated", "paused", "needs_review"] = "active"
    prompt_version: str
    prompt_body: str
    schema_contract: str
    validator_key: str
    fallback_policy: str
    max_iterations: int
    prompt_override: dict[str, object] = PydanticField(default_factory=dict)
    versions: list[DeliverablePromptVersionEntry] = PydanticField(default_factory=list)


class DeliverablePromptUpdate(ContractModel):
    prompt_status: Literal["active", "draft", "deprecated", "paused", "needs_review"] = "active"
    prompt_body: str
    schema_contract: str = ""
    validator_key: str = ""
    fallback_policy: str = ""
    version: str = ""
    change_reason: str = ""
    metadata: dict[str, object] = PydanticField(default_factory=dict)

    @field_validator("prompt_body")
    @classmethod
    def validate_prompt_body(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("prompt_body is required")
        return normalized


class DeliverablePromptValidationRequest(ContractModel):
    prompt_body: str
    schema_contract: str = ""
    validator_key: str = ""
    fallback_policy: str = ""


class DeliverablePromptValidationResponse(ContractModel):
    contract_version: Literal["deliverable-prompt-validation.v1"] = "deliverable-prompt-validation.v1"
    valid: bool
    errors: list[str] = PydanticField(default_factory=list)
    warnings: list[str] = PydanticField(default_factory=list)
    required_schema_contract: str = ""
    required_validator_key: str = ""


class DeliverableQualityEvaluation(ContractModel):
    contract_version: Literal["deliverable-quality-evaluation.v1"] = "deliverable-quality-evaluation.v1"
    deliverable_key: str
    schema_contract: str
    validator_key: str
    state: Literal["passed", "warning", "failed"] = "failed"
    score: int = 0
    errors: list[str] = PydanticField(default_factory=list)
    warnings: list[str] = PydanticField(default_factory=list)
    checks: dict[str, object] = PydanticField(default_factory=dict)


class DeliverableStalenessReport(ContractModel):
    contract_version: Literal["deliverable-staleness-report.v1"] = "deliverable-staleness-report.v1"
    changed_dependency_keys: list[str] = PydanticField(default_factory=list)
    stale_deliverable_keys: list[str] = PydanticField(default_factory=list)
    unchanged_deliverable_keys: list[str] = PydanticField(default_factory=list)
    reasons_by_deliverable: dict[str, list[str]] = PydanticField(default_factory=dict)
    superseded_uncertainty_count: int = 0


class DeliverableRegenerationScope(ContractModel):
    contract_version: Literal["deliverable-regeneration-scope.v1"] = "deliverable-regeneration-scope.v1"
    source_deliverable_key: str = ""
    changed_dependency_keys: list[str] = PydanticField(default_factory=list)
    affected_deliverable_keys: list[str] = PydanticField(default_factory=list)
    ordered_regeneration_keys: list[str] = PydanticField(default_factory=list)
    unaffected_deliverable_keys: list[str] = PydanticField(default_factory=list)


class DeliverableGenerationTask(ContractModel):
    contract_version: Literal["deliverable-generation-task.v1"] = "deliverable-generation-task.v1"
    workspace_id: UUID
    session_id: UUID
    deliverable_key: str
    product_mode: str = "basic_free"
    current_stage: str = "discover"
    tier: CommercialTier = CommercialTier.blueprint
    idempotency_key: str
    requested_by_user_id: UUID | None = None
    context_payload: dict[str, object] = PydanticField(default_factory=dict)
    approved_context_refs: list[str] = PydanticField(default_factory=list)
    allow_llm: bool = False
    max_iterations: int = 3


class DeliverableGenerationTraceStep(ContractModel):
    step: Literal["reason", "act", "observe", "evaluate", "finish"]
    public_summary: str
    status: Literal["completed", "skipped", "failed"] = "completed"


class DeliverableGenerationResult(ContractModel):
    contract_version: Literal["deliverable-generation-result.v1"] = "deliverable-generation-result.v1"
    deliverable_key: str
    status: Literal["available", "requires_attention", "failed"] = "failed"
    output_payload: dict[str, object] = PydanticField(default_factory=dict)
    quality: DeliverableQualityEvaluation | None = None
    public_trace: list[DeliverableGenerationTraceStep] = PydanticField(default_factory=list)
    internal_trace_hash: str = ""
    iteration_count: int = 0
    provider_key: str = ""
    model_name: str = ""
    prompt_version: str = ""
    tokens_input: int = 0
    tokens_output: int = 0
    estimated_cost_usd: float = 0.0
    used_fallback: bool = False
    error_code: str = ""
    error_message: str = ""
    warnings: list[str] = PydanticField(default_factory=list)
