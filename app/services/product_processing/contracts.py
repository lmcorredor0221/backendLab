from __future__ import annotations

from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import field_validator

from app.models import CommercialTier, ContractModel, PydanticField


class ProductProcessingMode(StrEnum):
    basic_free = "basic_free"
    premium_enrichment = "premium_enrichment"
    acp_implementation = "acp_implementation"


class ProductBuildProductKey(StrEnum):
    blueprint_basic = "blueprint_basic"
    blueprint_pro = "blueprint_pro"
    acp = "acp"


class ProductBuildLifecycle(StrEnum):
    not_purchased = "not_purchased"
    payment_pending = "payment_pending"
    locked = "locked"
    ready_to_start = "ready_to_start"
    queued = "queued"
    preparing = "preparing"
    running = "running"
    requires_attention = "requires_attention"
    partial = "partial"
    completed = "completed"
    error = "error"


class ProductBuildDeliverableState(StrEnum):
    not_required = "not_required"
    locked = "locked"
    pending = "pending"
    queued = "queued"
    generating = "generating"
    available = "available"
    stale = "stale"
    requires_attention = "requires_attention"
    error = "error"
    skipped = "skipped"


class ProductBuildActionState(StrEnum):
    hidden = "hidden"
    disabled = "disabled"
    available = "available"
    recommended = "recommended"
    running = "running"
    blocked = "blocked"


class ProductBuildAttentionSeverity(StrEnum):
    info = "info"
    warning = "warning"
    blocking = "blocking"
    technical_error = "technical_error"


class ProductBuildProcessingQueueMode(StrEnum):
    process_pending = "process_pending"
    retry_failed = "retry_failed"


class ProductBuildProcessingItemStatus(StrEnum):
    pending = "pending"
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


def calculate_product_build_percent(completed_units: float, total_units: float) -> int:
    if total_units <= 0:
        return 0
    return min(max(round((completed_units / total_units) * 100), 0), 100)


class ProductBuildProgress(ContractModel):
    percent: int = 0
    completed_units: float = 0.0
    total_units: float = 0.0
    blocked_units: float = 0.0
    calculation: Literal["weighted_units", "manual", "not_applicable"] = "weighted_units"
    label: str = ""

    @field_validator("percent")
    @classmethod
    def validate_percent(cls, value: int) -> int:
        return min(max(int(value or 0), 0), 100)


class ProductBuildEntitlement(ContractModel):
    tier: CommercialTier = CommercialTier.blueprint
    access_state: Literal["allowed", "preview", "locked", "payment_pending"] = "preview"
    is_purchased: bool = False
    purchase_required: bool = False
    checkout_href: str = ""
    upgrade_label: str = ""


class ProductBuildCurrentActivity(ContractModel):
    activity_key: str = ""
    label: str = ""
    detail: str = ""
    step_key: str = ""
    status: Literal["idle", "queued", "running", "waiting_user", "completed", "failed"] = "idle"
    started_at: str = ""
    updated_at: str = ""


class ProductBuildStageStatus(ContractModel):
    stage_key: str
    label: str
    lifecycle: ProductBuildLifecycle = ProductBuildLifecycle.locked
    progress: ProductBuildProgress = PydanticField(default_factory=ProductBuildProgress)
    blocker_count: int = 0
    deliverable_count: int = 0


class ProductBuildDeliverableStatus(ContractModel):
    deliverable_key: str
    title: str
    deliverable_type: Literal["diagram", "document", "artifact", "prompt", "contract", "test", "package", "lineage"] = "artifact"
    state: ProductBuildDeliverableState = ProductBuildDeliverableState.pending
    product_surface: ProductBuildProductKey = ProductBuildProductKey.blueprint_basic
    stage_key: str = ""
    required: bool = True
    job_id: str = ""
    updated_at: str = ""
    href: str = ""


class ProductBuildAttentionItem(ContractModel):
    key: str
    title: str
    severity: ProductBuildAttentionSeverity = ProductBuildAttentionSeverity.info
    product_key: str = ""
    run_id: str = ""
    step_id: str = ""
    source: str = ""
    stage_key: str = ""
    deliverable_key: str = ""
    href: str = ""
    reason: str = ""
    blocking: bool = False


class ProductBuildAttentionSummary(ContractModel):
    total: int = 0
    blocking_count: int = 0
    warning_count: int = 0
    technical_error_count: int = 0
    items: list[ProductBuildAttentionItem] = PydanticField(default_factory=list)


class ProductBuildAction(ContractModel):
    action_key: str
    label: str
    state: ProductBuildActionState = ProductBuildActionState.hidden
    href: str = ""
    reason: str = ""
    primary: bool = False


class ProductBuildRecoverableError(ContractModel):
    code: str
    title: str
    message: str
    recoverable: bool = True
    technical_message: str = ""
    retry_action_key: str = ""
    trace_refs: list[str] = PydanticField(default_factory=list)


class ProductBuildProcessingQueueItem(ContractModel):
    deliverable_key: str
    title: str = ""
    deliverable_type: Literal["diagram", "document", "artifact", "prompt", "contract", "test", "package", "lineage"] = "artifact"
    stage_key: str = ""
    status: ProductBuildProcessingItemStatus = ProductBuildProcessingItemStatus.pending
    attempt_count: int = 0
    retried: bool = False
    error_message: str = ""
    href: str = ""
    job_id: str = ""
    updated_at: str = ""


class ProductBuildProcessingQueueStatus(ContractModel):
    active: bool = False
    queue_id: str = ""
    mode: ProductBuildProcessingQueueMode = ProductBuildProcessingQueueMode.process_pending
    status: Literal["idle", "queued", "running", "completed", "completed_with_errors"] = "idle"
    total_count: int = 0
    pending_count: int = 0
    processing_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    retried_count: int = 0
    started_at: str = ""
    completed_at: str = ""
    updated_at: str = ""
    current_deliverable_key: str = ""
    summary: str = ""
    completed_items: list[ProductBuildProcessingQueueItem] = PydanticField(default_factory=list)
    failed_items: list[ProductBuildProcessingQueueItem] = PydanticField(default_factory=list)


class ProductBuildStatus(ContractModel):
    contract_version: Literal["product-build-status.v1"] = "product-build-status.v1"
    workspace_id: UUID
    session_id: UUID
    product_key: ProductBuildProductKey
    product_mode: ProductProcessingMode
    product_label: str
    lifecycle: ProductBuildLifecycle = ProductBuildLifecycle.not_purchased
    entitlement: ProductBuildEntitlement = PydanticField(default_factory=ProductBuildEntitlement)
    progress: ProductBuildProgress = PydanticField(default_factory=ProductBuildProgress)
    current_activity: ProductBuildCurrentActivity | None = None
    stages: list[ProductBuildStageStatus] = PydanticField(default_factory=list)
    deliverables: list[ProductBuildDeliverableStatus] = PydanticField(default_factory=list)
    attention: ProductBuildAttentionSummary = PydanticField(default_factory=ProductBuildAttentionSummary)
    actions: list[ProductBuildAction] = PydanticField(default_factory=list)
    last_error: ProductBuildRecoverableError | None = None
    processing_queue: ProductBuildProcessingQueueStatus | None = None
    generated_at: str = ""
    source_contracts: list[str] = PydanticField(default_factory=list)


class ProductJourneyCurrentStage(ContractModel):
    stage_key: str = "discover"
    label: str = "Descubrir"
    lifecycle: ProductBuildLifecycle = ProductBuildLifecycle.ready_to_start
    progress_percent: int = 0
    product_key: ProductBuildProductKey = ProductBuildProductKey.blueprint_basic

    @field_validator("progress_percent")
    @classmethod
    def validate_progress_percent(cls, value: int) -> int:
        return min(max(int(value or 0), 0), 100)


class ProductJourneyRecommendedAction(ContractModel):
    action_key: str = ""
    label: str = ""
    state: ProductBuildActionState = ProductBuildActionState.hidden
    href: str = ""
    reason: str = ""
    product_key: ProductBuildProductKey = ProductBuildProductKey.blueprint_basic
    primary: bool = True


class ProductJourneyOutcome(ContractModel):
    key: str
    title: str
    detail: str = ""
    product_key: ProductBuildProductKey = ProductBuildProductKey.blueprint_basic
    stage_key: str = ""
    href: str = ""


class ProductJourneyDeliverableSummary(ContractModel):
    total_count: int = 0
    available_count: int = 0
    pending_count: int = 0
    running_count: int = 0
    locked_count: int = 0
    stale_count: int = 0
    attention_count: int = 0
    error_count: int = 0


class ProductJourneyProductSummary(ContractModel):
    product_key: ProductBuildProductKey
    product_label: str
    lifecycle: ProductBuildLifecycle = ProductBuildLifecycle.not_purchased
    access_state: Literal["allowed", "preview", "locked", "payment_pending"] = "preview"
    is_purchased: bool = False
    purchase_required: bool = False
    progress_percent: int = 0
    available_deliverable_count: int = 0
    total_deliverable_count: int = 0
    blocking_attention_count: int = 0
    warning_attention_count: int = 0
    technical_error_count: int = 0
    active_operation: ProductBuildCurrentActivity | None = None
    primary_action: ProductJourneyRecommendedAction | None = None

    @field_validator("progress_percent")
    @classmethod
    def validate_progress_percent(cls, value: int) -> int:
        return min(max(int(value or 0), 0), 100)


class ProductJourneyOverview(ContractModel):
    contract_version: Literal["product-journey-overview.v2"] = "product-journey-overview.v2"
    workspace_id: UUID
    session_id: UUID
    project_title: str = ""
    current_stage: ProductJourneyCurrentStage = PydanticField(default_factory=ProductJourneyCurrentStage)
    achieved_outcomes: list[ProductJourneyOutcome] = PydanticField(default_factory=list)
    active_operation: ProductBuildCurrentActivity | None = None
    blocking_attention_count: int = 0
    warning_attention_count: int = 0
    technical_error_count: int = 0
    recommended_next_action: ProductJourneyRecommendedAction | None = None
    products: list[ProductJourneyProductSummary] = PydanticField(default_factory=list)
    deliverable_summary: ProductJourneyDeliverableSummary = PydanticField(default_factory=ProductJourneyDeliverableSummary)
    generated_at: str = ""
    source_contracts: list[str] = PydanticField(default_factory=list)


class ProductBuildTelemetryEvent(ContractModel):
    event_key: str
    event_type: Literal["cta", "checkout", "activation", "run", "retry", "resume", "attention", "error", "other"] = "other"
    workspace_id: UUID
    session_id: UUID
    product_key: ProductBuildProductKey
    run_id: str = ""
    step_id: str = ""
    stage_key: str = ""
    deliverable_key: str = ""
    source: str = ""
    status: str = ""
    created_at: str = ""
    metadata_keys: list[str] = PydanticField(default_factory=list)


class ProductBuildTelemetryProductSummary(ContractModel):
    product_key: ProductBuildProductKey
    product_label: str
    run_id: str = ""
    lifecycle: str = "not_started"
    run_count: int = 0
    step_count: int = 0
    deliverable_count: int = 0
    event_count: int = 0
    cta_count: int = 0
    checkout_count: int = 0
    activation_count: int = 0
    run_started_count: int = 0
    run_completed_count: int = 0
    run_error_count: int = 0
    requires_attention_count: int = 0
    retry_count: int = 0
    resume_count: int = 0
    run_duration_seconds: int = 0
    deliverable_duration_seconds: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_total: int = 0
    estimated_cost_usd: float = 0.0
    latest_at: str = ""


class ProductBuildTelemetryTotals(ContractModel):
    product_count: int = 0
    run_count: int = 0
    step_count: int = 0
    deliverable_count: int = 0
    event_count: int = 0
    requires_attention_count: int = 0
    retry_count: int = 0
    resume_count: int = 0
    tokens_total: int = 0
    estimated_cost_usd: float = 0.0


class ProductBuildTelemetryReport(ContractModel):
    contract_version: Literal["product-build-telemetry.v1"] = "product-build-telemetry.v1"
    workspace_id: UUID
    session_id: UUID
    requested_by_user_id: UUID | None = None
    generated_at: str = ""
    redaction_policy: str = (
        "Only operational metadata, ids, counts, timing, token totals and cost aggregates are exposed. "
        "Raw prompts, reasoning traces, document content, diagram source, secrets and credentials are not returned."
    )
    products: list[ProductBuildTelemetryProductSummary] = PydanticField(default_factory=list)
    events: list[ProductBuildTelemetryEvent] = PydanticField(default_factory=list)
    totals: ProductBuildTelemetryTotals = PydanticField(default_factory=ProductBuildTelemetryTotals)
    warnings: list[str] = PydanticField(default_factory=list)
    source_contracts: list[str] = PydanticField(default_factory=list)


class QuestionPolicyMode(StrEnum):
    infer_defer = "infer_defer"
    prioritized_enrichment = "prioritized_enrichment"
    full_readiness = "full_readiness"


class UncertaintyDisposition(StrEnum):
    infer = "infer"
    defer = "defer"
    block = "block"
    resolve_now = "resolve_now"


class UncertaintyKind(StrEnum):
    question = "question"
    gap = "gap"
    assumption = "assumption"
    decision = "decision"
    hitl = "hitl"
    runtime_error = "runtime_error"
    stale_dependency = "stale_dependency"


class UncertaintyBacklogStatus(StrEnum):
    open = "open"
    in_progress = "in_progress"
    resolved = "resolved"
    deferred = "deferred"
    superseded = "superseded"
    dismissed = "dismissed"


class UncertaintyOption(ContractModel):
    key: str
    label: str
    description: str = ""
    impact: str = ""
    recommended: bool = False
    confidence: float = 0.0

    @field_validator("key", "label")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("option key and label are required")
        return normalized

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        return min(max(float(value or 0.0), 0.0), 1.0)


class BlueprintUncertainty(ContractModel):
    contract_version: Literal["blueprint-uncertainty.v1"] = "blueprint-uncertainty.v1"
    key: str
    kind: UncertaintyKind = UncertaintyKind.question
    stage: str
    title: str
    description: str = ""
    reason: str = ""
    impact: str = ""
    confidence: float = 0.0
    source_refs: list[str] = PydanticField(default_factory=list)
    affected_deliverable_keys: list[str] = PydanticField(default_factory=list)
    product_targets: list[ProductProcessingMode] = PydanticField(default_factory=list)
    disposition: UncertaintyDisposition = UncertaintyDisposition.defer
    deferral_target_stage: str = ""
    assumed_answer: str = ""
    suggested_answer: str = ""
    answer_options: list[UncertaintyOption] = PydanticField(default_factory=list)
    blocking: bool = False
    required_for_implementation: bool = False

    @field_validator("key", "stage", "title")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("uncertainty key, stage and title are required")
        return normalized

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        return min(max(float(value or 0.0), 0.0), 1.0)


class ProductProcessingProfile(ContractModel):
    contract_version: Literal["product-processing-profile.v1"] = "product-processing-profile.v1"
    mode: ProductProcessingMode
    commercial_tier: CommercialTier
    label: str
    question_policy: QuestionPolicyMode
    default_disposition: UncertaintyDisposition
    max_questions_per_stage: int = 0
    max_llm_iterations_per_stage: int = 0
    max_llm_calls_per_stage: int = 0
    cost_budget_units_per_stage: int = 0
    deliverable_generation_budget: Literal["low", "balanced", "full"] = "balanced"
    reprocess_strategy: Literal["none", "selective", "full"] = "none"
    allow_inferred_assumptions: bool = True
    allow_nonblocking_continuation: bool = True
    require_stage_readiness: bool = False
    require_all_readiness_blockers: bool = False
    surface_deferred_questions: bool = False
    surface_technical_questions: bool = False
    create_attention_for_nonblocking: bool = False


class BlueprintTierPolicy(ContractModel):
    contract_version: Literal["blueprint-tier-policy.v1"] = "blueprint-tier-policy.v1"
    profiles: dict[ProductProcessingMode, ProductProcessingProfile]
    infer_confidence_threshold: float = 0.72
    premium_priority_threshold: float = 0.55

    @field_validator("infer_confidence_threshold", "premium_priority_threshold")
    @classmethod
    def validate_threshold(cls, value: float) -> float:
        return min(max(float(value or 0.0), 0.0), 1.0)


class UncertaintyClassification(ContractModel):
    contract_version: Literal["uncertainty-classification.v1"] = "uncertainty-classification.v1"
    uncertainty: BlueprintUncertainty
    profile_mode: ProductProcessingMode
    disposition: UncertaintyDisposition
    reason: str = ""
    target_stage: str = ""
    should_surface_to_user: bool = False
    should_create_attention: bool = False
    should_continue_processing: bool = True


class UncertaintyBacklogEntry(ContractModel):
    contract_version: Literal["uncertainty-backlog-entry.v1"] = "uncertainty-backlog-entry.v1"
    id: str
    workspace_id: str
    session_id: str
    uncertainty_key: str
    product_mode: ProductProcessingMode
    source_stage: str
    target_stage: str = ""
    kind: UncertaintyKind = UncertaintyKind.question
    disposition: UncertaintyDisposition
    status: UncertaintyBacklogStatus = UncertaintyBacklogStatus.open
    title: str
    reason: str = ""
    impact: str = ""
    confidence: float = 0.0
    cost_to_resolve_units: int = 1
    assumed_answer: str = ""
    suggested_answer: str = ""
    answer_options: list[UncertaintyOption] = PydanticField(default_factory=list)
    source_refs: list[str] = PydanticField(default_factory=list)
    affected_deliverable_keys: list[str] = PydanticField(default_factory=list)
    dependency_keys: list[str] = PydanticField(default_factory=list)
    created_from: str = "runtime"


class PremiumEnrichmentItem(ContractModel):
    contract_version: Literal["premium-enrichment-item.v1"] = "premium-enrichment-item.v1"
    entry: UncertaintyBacklogEntry
    priority_score: float = 0.0
    priority_reason: str = ""
    changed_dependency_keys: list[str] = PydanticField(default_factory=list)
    affected_deliverable_keys: list[str] = PydanticField(default_factory=list)
    ordered_regeneration_keys: list[str] = PydanticField(default_factory=list)
    unaffected_deliverable_count: int = 0


class PremiumEnrichmentWorkspace(ContractModel):
    contract_version: Literal["premium-enrichment-workspace.v1"] = "premium-enrichment-workspace.v1"
    workspace_id: UUID
    session_id: UUID
    current_tier: CommercialTier = CommercialTier.blueprint
    product_mode: ProductProcessingMode = ProductProcessingMode.premium_enrichment
    selectable_limit: int = 6
    total_uncertainties: int = 0
    prioritized_count: int = 0
    deferred_count: int = 0
    resolved_count: int = 0
    items: list[PremiumEnrichmentItem] = PydanticField(default_factory=list)
    value_summary: str = ""
    processing_guidance: str = ""


class PremiumUncertaintyResolutionRequest(ContractModel):
    answer: str = ""
    selected_option_key: str = ""
    regenerate: bool = False
    execution_mode: Literal["analyze_only", "apply_reprocess"] = "analyze_only"
    max_deliverables: int = 5

    @field_validator("max_deliverables")
    @classmethod
    def validate_max_deliverables(cls, value: int) -> int:
        return min(max(int(value or 0), 0), 12)


class PremiumSelectiveReprocessResult(ContractModel):
    contract_version: Literal["premium-selective-reprocess-result.v1"] = "premium-selective-reprocess-result.v1"
    resolved_entry: UncertaintyBacklogEntry
    changed_dependency_keys: list[str] = PydanticField(default_factory=list)
    stale_deliverable_keys: list[str] = PydanticField(default_factory=list)
    ordered_regeneration_keys: list[str] = PydanticField(default_factory=list)
    regenerated_deliverable_keys: list[str] = PydanticField(default_factory=list)
    preserved_deliverable_keys: list[str] = PydanticField(default_factory=list)
    material_impact: bool = False
    reprocess_decision: Literal["document_only", "localized_reprocess", "structural_reprocess"] = "document_only"
    recommended_action: str = ""
    impact_summary: str = ""
    generation_job_ids: list[str] = PydanticField(default_factory=list)
    generation_status_by_deliverable: dict[str, str] = PydanticField(default_factory=dict)
    superseded_uncertainty_count: int = 0
    comparison_summary: str = ""
    queue_total: int = 0
    queue_completed: int = 0
    queue_status: str = "completed"
    queue_processed_keys: list[str] = PydanticField(default_factory=list)


class AcpStageReadinessEntry(ContractModel):
    stage_key: str
    label: str
    completed: bool = False
    justified: bool = False
    justification: str = ""
    technical_question_count: int = 0
    blocking_question_count: int = 0
    next_action: str = ""


class AcpDirectRouteResolution(ContractModel):
    contract_version: Literal["acp-direct-route-resolution.v1"] = "acp-direct-route-resolution.v1"
    workspace_id: UUID
    session_id: UUID
    current_tier: CommercialTier = CommercialTier.blueprint
    route_kind: Literal["acp_direct", "acp_after_blueprint"] = "acp_direct"
    product_mode: ProductProcessingMode = ProductProcessingMode.acp_implementation
    question_policy: QuestionPolicyMode = QuestionPolicyMode.full_readiness
    required_stage_keys: list[str] = PydanticField(default_factory=list)
    completed_stage_keys: list[str] = PydanticField(default_factory=list)
    missing_stage_keys: list[str] = PydanticField(default_factory=list)
    justified_stage_keys: list[str] = PydanticField(default_factory=list)
    stages: list[AcpStageReadinessEntry] = PydanticField(default_factory=list)
    can_start_package: bool = False
    can_export_package: bool = False
    next_stage_key: str = "discover"
    readiness_blockers: list[str] = PydanticField(default_factory=list)
    total_technical_questions: int = 0
    total_blocking_questions: int = 0
    catalog_counts: dict[str, int] = PydanticField(default_factory=dict)
    portable_catalog_paths: list[str] = PydanticField(default_factory=list)
    processing_guidance: str = ""


class ProductBuildCommandRequest(ContractModel):
    action: Literal["start", "resume", "retry", "process_pending", "retry_failed"] = "start"
    idempotency_key: str = ""
    allow_llm: bool = False
