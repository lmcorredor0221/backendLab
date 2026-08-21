from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import AliasChoices, BaseModel, ConfigDict, Field as PydanticField, field_validator, model_validator
from sqlalchemy import Column, Index, JSON, String, UniqueConstraint
from sqlmodel import Field, SQLModel

from app.diagnostics import (
    CASE_TYPE_COPILOT,
    AUTONOMY_MEDIUM,
    normalize_autonomy_level,
    normalize_case_type,
)


def utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class SessionStage(str, Enum):
    draft_capture = "draft_capture"
    input_validation = "input_validation"
    normalize_discovery = "normalize_discovery"
    build_canvas = "build_canvas"
    build_blueprint = "build_blueprint"
    post_validation = "post_validation"
    ready_for_export = "ready_for_export"


class CommercialTier(str, Enum):
    blueprint = "blueprint"
    blueprint_pro = "blueprint_pro"
    acp = "acp"


class ProjectTitleSource(str, Enum):
    generated = "generated"
    manual = "manual"
    migrated = "migrated"


class CommercialProductType(str, Enum):
    blueprint = "blueprint"
    acp = "acp"


class CommercialProductStatus(str, Enum):
    active = "active"
    archived = "archived"


class CommercialPriceStatus(str, Enum):
    active = "active"
    archived = "archived"


class CommercialOrderStatus(str, Enum):
    pending = "pending"
    paid = "paid"
    failed = "failed"
    canceled = "canceled"
    refunded = "refunded"


class CommercialPaymentStatus(str, Enum):
    pending = "pending"
    succeeded = "succeeded"
    failed = "failed"
    canceled = "canceled"
    refunded = "refunded"


class CommercialEntitlementStatus(str, Enum):
    pending_activation = "pending_activation"
    active = "active"
    suspended = "suspended"
    expired = "expired"
    revoked = "revoked"
    refunded = "refunded"


class CommercialEntitlementSource(str, Enum):
    checkout = "checkout"
    admin_grant = "admin_grant"
    legacy_backfill = "legacy_backfill"
    legacy_migration = "legacy_migration"


class CommercialAccessRequestStatus(str, Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"
    canceled = "canceled"


class ACPWorkflowRunStatus(str, Enum):
    not_started = "not_started"
    running = "running"
    waiting_user = "waiting_user"
    blocked = "blocked"
    completed = "completed"
    completed_with_observations = "completed_with_observations"
    failed = "failed"
    stale = "stale"
    canceled = "canceled"


class ExportJobStatus(str, Enum):
    queued = "queued"
    running = "running"
    ready = "ready"
    failed = "failed"
    canceled = "canceled"
    expired = "expired"


class StageOperationStatus(str, Enum):
    queued = "queued"
    running = "running"
    waiting_for_user = "waiting_for_user"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class WorkspaceRole(str, Enum):
    owner = "owner"
    admin = "admin"
    editor = "editor"
    viewer = "viewer"


class PlatformRole(str, Enum):
    platform_admin = "platform_admin"
    platform_operator = "platform_operator"


class ArtifactStatus(str, Enum):
    draft = "draft"
    ready = "ready"
    needs_review = "needs_review"
    failed = "failed"


class JourneyArtifactState(str, Enum):
    generated = "generated"
    reviewed = "reviewed"
    approved = "approved"
    rejected = "rejected"
    stale = "stale"
    approved_legacy = "approved_legacy"
    needs_review_legacy = "needs_review_legacy"


class JourneyDecisionType(str, Enum):
    create = "create"
    patch = "patch"
    approve = "approve"
    reject = "reject"
    replace = "replace"
    mark_stale = "mark_stale"
    backfill_legacy = "backfill_legacy"


class EvidenceSource(str, Enum):
    form_input = "form_input"
    rule_engine = "rule_engine"
    llm_inference = "llm_inference"


class ReviewState(str, Enum):
    complete = "complete"
    partial = "partial"
    blocked = "blocked"


class KnowledgeScope(str, Enum):
    platform = "platform"
    workspace = "workspace"
    session = "session"


class KnowledgeVisibility(str, Enum):
    platform = "platform"
    workspace = "workspace"
    session = "session"
    restricted = "restricted"


class KnowledgeDocumentStatus(str, Enum):
    draft = "draft"
    approved = "approved"
    disabled = "disabled"
    expired = "expired"


class ApprovalStatus(str, Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"


class LLMProviderKey(str, Enum):
    openai = "openai"
    deepseek = "deepseek"
    codex_local = "codex_local"
    antigravity_cli = "antigravity_cli"


class LLMBudgetScopeType(str, Enum):
    workspace = "workspace"
    user = "user"
    project = "project"
    initiative = "initiative"
    stage = "stage"
    provider = "provider"
    model = "model"


class LLMBudgetPeriodType(str, Enum):
    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"
    custom = "custom"


class EstimationMaturityStage(str, Enum):
    canvas = "canvas"
    blueprint = "blueprint"
    ready_to_build = "ready_to_build"


class EstimationScenarioType(str, Enum):
    traditional = "traditional"
    agentic = "agentic"


class EstimationComplexityLevel(str, Enum):
    simple = "simple"
    moderate = "moderate"
    complex = "complex"
    critical = "critical"


class EstimationConfidenceLabel(str, Enum):
    low = "low"
    medium_low = "medium_low"
    medium = "medium"
    medium_high = "medium_high"
    high = "high"


class CodexLocalCostPolicy(str, Enum):
    marginal_only = "marginal_only"
    fully_loaded = "fully_loaded"
    hybrid = "hybrid"


class CodexAuthMode(str, Enum):
    auto = "auto"
    api_key = "api_key"
    access_token = "access_token"
    chatgpt_session = "chatgpt_session"
    profile = "profile"


class AgentExecutionBackend(str, Enum):
    provider_native = "provider_native"
    codex_cli = "codex_cli"
    shadow_codex_cli = "shadow_codex_cli"
    antigravity_cli = "antigravity_cli"
    shadow_antigravity_cli = "shadow_antigravity_cli"


class KnowledgeAccessBackend(str, Enum):
    inline_context = "inline_context"
    workspace_staged = "workspace_staged"
    hybrid = "hybrid"


class RuntimeGovernanceScopeType(str, Enum):
    platform = "platform"
    workspace = "workspace"


class RuntimeProviderReleaseStage(str, Enum):
    preview = "preview"
    general_availability = "general_availability"
    deprecated = "deprecated"


class RuntimeSecretStatus(str, Enum):
    not_configured = "not_configured"
    configured = "configured"
    invalid = "invalid"
    rotating = "rotating"


ALLOWED_DELIVERABLE_KEYS = {
    "prd",
    "technical_spec",
    "system_prompt",
    "skill_spec",
    "tool_schema",
    "state_flow",
    "decision_trace",
    "component_checklist",
    "test_cases",
    "risk_matrix",
    "mvp_backlog",
    "evolution_roadmap",
}

ALLOWED_COMPONENT_KEYS = {"tools", "memory", "knowledge", "security", "llm_policy"}
ALLOWED_PATTERN_FAMILIES = {"architecture", "reasoning", "memory"}
ALLOWED_ACP_FILE_STATUSES = {"complete", "incomplete", "needs_review"}
ALLOWED_ACP_VALIDATION_SEVERITIES = {"info", "warning", "error"}
ALLOWED_CONSTRUCTION_GAP_SEVERITIES = {"info", "warning", "blocking"}
ALLOWED_CONSTRUCTION_GAP_STATUSES = {"open", "answered", "waived", "resolved"}
ALLOWED_CONSTRUCTION_QUESTION_STATUSES = {"open", "answered", "resolved"}
ALLOWED_CONSTRUCTION_READINESS_STATUSES = {"not_started", "needs_questions", "blocked", "ready_to_build"}


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        candidates = value.split(",")
    elif isinstance(value, list):
        candidates = value
    else:
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        token = str(item).strip()
        if not token:
            continue
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(token)
    return normalized


class OpenAIProviderConfig(ContractModel):
    fast_model: str = ""
    reasoning_model: str = ""
    reasoning_effort: str = "low"
    api_key_configured: bool = False
    available: bool = False
    secret_source: str = "platform_managed"
    last_rotated_at: datetime | None = None
    health_status: str = "platform_missing"
    status_note: str = ""


class DeepSeekProviderConfig(ContractModel):
    base_url: str = "https://api.deepseek.com"
    fast_model: str = ""
    reasoning_model: str = ""
    reasoning_effort: str = "high"
    api_key_configured: bool = False
    available: bool = False
    secret_source: str = "platform_managed"
    last_rotated_at: datetime | None = None
    health_status: str = "platform_missing"
    status_note: str = ""


class CodexLocalProviderConfig(ContractModel):
    command: str = "codex"
    model: str = ""
    profile: str = ""
    cost_policy: CodexLocalCostPolicy = CodexLocalCostPolicy.hybrid
    timeout_ms: int = 150000
    max_concurrency: int = 1
    runner_id: str = "local"
    auth_mode: CodexAuthMode = CodexAuthMode.auto
    fallback_models: list[str] = PydanticField(default_factory=list)
    primary_agents: list[str] = PydanticField(default_factory=list)
    shadow_agents: list[str] = PydanticField(default_factory=list)
    staged_agents: list[str] = PydanticField(default_factory=list)
    available: bool = False
    executable_found: bool = False
    secret_source: str = "local_runtime"
    last_rotated_at: datetime | None = None
    health_status: str = "local_runtime_missing"
    status_note: str = ""

    @field_validator("fallback_models", "primary_agents", "shadow_agents", "staged_agents", mode="before")
    @classmethod
    def normalize_string_lists(cls, value: Any) -> list[str]:
        return _coerce_string_list(value)


class AntigravityProviderConfig(ContractModel):
    """Configuracion del proveedor Antigravity CLI (agy)."""

    executable: str = "agy"
    model: str = ""
    effort: str = "high"
    timeout_ms: int = 1200000
    max_concurrency: int = 1
    runner_id: str = "local-antigravity-cli"
    auth_mode: str = "auto"
    fallback_models: list[str] = PydanticField(default_factory=list)
    primary_agents: list[str] = PydanticField(default_factory=list)
    shadow_agents: list[str] = PydanticField(default_factory=list)
    staged_agents: list[str] = PydanticField(default_factory=list)
    available: bool = False
    executable_found: bool = False
    secret_source: str = "local_runtime"
    last_rotated_at: datetime | None = None
    health_status: str = "local_runtime_missing"
    status_note: str = ""

    @field_validator("fallback_models", "primary_agents", "shadow_agents", "staged_agents", mode="before")
    @classmethod
    def normalize_string_lists(cls, value: Any) -> list[str]:
        return _coerce_string_list(value)


class AntigravityProviderConfigUpdate(ContractModel):
    """Payload editable de configuracion del proveedor Antigravity CLI."""

    executable: str = "agy"
    model: str = ""
    effort: str = "high"
    timeout_ms: int = 1200000
    max_concurrency: int = 1
    runner_id: str = "local-antigravity-cli"
    auth_mode: str = "auto"
    fallback_models: list[str] = PydanticField(default_factory=list)
    primary_agents: list[str] = PydanticField(default_factory=list)
    shadow_agents: list[str] = PydanticField(default_factory=list)
    staged_agents: list[str] = PydanticField(default_factory=list)

    @field_validator("fallback_models", "primary_agents", "shadow_agents", "staged_agents", mode="before")
    @classmethod
    def normalize_string_lists(cls, value: Any) -> list[str]:
        return _coerce_string_list(value)


class LLMProviderOption(ContractModel):
    key: str = ""
    label: str = ""
    description: str = ""
    configured: bool = False
    reachable: bool = False
    selected: bool = False
    supports_structured_output: bool = True
    metadata: dict[str, Any] = PydanticField(default_factory=dict)


class MemoryRolloutPhaseEntry(ContractModel):
    phase_key: str = ""
    label: str = ""
    description: str = ""
    enabled: bool = False
    stage_keys: list[str] = PydanticField(default_factory=list)


class MemoryRolloutStageEntry(ContractModel):
    stage_key: str = ""
    label: str = ""
    phase_key: str = ""
    enabled: bool = False
    expects_llm_call: bool = False
    requested_backend: str = ""
    effective_backend: str = ""


class MemoryRolloutSummary(ContractModel):
    status: str = "not_ready"
    manifest_ready: bool = False
    requested_backend: str = ""
    effective_default_backend: str = ""
    phases: list[MemoryRolloutPhaseEntry] = PydanticField(default_factory=list)
    stages: list[MemoryRolloutStageEntry] = PydanticField(default_factory=list)
    notes: list[str] = PydanticField(default_factory=list)


class LLMRuntimeSettings(ContractModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True, validate_assignment=True)

    active_provider: LLMProviderKey = LLMProviderKey.openai
    agent_execution_backend: AgentExecutionBackend = AgentExecutionBackend.provider_native
    knowledge_access_backend: KnowledgeAccessBackend = KnowledgeAccessBackend.workspace_staged
    uses_platform_credentials: bool = True
    openai: OpenAIProviderConfig = PydanticField(default_factory=OpenAIProviderConfig)
    deepseek: DeepSeekProviderConfig = PydanticField(default_factory=DeepSeekProviderConfig)
    codex_local: CodexLocalProviderConfig = PydanticField(default_factory=CodexLocalProviderConfig)
    antigravity_cli: AntigravityProviderConfig = PydanticField(
        default_factory=AntigravityProviderConfig,
        validation_alias=AliasChoices("antigravity_cli", "antigravity"),
    )
    provider_options: list[LLMProviderOption] = PydanticField(default_factory=list)
    memory_rollout: MemoryRolloutSummary | None = None
    compatibility_mode: str = "backward_compatible"
    field_origins: dict[str, str] = PydanticField(default_factory=dict)
    updated_at: datetime | None = None

    @property
    def antigravity(self) -> AntigravityProviderConfig:
        return self.antigravity_cli


class OpenAIProviderConfigUpdate(ContractModel):
    fast_model: str = ""
    reasoning_model: str = ""
    reasoning_effort: str = "low"


class DeepSeekProviderConfigUpdate(ContractModel):
    base_url: str = "https://api.deepseek.com"
    fast_model: str = ""
    reasoning_model: str = ""
    reasoning_effort: str = "high"


class CodexLocalProviderConfigUpdate(ContractModel):
    command: str = "codex"
    model: str = ""
    profile: str = ""
    cost_policy: CodexLocalCostPolicy = CodexLocalCostPolicy.hybrid
    timeout_ms: int = 150000
    max_concurrency: int = 1
    runner_id: str = "local"
    auth_mode: CodexAuthMode = CodexAuthMode.auto
    fallback_models: list[str] = PydanticField(default_factory=list)
    primary_agents: list[str] = PydanticField(default_factory=list)
    shadow_agents: list[str] = PydanticField(default_factory=list)
    staged_agents: list[str] = PydanticField(default_factory=list)

    @field_validator("fallback_models", "primary_agents", "shadow_agents", "staged_agents", mode="before")
    @classmethod
    def normalize_string_lists(cls, value: Any) -> list[str]:
        return _coerce_string_list(value)


class LLMRuntimeSettingsUpdateRequest(ContractModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True, validate_assignment=True)

    active_provider: LLMProviderKey = LLMProviderKey.openai
    agent_execution_backend: AgentExecutionBackend = AgentExecutionBackend.provider_native
    knowledge_access_backend: KnowledgeAccessBackend = KnowledgeAccessBackend.workspace_staged
    uses_platform_credentials: bool | None = None
    openai: OpenAIProviderConfigUpdate = PydanticField(default_factory=OpenAIProviderConfigUpdate)
    deepseek: DeepSeekProviderConfigUpdate = PydanticField(default_factory=DeepSeekProviderConfigUpdate)
    codex_local: CodexLocalProviderConfigUpdate = PydanticField(default_factory=CodexLocalProviderConfigUpdate)
    antigravity_cli: AntigravityProviderConfigUpdate = PydanticField(
        default_factory=AntigravityProviderConfigUpdate,
        validation_alias=AliasChoices("antigravity_cli", "antigravity"),
    )

    @property
    def antigravity(self) -> AntigravityProviderConfigUpdate:
        return self.antigravity_cli


class WorkspaceProviderSecretUpsertRequest(ContractModel):
    secret_value: str = ""
    secret_ref: str = ""
    secret_kind: str = "api_key"
    activate_for_runtime: bool = True

    @model_validator(mode="after")
    def validate_secret_input(self) -> "WorkspaceProviderSecretUpsertRequest":
        if not self.secret_value.strip() and not self.secret_ref.strip():
            raise ValueError("Debes enviar secret_value o secret_ref.")
        return self


class WorkspaceProviderSecretResponse(ContractModel):
    provider_key: LLMProviderKey = LLMProviderKey.openai
    workspace_id: UUID
    secret_kind: str = "api_key"
    configured: bool = False
    uses_platform_credentials: bool = True
    secret_source: str = "platform_managed"
    status: RuntimeSecretStatus = RuntimeSecretStatus.not_configured
    health_status: str = "platform_missing"
    last_rotated_at: datetime | None = None
    updated_at: datetime | None = None
    active_for_runtime: bool = False
    storage_mode: str = "none"
    supports_workspace_secrets: bool = True


class PlatformRuntimeProviderResponse(ContractModel):
    provider_key: LLMProviderKey = LLMProviderKey.openai
    label: str = ""
    is_enabled: bool = True
    allowed_models: list[str] = PydanticField(default_factory=list)
    default_models: dict[str, Any] = PydanticField(default_factory=dict)
    allowed_auth_modes: list[str] = PydanticField(default_factory=list)
    supports_workspace_secrets: bool = True
    supports_platform_managed_credentials: bool = True
    release_stage: RuntimeProviderReleaseStage = RuntimeProviderReleaseStage.general_availability
    health_policy: dict[str, Any] = PydanticField(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class PlatformRuntimeProviderUpdateRequest(ContractModel):
    label: str | None = None
    is_enabled: bool | None = None
    allowed_models: list[str] | None = None
    default_models: dict[str, Any] | None = None
    allowed_auth_modes: list[str] | None = None
    supports_workspace_secrets: bool | None = None
    supports_platform_managed_credentials: bool | None = None
    release_stage: RuntimeProviderReleaseStage | None = None
    health_policy: dict[str, Any] | None = None


class RuntimeSettingsAuditEntry(ContractModel):
    id: UUID
    scope_type: RuntimeGovernanceScopeType = RuntimeGovernanceScopeType.workspace
    scope_id: str = ""
    change_type: str = ""
    before_payload_redacted: dict[str, Any] = PydanticField(default_factory=dict)
    after_payload_redacted: dict[str, Any] = PydanticField(default_factory=dict)
    actor_user_id: UUID | None = None
    actor_email: str = ""
    created_at: datetime


class RuntimeSettingsAuditListResponse(ContractModel):
    items: list[RuntimeSettingsAuditEntry] = PydanticField(default_factory=list)


class WorkspaceRuntimeHealthCheckEntry(ContractModel):
    check_key: str = ""
    label: str = ""
    status: str = "unknown"
    detail: str = ""


class WorkspaceRuntimeHealthResponse(ContractModel):
    workspace_id: UUID
    mode: str = "health"
    overall_status: str = "unknown"
    provider_key: LLMProviderKey = LLMProviderKey.openai
    provider_label: str = ""
    secret_source: str = "platform_managed"
    health_status: str = "platform_missing"
    uses_platform_credentials: bool = True
    agent_execution_backend: AgentExecutionBackend = AgentExecutionBackend.provider_native
    knowledge_access_backend: KnowledgeAccessBackend = KnowledgeAccessBackend.workspace_staged
    checked_at: datetime
    checks: list[WorkspaceRuntimeHealthCheckEntry] = PydanticField(default_factory=list)


class WorkspaceRecord(SQLModel, table=True):
    __tablename__ = "workspaces"

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    name: str = Field(index=True)
    slug: str = Field(index=True, unique=True)
    created_by_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True)
    is_active: bool = Field(default=True, nullable=False)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class SessionRecord(SQLModel, table=True):
    __tablename__ = "sessions"

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    user_id: UUID = Field(foreign_key="users.id", index=True)
    workspace_id: UUID | None = Field(default=None, foreign_key="workspaces.id", index=True, nullable=True)
    title: str = Field(default="Nueva sesion")
    suggested_title: str | None = Field(default=None, nullable=True)
    title_source: ProjectTitleSource = Field(default=ProjectTitleSource.generated, nullable=False)
    row_version: int = Field(default=1, nullable=False)
    status: ArtifactStatus = Field(default=ArtifactStatus.draft)
    current_stage: SessionStage = Field(default=SessionStage.draft_capture)
    commercial_tier: CommercialTier = Field(default=CommercialTier.blueprint, nullable=False)
    selected_workflow_template_key: str = Field(default="", nullable=False)
    archived_at: datetime | None = Field(default=None, index=True, nullable=True)
    archived_by_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True)
    deleted_at: datetime | None = Field(default=None, index=True, nullable=True)
    deleted_by_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class ProductCatalogRecord(SQLModel, table=True):
    __tablename__ = "product_catalog"
    __table_args__ = (UniqueConstraint("product_key", "version", name="uq_product_catalog_key_version"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    product_key: str = Field(index=True, nullable=False)
    tier: CommercialTier = Field(default=CommercialTier.blueprint_pro, nullable=False)
    product_type: CommercialProductType = Field(default=CommercialProductType.blueprint, nullable=False)
    status: CommercialProductStatus = Field(default=CommercialProductStatus.active, nullable=False)
    name: str = Field(default="", nullable=False)
    description: str = Field(default="", nullable=False)
    scope: str = Field(default="project", nullable=False)
    benefits: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    exclusions: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    capabilities: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    metadata_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSON, nullable=False))
    version: int = Field(default=1, nullable=False, index=True)
    is_active: bool = Field(default=True, nullable=False)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class ProductPriceRecord(SQLModel, table=True):
    __tablename__ = "product_prices"
    __table_args__ = (UniqueConstraint("price_code", name="uq_product_price_code"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    product_key: str = Field(index=True, nullable=False)
    price_code: str = Field(index=True, nullable=False)
    currency: str = Field(default="USD", nullable=False)
    unit_amount_cents: int = Field(default=0, nullable=False)
    unit_amount_usd_cents: int = Field(default=0, nullable=False)
    billing_period: str = Field(default="one_time", nullable=False)
    status: CommercialPriceStatus = Field(default=CommercialPriceStatus.active, nullable=False)
    version: int = Field(default=1, nullable=False, index=True)
    metadata_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class CommercialOrderRecord(SQLModel, table=True):
    __tablename__ = "commercial_orders"
    __table_args__ = (UniqueConstraint("checkout_ref", name="uq_commercial_order_checkout_ref"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    session_id: UUID | None = Field(default=None, foreign_key="sessions.id", index=True, nullable=True)
    buyer_user_id: UUID = Field(foreign_key="users.id", index=True)
    status: CommercialOrderStatus = Field(default=CommercialOrderStatus.pending, nullable=False)
    currency: str = Field(default="COP", nullable=False)
    subtotal_cents: int = Field(default=0, nullable=False)
    tax_cents: int = Field(default=0, nullable=False)
    total_cents: int = Field(default=0, nullable=False)
    provider: str = Field(default="sandbox", nullable=False)
    checkout_ref: str = Field(default="", index=True, nullable=False)
    checkout_url: str = Field(default="", nullable=False)
    idempotency_key: str = Field(default="", index=True, nullable=False)
    metadata_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)
    paid_at: datetime | None = Field(default=None, nullable=True)


class CommercialOrderLineRecord(SQLModel, table=True):
    __tablename__ = "commercial_order_lines"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    order_id: UUID = Field(foreign_key="commercial_orders.id", index=True)
    product_key: str = Field(index=True, nullable=False)
    price_code: str = Field(default="", nullable=False)
    quantity: int = Field(default=1, nullable=False)
    unit_amount_cents: int = Field(default=0, nullable=False)
    total_amount_cents: int = Field(default=0, nullable=False)
    metadata_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class CommercialPaymentRecord(SQLModel, table=True):
    __tablename__ = "commercial_payments"
    __table_args__ = (UniqueConstraint("provider", "provider_payment_id", name="uq_commercial_payment_provider_id"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    session_id: UUID | None = Field(default=None, foreign_key="sessions.id", index=True, nullable=True)
    order_id: UUID = Field(foreign_key="commercial_orders.id", index=True)
    provider: str = Field(default="sandbox", nullable=False)
    provider_payment_id: str = Field(default="", nullable=False)
    provider_checkout_ref: str = Field(default="", index=True, nullable=False)
    status: CommercialPaymentStatus = Field(default=CommercialPaymentStatus.pending, nullable=False)
    amount_cents: int = Field(default=0, nullable=False)
    currency: str = Field(default="COP", nullable=False)
    idempotency_key: str = Field(default="", index=True, nullable=False)
    metadata_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class CommercialEntitlementRecord(SQLModel, table=True):
    __tablename__ = "commercial_entitlements"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    session_id: UUID | None = Field(default=None, foreign_key="sessions.id", index=True, nullable=True)
    product_key: str = Field(index=True, nullable=False)
    tier: CommercialTier = Field(default=CommercialTier.blueprint_pro, nullable=False)
    status: CommercialEntitlementStatus = Field(default=CommercialEntitlementStatus.active, nullable=False)
    source: CommercialEntitlementSource = Field(default=CommercialEntitlementSource.checkout, nullable=False)
    order_id: UUID | None = Field(default=None, foreign_key="commercial_orders.id", nullable=True)
    order_line_id: UUID | None = Field(default=None, foreign_key="commercial_order_lines.id", nullable=True)
    payment_id: UUID | None = Field(default=None, foreign_key="commercial_payments.id", nullable=True)
    granted_by_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True)
    revoked_by_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True)
    starts_at: datetime = Field(default_factory=utc_now, nullable=False)
    ends_at: datetime | None = Field(default=None, nullable=True)
    version: int = Field(default=1, nullable=False, index=True)
    metadata_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class CommercialAccessRequestRecord(SQLModel, table=True):
    __tablename__ = "commercial_access_requests"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    requester_user_id: UUID = Field(foreign_key="users.id", index=True)
    resolver_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True)
    capability: str = Field(default="", index=True, nullable=False)
    product_key: str = Field(default="", index=True, nullable=False)
    target_tier: CommercialTier = Field(default=CommercialTier.blueprint_pro, nullable=False)
    status: CommercialAccessRequestStatus = Field(default=CommercialAccessRequestStatus.pending, nullable=False)
    reason: str = Field(default="", nullable=False)
    resolution_note: str = Field(default="", nullable=False)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)
    resolved_at: datetime | None = Field(default=None, nullable=True)


class CommercialEventRecord(SQLModel, table=True):
    __tablename__ = "commercial_events"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    session_id: UUID | None = Field(default=None, foreign_key="sessions.id", index=True, nullable=True)
    user_id: UUID | None = Field(default=None, foreign_key="users.id", index=True, nullable=True)
    event_key: str = Field(default="", index=True, nullable=False)
    product_key: str = Field(default="", index=True, nullable=False)
    source: str = Field(default="", nullable=False)
    correlation_id: str = Field(default="", index=True, nullable=False)
    revenue_cents: int = Field(default=0, nullable=False)
    currency: str = Field(default="", nullable=False)
    metadata_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class HotmartIntegrationConfigRecord(SQLModel, table=True):
    __tablename__ = "hotmart_integration_configs"
    __table_args__ = (UniqueConstraint("workspace_id", "environment", name="uq_hotmart_config_workspace_environment"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    environment: str = Field(default="sandbox", index=True, nullable=False)
    enabled: bool = Field(default=False, nullable=False)
    status: str = Field(default="not_configured", index=True, nullable=False)
    client_id_configured: bool = Field(default=False, nullable=False)
    client_secret_configured: bool = Field(default=False, nullable=False)
    basic_token_configured: bool = Field(default=False, nullable=False)
    hottok_configured: bool = Field(default=False, nullable=False)
    api_base_url: str = Field(default="", nullable=False)
    auth_base_url: str = Field(default="", nullable=False)
    webhook_public_url: str = Field(default="", nullable=False)
    last_health_check_at: datetime | None = Field(default=None, nullable=True)
    last_health_status: str = Field(default="", nullable=False)
    last_health_message: str = Field(default="", nullable=False)
    last_sync_at: datetime | None = Field(default=None, nullable=True)
    metadata_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSON, nullable=False))
    created_by_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True)
    updated_by_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class HotmartIntegrationSecretRecord(SQLModel, table=True):
    __tablename__ = "hotmart_integration_secrets"
    __table_args__ = (
        UniqueConstraint("workspace_id", "environment", "secret_kind", name="uq_hotmart_secret_workspace_environment_kind"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    environment: str = Field(default="sandbox", index=True, nullable=False)
    secret_kind: str = Field(default="", index=True, nullable=False)
    secret_ciphertext: str = Field(default="", nullable=False)
    secret_ref: str = Field(default="", nullable=False)
    status: str = Field(default="not_configured", nullable=False)
    last_rotated_at: datetime | None = Field(default=None, nullable=True)
    metadata_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class HotmartProductMappingRecord(SQLModel, table=True):
    __tablename__ = "hotmart_product_mappings"
    __table_args__ = (
        UniqueConstraint("workspace_id", "internal_product_key", "environment", name="uq_hotmart_mapping_workspace_product_environment"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    environment: str = Field(default="sandbox", index=True, nullable=False)
    internal_product_key: str = Field(default="", index=True, nullable=False)
    hotmart_product_id: str = Field(default="", index=True, nullable=False)
    hotmart_product_ucode: str = Field(default="", nullable=False)
    offer_code: str = Field(default="", nullable=False)
    plan_code: str = Field(default="", nullable=False)
    billing_mode: str = Field(default="one_time", nullable=False)
    currency: str = Field(default="USD", nullable=False)
    internal_base_currency: str = Field(default="USD", nullable=False)
    internal_unit_amount_usd_cents: int = Field(default=0, nullable=False)
    hotmart_price_strategy: str = Field(default="internal_net_amount", nullable=False)
    trm_policy: str = Field(default="display_only", nullable=False)
    grants_tier: CommercialTier = Field(default=CommercialTier.blueprint_pro, nullable=False)
    entitlement_scope: str = Field(default="project", nullable=False)
    is_active: bool = Field(default=True, nullable=False)
    metadata_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class HotmartPaymentLinkRecord(SQLModel, table=True):
    __tablename__ = "hotmart_payment_links"
    __table_args__ = (UniqueConstraint("workspace_id", "hotmart_payment_link_id", name="uq_hotmart_payment_link_provider_id"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    order_id: UUID | None = Field(default=None, foreign_key="commercial_orders.id", index=True, nullable=True)
    created_by_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True)
    environment: str = Field(default="sandbox", index=True, nullable=False)
    internal_product_key: str = Field(default="", index=True, nullable=False)
    hotmart_payment_link_id: str = Field(default="", index=True, nullable=False)
    checkout_url: str = Field(default="", nullable=False)
    activation_status: str = Field(default="draft", index=True, nullable=False)
    provider_ref: str = Field(default="", index=True, nullable=False)
    gross_amount_cents: int = Field(default=0, nullable=False)
    discount_amount_cents: int = Field(default=0, nullable=False)
    net_amount_cents: int = Field(default=0, nullable=False)
    currency: str = Field(default="USD", nullable=False)
    internal_unit_amount_usd_cents: int = Field(default=0, nullable=False)
    trm_cop_applied: float | None = Field(default=None, nullable=True)
    discount_origin: str = Field(default="", nullable=False)
    request_payload_redacted: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    response_payload_redacted: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    expires_at: datetime | None = Field(default=None, nullable=True)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class HotmartPromotionRecord(SQLModel, table=True):
    __tablename__ = "hotmart_promotions"
    __table_args__ = (UniqueConstraint("workspace_id", "environment", "coupon_code", name="uq_hotmart_promotion_coupon"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    environment: str = Field(default="sandbox", index=True, nullable=False)
    internal_campaign_key: str = Field(default="", index=True, nullable=False)
    internal_product_key: str = Field(default="", index=True, nullable=False)
    hotmart_product_id: str = Field(default="", index=True, nullable=False)
    offer_codes: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    coupon_id: str = Field(default="", index=True, nullable=False)
    coupon_code: str = Field(default="", index=True, nullable=False)
    discount_percent: float = Field(default=0.0, nullable=False)
    discount_origin: str = Field(default="provider_coupon", nullable=False)
    discount_type: str = Field(default="percent", nullable=False)
    discount_amount_cents: int | None = Field(default=None, nullable=True)
    starts_at: datetime | None = Field(default=None, nullable=True)
    ends_at: datetime | None = Field(default=None, nullable=True)
    status: str = Field(default="draft", index=True, nullable=False)
    published_at: datetime | None = Field(default=None, nullable=True)
    created_by_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True)
    metadata_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class HotmartSyncRunRecord(SQLModel, table=True):
    __tablename__ = "hotmart_sync_runs"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    environment: str = Field(default="sandbox", index=True, nullable=False)
    resource: str = Field(default="", index=True, nullable=False)
    status: str = Field(default="idle", index=True, nullable=False)
    started_by_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True)
    started_at: datetime = Field(default_factory=utc_now, nullable=False)
    finished_at: datetime | None = Field(default=None, nullable=True)
    cursor_before: str = Field(default="", nullable=False)
    cursor_after: str = Field(default="", nullable=False)
    records_read: int = Field(default=0, nullable=False)
    records_created: int = Field(default=0, nullable=False)
    records_updated: int = Field(default=0, nullable=False)
    records_skipped: int = Field(default=0, nullable=False)
    error_summary: str = Field(default="", nullable=False)
    metadata_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSON, nullable=False))


class HotmartSyncCursorRecord(SQLModel, table=True):
    __tablename__ = "hotmart_sync_cursors"
    __table_args__ = (UniqueConstraint("workspace_id", "environment", "resource", name="uq_hotmart_sync_cursor_resource"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    environment: str = Field(default="sandbox", index=True, nullable=False)
    resource: str = Field(default="", index=True, nullable=False)
    page_token: str = Field(default="", nullable=False)
    last_event_at: datetime | None = Field(default=None, nullable=True)
    last_transaction: str = Field(default="", nullable=False)
    last_success_at: datetime | None = Field(default=None, nullable=True)
    metadata_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class HotmartWebhookEventRecord(SQLModel, table=True):
    __tablename__ = "hotmart_webhook_events"
    __table_args__ = (UniqueConstraint("event_id", name="uq_hotmart_webhook_event_id"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    event_id: str = Field(default="", index=True, nullable=False)
    event_type: str = Field(default="", index=True, nullable=False)
    transaction: str = Field(default="", index=True, nullable=False)
    workspace_id: UUID | None = Field(default=None, foreign_key="workspaces.id", index=True, nullable=True)
    order_id: UUID | None = Field(default=None, foreign_key="commercial_orders.id", index=True, nullable=True)
    payment_id: UUID | None = Field(default=None, foreign_key="commercial_payments.id", nullable=True)
    hottok_validated: bool = Field(default=False, nullable=False)
    processing_status: str = Field(default="received", index=True, nullable=False)
    payload_hash: str = Field(default="", index=True, nullable=False)
    payload_redacted: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    error_code: str = Field(default="", nullable=False)
    error_message: str = Field(default="", nullable=False)
    retries: int = Field(default=0, nullable=False)
    processed_at: datetime | None = Field(default=None, nullable=True)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class HotmartReconciliationIssueRecord(SQLModel, table=True):
    __tablename__ = "hotmart_reconciliation_issues"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    environment: str = Field(default="sandbox", index=True, nullable=False)
    issue_type: str = Field(default="", index=True, nullable=False)
    severity: str = Field(default="medium", index=True, nullable=False)
    status: str = Field(default="open", index=True, nullable=False)
    provider_ref: str = Field(default="", index=True, nullable=False)
    internal_ref: str = Field(default="", index=True, nullable=False)
    summary: str = Field(default="", nullable=False)
    suggested_action: str = Field(default="", nullable=False)
    resolution_action: str = Field(default="", nullable=False)
    resolution_note: str = Field(default="", nullable=False)
    resolved_by_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True)
    resolved_at: datetime | None = Field(default=None, nullable=True)
    metadata_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class ACPBuildRunRecord(SQLModel, table=True):
    __tablename__ = "acp_build_runs"
    __table_args__ = (UniqueConstraint("workspace_id", "session_id", "idempotency_key", name="uq_acp_build_run_idempotency"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    created_by_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True)
    blueprint_version_number: int | None = Field(default=None, nullable=True)
    status: ACPWorkflowRunStatus = Field(default=ACPWorkflowRunStatus.not_started, nullable=False, index=True)
    current_phase_key: str = Field(default="", index=True, nullable=False)
    phase_order: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    progress_percent: int = Field(default=0, nullable=False)
    checkpoints: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    artifacts: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    blockers: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    warnings: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    idempotency_key: str = Field(default="", index=True, nullable=False)
    started_at: datetime | None = Field(default=None, nullable=True)
    completed_at: datetime | None = Field(default=None, nullable=True)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class ACPPhaseRunRecord(SQLModel, table=True):
    __tablename__ = "acp_phase_runs"
    __table_args__ = (UniqueConstraint("run_id", "phase_key", name="uq_acp_phase_run_phase"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    run_id: UUID = Field(foreign_key="acp_build_runs.id", index=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    phase_key: str = Field(index=True, nullable=False)
    phase_label: str = Field(default="", nullable=False)
    phase_order: int = Field(default=0, nullable=False)
    status: ACPWorkflowRunStatus = Field(default=ACPWorkflowRunStatus.not_started, nullable=False, index=True)
    attempt_count: int = Field(default=0, nullable=False)
    input_refs: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    output_refs: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    checkpoints: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    blockers: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    warnings: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    idempotency_key: str = Field(default="", index=True, nullable=False)
    started_at: datetime | None = Field(default=None, nullable=True)
    completed_at: datetime | None = Field(default=None, nullable=True)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class ExportJobRecord(SQLModel, table=True):
    __tablename__ = "export_jobs"
    __table_args__ = (UniqueConstraint("workspace_id", "session_id", "idempotency_key", name="uq_export_job_idempotency"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True)
    product_key: str = Field(default="", index=True, nullable=False)
    profile: str = Field(default="", index=True, nullable=False)
    artifact_kind: str = Field(default="", index=True, nullable=False)
    status: ExportJobStatus = Field(default=ExportJobStatus.queued, nullable=False, index=True)
    idempotency_key: str = Field(default="", index=True, nullable=False)
    content_type: str = Field(default="application/json", nullable=False)
    file_name: str = Field(default="", nullable=False)
    storage_key: str = Field(default="", index=True, nullable=False)
    checksum_sha256: str = Field(default="", nullable=False)
    size_bytes: int = Field(default=0, nullable=False)
    expires_at: datetime | None = Field(default=None, nullable=True)
    error_message: str = Field(default="", nullable=False)
    metadata_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)
    completed_at: datetime | None = Field(default=None, nullable=True)


class ACPLaunchReportRecord(SQLModel, table=True):
    __tablename__ = "acp_launch_reports"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True)
    report_path: str = Field(default="ACP/launcher/launch-report.json", nullable=False)
    launcher_version: str = Field(default="", index=True, nullable=False)
    detected_tool: str = Field(default="", index=True, nullable=False)
    detected_ide: str = Field(default="", index=True, nullable=False)
    status: str = Field(default="received", index=True, nullable=False)
    summary: str = Field(default="", nullable=False)
    report_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column("report", JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class UserRecord(SQLModel, table=True):
    __tablename__ = "users"

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    email: str = Field(index=True, unique=True)
    full_name: str
    password_hash: str
    phone_number: str | None = Field(default=None, nullable=True)
    preferred_currency: str = Field(default="COP", nullable=False)
    preferred_language: str = Field(default="es", nullable=False)
    email_verified: bool = Field(default=False, nullable=False)
    email_verified_at: datetime | None = Field(default=None, nullable=True)
    verification_code: str | None = Field(default=None, nullable=True)
    consent_system_notifications: bool = Field(default=False, nullable=False)
    consent_commercial_promotions: bool = Field(default=False, nullable=False)
    consent_events_newsletters: bool = Field(default=False, nullable=False)
    default_workspace_id: UUID | None = Field(default=None, foreign_key="workspaces.id", nullable=True)
    is_active: bool = Field(default=True, nullable=False)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class UserLegalAcceptanceRecord(SQLModel, table=True):
    __tablename__ = "user_legal_acceptances"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(foreign_key="users.id", index=True, nullable=False)
    document_type: str = Field(index=True, nullable=False)
    document_version: str = Field(default="v1.0-2026-08", nullable=False)
    accepted: bool = Field(default=True, nullable=False)
    accepted_at: datetime = Field(default_factory=utc_now, nullable=False)
    ip_address: str | None = Field(default=None, nullable=True)
    user_agent: str | None = Field(default=None, nullable=True)


class WorkspaceMembershipRecord(SQLModel, table=True):
    __tablename__ = "workspace_memberships"
    __table_args__ = (UniqueConstraint("workspace_id", "user_id", name="uq_workspace_membership"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    user_id: UUID = Field(foreign_key="users.id", index=True)
    role: WorkspaceRole = Field(default=WorkspaceRole.owner, nullable=False)
    is_active: bool = Field(default=True, nullable=False)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class AdminUserInvitationRecord(SQLModel, table=True):
    __tablename__ = "admin_user_invitations"
    __table_args__ = (
        Index("ix_admin_user_invitations_workspace_status", "workspace_id", "status", "created_at"),
        Index("ix_admin_user_invitations_email_status", "email", "status"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    email: str = Field(index=True, nullable=False)
    full_name: str = Field(default="", nullable=False)
    role: WorkspaceRole = Field(default=WorkspaceRole.viewer, nullable=False)
    status: str = Field(default="pending", index=True, nullable=False)
    invited_by_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True)
    accepted_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True)
    expires_at: datetime | None = Field(default=None, nullable=True)
    message: str = Field(default="", nullable=False)
    metadata_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class AuthTokenRecord(SQLModel, table=True):
    __tablename__ = "auth_tokens"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(foreign_key="users.id", index=True)
    token_hash: str = Field(index=True, unique=True)
    expires_at: datetime = Field(nullable=False)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    last_used_at: datetime = Field(default_factory=utc_now, nullable=False)


class PlatformRoleAssignmentRecord(SQLModel, table=True):
    __tablename__ = "platform_role_assignments"
    __table_args__ = (UniqueConstraint("user_id", "role", name="uq_platform_role_assignment"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(foreign_key="users.id", index=True)
    role: PlatformRole = Field(default=PlatformRole.platform_operator, nullable=False)
    is_active: bool = Field(default=True, nullable=False)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class PlatformRuntimeProviderRecord(SQLModel, table=True):
    __tablename__ = "platform_runtime_providers"
    __table_args__ = (UniqueConstraint("provider_key", name="uq_platform_runtime_provider_key"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    provider_key: LLMProviderKey = Field(default=LLMProviderKey.openai, index=True, nullable=False)
    label: str = Field(default="", nullable=False)
    is_enabled: bool = Field(default=True, nullable=False)
    allowed_models: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    default_models: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    allowed_auth_modes: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    supports_workspace_secrets: bool = Field(default=True, nullable=False)
    supports_platform_managed_credentials: bool = Field(default=True, nullable=False)
    release_stage: RuntimeProviderReleaseStage = Field(
        default=RuntimeProviderReleaseStage.general_availability,
        nullable=False,
    )
    health_policy: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class PlatformRuntimeDefaultsRecord(SQLModel, table=True):
    __tablename__ = "platform_runtime_defaults"
    __table_args__ = (UniqueConstraint("version", name="uq_platform_runtime_defaults_version"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    active_provider_default: LLMProviderKey = Field(default=LLMProviderKey.openai, nullable=False)
    agent_execution_backend_default: AgentExecutionBackend = Field(
        default=AgentExecutionBackend.provider_native,
        nullable=False,
    )
    knowledge_access_backend_default: KnowledgeAccessBackend = Field(
        default=KnowledgeAccessBackend.workspace_staged,
        nullable=False,
    )
    per_provider_defaults: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    is_active: bool = Field(default=True, nullable=False)
    version: int = Field(default=1, nullable=False, index=True)
    updated_by_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class WorkspaceRuntimeSettingsRecord(SQLModel, table=True):
    __tablename__ = "workspace_runtime_settings"
    __table_args__ = (UniqueConstraint("workspace_id", "version", name="uq_workspace_runtime_settings_version"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    active_provider: LLMProviderKey = Field(default=LLMProviderKey.openai, nullable=False)
    agent_execution_backend: AgentExecutionBackend = Field(
        default=AgentExecutionBackend.provider_native,
        nullable=False,
    )
    knowledge_access_backend: KnowledgeAccessBackend = Field(
        default=KnowledgeAccessBackend.workspace_staged,
        nullable=False,
    )
    provider_overrides: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    uses_platform_credentials: bool = Field(default=True, nullable=False)
    is_active: bool = Field(default=True, nullable=False)
    version: int = Field(default=1, nullable=False, index=True)
    updated_by_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class WorkspaceProviderSecretRecord(SQLModel, table=True):
    __tablename__ = "workspace_provider_secrets"
    __table_args__ = (
        UniqueConstraint("workspace_id", "provider_key", "secret_kind", name="uq_workspace_provider_secret"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    provider_key: LLMProviderKey = Field(default=LLMProviderKey.openai, index=True, nullable=False)
    secret_kind: str = Field(default="api_key", nullable=False)
    secret_ciphertext: str = Field(default="", nullable=False)
    secret_ref: str = Field(default="", nullable=False)
    status: RuntimeSecretStatus = Field(default=RuntimeSecretStatus.not_configured, nullable=False)
    last_rotated_at: datetime | None = Field(default=None, nullable=True)
    updated_by_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class RuntimeSettingsAuditRecord(SQLModel, table=True):
    __tablename__ = "runtime_settings_audit"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    scope_type: RuntimeGovernanceScopeType = Field(default=RuntimeGovernanceScopeType.workspace, nullable=False)
    scope_id: str = Field(default="", index=True, nullable=False)
    change_type: str = Field(default="", nullable=False)
    before_payload_redacted: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    after_payload_redacted: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    actor_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True)
    actor_email: str = Field(default="", nullable=False)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class LLMUsageLedgerRecord(SQLModel, table=True):
    __tablename__ = "llm_usage_ledger"
    __table_args__ = (
        Index("ix_llm_usage_ledger_workspace_started", "workspace_id", "started_at"),
        Index("ix_llm_usage_ledger_user_started", "workspace_id", "user_id", "started_at"),
        Index("ix_llm_usage_ledger_session_started", "workspace_id", "session_id", "started_at"),
        Index("ix_llm_usage_ledger_project_started", "workspace_id", "project_id", "started_at"),
        Index("ix_llm_usage_ledger_stage_capability_started", "workspace_id", "stage", "capability_key", "started_at"),
        Index("ix_llm_usage_ledger_provider_model_started", "workspace_id", "provider_key", "model_name", "started_at"),
        Index("ix_llm_usage_ledger_request_attempt", "request_id", "attempt_number"),
        Index("ix_llm_usage_ledger_operation", "operation_id"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID | None = Field(default=None, foreign_key="workspaces.id", index=True, nullable=True)
    user_id: UUID | None = Field(default=None, foreign_key="users.id", index=True, nullable=True)
    session_id: UUID | None = Field(default=None, foreign_key="sessions.id", index=True, nullable=True)
    project_id: UUID | None = Field(default=None, index=True, nullable=True)
    initiative_id: UUID | None = Field(default=None, index=True, nullable=True)
    stage: str = Field(default="", index=True, nullable=False)
    substage: str = Field(default="", index=True, nullable=False)
    agent_key: str = Field(default="", index=True, nullable=False)
    capability_key: str = Field(default="", index=True, nullable=False)
    action_key: str = Field(default="", index=True, nullable=False)
    operation_id: UUID | None = Field(default=None, index=True, nullable=True)
    parent_run_id: str = Field(default="", index=True, nullable=False)
    correlation_id: str = Field(default="", index=True, nullable=False)
    provider_key: str = Field(default="", index=True, nullable=False)
    model_name: str = Field(default="", index=True, nullable=False)
    requested_model: str = Field(default="", nullable=False)
    execution_backend: str = Field(default="", index=True, nullable=False)
    execution_mode: str = Field(default="primary", index=True, nullable=False)
    request_id: str = Field(default="", index=True, nullable=False)
    provider_request_id: str = Field(default="", index=True, nullable=False)
    attempt_number: int = Field(default=1, nullable=False)
    retry_count: int = Field(default=0, nullable=False)
    fallback_used: bool = Field(default=False, nullable=False)
    shadow_provider_key: str = Field(default="", index=True, nullable=False)
    status: str = Field(default="succeeded", index=True, nullable=False)
    failure_kind: str = Field(default="", index=True, nullable=False)
    failure_detail_redacted: str = Field(default="", nullable=False)
    started_at: datetime = Field(default_factory=utc_now, index=True, nullable=False)
    finished_at: datetime | None = Field(default=None, nullable=True)
    duration_ms: int = Field(default=0, nullable=False)
    queue_wait_ms: int = Field(default=0, nullable=False)
    input_tokens: int = Field(default=0, nullable=False)
    output_tokens: int = Field(default=0, nullable=False)
    total_tokens: int = Field(default=0, nullable=False)
    cached_input_tokens: int = Field(default=0, nullable=False)
    reasoning_tokens: int = Field(default=0, nullable=False)
    other_token_metrics: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    provider_metrics: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    cost_input: float = Field(default=0.0, nullable=False)
    cost_output: float = Field(default=0.0, nullable=False)
    cost_other: float = Field(default=0.0, nullable=False)
    cost_total: float = Field(default=0.0, index=True, nullable=False)
    currency: str = Field(default="USD", index=True, nullable=False)
    fx_rate: float = Field(default=1.0, nullable=False)
    pricing_profile_key: str = Field(default="", index=True, nullable=False)
    pricing_snapshot: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    usage_raw_redacted: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    prompt_hash: str = Field(default="", index=True, nullable=False)
    response_hash: str = Field(default="", index=True, nullable=False)
    schema_validation_status: str = Field(default="", index=True, nullable=False)
    finish_reason: str = Field(default="", nullable=False)
    value_signal: str = Field(default="", index=True, nullable=False)
    metadata_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class LLMBudgetPolicyRecord(SQLModel, table=True):
    __tablename__ = "llm_budget_policies"
    __table_args__ = (
        UniqueConstraint("workspace_id", "policy_key", name="uq_llm_budget_policy_workspace_key"),
        Index("ix_llm_budget_policies_workspace_scope", "workspace_id", "scope_type", "scope_value"),
        Index("ix_llm_budget_policies_active_period", "workspace_id", "is_active", "period_type"),
        Index("ix_llm_budget_policies_provider_model", "workspace_id", "provider_key", "model_name"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True, nullable=False)
    policy_key: str = Field(default="", index=True, nullable=False)
    name: str = Field(default="", nullable=False)
    description: str = Field(default="", nullable=False)
    scope_type: LLMBudgetScopeType = Field(default=LLMBudgetScopeType.workspace, index=True, nullable=False)
    scope_value: str = Field(default="", index=True, nullable=False)
    user_id: UUID | None = Field(default=None, foreign_key="users.id", index=True, nullable=True)
    project_id: UUID | None = Field(default=None, index=True, nullable=True)
    initiative_id: UUID | None = Field(default=None, index=True, nullable=True)
    stage: str = Field(default="", index=True, nullable=False)
    provider_key: str = Field(default="", index=True, nullable=False)
    model_name: str = Field(default="", index=True, nullable=False)
    period_type: LLMBudgetPeriodType = Field(default=LLMBudgetPeriodType.monthly, index=True, nullable=False)
    custom_period_start: datetime | None = Field(default=None, nullable=True)
    custom_period_end: datetime | None = Field(default=None, nullable=True)
    limit_amount: float = Field(default=0.0, nullable=False)
    currency: str = Field(default="USD", index=True, nullable=False)
    threshold_percentages: list[float] = Field(default_factory=lambda: [50.0, 80.0, 95.0, 100.0], sa_column=Column(JSON, nullable=False))
    hard_limit_percent: float = Field(default=100.0, nullable=False)
    is_active: bool = Field(default=True, index=True, nullable=False)
    created_by_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True)
    updated_by_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True)
    metadata_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class LLMFinOpsAlertRecord(SQLModel, table=True):
    __tablename__ = "llm_finops_alerts"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            "alert_key",
            "period_start",
            "period_end",
            name="uq_llm_finops_alert_period_key",
        ),
        Index("ix_llm_finops_alerts_workspace_status_created", "workspace_id", "status", "created_at"),
        Index("ix_llm_finops_alerts_policy_threshold", "budget_policy_id", "threshold_percent"),
        Index("ix_llm_finops_alerts_scope_period", "workspace_id", "scope_type", "scope_value", "period_start"),
        Index("ix_llm_finops_alerts_provider_model", "workspace_id", "provider_key", "model_name"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True, nullable=False)
    budget_policy_id: UUID | None = Field(default=None, foreign_key="llm_budget_policies.id", index=True, nullable=True)
    usage_record_id: UUID | None = Field(default=None, foreign_key="llm_usage_ledger.id", index=True, nullable=True)
    alert_key: str = Field(default="", index=True, nullable=False)
    alert_type: str = Field(default="", index=True, nullable=False)
    severity: str = Field(default="medium", index=True, nullable=False)
    title: str = Field(default="", nullable=False)
    message: str = Field(default="", nullable=False)
    status: str = Field(default="active", index=True, nullable=False)
    scope_type: str = Field(default="", index=True, nullable=False)
    scope_value: str = Field(default="", index=True, nullable=False)
    provider_key: str = Field(default="", index=True, nullable=False)
    model_name: str = Field(default="", index=True, nullable=False)
    stage: str = Field(default="", index=True, nullable=False)
    threshold_percent: float = Field(default=0.0, nullable=False)
    period_start: datetime = Field(index=True, nullable=False)
    period_end: datetime = Field(index=True, nullable=False)
    consumed_amount: float = Field(default=0.0, nullable=False)
    limit_amount: float = Field(default=0.0, nullable=False)
    currency: str = Field(default="USD", index=True, nullable=False)
    evidence: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    metadata_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)
    resolved_at: datetime | None = Field(default=None, nullable=True)


class LLMValueAnnotationRecord(SQLModel, table=True):
    __tablename__ = "llm_value_annotations"
    __table_args__ = (
        Index("ix_llm_value_annotations_usage", "usage_record_id"),
        Index("ix_llm_value_annotations_workspace_artifact", "workspace_id", "artifact_type", "artifact_id"),
        Index("ix_llm_value_annotations_workspace_result", "workspace_id", "result_type", "result_id"),
        Index("ix_llm_value_annotations_workspace_stage", "workspace_id", "stage", "created_at"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID | None = Field(default=None, foreign_key="workspaces.id", index=True, nullable=True)
    usage_record_id: UUID | None = Field(default=None, foreign_key="llm_usage_ledger.id", index=True, nullable=True)
    artifact_type: str = Field(default="", index=True, nullable=False)
    artifact_id: str = Field(default="", index=True, nullable=False)
    result_type: str = Field(default="", index=True, nullable=False)
    result_id: str = Field(default="", index=True, nullable=False)
    stage: str = Field(default="", index=True, nullable=False)
    decision_key: str = Field(default="", index=True, nullable=False)
    value_signal: str = Field(default="", index=True, nullable=False)
    artifact_created: bool = Field(default=False, nullable=False)
    stage_completed: bool = Field(default=False, nullable=False)
    evaluation_passed: bool = Field(default=False, nullable=False)
    human_review_needed: bool = Field(default=False, nullable=False)
    created_by_user_id: UUID | None = Field(default=None, foreign_key="users.id", index=True, nullable=True)
    metadata_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class OpportunityRecord(SQLModel, table=True):
    __tablename__ = "opportunities"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True, unique=True)
    problem_statement: str
    current_user: str
    current_process: str
    desired_outcome: str
    autonomy_level: str
    constraints: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    operational_baseline: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    mvp_definition: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    case_type: str = Field(default=CASE_TYPE_COPILOT)
    value_statement: str = Field(default="")
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class CanvasRecord(SQLModel, table=True):
    __tablename__ = "canvases"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True, unique=True)
    user_goal: str
    mvp_scope: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    out_of_scope: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    success_metric: str
    primary_risk: str
    agent_profile: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class BlueprintRecord(SQLModel, table=True):
    __tablename__ = "blueprints"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True, unique=True)
    architecture: str
    reasoning_pattern: str
    memory_strategy: str
    tools: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    llm_policy: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    memory_profile: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    knowledge_profile: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    safety_checks: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    guardrails: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    delivery_package: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    readiness_state: ReviewState = Field(default=ReviewState.partial)
    narrative: str
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class EvaluationRecord(SQLModel, table=True):
    __tablename__ = "evaluations"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True, unique=True)
    report: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    status: ArtifactStatus = Field(default=ArtifactStatus.draft)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class EvaluationDatasetRecord(SQLModel, table=True):
    __tablename__ = "evaluation_datasets"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    blueprint_version_number: int | None = Field(default=None, nullable=True)
    version_number: int = Field(default=1, nullable=False)
    source_action: str = Field(default="bootstrap")
    status: ArtifactStatus = Field(default=ArtifactStatus.draft)
    summary: str = Field(default="")
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class EvaluationCaseRecord(SQLModel, table=True):
    __tablename__ = "evaluation_cases"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    dataset_id: UUID = Field(foreign_key="evaluation_datasets.id", index=True)
    case_key: str = Field(index=True)
    title: str = Field(default="")
    category: str = Field(default="")
    scenario: str = Field(default="")
    expected_result: str = Field(default="")
    source: str = Field(default="generated")
    priority: str = Field(default="core")
    sort_order: int = Field(default=0, nullable=False)
    is_active: bool = Field(default=True, nullable=False)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class EvaluationRubricRecord(SQLModel, table=True):
    __tablename__ = "evaluation_rubrics"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    blueprint_version_number: int | None = Field(default=None, nullable=True)
    version_number: int = Field(default=1, nullable=False)
    source_action: str = Field(default="bootstrap")
    summary: str = Field(default="")
    dimensions: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class EvaluationRunRecord(SQLModel, table=True):
    __tablename__ = "evaluation_runs"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    dataset_id: UUID = Field(foreign_key="evaluation_datasets.id", index=True)
    rubric_id: UUID = Field(foreign_key="evaluation_rubrics.id", index=True)
    blueprint_version_number: int | None = Field(default=None, nullable=True)
    source_action: str = Field(default="manual_run")
    status: ArtifactStatus = Field(default=ArtifactStatus.draft)
    overall_score: int = Field(default=0, nullable=False)
    summary: str = Field(default="")
    category_scores: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    dimension_scores: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    blocking_issues: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    recommendations: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class EvaluationResultRecord(SQLModel, table=True):
    __tablename__ = "evaluation_results"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    run_id: UUID = Field(foreign_key="evaluation_runs.id", index=True)
    case_id: UUID | None = Field(default=None, foreign_key="evaluation_cases.id", nullable=True)
    case_key: str = Field(index=True)
    title: str = Field(default="")
    category: str = Field(default="")
    status: ArtifactStatus = Field(default=ArtifactStatus.draft)
    score: int = Field(default=0, nullable=False)
    summary: str = Field(default="")
    observed_result: str = Field(default="")
    evidence: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    blocking_issues: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    recommendations: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class ValidationSimulationRunStateRecord(SQLModel, table=True):
    __tablename__ = "validation_simulation_runs"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    specification_artifact_id: UUID | None = Field(default=None, foreign_key="journey_stage_artifacts.id", nullable=True)
    blueprint_version_number: int | None = Field(default=None, nullable=True)
    scenario_key: str = Field(default="", index=True)
    scenario_title: str = Field(default="")
    scenario_version_number: int = Field(default=1, nullable=False)
    source_action: str = Field(default="run_validation_simulation")
    status: ArtifactStatus = Field(default=ArtifactStatus.draft)
    execution_state: str = Field(default="completed")
    hard_gate_status: str = Field(default="needs_revision")
    final_status: str = Field(default="needs_revision")
    active_node_key: str = Field(default="")
    summary: str = Field(default="")
    injected_conditions: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    deterministic_signature: str = Field(default="")
    events: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    judgement: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class BlueprintVersionRecord(SQLModel, table=True):
    __tablename__ = "blueprint_versions"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    version_number: int
    source_action: str
    status: ArtifactStatus = Field(default=ArtifactStatus.draft)
    blueprint_snapshot: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class ValidationReportRecord(SQLModel, table=True):
    __tablename__ = "validation_reports"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    artifact_name: str
    status: ArtifactStatus
    missing_fields: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    warnings: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class ExecutionLogRecord(SQLModel, table=True):
    __tablename__ = "execution_logs"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    stage: SessionStage
    status: ArtifactStatus
    message: str
    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class StageOperationRecord(SQLModel, table=True):
    __tablename__ = "stage_operations"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            "session_id",
            "action",
            "idempotency_key",
            name="uq_stage_operations_workspace_session_action_idempotency",
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    user_id: UUID = Field(foreign_key="users.id", index=True)
    stage_key: str = Field(index=True)
    action: str = Field(index=True)
    idempotency_key: str = Field(default="", index=True, nullable=False)
    attempt_count: int = Field(default=1, nullable=False)
    status: StageOperationStatus = Field(default=StageOperationStatus.queued, index=True)
    current_step: str = Field(default="")
    detail: str = Field(default="")
    request_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    steps: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    result_artifact_id: UUID | None = Field(default=None, foreign_key="journey_stage_artifacts.id", index=True)
    error_message: str = Field(default="")
    technical_detail: str = Field(default="")
    cancel_requested_at: datetime | None = Field(default=None, nullable=True, index=True)
    heartbeat_at: datetime | None = Field(default=None, nullable=True, index=True)
    expires_at: datetime | None = Field(default=None, nullable=True, index=True)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)
    completed_at: datetime | None = Field(default=None, nullable=True)


class ApprovalGateRecord(SQLModel, table=True):
    __tablename__ = "approval_gates"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    gate_key: str = Field(index=True)
    title: str
    rationale: str = Field(default="")
    instructions: str = Field(default="")
    requested_in_stage: SessionStage = Field(default=SessionStage.post_validation)
    status: ApprovalStatus = Field(default=ApprovalStatus.pending)
    resolution_note: str = Field(default="")
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    resolved_at: datetime | None = Field(default=None, nullable=True)


class SchemaMigrationRecord(SQLModel, table=True):
    __tablename__ = "schema_migrations"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    migration_key: str = Field(index=True, unique=True)
    description: str = Field(default="")
    applied_at: datetime = Field(default_factory=utc_now, nullable=False)


class RuntimeFeatureFlagRecord(SQLModel, table=True):
    __tablename__ = "runtime_feature_flags"
    __table_args__ = (UniqueConstraint("workspace_id", "flag_key", name="uq_runtime_feature_flag_workspace"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    flag_key: str = Field(index=True)
    enabled: bool = Field(default=False, nullable=False)
    description: str = Field(default="")
    stage_hint: str = Field(default="")
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class RuntimeCatalogEntryRecord(SQLModel, table=True):
    __tablename__ = "runtime_catalog_entries"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    catalog_key: str = Field(index=True)
    item_key: str = Field(index=True)
    label: str = Field(default="")
    version: str = Field(default="stage0.v1")
    order_index: int = Field(default=0, nullable=False)
    is_active: bool = Field(default=True, nullable=False)
    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class SkillCatalogRecord(SQLModel, table=True):
    __tablename__ = "skill_catalog"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    skill_key: str = Field(index=True, unique=True)
    label: str = Field(default="")
    stage_hint: str = Field(default="")
    summary: str = Field(default="")
    evidence_policy: str = Field(default="")
    input_schema: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    output_schema: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    is_active: bool = Field(default=True, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class SkillRunRecord(SQLModel, table=True):
    __tablename__ = "skill_runs"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    skill_key: str = Field(index=True)
    stage: SessionStage
    blueprint_version_number: int | None = Field(default=None, nullable=True)
    source_action: str = Field(default="stage_run")
    status: ArtifactStatus = Field(default=ArtifactStatus.draft)
    duration_ms: int = Field(default=0, nullable=False)
    result_summary: str = Field(default="")
    warnings: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    evidence: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class SkillRunArtifactRecord(SQLModel, table=True):
    __tablename__ = "skill_run_artifacts"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    skill_run_id: UUID = Field(foreign_key="skill_runs.id", index=True)
    artifact_role: str = Field(index=True)
    artifact_kind: str = Field(default="")
    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class ArtifactRegistryRecord(SQLModel, table=True):
    __tablename__ = "artifact_records"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    blueprint_version_number: int | None = Field(default=None, nullable=True)
    artifact_key: str = Field(index=True)
    artifact_title: str = Field(default="")
    artifact_kind: str = Field(default="", index=True)
    stage: SessionStage = Field(default=SessionStage.post_validation)
    source_action: str = Field(default="")
    export_format: str = Field(default="")
    content_text: str = Field(default="")
    content_hash: str = Field(default="", index=True)
    artifact_metadata: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class JourneyStageArtifactRecord(SQLModel, table=True):
    __tablename__ = "journey_stage_artifacts"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            "session_id",
            "stage_key",
            "version_number",
            name="uq_journey_stage_artifact_version",
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    artifact_kind: str = Field(default="", index=True)
    stage_key: str = Field(default="", index=True)
    version_number: int = Field(default=1, nullable=False)
    state: JourneyArtifactState = Field(default=JourneyArtifactState.generated, nullable=False, index=True)
    source_action: str = Field(default="manual_draft", nullable=False)
    proposal_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    user_patch: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    source_stage_versions: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    input_fingerprint: str = Field(default="", index=True)
    context_fingerprint: str = Field(default="", index=True)
    output_fingerprint: str = Field(default="", index=True)
    corpus_hash: str = Field(default="")
    provider_key: str = Field(default="")
    model: str = Field(default="")
    execution_backend: str = Field(default="")
    prompt_version: str = Field(default="")
    schema_version: str = Field(default="")
    confidence: float | None = Field(default=None, nullable=True)
    missing_information: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    warnings: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    evidence_manifest: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    stale_reasons: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    based_on_artifact_id: UUID | None = Field(
        default=None,
        foreign_key="journey_stage_artifacts.id",
        nullable=True,
        index=True,
    )
    superseded_by_artifact_id: UUID | None = Field(
        default=None,
        foreign_key="journey_stage_artifacts.id",
        nullable=True,
        index=True,
    )
    approved_by_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True)
    reviewed_at: datetime | None = Field(default=None, nullable=True)
    approved_at: datetime | None = Field(default=None, nullable=True)
    rejected_at: datetime | None = Field(default=None, nullable=True)
    stale_at: datetime | None = Field(default=None, nullable=True)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class JourneyStageDecisionRecord(SQLModel, table=True):
    __tablename__ = "journey_stage_decisions"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    artifact_id: UUID = Field(foreign_key="journey_stage_artifacts.id", index=True)
    stage_key: str = Field(default="", index=True)
    decision_type: JourneyDecisionType = Field(default=JourneyDecisionType.create, nullable=False, index=True)
    previous_state: JourneyArtifactState | None = Field(default=None, nullable=True)
    next_state: JourneyArtifactState | None = Field(default=None, nullable=True)
    actor_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True)
    note: str = Field(default="")
    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class EstimationRunRecord(SQLModel, table=True):
    __tablename__ = "estimation_runs"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    blueprint_version_number: int | None = Field(default=None, nullable=True)
    source_action: str = Field(default="")
    maturity_stage: EstimationMaturityStage = Field(default=EstimationMaturityStage.canvas, index=True)
    active_provider: LLMProviderKey = Field(default=LLMProviderKey.openai, index=True)
    pricing_policy: str = Field(default="")
    confidence_score: int = Field(default=0, nullable=False)
    confidence_label: EstimationConfidenceLabel = Field(default=EstimationConfidenceLabel.low, index=True)
    uncertainty_band_percent: int = Field(default=0, nullable=False)
    traditional_hours_total: float = Field(default=0, nullable=False)
    traditional_duration_weeks: float = Field(default=0, nullable=False)
    traditional_cost_total: float = Field(default=0, nullable=False)
    agentic_hours_total: float = Field(default=0, nullable=False)
    agentic_duration_weeks: float = Field(default=0, nullable=False)
    agentic_cost_total: float = Field(default=0, nullable=False)
    automation_coverage_percent: int = Field(default=0, nullable=False)
    estimation_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class ProjectActualsRecord(SQLModel, table=True):
    __tablename__ = "project_actuals"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    estimation_run_id: UUID = Field(foreign_key="estimation_runs.id", index=True, unique=True)
    delivery_mode: EstimationScenarioType = Field(default=EstimationScenarioType.agentic, index=True)
    actual_provider: LLMProviderKey | None = Field(default=None, nullable=True)
    actual_hours_total: float = Field(default=0, nullable=False)
    actual_duration_weeks: float = Field(default=0, nullable=False)
    actual_cost_total: float = Field(default=0, nullable=False)
    actual_automation_coverage_percent: int = Field(default=0, nullable=False)
    notes: str = Field(default="")
    captured_by_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class EstimationErrorMetricRecord(SQLModel, table=True):
    __tablename__ = "estimation_error_metrics"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    estimation_run_id: UUID = Field(foreign_key="estimation_runs.id", index=True, unique=True)
    actuals_id: UUID = Field(foreign_key="project_actuals.id", index=True, unique=True)
    maturity_stage: EstimationMaturityStage = Field(default=EstimationMaturityStage.canvas, index=True)
    scenario_type: EstimationScenarioType = Field(default=EstimationScenarioType.agentic, index=True)
    active_provider: LLMProviderKey | None = Field(default=None, nullable=True)
    absolute_percentage_error_hours: float = Field(default=0, nullable=False)
    absolute_percentage_error_duration: float = Field(default=0, nullable=False)
    absolute_percentage_error_cost: float = Field(default=0, nullable=False)
    absolute_percentage_error_automation: float = Field(default=0, nullable=False)
    bias_hours_percent: float = Field(default=0, nullable=False)
    bias_duration_percent: float = Field(default=0, nullable=False)
    bias_cost_percent: float = Field(default=0, nullable=False)
    bias_automation_percent: float = Field(default=0, nullable=False)
    band_hit_hours: bool = Field(default=False, nullable=False)
    band_hit_duration: bool = Field(default=False, nullable=False)
    band_hit_cost: bool = Field(default=False, nullable=False)
    band_hit_overall: bool = Field(default=False, nullable=False)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class ConstructionQuestionResponseRecord(SQLModel, table=True):
    __tablename__ = "construction_question_responses"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    question_key: str = Field(index=True)
    gap_key: str = Field(default="", index=True)
    gap_title: str = Field(default="")
    domain: str = Field(default="", index=True)
    question_text: str = Field(default="")
    rationale: str = Field(default="")
    expected_answer_format: str = Field(default="")
    target_owner: str = Field(default="")
    blocking: bool = Field(default=False, nullable=False)
    status: str = Field(default="answered", index=True)
    answer_text: str = Field(default="")
    owner_role: str = Field(default="")
    impacted_artifacts: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    answered_by_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True)
    answered_by_display: str = Field(default="")
    answered_at: datetime | None = Field(default=None, nullable=True)
    resolved_at: datetime | None = Field(default=None, nullable=True)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class MetricSnapshotRecord(SQLModel, table=True):
    __tablename__ = "metric_snapshots"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    source_action: str = Field(default="")
    cost_estimate_usd: float = Field(default=0, nullable=False)
    total_duration_ms: int = Field(default=0, nullable=False)
    error_count: int = Field(default=0, nullable=False)
    warning_count: int = Field(default=0, nullable=False)
    approvals_pending: int = Field(default=0, nullable=False)
    approvals_resolved: int = Field(default=0, nullable=False)
    regenerations_count: int = Field(default=0, nullable=False)
    needs_review_count: int = Field(default=0, nullable=False)
    latest_evaluation_score: int | None = Field(default=None, nullable=True)
    latest_evaluation_status: str = Field(default="")
    export_count: int = Field(default=0, nullable=False)
    artifact_count: int = Field(default=0, nullable=False)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class AlertEventRecord(SQLModel, table=True):
    __tablename__ = "alert_events"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    alert_key: str = Field(index=True)
    severity: str = Field(default="")
    title: str = Field(default="")
    message: str = Field(default="")
    status: str = Field(default="active", index=True)
    evidence: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)
    resolved_at: datetime | None = Field(default=None, nullable=True)


class IntegrationStatusRecord(SQLModel, table=True):
    __tablename__ = "integration_statuses"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    integration_key: str = Field(index=True)
    label: str = Field(default="")
    status: str = Field(default="", index=True)
    configured: bool = Field(default=False, nullable=False)
    reachable: bool = Field(default=False, nullable=False)
    detail: str = Field(default="")
    checked_at: datetime = Field(default_factory=utc_now, nullable=False)


class WorkflowTemplateRecord(SQLModel, table=True):
    __tablename__ = "workflow_templates"
    __table_args__ = (UniqueConstraint("workspace_id", "template_key", name="uq_workflow_template_workspace"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    template_key: str = Field(index=True)
    label: str = Field(default="")
    summary: str = Field(default="")
    architecture_scope: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    supports_approvals: bool = Field(default=False, nullable=False)
    supports_handoffs: bool = Field(default=False, nullable=False)
    workflow_profile: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    governance_hints: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    is_active: bool = Field(default=True, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class HandoffRecord(SQLModel, table=True):
    __tablename__ = "handoff_records"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    blueprint_version_number: int | None = Field(default=None, nullable=True)
    handoff_key: str = Field(index=True)
    title: str = Field(default="")
    from_stage: SessionStage = Field(default=SessionStage.post_validation)
    to_stage: SessionStage = Field(default=SessionStage.ready_for_export)
    status: str = Field(default="pending", index=True)
    owner_role: str = Field(default="")
    triggered_by: str = Field(default="")
    summary: str = Field(default="")
    resolution_note: str = Field(default="")
    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)
    resolved_at: datetime | None = Field(default=None, nullable=True)


class GovernancePolicyRecord(SQLModel, table=True):
    __tablename__ = "governance_policies"
    __table_args__ = (UniqueConstraint("workspace_id", "policy_key", name="uq_governance_policy_workspace"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    policy_key: str = Field(index=True)
    label: str = Field(default="")
    summary: str = Field(default="")
    scope: str = Field(default="")
    is_active: bool = Field(default=True, nullable=False)
    policy_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class SubagentRunRecord(SQLModel, table=True):
    __tablename__ = "subagent_runs"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    blueprint_version_number: int | None = Field(default=None, nullable=True)
    run_kind: str = Field(index=True)
    title: str = Field(default="")
    status: ArtifactStatus = Field(default=ArtifactStatus.draft)
    feature_flag_key: str = Field(default="")
    summary: str = Field(default="")
    input_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    output_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class KnowledgeIngestionRunRecord(SQLModel, table=True):
    __tablename__ = "knowledge_ingestion_runs"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    source_root: str = Field(index=True)
    scope: KnowledgeScope = Field(default=KnowledgeScope.platform, index=True, nullable=False)
    workspace_id: UUID | None = Field(default=None, foreign_key="workspaces.id", nullable=True, index=True)
    session_id: UUID | None = Field(default=None, foreign_key="sessions.id", nullable=True, index=True)
    status: str = Field(default="ready", index=True)
    corpus_hash: str = Field(default="", index=True)
    document_count: int = Field(default=0, nullable=False)
    changed_document_count: int = Field(default=0, nullable=False)
    unchanged_document_count: int = Field(default=0, nullable=False)
    section_count: int = Field(default=0, nullable=False)
    lexical_term_count: int = Field(default=0, nullable=False)
    vector_dimensions: int = Field(default=0, nullable=False)
    filesystem_manifest_path: str = Field(default="")
    lexical_index_path: str = Field(default="")
    vector_index_path: str = Field(default="")
    changed_paths: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    metadata_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class KnowledgeDocumentRecord(SQLModel, table=True):
    __tablename__ = "knowledge_documents"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    source_root: str = Field(default="Docs", index=True)
    scope: KnowledgeScope = Field(default=KnowledgeScope.platform, index=True, nullable=False)
    workspace_id: UUID | None = Field(default=None, foreign_key="workspaces.id", nullable=True, index=True)
    session_id: UUID | None = Field(default=None, foreign_key="sessions.id", nullable=True, index=True)
    relative_path: str = Field(index=True)
    title: str = Field(default="")
    format: str = Field(default="", index=True)
    visibility: KnowledgeVisibility = Field(default=KnowledgeVisibility.platform, index=True, nullable=False)
    status: KnowledgeDocumentStatus = Field(default=KnowledgeDocumentStatus.approved, index=True, nullable=False)
    authority_level: str = Field(default="", index=True)
    memory_usage: str = Field(default="", index=True)
    stage_affinity: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    stage_affinity_text: str = Field(default="", index=True)
    agent_affinity: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    agent_affinity_text: str = Field(default="", index=True)
    content_hash: str = Field(default="", index=True)
    version_number: int = Field(default=1, nullable=False)
    file_size_bytes: int = Field(default=0, nullable=False)
    word_count: int = Field(default=0, nullable=False)
    section_count: int = Field(default=0, nullable=False)
    source_lineage: str = Field(default="")
    approved_by_user_id: UUID | None = Field(default=None, nullable=True, index=True)
    approved_at: datetime | None = Field(default=None, nullable=True)
    effective_from: datetime | None = Field(default=None, nullable=True, index=True)
    expires_at: datetime | None = Field(default=None, nullable=True, index=True)
    supersedes_document_id: UUID | None = Field(default=None, nullable=True, index=True)
    last_ingestion_run_id: UUID | None = Field(default=None, foreign_key="knowledge_ingestion_runs.id", nullable=True)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class KnowledgeSectionRecord(SQLModel, table=True):
    __tablename__ = "knowledge_sections"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    document_id: UUID = Field(foreign_key="knowledge_documents.id", index=True)
    source_root: str = Field(default="Docs", index=True)
    scope: KnowledgeScope = Field(default=KnowledgeScope.platform, index=True, nullable=False)
    workspace_id: UUID | None = Field(default=None, foreign_key="workspaces.id", nullable=True, index=True)
    session_id: UUID | None = Field(default=None, foreign_key="sessions.id", nullable=True, index=True)
    relative_path: str = Field(default="", index=True)
    section_key: str = Field(index=True)
    title: str = Field(default="")
    heading_path: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    heading_level: int = Field(default=0, nullable=False)
    sort_order: int = Field(default=0, nullable=False)
    start_line: int = Field(default=1, nullable=False)
    end_line: int = Field(default=1, nullable=False)
    visibility: KnowledgeVisibility = Field(default=KnowledgeVisibility.platform, index=True, nullable=False)
    status: KnowledgeDocumentStatus = Field(default=KnowledgeDocumentStatus.approved, index=True, nullable=False)
    authority_level: str = Field(default="", index=True)
    memory_usage: str = Field(default="", index=True)
    stage_affinity: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    stage_affinity_text: str = Field(default="", index=True)
    agent_affinity: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    agent_affinity_text: str = Field(default="", index=True)
    document_version_number: int = Field(default=1, nullable=False)
    content_hash: str = Field(default="", index=True)
    source_lineage: str = Field(default="", index=True)
    content_text: str = Field(default="")
    token_count: int = Field(default=0, nullable=False)
    lexical_terms: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    vector_payload: list[float] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    metadata_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    approved_by_user_id: UUID | None = Field(default=None, nullable=True, index=True)
    approved_at: datetime | None = Field(default=None, nullable=True)
    effective_from: datetime | None = Field(default=None, nullable=True, index=True)
    expires_at: datetime | None = Field(default=None, nullable=True, index=True)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class ShortTermSessionStateRecord(SQLModel, table=True):
    __tablename__ = "short_term_session_states"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True, unique=True)
    active_branch_key: str = Field(default="main", index=True)
    active_checkpoint_key: str = Field(default="", index=True)
    last_consistent_checkpoint_key: str = Field(default="", index=True)
    source_action: str = Field(default="")
    state_hash: str = Field(default="", index=True)
    state_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class ShortTermBranchRecord(SQLModel, table=True):
    __tablename__ = "short_term_branches"
    __table_args__ = (UniqueConstraint("session_id", "branch_key", name="uq_short_term_branches_session_branch"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    branch_key: str = Field(index=True)
    parent_branch_key: str = Field(default="")
    title: str = Field(default="")
    topology: str = Field(default="mainline", index=True)
    stage: str = Field(default="", index=True)
    status: str = Field(default="active", index=True)
    isolation_mode: str = Field(default="shared")
    summary: str = Field(default="")
    namespace_keys: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    checkpoint_count: int = Field(default=0, nullable=False)
    active_checkpoint_key: str = Field(default="")
    last_consistent_checkpoint_key: str = Field(default="")
    metadata_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)
    last_activity_at: datetime = Field(default_factory=utc_now, nullable=False)


class ShortTermCheckpointRecord(SQLModel, table=True):
    __tablename__ = "short_term_checkpoints"
    __table_args__ = (UniqueConstraint("session_id", "checkpoint_key", name="uq_short_term_checkpoints_session_key"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    branch_key: str = Field(index=True)
    checkpoint_key: str = Field(index=True)
    parent_checkpoint_key: str = Field(default="")
    checkpoint_number: int = Field(default=1, nullable=False)
    stage: str = Field(default="", index=True)
    source_action: str = Field(default="", index=True)
    status: str = Field(default="active", index=True)
    summary: str = Field(default="")
    state_hash: str = Field(default="", index=True)
    state_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    is_consistent: bool = Field(default=True, nullable=False)
    is_active: bool = Field(default=True, nullable=False)
    rollback_note: str = Field(default="")
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class DiscoveryInput(ContractModel):
    problem_statement: str = ""
    current_user: str = ""
    current_process: str = ""
    desired_outcome: str = ""
    autonomy_level: str = AUTONOMY_MEDIUM
    constraints: list[str] = PydanticField(default_factory=list)
    operational_baseline: "OperationalBaseline" = PydanticField(default_factory=lambda: OperationalBaseline())
    mvp_definition: "MvpDefinition" = PydanticField(default_factory=lambda: MvpDefinition())

    @field_validator("autonomy_level", mode="before")
    @classmethod
    def _normalize_autonomy_level(cls, value: object) -> str:
        if isinstance(value, str):
            return normalize_autonomy_level(value)
        return AUTONOMY_MEDIUM


class EvidenceItem(ContractModel):
    source: EvidenceSource
    detail: str


class LLMContextTrace(ContractModel):
    provider_key: str = ""
    execution_backend: str = ""
    execution_mode: str = ""
    shadow_provider_key: str = ""
    route_reason: str = ""
    knowledge_access_backend: str = ""
    effective_context_backend: str = ""
    context_used_sources: list[dict[str, Any]] = PydanticField(default_factory=list)
    context_stats: dict[str, Any] = PydanticField(default_factory=dict)
    capability_key: str = ""
    model_name: str = ""
    prompt_version: str = ""
    request_id: str = ""
    finish_reason: str = ""
    schema_validation_status: str = ""
    token_usage: dict[str, int] = PydanticField(default_factory=dict)
    failure_kind: str = ""
    failure_detail: str = ""
    retry_count: int = 0
    fallback_used: bool = False
    degraded: bool = False
    capability_policy: dict[str, Any] = PydanticField(default_factory=dict)
    rollout_comparison: dict[str, Any] = PydanticField(default_factory=dict)


class OperationalBaseline(ContractModel):
    current_time_spent: str = ""
    current_cost: str = ""
    frequent_errors: list[str] = PydanticField(default_factory=list)
    automation_opportunities: list[str] = PydanticField(default_factory=list)


class MvpDefinition(ContractModel):
    v1_scope: list[str] = PydanticField(default_factory=list)
    out_of_scope: list[str] = PydanticField(default_factory=list)
    north_star_metric: str = ""
    non_delegable_decisions: list[str] = PydanticField(default_factory=list)


class DiscoveryArtifact(ContractModel):
    problem_statement: str = ""
    current_user: str = ""
    current_process: str = ""
    desired_outcome: str = ""
    autonomy_level: str = AUTONOMY_MEDIUM
    constraints: list[str] = PydanticField(default_factory=list)
    operational_baseline: OperationalBaseline = PydanticField(default_factory=OperationalBaseline)
    mvp_definition: MvpDefinition = PydanticField(default_factory=MvpDefinition)
    case_type: str = ""
    value_statement: str = ""

    @field_validator("autonomy_level", mode="before")
    @classmethod
    def _normalize_autonomy_level(cls, value: object) -> str:
        if isinstance(value, str):
            return normalize_autonomy_level(value)
        return AUTONOMY_MEDIUM

    @field_validator("case_type", mode="before")
    @classmethod
    def _normalize_case_type(cls, value: object) -> str:
        if isinstance(value, str):
            return normalize_case_type(value)
        return ""


class AgentCanvasProfile(ContractModel):
    mission: str = ""
    primary_user: str = ""
    agent_task: str = ""
    allowed_decisions: list[str] = PydanticField(default_factory=list)
    prohibited_decisions: list[str] = PydanticField(default_factory=list)
    key_inputs: list[str] = PydanticField(default_factory=list)
    expected_outputs: list[str] = PydanticField(default_factory=list)
    human_approvals: list[str] = PydanticField(default_factory=list)
    success_metrics: list[str] = PydanticField(default_factory=list)


class CanvasArtifact(ContractModel):
    user_goal: str = ""
    mvp_scope: list[str] = PydanticField(default_factory=list)
    out_of_scope: list[str] = PydanticField(default_factory=list)
    success_metric: str = ""
    primary_risk: str = ""
    agent_profile: AgentCanvasProfile = PydanticField(default_factory=AgentCanvasProfile)


class BlueprintTool(ContractModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    name: str = ""
    purpose: str = ""
    owner: str = ""
    archetype: str = ""
    integration_kind: str = ""
    tool_type: str = "external"
    execution_stage: str = "tools"
    when_to_use: str = ""
    endpoint_reference: str = ""
    auth_reference: str = ""
    risk_level: str = ""
    requires_approval: bool = False
    inputs: list[str] = PydanticField(default_factory=list)
    outputs: list[str] = PydanticField(default_factory=list)
    request_schema: dict[str, Any] = PydanticField(default_factory=dict)
    response_schema: dict[str, Any] = PydanticField(default_factory=dict)
    usage_examples: list[dict[str, Any]] = PydanticField(default_factory=list)
    security_config: dict[str, Any] = PydanticField(default_factory=dict)
    registered_api_ref: str = ""
    validations: list[str] = PydanticField(default_factory=list)
    typed_errors: list[str] = PydanticField(default_factory=list)
    permissions: list[str] = PydanticField(default_factory=list)
    scopes: list[str] = PydanticField(default_factory=list)
    sensitive_data: list[str] = PydanticField(default_factory=list)
    audit_rules: list[str] = PydanticField(default_factory=list)
    has_side_effects: bool = False
    execution_mode: str = ""
    approval_policy: str = ""
    retry_strategy: str = ""
    idempotency_strategy: str = ""
    compensation_strategy: str = ""
    approval_reason: str = ""
    failure_mode: str = ""
    rate_limit_policy: str = ""
    timeout_policy: str = ""
    contract_review_state: str = "needs-review"


class BlueprintLLMFunctionPolicy(ContractModel):
    role: str = ""
    provider: str = ""
    model: str = ""
    reasoning_effort: str = "medium"
    max_tokens: int = 0
    fallback_model: str = ""
    tool_availability: list[str] = PydanticField(default_factory=list)


class BlueprintLLMPolicy(ContractModel):
    provider: str = ""
    fast_model: str = ""
    reasoning_model: str = ""
    fallback_model: str = ""
    context_policy: str = ""
    sampling_policy: str = ""
    fallback_policy: str = ""
    circuit_breaker_policy: str = ""
    budget_policy: str = ""
    output_validation_policy: str = ""
    log_redaction_policy: str = ""
    review_state: str = "needs-review"
    functions: list[BlueprintLLMFunctionPolicy] = PydanticField(default_factory=list)


class GroundingPolicy(ContractModel):
    citations_policy: str = ""
    confidence_policy: str = ""
    no_evidence_behavior: str = ""
    contradictory_evidence_behavior: str = ""


class MemoryProfile(ContractModel):
    strategy: str = ""
    storage_layers: list[str] = PydanticField(default_factory=list)
    write_policy: str = ""
    retrieval_policy: str = ""
    review_trigger: str = ""
    goal_drift_guard: str = ""
    retention_policy: str = ""
    ttl_policy: str = ""
    workspace_scope: str = ""
    agent_scope: str = ""
    grounding_policy: GroundingPolicy = PydanticField(default_factory=GroundingPolicy)
    sensitivity_rules: list[str] = PydanticField(default_factory=list)


class KnowledgeSource(ContractModel):
    key: str = ""
    title: str = ""
    source_type: str = ""
    uri: str = ""
    owner: str = ""
    sensitivity: str = ""
    license: str = ""
    description: str = ""
    source_version: str = ""


class IngestionPolicy(ContractModel):
    parser: str = ""
    chunking_policy: str = ""
    metadata_fields: list[str] = PydanticField(default_factory=list)
    include_filters: list[str] = PydanticField(default_factory=list)
    exclude_filters: list[str] = PydanticField(default_factory=list)


class EmbeddingPolicy(ContractModel):
    provider: str = ""
    model: str = ""
    dimensions: int = 0
    version: str = ""


class RetrievalPolicyProfile(ContractModel):
    top_k: int = 0
    filters: list[str] = PydanticField(default_factory=list)
    search_mode: str = ""
    reranking_policy: str = ""
    fallback_behavior: str = ""


class RefreshPolicy(ContractModel):
    frequency: str = ""
    triggers: list[str] = PydanticField(default_factory=list)
    expiration_policy: str = ""
    deletion_policy: str = ""


class KnowledgeProfile(ContractModel):
    mode: str = "none"
    sources: list[KnowledgeSource] = PydanticField(default_factory=list)
    ingestion_policy: IngestionPolicy = PydanticField(default_factory=IngestionPolicy)
    embedding_policy: EmbeddingPolicy = PydanticField(default_factory=EmbeddingPolicy)
    retrieval_policy: RetrievalPolicyProfile = PydanticField(default_factory=RetrievalPolicyProfile)
    refresh_policy: RefreshPolicy = PydanticField(default_factory=RefreshPolicy)
    grounding_policy: GroundingPolicy = PydanticField(default_factory=GroundingPolicy)
    sensitivity_rules: list[str] = PydanticField(default_factory=list)
    notes: str = ""


class SafetyCheck(ContractModel):
    category: str = ""
    risk: str = ""
    severity: str = ""
    mitigation: str = ""
    status: str = ""


class WorkflowStep(ContractModel):
    name: str = ""
    objective: str = ""
    actor: str = ""
    outputs: list[str] = PydanticField(default_factory=list)
    fallback: str = ""
    requires_approval: bool = False


class WorkflowProfile(ContractModel):
    execution_pattern: str = ""
    inbox_strategy: str = ""
    outbox_strategy: str = ""
    checkpoint_policy: str = ""
    retry_strategy: str = ""
    compensation_strategy: str = ""
    approval_pause: str = ""
    timeout_policy: str = ""
    steps: list[WorkflowStep] = PydanticField(default_factory=list)


class ObservabilityPlan(ContractModel):
    captured_signals: list[str] = PydanticField(default_factory=list)
    plan_summary_policy: str = ""
    tool_response_logging: str = ""
    decision_logging: str = ""
    cost_tracking: str = ""
    duration_tracking: str = ""
    alert_triggers: list[str] = PydanticField(default_factory=list)
    result_tracking: str = ""


class PatternCatalogEntry(ContractModel):
    family: str = ""
    key: str = ""
    label: str = ""
    summary: str = ""
    use_when: list[str] = PydanticField(default_factory=list)
    tradeoffs: list[str] = PydanticField(default_factory=list)
    fit_score: int = 0
    selected: bool = False


class DecisionTraceEntry(ContractModel):
    dimension: str = ""
    selected_value: str = ""
    selected_label: str = ""
    recommended_value: str = ""
    recommended_label: str = ""
    decision_source: str = ""
    rationale: str = ""
    evidence: list[str] = PydanticField(default_factory=list)
    review_note: str = ""


class ComponentCheckItem(ContractModel):
    key: str = ""
    title: str = ""
    status: ReviewState = ReviewState.partial
    detail: str = ""


class ComponentReadinessEntry(ContractModel):
    component: str = ""
    label: str = ""
    status: ReviewState = ReviewState.partial
    score: int = 0
    completed_checks: int = 0
    total_checks: int = 0
    blocking_issues: list[str] = PydanticField(default_factory=list)
    checks: list[ComponentCheckItem] = PydanticField(default_factory=list)


class RiskSummary(ContractModel):
    overall_status: ReviewState = ReviewState.partial
    total_checks: int = 0
    high_risks: int = 0
    medium_risks: int = 0
    low_risks: int = 0
    approval_gates_required: int = 0
    side_effect_tools: int = 0
    summary: str = ""


class GeneratedDeliverable(ContractModel):
    key: str = ""
    title: str = ""
    summary: str = ""
    content_markdown: str = ""


class RoadmapMilestone(ContractModel):
    release: str = ""
    title: str = ""
    objective: str = ""
    when_to_unlock: str = ""
    capabilities: list[str] = PydanticField(default_factory=list)


class RoadmapEvolution(ContractModel):
    current_release: str = "MVP 1"
    current_focus: str = ""
    milestones: list[RoadmapMilestone] = PydanticField(default_factory=list)


class BlueprintSectionCoverageEntry(ContractModel):
    key: str = ""
    title: str = ""
    status: ReviewState = ReviewState.partial
    source: str = ""
    note: str = ""


class BlueprintCoverageSummary(ContractModel):
    overall_status: ReviewState = ReviewState.partial
    covered_sections: int = 0
    total_sections: int = 14
    missing_sections: list[str] = PydanticField(default_factory=list)
    sections: list[BlueprintSectionCoverageEntry] = PydanticField(default_factory=list)


class DeliveryPackage(ContractModel):
    contract_version: str = "delivery-package.v1"
    workflow_profile: WorkflowProfile = PydanticField(default_factory=WorkflowProfile)
    observability_plan: ObservabilityPlan = PydanticField(default_factory=ObservabilityPlan)
    deliverables: list[GeneratedDeliverable] = PydanticField(default_factory=list)
    decision_summary: str = ""
    decision_trace: list[DecisionTraceEntry] = PydanticField(default_factory=list)
    pattern_catalog: list[PatternCatalogEntry] = PydanticField(default_factory=list)
    component_readiness: list[ComponentReadinessEntry] = PydanticField(default_factory=list)
    risk_summary: RiskSummary = PydanticField(default_factory=RiskSummary)
    roadmap_evolution: RoadmapEvolution = PydanticField(default_factory=RoadmapEvolution)
    blueprint_coverage: BlueprintCoverageSummary = PydanticField(default_factory=BlueprintCoverageSummary)

    @field_validator("deliverables")
    @classmethod
    def validate_deliverables(cls, value: list[GeneratedDeliverable]) -> list[GeneratedDeliverable]:
        seen: set[str] = set()
        for item in value:
            if item.key not in ALLOWED_DELIVERABLE_KEYS:
                raise ValueError(f"Ungoverned deliverable key: {item.key}")
            if item.key in seen:
                raise ValueError(f"Duplicated deliverable key: {item.key}")
            seen.add(item.key)
        return value

    @field_validator("component_readiness")
    @classmethod
    def validate_component_readiness(cls, value: list[ComponentReadinessEntry]) -> list[ComponentReadinessEntry]:
        seen: set[str] = set()
        for item in value:
            if item.component not in ALLOWED_COMPONENT_KEYS:
                raise ValueError(f"Ungoverned component key: {item.component}")
            if item.component in seen:
                raise ValueError(f"Duplicated component readiness key: {item.component}")
            seen.add(item.component)
        return value

    @field_validator("pattern_catalog")
    @classmethod
    def validate_pattern_catalog(cls, value: list[PatternCatalogEntry]) -> list[PatternCatalogEntry]:
        for item in value:
            if item.family not in ALLOWED_PATTERN_FAMILIES:
                raise ValueError(f"Ungoverned pattern family: {item.family}")
        return value


class EvaluationCase(ContractModel):
    name: str = ""
    category: str = ""
    scenario: str = ""
    expected_result: str = ""


class EvaluationArtifact(ContractModel):
    completeness_status: ReviewState
    coherence_status: ReviewState
    cases: list[EvaluationCase] = PydanticField(default_factory=list)
    gaps: list[str] = PydanticField(default_factory=list)
    recommendations: list[str] = PydanticField(default_factory=list)
    scores: dict[str, int] = PydanticField(default_factory=dict)


class EvaluationDatasetCase(ContractModel):
    id: UUID | None = None
    case_key: str = ""
    title: str = ""
    category: str = ""
    scenario: str = ""
    expected_result: str = ""
    source: str = "generated"
    priority: str = "core"
    is_active: bool = True


class EvaluationDatasetArtifact(ContractModel):
    id: UUID | None = None
    version_number: int = 1
    blueprint_version_number: int | None = None
    source_action: str = "bootstrap"
    status: ArtifactStatus = ArtifactStatus.draft
    summary: str = ""
    cases: list[EvaluationDatasetCase] = PydanticField(default_factory=list)


class EvaluationRubricDimension(ContractModel):
    key: str = ""
    label: str = ""
    description: str = ""
    weight: int = 0
    hard_block: bool = False


class EvaluationRubricArtifact(ContractModel):
    id: UUID | None = None
    version_number: int = 1
    blueprint_version_number: int | None = None
    source_action: str = "bootstrap"
    summary: str = ""
    dimensions: list[EvaluationRubricDimension] = PydanticField(default_factory=list)


class EvaluationCaseResult(ContractModel):
    case_key: str = ""
    title: str = ""
    category: str = ""
    status: ArtifactStatus = ArtifactStatus.draft
    score: int = 0
    summary: str = ""
    observed_result: str = ""
    evidence: list[str] = PydanticField(default_factory=list)
    blocking_issues: list[str] = PydanticField(default_factory=list)
    recommendations: list[str] = PydanticField(default_factory=list)


class EvaluationRunSummary(ContractModel):
    dataset_version_number: int = 1
    rubric_version_number: int = 1
    blueprint_version_number: int | None = None
    source_action: str = "manual_run"
    status: ArtifactStatus = ArtifactStatus.draft
    overall_score: int = 0
    summary: str = ""
    category_scores: dict[str, int] = PydanticField(default_factory=dict)
    dimension_scores: dict[str, int] = PydanticField(default_factory=dict)
    blocking_issues: list[str] = PydanticField(default_factory=list)
    recommendations: list[str] = PydanticField(default_factory=list)
    results: list[EvaluationCaseResult] = PydanticField(default_factory=list)


class EvaluationRunEntry(EvaluationRunSummary):
    id: UUID
    created_at: datetime


SimulationNodeType = Literal["agent", "decision", "tool", "memory", "human", "end"]
SimulationEventType = Literal[
    "start",
    "input",
    "decision",
    "tool_call",
    "tool_result",
    "memory_read",
    "memory_write",
    "approval_gate",
    "agent_response",
    "fault_injected",
    "issue",
    "end",
]
SimulationEventTone = Literal["info", "success", "warning", "error", "blocked"]
SimulationRunOutcome = Literal["pass", "needs_revision", "fail"]
SimulationExecutionState = Literal["running", "paused", "completed"]


class SimulationNode(ContractModel):
    node_key: str = ""
    label: str = ""
    node_type: SimulationNodeType = "agent"
    description: str = ""
    x: int = 0
    y: int = 0
    tags: list[str] = PydanticField(default_factory=list)


class SimulationEdge(ContractModel):
    edge_key: str = ""
    from_node_key: str = ""
    to_node_key: str = ""
    label: str = ""
    condition: str = ""
    transition_type: str = "default"


class SimulationScenario(ContractModel):
    scenario_key: str = ""
    title: str = ""
    actor: str = ""
    objective: str = ""
    priority: str = "medium"
    initial_input: str = ""
    preconditions: list[str] = PydanticField(default_factory=list)
    state_transitions: list[str] = PydanticField(default_factory=list)
    decision_criteria: list[str] = PydanticField(default_factory=list)
    tools_invoked: list[str] = PydanticField(default_factory=list)
    simulated_tool_responses: list[str] = PydanticField(default_factory=list)
    memory_reads: list[str] = PydanticField(default_factory=list)
    memory_writes: list[str] = PydanticField(default_factory=list)
    approval_gates: list[str] = PydanticField(default_factory=list)
    expected_outcome: str = ""
    success_criteria: list[str] = PydanticField(default_factory=list)
    blocking_failures: list[str] = PydanticField(default_factory=list)
    suggested_injections: list[str] = PydanticField(default_factory=list)
    source_refs: list[str] = PydanticField(default_factory=list)
    nodes: list[SimulationNode] = PydanticField(default_factory=list)
    edges: list[SimulationEdge] = PydanticField(default_factory=list)


class SimulationSpecificationArtifact(ContractModel):
    schema_version: str = "validation-simulation-spec.v1"
    summary: str = ""
    review_state: ReviewState = ReviewState.partial
    confidence: float = 0.0
    source_blueprint_version: int | None = None
    source_stage_versions: dict[str, int | None] = PydanticField(default_factory=dict)
    coverage_gaps: list[str] = PydanticField(default_factory=list)
    missing_information: list[str] = PydanticField(default_factory=list)
    warnings: list[str] = PydanticField(default_factory=list)
    scenarios: list[SimulationScenario] = PydanticField(default_factory=list)


class SimulationEvent(ContractModel):
    event_key: str = ""
    event_index: int = 0
    event_type: SimulationEventType = "start"
    tone: SimulationEventTone = "info"
    title: str = ""
    detail: str = ""
    actor: str = ""
    node_key: str = ""
    payload: dict[str, Any] = PydanticField(default_factory=dict)


class SimulationJudgementFinding(ContractModel):
    finding_key: str = ""
    title: str = ""
    severity: str = "warning"
    detail: str = ""
    suggested_action: str = ""
    source_refs: list[str] = PydanticField(default_factory=list)


class SimulationJudgement(ContractModel):
    scenario_key: str = ""
    hard_gate_status: SimulationRunOutcome = "needs_revision"
    llm_judgment: SimulationRunOutcome = "needs_revision"
    final_status: SimulationRunOutcome = "needs_revision"
    score: int = 0
    summary: str = ""
    hard_gate_findings: list[str] = PydanticField(default_factory=list)
    findings: list[SimulationJudgementFinding] = PydanticField(default_factory=list)


class SimulationRunRecord(ContractModel):
    id: UUID
    specification_artifact_id: UUID | None = None
    blueprint_version_number: int | None = None
    scenario_key: str = ""
    scenario_title: str = ""
    scenario_version_number: int = 1
    source_action: str = "run_validation_simulation"
    status: ArtifactStatus = ArtifactStatus.draft
    execution_state: SimulationExecutionState = "completed"
    hard_gate_status: SimulationRunOutcome = "needs_revision"
    final_status: SimulationRunOutcome = "needs_revision"
    active_node_key: str = ""
    summary: str = ""
    injected_conditions: list[str] = PydanticField(default_factory=list)
    deterministic_signature: str = ""
    is_stale: bool = False
    stale_reasons: list[str] = PydanticField(default_factory=list)
    events: list[SimulationEvent] = PydanticField(default_factory=list)
    judgement: SimulationJudgement | None = None
    created_at: datetime
    updated_at: datetime


class ValidationScenarioGenerationRequest(ContractModel):
    instructions: str = ""
    focus_areas: list[str] = PydanticField(default_factory=list)


class ValidationSimulationRunRequest(ContractModel):
    scenario_key: str = ""
    scenario_version_number: int | None = None
    initial_input_override: str = ""
    injected_conditions: list[str] = PydanticField(default_factory=list)


class ValidationSimulationEventInjectionRequest(ContractModel):
    run_id: UUID
    injection_type: str = ""
    note: str = ""


class ValidationSimulationJudgeRequest(ContractModel):
    run_id: UUID


class BlueprintArtifact(ContractModel):
    contract_version: str = "blueprint.v1"
    architecture: str = ""
    reasoning_pattern: str = ""
    memory_strategy: str = ""
    tools: list[BlueprintTool] = PydanticField(default_factory=list)
    llm_policy: BlueprintLLMPolicy = PydanticField(default_factory=BlueprintLLMPolicy)
    memory_profile: MemoryProfile = PydanticField(default_factory=MemoryProfile)
    knowledge_profile: KnowledgeProfile = PydanticField(default_factory=KnowledgeProfile)
    safety_checks: list[SafetyCheck] = PydanticField(default_factory=list)
    guardrails: list[str] = PydanticField(default_factory=list)
    delivery_package: DeliveryPackage = PydanticField(default_factory=DeliveryPackage)
    readiness_state: ReviewState = ReviewState.partial
    narrative: str = ""


class DesignCritiqueFinding(ContractModel):
    finding_key: str = ""
    title: str = ""
    severity: str = "warning"
    detail: str = ""
    suggested_action: str = ""
    source_refs: list[str] = PydanticField(default_factory=list)


class DesignRole(ContractModel):
    key: str = ""
    title: str = ""
    responsibility: str = ""
    limits: list[str] = PydanticField(default_factory=list)


class DesignHandoff(ContractModel):
    from_role: str = ""
    to_role: str = ""
    trigger: str = ""
    payload: str = ""
    approval_required: bool = False


class DesignFailureMode(ContractModel):
    scenario: str = ""
    retry_strategy: str = ""
    compensation_strategy: str = ""
    idempotency_notes: str = ""


class DesignBlueprintProjection(ContractModel):
    architecture: str = ""
    reasoning_pattern: str = ""
    safety_checks: list[SafetyCheck] = PydanticField(default_factory=list)
    guardrails: list[str] = PydanticField(default_factory=list)
    narrative: str = ""


class DesignAlternative(ContractModel):
    alternative_key: str = ""
    label: str = ""
    architecture: str = ""
    reasoning_pattern: str = ""
    coordination_model: str = ""
    summary: str = ""
    topology: str = ""
    roles: list[DesignRole] = PydanticField(default_factory=list)
    handoffs: list[DesignHandoff] = PydanticField(default_factory=list)
    approval_points: list[str] = PydanticField(default_factory=list)
    decision_policy: str = ""
    escalation_conditions: list[str] = PydanticField(default_factory=list)
    concurrency_strategy: str = ""
    failure_modes: list[DesignFailureMode] = PydanticField(default_factory=list)
    security_notes: list[str] = PydanticField(default_factory=list)
    operational_complexity: str = "medium"
    relative_cost: str = "medium"
    maintainability: str = "medium"
    tradeoffs: list[str] = PydanticField(default_factory=list)
    assumptions: list[str] = PydanticField(default_factory=list)
    fit_score: float = 0.0
    fit_rationale: list[str] = PydanticField(default_factory=list)
    evidence_refs: list[str] = PydanticField(default_factory=list)
    blueprint_projection: DesignBlueprintProjection = PydanticField(default_factory=DesignBlueprintProjection)


class DesignFitAlternativeScore(ContractModel):
    alternative_key: str = ""
    score: int = 0
    coverage_status: str = "partial"
    rationale: str = ""


class DesignFitMatrixEntry(ContractModel):
    requirement_key: str = ""
    requirement_title: str = ""
    category: str = "functional"
    priority: str = "medium"
    scores: list[DesignFitAlternativeScore] = PydanticField(default_factory=list)


class DesignRequirementCoverageEntry(ContractModel):
    requirement_key: str = ""
    requirement_title: str = ""
    category: str = "functional"
    priority: str = "medium"
    coverage_status: str = "partial"
    rationale: str = ""
    source_refs: list[str] = PydanticField(default_factory=list)


class DesignRecommendationConfidence(ContractModel):
    overall: float = 0.0
    band: str = "low"
    rationale: str = ""


class GuidedAnswerOptionEntry(ContractModel):
    key: str = ""
    label: str = ""
    description: str = ""
    impact: str = ""
    example: str = ""
    recommended: bool = False
    confidence: float = 0.0
    source_refs: list[str] = PydanticField(default_factory=list)


class GuidedQuestionEntry(ContractModel):
    key: str = ""
    question: str = ""
    rationale: str = ""
    priority: Literal["high", "medium", "low"] = "medium"
    blocking_stages: list[str] = PydanticField(default_factory=list)
    suggested_answer: str = ""
    answer_options: list[GuidedAnswerOptionEntry] = PydanticField(default_factory=list)
    stage_scope: str = ""
    deferral_target_stage: str = ""
    inference_summary: str = ""
    confidence: float = 0.0
    source_refs: list[str] = PydanticField(default_factory=list)


class DesignRecommendationArtifact(ContractModel):
    schema_version: str = "design-recommendation.v1"
    alternatives: list[DesignAlternative] = PydanticField(default_factory=list)
    fit_matrix: list[DesignFitMatrixEntry] = PydanticField(default_factory=list)
    recommended_alternative_key: str = ""
    critic_findings: list[DesignCritiqueFinding] = PydanticField(default_factory=list)
    remediation_summary: str = ""
    selected_design: DesignAlternative | None = None
    decision_rationale: str = ""
    requirements_coverage: list[DesignRequirementCoverageEntry] = PydanticField(default_factory=list)
    evidence_refs: list[str] = PydanticField(default_factory=list)
    confidence: DesignRecommendationConfidence = PydanticField(default_factory=DesignRecommendationConfidence)
    open_questions: list[str] = PydanticField(default_factory=list)
    guided_questions: list[GuidedQuestionEntry] = PydanticField(default_factory=list)
    missing_information: list[str] = PydanticField(default_factory=list)
    review_state: ReviewState = ReviewState.partial
    summary: str = ""


class ToolRecommendationSourceStageVersions(ContractModel):
    discover: int | None = None
    define: int | None = None
    design: int | None = None


class ToolRecommendationContextDigest(ContractModel):
    digest_sha256: str = ""
    workflow_summary: str = ""
    constraints_summary: str = ""
    source_refs: list[str] = PydanticField(default_factory=list)


class ToolRecommendationEntry(ContractModel):
    tool_key: str = ""
    tool_label: str = ""
    classification: str = "optional"
    capability_covered: str = ""
    decision_reason: str = ""
    source_evidence: list[str] = PydanticField(default_factory=list)
    dependencies: list[str] = PydanticField(default_factory=list)
    incompatibilities: list[str] = PydanticField(default_factory=list)
    redundant_with: list[str] = PydanticField(default_factory=list)
    confidence: float = 0.0
    contract_seed: BlueprintTool | None = None


class ToolRequirementCoverageEntry(ContractModel):
    requirement_key: str = ""
    requirement_title: str = ""
    category: str = "functional"
    priority: str = "medium"
    coverage_status: str = "partial"
    covered_by_tool_keys: list[str] = PydanticField(default_factory=list)
    rationale: str = ""
    source_refs: list[str] = PydanticField(default_factory=list)


class ToolDesignRoleCoverageEntry(ContractModel):
    role_key: str = ""
    role_title: str = ""
    responsibility: str = ""
    coverage_status: str = "partial"
    covered_by_tool_keys: list[str] = PydanticField(default_factory=list)
    rationale: str = ""
    source_refs: list[str] = PydanticField(default_factory=list)


class ToolRecommendationGap(ContractModel):
    gap_key: str = ""
    title: str = ""
    question: str = ""
    reason: str = ""
    impact: str = ""
    severity: str = "medium"
    suggested_answer: str = ""
    answer_options: list[GuidedAnswerOptionEntry] = PydanticField(default_factory=list)


class ToolRecommendationConfidence(ContractModel):
    overall: float = 0.0
    band: str = "low"
    rationale: str = ""


class ToolRecommendationReviewDecision(ContractModel):
    tool_key: str = ""
    classification: str = ""
    decision: str = "approved"
    decision_reason: str = ""


class ToolApiBindRequest(ContractModel):
    tool_name: str = ""
    registered_api_ref: str = ""
    endpoint_reference: str = ""
    auth_reference: str = ""
    openapi_spec: dict[str, Any] = PydanticField(default_factory=dict)


class ApprovedToolDigestEntry(ContractModel):
    tool_key: str = ""
    tool_label: str = ""
    blueprint_tool_name: str = ""
    classification: str = "optional"
    integration_kind: str = ""
    owner: str = ""
    requires_approval: bool = False
    has_side_effects: bool = False
    memory_implications: list[str] = PydanticField(default_factory=list)


class ApprovedToolsDigest(ContractModel):
    digest_version: str = "approved-tools-digest.v1"
    digest_sha256: str = ""
    source_session_id: UUID | None = None
    source_blueprint_version: int | None = None
    promoted_blueprint_version: int | None = None
    tool_count: int = 0
    approved_tool_keys: list[str] = PydanticField(default_factory=list)
    mandatory_tool_keys: list[str] = PydanticField(default_factory=list)
    optional_tool_keys: list[str] = PydanticField(default_factory=list)
    side_effect_tool_keys: list[str] = PydanticField(default_factory=list)
    approval_required_tool_keys: list[str] = PydanticField(default_factory=list)
    knowledge_tool_keys: list[str] = PydanticField(default_factory=list)
    selected_blueprint_tool_names: list[str] = PydanticField(default_factory=list)
    retrieval_scopes: list[str] = PydanticField(default_factory=list)
    memory_hints: list[str] = PydanticField(default_factory=list)
    recommended_memory_strategy: str = ""
    summary: str = ""


class ToolRecommendationFinding(ContractModel):
    finding_key: str = ""
    title: str = ""
    detail: str = ""
    severity: Literal["info", "warning", "blocking"] = "info"
    category: str = ""
    affected_tool_keys: list[str] = PydanticField(default_factory=list)
    suggested_action: str = ""


class ToolRecommendationEvaluation(ContractModel):
    overall_status: ReviewState = ReviewState.partial
    coverage_status: ReviewState = ReviewState.partial
    minimality_status: ReviewState = ReviewState.partial
    compatibility_status: ReviewState = ReviewState.partial
    governance_status: ReviewState = ReviewState.partial
    promotion_blocked: bool = False
    findings: list[ToolRecommendationFinding] = PydanticField(default_factory=list)
    recommended_actions: list[str] = PydanticField(default_factory=list)
    summary: str = ""


class ToolRecommendationAllowedToolKey(str, Enum):
    read_system_of_record = "read_system_of_record"
    approval_gate = "approval_gate"
    transactional_write = "transactional_write"
    knowledge_retrieval = "knowledge_retrieval"
    document_ingestion = "document_ingestion"
    outbound_notification = "outbound_notification"
    human_handoff = "human_handoff"
    scheduler = "scheduler"


class ToolRecommendationPromptToolOption(ContractModel):
    tool_key: ToolRecommendationAllowedToolKey = ToolRecommendationAllowedToolKey.read_system_of_record
    tool_label: str = ""
    family_key: str = ""
    family_status: str = "candidate"
    capability_covered: str = ""
    reason: str = ""
    selection_notes: list[str] = PydanticField(default_factory=list)


class ToolRecommendationPromptInput(ContractModel):
    prompt_version: str = "tool-recommendation-prompt.v1"
    source_session_id: UUID | None = None
    source_blueprint_version: int | None = None
    case_classification: str = ""
    agent_goal: str = ""
    primary_user: str = ""
    workflow_summary: str = ""
    constraints_summary: str = ""
    source_refs: list[str] = PydanticField(default_factory=list)
    core_workflows: list[str] = PydanticField(default_factory=list)
    interaction_modes: list[str] = PydanticField(default_factory=list)
    required_information_sources: list[str] = PydanticField(default_factory=list)
    required_write_actions: list[str] = PydanticField(default_factory=list)
    approval_boundaries: list[str] = PydanticField(default_factory=list)
    hard_constraints: list[str] = PydanticField(default_factory=list)
    mandatory_tool_keys: list[ToolRecommendationAllowedToolKey] = PydanticField(default_factory=list)
    forbidden_tool_keys: list[ToolRecommendationAllowedToolKey] = PydanticField(default_factory=list)
    candidate_tools: list[ToolRecommendationPromptToolOption] = PydanticField(default_factory=list)
    requirements_coverage: list[ToolRequirementCoverageEntry] = PydanticField(default_factory=list)
    design_role_coverage: list[ToolDesignRoleCoverageEntry] = PydanticField(default_factory=list)
    existing_gaps: list[ToolRecommendationGap] = PydanticField(default_factory=list)
    compact_evidence: list[str] = PydanticField(default_factory=list)


class ToolRecommendationLLMDecision(ContractModel):
    tool_key: ToolRecommendationAllowedToolKey = ToolRecommendationAllowedToolKey.read_system_of_record
    classification: Literal["mandatory", "optional", "unnecessary"] = "optional"
    decision_reason: str = ""
    source_evidence: list[str] = PydanticField(default_factory=list)
    dependencies: list[ToolRecommendationAllowedToolKey] = PydanticField(default_factory=list)
    incompatibilities: list[str] = PydanticField(default_factory=list)
    redundant_with: list[ToolRecommendationAllowedToolKey] = PydanticField(default_factory=list)
    confidence: float = 0.0


class ToolRecommendationLLMOutput(ContractModel):
    summary: str = ""
    confidence: ToolRecommendationConfidence = PydanticField(default_factory=ToolRecommendationConfidence)
    tool_decisions: list[ToolRecommendationLLMDecision] = PydanticField(default_factory=list)
    needs_information: list[ToolRecommendationGap] = PydanticField(default_factory=list)
    coverage_gaps: list[ToolRecommendationGap] = PydanticField(default_factory=list)


class ToolPreflightCapability(ContractModel):
    capability_key: str = ""
    label: str = ""
    required: bool = False
    reason: str = ""
    source_evidence: list[str] = PydanticField(default_factory=list)
    confidence: float = 0.0


class ToolFamilyCandidate(ContractModel):
    family_key: str = ""
    label: str = ""
    status: str = "candidate"
    supported_capabilities: list[str] = PydanticField(default_factory=list)
    matched_signals: list[str] = PydanticField(default_factory=list)
    rejected_by_constraints: list[str] = PydanticField(default_factory=list)
    suggested_tool_keys: list[str] = PydanticField(default_factory=list)
    estimated_complexity: str = "medium"
    reason: str = ""


class ToolRecommendationPreflight(ContractModel):
    case_classification: str = ""
    agent_goal: str = ""
    primary_user: str = ""
    core_workflows: list[str] = PydanticField(default_factory=list)
    interaction_modes: list[str] = PydanticField(default_factory=list)
    required_information_sources: list[str] = PydanticField(default_factory=list)
    required_write_actions: list[str] = PydanticField(default_factory=list)
    approval_boundaries: list[str] = PydanticField(default_factory=list)
    hard_constraints: list[str] = PydanticField(default_factory=list)
    mandatory_capabilities: list[ToolPreflightCapability] = PydanticField(default_factory=list)
    forbidden_capabilities: list[str] = PydanticField(default_factory=list)
    candidate_tool_families: list[ToolFamilyCandidate] = PydanticField(default_factory=list)
    missing_information: list[ToolRecommendationGap] = PydanticField(default_factory=list)


class ToolRecommendationArtifact(ContractModel):
    schema_version: str = "tool-recommendation.v1"
    source_session_id: UUID | None = None
    source_blueprint_version: int | None = None
    current_blueprint_version: int | None = None
    generation_instructions: str = ""
    is_stale: bool = False
    stale_reasons: list[str] = PydanticField(default_factory=list)
    source_stage_versions: ToolRecommendationSourceStageVersions = PydanticField(
        default_factory=ToolRecommendationSourceStageVersions
    )
    context_digest: ToolRecommendationContextDigest = PydanticField(default_factory=ToolRecommendationContextDigest)
    preflight: ToolRecommendationPreflight = PydanticField(default_factory=ToolRecommendationPreflight)
    recommended_tools: list[ToolRecommendationEntry] = PydanticField(default_factory=list)
    optional_tools: list[ToolRecommendationEntry] = PydanticField(default_factory=list)
    rejected_tools: list[ToolRecommendationEntry] = PydanticField(default_factory=list)
    requirements_coverage: list[ToolRequirementCoverageEntry] = PydanticField(default_factory=list)
    design_role_coverage: list[ToolDesignRoleCoverageEntry] = PydanticField(default_factory=list)
    coverage_gaps: list[ToolRecommendationGap] = PydanticField(default_factory=list)
    needs_information: list[ToolRecommendationGap] = PydanticField(default_factory=list)
    confidence: ToolRecommendationConfidence = PydanticField(default_factory=ToolRecommendationConfidence)
    review_decisions: list[ToolRecommendationReviewDecision] = PydanticField(default_factory=list)
    approved_tools_digest: ApprovedToolsDigest | None = None
    evaluation: ToolRecommendationEvaluation = PydanticField(default_factory=ToolRecommendationEvaluation)
    review_state: ReviewState = ReviewState.partial
    summary: str = ""


class MemoryRecommendationSourceStageVersions(ContractModel):
    discover: int | None = None
    define: int | None = None
    design: int | None = None
    tools: int | None = None


class MemoryNeedDecision(ContractModel):
    mode: str = "session_only"
    required: bool = True
    summary: str = ""
    rationale: str = ""
    source_refs: list[str] = PydanticField(default_factory=list)


class MemoryLayerDesign(ContractModel):
    layer_key: str = ""
    label: str = ""
    owner: str = ""
    summary: str = ""
    stores: list[str] = PydanticField(default_factory=list)
    write_triggers: list[str] = PydanticField(default_factory=list)
    read_paths: list[str] = PydanticField(default_factory=list)
    compaction_policy: str = ""
    retention_policy: str = ""


class MemoryKnowledgeDesign(ContractModel):
    mode: str = "none"
    rag_required: bool = False
    summary: str = ""
    source_scope: str = ""
    approved_sources: list[KnowledgeSource] = PydanticField(default_factory=list)
    ingestion_policy: IngestionPolicy = PydanticField(default_factory=IngestionPolicy)
    embedding_policy: EmbeddingPolicy = PydanticField(default_factory=EmbeddingPolicy)
    retrieval_policy: RetrievalPolicyProfile = PydanticField(default_factory=RetrievalPolicyProfile)
    refresh_policy: RefreshPolicy = PydanticField(default_factory=RefreshPolicy)
    grounding_policy: GroundingPolicy = PydanticField(default_factory=GroundingPolicy)
    notes: list[str] = PydanticField(default_factory=list)


class MemoryContextBudgetEntry(ContractModel):
    role: str = ""
    task_kind: str = ""
    max_context_tokens: int = 0
    max_short_term_items: int = 0
    max_retrieved_sources: int = 0
    strategy: str = ""
    source_refs: list[str] = PydanticField(default_factory=list)


class MemoryWriteReadRule(ContractModel):
    scope: str = ""
    owner: str = ""
    write_when: str = ""
    do_not_write_when: str = ""
    read_when: str = ""
    compact_when: str = ""


class MemoryRetentionDeletionRule(ContractModel):
    scope: str = ""
    retention_policy: str = ""
    ttl_policy: str = ""
    deletion_policy: str = ""
    residency: str = ""
    source_refs: list[str] = PydanticField(default_factory=list)


class MemorySensitivityIsolationRule(ContractModel):
    scope: str = ""
    isolation_mode: str = ""
    data_classes: list[str] = PydanticField(default_factory=list)
    restrictions: list[str] = PydanticField(default_factory=list)
    source_refs: list[str] = PydanticField(default_factory=list)


class MemoryToolDependency(ContractModel):
    tool_key: str = ""
    required: bool = False
    status: str = "optional"
    reason: str = ""
    capabilities: list[str] = PydanticField(default_factory=list)


class MemoryRecommendationFinding(ContractModel):
    finding_key: str = ""
    title: str = ""
    detail: str = ""
    severity: Literal["info", "warning", "blocking"] = "info"
    category: str = ""
    suggested_action: str = ""
    source_refs: list[str] = PydanticField(default_factory=list)


class MemoryRecommendationConfidence(ContractModel):
    overall: float = 0.0
    band: str = "low"
    rationale: str = ""


class MemoryDryCompileStatus(ContractModel):
    status: str = "pending"
    summary: str = ""
    generated_contracts: list[str] = PydanticField(default_factory=list)
    blocking_issues: list[str] = PydanticField(default_factory=list)


class MemoryRecommendationArtifact(ContractModel):
    schema_version: str = "memory-recommendation.v1"
    source_session_id: UUID | None = None
    source_blueprint_version: int | None = None
    current_blueprint_version: int | None = None
    generation_instructions: str = ""
    is_stale: bool = False
    stale_reasons: list[str] = PydanticField(default_factory=list)
    source_stage_versions: MemoryRecommendationSourceStageVersions = PydanticField(
        default_factory=MemoryRecommendationSourceStageVersions
    )
    summary: str = ""
    memory_need_decision: MemoryNeedDecision = PydanticField(default_factory=MemoryNeedDecision)
    short_term_design: MemoryLayerDesign = PydanticField(default_factory=MemoryLayerDesign)
    working_memory_design: MemoryLayerDesign = PydanticField(default_factory=MemoryLayerDesign)
    long_term_design: MemoryLayerDesign = PydanticField(default_factory=MemoryLayerDesign)
    knowledge_design: MemoryKnowledgeDesign = PydanticField(default_factory=MemoryKnowledgeDesign)
    context_budget_plan: list[MemoryContextBudgetEntry] = PydanticField(default_factory=list)
    write_read_matrix: list[MemoryWriteReadRule] = PydanticField(default_factory=list)
    retention_and_deletion: list[MemoryRetentionDeletionRule] = PydanticField(default_factory=list)
    sensitivity_and_isolation: list[MemorySensitivityIsolationRule] = PydanticField(default_factory=list)
    tool_dependencies: list[MemoryToolDependency] = PydanticField(default_factory=list)
    critic_findings: list[MemoryRecommendationFinding] = PydanticField(default_factory=list)
    evidence_refs: list[str] = PydanticField(default_factory=list)
    open_questions: list[str] = PydanticField(default_factory=list)
    guided_questions: list[GuidedQuestionEntry] = PydanticField(default_factory=list)
    missing_information: list[str] = PydanticField(default_factory=list)
    confidence: MemoryRecommendationConfidence = PydanticField(default_factory=MemoryRecommendationConfidence)
    dry_compile_status: MemoryDryCompileStatus = PydanticField(default_factory=MemoryDryCompileStatus)
    proposed_memory_profile: MemoryProfile = PydanticField(default_factory=MemoryProfile)
    proposed_knowledge_profile: KnowledgeProfile = PydanticField(default_factory=KnowledgeProfile)
    review_state: ReviewState = ReviewState.partial


class FeatureFlagEntry(ContractModel):
    key: str = ""
    enabled: bool = False
    description: str = ""
    stage_hint: str = ""


class CatalogItemSummary(ContractModel):
    item_key: str = ""
    label: str = ""
    status: str = ""
    summary: str = ""


class CatalogSummaryEntry(ContractModel):
    catalog_key: str = ""
    version: str = ""
    item_count: int = 0
    active_count: int = 0
    items: list[CatalogItemSummary] = PydanticField(default_factory=list)


class WorkspaceSectionEntry(ContractModel):
    key: str = ""
    label: str = ""
    view_kind: str = ""
    capability_status: str = ""
    source_of_truth: str = ""
    read_only: bool = False
    summary: str = ""


class WorkspaceContract(ContractModel):
    contract_version: str = "workspace-contract.v1"
    sections: list[WorkspaceSectionEntry] = PydanticField(default_factory=list)
    feature_flags: list[FeatureFlagEntry] = PydanticField(default_factory=list)
    catalogs: list[CatalogSummaryEntry] = PydanticField(default_factory=list)

    @field_validator("sections")
    @classmethod
    def validate_sections(cls, value: list[WorkspaceSectionEntry]) -> list[WorkspaceSectionEntry]:
        seen: set[str] = set()
        for item in value:
            if item.key in seen:
                raise ValueError(f"Duplicated workspace section key: {item.key}")
            seen.add(item.key)
        return value


class SkillDefinition(ContractModel):
    skill_key: str = ""
    label: str = ""
    stage_hint: str = ""
    summary: str = ""
    evidence_policy: str = ""
    input_schema: dict[str, Any] = PydanticField(default_factory=dict)
    output_schema: dict[str, Any] = PydanticField(default_factory=dict)
    is_active: bool = True


class SkillRunArtifact(ContractModel):
    artifact_role: str = ""
    artifact_kind: str = ""
    payload: dict[str, Any] = PydanticField(default_factory=dict)


class SkillRunEntry(ContractModel):
    id: UUID
    skill_key: str = ""
    label: str = ""
    stage: SessionStage
    blueprint_version_number: int | None = None
    source_action: str = ""
    status: ArtifactStatus
    duration_ms: int = 0
    result_summary: str = ""
    warnings: list[str] = PydanticField(default_factory=list)
    evidence: list[EvidenceItem] = PydanticField(default_factory=list)
    llm_trace: LLMContextTrace | None = None
    artifacts: list[SkillRunArtifact] = PydanticField(default_factory=list)
    created_at: datetime


class ArtifactRecordEntry(ContractModel):
    id: UUID
    blueprint_version_number: int | None = None
    artifact_key: str = ""
    artifact_title: str = ""
    artifact_kind: str = ""
    stage: SessionStage
    source_action: str = ""
    export_format: str = ""
    content_text: str = ""
    content_hash: str = ""
    artifact_metadata: dict[str, Any] = PydanticField(default_factory=dict)
    created_at: datetime


class JourneyArtifactEvidenceEntry(ContractModel):
    source_type: str = ""
    source_id: str = ""
    source_version: str = ""
    source_lineage: list[str] = PydanticField(default_factory=list)
    section_key: str = ""
    artifact_ref: str = ""
    retrieval_score: float | None = None
    authority_level: str = ""
    used_for: str = ""
    citation_label: str = ""
    detail: str = ""


class JourneyStageDecisionEntry(ContractModel):
    id: UUID
    artifact_id: UUID
    stage_key: str = ""
    decision_type: JourneyDecisionType
    previous_state: JourneyArtifactState | None = None
    next_state: JourneyArtifactState | None = None
    actor_user_id: UUID | None = None
    note: str = ""
    payload: dict[str, Any] = PydanticField(default_factory=dict)
    created_at: datetime


class JourneyStageArtifactEntry(ContractModel):
    id: UUID
    workspace_id: UUID
    session_id: UUID
    artifact_kind: str = ""
    stage_key: str = ""
    version_number: int = 1
    state: JourneyArtifactState = JourneyArtifactState.generated
    source_action: str = "manual_draft"
    proposal_payload: dict[str, Any] = PydanticField(default_factory=dict)
    user_patch: dict[str, Any] = PydanticField(default_factory=dict)
    source_stage_versions: dict[str, Any] = PydanticField(default_factory=dict)
    input_fingerprint: str = ""
    context_fingerprint: str = ""
    output_fingerprint: str = ""
    corpus_hash: str = ""
    provider_key: str = ""
    model: str = ""
    execution_backend: str = ""
    prompt_version: str = ""
    schema_version: str = ""
    confidence: float | None = None
    missing_information: list[str] = PydanticField(default_factory=list)
    warnings: list[str] = PydanticField(default_factory=list)
    evidence_manifest: list[JourneyArtifactEvidenceEntry] = PydanticField(default_factory=list)
    stale_reasons: list[str] = PydanticField(default_factory=list)
    based_on_artifact_id: UUID | None = None
    superseded_by_artifact_id: UUID | None = None
    approved_by_user_id: UUID | None = None
    reviewed_at: datetime | None = None
    approved_at: datetime | None = None
    rejected_at: datetime | None = None
    stale_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    decisions: list[JourneyStageDecisionEntry] = PydanticField(default_factory=list)


class JourneyStageArtifactListResponse(ContractModel):
    items: list[JourneyStageArtifactEntry] = PydanticField(default_factory=list)
    latest: JourneyStageArtifactEntry | None = None


class BlueprintConsistencyIssue(ContractModel):
    issue_key: str = ""
    severity: Literal["info", "warning", "blocking"] = "warning"
    category: str = ""
    title: str = ""
    detail: str = ""
    affected_stage_keys: list[str] = PydanticField(default_factory=list)
    source_refs: list[str] = PydanticField(default_factory=list)
    citations: list[str] = PydanticField(default_factory=list)


class ApprovedStageLineageEntry(ContractModel):
    stage_key: str = ""
    artifact_id: UUID | None = None
    artifact_kind: str = ""
    source_action: str = ""
    version_number: int | None = None
    state: str = ""
    approved_at: datetime | None = None
    decision_count: int = 0
    rejection_count: int = 0
    citation_labels: list[str] = PydanticField(default_factory=list)
    lineage_refs: list[str] = PydanticField(default_factory=list)


class BlueprintConsistencyReport(ContractModel):
    overall_status: ReviewState = ReviewState.partial
    summary: str = ""
    generated_from_blueprint_version: int | None = None
    approved_stage_lineage: list[ApprovedStageLineageEntry] = PydanticField(default_factory=list)
    issues: list[BlueprintConsistencyIssue] = PydanticField(default_factory=list)
    blocking_issues: list[str] = PydanticField(default_factory=list)
    warnings: list[str] = PydanticField(default_factory=list)
    uncovered_requirement_keys: list[str] = PydanticField(default_factory=list)
    orphan_design_role_keys: list[str] = PydanticField(default_factory=list)
    orphan_tool_keys: list[str] = PydanticField(default_factory=list)
    orphan_memory_dependency_keys: list[str] = PydanticField(default_factory=list)
    stale_stage_keys: list[str] = PydanticField(default_factory=list)
    exportable_lineage: list[str] = PydanticField(default_factory=list)
    restricted_lineage: list[str] = PydanticField(default_factory=list)
    decision_history: list[dict[str, Any]] = PydanticField(default_factory=list)


class MetricSnapshotEntry(ContractModel):
    id: UUID
    source_action: str = ""
    cost_estimate_usd: float = 0
    total_duration_ms: int = 0
    error_count: int = 0
    warning_count: int = 0
    approvals_pending: int = 0
    approvals_resolved: int = 0
    regenerations_count: int = 0
    needs_review_count: int = 0
    latest_evaluation_score: int | None = None
    latest_evaluation_status: str = ""
    export_count: int = 0
    artifact_count: int = 0
    created_at: datetime


class MemoryObservabilityMetric(ContractModel):
    key: str = ""
    label: str = ""
    value: float = 0
    unit: str = "%"
    numerator: int = 0
    denominator: int = 0
    status: str = "ok"
    detail: str = ""


class MemoryDashboardEntry(ContractModel):
    scope_key: str = ""
    label: str = ""
    llm_runs: int = 0
    grounded_hit_rate: float = 0
    citation_coverage: float = 0
    stale_rate: float = 0
    average_budget_utilization: float = 0
    average_compression_gain: float = 0


class MemoryValidationCheckEntry(ContractModel):
    check_key: str = ""
    label: str = ""
    status: str = "not_applicable"
    summary: str = ""
    evidence: list[str] = PydanticField(default_factory=list)


class MemoryObservabilityReport(ContractModel):
    llm_run_count: int = 0
    traced_source_count: int = 0
    grounded_hit_runs: int = 0
    stale_source_count: int = 0
    recent_warnings: list[str] = PydanticField(default_factory=list)
    metrics: list[MemoryObservabilityMetric] = PydanticField(default_factory=list)
    by_agent: list[MemoryDashboardEntry] = PydanticField(default_factory=list)
    by_stage: list[MemoryDashboardEntry] = PydanticField(default_factory=list)
    validations: list[MemoryValidationCheckEntry] = PydanticField(default_factory=list)


class AlertEventEntry(ContractModel):
    id: UUID
    alert_key: str = ""
    severity: str = ""
    title: str = ""
    message: str = ""
    status: str = "active"
    evidence: list[str] = PydanticField(default_factory=list)
    created_at: datetime
    updated_at: datetime
    resolved_at: datetime | None = None


class IntegrationStatusEntry(ContractModel):
    id: UUID
    integration_key: str = ""
    label: str = ""
    status: str = ""
    configured: bool = False
    reachable: bool = False
    detail: str = ""
    checked_at: datetime


class MonitoringContextBackendEntry(ContractModel):
    key: str = ""
    label: str = ""
    run_count: int = 0
    share_percent: float = 0.0


class MonitoringProviderObservabilityEntry(ContractModel):
    provider_key: str = ""
    model_name: str = ""
    execution_backend: str = ""
    effective_context_backend: str = ""
    run_count: int = 0
    fallback_count: int = 0
    degraded_count: int = 0
    long_term_hit_count: int = 0
    total_duration_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class MonitoringStageObservabilityEntry(ContractModel):
    stage_key: str = ""
    label: str = ""
    run_count: int = 0
    success_count: int = 0
    needs_review_count: int = 0
    failure_count: int = 0
    approved_artifact_count: int = 0
    stale_artifact_count: int = 0
    rerun_count: int = 0
    long_term_hit_count: int = 0
    average_confidence: float = 0.0
    simulation_run_count: int = 0
    simulation_pass_rate: float = 0.0


class MonitoringCapabilityObservabilityEntry(ContractModel):
    capability_key: str = ""
    label: str = ""
    run_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    fallback_count: int = 0
    degraded_count: int = 0
    long_term_hit_count: int = 0


class MonitoringReleaseGateEntry(ContractModel):
    gate_key: str = ""
    label: str = ""
    status: str = "warning"
    detail: str = ""
    evidence: list[str] = PydanticField(default_factory=list)


class MonitoringReleaseObservability(ContractModel):
    total_llm_runs: int = 0
    real_llm_runs: int = 0
    fallback_runs: int = 0
    fallback_rate: float = 0.0
    degraded_runs: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    average_latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    average_compaction_ratio: float = 0.0
    context_fingerprint_coverage: float = 0.0
    source_version_coverage: float = 0.0
    approval_resolution_rate: float = 0.0
    stale_artifact_count: int = 0
    rerun_count: int = 0
    long_term_hit_count: int = 0
    simulation_run_count: int = 0
    simulation_pass_rate: float = 0.0
    auth_or_isolation_error_count: int = 0
    project_actuals_count: int = 0
    estimation_error_metric_count: int = 0
    estimation_band_hit_rate: float = 0.0
    context_backends: list[MonitoringContextBackendEntry] = PydanticField(default_factory=list)
    providers: list[MonitoringProviderObservabilityEntry] = PydanticField(default_factory=list)
    stages: list[MonitoringStageObservabilityEntry] = PydanticField(default_factory=list)
    capabilities: list[MonitoringCapabilityObservabilityEntry] = PydanticField(default_factory=list)
    release_gates: list[MonitoringReleaseGateEntry] = PydanticField(default_factory=list)


class WorkflowTemplateEntry(ContractModel):
    id: UUID
    template_key: str = ""
    label: str = ""
    summary: str = ""
    architecture_scope: list[str] = PydanticField(default_factory=list)
    supports_approvals: bool = False
    supports_handoffs: bool = False
    workflow_profile: WorkflowProfile = PydanticField(default_factory=WorkflowProfile)
    governance_hints: list[str] = PydanticField(default_factory=list)
    is_active: bool = True
    updated_at: datetime


class HandoffRecordEntry(ContractModel):
    id: UUID
    blueprint_version_number: int | None = None
    handoff_key: str = ""
    title: str = ""
    from_stage: SessionStage
    to_stage: SessionStage
    status: str = "pending"
    owner_role: str = ""
    triggered_by: str = ""
    summary: str = ""
    resolution_note: str = ""
    payload: dict[str, Any] = PydanticField(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    resolved_at: datetime | None = None


class GovernancePolicyEntry(ContractModel):
    id: UUID
    policy_key: str = ""
    label: str = ""
    summary: str = ""
    scope: str = ""
    is_active: bool = True
    policy_payload: dict[str, Any] = PydanticField(default_factory=dict)
    compliance_status: str = "unknown"
    evidence: list[str] = PydanticField(default_factory=list)
    updated_at: datetime


class SubagentRunEntry(ContractModel):
    id: UUID
    blueprint_version_number: int | None = None
    run_kind: str = ""
    title: str = ""
    status: ArtifactStatus = ArtifactStatus.draft
    feature_flag_key: str = ""
    summary: str = ""
    input_payload: dict[str, Any] = PydanticField(default_factory=dict)
    output_payload: dict[str, Any] = PydanticField(default_factory=dict)
    created_at: datetime


class MonitoringWorkspace(ContractModel):
    current_metrics: MetricSnapshotEntry | None = None
    history: list[MetricSnapshotEntry] = PydanticField(default_factory=list)
    alerts: list[AlertEventEntry] = PydanticField(default_factory=list)
    recent_errors: list[ExecutionLogEntry] = PydanticField(default_factory=list)
    integrations: list[IntegrationStatusEntry] = PydanticField(default_factory=list)
    memory_observability: MemoryObservabilityReport | None = None
    release_observability: MonitoringReleaseObservability | None = None


class ArtifactBrowserResponse(ContractModel):
    items: list[ArtifactRecordEntry] = PydanticField(default_factory=list)


class KnowledgeDocumentEntry(ContractModel):
    id: UUID | None = None
    scope: KnowledgeScope = KnowledgeScope.platform
    workspace_id: UUID | None = None
    session_id: UUID | None = None
    relative_path: str = ""
    title: str = ""
    format: str = ""
    visibility: KnowledgeVisibility = KnowledgeVisibility.platform
    status: KnowledgeDocumentStatus = KnowledgeDocumentStatus.approved
    authority_level: str = ""
    memory_usage: str = ""
    stage_affinity: list[str] = PydanticField(default_factory=list)
    agent_affinity: list[str] = PydanticField(default_factory=list)
    version_number: int = 0
    section_count: int = 0
    content_hash: str = ""
    source_lineage: str = ""
    approved_at: datetime | None = None
    effective_from: datetime | None = None
    expires_at: datetime | None = None
    updated_at: datetime | None = None


class KnowledgeSearchHit(ContractModel):
    document_id: UUID | None = None
    scope: KnowledgeScope = KnowledgeScope.platform
    workspace_id: UUID | None = None
    session_id: UUID | None = None
    relative_path: str = ""
    section_key: str = ""
    title: str = ""
    heading_path: list[str] = PydanticField(default_factory=list)
    visibility: KnowledgeVisibility = KnowledgeVisibility.platform
    status: KnowledgeDocumentStatus = KnowledgeDocumentStatus.approved
    authority_level: str = ""
    memory_usage: str = ""
    stage_affinity: list[str] = PydanticField(default_factory=list)
    agent_affinity: list[str] = PydanticField(default_factory=list)
    source_lineage: str = ""
    preview: str = ""
    score: float = 0
    lexical_score: float = 0
    vector_score: float = 0
    version_number: int = 0
    approved_at: datetime | None = None
    effective_from: datetime | None = None
    expires_at: datetime | None = None


class KnowledgeSearchResponse(ContractModel):
    query: str = ""
    role: str = ""
    total_hits: int = 0
    grounded_hits: int = 0
    corpus_hash: str = ""
    evidence_status: str = ""
    absence_reason: str = ""
    applied_filters: list[str] = PydanticField(default_factory=list)
    authorized_scopes: list[str] = PydanticField(default_factory=list)
    citations: list[str] = PydanticField(default_factory=list)
    next_cursor: str = ""
    discarded_hits: int = 0
    items: list[KnowledgeSearchHit] = PydanticField(default_factory=list)


class KnowledgeIngestionReport(ContractModel):
    run_id: UUID | None = None
    source_root: str = ""
    scope: KnowledgeScope = KnowledgeScope.platform
    workspace_id: UUID | None = None
    session_id: UUID | None = None
    status: str = ""
    corpus_hash: str = ""
    document_count: int = 0
    changed_document_count: int = 0
    unchanged_document_count: int = 0
    section_count: int = 0
    lexical_term_count: int = 0
    vector_dimensions: int = 0
    filesystem_manifest_path: str = ""
    lexical_index_path: str = ""
    vector_index_path: str = ""
    changed_paths: list[str] = PydanticField(default_factory=list)
    documents: list[KnowledgeDocumentEntry] = PydanticField(default_factory=list)
    created_at: datetime | None = None


class KnowledgeCorpusStatus(ContractModel):
    source_root: str = ""
    scope: KnowledgeScope = KnowledgeScope.platform
    workspace_id: UUID | None = None
    session_id: UUID | None = None
    status: str = ""
    corpus_hash: str = ""
    document_count: int = 0
    section_count: int = 0
    lexical_term_count: int = 0
    vector_dimensions: int = 0
    filesystem_manifest_path: str = ""
    lexical_index_path: str = ""
    vector_index_path: str = ""
    last_ingested_at: datetime | None = None
    latest_documents: list[KnowledgeDocumentEntry] = PydanticField(default_factory=list)


class KnowledgeManagedDocumentUpsertRequest(ContractModel):
    scope: KnowledgeScope = KnowledgeScope.workspace
    session_id: UUID | None = None
    relative_path: str = ""
    content_text: str = ""
    authority_level: str = "operational"
    memory_usage: str = "candidate_retrieval"
    stage_affinity: list[str] = PydanticField(default_factory=list)
    agent_affinity: list[str] = PydanticField(default_factory=list)
    visibility: KnowledgeVisibility | None = None
    status: KnowledgeDocumentStatus = KnowledgeDocumentStatus.approved
    effective_from: datetime | None = None
    expires_at: datetime | None = None


class KnowledgeDocumentGovernancePatchRequest(ContractModel):
    authority_level: str | None = None
    memory_usage: str | None = None
    stage_affinity: list[str] | None = None
    agent_affinity: list[str] | None = None
    visibility: KnowledgeVisibility | None = None
    status: KnowledgeDocumentStatus | None = None
    effective_from: datetime | None = None
    expires_at: datetime | None = None


ACPFileStatus = Literal["complete", "incomplete", "needs_review"]
ACPValidationSeverity = Literal["info", "warning", "error"]
ConstructionGapSeverity = Literal["info", "warning", "blocking"]
ConstructionGapStatus = Literal["open", "answered", "waived", "resolved"]
ConstructionQuestionStatus = Literal["open", "answered", "resolved"]
ConstructionReadinessStatus = Literal["not_started", "needs_questions", "blocked", "ready_to_build"]


class ACPFileEntry(ContractModel):
    path: str = ""
    domain: str = ""
    title: str = ""
    format: str = ""
    status: ACPFileStatus = "incomplete"
    source_sections: list[str] = PydanticField(default_factory=list)
    missing_fields: list[str] = PydanticField(default_factory=list)
    warnings: list[str] = PydanticField(default_factory=list)
    content_text: str = ""
    content_hash: str = ""

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        if value not in ALLOWED_ACP_FILE_STATUSES:
            raise ValueError(f"Ungoverned ACP file status: {value}")
        return value


class ACPValidationIssue(ContractModel):
    code: str = ""
    severity: ACPValidationSeverity = "warning"
    path: str = ""
    message: str = ""
    remediation: str = ""
    source_sections: list[str] = PydanticField(default_factory=list)
    blocking: bool = False

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, value: str) -> str:
        if value not in ALLOWED_ACP_VALIDATION_SEVERITIES:
            raise ValueError(f"Ungoverned ACP validation severity: {value}")
        return value


class ACPValidationReport(ContractModel):
    overall_status: ACPFileStatus = "needs_review"
    completeness_percent: int = 0
    can_export_zip: bool = False
    issues: list[ACPValidationIssue] = PydanticField(default_factory=list)

    @field_validator("overall_status")
    @classmethod
    def validate_overall_status(cls, value: str) -> str:
        if value not in ALLOWED_ACP_FILE_STATUSES:
            raise ValueError(f"Ungoverned ACP overall status: {value}")
        return value

    @field_validator("completeness_percent")
    @classmethod
    def validate_completeness_percent(cls, value: int) -> int:
        if not 0 <= value <= 100:
            raise ValueError("ACP completeness percent must be between 0 and 100")
        return value


class ConstructionQuestionOption(ContractModel):
    key: str = ""
    label: str = ""
    description: str = ""
    impact: str = ""
    example: str = ""
    recommended: bool = False
    confidence: float = 0.0
    source_refs: list[str] = PydanticField(default_factory=list)


class ConstructionQuestionEntry(ContractModel):
    question_key: str = ""
    question_text: str = ""
    rationale: str = ""
    purpose: str = ""
    expected_answer_format: str = ""
    target_owner: str = ""
    blocking: bool = False
    options: list[ConstructionQuestionOption] = PydanticField(default_factory=list)


class ConstructionQuestionViewEntry(ContractModel):
    question_key: str = ""
    gap_key: str = ""
    gap_title: str = ""
    domain: str = ""
    question_text: str = ""
    rationale: str = ""
    purpose: str = ""
    expected_answer_format: str = ""
    target_owner: str = ""
    blocking: bool = False
    status: ConstructionQuestionStatus = "open"
    answer_text: str = ""
    owner_role: str = ""
    answered_by_display: str = ""
    answered_at: datetime | None = None
    resolved_at: datetime | None = None
    impacted_artifacts: list[str] = PydanticField(default_factory=list)
    options: list[ConstructionQuestionOption] = PydanticField(default_factory=list)

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        if value not in ALLOWED_CONSTRUCTION_QUESTION_STATUSES:
            raise ValueError(f"Ungoverned construction question status: {value}")
        return value


class ConstructionQuestionAnswerRequest(ContractModel):
    answer_text: str = ""
    owner_role: str = ""
    impacted_artifacts: list[str] = PydanticField(default_factory=list)

    @field_validator("answer_text")
    @classmethod
    def validate_answer_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Construction question answer cannot be empty")
        return normalized


class ConstructionGapEntry(ContractModel):
    gap_key: str = ""
    title: str = ""
    domain: str = ""
    severity: ConstructionGapSeverity = "warning"
    status: ConstructionGapStatus = "open"
    blocking_stage: str = ""
    summary: str = ""
    remediation: str = ""
    evidence_paths: list[str] = PydanticField(default_factory=list)
    source_sections: list[str] = PydanticField(default_factory=list)
    current_assumptions: list[str] = PydanticField(default_factory=list)
    closure_criteria: list[str] = PydanticField(default_factory=list)
    questions: list[ConstructionQuestionEntry] = PydanticField(default_factory=list)

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, value: str) -> str:
        if value not in ALLOWED_CONSTRUCTION_GAP_SEVERITIES:
            raise ValueError(f"Ungoverned construction gap severity: {value}")
        return value

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        if value not in ALLOWED_CONSTRUCTION_GAP_STATUSES:
            raise ValueError(f"Ungoverned construction gap status: {value}")
        return value


class ConstructionReadinessReport(ContractModel):
    overall_status: ConstructionReadinessStatus = "not_started"
    can_start_build: bool = False
    blocking_gaps: int = 0
    open_questions: int = 0
    assumptions_count: int = 0
    gaps: list[ConstructionGapEntry] = PydanticField(default_factory=list)
    next_recommended_action: str = ""

    @field_validator("overall_status")
    @classmethod
    def validate_overall_status(cls, value: str) -> str:
        if value not in ALLOWED_CONSTRUCTION_READINESS_STATUSES:
            raise ValueError(f"Ungoverned construction readiness status: {value}")
        return value

    @field_validator("blocking_gaps", "open_questions", "assumptions_count")
    @classmethod
    def validate_non_negative_counts(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Construction readiness counts must be non-negative")
        return value


class ACPPreview(ContractModel):
    package_version: str = "acp.v1"
    session_id: UUID
    blueprint_version_number: int | None = None
    manifest_path: str = "ACP/manifest.yaml"
    files: list[ACPFileEntry] = PydanticField(default_factory=list)
    validation: ACPValidationReport = PydanticField(default_factory=ACPValidationReport)
    construction_readiness: ConstructionReadinessReport = PydanticField(default_factory=ConstructionReadinessReport)

    @field_validator("files")
    @classmethod
    def validate_files(cls, value: list[ACPFileEntry]) -> list[ACPFileEntry]:
        seen: set[str] = set()
        for item in value:
            if item.path in seen:
                raise ValueError(f"Duplicated ACP file path: {item.path}")
            seen.add(item.path)
        return value


class WorkstreamEstimate(ContractModel):
    workstream_key: str = ""
    label: str = ""
    estimated_hours: float = 0
    estimated_cost: float = 0
    duration_days: float = 0
    automation_percent: int = 0
    notes: list[str] = PydanticField(default_factory=list)

    @field_validator("estimated_hours", "estimated_cost", "duration_days")
    @classmethod
    def validate_non_negative_metric(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Workstream estimate metrics must be non-negative")
        return value

    @field_validator("automation_percent")
    @classmethod
    def validate_automation_percent(cls, value: int) -> int:
        if not 0 <= value <= 100:
            raise ValueError("Automation percent must be between 0 and 100")
        return value


class EstimateScenarioBase(ContractModel):
    scenario_type: EstimationScenarioType
    estimated_hours_total: float = 0
    estimated_duration_weeks: float = 0
    estimated_cost: float = 0
    team_shape: list[str] = PydanticField(default_factory=list)
    workstream_breakdown: list[WorkstreamEstimate] = PydanticField(default_factory=list)
    assumptions: list[str] = PydanticField(default_factory=list)
    warnings: list[str] = PydanticField(default_factory=list)

    @field_validator("estimated_hours_total", "estimated_duration_weeks", "estimated_cost")
    @classmethod
    def validate_non_negative_totals(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Estimate totals must be non-negative")
        return value


class TraditionalEstimate(EstimateScenarioBase):
    scenario_type: EstimationScenarioType = EstimationScenarioType.traditional


class EstimationPricingSnapshot(ContractModel):
    provider: LLMProviderKey = LLMProviderKey.openai
    profile_key: str = ""
    label: str = ""
    model: str = ""
    pricing_mode: str = ""
    effective_from: str = ""
    is_local_inference: bool = False
    local_cost_policy: CodexLocalCostPolicy | None = None
    cop_per_usd: float = 0
    assumptions: list[str] = PydanticField(default_factory=list)
    rates: list[LLMPricingRateEntry] = PydanticField(default_factory=list)

    @field_validator("cop_per_usd")
    @classmethod
    def validate_snapshot_cop_per_usd(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Pricing snapshot FX assumptions must be non-negative")
        return value


class AutomationFamilyAssessment(ContractModel):
    family_key: str = ""
    label: str = ""
    complexity: EstimationComplexityLevel = EstimationComplexityLevel.simple
    coverage_percent: int = 0
    risk_tier: str = ""
    mandatory_human_review: bool = False
    blocking_conditions: list[str] = PydanticField(default_factory=list)
    penalties_applied: list[str] = PydanticField(default_factory=list)
    bonuses_applied: list[str] = PydanticField(default_factory=list)
    non_automatable_reasons: list[str] = PydanticField(default_factory=list)
    notes: list[str] = PydanticField(default_factory=list)

    @field_validator("coverage_percent")
    @classmethod
    def validate_coverage_percent(cls, value: int) -> int:
        if not 0 <= value <= 100:
            raise ValueError("Automation family coverage percent must be between 0 and 100")
        return value


class AgenticEstimate(EstimateScenarioBase):
    scenario_type: EstimationScenarioType = EstimationScenarioType.agentic
    active_provider: LLMProviderKey = LLMProviderKey.openai
    pricing_policy: str = ""
    provider_model: str = ""
    economic_model: str = ""
    blueprint_design_coverage_percent: int = 0
    acp_package_readiness_percent: int = 0
    implementation_scope_coverage_percent: int = 0
    automation_coverage_percent: int = 0
    human_supervision_hours: float = 0
    human_delivery_cost: float = 0
    human_supervision_cost: float = 0
    llm_runtime_cost_usd: float = 0
    tool_runtime_cost_usd: float = 0
    platform_overhead_cost_usd: float = 0
    provider_runtime_cost_total_usd: float = 0
    provider_runtime_cost_total_cop: float = 0
    tooling_cost_usd: float = 0
    platform_cost_usd: float = 0
    net_savings_vs_traditional: float = 0
    automation_coverage_by_workstream: dict[str, int] = PydanticField(default_factory=dict)
    automation_coverage_by_artifact_family: dict[str, int] = PydanticField(default_factory=dict)
    automation_assessments: list[AutomationFamilyAssessment] = PydanticField(default_factory=list)
    pricing_assumptions: list[str] = PydanticField(default_factory=list)
    pricing_snapshot: EstimationPricingSnapshot | None = None

    @field_validator(
        "blueprint_design_coverage_percent",
        "acp_package_readiness_percent",
        "implementation_scope_coverage_percent",
        "automation_coverage_percent",
    )
    @classmethod
    def validate_agentic_percent(cls, value: int) -> int:
        if not 0 <= value <= 100:
            raise ValueError("Automation coverage percent must be between 0 and 100")
        return value

    @field_validator(
        "human_supervision_hours",
        "human_delivery_cost",
        "human_supervision_cost",
        "llm_runtime_cost_usd",
        "tool_runtime_cost_usd",
        "platform_overhead_cost_usd",
        "provider_runtime_cost_total_usd",
        "provider_runtime_cost_total_cop",
        "tooling_cost_usd",
        "platform_cost_usd",
    )
    @classmethod
    def validate_non_negative_costs(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Agentic cost components must be non-negative")
        return value


class ConfidenceBreakdown(ContractModel):
    score: int = 0
    label: EstimationConfidenceLabel = EstimationConfidenceLabel.low
    uncertainty_band_percent: int = 0
    blocking_gaps: int = 0
    open_questions: int = 0
    design_gap_count: int = 0
    implementation_gap_count: int = 0
    design_open_questions: int = 0
    implementation_open_questions: int = 0
    assumptions_count: int = 0
    subscores: dict[str, int] = PydanticField(default_factory=dict)
    positive_signals: list[str] = PydanticField(default_factory=list)
    negative_signals: list[str] = PydanticField(default_factory=list)
    recommended_next_actions: list[str] = PydanticField(default_factory=list)

    @field_validator("score", "uncertainty_band_percent")
    @classmethod
    def validate_score_range(cls, value: int) -> int:
        if not 0 <= value <= 100:
            raise ValueError("Confidence values must be between 0 and 100")
        return value

    @field_validator(
        "blocking_gaps",
        "open_questions",
        "design_gap_count",
        "implementation_gap_count",
        "design_open_questions",
        "implementation_open_questions",
        "assumptions_count",
    )
    @classmethod
    def validate_non_negative_counts(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Confidence counts must be non-negative")
        return value


class EstimationComplexityDriver(ContractModel):
    driver_key: str = ""
    title: str = ""
    workstream_key: str = ""
    impact_level: Literal["low", "medium", "high"] = "medium"
    summary: str = ""
    evidence_refs: list[str] = PydanticField(default_factory=list)


class EstimationRiskRegisterEntry(ContractModel):
    risk_key: str = ""
    title: str = ""
    severity: Literal["low", "medium", "high"] = "medium"
    likelihood: Literal["low", "medium", "high"] = "medium"
    impact: str = ""
    mitigation: str = ""
    evidence_refs: list[str] = PydanticField(default_factory=list)


class EstimationUncertaintyFactor(ContractModel):
    factor_key: str = ""
    title: str = ""
    category: str = ""
    impact_area: Literal["scope", "schedule", "cost", "confidence", "operations"] = "confidence"
    summary: str = ""
    evidence_refs: list[str] = PydanticField(default_factory=list)


class EstimationBenchmarkRef(ContractModel):
    benchmark_key: str = ""
    title: str = ""
    source_kind: Literal["workspace_actuals", "pricing_catalog", "knowledge_document", "platform_benchmark"] = "knowledge_document"
    source_ref: str = ""
    sample_size: int = 0
    captured_at: str = ""
    freshness: str = ""
    summary: str = ""
    workspace_scoped: bool = False

    @field_validator("sample_size")
    @classmethod
    def validate_sample_size(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Benchmark sample size must be non-negative")
        return value


class EstimationScenarioAdjustment(ContractModel):
    scenario_key: Literal["optimistic", "base", "conservative"] = "base"
    hours_multiplier: float = 1.0
    duration_multiplier: float = 1.0
    cost_multiplier: float = 1.0
    rationale: str = ""
    evidence_refs: list[str] = PydanticField(default_factory=list)

    @field_validator("hours_multiplier", "duration_multiplier", "cost_multiplier")
    @classmethod
    def validate_scenario_multiplier(cls, value: float) -> float:
        if value < 0.75 or value > 1.35:
            raise ValueError("Scenario multipliers must remain between 0.75 and 1.35")
        return value


class EstimationConstructionScenario(ContractModel):
    scenario_key: Literal[
        "traditional_blueprint",
        "blueprint_basic",
        "blueprint_premium",
        "agentic_blueprint",
        "acp_manual",
        "acp_agentic",
        "done_for_you_factory",
    ] = "traditional_blueprint"
    label: str = ""
    description: str = ""
    estimated_hours_total: float = 0
    estimated_duration_weeks: float = 0
    estimated_cost: float = 0
    human_intervention_percent: int = 0
    automation_leverage_percent: int = 0
    effort_reduction_vs_traditional_percent: int = 0
    cost_savings_vs_traditional: float = 0
    notes: list[str] = PydanticField(default_factory=list)

    @field_validator("estimated_hours_total", "estimated_duration_weeks", "estimated_cost", "cost_savings_vs_traditional")
    @classmethod
    def validate_non_negative_scenario_metrics(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Construction scenario metrics must be non-negative")
        return value

    @field_validator("human_intervention_percent", "automation_leverage_percent", "effort_reduction_vs_traditional_percent")
    @classmethod
    def validate_construction_scenario_percent(cls, value: int) -> int:
        if not 0 <= value <= 100:
            raise ValueError("Construction scenario percent values must be between 0 and 100")
        return value


class EstimationSavingsOpportunity(ContractModel):
    opportunity_key: str = ""
    title: str = ""
    summary: str = ""
    expected_impact: str = ""
    prerequisites: list[str] = PydanticField(default_factory=list)
    evidence_refs: list[str] = PydanticField(default_factory=list)


class EstimationQuestion(ContractModel):
    question_key: str = ""
    question: str = ""
    rationale: str = ""
    blocking: bool = False


class EstimationConfidenceAdjustmentProposal(ContractModel):
    proposed_score_delta: int = 0
    proposed_uncertainty_band_delta: int = 0
    rationale: str = ""
    evidence_refs: list[str] = PydanticField(default_factory=list)

    @field_validator("proposed_score_delta")
    @classmethod
    def validate_score_delta(cls, value: int) -> int:
        if value < -20 or value > 10:
            raise ValueError("Confidence score delta must remain between -20 and 10")
        return value

    @field_validator("proposed_uncertainty_band_delta")
    @classmethod
    def validate_band_delta(cls, value: int) -> int:
        if value < -10 or value > 20:
            raise ValueError("Confidence band delta must remain between -10 and 20")
        return value


class EstimationAnalysisArtifact(ContractModel):
    schema_version: str = "estimation-analysis.v1"
    summary: str = ""
    complexity_drivers: list[EstimationComplexityDriver] = PydanticField(default_factory=list)
    risk_register: list[EstimationRiskRegisterEntry] = PydanticField(default_factory=list)
    uncertainty_factors: list[EstimationUncertaintyFactor] = PydanticField(default_factory=list)
    benchmark_refs: list[EstimationBenchmarkRef] = PydanticField(default_factory=list)
    scenario_adjustments: list[EstimationScenarioAdjustment] = PydanticField(default_factory=list)
    savings_opportunities: list[EstimationSavingsOpportunity] = PydanticField(default_factory=list)
    assumptions: list[str] = PydanticField(default_factory=list)
    questions: list[EstimationQuestion] = PydanticField(default_factory=list)
    evidence_refs: list[str] = PydanticField(default_factory=list)
    confidence_adjustment_proposal: EstimationConfidenceAdjustmentProposal = PydanticField(
        default_factory=EstimationConfidenceAdjustmentProposal
    )


class EstimationAnalysisDecision(ContractModel):
    decision: Literal["pending", "accepted", "rejected"] = "pending"
    note: str = ""
    decided_at: datetime | None = None


class EstimationDeterministicInputs(ContractModel):
    pricing_catalog_signature: str = ""
    validation_fingerprint: str = ""
    benchmark_corpus_hash: str = ""
    catalogs_used: list[str] = PydanticField(default_factory=list)
    benchmark_ids: list[str] = PydanticField(default_factory=list)
    formula_notes: list[str] = PydanticField(default_factory=list)
    calibration_sample_size: int = 0

    @field_validator("calibration_sample_size")
    @classmethod
    def validate_calibration_sample_size(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Calibration sample size must be non-negative")
        return value


class EstimationPackagePolicyState(ContractModel):
    preliminary: bool = False
    can_continue_to_package: bool = False
    package_block_reasons: list[str] = PydanticField(default_factory=list)
    commercial_blocked: bool = False


class EstimationReportArtifact(ContractModel):
    contract_version: str = "estimation-report.v1"
    maturity_stage: EstimationMaturityStage = EstimationMaturityStage.canvas
    blueprint_version_number: int | None = None
    current_blueprint_version_number: int | None = None
    generated_at: datetime | None = None
    is_stale: bool = False
    stale_reasons: list[str] = PydanticField(default_factory=list)
    source_artifacts: list[str] = PydanticField(default_factory=list)
    assumptions: list[str] = PydanticField(default_factory=list)
    risk_drivers: list[str] = PydanticField(default_factory=list)
    traditional: TraditionalEstimate = PydanticField(default_factory=TraditionalEstimate)
    agentic: AgenticEstimate = PydanticField(default_factory=AgenticEstimate)
    confidence: ConfidenceBreakdown = PydanticField(default_factory=ConfidenceBreakdown)
    base_confidence: ConfidenceBreakdown | None = None
    analysis: EstimationAnalysisArtifact | None = None
    analysis_decision: EstimationAnalysisDecision = PydanticField(default_factory=EstimationAnalysisDecision)
    construction_scenarios: list[EstimationConstructionScenario] = PydanticField(default_factory=list)
    deterministic_inputs: EstimationDeterministicInputs = PydanticField(default_factory=EstimationDeterministicInputs)
    package_policy: EstimationPackagePolicyState = PydanticField(default_factory=EstimationPackagePolicyState)
    notes: list[str] = PydanticField(default_factory=list)


class EstimationRunEntry(ContractModel):
    id: UUID
    blueprint_version_number: int | None = None
    source_action: str = ""
    maturity_stage: EstimationMaturityStage = EstimationMaturityStage.canvas
    active_provider: LLMProviderKey = LLMProviderKey.openai
    pricing_policy: str = ""
    confidence_score: int = 0
    confidence_label: EstimationConfidenceLabel = EstimationConfidenceLabel.low
    uncertainty_band_percent: int = 0
    traditional_hours_total: float = 0
    traditional_duration_weeks: float = 0
    traditional_cost_total: float = 0
    agentic_hours_total: float = 0
    agentic_duration_weeks: float = 0
    agentic_cost_total: float = 0
    automation_coverage_percent: int = 0
    created_at: datetime


class ProjectActualsEntry(ContractModel):
    id: UUID
    estimation_run_id: UUID
    delivery_mode: EstimationScenarioType = EstimationScenarioType.agentic
    actual_provider: LLMProviderKey | None = None
    actual_hours_total: float = 0
    actual_duration_weeks: float = 0
    actual_cost_total: float = 0
    actual_automation_coverage_percent: int = 0
    notes: str = ""
    created_at: datetime
    updated_at: datetime


class EstimationErrorMetricEntry(ContractModel):
    id: UUID
    estimation_run_id: UUID
    maturity_stage: EstimationMaturityStage = EstimationMaturityStage.canvas
    scenario_type: EstimationScenarioType = EstimationScenarioType.agentic
    active_provider: LLMProviderKey | None = None
    absolute_percentage_error_hours: float = 0
    absolute_percentage_error_duration: float = 0
    absolute_percentage_error_cost: float = 0
    absolute_percentage_error_automation: float = 0
    bias_hours_percent: float = 0
    bias_duration_percent: float = 0
    bias_cost_percent: float = 0
    bias_automation_percent: float = 0
    band_hit_hours: bool = False
    band_hit_duration: bool = False
    band_hit_cost: bool = False
    band_hit_overall: bool = False
    created_at: datetime
    updated_at: datetime


class EstimationCalibrationStageSummary(ContractModel):
    maturity_stage: EstimationMaturityStage = EstimationMaturityStage.canvas
    total_runs: int = 0
    calibrated_runs: int = 0
    mean_absolute_percentage_error_hours: float = 0
    mean_absolute_percentage_error_duration: float = 0
    mean_absolute_percentage_error_cost: float = 0
    mean_absolute_percentage_error_automation: float = 0
    mean_bias_hours_percent: float = 0
    mean_bias_duration_percent: float = 0
    mean_bias_cost_percent: float = 0
    mean_bias_automation_percent: float = 0
    band_hit_rate: float = 0


class EstimationRecentCalibrationEntry(ContractModel):
    session_id: UUID
    session_title: str = ""
    estimation_run_id: UUID
    maturity_stage: EstimationMaturityStage = EstimationMaturityStage.canvas
    scenario_type: EstimationScenarioType = EstimationScenarioType.agentic
    provider: LLMProviderKey | None = None
    estimated_cost_total: float = 0
    actual_cost_total: float = 0
    cost_absolute_percentage_error: float = 0
    band_hit_overall: bool = False
    updated_at: datetime


class EstimationCalibrationDashboard(ContractModel):
    generated_at: datetime | None = None
    total_runs: int = 0
    calibrated_runs: int = 0
    coverage_percent: float = 0
    mean_absolute_percentage_error_hours: float = 0
    mean_absolute_percentage_error_duration: float = 0
    mean_absolute_percentage_error_cost: float = 0
    mean_absolute_percentage_error_automation: float = 0
    mean_bias_cost_percent: float = 0
    band_hit_rate: float = 0
    precision_by_stage: list[EstimationCalibrationStageSummary] = PydanticField(default_factory=list)
    recent_projects: list[EstimationRecentCalibrationEntry] = PydanticField(default_factory=list)


class EstimationActualsUpsertRequest(ContractModel):
    estimation_run_id: UUID
    delivery_mode: EstimationScenarioType = EstimationScenarioType.agentic
    actual_provider: LLMProviderKey | None = None
    actual_hours_total: float = 0
    actual_duration_weeks: float = 0
    actual_cost_total: float = 0
    actual_automation_coverage_percent: int = 0
    notes: str = ""

    @field_validator(
        "actual_hours_total",
        "actual_duration_weeks",
        "actual_cost_total",
    )
    @classmethod
    def validate_non_negative_actual_values(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Actual metrics must be greater than zero")
        return value

    @field_validator("actual_automation_coverage_percent")
    @classmethod
    def validate_actual_automation_percent(cls, value: int) -> int:
        if not 0 <= value <= 100:
            raise ValueError("Actual automation coverage percent must be between 0 and 100")
        return value

    @field_validator("notes")
    @classmethod
    def normalize_notes(cls, value: str) -> str:
        return value.strip()


class EstimationAnalysisDecisionRequest(ContractModel):
    decision: Literal["accepted", "rejected"] = "accepted"
    note: str = ""

    @field_validator("note")
    @classmethod
    def normalize_decision_note(cls, value: str) -> str:
        return value.strip()


class RoleRateCatalogEntry(ContractModel):
    role_key: str = ""
    label: str = ""
    seniority: str = ""
    currency: str = "USD"
    hourly_rate: float = 0
    source_note: str = ""

    @field_validator("hourly_rate")
    @classmethod
    def validate_hourly_rate(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Hourly rate must be non-negative")
        return value


class WorkstreamEffortBand(ContractModel):
    complexity: EstimationComplexityLevel = EstimationComplexityLevel.moderate
    relative_weight: float = 1
    base_hours_min: float = 0
    base_hours_max: float = 0

    @field_validator("relative_weight", "base_hours_min", "base_hours_max")
    @classmethod
    def validate_non_negative_effort(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Workstream effort values must be non-negative")
        return value


class WorkstreamEffortProfile(ContractModel):
    workstream_key: str = ""
    label: str = ""
    default_role_keys: list[str] = PydanticField(default_factory=list)
    bands: list[WorkstreamEffortBand] = PydanticField(default_factory=list)
    notes: list[str] = PydanticField(default_factory=list)

    @field_validator("bands")
    @classmethod
    def validate_unique_effort_bands(cls, value: list[WorkstreamEffortBand]) -> list[WorkstreamEffortBand]:
        seen: set[EstimationComplexityLevel] = set()
        for item in value:
            if item.complexity in seen:
                raise ValueError(f"Duplicated workstream effort complexity: {item.complexity}")
            if item.base_hours_max < item.base_hours_min:
                raise ValueError("Workstream base_hours_max must be greater than or equal to base_hours_min")
            seen.add(item.complexity)
        return value


class AutomationBandProfile(ContractModel):
    complexity: EstimationComplexityLevel = EstimationComplexityLevel.simple
    base_automation_percent: int = 0
    automation_ceiling_percent: int = 0
    mandatory_human_review: bool = False
    risk_tier: str = ""

    @field_validator("base_automation_percent", "automation_ceiling_percent")
    @classmethod
    def validate_automation_band_percent(cls, value: int) -> int:
        if not 0 <= value <= 100:
            raise ValueError("Automation matrix percentages must be between 0 and 100")
        return value


class AutomationAdjustmentRule(ContractModel):
    rule_key: str = ""
    label: str = ""
    delta_percent: int = 0
    rationale: str = ""

    @field_validator("delta_percent")
    @classmethod
    def validate_delta_percent(cls, value: int) -> int:
        if not 0 <= value <= 100:
            raise ValueError("Automation adjustment percentages must be between 0 and 100")
        return value


class AutomationMatrixProfile(ContractModel):
    family_key: str = ""
    label: str = ""
    bands: list[AutomationBandProfile] = PydanticField(default_factory=list)
    blocking_conditions: list[str] = PydanticField(default_factory=list)
    penalty_rules: list[AutomationAdjustmentRule] = PydanticField(default_factory=list)
    bonus_rules: list[AutomationAdjustmentRule] = PydanticField(default_factory=list)
    notes: list[str] = PydanticField(default_factory=list)

    @field_validator("bands")
    @classmethod
    def validate_unique_automation_bands(cls, value: list[AutomationBandProfile]) -> list[AutomationBandProfile]:
        seen: set[EstimationComplexityLevel] = set()
        for item in value:
            if item.complexity in seen:
                raise ValueError(f"Duplicated automation matrix complexity: {item.complexity}")
            if item.automation_ceiling_percent < item.base_automation_percent:
                raise ValueError(
                    "Automation ceiling percent must be greater than or equal to base automation percent"
                )
            seen.add(item.complexity)
        return value


class LLMPricingRateEntry(ContractModel):
    metric_key: str = ""
    label: str = ""
    unit: str = ""
    amount_usd: float = 0

    @field_validator("amount_usd")
    @classmethod
    def validate_amount_usd(cls, value: float) -> float:
        if value < 0:
            raise ValueError("LLM pricing amounts must be non-negative")
        return value


class LLMPricingProfile(ContractModel):
    profile_key: str = ""
    label: str = ""
    provider: LLMProviderKey = LLMProviderKey.openai
    model: str = ""
    mode: str = ""
    is_local_inference: bool = False
    local_cost_policy: CodexLocalCostPolicy | None = None
    effective_from: str = ""
    cop_per_usd: float = 0
    rates: list[LLMPricingRateEntry] = PydanticField(default_factory=list)
    notes: list[str] = PydanticField(default_factory=list)

    @field_validator("cop_per_usd")
    @classmethod
    def validate_cop_per_usd(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Pricing FX assumptions must be non-negative")
        return value


class BlueprintGraphNodeEntry(ContractModel):
    id: str
    label: str = ""
    type: str = ""
    description: str = ""
    source_artifacts: list[str] = PydanticField(default_factory=list)
    properties: dict[str, Any] = PydanticField(default_factory=dict)


class BlueprintGraphEdgeEntry(ContractModel):
    source: str
    target: str
    type: str = ""
    description: str = ""
    source_artifacts: list[str] = PydanticField(default_factory=list)


class BlueprintKnowledgeGraph(ContractModel):
    graph_version: str = "blueprint-graph.v1"
    generated_from_session_id: str = ""
    generated_at: str = ""
    nodes: list[BlueprintGraphNodeEntry] = PydanticField(default_factory=list)
    edges: list[BlueprintGraphEdgeEntry] = PydanticField(default_factory=list)


class OperationEnvelopeBase(ContractModel):
    status: ArtifactStatus
    stage: SessionStage
    missing_fields: list[str] = PydanticField(default_factory=list)
    assumptions: list[str] = PydanticField(default_factory=list)
    warnings: list[str] = PydanticField(default_factory=list)
    evidence: list[EvidenceItem] = PydanticField(default_factory=list)
    llm_trace: LLMContextTrace | None = None
    next_action: str


class DiscoveryEnvelope(OperationEnvelopeBase):
    data: DiscoveryArtifact


class CanvasEnvelope(OperationEnvelopeBase):
    data: CanvasArtifact


class BlueprintEnvelope(OperationEnvelopeBase):
    data: BlueprintArtifact


class EvaluationEnvelope(OperationEnvelopeBase):
    data: EvaluationArtifact


class EstimationEnvelope(OperationEnvelopeBase):
    data: EstimationReportArtifact


class ToolRecommendationEnvelope(OperationEnvelopeBase):
    data: ToolRecommendationArtifact


class BlueprintPatchRequest(ContractModel):
    architecture: str | None = None
    reasoning_pattern: str | None = None
    memory_strategy: str | None = None
    memory_profile: MemoryProfile | None = None
    knowledge_profile: KnowledgeProfile | None = None
    tools: list[BlueprintTool] | None = None
    llm_policy: BlueprintLLMPolicy | None = None
    safety_checks: list[SafetyCheck] | None = None
    guardrails: list[str] | None = None
    delivery_package: DeliveryPackage | None = None
    readiness_state: ReviewState | None = None
    narrative: str | None = None


class ApproveToolsSelectionRequest(ContractModel):
    include_optional_tool_keys: list[str] = PydanticField(default_factory=list)


class ToolRecommendationRequest(ContractModel):
    instructions: str = ""


class MemoryRecommendationRequest(ContractModel):
    instructions: str = ""


class SessionOwnerSummary(ContractModel):
    id: UUID
    name: str = ""


class SessionCapabilities(ContractModel):
    can_open: bool = True
    can_rename: bool = False
    can_archive: bool = False
    can_restore: bool = False
    can_delete: bool = False


class SessionCreateResponse(ContractModel):
    id: UUID
    workspace_id: UUID | None = None
    title: str
    suggested_title: str | None = None
    title_source: ProjectTitleSource = ProjectTitleSource.generated
    row_version: int = 1
    status: ArtifactStatus
    current_stage: SessionStage
    commercial_tier: CommercialTier = CommercialTier.blueprint
    owner: SessionOwnerSummary | None = None
    pending_attention_count: int = 0
    progress_percent: int = 0
    archived_at: datetime | None = None
    deleted_at: datetime | None = None
    capabilities: SessionCapabilities = PydanticField(default_factory=SessionCapabilities)
    created_at: datetime
    updated_at: datetime


class SessionListPageInfo(ContractModel):
    next_cursor: str | None = None
    total: int = 0


class SessionListFacets(ContractModel):
    active: int = 0
    needs_review: int = 0
    archived: int = 0
    trash: int = 0


class SessionListResponse(ContractModel):
    items: list[SessionCreateResponse] = PydanticField(default_factory=list)
    page: SessionListPageInfo = PydanticField(default_factory=SessionListPageInfo)
    facets: SessionListFacets = PydanticField(default_factory=SessionListFacets)


class SessionRenameRequest(ContractModel):
    title: str
    expected_version: int | None = None

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        normalized = " ".join(value.strip().split())
        if len(normalized) < 3:
            raise ValueError("El nombre del proyecto debe tener al menos 3 caracteres.")
        if len(normalized) > 100:
            raise ValueError("El nombre del proyecto no puede superar 100 caracteres.")
        return normalized


class SessionDeleteRequest(ContractModel):
    confirm_title: str


class WorkspaceMembershipSummary(ContractModel):
    workspace_id: UUID
    workspace_name: str = ""
    workspace_slug: str = ""
    role: WorkspaceRole = WorkspaceRole.viewer
    is_active: bool = True


class AuthUser(ContractModel):
    id: UUID
    email: str
    full_name: str
    active_workspace_id: UUID | None = None
    active_workspace_name: str = ""
    workspaces: list[WorkspaceMembershipSummary] = PydanticField(default_factory=list)


class LoginRequest(ContractModel):
    email: str
    password: str


class LoginResponse(ContractModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime
    user: AuthUser


class WorkspaceSelectionRequest(ContractModel):
    workspace_id: UUID


class ExecutionLogEntry(ContractModel):
    stage: SessionStage
    status: ArtifactStatus
    message: str
    payload: dict[str, Any] = PydanticField(default_factory=dict)
    created_at: datetime


class BlueprintVersionEntry(ContractModel):
    version_number: int
    source_action: str
    status: ArtifactStatus
    readiness_state: ReviewState
    architecture: str
    reasoning_pattern: str
    created_at: datetime


class ApprovalGateEntry(ContractModel):
    id: UUID
    gate_key: str
    title: str
    rationale: str = ""
    instructions: str = ""
    requested_in_stage: SessionStage
    status: ApprovalStatus
    resolution_note: str = ""
    created_at: datetime
    resolved_at: datetime | None = None


class ApprovalResolutionRequest(ContractModel):
    decision: ApprovalStatus
    resolution_note: str = ""


class JourneyStageArtifactCreateRequest(ContractModel):
    artifact_kind: str = ""
    source_action: str = "manual_draft"
    proposal_payload: dict[str, Any] = PydanticField(default_factory=dict)
    user_patch: dict[str, Any] = PydanticField(default_factory=dict)
    source_stage_versions: dict[str, Any] = PydanticField(default_factory=dict)
    input_fingerprint: str = ""
    context_fingerprint: str = ""
    output_fingerprint: str = ""
    corpus_hash: str = ""
    provider_key: str = ""
    model: str = ""
    execution_backend: str = ""
    prompt_version: str = ""
    schema_version: str = ""
    confidence: float | None = None
    missing_information: list[str] = PydanticField(default_factory=list)
    warnings: list[str] = PydanticField(default_factory=list)
    evidence_manifest: list[JourneyArtifactEvidenceEntry] = PydanticField(default_factory=list)
    note: str = ""


class JourneyStageArtifactPatchRequest(ContractModel):
    artifact_kind: str | None = None
    proposal_payload: dict[str, Any] | None = None
    user_patch: dict[str, Any] | None = None
    source_stage_versions: dict[str, Any] | None = None
    input_fingerprint: str | None = None
    context_fingerprint: str | None = None
    output_fingerprint: str | None = None
    corpus_hash: str | None = None
    provider_key: str | None = None
    model: str | None = None
    execution_backend: str | None = None
    prompt_version: str | None = None
    schema_version: str | None = None
    confidence: float | None = None
    missing_information: list[str] | None = None
    warnings: list[str] | None = None
    evidence_manifest: list[JourneyArtifactEvidenceEntry] | None = None
    note: str = ""


class JourneyStageArtifactApprovalRequest(ContractModel):
    note: str = ""
    decision_payload: dict[str, Any] = PydanticField(default_factory=dict)


class JourneyStageArtifactRejectionRequest(ContractModel):
    note: str = ""
    decision_payload: dict[str, Any] = PydanticField(default_factory=dict)


class DesignProposalRequest(ContractModel):
    instructions: str = ""


class EvaluationDatasetUpdateRequest(ContractModel):
    cases: list[EvaluationDatasetCase] = PydanticField(default_factory=list)


class EvaluationRubricUpdateRequest(ContractModel):
    summary: str | None = None
    dimensions: list[EvaluationRubricDimension] = PydanticField(default_factory=list)


class WorkflowTemplateApplyRequest(ContractModel):
    template_key: str


class HandoffResolutionRequest(ContractModel):
    decision: str
    resolution_note: str = ""


class FeatureFlagUpdateRequest(ContractModel):
    enabled: bool


class CommercialTierUpdateRequest(ContractModel):
    tier: CommercialTier


class TRMResponse(ContractModel):
    contract_version: str = "trm.v1"
    unit_usd: float = 1.0
    trm_cop: float = 3171.93
    date: str = ""
    source: str = "Datos Abiertos Colombia / Superfinanciera"
    updated_at: datetime = PydanticField(default_factory=utc_now)


class BasePricesUpdateRequest(ContractModel):
    blueprint_pro_usd: float
    acp_premium_usd: float


class BasePricesResponse(ContractModel):
    contract_version: str = "base-prices.v1"
    blueprint_free_usd: float = 0.0
    blueprint_pro_usd: float = 49.0
    acp_premium_usd: float = 149.0
    trm_cop: float = 3171.93
    updated_at: datetime = PydanticField(default_factory=utc_now)


class ProductPriceResponse(ContractModel):
    price_code: str = ""
    currency: str = "USD"
    unit_amount_cents: int = 0
    unit_amount_usd_cents: int = 0
    unit_amount_usd: float = 0.0
    unit_amount_cop_calculated: float = 0.0
    trm_applied: float = 3171.93
    billing_period: str = "one_time"
    version: int = 1


class ProductCatalogResponse(ContractModel):
    product_key: str = ""
    tier: CommercialTier = CommercialTier.blueprint_pro
    product_type: CommercialProductType = CommercialProductType.blueprint
    name: str = ""
    description: str = ""
    scope: str = "project"
    benefits: list[str] = PydanticField(default_factory=list)
    exclusions: list[str] = PydanticField(default_factory=list)
    capabilities: list[str] = PydanticField(default_factory=list)
    price: ProductPriceResponse | None = None
    version: int = 1


class CommercialEntitlementSummary(ContractModel):
    id: UUID
    product_key: str = ""
    tier: CommercialTier = CommercialTier.blueprint_pro
    status: CommercialEntitlementStatus = CommercialEntitlementStatus.active
    source: CommercialEntitlementSource = CommercialEntitlementSource.checkout
    scope: str = "project"
    starts_at: datetime
    ends_at: datetime | None = None
    purchase_ref: str = ""
    non_revenue: bool = False


class CommercialCapabilityDecisionEntry(ContractModel):
    capability: str = ""
    allowed: bool = False
    current_tier: CommercialTier = CommercialTier.blueprint
    required_tier: CommercialTier = CommercialTier.blueprint
    product: str = ""
    label: str = ""
    reason_code: str = "allowed"
    cta_label: str = ""


class CommercialAccessSnapshotV2(ContractModel):
    contract_version: str = "commercial-access.v2"
    workspace_id: UUID | None = None
    session_id: UUID | None = None
    user_id: UUID | None = None
    role: WorkspaceRole | None = None
    tier: CommercialTier = CommercialTier.blueprint
    tier_label: str = "Blueprint"
    reason_code: str = "free_access"
    checkout_state: str = "not_started"
    purchase_refs: list[str] = PydanticField(default_factory=list)
    entitlements: list[CommercialEntitlementSummary] = PydanticField(default_factory=list)
    capabilities: list[CommercialCapabilityDecisionEntry] = PydanticField(default_factory=list)


class CommercialCheckoutSessionRequest(ContractModel):
    session_id: UUID
    product_key: str
    price_code: str = ""
    provider: Literal["sandbox", "hotmart"] | None = None
    success_url: str = ""
    cancel_url: str = ""
    idempotency_key: str = ""


class CommercialCheckoutSessionResponse(ContractModel):
    contract_version: str = "commerce-checkout-session.v1"
    checkout_ref: str
    order_id: UUID
    session_id: UUID
    workspace_id: UUID
    product_key: str
    provider: str = "sandbox"
    status: CommercialOrderStatus = CommercialOrderStatus.pending
    checkout_url: str = ""
    total_cents: int = 0
    currency: str = "COP"
    expires_at: datetime | None = None
    entitlement: CommercialEntitlementSummary | None = None
    next_action: str = "open_checkout"


class CommercialCheckoutCompletionRequest(ContractModel):
    outcome: Literal["success", "failure", "cancel"] = "success"
    provider_payment_id: str = ""


class CommercialOrderLineResponse(ContractModel):
    product_key: str = ""
    price_code: str = ""
    quantity: int = 1
    total_amount_cents: int = 0


class CommercialOrderResponse(ContractModel):
    id: UUID
    workspace_id: UUID
    session_id: UUID | None = None
    buyer_user_id: UUID
    status: CommercialOrderStatus = CommercialOrderStatus.pending
    provider: str = "sandbox"
    checkout_ref: str = ""
    checkout_url: str = ""
    currency: str = "COP"
    total_cents: int = 0
    lines: list[CommercialOrderLineResponse] = PydanticField(default_factory=list)
    entitlement: CommercialEntitlementSummary | None = None
    created_at: datetime
    updated_at: datetime


class HotmartCredentialUpsertRequest(ContractModel):
    environment: Literal["sandbox", "production"] = "sandbox"
    enabled: bool = True
    client_id: str = ""
    client_secret: str = ""
    basic_token: str = ""
    hottok: str = ""
    api_base_url: str = ""
    auth_base_url: str = ""
    webhook_public_url: str = ""

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        candidate = value.strip().lower()
        if candidate not in {"sandbox", "production"}:
            raise ValueError("environment must be sandbox or production.")
        return candidate


class HotmartIntegrationStatusResponse(ContractModel):
    contract_version: str = "hotmart-integration-status.v1"
    workspace_id: UUID
    environment: Literal["sandbox", "production"] = "sandbox"
    enabled: bool = False
    status: str = "not_configured"
    client_id_configured: bool = False
    client_secret_configured: bool = False
    basic_token_configured: bool = False
    hottok_configured: bool = False
    api_base_url: str = ""
    auth_base_url: str = ""
    webhook_public_url: str = ""
    last_health_check_at: datetime | None = None
    last_health_status: str = ""
    last_health_message: str = ""
    last_sync_at: datetime | None = None
    storage_mode: str = "none"
    updated_at: datetime | None = None


class HotmartTestConnectionResponse(ContractModel):
    contract_version: str = "hotmart-test-connection.v1"
    workspace_id: UUID
    environment: Literal["sandbox", "production"] = "sandbox"
    reachable: bool = False
    status: str = "not_configured"
    message: str = ""
    token_expires_in: int | None = None
    http_status: int | None = None
    rate_limit_remaining: int | None = None
    checked_at: datetime


class HotmartProductMappingUpsertRequest(ContractModel):
    environment: Literal["sandbox", "production"] = "sandbox"
    internal_product_key: str
    hotmart_product_id: str = ""
    hotmart_product_ucode: str = ""
    offer_code: str = ""
    plan_code: str = ""
    billing_mode: str = "one_time"
    currency: str = "USD"
    hotmart_price_strategy: str = "net_order_amount"
    trm_policy: str = "internal_usd"
    grants_tier: CommercialTier = CommercialTier.blueprint_pro
    entitlement_scope: str = "project"
    is_active: bool = True
    metadata: dict[str, Any] = PydanticField(default_factory=dict)


class HotmartProductMappingResponse(ContractModel):
    contract_version: str = "hotmart-product-mapping.v1"
    id: UUID
    workspace_id: UUID
    environment: Literal["sandbox", "production"] = "sandbox"
    internal_product_key: str
    hotmart_product_id: str = ""
    hotmart_product_ucode: str = ""
    offer_code: str = ""
    plan_code: str = ""
    billing_mode: str = "one_time"
    currency: str = "USD"
    internal_unit_amount_usd_cents: int = 0
    hotmart_price_strategy: str = "net_order_amount"
    trm_policy: str = "internal_usd"
    grants_tier: CommercialTier = CommercialTier.blueprint_pro
    entitlement_scope: str = "project"
    is_active: bool = True
    updated_at: datetime


class HotmartPaymentLinkCreateRequest(ContractModel):
    order_id: UUID | None = None
    checkout_ref: str = ""
    environment: Literal["sandbox", "production"] = "sandbox"
    link_name: str = ""
    callback_url: str = ""
    force_new: bool = False


class HotmartPaymentLinkResponse(ContractModel):
    contract_version: str = "hotmart-payment-link.v1"
    id: UUID
    workspace_id: UUID
    order_id: UUID
    internal_product_key: str = ""
    hotmart_payment_link_id: str = ""
    provider_ref: str = ""
    checkout_url: str = ""
    activation_status: str = "pending_activation"
    gross_amount_cents: int = 0
    discount_amount_cents: int = 0
    net_amount_cents: int = 0
    currency: str = "USD"
    discount_origin: str = "none"
    created_at: datetime
    updated_at: datetime


class HotmartPromotionCreateRequest(ContractModel):
    environment: Literal["sandbox", "production"] = "sandbox"
    internal_campaign_key: str = ""
    internal_product_key: str
    coupon_code: str
    discount_percent: float
    offer_codes: list[str] = PydanticField(default_factory=list)
    starts_at: datetime | None = None
    ends_at: datetime | None = None
    affiliate_id: str = ""
    publish: bool = True
    metadata: dict[str, Any] = PydanticField(default_factory=dict)


class HotmartPromotionResponse(ContractModel):
    contract_version: str = "hotmart-promotion.v1"
    id: UUID
    workspace_id: UUID
    environment: Literal["sandbox", "production"] = "sandbox"
    internal_campaign_key: str = ""
    internal_product_key: str = ""
    hotmart_product_id: str = ""
    offer_codes: list[str] = PydanticField(default_factory=list)
    coupon_id: str = ""
    coupon_code: str = ""
    discount_percent: float = 0.0
    discount_origin: str = "provider_coupon"
    discount_type: str = "percent"
    discount_amount_cents: int | None = None
    starts_at: datetime | None = None
    ends_at: datetime | None = None
    status: str = "draft"
    published_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class HotmartPromotionDeleteResponse(ContractModel):
    contract_version: str = "hotmart-promotion-delete.v1"
    id: UUID
    coupon_id: str = ""
    coupon_code: str = ""
    status: str = "deleted"
    deleted_remote: bool = False
    message: str = ""


class HotmartPromotionMetricsResponse(ContractModel):
    contract_version: str = "hotmart-promotion-metrics.v1"
    total: int = 0
    active: int = 0
    scheduled: int = 0
    expired: int = 0
    deleted: int = 0
    sync_error: int = 0
    provider_coupon_count: int = 0
    internal_upgrade_credit_count: int = 0


class HotmartSyncRequest(ContractModel):
    environment: Literal["sandbox", "production"] = "sandbox"
    resource: Literal["products", "offers", "plans", "sales", "subscriptions", "coupons", "payment_links"]
    force_reset: bool = False
    max_results: int = 50
    page_token: str = ""
    product_id: str = ""
    filters: dict[str, Any] = PydanticField(default_factory=dict)


class HotmartSyncRunResponse(ContractModel):
    contract_version: str = "hotmart-sync-run.v1"
    id: UUID
    workspace_id: UUID
    environment: Literal["sandbox", "production"] = "sandbox"
    resource: str = ""
    status: str = "idle"
    started_by_user_id: UUID | None = None
    started_at: datetime
    finished_at: datetime | None = None
    cursor_before: str = ""
    cursor_after: str = ""
    records_read: int = 0
    records_created: int = 0
    records_updated: int = 0
    records_skipped: int = 0
    error_summary: str = ""
    issue_count: int = 0


class HotmartSyncCursorResponse(ContractModel):
    contract_version: str = "hotmart-sync-cursor.v1"
    id: UUID
    workspace_id: UUID
    environment: Literal["sandbox", "production"] = "sandbox"
    resource: str = ""
    page_token: str = ""
    last_event_at: datetime | None = None
    last_transaction: str = ""
    last_success_at: datetime | None = None
    updated_at: datetime


class HotmartReconciliationIssueResponse(ContractModel):
    contract_version: str = "hotmart-reconciliation-issue.v1"
    id: UUID
    workspace_id: UUID
    environment: Literal["sandbox", "production"] = "sandbox"
    issue_type: str = ""
    severity: str = "medium"
    status: str = "open"
    provider_ref: str = ""
    internal_ref: str = ""
    summary: str = ""
    suggested_action: str = ""
    resolution_action: str = ""
    resolution_note: str = ""
    resolved_by_user_id: UUID | None = None
    resolved_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class HotmartReconciliationResolveRequest(ContractModel):
    resolution_action: str
    resolution_note: str = ""
    status: Literal["resolved", "ignored", "needs_review"] = "resolved"


class HotmartWebhookReplayResponse(ContractModel):
    contract_version: str = "hotmart-webhook-replay.v1"
    event_id: str
    processing_status: str = ""
    retries: int = 0
    issue_id: UUID | None = None
    message: str = ""


class HotmartClubSyncRequest(ContractModel):
    environment: Literal["sandbox", "production"] = "sandbox"
    subdomain: str
    sync_modules: bool = True
    sync_pages: bool = True
    sync_students: bool = True
    sync_progress: bool = False
    module_id: str = ""
    user_id: str = ""
    is_extra: bool | None = None


class HotmartClubOverviewResponse(ContractModel):
    contract_version: str = "hotmart-club-overview.v1"
    workspace_id: UUID
    environment: Literal["sandbox", "production"] = "sandbox"
    subdomain: str = ""
    modules_count: int = 0
    pages_count: int = 0
    students_count: int = 0
    progress_count: int = 0
    open_issue_count: int = 0
    last_sync_status: str = "idle"
    last_sync_at: datetime | None = None


class HotmartClubModuleResponse(ContractModel):
    contract_version: str = "hotmart-club-module.v1"
    module_id: str = ""
    name: str = ""
    sequence: int = 0
    is_public: bool = False
    is_extra: bool = False
    is_extra_paid: bool = False
    total_pages: int = 0


class HotmartClubPageResponse(ContractModel):
    contract_version: str = "hotmart-club-page.v1"
    page_id: str = ""
    module_id: str = ""
    name: str = ""
    page_order: int = 0
    type: str = ""


class HotmartClubStudentResponse(ContractModel):
    contract_version: str = "hotmart-club-student.v1"
    user_id: str = ""
    name: str = ""
    email: str = ""
    status: str = ""
    engagement: str = ""
    progress: dict[str, Any] = PydanticField(default_factory=dict)


class HotmartClubProgressResponse(ContractModel):
    contract_version: str = "hotmart-club-progress.v1"
    user_id: str = ""
    email: str = ""
    page_id: str = ""
    page_name: str = ""
    completed: bool = False
    completed_at: datetime | None = None
    progress_payload: dict[str, Any] = PydanticField(default_factory=dict)


class HotmartReleaseChecklistItemResponse(ContractModel):
    contract_version: str = "hotmart-release-checklist-item.v1"
    key: str = ""
    label: str = ""
    status: Literal["failed", "manual", "passed", "warning"] = "manual"
    severity: Literal["critical", "high", "medium", "low"] = "medium"
    required: bool = True
    detail: str = ""
    evidence: list[str] = PydanticField(default_factory=list)


class HotmartOperationalAlertResponse(ContractModel):
    contract_version: str = "hotmart-operational-alert.v1"
    key: str = ""
    severity: Literal["critical", "high", "medium", "low"] = "medium"
    status: Literal["active", "resolved"] = "active"
    title: str = ""
    message: str = ""
    evidence: list[str] = PydanticField(default_factory=list)
    created_at: datetime = PydanticField(default_factory=utc_now)


class HotmartRunbookSectionResponse(ContractModel):
    contract_version: str = "hotmart-runbook-section.v1"
    key: str = ""
    title: str = ""
    steps: list[str] = PydanticField(default_factory=list)
    links: list[str] = PydanticField(default_factory=list)


class HotmartReleaseReadinessResponse(ContractModel):
    contract_version: str = "hotmart-release-readiness.v1"
    workspace_id: UUID
    environment: Literal["sandbox", "production"] = "sandbox"
    generated_at: datetime = PydanticField(default_factory=utc_now)
    overall_status: Literal["blocked", "needs_attention", "ready"] = "needs_attention"
    release_candidate: bool = False
    metrics: dict[str, int] = PydanticField(default_factory=dict)
    checklist: list[HotmartReleaseChecklistItemResponse] = PydanticField(default_factory=list)
    alerts: list[HotmartOperationalAlertResponse] = PydanticField(default_factory=list)
    runbook: list[HotmartRunbookSectionResponse] = PydanticField(default_factory=list)


class HotmartWebhookIngestResponse(ContractModel):
    contract_version: str = "hotmart-webhook-ingest.v1"
    event_id: str
    event_type: str = ""
    transaction: str = ""
    processing_status: str = "received"
    duplicate: bool = False
    workspace_id: UUID | None = None
    order_id: UUID | None = None
    payment_id: UUID | None = None
    entitlement_id: UUID | None = None
    message: str = ""


class AccessRequestCreateRequest(ContractModel):
    session_id: UUID
    capability: str
    reason: str = ""


class AccessRequestResolveRequest(ContractModel):
    decision: Literal["approved", "rejected", "canceled"]
    resolution_note: str = ""


class AccessRequestResponse(ContractModel):
    id: UUID
    workspace_id: UUID
    session_id: UUID
    requester_user_id: UUID
    capability: str = ""
    product_key: str = ""
    target_tier: CommercialTier = CommercialTier.blueprint_pro
    status: CommercialAccessRequestStatus = CommercialAccessRequestStatus.pending
    reason: str = ""
    resolution_note: str = ""
    created_at: datetime
    updated_at: datetime
    resolved_at: datetime | None = None
    project_title: str = ""
    workspace_name: str = ""
    requester_name: str = ""
    requester_email: str = ""


class ProductOverviewItem(ContractModel):
    key: str = ""
    label: str = ""
    status: str = "available"
    access_state: str = "allowed"
    cta_label: str = ""
    href: str = ""
    progress_percent: int = 0
    detail: str = ""


class ProductAttentionItem(ContractModel):
    key: str = ""
    title: str = ""
    severity: Literal["info", "warning", "blocking"] = "info"
    stage: str = ""
    reason: str = ""
    href: str = ""


class ProductOverviewResponse(ContractModel):
    contract_version: str = "product-overview.v1"
    session_id: UUID
    workspace_id: UUID
    project_title: str = ""
    active_stage: str = ""
    lean_progress_percent: int = 0
    access: CommercialAccessSnapshotV2
    products: list[ProductOverviewItem] = PydanticField(default_factory=list)
    attention: list[ProductAttentionItem] = PydanticField(default_factory=list)
    exports: list[ProductOverviewItem] = PydanticField(default_factory=list)
    navigation: list[ProductOverviewItem] = PydanticField(default_factory=list)
    canonical_overview_contract: str = ""
    recommended_next_action: dict[str, Any] = PydanticField(default_factory=dict)
    source_contracts: list[str] = PydanticField(default_factory=list)
    generated_at: datetime = PydanticField(default_factory=utc_now)


class BlueprintResultResponse(ContractModel):
    contract_version: str = "blueprint-result.v1"
    session_id: UUID
    workspace_id: UUID
    title: str = ""
    version_number: int | None = None
    state: str = "not_generated"
    stale: bool = False
    access: CommercialAccessSnapshotV2
    summary: str = ""
    architecture_sample: str = ""
    sections: list[dict[str, Any]] = PydanticField(default_factory=list)
    diagrams: list[DiagramCatalogEntry] = PydanticField(default_factory=list)
    estimation: dict[str, Any] = PydanticField(default_factory=dict)
    protection: dict[str, Any] = PydanticField(default_factory=dict)
    generated_at: datetime = PydanticField(default_factory=utc_now)


class ProductOfferResponse(ContractModel):
    contract_version: str = "product-offer.v1"
    session_id: UUID
    workspace_id: UUID
    product: ProductCatalogResponse
    access: CommercialAccessSnapshotV2
    can_checkout: bool = False
    checkout_disabled_reason: str = ""
    comparison: dict[str, Any] = PydanticField(default_factory=dict)
    generated_at: datetime = PydanticField(default_factory=utc_now)


class AcpInvitationResponse(ContractModel):
    contract_version: str = "acp-invitation.v1"
    session_id: UUID
    workspace_id: UUID
    access: CommercialAccessSnapshotV2
    state: str = "purchase_required"
    metrics: dict[str, Any] = PydanticField(default_factory=dict)
    comparison: dict[str, Any] = PydanticField(default_factory=dict)
    benefits: list[str] = PydanticField(default_factory=list)
    next_action: str = "checkout"
    generated_at: datetime = PydanticField(default_factory=utc_now)


class SessionCommercialAccess(ContractModel):
    tier: CommercialTier = CommercialTier.blueprint
    tier_label: str = "Blueprint"
    tier_rank: int = 1
    contract_version: str = "commercial-access.v1-compatible"
    reason_code: str = "free_access"
    checkout_state: str = "not_started"
    purchase_refs: list[str] = PydanticField(default_factory=list)
    capability_reasons: dict[str, str] = PydanticField(default_factory=dict)
    upgrade_cta_label: str = "Upgrade a Blueprint Profesional"
    upgrade_message: str = ""
    can_view_in_app_blueprint: bool = True
    can_view_blueprint: bool = True
    can_download_blueprint: bool = False
    can_export_blueprint_document: bool = False
    can_export_markdown: bool = False
    can_export_json: bool = False
    can_export_blueprint_core: bool = False
    can_export_estimation_pack: bool = False
    can_export_construction_pack: bool = False
    can_export_prompt_pack: bool = False
    can_export_test_pack: bool = False
    can_export_acp_zip: bool = False
    can_invite_acp: bool = True
    can_build_acp: bool = False
    can_download_acp: bool = False
    can_view_diagram_sample: bool = True
    can_view_diagram_blueprint: bool = False
    can_view_diagram_acp: bool = False
    can_access_library_workspace: bool = False
    available_upgrades: list[CommercialTier] = PydanticField(default_factory=list)


class CommercialEventRequest(ContractModel):
    event_key: str
    product: str = ""
    source: str = ""
    metadata: dict[str, Any] = PydanticField(default_factory=dict)


class CommercialAuditMetric(ContractModel):
    key: str
    label: str
    value: int | float | str
    unit: str = ""
    tone: Literal["blue", "green", "orange", "red", "slate", "violet"] = "slate"
    detail: str = ""


class CommercialAuditFunnelStep(ContractModel):
    key: str
    label: str
    product: str = ""
    event_keys: list[str] = PydanticField(default_factory=list)
    count: int = 0
    completed: bool = False
    conversion_percent: int = 0
    latest_at: datetime | None = None


class CommercialAuditProductSummary(ContractModel):
    product: str
    views: int = 0
    blocked_events: int = 0
    cta_clicks: int = 0
    purchases: int = 0
    exports: int = 0


class CommercialAuditEventEntry(ContractModel):
    event_key: str
    product: str = ""
    source: str = ""
    stage: SessionStage
    status: ArtifactStatus
    message: str
    metadata: dict[str, Any] = PydanticField(default_factory=dict)
    created_at: datetime


class CommercialAuditReport(ContractModel):
    contract_version: str = "commercial-audit.v1"
    session_id: UUID
    workspace_id: UUID | None = None
    requested_by_user_id: UUID | None = None
    current_tier: CommercialTier = CommercialTier.blueprint
    generated_at: datetime = PydanticField(default_factory=utc_now)
    redaction_policy: str = (
        "Secrets, credentials, tokens, raw prompts, diagram content and export payloads are redacted; "
        "only ids, event keys, counts, hashes and short safe metadata are exposed."
    )
    metrics: list[CommercialAuditMetric] = PydanticField(default_factory=list)
    funnel: list[CommercialAuditFunnelStep] = PydanticField(default_factory=list)
    product_summary: list[CommercialAuditProductSummary] = PydanticField(default_factory=list)
    recent_events: list[CommercialAuditEventEntry] = PydanticField(default_factory=list)
    warnings: list[str] = PydanticField(default_factory=list)


class ACPPhaseDefinitionResponse(ContractModel):
    key: str = ""
    label: str = ""
    objective: str = ""
    order: int = 0
    required: bool = True


class ACPPhaseRunResponse(ContractModel):
    id: UUID | None = None
    phase_key: str = ""
    phase_label: str = ""
    phase_order: int = 0
    status: ACPWorkflowRunStatus = ACPWorkflowRunStatus.not_started
    attempt_count: int = 0
    input_refs: list[dict[str, Any]] = PydanticField(default_factory=list)
    output_refs: list[dict[str, Any]] = PydanticField(default_factory=list)
    checkpoints: dict[str, Any] = PydanticField(default_factory=dict)
    blockers: list[dict[str, Any]] = PydanticField(default_factory=list)
    warnings: list[str] = PydanticField(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime | None = None


class ACPBuildRunResponse(ContractModel):
    id: UUID
    workspace_id: UUID
    session_id: UUID
    blueprint_version_number: int | None = None
    status: ACPWorkflowRunStatus = ACPWorkflowRunStatus.not_started
    current_phase_key: str = ""
    progress_percent: int = 0
    phase_order: list[str] = PydanticField(default_factory=list)
    checkpoints: dict[str, Any] = PydanticField(default_factory=dict)
    artifacts: dict[str, Any] = PydanticField(default_factory=dict)
    blockers: list[dict[str, Any]] = PydanticField(default_factory=list)
    warnings: list[str] = PydanticField(default_factory=list)
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None


class ACPWorkspaceResponse(ContractModel):
    contract_version: str = "acp-workspace.v1"
    session_id: UUID
    workspace_id: UUID
    access: CommercialAccessSnapshotV2
    run: ACPBuildRunResponse
    phases: list[ACPPhaseRunResponse] = PydanticField(default_factory=list)
    phase_definitions: list[ACPPhaseDefinitionResponse] = PydanticField(default_factory=list)
    readiness: ConstructionReadinessReport = PydanticField(default_factory=ConstructionReadinessReport)
    validation: ACPValidationReport = PydanticField(default_factory=ACPValidationReport)
    next_action: str = ""
    generated_at: datetime = PydanticField(default_factory=utc_now)


class ACPPhaseCommandRequest(ContractModel):
    idempotency_key: str = ""
    force: bool = False


class AttentionItemResponse(ContractModel):
    key: str = ""
    title: str = ""
    type: Literal["question", "gap", "approval", "checkout", "entitlement", "warning", "info"] = "info"
    severity: Literal["info", "warning", "blocking"] = "info"
    stage: str = ""
    source: str = ""
    reason: str = ""
    impact: str = ""
    status: str = "open"
    owner_role: str = ""
    action_label: str = ""
    href: str = ""
    detected_at: datetime | None = None
    metadata: dict[str, Any] = PydanticField(default_factory=dict)


class AttentionResponse(ContractModel):
    contract_version: str = "attention.v1"
    session_id: UUID
    workspace_id: UUID
    total_count: int = 0
    blocking_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    items: list[AttentionItemResponse] = PydanticField(default_factory=list)
    generated_at: datetime = PydanticField(default_factory=utc_now)


class AttentionSourceRefV2(ContractModel):
    artifact_id: str | None = None
    artifact_version: int | None = None
    entity_id: str | None = None
    field_path: str | None = None


class AttentionOptionV2(ContractModel):
    key: str
    label: str
    description: str = ""
    impact: str = ""
    example: str = ""
    recommended: bool = False
    confidence: float = 0.0
    source_refs: list[str] = PydanticField(default_factory=list)


class AttentionDiagnosticsV2(ContractModel):
    summary: str = ""
    technical_message: str = ""
    error_kind: str = ""
    capability: str = ""
    capability_label: str = ""
    operation_id: str = ""
    retry_policy: str = ""
    repair_hint: str = ""
    trace_refs: list[str] = PydanticField(default_factory=list)


class AttentionActionV2(ContractModel):
    kind: Literal["navigate", "answer", "approve", "reject", "confirm", "regenerate", "retry"]
    label: str
    href: str
    return_href: str
    can_resolve_inline: bool = False


class AttentionItemV2(ContractModel):
    key: str
    type: Literal[
        "question",
        "gap",
        "decision",
        "approval",
        "confirmation",
        "validation",
        "hitl",
        "inconsistency",
        "stale",
        "runtime_error",
        "access_request",
    ]
    severity: Literal["info", "warning", "blocking"]
    blocking: bool
    product: Literal["blueprint", "acp", "commercial"]
    stage: str
    source: str
    source_ref: AttentionSourceRefV2 = PydanticField(default_factory=AttentionSourceRefV2)
    title: str
    reason: str
    impact: str
    consequence_if_unresolved: str
    status: Literal["open", "in_progress", "deferred", "resolved", "dismissed", "superseded"] = "open"
    owner_role: str = ""
    owner_user_id: str = ""
    options: list[AttentionOptionV2] = PydanticField(default_factory=list)
    suggested_answer: str = ""
    unblocks: str = ""
    resume_action: str = ""
    action: AttentionActionV2
    affected_artifact_refs: list[str] = PydanticField(default_factory=list)
    diagnostics: AttentionDiagnosticsV2 | None = None
    detected_at: datetime | None = None
    updated_at: datetime | None = None

    @field_validator("key", "stage", "source", "title", "reason", "consequence_if_unresolved")
    @classmethod
    def validate_attention_v2_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Attention v2 required text fields cannot be empty")
        return normalized

    @model_validator(mode="after")
    def validate_attention_v2_blocking_alignment(self) -> "AttentionItemV2":
        if self.severity == "blocking" and not self.blocking:
            raise ValueError("Blocking severity must set blocking=true")
        if self.blocking and self.severity != "blocking":
            raise ValueError("blocking=true requires severity=blocking")
        return self


class AttentionResponseV2(ContractModel):
    contract_version: Literal["attention.v2"] = "attention.v2"
    session_id: UUID
    workspace_id: UUID
    current_stage: str = ""
    total_count: int = 0
    actionable_count: int = 0
    blocking_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    counts_by_stage: dict[str, int] = PydanticField(default_factory=dict)
    counts_by_type: dict[str, int] = PydanticField(default_factory=dict)
    counts_by_product: dict[str, int] = PydanticField(default_factory=dict)
    primary_item: AttentionItemV2 | None = None
    items: list[AttentionItemV2] = PydanticField(default_factory=list)
    cursor: str = ""
    generated_at: datetime = PydanticField(default_factory=utc_now)


class AttentionActionRequestV2(ContractModel):
    action_kind: Literal["navigate", "answer", "approve", "reject", "confirm", "regenerate", "retry", "defer"]
    idempotency_key: str = ""
    answer_text: str = ""
    selected_option_key: str = ""
    was_suggested_answer_used: bool = False
    decision: str = ""
    resolution_note: str = ""
    source_artifact_version: int | None = None
    payload: dict[str, Any] = PydanticField(default_factory=dict)


class AttentionActionResultV2(ContractModel):
    contract_version: Literal["attention-action.v2"] = "attention-action.v2"
    session_id: UUID
    workspace_id: UUID
    item_key: str
    action_kind: str
    status: Literal["applied", "duplicate", "unsupported", "not_found", "conflict", "forbidden"]
    message: str = ""
    attention: AttentionResponseV2


class ExportCatalogItemResponse(ContractModel):
    key: str = ""
    label: str = ""
    description: str = ""
    product_key: str = ""
    required_capability: str = ""
    profile: str = ""
    content_type: str = "application/json"
    file_extension: str = "json"
    access_state: Literal["allowed", "locked", "blocked"] = "locked"
    locked_reason: str = ""
    cta_label: str = ""


class ExportCatalogResponse(ContractModel):
    contract_version: str = "export-catalog.v1"
    session_id: UUID
    workspace_id: UUID
    items: list[ExportCatalogItemResponse] = PydanticField(default_factory=list)
    generated_at: datetime = PydanticField(default_factory=utc_now)


class ExportJobCreateRequest(ContractModel):
    artifact_kind: str
    idempotency_key: str = ""
    profile: str = ""


class ExportJobResponse(ContractModel):
    id: UUID
    workspace_id: UUID
    session_id: UUID
    product_key: str = ""
    artifact_kind: str = ""
    profile: str = ""
    status: ExportJobStatus = ExportJobStatus.queued
    content_type: str = "application/json"
    file_name: str = ""
    checksum_sha256: str = ""
    size_bytes: int = 0
    download_url: str = ""
    expires_at: datetime | None = None
    error_message: str = ""
    metadata: dict[str, Any] = PydanticField(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None


class LauncherScriptResponse(ContractModel):
    platform: str = ""
    path: str = ""
    command: str = ""
    available: bool = False


class LauncherMetadataResponse(ContractModel):
    contract_version: str = "acp-launcher-metadata.v1"
    session_id: UUID
    workspace_id: UUID
    manifest_path: str = "ACP/launcher/launch-manifest.json"
    launcher_version: str = ""
    package_name: str = ""
    requires_lean_backend: bool = False
    report_output: str = ""
    scripts: list[LauncherScriptResponse] = PydanticField(default_factory=list)
    restrictions: list[str] = PydanticField(default_factory=list)
    safe_defaults: dict[str, Any] = PydanticField(default_factory=dict)
    generated_at: datetime = PydanticField(default_factory=utc_now)


class LauncherReportSubmitRequest(ContractModel):
    report_path: str = "ACP/launcher/launch-report.json"
    launcher_version: str = ""
    detected_tool: str = ""
    detected_ide: str = ""
    status: str = "received"
    summary: str = ""
    report: dict[str, Any] = PydanticField(default_factory=dict)


class LauncherReportResponse(ContractModel):
    contract_version: str = "acp-launch-report.v1"
    id: UUID
    workspace_id: UUID
    session_id: UUID
    report_path: str = ""
    launcher_version: str = ""
    detected_tool: str = ""
    detected_ide: str = ""
    status: str = "received"
    summary: str = ""
    created_at: datetime


class ActivityTimelineEntry(ContractModel):
    key: str = ""
    type: Literal["commercial", "execution", "export", "workflow"] = "commercial"
    title: str = ""
    product_key: str = ""
    source: str = ""
    status: str = ""
    revenue_cents: int = 0
    currency: str = ""
    created_at: datetime
    metadata: dict[str, Any] = PydanticField(default_factory=dict)


class ActivityResponse(ContractModel):
    contract_version: str = "activity.v1"
    session_id: UUID
    workspace_id: UUID
    metrics: list[CommercialAuditMetric] = PydanticField(default_factory=list)
    funnel: list[CommercialAuditFunnelStep] = PydanticField(default_factory=list)
    timeline: list[ActivityTimelineEntry] = PydanticField(default_factory=list)
    generated_at: datetime = PydanticField(default_factory=utc_now)


class PlanAccessResponse(ContractModel):
    contract_version: str = "plan-access.v1"
    session_id: UUID
    workspace_id: UUID
    access: CommercialAccessSnapshotV2
    products: list[ProductCatalogResponse] = PydanticField(default_factory=list)
    pending_requests: list[AccessRequestResponse] = PydanticField(default_factory=list)
    entitlements: list[CommercialEntitlementSummary] = PydanticField(default_factory=list)
    generated_at: datetime = PydanticField(default_factory=utc_now)


DiagramAccessState = Literal[
    "unlocked",
    "sample",
    "locked_blueprint",
    "locked_acp",
    "stage_locked",
    "not_generated",
]
DiagramGenerationState = Literal["generated", "planned", "pending_generation", "not_generated"]


class DiagramContentProtection(ContractModel):
    disable_copy: bool = True
    disable_context_menu: bool = True
    disable_download: bool = True
    watermark_sample: bool = True


class DiagramUpsellMessage(ContractModel):
    title: str = ""
    message: str = ""
    cta_label: str = ""
    target_tier: CommercialTier = CommercialTier.blueprint_pro
    product: str = "blueprint"


class DiagramAccessPolicy(ContractModel):
    diagram_key: str = ""
    title: str = ""
    category: str = ""
    description: str = ""
    enabled_from_stage: str = ""
    product_scope: list[str] = PydanticField(default_factory=list)
    required_tier: CommercialTier = CommercialTier.blueprint
    access_level: str = "view_only"
    diagram_surface: str = ""
    sample_enabled: bool = False
    sample_tier: CommercialTier = CommercialTier.blueprint
    visible_to_user_types: list[str] = PydanticField(default_factory=list)
    requires_purchase: bool = False
    default_generation_state: DiagramGenerationState = "pending_generation"
    content_protection: DiagramContentProtection = PydanticField(default_factory=DiagramContentProtection)
    upsell: dict[str, str] = PydanticField(default_factory=dict)
    preferred_format: str = ""
    available_formats: list[str] = PydanticField(default_factory=list)
    source_artifact_keys: list[str] = PydanticField(default_factory=list)
    portable_paths: list[str] = PydanticField(default_factory=list)
    sort_order: int = 0
    is_active: bool = True


class DiagramCatalogEntry(ContractModel):
    diagram_key: str = ""
    title: str = ""
    category: str = ""
    summary: str = ""
    diagram_surface: str = ""
    product_scope: list[str] = PydanticField(default_factory=list)
    required_tier: CommercialTier = CommercialTier.blueprint
    enabled_from_stage: str = ""
    generation_state: DiagramGenerationState = "pending_generation"
    access_state: DiagramAccessState = "locked_blueprint"
    locked_reason: str = ""
    upgrade_cta_label: str = ""
    preferred_format: str = ""
    available_formats: list[str] = PydanticField(default_factory=list)
    available_content_formats: list[str] = PydanticField(default_factory=list)
    preview_thumbnail: str | None = None
    source_artifact_count: int = 0
    source_paths: list[str] = PydanticField(default_factory=list)
    protection: DiagramContentProtection = PydanticField(default_factory=DiagramContentProtection)
    upsell: DiagramUpsellMessage | None = None


class DiagramCatalogResponse(ContractModel):
    session_id: UUID
    workspace_id: UUID
    current_stage: str = ""
    tier: CommercialTier = CommercialTier.blueprint
    total_count: int = 0
    unlocked_count: int = 0
    locked_count: int = 0
    sample_count: int = 0
    pending_count: int = 0
    entries: list[DiagramCatalogEntry] = PydanticField(default_factory=list)


class DiagramCatalogV2Response(DiagramCatalogResponse):
    contract_version: str = "diagram-catalog.v2"
    limit: int = 20
    next_cursor: str | None = None
    has_more: bool = False


class DiagramContentResponse(ContractModel):
    diagram_key: str = ""
    access_state: DiagramAccessState = "locked_blueprint"
    generation_state: DiagramGenerationState = "pending_generation"
    format: str = ""
    content: str | None = None
    asset_url: str | None = None
    protection: DiagramContentProtection = PydanticField(default_factory=DiagramContentProtection)
    upsell: DiagramUpsellMessage | None = None
    metadata: dict[str, Any] = PydanticField(default_factory=dict)


class ShortTermMemoryStateRef(ContractModel):
    key: str
    kind: str
    stage: str = ""
    source: str = ""
    summary: str = ""
    status: str = ""
    created_at: str = ""
    blueprint_version_number: int | None = None
    evidence_paths: list[str] = PydanticField(default_factory=list)


class ShortTermMemoryStateNamespace(ContractModel):
    namespace: str
    summary: str = ""
    ref_keys: list[str] = PydanticField(default_factory=list)
    freshness: str = ""
    read_roles: list[str] = PydanticField(default_factory=list)
    write_roles: list[str] = PydanticField(default_factory=list)


class ShortTermMemoryStateCompaction(ContractModel):
    summary_policy: str = ""
    invalidation_policy: str = ""
    eviction_policy: str = ""
    last_compacted_at: str = ""


class ShortTermMemoryState(ContractModel):
    schema_version: str = "short-term-memory.state.v1"
    active_stage: str = ""
    active_goal: str = ""
    current_focus: str = ""
    pending_approvals: list[str] = PydanticField(default_factory=list)
    open_handoffs: list[str] = PydanticField(default_factory=list)
    recent_decisions: list[str] = PydanticField(default_factory=list)
    namespaces: list[ShortTermMemoryStateNamespace] = PydanticField(default_factory=list)
    checkpoint_refs: list[ShortTermMemoryStateRef] = PydanticField(default_factory=list)
    artifact_refs: list[ShortTermMemoryStateRef] = PydanticField(default_factory=list)
    skill_run_refs: list[ShortTermMemoryStateRef] = PydanticField(default_factory=list)
    branch_refs: list[ShortTermMemoryStateRef] = PydanticField(default_factory=list)
    compaction: ShortTermMemoryStateCompaction = PydanticField(default_factory=ShortTermMemoryStateCompaction)


class ShortTermBranchBoardEntry(ContractModel):
    branch_key: str
    parent_branch_key: str = ""
    title: str = ""
    topology: str = ""
    stage: str = ""
    status: str = ""
    isolation_mode: str = ""
    summary: str = ""
    namespace_keys: list[str] = PydanticField(default_factory=list)
    checkpoint_count: int = 0
    active_checkpoint_key: str = ""
    last_consistent_checkpoint_key: str = ""
    last_activity_at: datetime | None = None


class ShortTermCheckpointEntry(ContractModel):
    checkpoint_key: str
    branch_key: str
    checkpoint_number: int = 1
    parent_checkpoint_key: str = ""
    stage: str = ""
    source_action: str = ""
    status: str = ""
    summary: str = ""
    state_hash: str = ""
    is_consistent: bool = True
    is_active: bool = False
    rollback_note: str = ""
    created_at: datetime
    updated_at: datetime


class ShortTermMemoryRuntimeState(ContractModel):
    contract_version: str = "short-term-memory-runtime.v1"
    session_id: UUID
    source_action: str = ""
    active_branch_key: str = "main"
    active_checkpoint_key: str = ""
    last_consistent_checkpoint_key: str = ""
    resume_supported: bool = True
    rollback_available: bool = False
    branch_count: int = 0
    checkpoint_count: int = 0
    memory: ShortTermMemoryState = PydanticField(default_factory=ShortTermMemoryState)
    branch_board: list[ShortTermBranchBoardEntry] = PydanticField(default_factory=list)
    checkpoint_history: list[ShortTermCheckpointEntry] = PydanticField(default_factory=list)
    updated_at: datetime | None = None


class ShortTermMemoryRollbackRequest(ContractModel):
    checkpoint_key: str | None = None
    branch_key: str | None = None
    reason: str = ""


class SessionSnapshot(ContractModel):
    contract_version: str = "session-snapshot.v1"
    session: SessionCreateResponse
    commercial_access: SessionCommercialAccess = PydanticField(default_factory=SessionCommercialAccess)
    discovery: DiscoveryArtifact | None = None
    canvas: CanvasArtifact | None = None
    blueprint: BlueprintArtifact | None = None
    latest_tool_recommendation: ToolRecommendationArtifact | None = None
    evaluation: EvaluationArtifact | None = None
    estimation_report: EstimationReportArtifact | None = None
    estimation_runs: list[EstimationRunEntry] = PydanticField(default_factory=list)
    project_actuals: list[ProjectActualsEntry] = PydanticField(default_factory=list)
    estimation_error_metrics: list[EstimationErrorMetricEntry] = PydanticField(default_factory=list)
    evaluation_dataset: EvaluationDatasetArtifact | None = None
    evaluation_rubric: EvaluationRubricArtifact | None = None
    evaluation_runs: list[EvaluationRunEntry] = PydanticField(default_factory=list)
    simulation_runs: list[SimulationRunRecord] = PydanticField(default_factory=list)
    validations: list[dict[str, Any]] = PydanticField(default_factory=list)
    activity: list[ExecutionLogEntry] = PydanticField(default_factory=list)
    blueprint_versions: list[BlueprintVersionEntry] = PydanticField(default_factory=list)
    selected_workflow_template_key: str = ""
    approvals: list[ApprovalGateEntry] = PydanticField(default_factory=list)
    journey_artifacts: list[JourneyStageArtifactEntry] = PydanticField(default_factory=list)
    journey_latest_artifacts: dict[str, JourneyStageArtifactEntry] = PydanticField(default_factory=dict)
    artifact_records: list[ArtifactRecordEntry] = PydanticField(default_factory=list)
    metric_snapshots: list[MetricSnapshotEntry] = PydanticField(default_factory=list)
    alert_events: list[AlertEventEntry] = PydanticField(default_factory=list)
    integration_statuses: list[IntegrationStatusEntry] = PydanticField(default_factory=list)
    workflow_templates: list[WorkflowTemplateEntry] = PydanticField(default_factory=list)
    handoff_records: list[HandoffRecordEntry] = PydanticField(default_factory=list)
    governance_policies: list[GovernancePolicyEntry] = PydanticField(default_factory=list)
    subagent_runs: list[SubagentRunEntry] = PydanticField(default_factory=list)
    workspace_contract: WorkspaceContract = PydanticField(default_factory=WorkspaceContract)
    skill_catalog: list[SkillDefinition] = PydanticField(default_factory=list)
    skill_runs: list[SkillRunEntry] = PydanticField(default_factory=list)
    short_term_memory: ShortTermMemoryRuntimeState | None = None
    blueprint_consistency: BlueprintConsistencyReport = PydanticField(default_factory=BlueprintConsistencyReport)


class UserRegisterRequest(ContractModel):
    full_name: str
    email: str
    password: str
    confirm_password: str
    workspace_name: str | None = None
    accept_terms: bool = False
    accept_privacy: bool = False
    accept_data_treatment: bool = False
    consent_system_notifications: bool = False
    consent_commercial_promotions: bool = False
    consent_events_newsletters: bool = False
    honeypot_field: str | None = None


class UserConsentUpdateRequest(ContractModel):
    consent_system_notifications: bool
    consent_commercial_promotions: bool
    consent_events_newsletters: bool


class UserConsentResponse(ContractModel):
    user_id: UUID
    consent_system_notifications: bool
    consent_commercial_promotions: bool
    consent_events_newsletters: bool
    updated_at: datetime


class SkillRerunResponse(ContractModel):
    skill_run: SkillRunEntry
    snapshot: SessionSnapshot


class UserLanguageUpdateRequest(ContractModel):
    preferred_language: str


class UserLanguageResponse(ContractModel):
    user_id: UUID
    preferred_language: str
    updated_at: datetime


class InitiativeEvaluationRequest(ContractModel):
    initiative_text: str = PydanticField(
        ...,
        min_length=5,
        max_length=4000,
        description="Descripcion de la iniciativa o caso de uso a evaluar para construccion de agente.",
    )
    language: str = PydanticField(default="es", description="Idioma preferido para la respuesta (es, en, pt).")
    business_context: str | None = PydanticField(default=None, max_length=1000)
    expected_users: str | None = PydanticField(default=None, max_length=500)


class InitiativeDimensionScore(ContractModel):
    dimension_key: str
    dimension_name: str
    score: int = PydanticField(ge=0, le=100)
    weight: float = PydanticField(ge=0.0, le=1.0)
    justification: str
    status: Literal["optimal", "acceptable", "critical"] = "optimal"


class InitiativeAlternativeRecommendation(ContractModel):
    recommended_technology: str
    technology_category: Literal["rpa", "deterministic_script", "traditional_software", "workflow_webhook", "prompt_chain"] = "deterministic_script"
    why_not_agent: str
    estimated_cost_risk: str
    suggested_next_step: str


class InitiativeEvaluationResponse(ContractModel):
    is_viable: bool
    readiness_score: int = PydanticField(ge=0, le=100)
    verdict_badge: Literal["viable", "partially_viable", "not_recommended"]
    verdict_title: str
    verdict_summary: str
    suggested_archetype: str | None = None
    suggested_tier: CommercialTier | None = None
    dimensions: list[InitiativeDimensionScore] = PydanticField(default_factory=list)
    key_strengths: list[str] = PydanticField(default_factory=list)
    key_risks_or_gaps: list[str] = PydanticField(default_factory=list)
    alternative: InitiativeAlternativeRecommendation | None = None
    prefilled_project_data: dict[str, Any] = PydanticField(default_factory=dict)
    token_usage: dict[str, int] = PydanticField(default_factory=dict)
    evaluation_id: str = ""

