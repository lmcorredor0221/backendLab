from __future__ import annotations

from datetime import datetime
from enum import StrEnum
import re
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


SAFE_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,119}$")


class DiagramNotation(StrEnum):
    flowchart = "flowchart"
    sequence = "sequence"
    class_diagram = "class"
    entity_relationship = "er"
    state = "state"
    journey = "journey"
    c4 = "c4"
    bpmn = "bpmn"
    deployment = "deployment"
    package = "package"
    capability = "capability"


class DiagramNode(BaseModel):
    id: str
    label: str
    kind: str = "component"
    description: str = ""
    group_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not SAFE_IDENTIFIER.fullmatch(normalized):
            raise ValueError("node id must be a stable safe identifier")
        return normalized

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("node label cannot be empty")
        return normalized[:180]


class DiagramEdge(BaseModel):
    id: str
    source: str
    target: str
    label: str = ""
    kind: str = "relationship"
    order: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("id", "source", "target")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not SAFE_IDENTIFIER.fullmatch(normalized):
            raise ValueError("edge identifiers must be stable and safe")
        return normalized


class DiagramGroup(BaseModel):
    id: str
    label: str
    kind: str = "boundary"
    parent_id: str | None = None
    description: str = ""

    @field_validator("id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not SAFE_IDENTIFIER.fullmatch(normalized):
            raise ValueError("group id must be a stable safe identifier")
        return normalized


class DiagramLegendItem(BaseModel):
    key: str
    label: str
    description: str = ""


class DiagramModel(BaseModel):
    schema_version: Literal["diagram-model.v1"] = "diagram-model.v1"
    diagram_key: str
    title: str
    description: str = ""
    notation: DiagramNotation = DiagramNotation.flowchart
    direction: Literal["TB", "TD", "BT", "RL", "LR"] = "LR"
    nodes: list[DiagramNode] = Field(default_factory=list)
    edges: list[DiagramEdge] = Field(default_factory=list)
    groups: list[DiagramGroup] = Field(default_factory=list)
    legend: list[DiagramLegendItem] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("diagram_key")
    @classmethod
    def validate_diagram_key(cls, value: str) -> str:
        normalized = value.strip()
        if not SAFE_IDENTIFIER.fullmatch(normalized):
            raise ValueError("diagram key must be a stable safe identifier")
        return normalized

    @model_validator(mode="after")
    def validate_graph_integrity(self) -> "DiagramModel":
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("node ids must be unique")
        edge_ids = [edge.id for edge in self.edges]
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("edge ids must be unique")
        group_ids = [group.id for group in self.groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("group ids must be unique")
        known_nodes = set(node_ids)
        invalid_edges = [edge.id for edge in self.edges if edge.source not in known_nodes or edge.target not in known_nodes]
        if invalid_edges:
            raise ValueError(f"edges reference unknown nodes: {', '.join(invalid_edges)}")
        known_groups = set(group_ids)
        invalid_group_nodes = [node.id for node in self.nodes if node.group_id and node.group_id not in known_groups]
        if invalid_group_nodes:
            raise ValueError(f"nodes reference unknown groups: {', '.join(invalid_group_nodes)}")
        return self


class DiagramGenerationInput(BaseModel):
    diagram_key: str
    title: str
    objective: str
    notation: DiagramNotation
    detail_level: Literal["executive", "standard", "detailed"] = "standard"
    required_inputs: list[str] = Field(default_factory=list)
    source_context: dict[str, Any] = Field(default_factory=dict)
    source_refs: list[str] = Field(default_factory=list)
    semantic_rules: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    prompt_spec_version: str = ""


class DiagramRegistryEntry(BaseModel):
    key: str
    title: str
    description: str
    benefit: str
    category: str
    type: str
    family: str
    notation: DiagramNotation
    complexity: Literal["basic", "intermediate", "advanced"] = "intermediate"
    stage: str
    required_tier: str
    preview_mode: Literal["full", "limited", "none"] = "limited"
    products: list[str] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    objective: str
    semantic_rules: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    sort_order: int = 0
    active: bool = True


class DiagramRegistry(BaseModel):
    schema_version: Literal["diagram-registry.v1"] = "diagram-registry.v1"
    prompt_spec_version: str
    entries: list[DiagramRegistryEntry]

    @model_validator(mode="after")
    def validate_unique_keys(self) -> "DiagramRegistry":
        keys = [entry.key for entry in self.entries]
        if len(keys) != len(set(keys)):
            raise ValueError("diagram registry keys must be unique")
        return self


class DiagramQualityReport(BaseModel):
    schema_version: Literal["diagram-quality-report.v1"] = "diagram-quality-report.v1"
    valid: bool
    score: int = Field(ge=0, le=100)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    checks: dict[str, bool] = Field(default_factory=dict)


class DiagramPolicyDecision(BaseModel):
    schema_version: Literal["diagram-policy-decision.v1"] = "diagram-policy-decision.v1"
    visible: bool = True
    access_state: Literal["available", "preview", "locked", "stage_locked", "disabled"] = "locked"
    can_generate: bool = False
    can_view: bool = False
    can_download: bool = False
    can_regenerate: bool = False
    can_compare: bool = False
    reason_code: str = ""
    reason: str = ""
    cta_label: str = ""
    required_tier: str = ""


class DiagramVersionSummary(BaseModel):
    id: UUID
    version_number: int
    state: str
    provider_key: str = ""
    model_name: str = ""
    prompt_spec_version: str = ""
    quality_score: int = 0
    created_at: datetime


class DiagramCatalogItemV3(BaseModel):
    key: str
    title: str
    description: str
    benefit: str
    category: str
    type: str
    family: str
    notation: DiagramNotation
    complexity: str
    stage: str
    required_tier: str
    products: list[str] = Field(default_factory=list)
    generation_state: Literal["pending", "queued", "generating", "available", "error", "updating"] = "pending"
    access: DiagramPolicyDecision
    updated_at: datetime | None = None
    current_version: DiagramVersionSummary | None = None
    available_actions: list[str] = Field(default_factory=list)


class DiagramCatalogV3Response(BaseModel):
    contract_version: Literal["diagram-catalog.v3"] = "diagram-catalog.v3"
    project_id: UUID
    workspace_id: UUID
    current_stage: str
    tier: str
    provider_key: str
    total_count: int
    available_count: int
    preview_count: int
    locked_count: int
    entries: list[DiagramCatalogItemV3]


class DiagramDetailV3Response(BaseModel):
    contract_version: Literal["diagram-detail.v3"] = "diagram-detail.v3"
    project_id: UUID
    item: DiagramCatalogItemV3
    model: DiagramModel | None = None
    renderings: dict[str, str] = Field(default_factory=dict)
    quality: DiagramQualityReport | None = None
    versions: list[DiagramVersionSummary] = Field(default_factory=list)


class DiagramGenerationRequest(BaseModel):
    detail_level: Literal["executive", "standard", "detailed"] = "standard"
    reason: str = "user_request"
    idempotency_key: str = ""


class DiagramGenerationJobResponse(BaseModel):
    contract_version: Literal["diagram-generation-job.v1"] = "diagram-generation-job.v1"
    id: UUID
    project_id: UUID
    diagram_key: str
    status: Literal["queued", "generating", "available", "error", "updating"]
    provider_key: str = ""
    model_name: str = ""
    version_id: UUID | None = None
    error_code: str = ""
    error_message: str = ""
    requested_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class DiagramGovernanceUpdate(BaseModel):
    enabled: bool = True
    generation_enabled: bool = True
    required_tier_override: str = ""
    preview_mode_override: str = ""
    prompt_status: Literal["draft", "active", "retired"] = "active"
    prompt_override: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""


class DiagramGovernanceEntry(BaseModel):
    diagram_key: str
    title: str
    enabled: bool
    generation_enabled: bool
    required_tier: str
    preview_mode: str
    prompt_spec_version: str
    prompt_status: str
    prompt_override: dict[str, Any] = Field(default_factory=dict)
    prompt_spec: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""
    updated_at: datetime | None = None


class DiagramGovernanceResponse(BaseModel):
    contract_version: Literal["diagram-governance.v1"] = "diagram-governance.v1"
    entries: list[DiagramGovernanceEntry]


class DiagramGovernanceAuditEntry(BaseModel):
    id: UUID
    diagram_key: str
    action: str
    changed_fields: list[str] = Field(default_factory=list)
    actor_user_id: UUID | None = None
    reason: str = ""
    created_at: datetime


class DiagramGovernanceOverview(BaseModel):
    contract_version: Literal["diagram-governance-overview.v1"] = "diagram-governance-overview.v1"
    active_provider: str
    provider_mode: str = ""
    model_name: str = ""
    provider_configured: bool = False
    registry_version: str
    prompt_spec_version: str
    job_counts: dict[str, int] = Field(default_factory=dict)
    total_versions: int = 0
    average_quality_score: int = 0
    recent_jobs: list[DiagramGenerationJobResponse] = Field(default_factory=list)
    recent_audit: list[DiagramGovernanceAuditEntry] = Field(default_factory=list)
