from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Column, JSON, UniqueConstraint
from sqlmodel import Field, SQLModel

from app.models import utc_now


class DeliverableGovernanceRecord(SQLModel, table=True):
    __tablename__ = "deliverable_governance_v1"
    __table_args__ = (UniqueConstraint("scope_key", "deliverable_key", name="uq_deliverable_governance_scope_key_v1"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    scope_key: str = Field(default="platform", index=True, nullable=False)
    workspace_id: UUID | None = Field(default=None, foreign_key="workspaces.id", index=True, nullable=True)
    deliverable_key: str = Field(index=True, nullable=False)
    enabled: bool = Field(default=True, nullable=False)
    generation_enabled: bool = Field(default=True, nullable=False)
    required_tier_override: str = Field(default="", nullable=False)
    preview_mode_override: str = Field(default="", nullable=False)
    prompt_status: str = Field(default="active", nullable=False)
    prompt_override: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    notes: str = Field(default="", nullable=False)
    updated_by_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True, index=True)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class DeliverableGovernanceAuditRecord(SQLModel, table=True):
    __tablename__ = "deliverable_governance_audit_v1"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    scope_key: str = Field(default="platform", index=True, nullable=False)
    workspace_id: UUID | None = Field(default=None, foreign_key="workspaces.id", index=True, nullable=True)
    deliverable_key: str = Field(index=True, nullable=False)
    action: str = Field(default="governance_updated", index=True, nullable=False)
    changed_fields: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    before_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    after_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    actor_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True, index=True)
    reason: str = Field(default="", nullable=False)
    created_at: datetime = Field(default_factory=utc_now, nullable=False, index=True)


class DeliverableGenerationJobRecord(SQLModel, table=True):
    __tablename__ = "deliverable_generation_jobs_v1"
    __table_args__ = (
        UniqueConstraint("workspace_id", "idempotency_key", name="uq_deliverable_job_workspace_idempotency_v1"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    deliverable_key: str = Field(index=True, nullable=False)
    requested_by_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True, index=True)
    status: str = Field(default="queued", index=True, nullable=False)
    product_mode: str = Field(default="basic_free", index=True, nullable=False)
    generation_mode: str = Field(default="deterministic", index=True, nullable=False)
    idempotency_key: str = Field(index=True, nullable=False)
    provider_key: str = Field(default="", index=True, nullable=False)
    model_name: str = Field(default="", nullable=False)
    prompt_version_id: UUID | None = Field(default=None, foreign_key="deliverable_prompt_versions_v1.id", nullable=True, index=True)
    output_version_id: UUID | None = Field(default=None, nullable=True, index=True)
    error_code: str = Field(default="", nullable=False)
    error_message: str = Field(default="", nullable=False)
    tokens_input: int = Field(default=0, nullable=False)
    tokens_output: int = Field(default=0, nullable=False)
    estimated_cost_usd: float = Field(default=0.0, nullable=False)
    request_metadata: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    requested_at: datetime = Field(default_factory=utc_now, nullable=False)
    started_at: datetime | None = Field(default=None, nullable=True)
    completed_at: datetime | None = Field(default=None, nullable=True)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class DeliverableQualitySnapshotRecord(SQLModel, table=True):
    __tablename__ = "deliverable_quality_snapshots_v1"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    deliverable_key: str = Field(index=True, nullable=False)
    version_ref: str = Field(default="", index=True, nullable=False)
    state: str = Field(default="unknown", index=True, nullable=False)
    score: int = Field(default=0, nullable=False)
    errors: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    warnings: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    checks: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    source_fingerprint: str = Field(default="", index=True, nullable=False)
    created_at: datetime = Field(default_factory=utc_now, nullable=False, index=True)


class DeliverablePromptVersionRecord(SQLModel, table=True):
    __tablename__ = "deliverable_prompt_versions_v1"
    __table_args__ = (
        UniqueConstraint("scope_key", "deliverable_key", "version", name="uq_deliverable_prompt_scope_version_v1"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    scope_key: str = Field(default="platform", index=True, nullable=False)
    workspace_id: UUID | None = Field(default=None, foreign_key="workspaces.id", index=True, nullable=True)
    deliverable_key: str = Field(index=True, nullable=False)
    version: str = Field(default="1.0.0", nullable=False)
    status: str = Field(default="active", index=True, nullable=False)
    prompt_template_key: str = Field(default="", index=True, nullable=False)
    prompt_body: str = Field(default="", nullable=False)
    schema_contract: str = Field(default="", nullable=False)
    validator_key: str = Field(default="", nullable=False)
    fallback_policy: str = Field(default="", nullable=False)
    metadata_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSON, nullable=False))
    created_by_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True, index=True)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class DeliverablePromptAuditRecord(SQLModel, table=True):
    __tablename__ = "deliverable_prompt_audit_v1"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    scope_key: str = Field(default="platform", index=True, nullable=False)
    workspace_id: UUID | None = Field(default=None, foreign_key="workspaces.id", index=True, nullable=True)
    deliverable_key: str = Field(index=True, nullable=False)
    prompt_version_id: UUID | None = Field(default=None, foreign_key="deliverable_prompt_versions_v1.id", nullable=True, index=True)
    action: str = Field(default="prompt_updated", index=True, nullable=False)
    changed_fields: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    before_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    after_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    actor_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True, index=True)
    reason: str = Field(default="", nullable=False)
    created_at: datetime = Field(default_factory=utc_now, nullable=False, index=True)
