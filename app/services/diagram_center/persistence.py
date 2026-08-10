from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Column, JSON, UniqueConstraint
from sqlmodel import Field, SQLModel

from app.models import utc_now


class DiagramGenerationJobRecord(SQLModel, table=True):
    __tablename__ = "diagram_generation_jobs_v3"
    __table_args__ = (
        UniqueConstraint("workspace_id", "idempotency_key", name="uq_diagram_job_workspace_idempotency"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    diagram_key: str = Field(index=True, nullable=False)
    requested_by_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True)
    status: str = Field(default="queued", index=True, nullable=False)
    detail_level: str = Field(default="standard", nullable=False)
    reason: str = Field(default="user_request", nullable=False)
    idempotency_key: str = Field(index=True, nullable=False)
    provider_key: str = Field(default="", index=True, nullable=False)
    model_name: str = Field(default="", nullable=False)
    prompt_spec_version: str = Field(default="", nullable=False)
    version_id: UUID | None = Field(default=None, nullable=True, index=True)
    error_code: str = Field(default="", nullable=False)
    error_message: str = Field(default="", nullable=False)
    request_metadata: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    requested_at: datetime = Field(default_factory=utc_now, nullable=False)
    started_at: datetime | None = Field(default=None, nullable=True)
    completed_at: datetime | None = Field(default=None, nullable=True)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class DiagramVersionRecord(SQLModel, table=True):
    __tablename__ = "diagram_versions_v3"
    __table_args__ = (
        UniqueConstraint("session_id", "diagram_key", "version_number", name="uq_diagram_version_number_v3"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    diagram_key: str = Field(index=True, nullable=False)
    job_id: UUID | None = Field(default=None, foreign_key="diagram_generation_jobs_v3.id", nullable=True, index=True)
    version_number: int = Field(default=1, nullable=False)
    state: str = Field(default="available", index=True, nullable=False)
    diagram_model: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    renderings: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    quality_report: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    source_fingerprint: str = Field(default="", index=True, nullable=False)
    source_refs: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    provider_key: str = Field(default="", index=True, nullable=False)
    model_name: str = Field(default="", nullable=False)
    prompt_spec_version: str = Field(default="", nullable=False)
    request_id: str = Field(default="", nullable=False)
    created_by_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class DiagramGovernanceRecord(SQLModel, table=True):
    __tablename__ = "diagram_governance_v3"
    __table_args__ = (UniqueConstraint("diagram_key", name="uq_diagram_governance_key_v3"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    diagram_key: str = Field(index=True, nullable=False)
    enabled: bool = Field(default=True, nullable=False)
    generation_enabled: bool = Field(default=True, nullable=False)
    required_tier_override: str = Field(default="", nullable=False)
    preview_mode_override: str = Field(default="", nullable=False)
    prompt_status: str = Field(default="active", nullable=False)
    prompt_override: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    notes: str = Field(default="", nullable=False)
    updated_by_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class DiagramGovernanceAuditRecord(SQLModel, table=True):
    __tablename__ = "diagram_governance_audit_v3"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    diagram_key: str = Field(index=True, nullable=False)
    action: str = Field(default="governance_updated", index=True, nullable=False)
    changed_fields: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    before_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    after_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    actor_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True, index=True)
    reason: str = Field(default="", nullable=False)
    created_at: datetime = Field(default_factory=utc_now, nullable=False, index=True)
