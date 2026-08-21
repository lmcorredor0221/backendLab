from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Column, JSON, UniqueConstraint
from sqlmodel import Field, SQLModel

from app.models import utc_now


class UncertaintyBacklogRecord(SQLModel, table=True):
    __tablename__ = "uncertainty_backlog_v1"
    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "uncertainty_key",
            "product_mode",
            name="uq_uncertainty_backlog_session_key_mode_v1",
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    uncertainty_key: str = Field(index=True, nullable=False)
    product_mode: str = Field(index=True, nullable=False)
    source_stage: str = Field(index=True, nullable=False)
    target_stage: str = Field(default="", index=True, nullable=False)
    kind: str = Field(default="question", index=True, nullable=False)
    disposition: str = Field(default="defer", index=True, nullable=False)
    status: str = Field(default="open", index=True, nullable=False)
    title: str = Field(default="", nullable=False)
    description: str = Field(default="", nullable=False)
    reason: str = Field(default="", nullable=False)
    impact: str = Field(default="", nullable=False)
    confidence: float = Field(default=0.0, nullable=False)
    cost_to_resolve_units: int = Field(default=1, nullable=False)
    assumed_answer: str = Field(default="", nullable=False)
    suggested_answer: str = Field(default="", nullable=False)
    answer_options: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    source_refs: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    affected_deliverable_keys: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    dependency_keys: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    created_from: str = Field(default="runtime", nullable=False)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)
    resolved_at: datetime | None = Field(default=None, nullable=True, index=True)
    superseded_at: datetime | None = Field(default=None, nullable=True, index=True)


class ProductBuildRunRecord(SQLModel, table=True):
    __tablename__ = "product_build_runs_v1"
    __table_args__ = (
        UniqueConstraint("workspace_id", "idempotency_key", name="uq_product_build_run_workspace_idempotency_v1"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    product_key: str = Field(index=True, nullable=False)
    product_mode: str = Field(index=True, nullable=False)
    entitlement_tier: str = Field(default="blueprint", index=True, nullable=False)
    access_state: str = Field(default="preview", index=True, nullable=False)
    lifecycle: str = Field(default="ready_to_start", index=True, nullable=False)
    progress_percent: int = Field(default=0, nullable=False)
    completed_units: float = Field(default=0.0, nullable=False)
    total_units: float = Field(default=0.0, nullable=False)
    blocked_units: float = Field(default=0.0, nullable=False)
    idempotency_key: str = Field(index=True, nullable=False)
    checkpoint_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    error_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    created_by_user_id: UUID | None = Field(default=None, foreign_key="users.id", nullable=True, index=True)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    started_at: datetime | None = Field(default=None, nullable=True)
    completed_at: datetime | None = Field(default=None, nullable=True)
    requires_attention_at: datetime | None = Field(default=None, nullable=True)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class ProductBuildStepRecord(SQLModel, table=True):
    __tablename__ = "product_build_steps_v1"
    __table_args__ = (UniqueConstraint("run_id", "step_key", name="uq_product_build_step_run_key_v1"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    run_id: UUID = Field(foreign_key="product_build_runs_v1.id", index=True)
    workspace_id: UUID = Field(foreign_key="workspaces.id", index=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    step_key: str = Field(index=True, nullable=False)
    stage_key: str = Field(default="", index=True, nullable=False)
    deliverable_key: str = Field(default="", index=True, nullable=False)
    job_id: UUID | None = Field(default=None, nullable=True, index=True)
    dependency_key: str = Field(default="", index=True, nullable=False)
    status: str = Field(default="queued", index=True, nullable=False)
    sequence: int = Field(default=0, nullable=False)
    progress_percent: int = Field(default=0, nullable=False)
    checkpoint_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    error_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    started_at: datetime | None = Field(default=None, nullable=True)
    completed_at: datetime | None = Field(default=None, nullable=True)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)
