"""llm_finops_ledger

Revision ID: 20260813_0008
Revises: 20260810_0007
Create Date: 2026-08-13 12:00:00.000000

"""
from __future__ import annotations

from collections.abc import Iterable

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260813_0008"
down_revision = "20260810_0007"
branch_labels = None
depends_on = None


def _uuid_type() -> sa.types.TypeEngine:
    if op.get_bind().dialect.name == "postgresql":
        return postgresql.UUID(as_uuid=True)
    return sa.String(length=36)


def _json_type() -> sa.types.TypeEngine:
    if op.get_bind().dialect.name == "postgresql":
        return postgresql.JSONB()
    return sa.JSON()


def _has_table(table_name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(table_name)


def _create_indexes(table_name: str, indexes: Iterable[tuple[str, list[str]]]) -> None:
    for index_name, columns in indexes:
        op.create_index(index_name, table_name, columns)


def upgrade() -> None:
    if _has_table("llm_usage_ledger"):
        return
    uuid = _uuid_type()
    json = _json_type()
    op.create_table(
        "llm_usage_ledger",
        sa.Column("id", uuid, primary_key=True),
        sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=True),
        sa.Column("user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
        sa.Column("session_id", uuid, sa.ForeignKey("sessions.id"), nullable=True),
        sa.Column("project_id", uuid, nullable=True),
        sa.Column("initiative_id", uuid, nullable=True),
        sa.Column("stage", sa.String(), nullable=False),
        sa.Column("substage", sa.String(), nullable=False),
        sa.Column("agent_key", sa.String(), nullable=False),
        sa.Column("capability_key", sa.String(), nullable=False),
        sa.Column("action_key", sa.String(), nullable=False),
        sa.Column("operation_id", uuid, nullable=True),
        sa.Column("parent_run_id", sa.String(), nullable=False),
        sa.Column("correlation_id", sa.String(), nullable=False),
        sa.Column("provider_key", sa.String(), nullable=False),
        sa.Column("model_name", sa.String(), nullable=False),
        sa.Column("requested_model", sa.String(), nullable=False),
        sa.Column("execution_backend", sa.String(), nullable=False),
        sa.Column("execution_mode", sa.String(), nullable=False),
        sa.Column("request_id", sa.String(), nullable=False),
        sa.Column("provider_request_id", sa.String(), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("fallback_used", sa.Boolean(), nullable=False),
        sa.Column("shadow_provider_key", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("failure_kind", sa.String(), nullable=False),
        sa.Column("failure_detail_redacted", sa.String(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("queue_wait_ms", sa.Integer(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("total_tokens", sa.Integer(), nullable=False),
        sa.Column("cached_input_tokens", sa.Integer(), nullable=False),
        sa.Column("reasoning_tokens", sa.Integer(), nullable=False),
        sa.Column("other_token_metrics", json, nullable=False),
        sa.Column("provider_metrics", json, nullable=False),
        sa.Column("cost_input", sa.Float(), nullable=False),
        sa.Column("cost_output", sa.Float(), nullable=False),
        sa.Column("cost_other", sa.Float(), nullable=False),
        sa.Column("cost_total", sa.Float(), nullable=False),
        sa.Column("currency", sa.String(), nullable=False),
        sa.Column("fx_rate", sa.Float(), nullable=False),
        sa.Column("pricing_profile_key", sa.String(), nullable=False),
        sa.Column("pricing_snapshot", json, nullable=False),
        sa.Column("usage_raw_redacted", json, nullable=False),
        sa.Column("prompt_hash", sa.String(), nullable=False),
        sa.Column("response_hash", sa.String(), nullable=False),
        sa.Column("schema_validation_status", sa.String(), nullable=False),
        sa.Column("finish_reason", sa.String(), nullable=False),
        sa.Column("value_signal", sa.String(), nullable=False),
        sa.Column("metadata", json, nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    _create_indexes(
        "llm_usage_ledger",
        (
            ("ix_llm_usage_ledger_workspace_id", ["workspace_id"]),
            ("ix_llm_usage_ledger_user_id", ["user_id"]),
            ("ix_llm_usage_ledger_session_id", ["session_id"]),
            ("ix_llm_usage_ledger_project_id", ["project_id"]),
            ("ix_llm_usage_ledger_initiative_id", ["initiative_id"]),
            ("ix_llm_usage_ledger_stage", ["stage"]),
            ("ix_llm_usage_ledger_substage", ["substage"]),
            ("ix_llm_usage_ledger_agent_key", ["agent_key"]),
            ("ix_llm_usage_ledger_capability_key", ["capability_key"]),
            ("ix_llm_usage_ledger_action_key", ["action_key"]),
            ("ix_llm_usage_ledger_operation_id", ["operation_id"]),
            ("ix_llm_usage_ledger_parent_run_id", ["parent_run_id"]),
            ("ix_llm_usage_ledger_correlation_id", ["correlation_id"]),
            ("ix_llm_usage_ledger_provider_key", ["provider_key"]),
            ("ix_llm_usage_ledger_model_name", ["model_name"]),
            ("ix_llm_usage_ledger_execution_backend", ["execution_backend"]),
            ("ix_llm_usage_ledger_execution_mode", ["execution_mode"]),
            ("ix_llm_usage_ledger_request_id", ["request_id"]),
            ("ix_llm_usage_ledger_provider_request_id", ["provider_request_id"]),
            ("ix_llm_usage_ledger_shadow_provider_key", ["shadow_provider_key"]),
            ("ix_llm_usage_ledger_status", ["status"]),
            ("ix_llm_usage_ledger_failure_kind", ["failure_kind"]),
            ("ix_llm_usage_ledger_started_at", ["started_at"]),
            ("ix_llm_usage_ledger_cost_total", ["cost_total"]),
            ("ix_llm_usage_ledger_currency", ["currency"]),
            ("ix_llm_usage_ledger_pricing_profile_key", ["pricing_profile_key"]),
            ("ix_llm_usage_ledger_prompt_hash", ["prompt_hash"]),
            ("ix_llm_usage_ledger_response_hash", ["response_hash"]),
            ("ix_llm_usage_ledger_schema_validation_status", ["schema_validation_status"]),
            ("ix_llm_usage_ledger_value_signal", ["value_signal"]),
            ("ix_llm_usage_ledger_workspace_started", ["workspace_id", "started_at"]),
            ("ix_llm_usage_ledger_user_started", ["workspace_id", "user_id", "started_at"]),
            ("ix_llm_usage_ledger_session_started", ["workspace_id", "session_id", "started_at"]),
            ("ix_llm_usage_ledger_project_started", ["workspace_id", "project_id", "started_at"]),
            ("ix_llm_usage_ledger_stage_capability_started", ["workspace_id", "stage", "capability_key", "started_at"]),
            ("ix_llm_usage_ledger_provider_model_started", ["workspace_id", "provider_key", "model_name", "started_at"]),
            ("ix_llm_usage_ledger_request_attempt", ["request_id", "attempt_number"]),
            ("ix_llm_usage_ledger_operation", ["operation_id"]),
        ),
    )


def downgrade() -> None:
    if _has_table("llm_usage_ledger"):
        op.drop_table("llm_usage_ledger")
