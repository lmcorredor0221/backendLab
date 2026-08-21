"""llm_finops_alerts

Revision ID: 20260813_0011
Revises: 20260813_0010
Create Date: 2026-08-13 13:30:00.000000

"""
from __future__ import annotations

from collections.abc import Iterable

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260813_0011"
down_revision = "20260813_0010"
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
    if _has_table("llm_finops_alerts"):
        return
    uuid = _uuid_type()
    json = _json_type()
    op.create_table(
        "llm_finops_alerts",
        sa.Column("id", uuid, primary_key=True),
        sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
        sa.Column("budget_policy_id", uuid, sa.ForeignKey("llm_budget_policies.id"), nullable=True),
        sa.Column("usage_record_id", uuid, sa.ForeignKey("llm_usage_ledger.id"), nullable=True),
        sa.Column("alert_key", sa.String(), nullable=False),
        sa.Column("alert_type", sa.String(), nullable=False),
        sa.Column("severity", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("message", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("scope_type", sa.String(), nullable=False),
        sa.Column("scope_value", sa.String(), nullable=False),
        sa.Column("provider_key", sa.String(), nullable=False),
        sa.Column("model_name", sa.String(), nullable=False),
        sa.Column("stage", sa.String(), nullable=False),
        sa.Column("threshold_percent", sa.Float(), nullable=False),
        sa.Column("period_start", sa.DateTime(), nullable=False),
        sa.Column("period_end", sa.DateTime(), nullable=False),
        sa.Column("consumed_amount", sa.Float(), nullable=False),
        sa.Column("limit_amount", sa.Float(), nullable=False),
        sa.Column("currency", sa.String(), nullable=False),
        sa.Column("evidence", json, nullable=False),
        sa.Column("metadata", json, nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("resolved_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint(
            "workspace_id",
            "alert_key",
            "period_start",
            "period_end",
            name="uq_llm_finops_alert_period_key",
        ),
    )
    _create_indexes(
        "llm_finops_alerts",
        (
            ("ix_llm_finops_alerts_workspace_id", ["workspace_id"]),
            ("ix_llm_finops_alerts_budget_policy_id", ["budget_policy_id"]),
            ("ix_llm_finops_alerts_usage_record_id", ["usage_record_id"]),
            ("ix_llm_finops_alerts_alert_key", ["alert_key"]),
            ("ix_llm_finops_alerts_alert_type", ["alert_type"]),
            ("ix_llm_finops_alerts_severity", ["severity"]),
            ("ix_llm_finops_alerts_status", ["status"]),
            ("ix_llm_finops_alerts_scope_type", ["scope_type"]),
            ("ix_llm_finops_alerts_scope_value", ["scope_value"]),
            ("ix_llm_finops_alerts_provider_key", ["provider_key"]),
            ("ix_llm_finops_alerts_model_name", ["model_name"]),
            ("ix_llm_finops_alerts_stage", ["stage"]),
            ("ix_llm_finops_alerts_period_start", ["period_start"]),
            ("ix_llm_finops_alerts_period_end", ["period_end"]),
            ("ix_llm_finops_alerts_currency", ["currency"]),
            ("ix_llm_finops_alerts_workspace_status_created", ["workspace_id", "status", "created_at"]),
            ("ix_llm_finops_alerts_policy_threshold", ["budget_policy_id", "threshold_percent"]),
            ("ix_llm_finops_alerts_scope_period", ["workspace_id", "scope_type", "scope_value", "period_start"]),
            ("ix_llm_finops_alerts_provider_model", ["workspace_id", "provider_key", "model_name"]),
        ),
    )


def downgrade() -> None:
    if _has_table("llm_finops_alerts"):
        op.drop_table("llm_finops_alerts")
