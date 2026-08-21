"""llm_finops_budgets

Revision ID: 20260813_0010
Revises: 20260813_0008, 20260813_0009
Create Date: 2026-08-13 13:00:00.000000

"""
from __future__ import annotations

from collections.abc import Iterable

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260813_0010"
down_revision = ("20260813_0008", "20260813_0009")
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
    if _has_table("llm_budget_policies"):
        return
    uuid = _uuid_type()
    json = _json_type()
    op.create_table(
        "llm_budget_policies",
        sa.Column("id", uuid, primary_key=True),
        sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
        sa.Column("policy_key", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("scope_type", sa.String(), nullable=False),
        sa.Column("scope_value", sa.String(), nullable=False),
        sa.Column("user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
        sa.Column("project_id", uuid, nullable=True),
        sa.Column("initiative_id", uuid, nullable=True),
        sa.Column("stage", sa.String(), nullable=False),
        sa.Column("provider_key", sa.String(), nullable=False),
        sa.Column("model_name", sa.String(), nullable=False),
        sa.Column("period_type", sa.String(), nullable=False),
        sa.Column("custom_period_start", sa.DateTime(), nullable=True),
        sa.Column("custom_period_end", sa.DateTime(), nullable=True),
        sa.Column("limit_amount", sa.Float(), nullable=False),
        sa.Column("currency", sa.String(), nullable=False),
        sa.Column("threshold_percentages", json, nullable=False),
        sa.Column("hard_limit_percent", sa.Float(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_by_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
        sa.Column("updated_by_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
        sa.Column("metadata", json, nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("workspace_id", "policy_key", name="uq_llm_budget_policy_workspace_key"),
    )
    _create_indexes(
        "llm_budget_policies",
        (
            ("ix_llm_budget_policies_workspace_id", ["workspace_id"]),
            ("ix_llm_budget_policies_policy_key", ["policy_key"]),
            ("ix_llm_budget_policies_scope_type", ["scope_type"]),
            ("ix_llm_budget_policies_scope_value", ["scope_value"]),
            ("ix_llm_budget_policies_user_id", ["user_id"]),
            ("ix_llm_budget_policies_project_id", ["project_id"]),
            ("ix_llm_budget_policies_initiative_id", ["initiative_id"]),
            ("ix_llm_budget_policies_stage", ["stage"]),
            ("ix_llm_budget_policies_provider_key", ["provider_key"]),
            ("ix_llm_budget_policies_model_name", ["model_name"]),
            ("ix_llm_budget_policies_period_type", ["period_type"]),
            ("ix_llm_budget_policies_currency", ["currency"]),
            ("ix_llm_budget_policies_is_active", ["is_active"]),
            ("ix_llm_budget_policies_workspace_scope", ["workspace_id", "scope_type", "scope_value"]),
            ("ix_llm_budget_policies_active_period", ["workspace_id", "is_active", "period_type"]),
            ("ix_llm_budget_policies_provider_model", ["workspace_id", "provider_key", "model_name"]),
        ),
    )


def downgrade() -> None:
    if _has_table("llm_budget_policies"):
        op.drop_table("llm_budget_policies")
