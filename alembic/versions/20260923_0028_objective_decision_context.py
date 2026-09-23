"""objective decision context for ACP questions

Revision ID: 20260923_0028
Revises: 20260921_0027
Create Date: 2026-09-23 00:00:00.000000
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260923_0028"
down_revision = "20260921_0027"
branch_labels = None
depends_on = None


def _has_table(table_name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(table_name)


def _has_column(table_name: str, column_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return any(column["name"] == column_name for column in inspector.get_columns(table_name))


def _json_type() -> sa.types.TypeEngine:
    if op.get_bind().dialect.name == "postgresql":
        return postgresql.JSONB(astext_type=sa.Text())
    return sa.JSON()


def upgrade() -> None:
    if not _has_table("construction_question_responses"):
        return
    if _has_column("construction_question_responses", "decision_context"):
        return
    server_default = sa.text("'{}'::jsonb") if op.get_bind().dialect.name == "postgresql" else sa.text("'{}'")
    op.add_column(
        "construction_question_responses",
        sa.Column("decision_context", _json_type(), nullable=False, server_default=server_default),
    )


def downgrade() -> None:
    if not _has_table("construction_question_responses"):
        return
    if not _has_column("construction_question_responses", "decision_context"):
        return
    op.drop_column("construction_question_responses", "decision_context")
