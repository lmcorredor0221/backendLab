"""Add operational profile to opportunities.

Revision ID: 20260923_0030
Revises: 20260923_0029
Create Date: 2026-09-23 11:30:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260923_0030"
down_revision = "20260923_0029"
branch_labels = None
depends_on = None


def _json_type() -> sa.types.TypeEngine:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        return postgresql.JSONB()
    return sa.JSON()


def _json_server_default() -> sa.TextClause:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        return sa.text("'{}'::jsonb")
    return sa.text("'{}'")


def _has_table(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return table_name in inspector.get_table_names()


def _has_column(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return any(column["name"] == column_name for column in inspector.get_columns(table_name))


def upgrade() -> None:
    if not _has_table("opportunities") or _has_column("opportunities", "operational_profile"):
        return
    op.add_column(
        "opportunities",
        sa.Column(
            "operational_profile",
            _json_type(),
            nullable=False,
            server_default=_json_server_default(),
        ),
    )


def downgrade() -> None:
    if _has_table("opportunities") and _has_column("opportunities", "operational_profile"):
        op.drop_column("opportunities", "operational_profile")
