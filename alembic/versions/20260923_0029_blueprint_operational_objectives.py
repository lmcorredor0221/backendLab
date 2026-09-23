"""Add operational and objective payloads to blueprints.

Revision ID: 20260923_0029
Revises: 20260923_0028
Create Date: 2026-09-23 10:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260923_0029"
down_revision = "20260923_0028"
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
    if not _has_table("blueprints"):
        return

    for column_name in ("operational_profile", "objective_contract"):
        if _has_column("blueprints", column_name):
            continue
        op.add_column(
            "blueprints",
            sa.Column(
                column_name,
                _json_type(),
                nullable=False,
                server_default=_json_server_default(),
            ),
        )


def downgrade() -> None:
    if not _has_table("blueprints"):
        return

    for column_name in ("objective_contract", "operational_profile"):
        if _has_column("blueprints", column_name):
            op.drop_column("blueprints", column_name)
