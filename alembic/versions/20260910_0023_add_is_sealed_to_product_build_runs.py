"""add_is_sealed_to_product_build_runs

Revision ID: 20260910_0023
Revises: 20260903_0022
Create Date: 2026-09-10 19:30:00.000000

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260910_0023"
down_revision = "20260903_0022"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("product_build_runs_v1"):
        columns = {c["name"] for c in inspector.get_columns("product_build_runs_v1")}
        if "is_sealed" not in columns:
            op.add_column(
                "product_build_runs_v1",
                sa.Column("is_sealed", sa.Boolean(), nullable=False, server_default=sa.text("false")),
            )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("product_build_runs_v1"):
        columns = {c["name"] for c in inspector.get_columns("product_build_runs_v1")}
        if "is_sealed" in columns:
            op.drop_column("product_build_runs_v1", "is_sealed")
