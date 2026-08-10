"""add_antigravity_cli_enum

Revision ID: 20260806_0006
Revises: 20260805_0005
Create Date: 2026-08-06 12:23:00.000000

"""
from __future__ import annotations

from alembic import op
from sqlalchemy import text


revision = "20260806_0006"
down_revision = "20260805_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(text("ALTER TYPE llmproviderkey ADD VALUE IF NOT EXISTS 'antigravity_cli';"))


def downgrade() -> None:
    pass
