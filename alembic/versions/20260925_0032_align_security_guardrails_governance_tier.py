"""Align security_guardrails deliverable governance tier to blueprint_pro.

Revision ID: 20260925_0032
Revises: 20260924_0031
Create Date: 2026-09-25 18:25:00.000000
"""

from __future__ import annotations

from datetime import datetime

from alembic import op
import sqlalchemy as sa


revision = "20260925_0032"
down_revision = "20260924_0031"
branch_labels = None
depends_on = None


def _has_table(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    if not _has_table("deliverable_governance_v1"):
        return

    bind = op.get_bind()
    bind.execute(
        sa.text(
            """
            UPDATE deliverable_governance_v1
            SET required_tier_override = 'blueprint_pro',
                preview_mode_override = 'limited',
                notes = 'Curacion Blueprint 2026-09: Movido a blueprint_pro',
                updated_at = :now
            WHERE deliverable_key = 'diagram.security_guardrails'
              AND scope_key = 'platform'
            """
        ),
        {"now": datetime.utcnow()},
    )


def downgrade() -> None:
    if not _has_table("deliverable_governance_v1"):
        return

    bind = op.get_bind()
    bind.execute(
        sa.text(
            """
            UPDATE deliverable_governance_v1
            SET required_tier_override = 'blueprint',
                preview_mode_override = 'full',
                notes = 'Rollback: diagram.security_guardrails a blueprint',
                updated_at = :now
            WHERE deliverable_key = 'diagram.security_guardrails'
              AND scope_key = 'platform'
            """
        ),
        {"now": datetime.utcnow()},
    )
