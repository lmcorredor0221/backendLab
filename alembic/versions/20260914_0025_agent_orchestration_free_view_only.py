"""agent_orchestration_free_view_only

Revision ID: 20260914_0025
Revises: 20260914_0024
Create Date: 2026-09-14 12:40:00.000000

"""
from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import sqlalchemy as sa
from alembic import op


revision = "20260914_0025"
down_revision = "20260914_0024"
branch_labels = None
depends_on = None


def _has_table(table_name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(table_name)


def _now() -> datetime:
    return datetime.utcnow()


def _diagram_table() -> sa.Table:
    return sa.table(
        "diagram_governance_v3",
        sa.column("id", sa.String()),
        sa.column("diagram_key", sa.String()),
        sa.column("enabled", sa.Boolean()),
        sa.column("generation_enabled", sa.Boolean()),
        sa.column("required_tier_override", sa.String()),
        sa.column("preview_mode_override", sa.String()),
        sa.column("prompt_status", sa.String()),
        sa.column("prompt_override", sa.JSON()),
        sa.column("notes", sa.String()),
        sa.column("updated_by_user_id", sa.String()),
        sa.column("created_at", sa.DateTime()),
        sa.column("updated_at", sa.DateTime()),
    )


def upgrade() -> None:
    if not _has_table("diagram_governance_v3"):
        return

    table = _diagram_table()
    bind = op.get_bind()
    existing_id = bind.execute(
        sa.select(table.c.id).where(table.c.diagram_key == "agent_orchestration")
    ).scalar_one_or_none()
    values = {
        "enabled": True,
        "generation_enabled": True,
        "required_tier_override": "blueprint",
        "preview_mode_override": "full",
        "prompt_status": "active",
        "notes": "Blueprint Free: visible y generable sin descarga; descarga/exportacion queda protegida por politica premium.",
        "updated_at": _now(),
    }
    if existing_id:
        bind.execute(table.update().where(table.c.diagram_key == "agent_orchestration").values(**values))
        return

    bind.execute(
        table.insert().values(
            id=str(uuid4()),
            diagram_key="agent_orchestration",
            prompt_override={},
            updated_by_user_id=None,
            created_at=_now(),
            **values,
        )
    )


def downgrade() -> None:
    if not _has_table("diagram_governance_v3"):
        return

    table = _diagram_table()
    op.get_bind().execute(
        table.update()
        .where(table.c.diagram_key == "agent_orchestration")
        .values(
            required_tier_override="blueprint_pro",
            preview_mode_override="limited",
            notes="Rollback: agent_orchestration vuelve a Blueprint Pro limitado.",
            updated_at=_now(),
        )
    )
