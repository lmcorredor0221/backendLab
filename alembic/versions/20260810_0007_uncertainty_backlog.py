"""uncertainty_backlog

Revision ID: 20260810_0007
Revises: 20260806_0006
Create Date: 2026-08-10 18:00:00.000000

"""
from __future__ import annotations

from collections.abc import Iterable

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260810_0007"
down_revision = "20260806_0006"
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
    if _has_table("uncertainty_backlog_v1"):
        return
    uuid = _uuid_type()
    json = _json_type()
    op.create_table(
        "uncertainty_backlog_v1",
        sa.Column("id", uuid, primary_key=True),
        sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
        sa.Column("session_id", uuid, sa.ForeignKey("sessions.id"), nullable=False),
        sa.Column("uncertainty_key", sa.String(), nullable=False),
        sa.Column("product_mode", sa.String(), nullable=False),
        sa.Column("source_stage", sa.String(), nullable=False),
        sa.Column("target_stage", sa.String(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("disposition", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("reason", sa.String(), nullable=False),
        sa.Column("impact", sa.String(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("cost_to_resolve_units", sa.Integer(), nullable=False),
        sa.Column("assumed_answer", sa.String(), nullable=False),
        sa.Column("suggested_answer", sa.String(), nullable=False),
        sa.Column("answer_options", json, nullable=False),
        sa.Column("source_refs", json, nullable=False),
        sa.Column("affected_deliverable_keys", json, nullable=False),
        sa.Column("dependency_keys", json, nullable=False),
        sa.Column("payload", json, nullable=False),
        sa.Column("created_from", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("resolved_at", sa.DateTime(), nullable=True),
        sa.Column("superseded_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint(
            "session_id",
            "uncertainty_key",
            "product_mode",
            name="uq_uncertainty_backlog_session_key_mode_v1",
        ),
    )
    _create_indexes(
        "uncertainty_backlog_v1",
        (
            ("ix_uncertainty_backlog_v1_workspace_id", ["workspace_id"]),
            ("ix_uncertainty_backlog_v1_session_id", ["session_id"]),
            ("ix_uncertainty_backlog_v1_uncertainty_key", ["uncertainty_key"]),
            ("ix_uncertainty_backlog_v1_product_mode", ["product_mode"]),
            ("ix_uncertainty_backlog_v1_source_stage", ["source_stage"]),
            ("ix_uncertainty_backlog_v1_target_stage", ["target_stage"]),
            ("ix_uncertainty_backlog_v1_kind", ["kind"]),
            ("ix_uncertainty_backlog_v1_disposition", ["disposition"]),
            ("ix_uncertainty_backlog_v1_status", ["status"]),
            ("ix_uncertainty_backlog_v1_resolved_at", ["resolved_at"]),
            ("ix_uncertainty_backlog_v1_superseded_at", ["superseded_at"]),
        ),
    )


def downgrade() -> None:
    if _has_table("uncertainty_backlog_v1"):
        op.drop_table("uncertainty_backlog_v1")
