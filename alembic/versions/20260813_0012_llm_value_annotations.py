"""llm_value_annotations

Revision ID: 20260813_0012
Revises: 20260813_0011
Create Date: 2026-08-13 18:10:00.000000

"""
from __future__ import annotations

from collections.abc import Iterable

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260813_0012"
down_revision = "20260813_0011"
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
    if _has_table("llm_value_annotations"):
        return
    uuid = _uuid_type()
    json = _json_type()
    op.create_table(
        "llm_value_annotations",
        sa.Column("id", uuid, primary_key=True),
        sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=True),
        sa.Column("usage_record_id", uuid, sa.ForeignKey("llm_usage_ledger.id"), nullable=True),
        sa.Column("artifact_type", sa.String(), nullable=False),
        sa.Column("artifact_id", sa.String(), nullable=False),
        sa.Column("result_type", sa.String(), nullable=False),
        sa.Column("result_id", sa.String(), nullable=False),
        sa.Column("stage", sa.String(), nullable=False),
        sa.Column("decision_key", sa.String(), nullable=False),
        sa.Column("value_signal", sa.String(), nullable=False),
        sa.Column("artifact_created", sa.Boolean(), nullable=False),
        sa.Column("stage_completed", sa.Boolean(), nullable=False),
        sa.Column("evaluation_passed", sa.Boolean(), nullable=False),
        sa.Column("human_review_needed", sa.Boolean(), nullable=False),
        sa.Column("created_by_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
        sa.Column("metadata", json, nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    _create_indexes(
        "llm_value_annotations",
        (
            ("ix_llm_value_annotations_workspace_id", ["workspace_id"]),
            ("ix_llm_value_annotations_usage_record_id", ["usage_record_id"]),
            ("ix_llm_value_annotations_artifact_type", ["artifact_type"]),
            ("ix_llm_value_annotations_artifact_id", ["artifact_id"]),
            ("ix_llm_value_annotations_result_type", ["result_type"]),
            ("ix_llm_value_annotations_result_id", ["result_id"]),
            ("ix_llm_value_annotations_stage", ["stage"]),
            ("ix_llm_value_annotations_decision_key", ["decision_key"]),
            ("ix_llm_value_annotations_value_signal", ["value_signal"]),
            ("ix_llm_value_annotations_created_by_user_id", ["created_by_user_id"]),
            ("ix_llm_value_annotations_usage", ["usage_record_id"]),
            ("ix_llm_value_annotations_workspace_artifact", ["workspace_id", "artifact_type", "artifact_id"]),
            ("ix_llm_value_annotations_workspace_result", ["workspace_id", "result_type", "result_id"]),
            ("ix_llm_value_annotations_workspace_stage", ["workspace_id", "stage", "created_at"]),
        ),
    )


def downgrade() -> None:
    if _has_table("llm_value_annotations"):
        op.drop_table("llm_value_annotations")
