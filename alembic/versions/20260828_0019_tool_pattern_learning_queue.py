"""tool pattern learning queue

Revision ID: 20260828_0019
Revises: 20260828_0018
Create Date: 2026-08-28 17:10:00.000000

"""
from __future__ import annotations

from collections.abc import Iterable

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260828_0019"
down_revision = "20260828_0018"
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
    if _has_table("tool_pattern_learning_candidates"):
        return
    uuid = _uuid_type()
    json = _json_type()
    op.create_table(
        "tool_pattern_learning_candidates",
        sa.Column("id", uuid, primary_key=True),
        sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
        sa.Column("session_id", uuid, sa.ForeignKey("sessions.id"), nullable=False),
        sa.Column("source_artifact_id", uuid, sa.ForeignKey("journey_stage_artifacts.id"), nullable=True),
        sa.Column("source_blueprint_version", sa.Integer(), nullable=True),
        sa.Column("candidate_pattern_id", sa.String(), nullable=False),
        sa.Column("capability_key", sa.String(), nullable=False),
        sa.Column("family_key", sa.String(), nullable=False),
        sa.Column("label", sa.String(), nullable=False),
        sa.Column("source_level", sa.String(), nullable=False),
        sa.Column("promotion_status", sa.String(), nullable=False),
        sa.Column("global_promotion_allowed", sa.Boolean(), nullable=False),
        sa.Column("dedupe_signature", sa.String(), nullable=False),
        sa.Column("replacement_global_pattern_id", sa.String(), nullable=False),
        sa.Column("contract_quality", sa.String(), nullable=False),
        sa.Column("risk_flags", json, nullable=False),
        sa.Column("source_refs", json, nullable=False),
        sa.Column("evidence_refs", json, nullable=False),
        sa.Column("contract_seed_payload", json, nullable=False),
        sa.Column("metadata", json, nullable=False),
        sa.Column("observation_count", sa.Integer(), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint(
            "workspace_id",
            "session_id",
            "dedupe_signature",
            name="uq_tool_pattern_learning_candidate_session_signature",
        ),
    )
    _create_indexes(
        "tool_pattern_learning_candidates",
        (
            ("ix_tool_pattern_learning_candidates_workspace_id", ["workspace_id"]),
            ("ix_tool_pattern_learning_candidates_session_id", ["session_id"]),
            ("ix_tool_pattern_learning_candidates_source_artifact_id", ["source_artifact_id"]),
            ("ix_tool_pattern_learning_candidates_source_blueprint_version", ["source_blueprint_version"]),
            ("ix_tool_pattern_learning_candidates_candidate_pattern_id", ["candidate_pattern_id"]),
            ("ix_tool_pattern_learning_candidates_capability_key", ["capability_key"]),
            ("ix_tool_pattern_learning_candidates_family_key", ["family_key"]),
            ("ix_tool_pattern_learning_candidates_source_level", ["source_level"]),
            ("ix_tool_pattern_learning_candidates_promotion_status", ["promotion_status"]),
            ("ix_tool_pattern_learning_candidates_global_promotion_allowed", ["global_promotion_allowed"]),
            ("ix_tool_pattern_learning_candidates_dedupe_signature", ["dedupe_signature"]),
            ("ix_tool_pattern_learning_candidates_replacement_global_pattern_id", ["replacement_global_pattern_id"]),
            ("ix_tool_pattern_learning_candidates_contract_quality", ["contract_quality"]),
            ("ix_tool_pattern_learning_candidates_last_seen_at", ["last_seen_at"]),
            ("ix_tool_pattern_learning_candidates_workspace_status", ["workspace_id", "promotion_status"]),
            ("ix_tool_pattern_learning_candidates_workspace_capability", ["workspace_id", "capability_key"]),
        ),
    )


def downgrade() -> None:
    if _has_table("tool_pattern_learning_candidates"):
        op.drop_table("tool_pattern_learning_candidates")
