"""journey state persistence

Revision ID: 20260828_0018
Revises: 20260824_0017
Create Date: 2026-08-28 08:30:00.000000

"""
from __future__ import annotations

from collections.abc import Iterable

import sqlalchemy as sa
from alembic import op
from alembic.runtime.migration import MigrationContext
from sqlalchemy.dialects import postgresql


revision = "20260828_0018"
down_revision = "20260824_0017"
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
    context = op.get_context()
    if isinstance(context, MigrationContext) and context.as_sql:
        return False
    return sa.inspect(op.get_bind()).has_table(table_name)


def _create_indexes(table_name: str, indexes: Iterable[tuple[str, list[str]]]) -> None:
    for index_name, columns in indexes:
        op.create_index(index_name, table_name, columns)


def upgrade() -> None:
    uuid = _uuid_type()
    json = _json_type()

    if not _has_table("journey_state_current"):
        op.create_table(
            "journey_state_current",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=True),
            sa.Column("session_id", uuid, sa.ForeignKey("sessions.id"), nullable=False),
            sa.Column("state_key", sa.String(), nullable=False),
            sa.Column("substate", sa.String(), nullable=False),
            sa.Column("product_key", sa.String(), nullable=False),
            sa.Column("stage_key", sa.String(), nullable=False),
            sa.Column("progress_percent", sa.Integer(), nullable=False),
            sa.Column("blocking", sa.Boolean(), nullable=False),
            sa.Column("revision", sa.Integer(), nullable=False),
            sa.Column("source_contracts", json, nullable=False),
            sa.Column("state_payload", json, nullable=False),
            sa.Column("last_transition_at", sa.DateTime(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("session_id", name="uq_journey_state_current_session"),
        )
        _create_indexes(
            "journey_state_current",
            (
                ("ix_journey_state_current_workspace_id", ["workspace_id"]),
                ("ix_journey_state_current_session_id", ["session_id"]),
                ("ix_journey_state_current_state_key", ["state_key"]),
                ("ix_journey_state_current_substate", ["substate"]),
                ("ix_journey_state_current_last_transition_at", ["last_transition_at"]),
            ),
        )

    if not _has_table("journey_state_transitions"):
        op.create_table(
            "journey_state_transitions",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=True),
            sa.Column("session_id", uuid, sa.ForeignKey("sessions.id"), nullable=False),
            sa.Column("sequence", sa.Integer(), nullable=False),
            sa.Column("event_key", sa.String(), nullable=False),
            sa.Column("from_state_key", sa.String(), nullable=False),
            sa.Column("from_substate", sa.String(), nullable=False),
            sa.Column("to_state_key", sa.String(), nullable=False),
            sa.Column("to_substate", sa.String(), nullable=False),
            sa.Column("actor_type", sa.String(), nullable=False),
            sa.Column("actor_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("reason", sa.String(), nullable=False),
            sa.Column("correlation_id", sa.String(), nullable=False),
            sa.Column("transition_payload", json, nullable=False),
            sa.Column("occurred_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("session_id", "sequence", name="uq_journey_state_transition_sequence"),
            sa.UniqueConstraint("session_id", "correlation_id", name="uq_journey_state_transition_correlation"),
        )
        _create_indexes(
            "journey_state_transitions",
            (
                ("ix_journey_state_transitions_workspace_id", ["workspace_id"]),
                ("ix_journey_state_transitions_session_id", ["session_id"]),
                ("ix_journey_state_transitions_event_key", ["event_key"]),
                ("ix_journey_state_transitions_actor_user_id", ["actor_user_id"]),
                ("ix_journey_state_transitions_occurred_at", ["occurred_at"]),
            ),
        )


def downgrade() -> None:
    if _has_table("journey_state_transitions"):
        op.drop_table("journey_state_transitions")
    if _has_table("journey_state_current"):
        op.drop_table("journey_state_current")
