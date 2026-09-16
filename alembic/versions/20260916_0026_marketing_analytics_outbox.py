"""marketing_analytics_outbox

Revision ID: 20260916_0026
Revises: 20260914_0025
Create Date: 2026-09-16 16:30:00.000000

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260916_0026"
down_revision = "20260914_0025"
branch_labels = None
depends_on = None


def _has_table(table_name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(table_name)


def _has_column(table_name: str, column_name: str) -> bool:
    if not _has_table(table_name):
        return False
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns(table_name)}
    return column_name in columns


def _json_type() -> sa.types.TypeEngine:
    if op.get_bind().dialect.name == "postgresql":
        return postgresql.JSONB()
    return sa.JSON()


def _json_default() -> sa.TextClause:
    if op.get_bind().dialect.name == "postgresql":
        return sa.text("'{}'::jsonb")
    return sa.text("'{}'")


def _uuid_type() -> sa.types.TypeEngine:
    if op.get_bind().dialect.name == "postgresql":
        return postgresql.UUID(as_uuid=True)
    return sa.String(length=36)


def upgrade() -> None:
    if _has_table("sessions") and not _has_column("sessions", "marketing_context"):
        op.add_column(
            "sessions",
            sa.Column("marketing_context", _json_type(), nullable=False, server_default=_json_default()),
        )
        op.alter_column("sessions", "marketing_context", server_default=None)

    if _has_table("commercial_orders") and not _has_column("commercial_orders", "marketing_context"):
        op.add_column(
            "commercial_orders",
            sa.Column("marketing_context", _json_type(), nullable=False, server_default=_json_default()),
        )
        op.alter_column("commercial_orders", "marketing_context", server_default=None)

    if not _has_table("marketing_analytics_outbox"):
        uuid = _uuid_type()
        op.create_table(
            "marketing_analytics_outbox",
            sa.Column("id", uuid, primary_key=True, nullable=False),
            sa.Column("workspace_id", uuid, nullable=True),
            sa.Column("session_id", uuid, nullable=True),
            sa.Column("user_id", uuid, nullable=True),
            sa.Column("order_id", uuid, nullable=True),
            sa.Column("event_name", sa.String(), nullable=False),
            sa.Column("business_key", sa.String(), nullable=False),
            sa.Column("source", sa.String(), nullable=False),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("next_attempt_at", sa.DateTime(), nullable=False),
            sa.Column("last_attempt_at", sa.DateTime(), nullable=True),
            sa.Column("sent_at", sa.DateTime(), nullable=True),
            sa.Column("error_code", sa.String(), nullable=False, server_default=""),
            sa.Column("error_message", sa.String(), nullable=False, server_default=""),
            sa.Column("payload", _json_type(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"]),
            sa.ForeignKeyConstraint(["session_id"], ["sessions.id"]),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
            sa.ForeignKeyConstraint(["order_id"], ["commercial_orders.id"]),
            sa.UniqueConstraint("business_key", name="uq_marketing_analytics_outbox_business_key"),
        )
        op.create_index("ix_marketing_analytics_outbox_event_name", "marketing_analytics_outbox", ["event_name"])
        op.create_index("ix_marketing_analytics_outbox_order_id", "marketing_analytics_outbox", ["order_id"])
        op.create_index("ix_marketing_analytics_outbox_session_id", "marketing_analytics_outbox", ["session_id"])
        op.create_index("ix_marketing_analytics_outbox_status", "marketing_analytics_outbox", ["status"])
        op.create_index(
            "ix_marketing_analytics_outbox_status_next",
            "marketing_analytics_outbox",
            ["status", "next_attempt_at"],
        )
        op.create_index("ix_marketing_analytics_outbox_user_id", "marketing_analytics_outbox", ["user_id"])
        op.create_index("ix_marketing_analytics_outbox_workspace_id", "marketing_analytics_outbox", ["workspace_id"])

    if op.get_bind().dialect.name == "postgresql" and _has_table("marketing_analytics_outbox"):
        op.execute("ALTER TABLE marketing_analytics_outbox ENABLE ROW LEVEL SECURITY")


def downgrade() -> None:
    if _has_table("marketing_analytics_outbox"):
        op.drop_index("ix_marketing_analytics_outbox_workspace_id", table_name="marketing_analytics_outbox")
        op.drop_index("ix_marketing_analytics_outbox_user_id", table_name="marketing_analytics_outbox")
        op.drop_index("ix_marketing_analytics_outbox_status_next", table_name="marketing_analytics_outbox")
        op.drop_index("ix_marketing_analytics_outbox_status", table_name="marketing_analytics_outbox")
        op.drop_index("ix_marketing_analytics_outbox_session_id", table_name="marketing_analytics_outbox")
        op.drop_index("ix_marketing_analytics_outbox_order_id", table_name="marketing_analytics_outbox")
        op.drop_index("ix_marketing_analytics_outbox_event_name", table_name="marketing_analytics_outbox")
        op.drop_table("marketing_analytics_outbox")

    if _has_column("commercial_orders", "marketing_context"):
        op.drop_column("commercial_orders", "marketing_context")
    if _has_column("sessions", "marketing_context"):
        op.drop_column("sessions", "marketing_context")
