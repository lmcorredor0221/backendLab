from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260805_0004"
down_revision = "20260805_0003"
branch_labels = None
depends_on = None


def _uuid_type() -> sa.types.TypeEngine:
    if op.get_bind().dialect.name == "postgresql":
        return postgresql.UUID(as_uuid=True)
    return sa.String(length=36)


def _has_column(table_name: str, column_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table(table_name):
        return False
    return column_name in {column["name"] for column in inspector.get_columns(table_name)}


def _has_index(table_name: str, index_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table(table_name):
        return False
    return index_name in {index["name"] for index in inspector.get_indexes(table_name)}


def _add_column_if_missing(table_name: str, column: sa.Column) -> None:
    if not _has_column(table_name, column.name):
        op.add_column(table_name, column)


def _create_index_if_missing(index_name: str, table_name: str, columns: list[str]) -> None:
    if not _has_index(table_name, index_name):
        op.create_index(index_name, table_name, columns)


def upgrade() -> None:
    uuid = _uuid_type()
    _add_column_if_missing("sessions", sa.Column("suggested_title", sa.String(), nullable=True))
    _add_column_if_missing(
        "sessions",
        sa.Column("title_source", sa.String(), nullable=False, server_default="migrated"),
    )
    _add_column_if_missing(
        "sessions",
        sa.Column("row_version", sa.Integer(), nullable=False, server_default="1"),
    )
    _add_column_if_missing("sessions", sa.Column("archived_at", sa.DateTime(), nullable=True))
    _add_column_if_missing("sessions", sa.Column("archived_by_user_id", uuid, nullable=True))
    _add_column_if_missing("sessions", sa.Column("deleted_at", sa.DateTime(), nullable=True))
    _add_column_if_missing("sessions", sa.Column("deleted_by_user_id", uuid, nullable=True))

    _create_index_if_missing("ix_sessions_archived_at", "sessions", ["archived_at"])
    _create_index_if_missing("ix_sessions_deleted_at", "sessions", ["deleted_at"])
    _create_index_if_missing("ix_sessions_updated_at", "sessions", ["updated_at"])
    _create_index_if_missing("ix_sessions_status", "sessions", ["status"])
    _create_index_if_missing("ix_sessions_commercial_tier", "sessions", ["commercial_tier"])


def downgrade() -> None:
    for index_name in (
        "ix_sessions_commercial_tier",
        "ix_sessions_status",
        "ix_sessions_updated_at",
        "ix_sessions_deleted_at",
        "ix_sessions_archived_at",
    ):
        if _has_index("sessions", index_name):
            op.drop_index(index_name, table_name="sessions")

    for column_name in (
        "deleted_by_user_id",
        "deleted_at",
        "archived_by_user_id",
        "archived_at",
        "row_version",
        "title_source",
        "suggested_title",
    ):
        if _has_column("sessions", column_name):
            op.drop_column("sessions", column_name)
