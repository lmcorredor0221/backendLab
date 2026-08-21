"""stage_operation_controls

Revision ID: 20260816_0014
Revises: 20260815_0013
Create Date: 2026-08-16 00:14:00.000000

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260816_0014"
down_revision = "20260815_0013"
branch_labels = None
depends_on = None


STAGE_OPERATION_TABLE = "stage_operations"
STAGE_OPERATION_CONTROL_INDEXES = (
    ("ix_stage_operations_idempotency_key", ["idempotency_key"], False),
    ("ix_stage_operations_cancel_requested_at", ["cancel_requested_at"], False),
    ("ix_stage_operations_heartbeat_at", ["heartbeat_at"], False),
    ("ix_stage_operations_expires_at", ["expires_at"], False),
    (
        "uq_stage_operations_workspace_session_action_idempotency",
        ["workspace_id", "session_id", "action", "idempotency_key"],
        True,
    ),
)


def _has_table(table_name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(table_name)


def _columns(table_name: str) -> set[str]:
    return {column["name"] for column in sa.inspect(op.get_bind()).get_columns(table_name)}


def _indexes(table_name: str) -> set[str]:
    return {index["name"] for index in sa.inspect(op.get_bind()).get_indexes(table_name)}


def _add_missing_columns() -> None:
    existing = _columns(STAGE_OPERATION_TABLE)
    with op.batch_alter_table(STAGE_OPERATION_TABLE, reflect_kwargs={"resolve_fks": False}) as batch:
        if "idempotency_key" not in existing:
            batch.add_column(sa.Column("idempotency_key", sa.String(), nullable=True))
        if "attempt_count" not in existing:
            batch.add_column(sa.Column("attempt_count", sa.Integer(), nullable=True))
        if "cancel_requested_at" not in existing:
            batch.add_column(sa.Column("cancel_requested_at", sa.DateTime(), nullable=True))
        if "heartbeat_at" not in existing:
            batch.add_column(sa.Column("heartbeat_at", sa.DateTime(), nullable=True))
        if "expires_at" not in existing:
            batch.add_column(sa.Column("expires_at", sa.DateTime(), nullable=True))


def _backfill_controls() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    if dialect == "postgresql":
        bind.execute(
            sa.text(
                """
                UPDATE stage_operations
                SET idempotency_key = CONCAT('legacy:', id)
                WHERE idempotency_key IS NULL OR idempotency_key = ''
                """
            )
        )
    else:
        bind.execute(
            sa.text(
                """
                UPDATE stage_operations
                SET idempotency_key = 'legacy:' || id
                WHERE idempotency_key IS NULL OR idempotency_key = ''
                """
            )
        )
    bind.execute(sa.text("UPDATE stage_operations SET attempt_count = 1 WHERE attempt_count IS NULL OR attempt_count < 1"))
    bind.execute(sa.text("UPDATE stage_operations SET heartbeat_at = updated_at WHERE heartbeat_at IS NULL"))


def _enforce_not_null() -> None:
    with op.batch_alter_table(STAGE_OPERATION_TABLE, reflect_kwargs={"resolve_fks": False}) as batch:
        batch.alter_column("idempotency_key", existing_type=sa.String(), nullable=False)
        batch.alter_column("attempt_count", existing_type=sa.Integer(), nullable=False)


def _widen_status_for_waiting_state() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.alter_column(
            STAGE_OPERATION_TABLE,
            "status",
            existing_type=sa.String(length=9),
            type_=sa.String(length=32),
            existing_nullable=False,
        )


def _create_indexes() -> None:
    existing = _indexes(STAGE_OPERATION_TABLE)
    for index_name, columns, unique in STAGE_OPERATION_CONTROL_INDEXES:
        if index_name not in existing:
            op.create_index(index_name, STAGE_OPERATION_TABLE, columns, unique=unique)


def _drop_indexes() -> None:
    existing = _indexes(STAGE_OPERATION_TABLE)
    for index_name, _columns_to_drop, _unique in reversed(STAGE_OPERATION_CONTROL_INDEXES):
        if index_name in existing:
            op.drop_index(index_name, table_name=STAGE_OPERATION_TABLE)


def upgrade() -> None:
    if not _has_table(STAGE_OPERATION_TABLE):
        return

    _add_missing_columns()
    _backfill_controls()
    _enforce_not_null()
    _widen_status_for_waiting_state()
    _create_indexes()


def downgrade() -> None:
    if not _has_table(STAGE_OPERATION_TABLE):
        return

    _drop_indexes()
    with op.batch_alter_table(STAGE_OPERATION_TABLE, reflect_kwargs={"resolve_fks": False}) as batch:
        for column_name in ("expires_at", "heartbeat_at", "cancel_requested_at", "attempt_count", "idempotency_key"):
            if column_name in _columns(STAGE_OPERATION_TABLE):
                batch.drop_column(column_name)

    if op.get_bind().dialect.name == "postgresql":
        op.alter_column(
            STAGE_OPERATION_TABLE,
            "status",
            existing_type=sa.String(length=32),
            type_=sa.String(length=9),
            existing_nullable=False,
        )
