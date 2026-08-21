"""product_build_stage_operations

Revision ID: 20260815_0013
Revises: 20260813_0012
Create Date: 2026-08-15 00:13:00.000000

"""
from __future__ import annotations

from collections.abc import Iterable

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260815_0013"
down_revision = "20260813_0012"
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


def _columns(table_name: str) -> set[str]:
    return {column["name"] for column in sa.inspect(op.get_bind()).get_columns(table_name)}


def _index_names(table_name: str) -> set[str]:
    return {index["name"] for index in sa.inspect(op.get_bind()).get_indexes(table_name)}


def _unique_constraint_names(table_name: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    names = {constraint["name"] for constraint in inspector.get_unique_constraints(table_name) if constraint.get("name")}
    names.update(index["name"] for index in inspector.get_indexes(table_name) if index.get("unique") and index.get("name"))
    return names


def _row_count(table_name: str) -> int:
    result = op.get_bind().execute(sa.text(f'SELECT COUNT(*) FROM "{table_name}"'))
    return int(result.scalar_one())


def _create_indexes(table_name: str, indexes: Iterable[tuple[str, list[str]]]) -> None:
    existing = _index_names(table_name)
    for index_name, columns in indexes:
        if index_name not in existing:
            op.create_index(index_name, table_name, columns)


def _ensure_column(table_name: str, column: sa.Column, *, existing_rows_default: object | None = None) -> None:
    if column.name in _columns(table_name):
        return
    add_column = column
    if op.get_bind().dialect.name == "sqlite" and column.foreign_keys:
        add_column = sa.Column(column.name, column.type, nullable=column.nullable)
    if not column.nullable and _row_count(table_name) > 0 and existing_rows_default is not None:
        add_column.server_default = sa.DefaultClause(existing_rows_default)
    op.add_column(table_name, add_column)


def _assert_no_duplicates(table_name: str, columns: list[str], constraint_name: str) -> None:
    quoted_columns = ", ".join(f'"{column}"' for column in columns)
    result = op.get_bind().execute(
        sa.text(
            f'SELECT {quoted_columns}, COUNT(*) AS duplicate_count '
            f'FROM "{table_name}" '
            f"GROUP BY {quoted_columns} "
            "HAVING COUNT(*) > 1 "
            "LIMIT 1"
        )
    )
    duplicate = result.first()
    if duplicate is not None:
        raise RuntimeError(
            f"No se puede crear {constraint_name}: existen duplicados en {table_name} para {', '.join(columns)}."
        )


def _ensure_unique_constraint(table_name: str, name: str, columns: list[str]) -> None:
    if name in _unique_constraint_names(table_name):
        return
    _assert_no_duplicates(table_name, columns, name)
    if op.get_bind().dialect.name == "sqlite":
        op.create_index(name, table_name, columns, unique=True)
    else:
        op.create_unique_constraint(name, table_name, columns)


def _drop_index_if_exists(table_name: str, index_name: str) -> None:
    if index_name in _index_names(table_name):
        op.drop_index(index_name, table_name=table_name)


def _drop_unique_if_exists(table_name: str, name: str) -> None:
    if name not in _unique_constraint_names(table_name):
        return
    if op.get_bind().dialect.name == "sqlite":
        _drop_index_if_exists(table_name, name)
    else:
        op.drop_constraint(name, table_name, type_="unique")


def _product_build_runs_columns() -> list[sa.Column]:
    uuid = _uuid_type()
    json = _json_type()
    return [
        sa.Column("id", uuid, primary_key=True),
        sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
        sa.Column("session_id", uuid, sa.ForeignKey("sessions.id"), nullable=False),
        sa.Column("product_key", sa.String(), nullable=False),
        sa.Column("product_mode", sa.String(), nullable=False),
        sa.Column("entitlement_tier", sa.String(), nullable=False),
        sa.Column("access_state", sa.String(), nullable=False),
        sa.Column("lifecycle", sa.String(), nullable=False),
        sa.Column("progress_percent", sa.Integer(), nullable=False),
        sa.Column("completed_units", sa.Float(), nullable=False),
        sa.Column("total_units", sa.Float(), nullable=False),
        sa.Column("blocked_units", sa.Float(), nullable=False),
        sa.Column("idempotency_key", sa.String(), nullable=False),
        sa.Column("checkpoint_payload", json, nullable=False),
        sa.Column("error_payload", json, nullable=False),
        sa.Column("created_by_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("requires_attention_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    ]


def _product_build_steps_columns() -> list[sa.Column]:
    uuid = _uuid_type()
    json = _json_type()
    return [
        sa.Column("id", uuid, primary_key=True),
        sa.Column("run_id", uuid, sa.ForeignKey("product_build_runs_v1.id"), nullable=False),
        sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
        sa.Column("session_id", uuid, sa.ForeignKey("sessions.id"), nullable=False),
        sa.Column("step_key", sa.String(), nullable=False),
        sa.Column("stage_key", sa.String(), nullable=False),
        sa.Column("deliverable_key", sa.String(), nullable=False),
        sa.Column("job_id", uuid, nullable=True),
        sa.Column("dependency_key", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("progress_percent", sa.Integer(), nullable=False),
        sa.Column("checkpoint_payload", json, nullable=False),
        sa.Column("error_payload", json, nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    ]


def _stage_operations_columns() -> list[sa.Column]:
    uuid = _uuid_type()
    json = _json_type()
    return [
        sa.Column("id", uuid, primary_key=True),
        sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
        sa.Column("session_id", uuid, sa.ForeignKey("sessions.id"), nullable=False),
        sa.Column("user_id", uuid, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("stage_key", sa.String(), nullable=False),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("status", sa.String(length=9), nullable=False),
        sa.Column("current_step", sa.String(), nullable=False),
        sa.Column("detail", sa.String(), nullable=False),
        sa.Column("request_payload", json, nullable=False),
        sa.Column("steps", json, nullable=False),
        sa.Column("result_artifact_id", uuid, sa.ForeignKey("journey_stage_artifacts.id"), nullable=True),
        sa.Column("error_message", sa.String(), nullable=False),
        sa.Column("technical_detail", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
    ]


RUN_INDEXES = (
    ("ix_product_build_runs_v1_workspace_id", ["workspace_id"]),
    ("ix_product_build_runs_v1_session_id", ["session_id"]),
    ("ix_product_build_runs_v1_product_key", ["product_key"]),
    ("ix_product_build_runs_v1_product_mode", ["product_mode"]),
    ("ix_product_build_runs_v1_entitlement_tier", ["entitlement_tier"]),
    ("ix_product_build_runs_v1_access_state", ["access_state"]),
    ("ix_product_build_runs_v1_lifecycle", ["lifecycle"]),
    ("ix_product_build_runs_v1_idempotency_key", ["idempotency_key"]),
    ("ix_product_build_runs_v1_created_by_user_id", ["created_by_user_id"]),
)

STEP_INDEXES = (
    ("ix_product_build_steps_v1_run_id", ["run_id"]),
    ("ix_product_build_steps_v1_workspace_id", ["workspace_id"]),
    ("ix_product_build_steps_v1_session_id", ["session_id"]),
    ("ix_product_build_steps_v1_step_key", ["step_key"]),
    ("ix_product_build_steps_v1_stage_key", ["stage_key"]),
    ("ix_product_build_steps_v1_deliverable_key", ["deliverable_key"]),
    ("ix_product_build_steps_v1_job_id", ["job_id"]),
    ("ix_product_build_steps_v1_dependency_key", ["dependency_key"]),
    ("ix_product_build_steps_v1_status", ["status"]),
)

STAGE_OPERATION_INDEXES = (
    ("ix_stage_operations_workspace_id", ["workspace_id"]),
    ("ix_stage_operations_session_id", ["session_id"]),
    ("ix_stage_operations_user_id", ["user_id"]),
    ("ix_stage_operations_stage_key", ["stage_key"]),
    ("ix_stage_operations_action", ["action"]),
    ("ix_stage_operations_status", ["status"]),
    ("ix_stage_operations_result_artifact_id", ["result_artifact_id"]),
)

RUN_UNIQUE = ("uq_product_build_run_workspace_idempotency_v1", ["workspace_id", "idempotency_key"])
STEP_UNIQUE = ("uq_product_build_step_run_key_v1", ["run_id", "step_key"])


DEFAULTS_BY_COLUMN = {
    "entitlement_tier": "'blueprint'",
    "access_state": "'preview'",
    "lifecycle": "'ready_to_start'",
    "progress_percent": "0",
    "completed_units": "0",
    "total_units": "0",
    "blocked_units": "0",
    "checkpoint_payload": "'{}'",
    "error_payload": "'{}'",
    "stage_key": "''",
    "deliverable_key": "''",
    "dependency_key": "''",
    "status": "'queued'",
    "sequence": "0",
    "current_step": "''",
    "detail": "''",
    "request_payload": "'{}'",
    "steps": "'[]'",
    "error_message": "''",
    "technical_detail": "''",
    "created_at": "'1970-01-01 00:00:00'",
    "updated_at": "'1970-01-01 00:00:00'",
}


def _default_for(column_name: str) -> sa.TextClause | None:
    value = DEFAULTS_BY_COLUMN.get(column_name)
    if value is None:
        return None
    return sa.text(value)


def _ensure_table(
    table_name: str,
    columns: list[sa.Column],
    indexes: Iterable[tuple[str, list[str]]],
    uniques: Iterable[tuple[str, list[str]]] = (),
) -> None:
    if not _has_table(table_name):
        op.create_table(
            table_name,
            *columns,
            *(sa.UniqueConstraint(*unique_columns, name=unique_name) for unique_name, unique_columns in uniques),
        )
    else:
        for column in columns:
            _ensure_column(table_name, column, existing_rows_default=_default_for(column.name))

    for unique_name, unique_columns in uniques:
        _ensure_unique_constraint(table_name, unique_name, unique_columns)
    _create_indexes(table_name, indexes)


def upgrade() -> None:
    _ensure_table(
        "product_build_runs_v1",
        _product_build_runs_columns(),
        RUN_INDEXES,
        (RUN_UNIQUE,),
    )
    _ensure_table(
        "product_build_steps_v1",
        _product_build_steps_columns(),
        STEP_INDEXES,
        (STEP_UNIQUE,),
    )
    _ensure_table(
        "stage_operations",
        _stage_operations_columns(),
        STAGE_OPERATION_INDEXES,
    )


def downgrade() -> None:
    if _has_table("stage_operations"):
        for index_name, _columns_to_drop in reversed(STAGE_OPERATION_INDEXES):
            _drop_index_if_exists("stage_operations", index_name)
        op.drop_table("stage_operations")

    if _has_table("product_build_steps_v1"):
        _drop_unique_if_exists("product_build_steps_v1", STEP_UNIQUE[0])
        for index_name, _columns_to_drop in reversed(STEP_INDEXES):
            _drop_index_if_exists("product_build_steps_v1", index_name)
        op.drop_table("product_build_steps_v1")

    if _has_table("product_build_runs_v1"):
        _drop_unique_if_exists("product_build_runs_v1", RUN_UNIQUE[0])
        for index_name, _columns_to_drop in reversed(RUN_INDEXES):
            _drop_index_if_exists("product_build_runs_v1", index_name)
        op.drop_table("product_build_runs_v1")
