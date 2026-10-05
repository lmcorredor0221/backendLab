"""Persist deliverable generation cache identity.

Revision ID: 20260926_0033
Revises: 20260925_0032
Create Date: 2026-09-26 10:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260926_0033"
down_revision = "20260925_0032"
branch_labels = None
depends_on = None


TABLE_NAME = "deliverable_generation_jobs_v1"


def _inspector() -> sa.Inspector:
    return sa.inspect(op.get_bind())


def _has_table(table_name: str) -> bool:
    return table_name in _inspector().get_table_names()


def _has_column(table_name: str, column_name: str) -> bool:
    return column_name in {column["name"] for column in _inspector().get_columns(table_name)}


def _has_index(table_name: str, index_name: str) -> bool:
    return index_name in {index["name"] for index in _inspector().get_indexes(table_name)}


def _create_index_if_missing(index_name: str, columns: list[str]) -> None:
    if not _has_index(TABLE_NAME, index_name):
        op.create_index(index_name, TABLE_NAME, columns)


def _drop_index_if_exists(index_name: str) -> None:
    if _has_index(TABLE_NAME, index_name):
        op.drop_index(index_name, table_name=TABLE_NAME)


def upgrade() -> None:
    if not _has_table(TABLE_NAME):
        return

    if not _has_column(TABLE_NAME, "input_fingerprint"):
        op.add_column(TABLE_NAME, sa.Column("input_fingerprint", sa.String(), nullable=True))
    if not _has_column(TABLE_NAME, "builder_version"):
        op.add_column(TABLE_NAME, sa.Column("builder_version", sa.String(), nullable=True))
    if not _has_column(TABLE_NAME, "generation_profile_version"):
        op.add_column(TABLE_NAME, sa.Column("generation_profile_version", sa.String(), nullable=True))

    _create_index_if_missing("ix_deliverable_generation_jobs_v1_input_fingerprint", ["input_fingerprint"])
    _create_index_if_missing("ix_deliverable_generation_jobs_v1_builder_version", ["builder_version"])
    _create_index_if_missing(
        "ix_deliverable_generation_jobs_v1_cache_identity",
        [
            "workspace_id",
            "session_id",
            "deliverable_key",
            "input_fingerprint",
            "builder_version",
            "status",
        ],
    )


def downgrade() -> None:
    if not _has_table(TABLE_NAME):
        return

    _drop_index_if_exists("ix_deliverable_generation_jobs_v1_cache_identity")
    _drop_index_if_exists("ix_deliverable_generation_jobs_v1_builder_version")
    _drop_index_if_exists("ix_deliverable_generation_jobs_v1_input_fingerprint")

    if _has_column(TABLE_NAME, "generation_profile_version"):
        op.drop_column(TABLE_NAME, "generation_profile_version")
    if _has_column(TABLE_NAME, "builder_version"):
        op.drop_column(TABLE_NAME, "builder_version")
    if _has_column(TABLE_NAME, "input_fingerprint"):
        op.drop_column(TABLE_NAME, "input_fingerprint")
