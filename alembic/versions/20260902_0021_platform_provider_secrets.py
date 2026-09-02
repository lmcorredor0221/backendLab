"""platform_provider_secrets

Revision ID: 20260902_0021
Revises: 20260902_0020
Create Date: 2026-09-02 00:10:00.000000

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260902_0021"
down_revision = "20260902_0020"
branch_labels = None
depends_on = None


def _uuid_type() -> sa.types.TypeEngine:
    if op.get_bind().dialect.name == "postgresql":
        return postgresql.UUID(as_uuid=True)
    return sa.String(length=36)


def _has_table(table_name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(table_name)


def _can_create_table() -> bool:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return True
    return bool(
        bind.execute(
            sa.text("select has_schema_privilege(current_user, current_schema(), 'CREATE')")
        ).scalar()
    )


def upgrade() -> None:
    if _has_table("platform_provider_secrets"):
        return
    if not _can_create_table():
        return

    uuid = _uuid_type()
    op.create_table(
        "platform_provider_secrets",
        sa.Column("id", uuid, primary_key=True),
        sa.Column("provider_key", sa.String(), nullable=False),
        sa.Column("secret_kind", sa.String(), nullable=False),
        sa.Column("secret_ciphertext", sa.String(), nullable=False),
        sa.Column("secret_ref", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("last_rotated_at", sa.DateTime(), nullable=True),
        sa.Column("updated_by_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("provider_key", "secret_kind", name="uq_platform_provider_secret"),
    )
    op.create_index("ix_platform_provider_secrets_provider_key", "platform_provider_secrets", ["provider_key"])


def downgrade() -> None:
    if _has_table("platform_provider_secrets"):
        op.drop_table("platform_provider_secrets")
