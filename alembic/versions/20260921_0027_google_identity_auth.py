"""google identity authentication

Revision ID: 20260921_0027
Revises: 20260916_0026
Create Date: 2026-09-21 13:00:00.000000
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260921_0027"
down_revision = "20260916_0026"
branch_labels = None
depends_on = None


def _has_table(table_name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(table_name)


def _uuid_type() -> sa.types.TypeEngine:
    if op.get_bind().dialect.name == "postgresql":
        return postgresql.UUID(as_uuid=True)
    return sa.String(length=36)


def upgrade() -> None:
    if _has_table("users"):
        op.alter_column("users", "password_hash", existing_type=sa.String(), nullable=True)

    if not _has_table("user_auth_identities"):
        uuid = _uuid_type()
        op.create_table(
            "user_auth_identities",
            sa.Column("id", uuid, primary_key=True, nullable=False),
            sa.Column("user_id", uuid, nullable=False),
            sa.Column("provider", sa.String(), nullable=False),
            sa.Column("provider_subject", sa.String(), nullable=False),
            sa.Column("email_at_link", sa.String(), nullable=False, server_default=""),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("last_login_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
            sa.UniqueConstraint("provider", "provider_subject", name="uq_user_auth_identity_provider_subject"),
            sa.UniqueConstraint("user_id", "provider", name="uq_user_auth_identity_user_provider"),
        )
        op.create_index("ix_user_auth_identities_provider", "user_auth_identities", ["provider"])
        op.create_index("ix_user_auth_identities_provider_subject", "user_auth_identities", ["provider_subject"])
        op.create_index("ix_user_auth_identities_user_id", "user_auth_identities", ["user_id"])

    if op.get_bind().dialect.name == "postgresql" and _has_table("user_auth_identities"):
        op.execute("ALTER TABLE user_auth_identities ENABLE ROW LEVEL SECURITY")


def downgrade() -> None:
    if _has_table("user_auth_identities"):
        op.drop_index("ix_user_auth_identities_user_id", table_name="user_auth_identities")
        op.drop_index("ix_user_auth_identities_provider_subject", table_name="user_auth_identities")
        op.drop_index("ix_user_auth_identities_provider", table_name="user_auth_identities")
        op.drop_table("user_auth_identities")

    if _has_table("users"):
        op.execute("UPDATE users SET password_hash = 'google-only-disabled' WHERE password_hash IS NULL")
        op.alter_column("users", "password_hash", existing_type=sa.String(), nullable=False)
