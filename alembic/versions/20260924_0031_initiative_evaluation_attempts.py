"""Persist landing initiative evaluation attempts.

Revision ID: 20260924_0031
Revises: 20260923_0030
Create Date: 2026-09-24 09:30:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260924_0031"
down_revision = "20260923_0030"
branch_labels = None
depends_on = None


def _has_table(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return table_name in inspector.get_table_names()


def _json_type() -> sa.types.TypeEngine:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        return postgresql.JSONB()
    return sa.JSON()


def _uuid_type() -> sa.types.TypeEngine:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        return postgresql.UUID(as_uuid=True)
    return sa.String(length=36)


def upgrade() -> None:
    if _has_table("initiative_evaluation_attempts"):
        return

    uuid = _uuid_type()
    op.create_table(
        "initiative_evaluation_attempts",
        sa.Column("id", uuid, primary_key=True, nullable=False),
        sa.Column("evaluation_id", sa.String(), nullable=False),
        sa.Column("input_hash", sa.String(), nullable=False),
        sa.Column("normalized_text", sa.String(length=4000), nullable=False),
        sa.Column("initiative_text", sa.String(length=4000), nullable=False),
        sa.Column("language", sa.String(), nullable=False),
        sa.Column("input_type", sa.String(), nullable=False),
        sa.Column("example_id", sa.String(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("readiness_score", sa.Integer(), nullable=False),
        sa.Column("verdict_badge", sa.String(), nullable=False),
        sa.Column("suggested_archetype", sa.String(), nullable=False),
        sa.Column("suggested_tier", sa.String(), nullable=False),
        sa.Column("operational_profile", _json_type(), nullable=False),
        sa.Column("result_payload", _json_type(), nullable=False),
        sa.Column("token_usage", _json_type(), nullable=False),
        sa.Column("submission_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("example_submission_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("custom_submission_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("first_seen_at", sa.DateTime(), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("input_hash", name="uq_initiative_evaluation_attempts_input_hash"),
        sa.UniqueConstraint("evaluation_id", name="uq_initiative_evaluation_attempts_evaluation_id"),
    )
    op.create_index(
        "ix_initiative_evaluation_attempts_created_at",
        "initiative_evaluation_attempts",
        ["created_at"],
    )
    op.create_index(
        "ix_initiative_evaluation_attempts_example_id",
        "initiative_evaluation_attempts",
        ["example_id"],
    )
    op.create_index(
        "ix_initiative_evaluation_attempts_input_type",
        "initiative_evaluation_attempts",
        ["input_type"],
    )
    op.create_index(
        "ix_initiative_evaluation_attempts_last_seen_at",
        "initiative_evaluation_attempts",
        ["last_seen_at"],
    )
    op.create_index(
        "ix_initiative_evaluation_attempts_source",
        "initiative_evaluation_attempts",
        ["source"],
    )
    op.create_index(
        "ix_initiative_evaluation_attempts_verdict",
        "initiative_evaluation_attempts",
        ["verdict_badge"],
    )
    op.create_index(
        "ix_initiative_evaluation_attempts_language",
        "initiative_evaluation_attempts",
        ["language"],
    )
    if op.get_bind().dialect.name == "postgresql":
        op.execute("ALTER TABLE initiative_evaluation_attempts ENABLE ROW LEVEL SECURITY")
        op.execute(
            """
            DO $$
            BEGIN
                IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'luism_corredor_lab') THEN
                    GRANT USAGE ON SCHEMA public TO luism_corredor_lab;
                    GRANT SELECT, INSERT, UPDATE, DELETE ON public.initiative_evaluation_attempts TO luism_corredor_lab;
                    CREATE POLICY initiative_evaluation_attempts_backend_access
                        ON public.initiative_evaluation_attempts
                        FOR ALL
                        TO luism_corredor_lab
                        USING (true)
                        WITH CHECK (true);
                END IF;
            END
            $$;
            """
        )


def downgrade() -> None:
    if not _has_table("initiative_evaluation_attempts"):
        return

    op.drop_index("ix_initiative_evaluation_attempts_language", table_name="initiative_evaluation_attempts")
    op.drop_index("ix_initiative_evaluation_attempts_verdict", table_name="initiative_evaluation_attempts")
    op.drop_index("ix_initiative_evaluation_attempts_source", table_name="initiative_evaluation_attempts")
    op.drop_index("ix_initiative_evaluation_attempts_last_seen_at", table_name="initiative_evaluation_attempts")
    op.drop_index("ix_initiative_evaluation_attempts_input_type", table_name="initiative_evaluation_attempts")
    op.drop_index("ix_initiative_evaluation_attempts_example_id", table_name="initiative_evaluation_attempts")
    op.drop_index("ix_initiative_evaluation_attempts_created_at", table_name="initiative_evaluation_attempts")
    op.drop_table("initiative_evaluation_attempts")
