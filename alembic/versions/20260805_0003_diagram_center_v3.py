from __future__ import annotations

from collections.abc import Iterable

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260805_0003"
down_revision = "20260802_0002"
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
    uuid = _uuid_type()
    json = _json_type()
    if not _has_table("diagram_generation_jobs_v3"):
        op.create_table(
            "diagram_generation_jobs_v3",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
            sa.Column("session_id", uuid, sa.ForeignKey("sessions.id"), nullable=False),
            sa.Column("diagram_key", sa.String(), nullable=False),
            sa.Column("requested_by_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("detail_level", sa.String(), nullable=False),
            sa.Column("reason", sa.String(), nullable=False),
            sa.Column("idempotency_key", sa.String(), nullable=False),
            sa.Column("provider_key", sa.String(), nullable=False),
            sa.Column("model_name", sa.String(), nullable=False),
            sa.Column("prompt_spec_version", sa.String(), nullable=False),
            sa.Column("version_id", uuid, nullable=True),
            sa.Column("error_code", sa.String(), nullable=False),
            sa.Column("error_message", sa.String(), nullable=False),
            sa.Column("request_metadata", json, nullable=False),
            sa.Column("requested_at", sa.DateTime(), nullable=False),
            sa.Column("started_at", sa.DateTime(), nullable=True),
            sa.Column("completed_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("workspace_id", "idempotency_key", name="uq_diagram_job_workspace_idempotency"),
        )
        _create_indexes(
            "diagram_generation_jobs_v3",
            (
                ("ix_diagram_generation_jobs_v3_workspace_id", ["workspace_id"]),
                ("ix_diagram_generation_jobs_v3_session_id", ["session_id"]),
                ("ix_diagram_generation_jobs_v3_diagram_key", ["diagram_key"]),
                ("ix_diagram_generation_jobs_v3_status", ["status"]),
                ("ix_diagram_generation_jobs_v3_idempotency_key", ["idempotency_key"]),
                ("ix_diagram_generation_jobs_v3_provider_key", ["provider_key"]),
                ("ix_diagram_generation_jobs_v3_version_id", ["version_id"]),
            ),
        )

    if not _has_table("diagram_versions_v3"):
        op.create_table(
            "diagram_versions_v3",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
            sa.Column("session_id", uuid, sa.ForeignKey("sessions.id"), nullable=False),
            sa.Column("diagram_key", sa.String(), nullable=False),
            sa.Column("job_id", uuid, sa.ForeignKey("diagram_generation_jobs_v3.id"), nullable=True),
            sa.Column("version_number", sa.Integer(), nullable=False),
            sa.Column("state", sa.String(), nullable=False),
            sa.Column("diagram_model", json, nullable=False),
            sa.Column("renderings", json, nullable=False),
            sa.Column("quality_report", json, nullable=False),
            sa.Column("source_fingerprint", sa.String(), nullable=False),
            sa.Column("source_refs", json, nullable=False),
            sa.Column("provider_key", sa.String(), nullable=False),
            sa.Column("model_name", sa.String(), nullable=False),
            sa.Column("prompt_spec_version", sa.String(), nullable=False),
            sa.Column("request_id", sa.String(), nullable=False),
            sa.Column("created_by_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("session_id", "diagram_key", "version_number", name="uq_diagram_version_number_v3"),
        )
        _create_indexes(
            "diagram_versions_v3",
            (
                ("ix_diagram_versions_v3_workspace_id", ["workspace_id"]),
                ("ix_diagram_versions_v3_session_id", ["session_id"]),
                ("ix_diagram_versions_v3_diagram_key", ["diagram_key"]),
                ("ix_diagram_versions_v3_job_id", ["job_id"]),
                ("ix_diagram_versions_v3_state", ["state"]),
                ("ix_diagram_versions_v3_source_fingerprint", ["source_fingerprint"]),
                ("ix_diagram_versions_v3_provider_key", ["provider_key"]),
            ),
        )

    if not _has_table("diagram_governance_v3"):
        op.create_table(
            "diagram_governance_v3",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("diagram_key", sa.String(), nullable=False),
            sa.Column("enabled", sa.Boolean(), nullable=False),
            sa.Column("generation_enabled", sa.Boolean(), nullable=False),
            sa.Column("required_tier_override", sa.String(), nullable=False),
            sa.Column("preview_mode_override", sa.String(), nullable=False),
            sa.Column("prompt_status", sa.String(), nullable=False),
            sa.Column("prompt_override", json, nullable=False),
            sa.Column("notes", sa.String(), nullable=False),
            sa.Column("updated_by_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("diagram_key", name="uq_diagram_governance_key_v3"),
        )
        op.create_index("ix_diagram_governance_v3_diagram_key", "diagram_governance_v3", ["diagram_key"])


def downgrade() -> None:
    for table_name in ("diagram_versions_v3", "diagram_generation_jobs_v3", "diagram_governance_v3"):
        if _has_table(table_name):
            op.drop_table(table_name)

