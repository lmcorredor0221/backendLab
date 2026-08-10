from __future__ import annotations

from collections.abc import Iterable

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260802_0002"
down_revision = "20260802_0001"
branch_labels = None
depends_on = None


def _uuid_type() -> sa.types.TypeEngine:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        return postgresql.UUID(as_uuid=True)
    return sa.String(length=36)


def _json_type() -> sa.types.TypeEngine:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
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

    if not _has_table("acp_build_runs"):
        op.create_table(
            "acp_build_runs",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
            sa.Column("session_id", uuid, sa.ForeignKey("sessions.id"), nullable=False),
            sa.Column("created_by_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("blueprint_version_number", sa.Integer(), nullable=True),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("current_phase_key", sa.String(), nullable=False),
            sa.Column("phase_order", json, nullable=False),
            sa.Column("progress_percent", sa.Integer(), nullable=False),
            sa.Column("checkpoints", json, nullable=False),
            sa.Column("artifacts", json, nullable=False),
            sa.Column("blockers", json, nullable=False),
            sa.Column("warnings", json, nullable=False),
            sa.Column("idempotency_key", sa.String(), nullable=False),
            sa.Column("started_at", sa.DateTime(), nullable=True),
            sa.Column("completed_at", sa.DateTime(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("workspace_id", "session_id", "idempotency_key", name="uq_acp_build_run_idempotency"),
        )
        _create_indexes(
            "acp_build_runs",
            (
                ("ix_acp_build_runs_workspace_id", ["workspace_id"]),
                ("ix_acp_build_runs_session_id", ["session_id"]),
                ("ix_acp_build_runs_status", ["status"]),
                ("ix_acp_build_runs_current_phase_key", ["current_phase_key"]),
                ("ix_acp_build_runs_idempotency_key", ["idempotency_key"]),
            ),
        )

    if not _has_table("acp_phase_runs"):
        op.create_table(
            "acp_phase_runs",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("run_id", uuid, sa.ForeignKey("acp_build_runs.id"), nullable=False),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
            sa.Column("session_id", uuid, sa.ForeignKey("sessions.id"), nullable=False),
            sa.Column("phase_key", sa.String(), nullable=False),
            sa.Column("phase_label", sa.String(), nullable=False),
            sa.Column("phase_order", sa.Integer(), nullable=False),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("attempt_count", sa.Integer(), nullable=False),
            sa.Column("input_refs", json, nullable=False),
            sa.Column("output_refs", json, nullable=False),
            sa.Column("checkpoints", json, nullable=False),
            sa.Column("blockers", json, nullable=False),
            sa.Column("warnings", json, nullable=False),
            sa.Column("idempotency_key", sa.String(), nullable=False),
            sa.Column("started_at", sa.DateTime(), nullable=True),
            sa.Column("completed_at", sa.DateTime(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("run_id", "phase_key", name="uq_acp_phase_run_phase"),
        )
        _create_indexes(
            "acp_phase_runs",
            (
                ("ix_acp_phase_runs_run_id", ["run_id"]),
                ("ix_acp_phase_runs_workspace_id", ["workspace_id"]),
                ("ix_acp_phase_runs_session_id", ["session_id"]),
                ("ix_acp_phase_runs_phase_key", ["phase_key"]),
                ("ix_acp_phase_runs_status", ["status"]),
                ("ix_acp_phase_runs_idempotency_key", ["idempotency_key"]),
            ),
        )

    if not _has_table("export_jobs"):
        op.create_table(
            "export_jobs",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
            sa.Column("session_id", uuid, sa.ForeignKey("sessions.id"), nullable=False),
            sa.Column("user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("product_key", sa.String(), nullable=False),
            sa.Column("profile", sa.String(), nullable=False),
            sa.Column("artifact_kind", sa.String(), nullable=False),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("idempotency_key", sa.String(), nullable=False),
            sa.Column("content_type", sa.String(), nullable=False),
            sa.Column("file_name", sa.String(), nullable=False),
            sa.Column("storage_key", sa.String(), nullable=False),
            sa.Column("checksum_sha256", sa.String(), nullable=False),
            sa.Column("size_bytes", sa.Integer(), nullable=False),
            sa.Column("expires_at", sa.DateTime(), nullable=True),
            sa.Column("error_message", sa.String(), nullable=False),
            sa.Column("metadata", json, nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.Column("completed_at", sa.DateTime(), nullable=True),
            sa.UniqueConstraint("workspace_id", "session_id", "idempotency_key", name="uq_export_job_idempotency"),
        )
        _create_indexes(
            "export_jobs",
            (
                ("ix_export_jobs_workspace_id", ["workspace_id"]),
                ("ix_export_jobs_session_id", ["session_id"]),
                ("ix_export_jobs_product_key", ["product_key"]),
                ("ix_export_jobs_profile", ["profile"]),
                ("ix_export_jobs_artifact_kind", ["artifact_kind"]),
                ("ix_export_jobs_status", ["status"]),
                ("ix_export_jobs_idempotency_key", ["idempotency_key"]),
                ("ix_export_jobs_storage_key", ["storage_key"]),
            ),
        )

    if not _has_table("acp_launch_reports"):
        op.create_table(
            "acp_launch_reports",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
            sa.Column("session_id", uuid, sa.ForeignKey("sessions.id"), nullable=False),
            sa.Column("user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("report_path", sa.String(), nullable=False),
            sa.Column("launcher_version", sa.String(), nullable=False),
            sa.Column("detected_tool", sa.String(), nullable=False),
            sa.Column("detected_ide", sa.String(), nullable=False),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("summary", sa.String(), nullable=False),
            sa.Column("report", json, nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )
        _create_indexes(
            "acp_launch_reports",
            (
                ("ix_acp_launch_reports_workspace_id", ["workspace_id"]),
                ("ix_acp_launch_reports_session_id", ["session_id"]),
                ("ix_acp_launch_reports_launcher_version", ["launcher_version"]),
                ("ix_acp_launch_reports_detected_tool", ["detected_tool"]),
                ("ix_acp_launch_reports_detected_ide", ["detected_ide"]),
                ("ix_acp_launch_reports_status", ["status"]),
            ),
        )


def downgrade() -> None:
    for table_name in (
        "acp_launch_reports",
        "export_jobs",
        "acp_phase_runs",
        "acp_build_runs",
    ):
        if _has_table(table_name):
            op.drop_table(table_name)
