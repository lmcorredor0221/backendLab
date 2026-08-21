"""deliverable_governance

Revision ID: 20260810_0008
Revises: 20260810_0007
Create Date: 2026-08-10 18:30:00.000000

"""
from __future__ import annotations

from collections.abc import Iterable

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260810_0008"
down_revision = "20260810_0007"
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

    if not _has_table("deliverable_governance_v1"):
        op.create_table(
            "deliverable_governance_v1",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("scope_key", sa.String(), nullable=False),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=True),
            sa.Column("deliverable_key", sa.String(), nullable=False),
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
            sa.UniqueConstraint("scope_key", "deliverable_key", name="uq_deliverable_governance_scope_key_v1"),
        )
        _create_indexes(
            "deliverable_governance_v1",
            (
                ("ix_deliverable_governance_v1_scope_key", ["scope_key"]),
                ("ix_deliverable_governance_v1_workspace_id", ["workspace_id"]),
                ("ix_deliverable_governance_v1_deliverable_key", ["deliverable_key"]),
                ("ix_deliverable_governance_v1_updated_by_user_id", ["updated_by_user_id"]),
            ),
        )

    if not _has_table("deliverable_governance_audit_v1"):
        op.create_table(
            "deliverable_governance_audit_v1",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("scope_key", sa.String(), nullable=False),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=True),
            sa.Column("deliverable_key", sa.String(), nullable=False),
            sa.Column("action", sa.String(), nullable=False),
            sa.Column("changed_fields", json, nullable=False),
            sa.Column("before_payload", json, nullable=False),
            sa.Column("after_payload", json, nullable=False),
            sa.Column("actor_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("reason", sa.String(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )
        _create_indexes(
            "deliverable_governance_audit_v1",
            (
                ("ix_deliverable_governance_audit_v1_scope_key", ["scope_key"]),
                ("ix_deliverable_governance_audit_v1_workspace_id", ["workspace_id"]),
                ("ix_deliverable_governance_audit_v1_deliverable_key", ["deliverable_key"]),
                ("ix_deliverable_governance_audit_v1_action", ["action"]),
                ("ix_deliverable_governance_audit_v1_actor_user_id", ["actor_user_id"]),
                ("ix_deliverable_governance_audit_v1_created_at", ["created_at"]),
            ),
        )

    if not _has_table("deliverable_prompt_versions_v1"):
        op.create_table(
            "deliverable_prompt_versions_v1",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("scope_key", sa.String(), nullable=False),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=True),
            sa.Column("deliverable_key", sa.String(), nullable=False),
            sa.Column("version", sa.String(), nullable=False),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("prompt_template_key", sa.String(), nullable=False),
            sa.Column("prompt_body", sa.String(), nullable=False),
            sa.Column("schema_contract", sa.String(), nullable=False),
            sa.Column("validator_key", sa.String(), nullable=False),
            sa.Column("fallback_policy", sa.String(), nullable=False),
            sa.Column("metadata", json, nullable=False),
            sa.Column("created_by_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("scope_key", "deliverable_key", "version", name="uq_deliverable_prompt_scope_version_v1"),
        )
        _create_indexes(
            "deliverable_prompt_versions_v1",
            (
                ("ix_deliverable_prompt_versions_v1_scope_key", ["scope_key"]),
                ("ix_deliverable_prompt_versions_v1_workspace_id", ["workspace_id"]),
                ("ix_deliverable_prompt_versions_v1_deliverable_key", ["deliverable_key"]),
                ("ix_deliverable_prompt_versions_v1_status", ["status"]),
                ("ix_deliverable_prompt_versions_v1_prompt_template_key", ["prompt_template_key"]),
                ("ix_deliverable_prompt_versions_v1_created_by_user_id", ["created_by_user_id"]),
            ),
        )

    if not _has_table("deliverable_generation_jobs_v1"):
        op.create_table(
            "deliverable_generation_jobs_v1",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
            sa.Column("session_id", uuid, sa.ForeignKey("sessions.id"), nullable=False),
            sa.Column("deliverable_key", sa.String(), nullable=False),
            sa.Column("requested_by_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("product_mode", sa.String(), nullable=False),
            sa.Column("generation_mode", sa.String(), nullable=False),
            sa.Column("idempotency_key", sa.String(), nullable=False),
            sa.Column("provider_key", sa.String(), nullable=False),
            sa.Column("model_name", sa.String(), nullable=False),
            sa.Column("prompt_version_id", uuid, sa.ForeignKey("deliverable_prompt_versions_v1.id"), nullable=True),
            sa.Column("output_version_id", uuid, nullable=True),
            sa.Column("error_code", sa.String(), nullable=False),
            sa.Column("error_message", sa.String(), nullable=False),
            sa.Column("tokens_input", sa.Integer(), nullable=False),
            sa.Column("tokens_output", sa.Integer(), nullable=False),
            sa.Column("estimated_cost_usd", sa.Float(), nullable=False),
            sa.Column("request_metadata", json, nullable=False),
            sa.Column("requested_at", sa.DateTime(), nullable=False),
            sa.Column("started_at", sa.DateTime(), nullable=True),
            sa.Column("completed_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("workspace_id", "idempotency_key", name="uq_deliverable_job_workspace_idempotency_v1"),
        )
        _create_indexes(
            "deliverable_generation_jobs_v1",
            (
                ("ix_deliverable_generation_jobs_v1_workspace_id", ["workspace_id"]),
                ("ix_deliverable_generation_jobs_v1_session_id", ["session_id"]),
                ("ix_deliverable_generation_jobs_v1_deliverable_key", ["deliverable_key"]),
                ("ix_deliverable_generation_jobs_v1_requested_by_user_id", ["requested_by_user_id"]),
                ("ix_deliverable_generation_jobs_v1_status", ["status"]),
                ("ix_deliverable_generation_jobs_v1_product_mode", ["product_mode"]),
                ("ix_deliverable_generation_jobs_v1_generation_mode", ["generation_mode"]),
                ("ix_deliverable_generation_jobs_v1_idempotency_key", ["idempotency_key"]),
                ("ix_deliverable_generation_jobs_v1_provider_key", ["provider_key"]),
                ("ix_deliverable_generation_jobs_v1_prompt_version_id", ["prompt_version_id"]),
                ("ix_deliverable_generation_jobs_v1_output_version_id", ["output_version_id"]),
            ),
        )

    if not _has_table("deliverable_quality_snapshots_v1"):
        op.create_table(
            "deliverable_quality_snapshots_v1",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
            sa.Column("session_id", uuid, sa.ForeignKey("sessions.id"), nullable=False),
            sa.Column("deliverable_key", sa.String(), nullable=False),
            sa.Column("version_ref", sa.String(), nullable=False),
            sa.Column("state", sa.String(), nullable=False),
            sa.Column("score", sa.Integer(), nullable=False),
            sa.Column("errors", json, nullable=False),
            sa.Column("warnings", json, nullable=False),
            sa.Column("checks", json, nullable=False),
            sa.Column("source_fingerprint", sa.String(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )
        _create_indexes(
            "deliverable_quality_snapshots_v1",
            (
                ("ix_deliverable_quality_snapshots_v1_workspace_id", ["workspace_id"]),
                ("ix_deliverable_quality_snapshots_v1_session_id", ["session_id"]),
                ("ix_deliverable_quality_snapshots_v1_deliverable_key", ["deliverable_key"]),
                ("ix_deliverable_quality_snapshots_v1_version_ref", ["version_ref"]),
                ("ix_deliverable_quality_snapshots_v1_state", ["state"]),
                ("ix_deliverable_quality_snapshots_v1_source_fingerprint", ["source_fingerprint"]),
                ("ix_deliverable_quality_snapshots_v1_created_at", ["created_at"]),
            ),
        )

    if not _has_table("deliverable_prompt_audit_v1"):
        op.create_table(
            "deliverable_prompt_audit_v1",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("scope_key", sa.String(), nullable=False),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=True),
            sa.Column("deliverable_key", sa.String(), nullable=False),
            sa.Column("prompt_version_id", uuid, sa.ForeignKey("deliverable_prompt_versions_v1.id"), nullable=True),
            sa.Column("action", sa.String(), nullable=False),
            sa.Column("changed_fields", json, nullable=False),
            sa.Column("before_payload", json, nullable=False),
            sa.Column("after_payload", json, nullable=False),
            sa.Column("actor_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("reason", sa.String(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )
        _create_indexes(
            "deliverable_prompt_audit_v1",
            (
                ("ix_deliverable_prompt_audit_v1_scope_key", ["scope_key"]),
                ("ix_deliverable_prompt_audit_v1_workspace_id", ["workspace_id"]),
                ("ix_deliverable_prompt_audit_v1_deliverable_key", ["deliverable_key"]),
                ("ix_deliverable_prompt_audit_v1_prompt_version_id", ["prompt_version_id"]),
                ("ix_deliverable_prompt_audit_v1_action", ["action"]),
                ("ix_deliverable_prompt_audit_v1_actor_user_id", ["actor_user_id"]),
                ("ix_deliverable_prompt_audit_v1_created_at", ["created_at"]),
            ),
        )


def downgrade() -> None:
    for table_name in (
        "deliverable_prompt_audit_v1",
        "deliverable_quality_snapshots_v1",
        "deliverable_generation_jobs_v1",
        "deliverable_prompt_versions_v1",
        "deliverable_governance_audit_v1",
        "deliverable_governance_v1",
    ):
        if _has_table(table_name):
            op.drop_table(table_name)
