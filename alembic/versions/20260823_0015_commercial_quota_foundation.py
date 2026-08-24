"""commercial_quota_foundation

Revision ID: 20260823_0015
Revises: 20260816_0014
Create Date: 2026-08-23 10:00:00.000000

"""
from __future__ import annotations

from collections.abc import Iterable

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260823_0015"
down_revision = "20260816_0014"
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

    if not _has_table("commercial_quota_product_configs"):
        op.create_table(
            "commercial_quota_product_configs",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("product_key", sa.String(), nullable=False),
            sa.Column("display_name", sa.String(), nullable=False),
            sa.Column("enabled", sa.Boolean(), nullable=False),
            sa.Column("initial_free_units", sa.Integer(), nullable=False),
            sa.Column("consumption_priority", json, nullable=False),
            sa.Column("checkout_required_on_zero_balance", sa.Boolean(), nullable=False),
            sa.Column("fifo_auto_approval_enabled", sa.Boolean(), nullable=False),
            sa.Column("default_blocked_request_ttl_hours", sa.Integer(), nullable=False),
            sa.Column("default_checkout_ttl_minutes", sa.Integer(), nullable=False),
            sa.Column("debt_enabled", sa.Boolean(), nullable=False),
            sa.Column("allow_manual_override_without_charge", sa.Boolean(), nullable=False),
            sa.Column("allow_courtesy", sa.Boolean(), nullable=False),
            sa.Column("allow_debt_pending", sa.Boolean(), nullable=False),
            sa.Column("catalog_priority_strategy", sa.String(), nullable=False),
            sa.Column("sync_retry_limit", sa.Integer(), nullable=False),
            sa.Column("duplicate_conflict_visibility", sa.String(), nullable=False),
            sa.Column("metadata", json, nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("product_key", name="uq_commercial_quota_product_config_product_key"),
        )
        _create_indexes(
            "commercial_quota_product_configs",
            (("ix_commercial_quota_product_configs_product_key", ["product_key"]),),
        )

    if not _has_table("commercial_quota_workspace_overrides"):
        op.create_table(
            "commercial_quota_workspace_overrides",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
            sa.Column("product_key", sa.String(), nullable=False),
            sa.Column("is_active", sa.Boolean(), nullable=False),
            sa.Column("enabled_override", sa.Boolean(), nullable=True),
            sa.Column("free_units_override", sa.Integer(), nullable=True),
            sa.Column("consumption_priority_override", json, nullable=False),
            sa.Column("checkout_required_on_zero_balance_override", sa.Boolean(), nullable=True),
            sa.Column("fifo_auto_approval_enabled_override", sa.Boolean(), nullable=True),
            sa.Column("default_blocked_request_ttl_hours_override", sa.Integer(), nullable=True),
            sa.Column("default_checkout_ttl_minutes_override", sa.Integer(), nullable=True),
            sa.Column("debt_enabled_override", sa.Boolean(), nullable=True),
            sa.Column("effective_from", sa.DateTime(), nullable=True),
            sa.Column("effective_to", sa.DateTime(), nullable=True),
            sa.Column("notes", sa.String(), nullable=False),
            sa.Column("updated_by_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("metadata", json, nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint(
                "workspace_id",
                "product_key",
                name="uq_commercial_quota_workspace_override_workspace_product",
            ),
        )
        _create_indexes(
            "commercial_quota_workspace_overrides",
            (
                ("ix_commercial_quota_workspace_overrides_workspace_id", ["workspace_id"]),
                ("ix_commercial_quota_workspace_overrides_product_key", ["product_key"]),
            ),
        )

    if not _has_table("commercial_balance_buckets"):
        op.create_table(
            "commercial_balance_buckets",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
            sa.Column("product_key", sa.String(), nullable=False),
            sa.Column("bucket_key", sa.String(), nullable=False),
            sa.Column("source_kind", sa.String(), nullable=False),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("units_granted", sa.Integer(), nullable=False),
            sa.Column("units_consumed", sa.Integer(), nullable=False),
            sa.Column("source_ref", sa.String(), nullable=False),
            sa.Column("granted_by_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("order_id", uuid, sa.ForeignKey("commercial_orders.id"), nullable=True),
            sa.Column("payment_id", uuid, sa.ForeignKey("commercial_payments.id"), nullable=True),
            sa.Column("starts_at", sa.DateTime(), nullable=False),
            sa.Column("ends_at", sa.DateTime(), nullable=True),
            sa.Column("metadata", json, nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint(
                "workspace_id",
                "product_key",
                "bucket_key",
                name="uq_commercial_balance_bucket_workspace_product_key",
            ),
        )
        _create_indexes(
            "commercial_balance_buckets",
            (
                ("ix_commercial_balance_buckets_workspace_id", ["workspace_id"]),
                ("ix_commercial_balance_buckets_product_key", ["product_key"]),
                ("ix_commercial_balance_buckets_bucket_key", ["bucket_key"]),
                ("ix_commercial_balance_buckets_source_ref", ["source_ref"]),
            ),
        )

    if not _has_table("commercial_balance_ledger"):
        op.create_table(
            "commercial_balance_ledger",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
            sa.Column("product_key", sa.String(), nullable=False),
            sa.Column("bucket_id", uuid, sa.ForeignKey("commercial_balance_buckets.id"), nullable=True),
            sa.Column("movement_type", sa.String(), nullable=False),
            sa.Column("source_kind", sa.String(), nullable=False),
            sa.Column("delta_units", sa.Integer(), nullable=False),
            sa.Column("balance_before_units", sa.Integer(), nullable=False),
            sa.Column("balance_after_units", sa.Integer(), nullable=False),
            sa.Column("bucket_balance_before_units", sa.Integer(), nullable=False),
            sa.Column("bucket_balance_after_units", sa.Integer(), nullable=False),
            sa.Column("source_ref", sa.String(), nullable=False),
            sa.Column("actor_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("order_id", uuid, sa.ForeignKey("commercial_orders.id"), nullable=True),
            sa.Column("payment_id", uuid, sa.ForeignKey("commercial_payments.id"), nullable=True),
            sa.Column("access_request_id", uuid, sa.ForeignKey("commercial_access_requests.id"), nullable=True),
            sa.Column("metadata", json, nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )
        _create_indexes(
            "commercial_balance_ledger",
            (
                ("ix_commercial_balance_ledger_workspace_id", ["workspace_id"]),
                ("ix_commercial_balance_ledger_product_key", ["product_key"]),
                ("ix_commercial_balance_ledger_bucket_id", ["bucket_id"]),
                ("ix_commercial_balance_ledger_source_ref", ["source_ref"]),
            ),
        )


def downgrade() -> None:
    if _has_table("commercial_balance_ledger"):
        op.drop_table("commercial_balance_ledger")
    if _has_table("commercial_balance_buckets"):
        op.drop_table("commercial_balance_buckets")
    if _has_table("commercial_quota_workspace_overrides"):
        op.drop_table("commercial_quota_workspace_overrides")
    if _has_table("commercial_quota_product_configs"):
        op.drop_table("commercial_quota_product_configs")
