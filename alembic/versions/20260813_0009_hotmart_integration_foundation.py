"""hotmart_integration_foundation

Revision ID: 20260813_0009
Revises: 20260810_0008
Create Date: 2026-08-13 10:00:00.000000

"""
from __future__ import annotations

from collections.abc import Iterable

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260813_0009"
down_revision = "20260810_0008"
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

    if not _has_table("hotmart_integration_configs"):
        op.create_table(
            "hotmart_integration_configs",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
            sa.Column("environment", sa.String(), nullable=False),
            sa.Column("enabled", sa.Boolean(), nullable=False),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("client_id_configured", sa.Boolean(), nullable=False),
            sa.Column("client_secret_configured", sa.Boolean(), nullable=False),
            sa.Column("basic_token_configured", sa.Boolean(), nullable=False),
            sa.Column("hottok_configured", sa.Boolean(), nullable=False),
            sa.Column("api_base_url", sa.String(), nullable=False),
            sa.Column("auth_base_url", sa.String(), nullable=False),
            sa.Column("webhook_public_url", sa.String(), nullable=False),
            sa.Column("last_health_check_at", sa.DateTime(), nullable=True),
            sa.Column("last_health_status", sa.String(), nullable=False),
            sa.Column("last_health_message", sa.String(), nullable=False),
            sa.Column("last_sync_at", sa.DateTime(), nullable=True),
            sa.Column("metadata", json, nullable=False),
            sa.Column("created_by_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("updated_by_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("workspace_id", "environment", name="uq_hotmart_config_workspace_environment"),
        )
        _create_indexes(
            "hotmart_integration_configs",
            (
                ("ix_hotmart_integration_configs_workspace_id", ["workspace_id"]),
                ("ix_hotmart_integration_configs_environment", ["environment"]),
                ("ix_hotmart_integration_configs_status", ["status"]),
            ),
        )

    if not _has_table("hotmart_integration_secrets"):
        op.create_table(
            "hotmart_integration_secrets",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
            sa.Column("environment", sa.String(), nullable=False),
            sa.Column("secret_kind", sa.String(), nullable=False),
            sa.Column("secret_ciphertext", sa.String(), nullable=False),
            sa.Column("secret_ref", sa.String(), nullable=False),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("last_rotated_at", sa.DateTime(), nullable=True),
            sa.Column("metadata", json, nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint(
                "workspace_id",
                "environment",
                "secret_kind",
                name="uq_hotmart_secret_workspace_environment_kind",
            ),
        )
        _create_indexes(
            "hotmart_integration_secrets",
            (
                ("ix_hotmart_integration_secrets_workspace_id", ["workspace_id"]),
                ("ix_hotmart_integration_secrets_environment", ["environment"]),
                ("ix_hotmart_integration_secrets_secret_kind", ["secret_kind"]),
            ),
        )

    if not _has_table("hotmart_product_mappings"):
        op.create_table(
            "hotmart_product_mappings",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
            sa.Column("environment", sa.String(), nullable=False),
            sa.Column("internal_product_key", sa.String(), nullable=False),
            sa.Column("hotmart_product_id", sa.String(), nullable=False),
            sa.Column("hotmart_product_ucode", sa.String(), nullable=False),
            sa.Column("offer_code", sa.String(), nullable=False),
            sa.Column("plan_code", sa.String(), nullable=False),
            sa.Column("billing_mode", sa.String(), nullable=False),
            sa.Column("currency", sa.String(), nullable=False),
            sa.Column("internal_base_currency", sa.String(), nullable=False),
            sa.Column("internal_unit_amount_usd_cents", sa.Integer(), nullable=False),
            sa.Column("hotmart_price_strategy", sa.String(), nullable=False),
            sa.Column("trm_policy", sa.String(), nullable=False),
            sa.Column("grants_tier", sa.String(), nullable=False),
            sa.Column("entitlement_scope", sa.String(), nullable=False),
            sa.Column("is_active", sa.Boolean(), nullable=False),
            sa.Column("metadata", json, nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint(
                "workspace_id",
                "internal_product_key",
                "environment",
                name="uq_hotmart_mapping_workspace_product_environment",
            ),
        )
        _create_indexes(
            "hotmart_product_mappings",
            (
                ("ix_hotmart_product_mappings_workspace_id", ["workspace_id"]),
                ("ix_hotmart_product_mappings_environment", ["environment"]),
                ("ix_hotmart_product_mappings_internal_product_key", ["internal_product_key"]),
                ("ix_hotmart_product_mappings_hotmart_product_id", ["hotmart_product_id"]),
            ),
        )

    if not _has_table("hotmart_payment_links"):
        op.create_table(
            "hotmart_payment_links",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
            sa.Column("order_id", uuid, sa.ForeignKey("commercial_orders.id"), nullable=True),
            sa.Column("created_by_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("environment", sa.String(), nullable=False),
            sa.Column("internal_product_key", sa.String(), nullable=False),
            sa.Column("hotmart_payment_link_id", sa.String(), nullable=False),
            sa.Column("checkout_url", sa.String(), nullable=False),
            sa.Column("activation_status", sa.String(), nullable=False),
            sa.Column("provider_ref", sa.String(), nullable=False),
            sa.Column("gross_amount_cents", sa.Integer(), nullable=False),
            sa.Column("discount_amount_cents", sa.Integer(), nullable=False),
            sa.Column("net_amount_cents", sa.Integer(), nullable=False),
            sa.Column("currency", sa.String(), nullable=False),
            sa.Column("internal_unit_amount_usd_cents", sa.Integer(), nullable=False),
            sa.Column("trm_cop_applied", sa.Float(), nullable=True),
            sa.Column("discount_origin", sa.String(), nullable=False),
            sa.Column("request_payload_redacted", json, nullable=False),
            sa.Column("response_payload_redacted", json, nullable=False),
            sa.Column("expires_at", sa.DateTime(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("workspace_id", "hotmart_payment_link_id", name="uq_hotmart_payment_link_provider_id"),
        )
        _create_indexes(
            "hotmart_payment_links",
            (
                ("ix_hotmart_payment_links_workspace_id", ["workspace_id"]),
                ("ix_hotmart_payment_links_order_id", ["order_id"]),
                ("ix_hotmart_payment_links_environment", ["environment"]),
                ("ix_hotmart_payment_links_internal_product_key", ["internal_product_key"]),
                ("ix_hotmart_payment_links_hotmart_payment_link_id", ["hotmart_payment_link_id"]),
                ("ix_hotmart_payment_links_activation_status", ["activation_status"]),
                ("ix_hotmart_payment_links_provider_ref", ["provider_ref"]),
            ),
        )

    if not _has_table("hotmart_promotions"):
        op.create_table(
            "hotmart_promotions",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
            sa.Column("environment", sa.String(), nullable=False),
            sa.Column("internal_campaign_key", sa.String(), nullable=False),
            sa.Column("internal_product_key", sa.String(), nullable=False),
            sa.Column("hotmart_product_id", sa.String(), nullable=False),
            sa.Column("offer_codes", json, nullable=False),
            sa.Column("coupon_id", sa.String(), nullable=False),
            sa.Column("coupon_code", sa.String(), nullable=False),
            sa.Column("discount_percent", sa.Float(), nullable=False),
            sa.Column("discount_origin", sa.String(), nullable=False),
            sa.Column("discount_type", sa.String(), nullable=False),
            sa.Column("discount_amount_cents", sa.Integer(), nullable=True),
            sa.Column("starts_at", sa.DateTime(), nullable=True),
            sa.Column("ends_at", sa.DateTime(), nullable=True),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("published_at", sa.DateTime(), nullable=True),
            sa.Column("created_by_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("metadata", json, nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("workspace_id", "environment", "coupon_code", name="uq_hotmart_promotion_coupon"),
        )
        _create_indexes(
            "hotmart_promotions",
            (
                ("ix_hotmart_promotions_workspace_id", ["workspace_id"]),
                ("ix_hotmart_promotions_environment", ["environment"]),
                ("ix_hotmart_promotions_internal_campaign_key", ["internal_campaign_key"]),
                ("ix_hotmart_promotions_internal_product_key", ["internal_product_key"]),
                ("ix_hotmart_promotions_hotmart_product_id", ["hotmart_product_id"]),
                ("ix_hotmart_promotions_coupon_id", ["coupon_id"]),
                ("ix_hotmart_promotions_coupon_code", ["coupon_code"]),
                ("ix_hotmart_promotions_status", ["status"]),
            ),
        )

    if not _has_table("hotmart_sync_runs"):
        op.create_table(
            "hotmart_sync_runs",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
            sa.Column("environment", sa.String(), nullable=False),
            sa.Column("resource", sa.String(), nullable=False),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("started_by_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("started_at", sa.DateTime(), nullable=False),
            sa.Column("finished_at", sa.DateTime(), nullable=True),
            sa.Column("cursor_before", sa.String(), nullable=False),
            sa.Column("cursor_after", sa.String(), nullable=False),
            sa.Column("records_read", sa.Integer(), nullable=False),
            sa.Column("records_created", sa.Integer(), nullable=False),
            sa.Column("records_updated", sa.Integer(), nullable=False),
            sa.Column("records_skipped", sa.Integer(), nullable=False),
            sa.Column("error_summary", sa.String(), nullable=False),
            sa.Column("metadata", json, nullable=False),
        )
        _create_indexes(
            "hotmart_sync_runs",
            (
                ("ix_hotmart_sync_runs_workspace_id", ["workspace_id"]),
                ("ix_hotmart_sync_runs_environment", ["environment"]),
                ("ix_hotmart_sync_runs_resource", ["resource"]),
                ("ix_hotmart_sync_runs_status", ["status"]),
            ),
        )

    if not _has_table("hotmart_sync_cursors"):
        op.create_table(
            "hotmart_sync_cursors",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
            sa.Column("environment", sa.String(), nullable=False),
            sa.Column("resource", sa.String(), nullable=False),
            sa.Column("page_token", sa.String(), nullable=False),
            sa.Column("last_event_at", sa.DateTime(), nullable=True),
            sa.Column("last_transaction", sa.String(), nullable=False),
            sa.Column("last_success_at", sa.DateTime(), nullable=True),
            sa.Column("metadata", json, nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("workspace_id", "environment", "resource", name="uq_hotmart_sync_cursor_resource"),
        )
        _create_indexes(
            "hotmart_sync_cursors",
            (
                ("ix_hotmart_sync_cursors_workspace_id", ["workspace_id"]),
                ("ix_hotmart_sync_cursors_environment", ["environment"]),
                ("ix_hotmart_sync_cursors_resource", ["resource"]),
            ),
        )

    if not _has_table("hotmart_webhook_events"):
        op.create_table(
            "hotmart_webhook_events",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("event_id", sa.String(), nullable=False),
            sa.Column("event_type", sa.String(), nullable=False),
            sa.Column("transaction", sa.String(), nullable=False),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=True),
            sa.Column("order_id", uuid, sa.ForeignKey("commercial_orders.id"), nullable=True),
            sa.Column("payment_id", uuid, sa.ForeignKey("commercial_payments.id"), nullable=True),
            sa.Column("hottok_validated", sa.Boolean(), nullable=False),
            sa.Column("processing_status", sa.String(), nullable=False),
            sa.Column("payload_hash", sa.String(), nullable=False),
            sa.Column("payload_redacted", json, nullable=False),
            sa.Column("error_code", sa.String(), nullable=False),
            sa.Column("error_message", sa.String(), nullable=False),
            sa.Column("retries", sa.Integer(), nullable=False),
            sa.Column("processed_at", sa.DateTime(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("event_id", name="uq_hotmart_webhook_event_id"),
        )
        _create_indexes(
            "hotmart_webhook_events",
            (
                ("ix_hotmart_webhook_events_event_id", ["event_id"]),
                ("ix_hotmart_webhook_events_event_type", ["event_type"]),
                ("ix_hotmart_webhook_events_transaction", ["transaction"]),
                ("ix_hotmart_webhook_events_workspace_id", ["workspace_id"]),
                ("ix_hotmart_webhook_events_order_id", ["order_id"]),
                ("ix_hotmart_webhook_events_processing_status", ["processing_status"]),
                ("ix_hotmart_webhook_events_payload_hash", ["payload_hash"]),
            ),
        )

    if not _has_table("hotmart_reconciliation_issues"):
        op.create_table(
            "hotmart_reconciliation_issues",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
            sa.Column("environment", sa.String(), nullable=False),
            sa.Column("issue_type", sa.String(), nullable=False),
            sa.Column("severity", sa.String(), nullable=False),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("provider_ref", sa.String(), nullable=False),
            sa.Column("internal_ref", sa.String(), nullable=False),
            sa.Column("summary", sa.String(), nullable=False),
            sa.Column("suggested_action", sa.String(), nullable=False),
            sa.Column("resolution_action", sa.String(), nullable=False),
            sa.Column("resolution_note", sa.String(), nullable=False),
            sa.Column("resolved_by_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("resolved_at", sa.DateTime(), nullable=True),
            sa.Column("metadata", json, nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
        )
        _create_indexes(
            "hotmart_reconciliation_issues",
            (
                ("ix_hotmart_reconciliation_issues_workspace_id", ["workspace_id"]),
                ("ix_hotmart_reconciliation_issues_environment", ["environment"]),
                ("ix_hotmart_reconciliation_issues_issue_type", ["issue_type"]),
                ("ix_hotmart_reconciliation_issues_severity", ["severity"]),
                ("ix_hotmart_reconciliation_issues_status", ["status"]),
                ("ix_hotmart_reconciliation_issues_provider_ref", ["provider_ref"]),
                ("ix_hotmart_reconciliation_issues_internal_ref", ["internal_ref"]),
            ),
        )


def downgrade() -> None:
    for table_name in (
        "hotmart_reconciliation_issues",
        "hotmart_webhook_events",
        "hotmart_sync_cursors",
        "hotmart_sync_runs",
        "hotmart_promotions",
        "hotmart_payment_links",
        "hotmart_product_mappings",
        "hotmart_integration_secrets",
        "hotmart_integration_configs",
    ):
        if _has_table(table_name):
            op.drop_table(table_name)
