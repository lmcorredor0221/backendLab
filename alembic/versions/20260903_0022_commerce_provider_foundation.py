"""commerce_provider_foundation

Revision ID: 20260903_0022
Revises: 20260902_0021
Create Date: 2026-09-03 00:22:00.000000

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260903_0022"
down_revision = "20260902_0021"
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


def _json_default() -> sa.TextClause:
    if op.get_bind().dialect.name == "postgresql":
        return sa.text("'{}'::jsonb")
    return sa.text("'{}'")


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
    if not _can_create_table():
        return
    uuid = _uuid_type()
    json_type = _json_type()
    json_default = _json_default()

    if not _has_table("commerce_provider_configs"):
        op.create_table(
            "commerce_provider_configs",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
            sa.Column("provider_key", sa.String(), nullable=False),
            sa.Column("environment", sa.String(), nullable=False),
            sa.Column("enabled", sa.Boolean(), nullable=False),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("api_base_url", sa.String(), nullable=False),
            sa.Column("webhook_public_url", sa.String(), nullable=False),
            sa.Column("capabilities", json_type, nullable=False, server_default=json_default),
            sa.Column("metadata", json_type, nullable=False, server_default=json_default),
            sa.Column("last_checked_at", sa.DateTime(), nullable=True),
            sa.Column("last_health_status", sa.String(), nullable=False),
            sa.Column("last_health_message", sa.String(), nullable=False),
            sa.Column("created_by_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("updated_by_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint(
                "workspace_id",
                "provider_key",
                "environment",
                name="uq_commerce_provider_config_workspace_provider_environment",
            ),
        )
        op.create_index(
            "ix_commerce_provider_config_workspace_provider_status",
            "commerce_provider_configs",
            ["workspace_id", "provider_key", "environment", "status"],
        )

    if not _has_table("commerce_provider_secrets"):
        op.create_table(
            "commerce_provider_secrets",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
            sa.Column("provider_key", sa.String(), nullable=False),
            sa.Column("environment", sa.String(), nullable=False),
            sa.Column("secret_kind", sa.String(), nullable=False),
            sa.Column("secret_ciphertext", sa.String(), nullable=False),
            sa.Column("secret_ref", sa.String(), nullable=False),
            sa.Column("configured", sa.Boolean(), nullable=False),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("last_rotated_at", sa.DateTime(), nullable=True),
            sa.Column("updated_by_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint(
                "workspace_id",
                "provider_key",
                "environment",
                "secret_kind",
                name="uq_commerce_provider_secret_workspace_provider_environment_kind",
            ),
        )
        op.create_index(
            "ix_commerce_provider_secret_workspace_provider",
            "commerce_provider_secrets",
            ["workspace_id", "provider_key", "environment"],
        )

    if not _has_table("commerce_provider_product_mappings"):
        op.create_table(
            "commerce_provider_product_mappings",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
            sa.Column("provider_key", sa.String(), nullable=False),
            sa.Column("environment", sa.String(), nullable=False),
            sa.Column("internal_product_key", sa.String(), nullable=False),
            sa.Column("package_code", sa.String(), nullable=False),
            sa.Column("billing_mode", sa.String(), nullable=False),
            sa.Column("currency", sa.String(), nullable=False),
            sa.Column("internal_unit_amount_usd_cents", sa.Integer(), nullable=False),
            sa.Column("provider_product_id", sa.String(), nullable=False),
            sa.Column("provider_plan_id", sa.String(), nullable=False),
            sa.Column("provider_price_id", sa.String(), nullable=False),
            sa.Column("provider_payment_link_id", sa.String(), nullable=False),
            sa.Column("provider_offer_ref", sa.String(), nullable=False),
            sa.Column("price_strategy", sa.String(), nullable=False),
            sa.Column("grants_tier", sa.String(), nullable=False),
            sa.Column("entitlement_scope", sa.String(), nullable=False),
            sa.Column("is_active", sa.Boolean(), nullable=False),
            sa.Column("metadata", json_type, nullable=False, server_default=json_default),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint(
                "workspace_id",
                "provider_key",
                "environment",
                "internal_product_key",
                "package_code",
                name="uq_commerce_provider_mapping_workspace_provider_product_package",
            ),
        )
        op.create_index(
            "ix_commerce_provider_mapping_workspace_provider_active",
            "commerce_provider_product_mappings",
            ["workspace_id", "provider_key", "environment", "is_active"],
        )

    if not _has_table("commerce_provider_checkout_records"):
        op.create_table(
            "commerce_provider_checkout_records",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
            sa.Column("provider_key", sa.String(), nullable=False),
            sa.Column("environment", sa.String(), nullable=False),
            sa.Column("order_id", uuid, sa.ForeignKey("commercial_orders.id"), nullable=False),
            sa.Column("checkout_ref", sa.String(), nullable=False),
            sa.Column("provider_checkout_id", sa.String(), nullable=False),
            sa.Column("provider_payment_link_id", sa.String(), nullable=False),
            sa.Column("provider_customer_id", sa.String(), nullable=False),
            sa.Column("checkout_url", sa.String(), nullable=False),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("amount_cents", sa.Integer(), nullable=False),
            sa.Column("currency", sa.String(), nullable=False),
            sa.Column("request_payload_redacted", json_type, nullable=False, server_default=json_default),
            sa.Column("response_payload_redacted", json_type, nullable=False, server_default=json_default),
            sa.Column("metadata", json_type, nullable=False, server_default=json_default),
            sa.Column("expires_at", sa.DateTime(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("provider_key", "checkout_ref", name="uq_commerce_provider_checkout_ref"),
        )
        op.create_index("ix_commerce_provider_checkout_order", "commerce_provider_checkout_records", ["order_id"])
        op.create_index(
            "ix_commerce_provider_checkout_provider_checkout_id",
            "commerce_provider_checkout_records",
            ["provider_key", "provider_checkout_id"],
        )
        op.create_index(
            "ix_commerce_provider_checkout_provider_payment_link_id",
            "commerce_provider_checkout_records",
            ["provider_key", "provider_payment_link_id"],
        )

    if not _has_table("commerce_provider_webhook_events"):
        op.create_table(
            "commerce_provider_webhook_events",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("provider_key", sa.String(), nullable=False),
            sa.Column("environment", sa.String(), nullable=False),
            sa.Column("event_id", sa.String(), nullable=False),
            sa.Column("event_type", sa.String(), nullable=False),
            sa.Column("provider_resource_id", sa.String(), nullable=False),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=True),
            sa.Column("order_id", uuid, sa.ForeignKey("commercial_orders.id"), nullable=True),
            sa.Column("payment_id", uuid, sa.ForeignKey("commercial_payments.id"), nullable=True),
            sa.Column("signature_validated", sa.Boolean(), nullable=False),
            sa.Column("processing_status", sa.String(), nullable=False),
            sa.Column("retries", sa.Integer(), nullable=False),
            sa.Column("payload_hash", sa.String(), nullable=False),
            sa.Column("payload_redacted", json_type, nullable=False, server_default=json_default),
            sa.Column("error_code", sa.String(), nullable=False),
            sa.Column("error_message", sa.String(), nullable=False),
            sa.Column("processed_at", sa.DateTime(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("provider_key", "event_id", "event_type", name="uq_commerce_provider_webhook_event"),
        )
        op.create_index(
            "ix_commerce_provider_webhook_provider_payload_hash",
            "commerce_provider_webhook_events",
            ["provider_key", "payload_hash"],
        )
        op.create_index(
            "ix_commerce_provider_webhook_provider_resource",
            "commerce_provider_webhook_events",
            ["provider_key", "provider_resource_id"],
        )
        op.create_index(
            "ix_commerce_provider_webhook_workspace_status",
            "commerce_provider_webhook_events",
            ["workspace_id", "processing_status"],
        )


def downgrade() -> None:
    for table_name in (
        "commerce_provider_webhook_events",
        "commerce_provider_checkout_records",
        "commerce_provider_product_mappings",
        "commerce_provider_secrets",
        "commerce_provider_configs",
    ):
        if _has_table(table_name):
            op.drop_table(table_name)
