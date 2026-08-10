from __future__ import annotations

from collections.abc import Iterable

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260802_0001"
down_revision = None
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

    if not _has_table("product_catalog"):
        op.create_table(
            "product_catalog",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("product_key", sa.String(), nullable=False),
            sa.Column("tier", sa.String(), nullable=False),
            sa.Column("product_type", sa.String(), nullable=False),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("description", sa.String(), nullable=False),
            sa.Column("scope", sa.String(), nullable=False),
            sa.Column("benefits", json, nullable=False),
            sa.Column("exclusions", json, nullable=False),
            sa.Column("capabilities", json, nullable=False),
            sa.Column("metadata", json, nullable=False),
            sa.Column("version", sa.Integer(), nullable=False),
            sa.Column("is_active", sa.Boolean(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("product_key", "version", name="uq_product_catalog_key_version"),
        )
        _create_indexes(
            "product_catalog",
            (
                ("ix_product_catalog_product_key", ["product_key"]),
                ("ix_product_catalog_version", ["version"]),
            ),
        )

    if not _has_table("product_prices"):
        op.create_table(
            "product_prices",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("product_key", sa.String(), nullable=False),
            sa.Column("price_code", sa.String(), nullable=False),
            sa.Column("currency", sa.String(), nullable=False),
            sa.Column("unit_amount_cents", sa.Integer(), nullable=False),
            sa.Column("billing_period", sa.String(), nullable=False),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("version", sa.Integer(), nullable=False),
            sa.Column("metadata", json, nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("price_code", name="uq_product_price_code"),
        )
        _create_indexes(
            "product_prices",
            (
                ("ix_product_prices_product_key", ["product_key"]),
                ("ix_product_prices_price_code", ["price_code"]),
                ("ix_product_prices_version", ["version"]),
            ),
        )

    if not _has_table("commercial_orders"):
        op.create_table(
            "commercial_orders",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
            sa.Column("session_id", uuid, sa.ForeignKey("sessions.id"), nullable=True),
            sa.Column("buyer_user_id", uuid, sa.ForeignKey("users.id"), nullable=False),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("currency", sa.String(), nullable=False),
            sa.Column("subtotal_cents", sa.Integer(), nullable=False),
            sa.Column("tax_cents", sa.Integer(), nullable=False),
            sa.Column("total_cents", sa.Integer(), nullable=False),
            sa.Column("provider", sa.String(), nullable=False),
            sa.Column("checkout_ref", sa.String(), nullable=False),
            sa.Column("checkout_url", sa.String(), nullable=False),
            sa.Column("idempotency_key", sa.String(), nullable=False),
            sa.Column("metadata", json, nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.Column("paid_at", sa.DateTime(), nullable=True),
            sa.UniqueConstraint("checkout_ref", name="uq_commercial_order_checkout_ref"),
        )
        _create_indexes(
            "commercial_orders",
            (
                ("ix_commercial_orders_workspace_id", ["workspace_id"]),
                ("ix_commercial_orders_session_id", ["session_id"]),
                ("ix_commercial_orders_buyer_user_id", ["buyer_user_id"]),
                ("ix_commercial_orders_checkout_ref", ["checkout_ref"]),
                ("ix_commercial_orders_idempotency_key", ["idempotency_key"]),
            ),
        )

    if not _has_table("commercial_order_lines"):
        op.create_table(
            "commercial_order_lines",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("order_id", uuid, sa.ForeignKey("commercial_orders.id"), nullable=False),
            sa.Column("product_key", sa.String(), nullable=False),
            sa.Column("price_code", sa.String(), nullable=False),
            sa.Column("quantity", sa.Integer(), nullable=False),
            sa.Column("unit_amount_cents", sa.Integer(), nullable=False),
            sa.Column("total_amount_cents", sa.Integer(), nullable=False),
            sa.Column("metadata", json, nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )
        _create_indexes(
            "commercial_order_lines",
            (
                ("ix_commercial_order_lines_order_id", ["order_id"]),
                ("ix_commercial_order_lines_product_key", ["product_key"]),
            ),
        )

    if not _has_table("commercial_payments"):
        op.create_table(
            "commercial_payments",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
            sa.Column("session_id", uuid, sa.ForeignKey("sessions.id"), nullable=True),
            sa.Column("order_id", uuid, sa.ForeignKey("commercial_orders.id"), nullable=False),
            sa.Column("provider", sa.String(), nullable=False),
            sa.Column("provider_payment_id", sa.String(), nullable=False),
            sa.Column("provider_checkout_ref", sa.String(), nullable=False),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("amount_cents", sa.Integer(), nullable=False),
            sa.Column("currency", sa.String(), nullable=False),
            sa.Column("idempotency_key", sa.String(), nullable=False),
            sa.Column("metadata", json, nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("provider", "provider_payment_id", name="uq_commercial_payment_provider_id"),
        )
        _create_indexes(
            "commercial_payments",
            (
                ("ix_commercial_payments_workspace_id", ["workspace_id"]),
                ("ix_commercial_payments_session_id", ["session_id"]),
                ("ix_commercial_payments_order_id", ["order_id"]),
                ("ix_commercial_payments_provider_checkout_ref", ["provider_checkout_ref"]),
                ("ix_commercial_payments_idempotency_key", ["idempotency_key"]),
            ),
        )

    if not _has_table("commercial_entitlements"):
        op.create_table(
            "commercial_entitlements",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
            sa.Column("session_id", uuid, sa.ForeignKey("sessions.id"), nullable=True),
            sa.Column("product_key", sa.String(), nullable=False),
            sa.Column("tier", sa.String(), nullable=False),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("source", sa.String(), nullable=False),
            sa.Column("order_id", uuid, sa.ForeignKey("commercial_orders.id"), nullable=True),
            sa.Column("order_line_id", uuid, sa.ForeignKey("commercial_order_lines.id"), nullable=True),
            sa.Column("payment_id", uuid, sa.ForeignKey("commercial_payments.id"), nullable=True),
            sa.Column("granted_by_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("revoked_by_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("starts_at", sa.DateTime(), nullable=False),
            sa.Column("ends_at", sa.DateTime(), nullable=True),
            sa.Column("version", sa.Integer(), nullable=False),
            sa.Column("metadata", json, nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
        )
        _create_indexes(
            "commercial_entitlements",
            (
                ("ix_commercial_entitlements_workspace_id", ["workspace_id"]),
                ("ix_commercial_entitlements_session_id", ["session_id"]),
                ("ix_commercial_entitlements_product_key", ["product_key"]),
                ("ix_commercial_entitlements_version", ["version"]),
            ),
        )

    if not _has_table("commercial_access_requests"):
        op.create_table(
            "commercial_access_requests",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
            sa.Column("session_id", uuid, sa.ForeignKey("sessions.id"), nullable=False),
            sa.Column("requester_user_id", uuid, sa.ForeignKey("users.id"), nullable=False),
            sa.Column("resolver_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("capability", sa.String(), nullable=False),
            sa.Column("product_key", sa.String(), nullable=False),
            sa.Column("target_tier", sa.String(), nullable=False),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("reason", sa.String(), nullable=False),
            sa.Column("resolution_note", sa.String(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.Column("resolved_at", sa.DateTime(), nullable=True),
        )
        _create_indexes(
            "commercial_access_requests",
            (
                ("ix_commercial_access_requests_workspace_id", ["workspace_id"]),
                ("ix_commercial_access_requests_session_id", ["session_id"]),
                ("ix_commercial_access_requests_requester_user_id", ["requester_user_id"]),
                ("ix_commercial_access_requests_capability", ["capability"]),
                ("ix_commercial_access_requests_product_key", ["product_key"]),
            ),
        )

    if not _has_table("commercial_events"):
        op.create_table(
            "commercial_events",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
            sa.Column("session_id", uuid, sa.ForeignKey("sessions.id"), nullable=True),
            sa.Column("user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("event_key", sa.String(), nullable=False),
            sa.Column("product_key", sa.String(), nullable=False),
            sa.Column("source", sa.String(), nullable=False),
            sa.Column("correlation_id", sa.String(), nullable=False),
            sa.Column("revenue_cents", sa.Integer(), nullable=False),
            sa.Column("currency", sa.String(), nullable=False),
            sa.Column("metadata", json, nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )
        _create_indexes(
            "commercial_events",
            (
                ("ix_commercial_events_workspace_id", ["workspace_id"]),
                ("ix_commercial_events_session_id", ["session_id"]),
                ("ix_commercial_events_user_id", ["user_id"]),
                ("ix_commercial_events_event_key", ["event_key"]),
                ("ix_commercial_events_product_key", ["product_key"]),
                ("ix_commercial_events_correlation_id", ["correlation_id"]),
            ),
        )


def downgrade() -> None:
    for table_name in (
        "commercial_events",
        "commercial_access_requests",
        "commercial_entitlements",
        "commercial_payments",
        "commercial_order_lines",
        "commercial_orders",
        "product_prices",
        "product_catalog",
    ):
        if _has_table(table_name):
            op.drop_table(table_name)
