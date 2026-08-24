"""commercial_catalog_and_debt

Revision ID: 20260823_0016
Revises: 20260823_0015
Create Date: 2026-08-23 13:30:00.000000

"""
from __future__ import annotations

from collections.abc import Iterable

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260823_0016"
down_revision = "20260823_0015"
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

    if not _has_table("commercial_package_catalog"):
        op.create_table(
            "commercial_package_catalog",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("package_code", sa.String(), nullable=False),
            sa.Column("display_name", sa.String(), nullable=False),
            sa.Column("product_key", sa.String(), nullable=False),
            sa.Column("package_type", sa.String(), nullable=False),
            sa.Column("enabled", sa.Boolean(), nullable=False),
            sa.Column("granted_units", sa.Integer(), nullable=False),
            sa.Column("granted_units_blueprint_pro", sa.Integer(), nullable=False),
            sa.Column("granted_units_acp", sa.Integer(), nullable=False),
            sa.Column("validity_days", sa.Integer(), nullable=True),
            sa.Column("billing_cycle", sa.String(), nullable=False),
            sa.Column("renewal_policy", sa.String(), nullable=False),
            sa.Column("recommendation_priority", sa.Integer(), nullable=False),
            sa.Column("hotmart_environment", sa.String(), nullable=False),
            sa.Column("hotmart_product_id", sa.String(), nullable=False),
            sa.Column("hotmart_product_ucode", sa.String(), nullable=False),
            sa.Column("offer_code", sa.String(), nullable=False),
            sa.Column("plan_code", sa.String(), nullable=False),
            sa.Column("checkout_currency_mode", sa.String(), nullable=False),
            sa.Column("hotmart_price_strategy", sa.String(), nullable=False),
            sa.Column("metadata", json, nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("package_code", name="uq_commercial_package_catalog_code"),
        )
        _create_indexes(
            "commercial_package_catalog",
            (
                ("ix_commercial_package_catalog_package_code", ["package_code"]),
                ("ix_commercial_package_catalog_product_key", ["product_key"]),
            ),
        )

    if not _has_table("commercial_debts"):
        op.create_table(
            "commercial_debts",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
            sa.Column("product_key", sa.String(), nullable=False),
            sa.Column("access_request_id", uuid, sa.ForeignKey("commercial_access_requests.id"), nullable=True),
            sa.Column("order_id", uuid, sa.ForeignKey("commercial_orders.id"), nullable=True),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("reason_code", sa.String(), nullable=False),
            sa.Column("reason_label", sa.String(), nullable=False),
            sa.Column("summary", sa.String(), nullable=False),
            sa.Column("amount_cents", sa.Integer(), nullable=False),
            sa.Column("settled_amount_cents", sa.Integer(), nullable=False),
            sa.Column("currency", sa.String(), nullable=False),
            sa.Column("opened_by_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("resolved_by_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("due_at", sa.DateTime(), nullable=True),
            sa.Column("resolved_at", sa.DateTime(), nullable=True),
            sa.Column("metadata", json, nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
        )
        _create_indexes(
            "commercial_debts",
            (
                ("ix_commercial_debts_workspace_id", ["workspace_id"]),
                ("ix_commercial_debts_product_key", ["product_key"]),
            ),
        )

    if not _has_table("commercial_debt_settlements"):
        op.create_table(
            "commercial_debt_settlements",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("debt_id", uuid, sa.ForeignKey("commercial_debts.id"), nullable=False),
            sa.Column("workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
            sa.Column("order_id", uuid, sa.ForeignKey("commercial_orders.id"), nullable=True),
            sa.Column("payment_id", uuid, sa.ForeignKey("commercial_payments.id"), nullable=True),
            sa.Column("settled_amount_cents", sa.Integer(), nullable=False),
            sa.Column("currency", sa.String(), nullable=False),
            sa.Column("settlement_kind", sa.String(), nullable=False),
            sa.Column("actor_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("metadata", json, nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )
        _create_indexes(
            "commercial_debt_settlements",
            (
                ("ix_commercial_debt_settlements_debt_id", ["debt_id"]),
                ("ix_commercial_debt_settlements_workspace_id", ["workspace_id"]),
            ),
        )

    if _has_table("hotmart_webhook_events"):
        with op.batch_alter_table("hotmart_webhook_events") as batch_op:
            try:
                batch_op.drop_constraint("uq_hotmart_webhook_event_id", type_="unique")
            except Exception:
                pass
            try:
                batch_op.create_unique_constraint(
                    "uq_hotmart_webhook_event_id_type",
                    ["event_id", "event_type"],
                )
            except Exception:
                pass


def downgrade() -> None:
    if _has_table("hotmart_webhook_events"):
        with op.batch_alter_table("hotmart_webhook_events") as batch_op:
            try:
                batch_op.drop_constraint("uq_hotmart_webhook_event_id_type", type_="unique")
            except Exception:
                pass
            try:
                batch_op.create_unique_constraint("uq_hotmart_webhook_event_id", ["event_id"])
            except Exception:
                pass
    if _has_table("commercial_debt_settlements"):
        op.drop_table("commercial_debt_settlements")
    if _has_table("commercial_debts"):
        op.drop_table("commercial_debts")
    if _has_table("commercial_package_catalog"):
        op.drop_table("commercial_package_catalog")
