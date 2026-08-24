"""hotmart_pending_activations

Revision ID: 20260824_0017
Revises: 20260823_0016
Create Date: 2026-08-24 10:20:00.000000

"""
from __future__ import annotations

from collections.abc import Iterable

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260824_0017"
down_revision = "20260823_0016"
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

    if _has_table("hotmart_pending_activations"):
        return

    op.create_table(
        "hotmart_pending_activations",
        sa.Column("id", uuid, primary_key=True),
        sa.Column("source_workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=False),
        sa.Column("environment", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("provider_ref", sa.String(), nullable=False),
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("webhook_event_id", uuid, sa.ForeignKey("hotmart_webhook_events.id"), nullable=True),
        sa.Column("hotmart_product_id", sa.String(), nullable=False),
        sa.Column("hotmart_product_ucode", sa.String(), nullable=False),
        sa.Column("offer_code", sa.String(), nullable=False),
        sa.Column("plan_code", sa.String(), nullable=False),
        sa.Column("product_key", sa.String(), nullable=False),
        sa.Column("package_code", sa.String(), nullable=False),
        sa.Column("resolution_strategy", sa.String(), nullable=False),
        sa.Column("buyer_name", sa.String(), nullable=False),
        sa.Column("buyer_email", sa.String(), nullable=False),
        sa.Column("buyer_document", sa.String(), nullable=False),
        sa.Column("currency", sa.String(), nullable=False),
        sa.Column("amount_cents", sa.Integer(), nullable=False),
        sa.Column("activation_token", sa.String(), nullable=False),
        sa.Column("claimed_by_user_id", uuid, sa.ForeignKey("users.id"), nullable=True),
        sa.Column("claimed_workspace_id", uuid, sa.ForeignKey("workspaces.id"), nullable=True),
        sa.Column("claimed_session_id", uuid, sa.ForeignKey("sessions.id"), nullable=True),
        sa.Column("adopted_order_id", uuid, sa.ForeignKey("commercial_orders.id"), nullable=True),
        sa.Column("adopted_payment_id", uuid, sa.ForeignKey("commercial_payments.id"), nullable=True),
        sa.Column("claimed_at", sa.DateTime(), nullable=True),
        sa.Column("canceled_at", sa.DateTime(), nullable=True),
        sa.Column("metadata", json, nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint(
            "source_workspace_id",
            "environment",
            "provider_ref",
            name="uq_hotmart_pending_activation_workspace_provider_ref",
        ),
        sa.UniqueConstraint("activation_token", name="uq_hotmart_pending_activation_token"),
    )
    _create_indexes(
        "hotmart_pending_activations",
        (
            ("ix_hotmart_pending_activations_source_workspace_id", ["source_workspace_id"]),
            ("ix_hotmart_pending_activations_environment", ["environment"]),
            ("ix_hotmart_pending_activations_status", ["status"]),
            ("ix_hotmart_pending_activations_provider_ref", ["provider_ref"]),
            ("ix_hotmart_pending_activations_event_id", ["event_id"]),
            ("ix_hotmart_pending_activations_hotmart_product_id", ["hotmart_product_id"]),
            ("ix_hotmart_pending_activations_product_key", ["product_key"]),
            ("ix_hotmart_pending_activations_buyer_email", ["buyer_email"]),
            ("ix_hotmart_pending_activations_activation_token", ["activation_token"]),
        ),
    )


def downgrade() -> None:
    if _has_table("hotmart_pending_activations"):
        op.drop_table("hotmart_pending_activations")
