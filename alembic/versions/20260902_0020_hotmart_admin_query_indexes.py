"""hotmart_admin_query_indexes

Revision ID: 20260902_0020
Revises: 20260828_0019
Create Date: 2026-09-02 00:00:00.000000

"""
from __future__ import annotations

from alembic import op


revision = "20260902_0020"
down_revision = "20260828_0019"
branch_labels = None
depends_on = None


INDEXES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("ix_commercial_debts_workspace_status_created", "commercial_debts", ("workspace_id", "status", "created_at")),
    ("ix_commercial_debts_workspace_product_status", "commercial_debts", ("workspace_id", "product_key", "status")),
    (
        "ix_hotmart_config_workspace_environment_status",
        "hotmart_integration_configs",
        ("workspace_id", "environment", "status"),
    ),
    (
        "ix_hotmart_mapping_workspace_environment_active",
        "hotmart_product_mappings",
        ("workspace_id", "environment", "is_active"),
    ),
    (
        "ix_hotmart_mapping_workspace_env_product_active",
        "hotmart_product_mappings",
        ("workspace_id", "environment", "internal_product_key", "is_active"),
    ),
    ("ix_hotmart_payment_links_workspace_created", "hotmart_payment_links", ("workspace_id", "created_at")),
    (
        "ix_hotmart_payment_links_workspace_env_status",
        "hotmart_payment_links",
        ("workspace_id", "environment", "activation_status"),
    ),
    ("ix_hotmart_promotions_workspace_env_updated", "hotmart_promotions", ("workspace_id", "environment", "updated_at")),
    ("ix_hotmart_promotions_workspace_env_status", "hotmart_promotions", ("workspace_id", "environment", "status")),
    ("ix_hotmart_sync_runs_workspace_env_started", "hotmart_sync_runs", ("workspace_id", "environment", "started_at")),
    (
        "ix_hotmart_sync_runs_workspace_env_resource_started",
        "hotmart_sync_runs",
        ("workspace_id", "environment", "resource", "started_at"),
    ),
    ("ix_hotmart_sync_runs_workspace_env_status", "hotmart_sync_runs", ("workspace_id", "environment", "status")),
    (
        "ix_hotmart_webhook_events_workspace_status_created",
        "hotmart_webhook_events",
        ("workspace_id", "processing_status", "created_at"),
    ),
    (
        "ix_hotmart_webhook_events_workspace_type_created",
        "hotmart_webhook_events",
        ("workspace_id", "event_type", "created_at"),
    ),
    (
        "ix_hotmart_pending_workspace_status_created",
        "hotmart_pending_activations",
        ("source_workspace_id", "status", "created_at"),
    ),
    (
        "ix_hotmart_pending_buyer_status_created",
        "hotmart_pending_activations",
        ("buyer_email", "status", "created_at"),
    ),
    (
        "ix_hotmart_reconciliation_workspace_env_status_updated",
        "hotmart_reconciliation_issues",
        ("workspace_id", "environment", "status", "updated_at"),
    ),
    (
        "ix_hotmart_reconciliation_workspace_env_severity_status",
        "hotmart_reconciliation_issues",
        ("workspace_id", "environment", "severity", "status"),
    ),
)


def _quoted_columns(columns: tuple[str, ...]) -> str:
    return ", ".join(f'"{column}"' for column in columns)


def upgrade() -> None:
    for index_name, table_name, columns in INDEXES:
        op.execute(
            f'CREATE INDEX IF NOT EXISTS "{index_name}" '
            f'ON "{table_name}" ({_quoted_columns(columns)})'
        )


def downgrade() -> None:
    for index_name, _table_name, _columns in reversed(INDEXES):
        op.execute(f'DROP INDEX IF EXISTS "{index_name}"')
