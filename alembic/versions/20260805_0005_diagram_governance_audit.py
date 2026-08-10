"""add diagram governance audit trail

Revision ID: 20260805_0005
Revises: 20260805_0004
Create Date: 2026-08-05
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260805_0005"
down_revision: str | None = "20260805_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "diagram_governance_audit_v3",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("diagram_key", sa.String(), nullable=False),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("changed_fields", postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column("before_payload", postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column("after_payload", postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column("actor_user_id", sa.Uuid(), nullable=True),
        sa.Column("reason", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["actor_user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_diagram_governance_audit_v3_diagram_key", "diagram_governance_audit_v3", ["diagram_key"])
    op.create_index("ix_diagram_governance_audit_v3_action", "diagram_governance_audit_v3", ["action"])
    op.create_index("ix_diagram_governance_audit_v3_actor_user_id", "diagram_governance_audit_v3", ["actor_user_id"])
    op.create_index("ix_diagram_governance_audit_v3_created_at", "diagram_governance_audit_v3", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_diagram_governance_audit_v3_created_at", table_name="diagram_governance_audit_v3")
    op.drop_index("ix_diagram_governance_audit_v3_actor_user_id", table_name="diagram_governance_audit_v3")
    op.drop_index("ix_diagram_governance_audit_v3_action", table_name="diagram_governance_audit_v3")
    op.drop_index("ix_diagram_governance_audit_v3_diagram_key", table_name="diagram_governance_audit_v3")
    op.drop_table("diagram_governance_audit_v3")
