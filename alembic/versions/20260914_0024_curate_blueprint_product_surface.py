"""curate_blueprint_product_surface

Revision ID: 20260914_0024
Revises: 20260910_0023
Create Date: 2026-09-14 10:00:00.000000

"""
from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import sqlalchemy as sa
from alembic import op


revision = "20260914_0024"
down_revision = "20260910_0023"
branch_labels = None
depends_on = None


DEPRECATED_DIAGRAMS: dict[str, str] = {
    "application_architecture": "Duplicado de c4_container",
    "logical_architecture": "Duplicado de architecture_overview",
    "solution_architecture": "Duplicado conceptual de c4_container",
    "data_lineage": "Duplicado de data_lineage_map",
    "capability_map": "Duplicado de target_capabilities_map",
    "context_map": "Solapado con c4_context",
    "bpmn_process": "Redundante con current_process_map",
    "activity_diagram": "Redundante con runtime_workflow",
    "state_diagram": "Duplicado de runtime_workflow",
    "integration_architecture": "Absorbido en c4_container",
    "integration_boundaries": "Absorbido en c4_container",
    "use_case_diagram": "UML obsoleto. Ver target_capabilities_map",
    "ux_navigation_flow": "Sin UI. Ver user_journey",
    "entity_relationship": "No agentico. Ver tools.contracts",
    "logical_data_model": "No agentico. Ver tools.contracts",
    "tool_contract_flow": "Nivel ACP. Ver tool_capability_map",
    "traceability_matrix": "Ver tabla en definition.requirements",
}

DEPRECATED_DELIVERABLES: dict[str, str] = {
    "discovery.problem_context_brief": "Fusionado en discovery.analysis",
    "definition.requirements_brief": "Fusionado en definition.requirements",
    "blueprint.patterns": "Fusionado en blueprint.architecture_spec",
    "acp.implementation_questions": "Fusionado en acp.gap_register",
    "diagrams.blueprint_bundle": "Integrado en proceso de PDF Pro",
    "acp.package_manifest": "Portada del ZIP ACP. No es artefacto de usuario",
    "provenance.producer_trace": "Log interno. No expuesto en UI",
    "diagram.traceability_matrix": "Ver tabla en definition.requirements",
}

TIER_OVERRIDES: dict[str, tuple[str, str]] = {
    "blueprint.architecture_spec": ("blueprint_pro", "limited"),
    "tools.minimum_set": ("blueprint_pro", "limited"),
    "memory.strategy": ("blueprint_pro", "limited"),
    "estimate.comparison": ("blueprint_pro", "limited"),
}

PRO_DIAGRAM_OVERRIDES: dict[str, tuple[str, str]] = {
    "agent_orchestration": ("blueprint_pro", "limited"),
    "security_guardrails": ("blueprint_pro", "limited"),
}


def _has_table(table_name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(table_name)


def _now() -> datetime:
    return datetime.utcnow()


def _json_type() -> sa.types.TypeEngine:
    return sa.JSON()


def _deliverable_table() -> sa.Table:
    return sa.table(
        "deliverable_governance_v1",
        sa.column("id", sa.String()),
        sa.column("scope_key", sa.String()),
        sa.column("workspace_id", sa.String()),
        sa.column("deliverable_key", sa.String()),
        sa.column("enabled", sa.Boolean()),
        sa.column("generation_enabled", sa.Boolean()),
        sa.column("required_tier_override", sa.String()),
        sa.column("preview_mode_override", sa.String()),
        sa.column("prompt_status", sa.String()),
        sa.column("prompt_override", _json_type()),
        sa.column("notes", sa.String()),
        sa.column("updated_by_user_id", sa.String()),
        sa.column("created_at", sa.DateTime()),
        sa.column("updated_at", sa.DateTime()),
    )


def _diagram_table() -> sa.Table:
    return sa.table(
        "diagram_governance_v3",
        sa.column("id", sa.String()),
        sa.column("diagram_key", sa.String()),
        sa.column("enabled", sa.Boolean()),
        sa.column("generation_enabled", sa.Boolean()),
        sa.column("required_tier_override", sa.String()),
        sa.column("preview_mode_override", sa.String()),
        sa.column("prompt_status", sa.String()),
        sa.column("prompt_override", _json_type()),
        sa.column("notes", sa.String()),
        sa.column("updated_by_user_id", sa.String()),
        sa.column("created_at", sa.DateTime()),
        sa.column("updated_at", sa.DateTime()),
    )


def _upsert_deliverable(
    table: sa.Table,
    *,
    deliverable_key: str,
    enabled: bool,
    generation_enabled: bool,
    prompt_status: str,
    notes: str,
    required_tier_override: str = "",
    preview_mode_override: str = "",
) -> None:
    bind = op.get_bind()
    existing_id = bind.execute(
        sa.select(table.c.id).where(
            table.c.scope_key == "platform",
            table.c.deliverable_key == deliverable_key,
        )
    ).scalar_one_or_none()
    values = {
        "enabled": enabled,
        "generation_enabled": generation_enabled,
        "required_tier_override": required_tier_override,
        "preview_mode_override": preview_mode_override,
        "prompt_status": prompt_status,
        "prompt_override": {},
        "notes": f"Curacion Blueprint 2026-09: {notes}",
        "updated_at": _now(),
    }
    if existing_id:
        bind.execute(
            table.update()
            .where(table.c.scope_key == "platform", table.c.deliverable_key == deliverable_key)
            .values(**values)
        )
        return

    bind.execute(
        table.insert().values(
            id=str(uuid4()),
            scope_key="platform",
            workspace_id=None,
            deliverable_key=deliverable_key,
            updated_by_user_id=None,
            created_at=_now(),
            **values,
        )
    )


def _upsert_diagram(
    table: sa.Table,
    *,
    diagram_key: str,
    enabled: bool,
    generation_enabled: bool,
    prompt_status: str,
    notes: str,
    required_tier_override: str = "",
    preview_mode_override: str = "",
) -> None:
    bind = op.get_bind()
    existing_id = bind.execute(sa.select(table.c.id).where(table.c.diagram_key == diagram_key)).scalar_one_or_none()
    values = {
        "enabled": enabled,
        "generation_enabled": generation_enabled,
        "required_tier_override": required_tier_override,
        "preview_mode_override": preview_mode_override,
        "prompt_status": prompt_status,
        "prompt_override": {},
        "notes": f"Curacion Blueprint 2026-09: {notes}",
        "updated_at": _now(),
    }
    if existing_id:
        bind.execute(table.update().where(table.c.diagram_key == diagram_key).values(**values))
        return

    bind.execute(
        table.insert().values(
            id=str(uuid4()),
            diagram_key=diagram_key,
            updated_by_user_id=None,
            created_at=_now(),
            **values,
        )
    )


def upgrade() -> None:
    if _has_table("diagram_governance_v3"):
        diagram_table = _diagram_table()
        for diagram_key, notes in DEPRECATED_DIAGRAMS.items():
            _upsert_diagram(
                diagram_table,
                diagram_key=diagram_key,
                enabled=False,
                generation_enabled=False,
                prompt_status="deprecated",
                notes=notes,
            )
        for diagram_key, (tier, preview_mode) in PRO_DIAGRAM_OVERRIDES.items():
            _upsert_diagram(
                diagram_table,
                diagram_key=diagram_key,
                enabled=True,
                generation_enabled=True,
                prompt_status="active",
                notes=f"Movido a {tier}",
                required_tier_override=tier,
                preview_mode_override=preview_mode,
            )

    if _has_table("deliverable_governance_v1"):
        deliverable_table = _deliverable_table()
        for deliverable_key, notes in DEPRECATED_DELIVERABLES.items():
            _upsert_deliverable(
                deliverable_table,
                deliverable_key=deliverable_key,
                enabled=False,
                generation_enabled=False,
                prompt_status="deprecated",
                notes=notes,
            )
        for deliverable_key, (tier, preview_mode) in TIER_OVERRIDES.items():
            _upsert_deliverable(
                deliverable_table,
                deliverable_key=deliverable_key,
                enabled=True,
                generation_enabled=True,
                prompt_status="active",
                notes=f"Movido a {tier}",
                required_tier_override=tier,
                preview_mode_override=preview_mode,
            )


def downgrade() -> None:
    if _has_table("diagram_governance_v3"):
        diagram_table = _diagram_table()
        bind = op.get_bind()
        for diagram_key in DEPRECATED_DIAGRAMS:
            bind.execute(
                diagram_table.update()
                .where(diagram_table.c.diagram_key == diagram_key)
                .values(enabled=True, generation_enabled=True, prompt_status="active", notes="Rollback curacion Blueprint 2026-09", updated_at=_now())
            )
        for diagram_key in PRO_DIAGRAM_OVERRIDES:
            bind.execute(
                diagram_table.update()
                .where(diagram_table.c.diagram_key == diagram_key)
                .values(required_tier_override="", preview_mode_override="", notes="Rollback curacion Blueprint 2026-09", updated_at=_now())
            )

    if _has_table("deliverable_governance_v1"):
        deliverable_table = _deliverable_table()
        bind = op.get_bind()
        for deliverable_key in DEPRECATED_DELIVERABLES:
            bind.execute(
                deliverable_table.update()
                .where(deliverable_table.c.scope_key == "platform", deliverable_table.c.deliverable_key == deliverable_key)
                .values(enabled=True, generation_enabled=True, prompt_status="active", notes="Rollback curacion Blueprint 2026-09", updated_at=_now())
            )
        for deliverable_key in TIER_OVERRIDES:
            bind.execute(
                deliverable_table.update()
                .where(deliverable_table.c.scope_key == "platform", deliverable_table.c.deliverable_key == deliverable_key)
                .values(required_tier_override="", preview_mode_override="", notes="Rollback curacion Blueprint 2026-09", updated_at=_now())
            )
