from __future__ import annotations

from uuid import UUID

from sqlmodel import Session, select

from app.models import CommercialTier, WorkspaceRole, utc_now
from app.services.commercial_access import tier_rank
from app.services.deliverable_catalog.contracts import (
    DeliverableGovernanceAuditEntry,
    DeliverableGovernanceEntry,
    DeliverableGovernanceUpdate,
    DeliverablePolicyContext,
    DeliverablePolicyDecision,
    DeliverableRegistryEntry,
    LEAN_STAGE_ORDER,
)
from app.services.deliverable_catalog.persistence import (
    DeliverableGovernanceAuditRecord,
    DeliverableGovernanceRecord,
)


WRITE_ROLES = {WorkspaceRole.owner, WorkspaceRole.admin, WorkspaceRole.editor}
ADMIN_ROLES = {WorkspaceRole.owner, WorkspaceRole.admin}
LEGACY_STAGE_MAP = {
    "draft_capture": "discover",
    "input_validation": "discover",
    "normalize_discovery": "discover",
    "build_canvas": "define",
    "build_blueprint": "design",
    "post_validation": "validate",
    "ready_for_export": "package",
}


def normalize_deliverable_stage(stage: str) -> str:
    normalized = str(stage or "").strip().lower()
    return LEGACY_STAGE_MAP.get(normalized, normalized)


def scope_key_for_workspace(workspace_id: UUID | None) -> str:
    return str(workspace_id) if workspace_id is not None else "platform"


def _stage_index(stage: str) -> int:
    try:
        return LEAN_STAGE_ORDER.index(normalize_deliverable_stage(stage))
    except ValueError:
        return -1


def _has_reached_stage(current_stage: str, enabled_from_stage: str) -> bool:
    current = _stage_index(current_stage)
    required = _stage_index(enabled_from_stage)
    return current >= required >= 0


def _effective_record(
    db: Session,
    entry: DeliverableRegistryEntry,
    *,
    workspace_id: UUID | None,
) -> DeliverableGovernanceRecord | None:
    records = db.exec(
        select(DeliverableGovernanceRecord).where(
            DeliverableGovernanceRecord.deliverable_key == entry.deliverable_key,
            DeliverableGovernanceRecord.scope_key.in_(["platform", scope_key_for_workspace(workspace_id)]),
        )
    ).all()
    by_scope = {record.scope_key: record for record in records}
    return by_scope.get(scope_key_for_workspace(workspace_id)) or by_scope.get("platform")


def effective_deliverable_governance(
    db: Session,
    entry: DeliverableRegistryEntry,
    *,
    workspace_id: UUID | None,
) -> tuple[bool, bool, CommercialTier, str, str, dict[str, object], DeliverableGovernanceRecord | None]:
    governance = _effective_record(db, entry, workspace_id=workspace_id)
    enabled = governance.enabled if governance is not None else True
    generation_enabled = governance.generation_enabled if governance is not None else True
    required_tier = (
        CommercialTier(governance.required_tier_override)
        if governance is not None and governance.required_tier_override
        else entry.required_tier
    )
    preview_mode = (
        governance.preview_mode_override
        if governance is not None and governance.preview_mode_override
        else entry.access_policy.preview_mode
    )
    prompt_status = governance.prompt_status if governance is not None else entry.prompt_policy.prompt_status
    prompt_override = governance.prompt_override if governance is not None else {}
    return enabled, generation_enabled, required_tier, preview_mode, prompt_status, prompt_override, governance


def resolve_deliverable_policy(
    db: Session,
    entry: DeliverableRegistryEntry,
    context: DeliverablePolicyContext,
) -> DeliverablePolicyDecision:
    enabled, generation_enabled, required_tier, preview_mode, prompt_status, _, _ = effective_deliverable_governance(
        db,
        entry,
        workspace_id=context.workspace_id,
    )
    can_write = context.role in WRITE_ROLES
    can_admin = context.role in ADMIN_ROLES
    prompt_allows_generation = prompt_status not in {"paused", "deprecated"}

    if not enabled or not entry.active:
        return DeliverablePolicyDecision(
            visible=True,
            access_state="disabled",
            reason_code="deliverable_disabled",
            reason="Este entregable esta deshabilitado por governance.",
            required_tier=required_tier,
            effective_prompt_status=prompt_status,
            preview_mode=preview_mode,
        )

    if not _has_reached_stage(context.current_stage, entry.enabled_from_stage):
        return DeliverablePolicyDecision(
            visible=True,
            access_state="stage_locked",
            reason_code="stage_not_reached",
            reason=f"Se habilita al llegar a la etapa {entry.enabled_from_stage}.",
            cta_label=f"Ir a {entry.enabled_from_stage}",
            required_tier=required_tier,
            effective_prompt_status=prompt_status,
            preview_mode=preview_mode,
        )

    if tier_rank(context.tier) < tier_rank(required_tier):
        if entry.access_policy.sample_enabled and preview_mode != "none":
            return DeliverablePolicyDecision(
                visible=True,
                access_state="preview",
                can_view=True,
                reason_code="preview_available",
                reason="Vista de muestra disponible; adquiere el producto requerido para acceso completo.",
                cta_label="Ver valor",
                required_tier=required_tier,
                effective_prompt_status=prompt_status,
                preview_mode=preview_mode,
            )
        return DeliverablePolicyDecision(
            visible=True,
            access_state="locked",
            reason_code="tier_required",
            reason=f"Requiere plan {required_tier.value}.",
            cta_label="Adquirir",
            required_tier=required_tier,
            effective_prompt_status=prompt_status,
            preview_mode=preview_mode,
        )

    if context.quality_state == "failed":
        return DeliverablePolicyDecision(
            visible=True,
            access_state="quality_failed",
            can_view=context.has_current_version,
            can_generate=can_write and generation_enabled and prompt_allows_generation,
            can_regenerate=can_write and generation_enabled and prompt_allows_generation,
            can_compare=context.has_current_version,
            reason_code="quality_failed",
            reason="El ultimo resultado no supero los criterios de calidad.",
            cta_label="Regenerar",
            required_tier=required_tier,
            effective_prompt_status=prompt_status,
            preview_mode=preview_mode,
        )

    if context.quality_state == "stale":
        return DeliverablePolicyDecision(
            visible=True,
            access_state="stale",
            can_view=context.has_current_version,
            can_generate=can_write and generation_enabled and prompt_allows_generation,
            can_regenerate=can_write and generation_enabled and prompt_allows_generation,
            can_download=False,
            can_compare=context.has_current_version,
            reason_code="stale_dependency",
            reason="Hay cambios en dependencias que requieren reprocesar el entregable.",
            cta_label="Actualizar",
            required_tier=required_tier,
            effective_prompt_status=prompt_status,
            preview_mode=preview_mode,
        )

    if not context.has_current_version and context.generation_state not in {"available", "generating", "queued", "updating"}:
        return DeliverablePolicyDecision(
            visible=True,
            access_state="not_generated",
            can_generate=can_write and generation_enabled and prompt_allows_generation,
            reason_code="not_generated",
            reason="El entregable esta definido en catalogo pero aun no se ha generado.",
            cta_label="Generar",
            required_tier=required_tier,
            effective_prompt_status=prompt_status,
            preview_mode=preview_mode,
        )

    is_premium_tier = tier_rank(context.tier) > tier_rank(CommercialTier.blueprint)
    return DeliverablePolicyDecision(
        visible=True,
        access_state="available",
        can_view=True,
        can_generate=can_write and generation_enabled and prompt_allows_generation and not context.has_current_version,
        can_regenerate=False,
        can_download=can_write and entry.exportable and is_premium_tier and tier_rank(context.tier) >= tier_rank(required_tier),
        can_compare=context.has_current_version and is_premium_tier,
        can_edit_prompt=can_admin,
        reason_code="allowed",
        reason="Entregable disponible para el contexto actual.",
        required_tier=required_tier,
        effective_prompt_status=prompt_status,
        preview_mode=preview_mode,
    )


def _governance_payload(record: DeliverableGovernanceRecord | None, entry: DeliverableRegistryEntry) -> dict[str, object]:
    if record is None:
        return {
            "enabled": True,
            "generation_enabled": True,
            "required_tier": entry.required_tier.value,
            "preview_mode": entry.access_policy.preview_mode,
            "prompt_status": entry.prompt_policy.prompt_status,
            "prompt_override": {},
        }
    return {
        "enabled": record.enabled,
        "generation_enabled": record.generation_enabled,
        "required_tier": record.required_tier_override or entry.required_tier.value,
        "preview_mode": record.preview_mode_override or entry.access_policy.preview_mode,
        "prompt_status": record.prompt_status,
        "prompt_override": record.prompt_override,
    }


def deliverable_governance_entry(
    db: Session,
    entry: DeliverableRegistryEntry,
    *,
    workspace_id: UUID | None = None,
) -> DeliverableGovernanceEntry:
    enabled, generation_enabled, required_tier, preview_mode, prompt_status, prompt_override, governance = (
        effective_deliverable_governance(db, entry, workspace_id=workspace_id)
    )
    return DeliverableGovernanceEntry(
        deliverable_key=entry.deliverable_key,
        title=entry.title,
        description=entry.description,
        deliverable_type=entry.deliverable_type,
        category=entry.category,
        stage=entry.stage,
        enabled_from_stage=entry.enabled_from_stage,
        product_scope=list(entry.product_scope),
        access_level=entry.access_level,
        formats=entry.formats,
        generation_mode=entry.generation_mode,
        prompt_policy=entry.prompt_policy,
        context_policy=entry.context_policy,
        quality_policy=entry.quality_policy,
        dependency_policy=entry.dependency_policy,
        access_policy=entry.access_policy,
        canonical_paths=list(entry.canonical_paths),
        portable_paths=list(entry.portable_paths),
        exportable=entry.exportable,
        blueprint_download=entry.blueprint_download,
        acp_download=entry.acp_download,
        active=entry.active,
        scope_key=scope_key_for_workspace(workspace_id),
        workspace_id=workspace_id,
        enabled=enabled,
        generation_enabled=generation_enabled,
        required_tier=required_tier,
        preview_mode=preview_mode,
        prompt_status=prompt_status,
        prompt_override=prompt_override,
        notes=governance.notes if governance is not None else "",
        updated_at=governance.updated_at if governance is not None else None,
    )


def governance_audit_entry(record: DeliverableGovernanceAuditRecord) -> DeliverableGovernanceAuditEntry:
    return DeliverableGovernanceAuditEntry(
        id=record.id,
        deliverable_key=record.deliverable_key,
        scope_key=record.scope_key,
        action=record.action,
        changed_fields=record.changed_fields,
        actor_user_id=record.actor_user_id,
        reason=record.reason,
        created_at=record.created_at,
    )


def upsert_deliverable_governance(
    db: Session,
    entry: DeliverableRegistryEntry,
    payload: DeliverableGovernanceUpdate,
    *,
    workspace_id: UUID | None = None,
    actor_user_id: UUID | None = None,
) -> DeliverableGovernanceEntry:
    scope_key = scope_key_for_workspace(workspace_id)
    governance = db.exec(
        select(DeliverableGovernanceRecord).where(
            DeliverableGovernanceRecord.scope_key == scope_key,
            DeliverableGovernanceRecord.deliverable_key == entry.deliverable_key,
        )
    ).first()
    before = _governance_payload(governance, entry)
    if governance is None:
        governance = DeliverableGovernanceRecord(
            scope_key=scope_key,
            workspace_id=workspace_id,
            deliverable_key=entry.deliverable_key,
        )
    governance.enabled = payload.enabled
    governance.generation_enabled = payload.generation_enabled
    governance.required_tier_override = payload.required_tier_override
    governance.preview_mode_override = payload.preview_mode_override
    governance.prompt_status = payload.prompt_status
    governance.prompt_override = payload.prompt_override
    governance.notes = payload.notes
    governance.updated_by_user_id = actor_user_id
    governance.updated_at = utc_now()
    db.add(governance)
    db.flush()
    after = _governance_payload(governance, entry)
    changed_fields = [key for key, value in after.items() if before.get(key) != value]
    db.add(
        DeliverableGovernanceAuditRecord(
            scope_key=scope_key,
            workspace_id=workspace_id,
            deliverable_key=entry.deliverable_key,
            changed_fields=changed_fields,
            before_payload=before,
            after_payload=after,
            actor_user_id=actor_user_id,
            reason=payload.notes,
        )
    )
    db.flush()
    return deliverable_governance_entry(db, entry, workspace_id=workspace_id)
