from __future__ import annotations

from sqlmodel import Session, select

from app.models import CommercialTier, JourneyStageArtifactRecord, SessionRecord, WorkspaceRole
from app.services.diagram_center.contracts import DiagramPolicyDecision, DiagramRegistryEntry
from app.services.diagram_center.persistence import DiagramGovernanceRecord


STAGE_ORDER = {
    "discover": 0,
    "define": 1,
    "design": 2,
    "tools": 3,
    "memory": 4,
    "validate": 5,
    "estimate": 6,
    "package": 7,
}

SESSION_STAGE_MAP = {
    "draft_capture": "discover",
    "input_validation": "discover",
    "normalize_discovery": "discover",
    "build_canvas": "define",
    "build_blueprint": "design",
    "post_validation": "validate",
    "ready_for_export": "estimate",
}

TIER_ORDER = {"blueprint": 1, "blueprint_pro": 2, "acp": 3, "enterprise": 4}
WRITE_ROLES = {WorkspaceRole.owner, WorkspaceRole.admin, WorkspaceRole.editor}


def resolve_project_stage(db: Session, record: SessionRecord) -> str:
    resolved = SESSION_STAGE_MAP.get(record.current_stage.value, "discover")
    artifact_stages = db.exec(
        select(JourneyStageArtifactRecord.stage_key).where(JourneyStageArtifactRecord.session_id == record.id)
    ).all()
    candidates = [resolved, *(str(value) for value in artifact_stages)]
    valid = [value for value in candidates if value in STAGE_ORDER]
    if record.commercial_tier == CommercialTier.acp and resolved == "estimate":
        valid.append("package")
    return max(valid, key=lambda value: STAGE_ORDER[value]) if valid else resolved


def effective_registry_policy(
    db: Session,
    entry: DiagramRegistryEntry,
) -> tuple[bool, bool, str, str, DiagramGovernanceRecord | None]:
    governance = db.exec(
        select(DiagramGovernanceRecord).where(DiagramGovernanceRecord.diagram_key == entry.key)
    ).first()
    enabled = governance.enabled if governance is not None else entry.active
    generation_enabled = governance.generation_enabled if governance is not None else True
    required_tier = governance.required_tier_override if governance and governance.required_tier_override else entry.required_tier
    preview_mode = governance.preview_mode_override if governance and governance.preview_mode_override else entry.preview_mode
    return enabled, generation_enabled, required_tier, preview_mode, governance


def resolve_diagram_policy(
    *,
    entry: DiagramRegistryEntry,
    project_stage: str,
    current_tier: str,
    role: WorkspaceRole | None,
    enabled: bool,
    generation_enabled: bool,
    required_tier: str,
    preview_mode: str,
) -> DiagramPolicyDecision:
    can_write = role in WRITE_ROLES
    if not enabled:
        return DiagramPolicyDecision(
            access_state="disabled",
            can_generate=False,
            reason_code="disabled_by_admin",
            reason="Este tipo de diagrama fue deshabilitado por el administrador de la plataforma.",
            cta_label="Contactar al administrador",
            required_tier=required_tier,
        )

    if STAGE_ORDER.get(project_stage, 0) < STAGE_ORDER.get(entry.stage, 0):
        return DiagramPolicyDecision(
            access_state="stage_locked",
            can_generate=False,
            reason_code="lean_stage_required",
            reason=f"Se habilita al completar la etapa {entry.stage.capitalize()} con sus artefactos aprobados.",
            cta_label=f"Continuar a {entry.stage.capitalize()}",
            required_tier=required_tier,
        )

    has_tier = TIER_ORDER.get(current_tier, 0) >= TIER_ORDER.get(required_tier, 0)
    if has_tier:
        return DiagramPolicyDecision(
            access_state="available",
            can_generate=generation_enabled and can_write,
            can_view=True,
            can_download=current_tier in {"blueprint_pro", "acp", "enterprise"},
            can_regenerate=generation_enabled and can_write,
            can_compare=current_tier in {"blueprint_pro", "acp", "enterprise"},
            reason_code="entitled",
            reason="Disponible según el plan, la etapa y las reglas administrativas vigentes.",
            required_tier=required_tier,
        )

    if preview_mode in {"full", "limited"}:
        return DiagramPolicyDecision(
            access_state="preview",
            can_view=True,
            reason_code="plan_preview",
            reason=f"Vista previa disponible. El acceso completo requiere el plan {required_tier.replace('_', ' ').title()}.",
            cta_label=f"Conocer {required_tier.replace('_', ' ').title()}",
            required_tier=required_tier,
        )

    product = "ACP" if required_tier == "acp" else "Blueprint Profesional"
    return DiagramPolicyDecision(
        access_state="locked",
        reason_code="plan_required",
        reason=f"Este diagrama es parte de {product}: {entry.benefit}",
        cta_label=f"Adquirir {product}",
        required_tier=required_tier,
    )

