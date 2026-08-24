from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlmodel import Session

from app.models import (
    CommercialAccessSnapshotV2,
    CommercialCapabilityDecisionEntry,
    CommercialTier,
    SessionCommercialAccess,
    SessionRecord,
    UserRecord,
    WorkspaceRole,
)
from app.services.commerce_service import (
    resolve_effective_entitlement_state,
    role_for_user,
)


CommercialCapability = str

COMMERCIAL_PRODUCTS = ("blueprint", "blueprint_pro", "acp")

TIER_LABELS: dict[CommercialTier, str] = {
    CommercialTier.blueprint: "Blueprint",
    CommercialTier.blueprint_pro: "Blueprint Profesional",
    CommercialTier.acp: "ACP Premium",
}

TIER_RANKS: dict[CommercialTier, int] = {
    CommercialTier.blueprint: 1,
    CommercialTier.blueprint_pro: 2,
    CommercialTier.acp: 3,
}

WRITE_ROLES = frozenset({WorkspaceRole.owner, WorkspaceRole.admin, WorkspaceRole.editor})


@dataclass(frozen=True)
class CapabilityPolicy:
    capability: CommercialCapability
    required_tier: CommercialTier
    product: str
    label: str
    allowed_roles: frozenset[WorkspaceRole] | None = None
    purchase_required: bool = False


@dataclass(frozen=True)
class CommercialEntitlementContext:
    workspace_id: UUID | None
    user_id: UUID | None
    role: WorkspaceRole | None
    tier: CommercialTier
    purchase_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class CommercialCapabilityDecision:
    capability: CommercialCapability
    allowed: bool
    current_tier: CommercialTier
    required_tier: CommercialTier
    product: str
    label: str
    workspace_id: UUID | None = None
    user_id: UUID | None = None
    role: WorkspaceRole | None = None
    purchase_refs: tuple[str, ...] = ()
    reason: str = "allowed"


CAPABILITY_POLICIES: dict[CommercialCapability, CapabilityPolicy] = {
    "blueprint.view": CapabilityPolicy(
        capability="blueprint.view",
        required_tier=CommercialTier.blueprint,
        product="blueprint",
        label="visualizar el Blueprint dentro de la plataforma",
    ),
    "blueprint.download": CapabilityPolicy(
        capability="blueprint.download",
        required_tier=CommercialTier.blueprint_pro,
        product="blueprint_pro",
        label="descargar el Blueprint Profesional",
        allowed_roles=WRITE_ROLES,
        purchase_required=True,
    ),
    "acp.invite": CapabilityPolicy(
        capability="acp.invite",
        required_tier=CommercialTier.blueprint,
        product="acp",
        label="ver la invitacion comercial del ACP",
    ),
    "acp.build": CapabilityPolicy(
        capability="acp.build",
        required_tier=CommercialTier.acp,
        product="acp",
        label="ejecutar funcionalidades de construccion ACP",
        allowed_roles=WRITE_ROLES,
        purchase_required=True,
    ),
    "acp.download": CapabilityPolicy(
        capability="acp.download",
        required_tier=CommercialTier.acp,
        product="acp",
        label="descargar el ACP portable",
        allowed_roles=WRITE_ROLES,
        purchase_required=True,
    ),
    "diagram.view.sample": CapabilityPolicy(
        capability="diagram.view.sample",
        required_tier=CommercialTier.blueprint,
        product="blueprint",
        label="visualizar diagramas de muestra",
    ),
    "diagram.view.blueprint": CapabilityPolicy(
        capability="diagram.view.blueprint",
        required_tier=CommercialTier.blueprint_pro,
        product="blueprint_pro",
        label="visualizar diagramas incluidos en el Blueprint Profesional",
        purchase_required=True,
    ),
    "diagram.view.acp": CapabilityPolicy(
        capability="diagram.view.acp",
        required_tier=CommercialTier.acp,
        product="acp",
        label="visualizar diagramas exclusivos del ACP",
        purchase_required=True,
    ),
    "export_markdown": CapabilityPolicy(
        capability="export_markdown",
        required_tier=CommercialTier.blueprint_pro,
        product="blueprint_pro",
        label="descargar el Blueprint Profesional en markdown",
        allowed_roles=WRITE_ROLES,
        purchase_required=True,
    ),
    "export_json": CapabilityPolicy(
        capability="export_json",
        required_tier=CommercialTier.blueprint_pro,
        product="blueprint_pro",
        label="descargar el Blueprint Profesional en JSON",
        allowed_roles=WRITE_ROLES,
        purchase_required=True,
    ),
    "export_blueprint_core": CapabilityPolicy(
        capability="export_blueprint_core",
        required_tier=CommercialTier.blueprint_pro,
        product="blueprint_pro",
        label="exportar Blueprint Core",
        allowed_roles=WRITE_ROLES,
        purchase_required=True,
    ),
    "export_estimation_pack": CapabilityPolicy(
        capability="export_estimation_pack",
        required_tier=CommercialTier.blueprint_pro,
        product="blueprint_pro",
        label="exportar Estimation Pack",
        allowed_roles=WRITE_ROLES,
        purchase_required=True,
    ),
    "export_construction_pack": CapabilityPolicy(
        capability="export_construction_pack",
        required_tier=CommercialTier.acp,
        product="acp",
        label="exportar Construction Pack",
        allowed_roles=WRITE_ROLES,
        purchase_required=True,
    ),
    "export_prompt_pack": CapabilityPolicy(
        capability="export_prompt_pack",
        required_tier=CommercialTier.acp,
        product="acp",
        label="exportar Prompt Pack",
        allowed_roles=WRITE_ROLES,
        purchase_required=True,
    ),
    "export_test_pack": CapabilityPolicy(
        capability="export_test_pack",
        required_tier=CommercialTier.acp,
        product="acp",
        label="exportar Test Pack",
        allowed_roles=WRITE_ROLES,
        purchase_required=True,
    ),
    "export_acp_zip": CapabilityPolicy(
        capability="export_acp_zip",
        required_tier=CommercialTier.acp,
        product="acp",
        label="descargar el ACP zip",
        allowed_roles=WRITE_ROLES,
        purchase_required=True,
    ),
    "library_workspace": CapabilityPolicy(
        capability="library_workspace",
        required_tier=CommercialTier.acp,
        product="acp",
        label="acceder a la biblioteca tecnica del ACP",
        allowed_roles=WRITE_ROLES,
        purchase_required=True,
    ),
}

CAPABILITY_REQUIRED_TIER: dict[CommercialCapability, CommercialTier] = {
    key: policy.required_tier for key, policy in CAPABILITY_POLICIES.items()
}

CAPABILITY_LABELS: dict[CommercialCapability, str] = {
    key: policy.label for key, policy in CAPABILITY_POLICIES.items()
}

ENTITLEMENT_MATRIX_CAPABILITIES: tuple[CommercialCapability, ...] = (
    "blueprint.view",
    "blueprint.download",
    "acp.invite",
    "acp.build",
    "acp.download",
    "diagram.view.sample",
    "diagram.view.blueprint",
    "diagram.view.acp",
)


@dataclass(frozen=True)
class CommercialAccessViolation:
    capability: CommercialCapability
    current_tier: CommercialTier
    required_tier: CommercialTier
    label: str = ""
    reason: str = "tier"
    role: WorkspaceRole | None = None
    workspace_id: UUID | None = None
    user_id: UUID | None = None

    @property
    def message(self) -> str:
        label = self.label or CAPABILITY_LABELS.get(self.capability, self.capability)
        if self.reason == "role":
            return f"Tu rol actual no permite {label}. Solicita acceso a un owner o admin del workspace."
        return (
            f"Tu plan actual es {TIER_LABELS[self.current_tier]}. "
            f"Actualiza a {TIER_LABELS[self.required_tier]} para {label}."
        )


def tier_rank(tier: CommercialTier) -> int:
    return TIER_RANKS[tier]


def policy_for_capability(capability: CommercialCapability) -> CapabilityPolicy:
    try:
        return CAPABILITY_POLICIES[capability]
    except KeyError as exc:
        raise KeyError(f"Unknown commercial capability: {capability}") from exc


def _purchase_refs_for_tier(record: SessionRecord) -> tuple[str, ...]:
    if record.commercial_tier == CommercialTier.blueprint:
        return ()
    return (f"legacy-session-tier:{record.id}:{record.commercial_tier.value}",)


def build_entitlement_context(
    *,
    tier: CommercialTier,
    workspace_id: UUID | None = None,
    user_id: UUID | None = None,
    role: WorkspaceRole | None = None,
    purchase_refs: tuple[str, ...] | list[str] | None = None,
) -> CommercialEntitlementContext:
    return CommercialEntitlementContext(
        workspace_id=workspace_id,
        user_id=user_id,
        role=role,
        tier=tier,
        purchase_refs=tuple(purchase_refs or ()),
    )


def resolve_session_entitlement_context(
    db: Session,
    record: SessionRecord,
    current_user: UserRecord,
) -> CommercialEntitlementContext:
    effective = resolve_effective_entitlement_state(db, record)
    role = role_for_user(db, workspace_id=record.workspace_id, user_id=current_user.id)
    return build_entitlement_context(
        tier=effective.tier,
        workspace_id=record.workspace_id,
        user_id=current_user.id,
        role=role,
        purchase_refs=effective.purchase_refs,
    )


def resolve_session_commercial_access(
    db: Session,
    record: SessionRecord,
    *,
    current_user: UserRecord | None = None,
) -> SessionCommercialAccess:
    effective = resolve_effective_entitlement_state(db, record)
    role = role_for_user(db, workspace_id=record.workspace_id, user_id=current_user.id) if current_user is not None else None
    return build_session_commercial_access(
        effective.tier,
        role=role,
        reason_code=effective.reason_code,
        checkout_state=effective.checkout_state,
        purchase_refs=list(effective.purchase_refs),
    )


def resolve_capability_access(
    context: CommercialEntitlementContext,
    capability: CommercialCapability,
) -> CommercialCapabilityDecision:
    policy = policy_for_capability(capability)
    if tier_rank(context.tier) < tier_rank(policy.required_tier):
        return CommercialCapabilityDecision(
            capability=capability,
            allowed=False,
            current_tier=context.tier,
            required_tier=policy.required_tier,
            product=policy.product,
            label=policy.label,
            workspace_id=context.workspace_id,
            user_id=context.user_id,
            role=context.role,
            purchase_refs=context.purchase_refs,
            reason="tier",
        )
    if context.role is not None and policy.allowed_roles is not None and context.role not in policy.allowed_roles:
        return CommercialCapabilityDecision(
            capability=capability,
            allowed=False,
            current_tier=context.tier,
            required_tier=policy.required_tier,
            product=policy.product,
            label=policy.label,
            workspace_id=context.workspace_id,
            user_id=context.user_id,
            role=context.role,
            purchase_refs=context.purchase_refs,
            reason="role",
        )
    return CommercialCapabilityDecision(
        capability=capability,
        allowed=True,
        current_tier=context.tier,
        required_tier=policy.required_tier,
        product=policy.product,
        label=policy.label,
        workspace_id=context.workspace_id,
        user_id=context.user_id,
        role=context.role,
        purchase_refs=context.purchase_refs,
    )


def has_capability(
    tier: CommercialTier,
    capability: CommercialCapability,
    *,
    role: WorkspaceRole | None = None,
) -> bool:
    context = build_entitlement_context(tier=tier, role=role)
    return resolve_capability_access(context, capability).allowed


def validate_capability(
    tier: CommercialTier,
    capability: CommercialCapability,
    *,
    context: CommercialEntitlementContext | None = None,
) -> CommercialAccessViolation | None:
    entitlement_context = context or build_entitlement_context(tier=tier)
    decision = resolve_capability_access(entitlement_context, capability)
    if decision.allowed:
        return None
    return CommercialAccessViolation(
        capability=capability,
        current_tier=decision.current_tier,
        required_tier=decision.required_tier,
        label=decision.label,
        reason=decision.reason,
        role=decision.role,
        workspace_id=decision.workspace_id,
        user_id=decision.user_id,
    )


def build_entitlement_matrix(
    context: CommercialEntitlementContext,
    capabilities: tuple[CommercialCapability, ...] = ENTITLEMENT_MATRIX_CAPABILITIES,
) -> list[CommercialCapabilityDecision]:
    return [resolve_capability_access(context, capability) for capability in capabilities]


def build_session_commercial_access(
    tier: CommercialTier,
    *,
    role: WorkspaceRole | None = None,
    reason_code: str = "free_access",
    checkout_state: str = "not_started",
    purchase_refs: list[str] | None = None,
) -> SessionCommercialAccess:
    available_upgrades = [
        candidate
        for candidate in (CommercialTier.blueprint_pro, CommercialTier.acp)
        if tier_rank(candidate) > tier_rank(tier)
    ]
    upgrade_target = available_upgrades[0] if available_upgrades else tier
    context = build_entitlement_context(tier=tier, role=role)
    matrix = build_entitlement_matrix(context, tuple(CAPABILITY_POLICIES.keys()))
    return SessionCommercialAccess(
        tier=tier,
        tier_label=TIER_LABELS[tier],
        tier_rank=tier_rank(tier),
        reason_code=reason_code,
        checkout_state=checkout_state,
        purchase_refs=purchase_refs or [],
        capability_reasons={item.capability: item.reason for item in matrix if not item.allowed},
        upgrade_cta_label=f"Upgrade a {TIER_LABELS[upgrade_target]}",
        upgrade_message=(
            "Explora el Blueprint en la plataforma y desbloquea exportables profesionales o el paquete ACP completo "
            "cuando quieras llevarlo a implementacion."
        ),
        can_view_in_app_blueprint=resolve_capability_access(context, "blueprint.view").allowed,
        can_view_blueprint=resolve_capability_access(context, "blueprint.view").allowed,
        can_download_blueprint=resolve_capability_access(context, "blueprint.download").allowed,
        can_export_blueprint_document=resolve_capability_access(context, "export_markdown").allowed,
        can_export_markdown=resolve_capability_access(context, "export_markdown").allowed,
        can_export_json=resolve_capability_access(context, "export_json").allowed,
        can_export_blueprint_core=resolve_capability_access(context, "export_blueprint_core").allowed,
        can_export_estimation_pack=resolve_capability_access(context, "export_estimation_pack").allowed,
        can_export_construction_pack=resolve_capability_access(context, "export_construction_pack").allowed,
        can_export_prompt_pack=resolve_capability_access(context, "export_prompt_pack").allowed,
        can_export_test_pack=resolve_capability_access(context, "export_test_pack").allowed,
        can_export_acp_zip=resolve_capability_access(context, "export_acp_zip").allowed,
        can_invite_acp=resolve_capability_access(context, "acp.invite").allowed,
        can_build_acp=resolve_capability_access(context, "acp.build").allowed,
        can_download_acp=resolve_capability_access(context, "acp.download").allowed,
        can_view_diagram_sample=resolve_capability_access(context, "diagram.view.sample").allowed,
        can_view_diagram_blueprint=resolve_capability_access(context, "diagram.view.blueprint").allowed,
        can_view_diagram_acp=resolve_capability_access(context, "diagram.view.acp").allowed,
        can_access_library_workspace=resolve_capability_access(context, "library_workspace").allowed,
        available_upgrades=available_upgrades,
    )


def build_commercial_access_snapshot_v2(
    db: Session,
    record: SessionRecord,
    *,
    current_user: UserRecord | None = None,
) -> CommercialAccessSnapshotV2:
    effective = resolve_effective_entitlement_state(db, record)
    role = role_for_user(db, workspace_id=record.workspace_id, user_id=current_user.id) if current_user is not None else None
    context = build_entitlement_context(
        tier=effective.tier,
        workspace_id=record.workspace_id,
        user_id=current_user.id if current_user is not None else None,
        role=role,
        purchase_refs=effective.purchase_refs,
    )
    decisions = build_entitlement_matrix(context, tuple(CAPABILITY_POLICIES.keys()))
    return CommercialAccessSnapshotV2(
        workspace_id=record.workspace_id,
        session_id=record.id,
        user_id=current_user.id if current_user is not None else None,
        role=role,
        tier=effective.tier,
        tier_label=TIER_LABELS[effective.tier],
        reason_code=effective.reason_code,
        checkout_state=effective.checkout_state,
        purchase_refs=list(effective.purchase_refs),
        entitlements=list(effective.entitlements),
        capabilities=[
            CommercialCapabilityDecisionEntry(
                capability=item.capability,
                allowed=item.allowed,
                current_tier=item.current_tier,
                required_tier=item.required_tier,
                product=item.product,
                label=item.label,
                reason_code=item.reason,
                cta_label="Solicitar acceso" if item.reason == "role" else f"Adquirir {TIER_LABELS[item.required_tier]}",
            )
            for item in decisions
        ],
    )
