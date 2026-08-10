from __future__ import annotations

from app.models import CommercialTier, DiagramAccessPolicy, DiagramContentProtection, WorkspaceRole
from app.services.artifact_diagram_taxonomy import get_diagram_taxonomy_entries


DEFAULT_DIAGRAM_PROTECTION = DiagramContentProtection(
    disable_copy=True,
    disable_context_menu=True,
    disable_download=True,
    watermark_sample=True,
)


def _requires_purchase(required_tier: CommercialTier, access_level: str) -> bool:
    if access_level in {"premium", "downloadable", "restricted"}:
        return required_tier != CommercialTier.blueprint
    return required_tier in {CommercialTier.blueprint_pro, CommercialTier.acp}


def build_diagram_access_policy(entry: dict) -> DiagramAccessPolicy:
    formats = entry.get("formats") if isinstance(entry.get("formats"), dict) else {}
    required_tier = CommercialTier(entry.get("required_tier", CommercialTier.blueprint.value))
    access_level = str(entry.get("access_level") or "view_only")
    return DiagramAccessPolicy(
        diagram_key=str(entry.get("diagram_key") or ""),
        title=str(entry.get("title") or ""),
        category=str(entry.get("category") or ""),
        description=str(entry.get("description") or ""),
        enabled_from_stage=str(entry.get("enabled_from_stage") or ""),
        product_scope=[str(item) for item in entry.get("product_scope", []) if str(item).strip()],
        required_tier=required_tier,
        access_level=access_level,
        diagram_surface=str(entry.get("diagram_surface") or ""),
        sample_enabled=access_level == "sample" or required_tier == CommercialTier.blueprint,
        sample_tier=CommercialTier.blueprint,
        visible_to_user_types=[role.value for role in WorkspaceRole],
        requires_purchase=_requires_purchase(required_tier, access_level),
        default_generation_state=entry.get("default_generation_state", "pending_generation"),
        content_protection=DEFAULT_DIAGRAM_PROTECTION,
        upsell={str(key): str(value) for key, value in (entry.get("upsell") or {}).items()},
        preferred_format=str(formats.get("preferred") or ""),
        available_formats=[str(item) for item in formats.get("available", []) if str(item).strip()],
        source_artifact_keys=[str(item) for item in entry.get("source_artifact_keys", []) if str(item).strip()],
        portable_paths=[str(item) for item in entry.get("portable_paths", []) if str(item).strip()],
        sort_order=int(entry.get("sort_order") or 0),
        is_active=bool(entry.get("is_active", True)),
    )


def list_diagram_access_policies() -> list[DiagramAccessPolicy]:
    return sorted(
        [
            build_diagram_access_policy(entry)
            for entry in get_diagram_taxonomy_entries()
            if entry.get("is_active", True)
        ],
        key=lambda item: (item.sort_order, item.diagram_key),
    )


def get_diagram_access_policy(diagram_key: str) -> DiagramAccessPolicy | None:
    normalized = diagram_key.strip()
    for policy in list_diagram_access_policies():
        if policy.diagram_key == normalized:
            return policy
    return None
