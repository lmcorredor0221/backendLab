from __future__ import annotations

from uuid import UUID

from sqlmodel import Session

from app.services.workspace_bootstrap import resolve_platform_admin_template_workspace_id


def resolve_commerce_provider_configuration_workspace_id(
    session: Session,
    *,
    workspace_id: UUID,
) -> UUID:
    platform_workspace_id = resolve_platform_admin_template_workspace_id(session)
    return platform_workspace_id or workspace_id
