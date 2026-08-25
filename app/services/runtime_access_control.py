from __future__ import annotations

from sqlmodel import Session

from app.models import UserRecord
from app.services.workspace_membership_service import is_platform_admin_user
from app.services.workspace_access import WorkspaceAccessContext


def is_platform_admin(session: Session, user: UserRecord) -> bool:
    return is_platform_admin_user(session, user=user)


def ensure_platform_admin(session: Session, user: UserRecord) -> None:
    if not is_platform_admin(session, user):
        raise PermissionError("Solo un platform admin puede ejecutar esta accion.")


def ensure_workspace_runtime_admin(
    session: Session,
    user: UserRecord,
    workspace_context: WorkspaceAccessContext,
) -> None:
    del workspace_context
    # Legacy compatibility name: runtime and technical administration are now
    # governed globally by the platform admin role only.
    ensure_platform_admin(session, user)
