from __future__ import annotations

from sqlmodel import Session, select

from app.models import PlatformRole, PlatformRoleAssignmentRecord, UserRecord, WorkspaceRole
from app.services.runtime_governance_bootstrap import backfill_platform_runtime_governance
from app.services.workspace_access import WorkspaceAccessContext


WORKSPACE_RUNTIME_ADMIN_ROLES = {WorkspaceRole.owner, WorkspaceRole.admin}


def is_platform_admin(session: Session, user: UserRecord) -> bool:
    backfill_platform_runtime_governance(session)
    assignment = session.exec(
        select(PlatformRoleAssignmentRecord).where(
            PlatformRoleAssignmentRecord.user_id == user.id,
            PlatformRoleAssignmentRecord.role == PlatformRole.platform_admin,
            PlatformRoleAssignmentRecord.is_active == True,  # noqa: E712
        )
    ).first()
    return assignment is not None


def ensure_platform_admin(session: Session, user: UserRecord) -> None:
    if not is_platform_admin(session, user):
        raise PermissionError("Solo un platform admin puede ejecutar esta accion.")


def ensure_workspace_runtime_admin(
    session: Session,
    user: UserRecord,
    workspace_context: WorkspaceAccessContext,
) -> None:
    if is_platform_admin(session, user):
        return
    if workspace_context.membership.role not in WORKSPACE_RUNTIME_ADMIN_ROLES:
        raise PermissionError("Solo un workspace owner/admin puede ejecutar esta accion.")
