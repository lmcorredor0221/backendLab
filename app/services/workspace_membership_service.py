from __future__ import annotations

from uuid import UUID

from sqlmodel import Session, select

from app.models import (
    PlatformRole,
    PlatformRoleAssignmentRecord,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceRecord,
    WorkspaceRole,
)
from app.services.runtime_governance_bootstrap import backfill_platform_runtime_governance


PLATFORM_ADMIN_WORKSPACE_ROLE = WorkspaceRole.owner


def is_platform_admin_user_id(session: Session, *, user_id: UUID) -> bool:
    backfill_platform_runtime_governance(session)
    assignment = session.exec(
        select(PlatformRoleAssignmentRecord).where(
            PlatformRoleAssignmentRecord.user_id == user_id,
            PlatformRoleAssignmentRecord.role == PlatformRole.platform_admin,
            PlatformRoleAssignmentRecord.is_active == True,  # noqa: E712
        )
    ).first()
    return assignment is not None


def is_platform_admin_user(session: Session, *, user: UserRecord) -> bool:
    return is_platform_admin_user_id(session, user_id=user.id)


def list_active_platform_roles_for_user(session: Session, *, user: UserRecord) -> list[PlatformRole]:
    backfill_platform_runtime_governance(session)
    assignments = session.exec(
        select(PlatformRoleAssignmentRecord).where(
            PlatformRoleAssignmentRecord.user_id == user.id,
            PlatformRoleAssignmentRecord.is_active == True,  # noqa: E712
        )
    ).all()
    seen: set[PlatformRole] = set()
    roles: list[PlatformRole] = []
    for assignment in assignments:
        if assignment.role in seen:
            continue
        seen.add(assignment.role)
        roles.append(assignment.role)
    return roles


def get_effective_workspace_membership(
    session: Session,
    *,
    workspace_id: UUID,
    user_id: UUID,
) -> WorkspaceMembershipRecord | None:
    membership = session.exec(
        select(WorkspaceMembershipRecord).where(
            WorkspaceMembershipRecord.workspace_id == workspace_id,
            WorkspaceMembershipRecord.user_id == user_id,
            WorkspaceMembershipRecord.is_active == True,  # noqa: E712
        )
    ).first()
    if membership is not None:
        return membership

    workspace = session.get(WorkspaceRecord, workspace_id)
    if workspace is None or not workspace.is_active:
        return None

    if not is_platform_admin_user_id(session, user_id=user_id):
        return None

    return WorkspaceMembershipRecord(
        workspace_id=workspace_id,
        user_id=user_id,
        role=PLATFORM_ADMIN_WORKSPACE_ROLE,
        is_active=True,
    )


def list_effective_workspace_memberships(
    session: Session,
    *,
    user: UserRecord,
) -> list[tuple[WorkspaceMembershipRecord, WorkspaceRecord]]:
    actual_memberships = session.exec(
        select(WorkspaceMembershipRecord)
        .where(WorkspaceMembershipRecord.user_id == user.id, WorkspaceMembershipRecord.is_active == True)  # noqa: E712
        .order_by(WorkspaceMembershipRecord.created_at.asc())
    ).all()
    workspace_ids = [item.workspace_id for item in actual_memberships]
    workspace_by_id: dict[UUID, WorkspaceRecord] = {}
    if workspace_ids:
        workspaces = session.exec(select(WorkspaceRecord).where(WorkspaceRecord.id.in_(workspace_ids))).all()
        workspace_by_id = {item.id: item for item in workspaces if item.is_active}

    resolved: list[tuple[WorkspaceMembershipRecord, WorkspaceRecord]] = []
    for membership in actual_memberships:
        workspace = workspace_by_id.get(membership.workspace_id)
        if workspace is None:
            continue
        resolved.append((membership, workspace))

    if not is_platform_admin_user_id(session, user_id=user.id):
        return resolved

    synthetic_workspace_ids = {workspace.id for _, workspace in resolved}
    all_active_workspaces = session.exec(
        select(WorkspaceRecord)
        .where(WorkspaceRecord.is_active == True)  # noqa: E712
        .order_by(WorkspaceRecord.created_at.asc())
    ).all()
    for workspace in all_active_workspaces:
        if workspace.id in synthetic_workspace_ids:
            continue
        resolved.append(
            (
                WorkspaceMembershipRecord(
                    workspace_id=workspace.id,
                    user_id=user.id,
                    role=PLATFORM_ADMIN_WORKSPACE_ROLE,
                    is_active=True,
                ),
                workspace,
            )
        )
    return resolved
