from __future__ import annotations

from dataclasses import dataclass
from typing import cast
from uuid import UUID

from fastapi import Depends, Header, HTTPException, status
from sqlmodel import Session, select

from app.db import get_session
from app.models import (
    AuthUser,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceMembershipSummary,
    WorkspaceRecord,
    WorkspaceRole,
    utc_now,
)
from app.services.auth_service import get_current_user
from app.services.workspace_membership_service import (
    is_platform_admin_user,
    list_active_platform_roles_for_user,
    list_effective_workspace_memberships,
)


@dataclass(frozen=True)
class WorkspaceAccessContext:
    workspace: WorkspaceRecord
    membership: WorkspaceMembershipRecord


def _coerce_uuid(value: UUID | str | None) -> UUID | None:
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            return UUID(candidate)
        except ValueError:
            return None
    return None


def _slugify(value: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "-" for char in value.strip())
    compact = "-".join(token for token in normalized.split("-") if token)
    return compact or "workspace"


def _workspace_name_for_user(user: UserRecord) -> str:
    label = user.full_name.strip() or user.email.split("@")[0].strip() or "Workspace"
    return f"{label} Workspace"


def list_workspace_memberships(session: Session, user: UserRecord) -> list[tuple[WorkspaceMembershipRecord, WorkspaceRecord]]:
    membership_rows = session.exec(
        select(WorkspaceMembershipRecord)
        .where(WorkspaceMembershipRecord.user_id == user.id, WorkspaceMembershipRecord.is_active == True)  # noqa: E712
        .order_by(WorkspaceMembershipRecord.created_at.asc())
    ).all()
    if not membership_rows:
        return []
    workspace_ids = [item.workspace_id for item in membership_rows]
    workspaces = session.exec(select(WorkspaceRecord).where(WorkspaceRecord.id.in_(workspace_ids))).all()
    workspace_by_id = {item.id: item for item in workspaces if item.is_active}
    resolved: list[tuple[WorkspaceMembershipRecord, WorkspaceRecord]] = []
    for membership in membership_rows:
        workspace = workspace_by_id.get(membership.workspace_id)
        if workspace is None:
            continue
        resolved.append((membership, workspace))
    return resolved


def workspace_ids_for_user(session: Session, user: UserRecord) -> set[UUID]:
    return {workspace.id for _, workspace in list_workspace_memberships(session, user)}


def _persist_default_workspace(session: Session, user: UserRecord, workspace_id: UUID) -> None:
    if _coerce_uuid(cast(UUID | str | None, user.default_workspace_id)) == workspace_id:
        return
    user.default_workspace_id = workspace_id
    user.updated_at = utc_now()
    session.add(user)
    session.commit()
    session.refresh(user)


def ensure_personal_workspace(session: Session, user: UserRecord, default_name: str = "") -> WorkspaceAccessContext:
    memberships = list_workspace_memberships(session, user)
    if memberships:
        default_workspace_id = _coerce_uuid(cast(UUID | str | None, user.default_workspace_id))
        default_workspace = next((workspace for _, workspace in memberships if workspace.id == default_workspace_id), None)
        default_membership = next((membership for membership, workspace in memberships if workspace.id == default_workspace_id), None)
        if default_workspace is not None and default_membership is not None:
            return WorkspaceAccessContext(workspace=default_workspace, membership=default_membership)
        membership, workspace = memberships[0]
        user.default_workspace_id = workspace.id
        user.updated_at = utc_now()
        session.add(user)
        session.commit()
        session.refresh(user)
        return WorkspaceAccessContext(workspace=workspace, membership=membership)

    name = default_name.strip() if default_name and default_name.strip() else _workspace_name_for_user(user)
    workspace = WorkspaceRecord(
        name=name,
        slug=f"{_slugify(name)}-{str(user.id)[:8]}",
        created_by_user_id=user.id,
    )
    session.add(workspace)
    session.flush()
    membership = WorkspaceMembershipRecord(
        workspace_id=workspace.id,
        user_id=user.id,
        role=WorkspaceRole.owner,
    )
    session.add(membership)
    user.default_workspace_id = workspace.id
    user.updated_at = utc_now()
    session.add(user)
    from app.services.commercial_quota_service import initialize_workspace_commercial_quota

    initialize_workspace_commercial_quota(
        session,
        workspace_id=workspace.id,
        actor_user_id=user.id,
    )
    session.commit()
    session.refresh(user)
    session.refresh(workspace)
    session.refresh(membership)
    return WorkspaceAccessContext(workspace=workspace, membership=membership)


def build_auth_user(
    session: Session,
    user: UserRecord,
    *,
    requested_workspace_id: UUID | None = None,
) -> AuthUser:
    active_context = resolve_workspace_access(session, user, requested_workspace_id=requested_workspace_id)
    memberships = list_effective_workspace_memberships(session, user=user)
    platform_roles = [role.value for role in list_active_platform_roles_for_user(session, user=user)]
    return AuthUser(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        preferred_currency=(user.preferred_currency or "COP").strip().upper() or "COP",
        active_workspace_id=active_context.workspace.id,
        active_workspace_name=active_context.workspace.name,
        platform_roles=platform_roles,
        workspaces=[
            WorkspaceMembershipSummary(
                workspace_id=workspace.id,
                workspace_name=workspace.name,
                workspace_slug=workspace.slug,
                role=membership.role,
                is_active=membership.is_active and workspace.is_active,
            )
            for membership, workspace in memberships
        ],
    )


def resolve_workspace_access(
    session: Session,
    user: UserRecord,
    *,
    requested_workspace_id: UUID | None = None,
) -> WorkspaceAccessContext:
    personal_context = ensure_personal_workspace(session, user)
    memberships = list_workspace_memberships(session, user)
    if not memberships:
        return personal_context

    membership_by_workspace_id = {workspace.id: (membership, workspace) for membership, workspace in memberships}
    resolved_requested_workspace_id = _coerce_uuid(requested_workspace_id)
    default_workspace_id = _coerce_uuid(cast(UUID | str | None, user.default_workspace_id))
    target_workspace_id = resolved_requested_workspace_id or default_workspace_id or memberships[0][1].id

    resolved_membership = membership_by_workspace_id.get(target_workspace_id)
    if resolved_membership is not None:
        membership, workspace = resolved_membership
        _persist_default_workspace(session, user, workspace.id)
        return WorkspaceAccessContext(workspace=workspace, membership=membership)

    if is_platform_admin_user(session, user=user):
        workspace = session.get(WorkspaceRecord, target_workspace_id)
        if workspace is not None and workspace.is_active:
            _persist_default_workspace(session, user, workspace.id)
            return WorkspaceAccessContext(
                workspace=workspace,
                membership=WorkspaceMembershipRecord(
                    workspace_id=workspace.id,
                    user_id=user.id,
                    role=WorkspaceRole.owner,
                    is_active=True,
                ),
            )

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="The requested workspace is not available for the current user.",
    )


def backfill_user_default_workspaces(session: Session) -> None:
    users = session.exec(select(UserRecord)).all()
    for user in users:
        ensure_personal_workspace(session, user)


def get_current_workspace_context(
    workspace_id: UUID | None = Header(default=None, alias="x-workspace-id"),
    current_user: UserRecord = Depends(get_current_user),
    db: Session = Depends(get_session),
) -> WorkspaceAccessContext:
    return resolve_workspace_access(db, current_user, requested_workspace_id=workspace_id)
