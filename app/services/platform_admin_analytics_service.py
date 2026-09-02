from __future__ import annotations

from collections import defaultdict
from uuid import UUID

from sqlalchemy import func
from sqlmodel import Session, select

from app.models import (
    HotmartIntegrationConfigRecord,
    LLMUsageLedgerRecord,
    PlatformAdminMembershipSummary,
    PlatformAdminProjectListResponse,
    PlatformAdminProjectSummary,
    PlatformAdminUserListResponse,
    PlatformAdminUserSummary,
    PlatformAdminWorkspaceListResponse,
    PlatformAdminWorkspaceSummary,
    PlatformRoleAssignmentRecord,
    SessionRecord,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceRecord,
    WorkspaceRuntimeSettingsRecord,
)


def _workspace_map(session: Session) -> dict[UUID, WorkspaceRecord]:
    return {workspace.id: workspace for workspace in session.exec(select(WorkspaceRecord)).all()}


def _platform_roles_by_user(session: Session) -> dict[UUID, list]:
    roles: dict[UUID, list] = defaultdict(list)
    rows = session.exec(
        select(PlatformRoleAssignmentRecord).where(PlatformRoleAssignmentRecord.is_active == True)  # noqa: E712
    ).all()
    for row in rows:
        roles[row.user_id].append(row.role)
    return roles


def _project_counts_by_user(session: Session) -> dict[UUID, int]:
    counts: dict[UUID, int] = defaultdict(int)
    rows = session.exec(select(SessionRecord.user_id, func.count()).group_by(SessionRecord.user_id)).all()
    for user_id, total in rows:
        counts[user_id] = int(total or 0)
    return counts


def _project_counts_by_workspace(session: Session) -> dict[UUID, int]:
    counts: dict[UUID, int] = defaultdict(int)
    rows = session.exec(select(SessionRecord.workspace_id, func.count()).group_by(SessionRecord.workspace_id)).all()
    for workspace_id, total in rows:
        counts[workspace_id] = int(total or 0)
    return counts


def _usage_by_user(session: Session) -> dict[UUID, tuple[int, float]]:
    usage: dict[UUID, tuple[int, float]] = defaultdict(lambda: (0, 0.0))
    rows = session.exec(
        select(
            LLMUsageLedgerRecord.user_id,
            func.coalesce(func.sum(LLMUsageLedgerRecord.total_tokens), 0),
            func.coalesce(func.sum(LLMUsageLedgerRecord.cost_total), 0),
        ).group_by(LLMUsageLedgerRecord.user_id)
    ).all()
    for user_id, tokens, cost in rows:
        if user_id is not None:
            usage[user_id] = (int(tokens or 0), float(cost or 0))
    return usage


def _usage_by_project(session: Session) -> dict[UUID, tuple[int, float]]:
    usage: dict[UUID, tuple[int, float]] = defaultdict(lambda: (0, 0.0))
    rows = session.exec(
        select(
            LLMUsageLedgerRecord.session_id,
            func.coalesce(func.sum(LLMUsageLedgerRecord.total_tokens), 0),
            func.coalesce(func.sum(LLMUsageLedgerRecord.cost_total), 0),
        ).group_by(LLMUsageLedgerRecord.session_id)
    ).all()
    for session_id, tokens, cost in rows:
        if session_id is not None:
            usage[session_id] = (int(tokens or 0), float(cost or 0))
    return usage


def list_platform_admin_users(
    session: Session,
    *,
    email: str = "",
    workspace_id: UUID | None = None,
    limit: int = 100,
) -> PlatformAdminUserListResponse:
    workspaces = _workspace_map(session)
    roles = _platform_roles_by_user(session)
    project_counts = _project_counts_by_user(session)
    usage = _usage_by_user(session)
    membership_rows = session.exec(select(WorkspaceMembershipRecord)).all()
    memberships_by_user: dict[UUID, list[WorkspaceMembershipRecord]] = defaultdict(list)
    allowed_user_ids: set[UUID] | None = None
    if workspace_id is not None:
        allowed_user_ids = set()
    for membership in membership_rows:
        memberships_by_user[membership.user_id].append(membership)
        if workspace_id is not None and membership.workspace_id == workspace_id:
            allowed_user_ids.add(membership.user_id)

    statement = select(UserRecord).order_by(UserRecord.created_at.desc(), UserRecord.email.asc()).limit(max(1, min(limit, 500)))
    rows = session.exec(statement).all()
    normalized_email = email.strip().lower()
    users: list[PlatformAdminUserSummary] = []
    for user in rows:
        if normalized_email and normalized_email not in user.email.lower():
            continue
        if allowed_user_ids is not None and user.id not in allowed_user_ids:
            continue
        memberships = memberships_by_user.get(user.id, [])
        tokens, cost = usage[user.id]
        users.append(
            PlatformAdminUserSummary(
                id=user.id,
                email=user.email,
                full_name=user.full_name,
                is_active=user.is_active,
                default_workspace_id=user.default_workspace_id,
                platform_roles=roles.get(user.id, []),
                workspace_count=len({membership.workspace_id for membership in memberships}),
                active_workspace_count=len({membership.workspace_id for membership in memberships if membership.is_active}),
                project_count=project_counts[user.id],
                llm_total_tokens=tokens,
                llm_total_cost=cost,
                memberships=[
                    PlatformAdminMembershipSummary(
                        workspace_id=membership.workspace_id,
                        workspace_name=workspaces.get(membership.workspace_id).name if workspaces.get(membership.workspace_id) else "",
                        workspace_slug=workspaces.get(membership.workspace_id).slug if workspaces.get(membership.workspace_id) else "",
                        role=membership.role,
                        is_active=membership.is_active,
                        created_at=membership.created_at,
                        updated_at=membership.updated_at,
                    )
                    for membership in memberships
                ],
                created_at=user.created_at,
                updated_at=user.updated_at,
            )
        )
    return PlatformAdminUserListResponse(users=users, total=len(users))


def list_platform_admin_workspaces(
    session: Session,
    *,
    limit: int = 100,
) -> PlatformAdminWorkspaceListResponse:
    project_counts = _project_counts_by_workspace(session)
    users = {user.id: user for user in session.exec(select(UserRecord)).all()}
    memberships = session.exec(select(WorkspaceMembershipRecord)).all()
    memberships_by_workspace: dict[UUID, list[WorkspaceMembershipRecord]] = defaultdict(list)
    for membership in memberships:
        memberships_by_workspace[membership.workspace_id].append(membership)
    runtime_rows = session.exec(
        select(WorkspaceRuntimeSettingsRecord).where(WorkspaceRuntimeSettingsRecord.is_active == True)  # noqa: E712
    ).all()
    runtime_by_workspace = {row.workspace_id: row for row in runtime_rows}
    hotmart_rows = session.exec(select(HotmartIntegrationConfigRecord)).all()
    hotmart_by_workspace = {row.workspace_id: row for row in hotmart_rows}
    workspaces = session.exec(
        select(WorkspaceRecord).order_by(WorkspaceRecord.created_at.desc(), WorkspaceRecord.name.asc()).limit(max(1, min(limit, 500)))
    ).all()
    return PlatformAdminWorkspaceListResponse(
        workspaces=[
            PlatformAdminWorkspaceSummary(
                id=workspace.id,
                name=workspace.name,
                slug=workspace.slug,
                owner_emails=[
                    users[membership.user_id].email
                    for membership in memberships_by_workspace.get(workspace.id, [])
                    if membership.role.value == "owner" and membership.user_id in users
                ],
                member_count=len([membership for membership in memberships_by_workspace.get(workspace.id, []) if membership.is_active]),
                project_count=project_counts[workspace.id],
                active_runtime_provider=runtime_by_workspace[workspace.id].active_provider if workspace.id in runtime_by_workspace else None,
                uses_platform_credentials=runtime_by_workspace[workspace.id].uses_platform_credentials if workspace.id in runtime_by_workspace else None,
                hotmart_enabled=hotmart_by_workspace[workspace.id].enabled if workspace.id in hotmart_by_workspace else None,
                hotmart_status=hotmart_by_workspace[workspace.id].status if workspace.id in hotmart_by_workspace else "",
                created_at=workspace.created_at,
                updated_at=workspace.updated_at,
            )
            for workspace in workspaces
        ],
        total=len(workspaces),
    )


def list_platform_admin_projects(
    session: Session,
    *,
    workspace_id: UUID | None = None,
    limit: int = 100,
) -> PlatformAdminProjectListResponse:
    workspaces = _workspace_map(session)
    users = {user.id: user for user in session.exec(select(UserRecord)).all()}
    usage = _usage_by_project(session)
    statement = select(SessionRecord).order_by(SessionRecord.updated_at.desc(), SessionRecord.created_at.desc()).limit(max(1, min(limit, 500)))
    if workspace_id is not None:
        statement = statement.where(SessionRecord.workspace_id == workspace_id)
    rows = session.exec(statement).all()
    projects: list[PlatformAdminProjectSummary] = []
    for project in rows:
        tokens, cost = usage[project.id]
        projects.append(
            PlatformAdminProjectSummary(
                id=project.id,
                workspace_id=project.workspace_id,
                workspace_name=workspaces.get(project.workspace_id).name if workspaces.get(project.workspace_id) else "",
                user_id=project.user_id,
                owner_email=users[project.user_id].email if project.user_id in users else "",
                title=project.title,
                current_stage=project.current_stage,
                status=project.status,
                commercial_tier=project.commercial_tier,
                llm_total_tokens=tokens,
                llm_total_cost=cost,
                created_at=project.created_at,
                updated_at=project.updated_at,
            )
        )
    return PlatformAdminProjectListResponse(projects=projects, total=len(projects))
