from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field as PydanticField
from sqlmodel import Session, select

from app.db import get_session
from app.models import (
    AdminUserInvitationRecord,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceRole,
    utc_now,
)
from app.services.admin_console_analytics import (
    AdminAnalyticsFilters,
    AdminConsoleAnalyticsService,
    audit_admin_change,
)
from app.services.auth_service import get_current_user
from app.services.runtime_access_control import ensure_platform_admin
from app.services.workspace_access import WorkspaceAccessContext, get_current_workspace_context


router = APIRouter(prefix="/admin", tags=["admin-console"])


class AdminUserPatchRequest(BaseModel):
    full_name: str | None = None
    is_active: bool | None = None
    membership_role: WorkspaceRole | None = None
    membership_is_active: bool | None = None


class AdminUserInvitationCreateRequest(BaseModel):
    email: str
    full_name: str = ""
    role: WorkspaceRole = WorkspaceRole.viewer
    expires_at: datetime | None = None
    message: str = ""
    metadata: dict[str, Any] = PydanticField(default_factory=dict)


@router.get("/overview")
def get_admin_overview_route(
    started_from: datetime | None = None,
    started_to: datetime | None = None,
    user_id: UUID | None = None,
    project_id: UUID | None = None,
    stage: str = "",
    provider_key: str = "",
    model_name: str = "",
    granularity: str = Query(default="day", pattern="^(day|week|month)$"),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> dict[str, Any]:
    _ensure_admin(db, current_user=current_user, workspace_context=workspace_context)
    service = AdminConsoleAnalyticsService()
    return service.overview(
        db,
        workspace=workspace_context.workspace,
        filters=_filters(
            workspace_context=workspace_context,
            platform_scope=True,
            started_from=started_from,
            started_to=started_to,
            user_id=user_id,
            project_id=project_id,
            stage=stage,
            provider_key=provider_key,
            model_name=model_name,
            granularity=granularity,
        ),
    )


@router.get("/projects/analytics")
def get_admin_projects_analytics_route(
    started_from: datetime | None = None,
    started_to: datetime | None = None,
    user_id: UUID | None = None,
    project_id: UUID | None = None,
    stage: str = "",
    granularity: str = Query(default="day", pattern="^(day|week|month)$"),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> dict[str, Any]:
    _ensure_admin(db, current_user=current_user, workspace_context=workspace_context)
    return AdminConsoleAnalyticsService().project_analytics(
        db,
        filters=_filters(
            workspace_context=workspace_context,
            platform_scope=True,
            started_from=started_from,
            started_to=started_to,
            user_id=user_id,
            project_id=project_id,
            stage=stage,
            granularity=granularity,
        ),
    )


@router.get("/users")
def list_admin_users_route(
    search: str = "",
    role: str = "",
    status_filter: str = Query(default="all", alias="status", pattern="^(all|active|inactive)$"),
    started_from: datetime | None = None,
    started_to: datetime | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> dict[str, Any]:
    _ensure_admin(db, current_user=current_user, workspace_context=workspace_context)
    return AdminConsoleAnalyticsService().list_users(
        db,
        filters=_filters(
            workspace_context=workspace_context,
            started_from=started_from,
            started_to=started_to,
        ),
        search=search,
        role=role,
        status_filter=status_filter,
        limit=limit,
        offset=offset,
    )


@router.get("/users/invitations")
def list_admin_user_invitations_route(
    status_filter: str = Query(default="pending", alias="status"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> dict[str, Any]:
    _ensure_admin(db, current_user=current_user, workspace_context=workspace_context)
    return AdminConsoleAnalyticsService().list_invitations(
        db,
        filters=_filters(workspace_context=workspace_context),
        status_filter=status_filter,
        limit=limit,
        offset=offset,
    )


@router.post("/users/invitations", status_code=status.HTTP_201_CREATED)
def create_admin_user_invitation_route(
    payload: AdminUserInvitationCreateRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> dict[str, Any]:
    _ensure_admin(db, current_user=current_user, workspace_context=workspace_context)
    email = payload.email.strip().lower()
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email de invitacion invalido.")

    existing_membership = _membership_for_email(db, workspace_id=workspace_context.workspace.id, email=email)
    if existing_membership is not None and existing_membership.is_active:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="El usuario ya pertenece al workspace.")

    existing_invitation = db.exec(
        select(AdminUserInvitationRecord).where(
            AdminUserInvitationRecord.workspace_id == workspace_context.workspace.id,
            AdminUserInvitationRecord.email == email,
            AdminUserInvitationRecord.status == "pending",
        )
    ).first()
    before = _invitation_payload(existing_invitation) if existing_invitation else {}
    invitation = existing_invitation or AdminUserInvitationRecord(
        workspace_id=workspace_context.workspace.id,
        email=email,
        invited_by_user_id=current_user.id,
    )
    invitation.full_name = payload.full_name.strip()
    invitation.role = payload.role
    invitation.expires_at = payload.expires_at
    invitation.message = payload.message.strip()
    invitation.metadata_payload = payload.metadata
    invitation.status = "pending"
    invitation.updated_at = utc_now()
    db.add(invitation)
    audit_admin_change(
        db,
        workspace_id=workspace_context.workspace.id,
        actor=current_user,
        change_type="admin_user_invitation_create",
        before=before,
        after={
            "email": invitation.email,
            "role": invitation.role.value,
            "status": invitation.status,
            "delivery_status": "manual_delivery_required",
        },
    )
    db.commit()
    db.refresh(invitation)
    return _invitation_payload(invitation)


@router.patch("/users/{user_id}")
def patch_admin_user_route(
    user_id: UUID,
    payload: AdminUserPatchRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> dict[str, Any]:
    _ensure_admin(db, current_user=current_user, workspace_context=workspace_context)
    target_user = db.get(UserRecord, user_id)
    if target_user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado.")
    membership = db.exec(
        select(WorkspaceMembershipRecord).where(
            WorkspaceMembershipRecord.workspace_id == workspace_context.workspace.id,
            WorkspaceMembershipRecord.user_id == user_id,
        )
    ).first()
    if membership is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="El usuario no pertenece al workspace.")

    proposed_role = payload.membership_role if payload.membership_role is not None else membership.role
    proposed_membership_active = (
        payload.membership_is_active if payload.membership_is_active is not None else membership.is_active
    )
    proposed_user_active = payload.is_active if payload.is_active is not None else target_user.is_active
    if current_user.id == target_user.id and proposed_user_active is False:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No puedes desactivar tu propio usuario.")
    if _removes_active_owner(membership, proposed_role=proposed_role, proposed_active=proposed_membership_active):
        _ensure_another_active_owner(db, workspace_id=workspace_context.workspace.id, excluding_user_id=target_user.id)

    before = {
        "user": {
            "id": str(target_user.id),
            "full_name": target_user.full_name,
            "is_active": target_user.is_active,
        },
        "membership": {
            "role": membership.role.value,
            "is_active": membership.is_active,
        },
    }
    if payload.full_name is not None:
        target_user.full_name = payload.full_name.strip() or target_user.full_name
    if payload.is_active is not None:
        target_user.is_active = payload.is_active
    if payload.membership_role is not None:
        membership.role = payload.membership_role
    if payload.membership_is_active is not None:
        membership.is_active = payload.membership_is_active
    target_user.updated_at = utc_now()
    membership.updated_at = utc_now()
    db.add(target_user)
    db.add(membership)
    audit_admin_change(
        db,
        workspace_id=workspace_context.workspace.id,
        actor=current_user,
        change_type="admin_user_update",
        before=before,
        after={
            "user": {
                "id": str(target_user.id),
                "full_name": target_user.full_name,
                "is_active": target_user.is_active,
            },
            "membership": {
                "role": membership.role.value,
                "is_active": membership.is_active,
            },
        },
    )
    db.commit()
    return AdminConsoleAnalyticsService().list_users(
        db,
        filters=_filters(workspace_context=workspace_context),
        search=target_user.email,
        status_filter="all",
        limit=1,
        offset=0,
    )["items"][0]


@router.get("/roles")
def get_admin_roles_route(
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> dict[str, Any]:
    _ensure_admin(db, current_user=current_user, workspace_context=workspace_context)
    return AdminConsoleAnalyticsService().roles_catalog(
        db,
        workspace_id=workspace_context.workspace.id,
        current_user_id=current_user.id,
    )


@router.get("/activity")
def get_admin_activity_route(
    started_from: datetime | None = None,
    started_to: datetime | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> dict[str, Any]:
    _ensure_admin(db, current_user=current_user, workspace_context=workspace_context)
    return AdminConsoleAnalyticsService().activity_feed(
        db,
        filters=_filters(
            workspace_context=workspace_context,
            started_from=started_from,
            started_to=started_to,
        ),
        limit=limit,
    )


def _ensure_admin(
    db: Session,
    *,
    current_user: UserRecord,
    workspace_context: WorkspaceAccessContext,
) -> None:
    del workspace_context
    try:
        ensure_platform_admin(db, current_user)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc


def _filters(
    *,
    workspace_context: WorkspaceAccessContext,
    platform_scope: bool = False,
    started_from: datetime | None = None,
    started_to: datetime | None = None,
    user_id: UUID | None = None,
    project_id: UUID | None = None,
    stage: str = "",
    provider_key: str = "",
    model_name: str = "",
    granularity: str = "day",
) -> AdminAnalyticsFilters:
    return AdminAnalyticsFilters(
        workspace_id=None if platform_scope else workspace_context.workspace.id,
        started_from=started_from,
        started_to=started_to,
        user_id=user_id,
        project_id=project_id,
        stage=stage.strip(),
        provider_key=provider_key.strip(),
        model_name=model_name.strip(),
        granularity=granularity,
    )


def _membership_for_email(
    db: Session,
    *,
    workspace_id: UUID,
    email: str,
) -> WorkspaceMembershipRecord | None:
    user = db.exec(select(UserRecord).where(UserRecord.email == email)).first()
    if user is None:
        return None
    return db.exec(
        select(WorkspaceMembershipRecord).where(
            WorkspaceMembershipRecord.workspace_id == workspace_id,
            WorkspaceMembershipRecord.user_id == user.id,
        )
    ).first()


def _removes_active_owner(
    membership: WorkspaceMembershipRecord,
    *,
    proposed_role: WorkspaceRole,
    proposed_active: bool,
) -> bool:
    return membership.role == WorkspaceRole.owner and membership.is_active and (
        proposed_role != WorkspaceRole.owner or not proposed_active
    )


def _ensure_another_active_owner(db: Session, *, workspace_id: UUID, excluding_user_id: UUID) -> None:
    owners = list(
        db.exec(
            select(WorkspaceMembershipRecord).where(
                WorkspaceMembershipRecord.workspace_id == workspace_id,
                WorkspaceMembershipRecord.role == WorkspaceRole.owner,
                WorkspaceMembershipRecord.is_active == True,  # noqa: E712
                WorkspaceMembershipRecord.user_id != excluding_user_id,
            )
        ).all()
    )
    if not owners:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El workspace debe conservar al menos un owner activo.",
        )


def _invitation_payload(record: AdminUserInvitationRecord | None) -> dict[str, Any]:
    if record is None:
        return {}
    return {
        "id": str(record.id),
        "workspace_id": str(record.workspace_id),
        "email": record.email,
        "full_name": record.full_name,
        "role": record.role.value,
        "status": record.status,
        "invited_by_user_id": str(record.invited_by_user_id) if record.invited_by_user_id else None,
        "accepted_user_id": str(record.accepted_user_id) if record.accepted_user_id else None,
        "expires_at": record.expires_at.isoformat() if record.expires_at else None,
        "message": record.message,
        "metadata": record.metadata_payload,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
        "delivery_status": "manual_delivery_required",
    }
