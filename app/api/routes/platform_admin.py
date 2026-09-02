from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session

from app.db import get_session
from app.models import (
    PlatformAdminProjectListResponse,
    PlatformAdminUserListResponse,
    PlatformAdminWorkspaceListResponse,
    UserRecord,
)
from app.services.auth_service import get_current_user
from app.services.platform_admin_analytics_service import (
    list_platform_admin_projects,
    list_platform_admin_users,
    list_platform_admin_workspaces,
)
from app.services.runtime_access_control import ensure_platform_admin


router = APIRouter(prefix="/platform/admin", tags=["platform-admin"])


def _ensure_platform_admin_or_403(db: Session, current_user: UserRecord) -> None:
    try:
        ensure_platform_admin(db, current_user)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc


@router.get("/users", response_model=PlatformAdminUserListResponse)
def list_platform_users_route(
    email: str = Query(default=""),
    workspace_id: UUID | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> PlatformAdminUserListResponse:
    _ensure_platform_admin_or_403(db, current_user)
    return list_platform_admin_users(db, email=email, workspace_id=workspace_id, limit=limit)


@router.get("/workspaces", response_model=PlatformAdminWorkspaceListResponse)
def list_platform_workspaces_route(
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> PlatformAdminWorkspaceListResponse:
    _ensure_platform_admin_or_403(db, current_user)
    return list_platform_admin_workspaces(db, limit=limit)


@router.get("/projects", response_model=PlatformAdminProjectListResponse)
def list_platform_projects_route(
    workspace_id: UUID | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> PlatformAdminProjectListResponse:
    _ensure_platform_admin_or_403(db, current_user)
    return list_platform_admin_projects(db, workspace_id=workspace_id, limit=limit)
