from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from app.db import get_session
from app.models import UserRecord
from app.services.auth_service import get_current_user
from app.services.operations_service import get_codex_runtime_status
from app.services.runtime_access_control import ensure_platform_admin
from app.services.workspace_access import WorkspaceAccessContext, get_current_workspace_context


router = APIRouter(prefix="/runtime", tags=["runtime-status"])


@router.get("/status")
def get_runtime_status_route(
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> dict[str, object]:
    try:
        ensure_platform_admin(db, current_user)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    return get_codex_runtime_status(db, workspace_id=workspace_context.workspace.id)
