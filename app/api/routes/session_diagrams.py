from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from app.api.routes.sessions import build_snapshot, get_or_404, resolve_acp_preview
from app.db import get_session
from app.models import DiagramCatalogResponse, DiagramContentResponse, UserRecord
from app.services.auth_service import get_current_user
from app.services.commercial_access import resolve_session_entitlement_context
from app.services.diagram_catalog_service import build_diagram_catalog, build_diagram_content


router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("/{session_id}/diagrams/catalog", response_model=DiagramCatalogResponse, deprecated=True)
def get_session_diagram_catalog_route(
    session_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> DiagramCatalogResponse:
    record = get_or_404(db, session_id, current_user.id)
    snapshot = build_snapshot(db, record)
    preview = resolve_acp_preview(db, record)
    context = resolve_session_entitlement_context(db, record, current_user)
    return build_diagram_catalog(
        snapshot=snapshot,
        preview=preview,
        context=context,
        workspace_id=record.workspace_id,
    )


@router.get("/{session_id}/diagrams/{diagram_key}", response_model=DiagramContentResponse, deprecated=True)
def get_session_diagram_content_route(
    session_id: UUID,
    diagram_key: str,
    format: str | None = None,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> DiagramContentResponse:
    record = get_or_404(db, session_id, current_user.id)
    snapshot = build_snapshot(db, record)
    preview = resolve_acp_preview(db, record)
    context = resolve_session_entitlement_context(db, record, current_user)
    response = build_diagram_content(
        diagram_key=diagram_key,
        snapshot=snapshot,
        preview=preview,
        context=context,
        workspace_id=record.workspace_id,
        requested_format=format,
    )
    if response is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Diagram not found")
    return response
