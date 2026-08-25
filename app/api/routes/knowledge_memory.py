from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session

from app.db import get_session
from app.models import (
    KnowledgeCorpusStatus,
    KnowledgeDocumentEntry,
    KnowledgeDocumentGovernancePatchRequest,
    KnowledgeDocumentRecord,
    KnowledgeIngestionReport,
    KnowledgeManagedDocumentUpsertRequest,
    KnowledgeScope,
    KnowledgeSearchResponse,
    SessionRecord,
    UserRecord,
)
from app.services.auth_service import get_current_user
from app.services.knowledge_memory import KnowledgeMemoryService
from app.services.runtime_access_control import ensure_platform_admin
from app.services.workspace_access import WorkspaceAccessContext, get_current_workspace_context


router = APIRouter(prefix="/knowledge", tags=["knowledge"])


def _resolve_authorized_session_id(
    db: Session,
    *,
    workspace_context: WorkspaceAccessContext,
    session_id: UUID | None,
    required: bool = False,
) -> UUID | None:
    if session_id is None:
        if required:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="session_id es obligatorio para el scope session.",
            )
        return None

    session_record = db.get(SessionRecord, session_id)
    if session_record is None or session_record.workspace_id != workspace_context.workspace.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="La sesion solicitada no pertenece al workspace activo.",
        )
    return session_record.id


def _workspace_id_for_scope(scope: KnowledgeScope, workspace_context: WorkspaceAccessContext) -> UUID | None:
    if scope == KnowledgeScope.platform:
        return None
    return workspace_context.workspace.id


def _ensure_scope_admin(
    db: Session,
    *,
    scope: KnowledgeScope,
    current_user: UserRecord,
    workspace_context: WorkspaceAccessContext,
) -> None:
    del scope, workspace_context
    try:
        ensure_platform_admin(db, current_user)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc


def _load_authorized_document(
    db: Session,
    *,
    document_id: UUID,
    workspace_context: WorkspaceAccessContext,
) -> KnowledgeDocumentRecord:
    record = db.get(KnowledgeDocumentRecord, document_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No existe el documento solicitado.")
    if record.scope != KnowledgeScope.platform and record.workspace_id != workspace_context.workspace.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="El documento solicitado no pertenece al workspace activo.",
        )
    return record


@router.get("/docs/status", response_model=KnowledgeCorpusStatus)
def get_docs_corpus_status(
    scope: KnowledgeScope = Query(default=KnowledgeScope.platform),
    session_id: UUID | None = Query(default=None),
    ensure: bool = Query(default=True),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
    db: Session = Depends(get_session),
) -> KnowledgeCorpusStatus:
    if scope != KnowledgeScope.session and session_id is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="session_id solo puede enviarse cuando scope=session.",
        )
    resolved_session_id = _resolve_authorized_session_id(
        db,
        workspace_context=workspace_context,
        session_id=session_id,
        required=scope == KnowledgeScope.session,
    )
    _ensure_scope_admin(
        db,
        scope=scope,
        current_user=current_user,
        workspace_context=workspace_context,
    )
    return KnowledgeMemoryService().build_corpus_status(
        db,
        scope=scope,
        workspace_id=_workspace_id_for_scope(scope, workspace_context),
        session_id=resolved_session_id,
        ensure_ingested=ensure,
    )


@router.post("/docs/reingest", response_model=KnowledgeIngestionReport)
def reingest_docs_corpus(
    scope: KnowledgeScope = Query(default=KnowledgeScope.platform),
    session_id: UUID | None = Query(default=None),
    force: bool = Query(default=False),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
    db: Session = Depends(get_session),
) -> KnowledgeIngestionReport:
    if scope != KnowledgeScope.session and session_id is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="session_id solo puede enviarse cuando scope=session.",
        )
    resolved_session_id = _resolve_authorized_session_id(
        db,
        workspace_context=workspace_context,
        session_id=session_id,
        required=scope == KnowledgeScope.session,
    )
    _ensure_scope_admin(
        db,
        scope=scope,
        current_user=current_user,
        workspace_context=workspace_context,
    )
    return KnowledgeMemoryService().sync_docs_corpus(
        db,
        scope=scope,
        workspace_id=_workspace_id_for_scope(scope, workspace_context),
        session_id=resolved_session_id,
        force=force,
    )


@router.get("/docs/search", response_model=KnowledgeSearchResponse)
def search_docs_corpus(
    q: str = Query(min_length=2),
    limit: int = Query(default=10, ge=1, le=25),
    ensure: bool = Query(default=True),
    role: str | None = Query(default=None),
    session_id: UUID | None = Query(default=None),
    stage: str | None = Query(default=None),
    authority: list[str] | None = Query(default=None),
    cursor: str | None = Query(default=None),
    corpus_hash: str | None = Query(default=None),
    _: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
    db: Session = Depends(get_session),
) -> KnowledgeSearchResponse:
    resolved_session_id = _resolve_authorized_session_id(
        db,
        workspace_context=workspace_context,
        session_id=session_id,
        required=False,
    )
    service = KnowledgeMemoryService()
    if isinstance(role, str) and role.strip():
        return service.search_governed(
            db,
            query=q,
            role=role.strip(),
            workspace_id=workspace_context.workspace.id,
            session_id=resolved_session_id,
            stage=stage,
            authority_allowlist=authority or [],
            corpus_hash=corpus_hash,
            limit=limit,
            ensure_ingested=ensure,
            cursor=cursor,
        )
    return service.search(
        db,
        query=q,
        workspace_id=workspace_context.workspace.id,
        session_id=resolved_session_id,
        stage=stage,
        authority_allowlist=authority or [],
        limit=limit,
        ensure_ingested=ensure,
        cursor=cursor,
    )


@router.post("/docs/entries", response_model=KnowledgeDocumentEntry)
def upsert_managed_knowledge_document(
    payload: KnowledgeManagedDocumentUpsertRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> KnowledgeDocumentEntry:
    if payload.scope == KnowledgeScope.platform:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La ingesta gestionada via API solo soporta scope workspace o session.",
        )
    if payload.scope != KnowledgeScope.session and payload.session_id is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="session_id solo puede enviarse cuando scope=session.",
        )
    resolved_session_id = _resolve_authorized_session_id(
        db,
        workspace_context=workspace_context,
        session_id=payload.session_id,
        required=payload.scope == KnowledgeScope.session,
    )
    _ensure_scope_admin(
        db,
        scope=payload.scope,
        current_user=current_user,
        workspace_context=workspace_context,
    )
    normalized_payload = payload.model_copy(update={"session_id": resolved_session_id})
    try:
        return KnowledgeMemoryService().upsert_managed_document(
            db,
            payload=normalized_payload,
            workspace_id=workspace_context.workspace.id,
            actor_user_id=current_user.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.patch("/docs/entries/{document_id}", response_model=KnowledgeDocumentEntry)
def patch_knowledge_document_governance(
    document_id: UUID,
    payload: KnowledgeDocumentGovernancePatchRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> KnowledgeDocumentEntry:
    record = _load_authorized_document(
        db,
        document_id=document_id,
        workspace_context=workspace_context,
    )
    _ensure_scope_admin(
        db,
        scope=record.scope,
        current_user=current_user,
        workspace_context=workspace_context,
    )
    try:
        return KnowledgeMemoryService().update_document_governance(
            db,
            document_id=document_id,
            payload=payload,
            actor_user_id=current_user.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
