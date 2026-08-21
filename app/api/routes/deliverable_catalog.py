from __future__ import annotations

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session, select

from app.db import get_session
from app.models import CommercialTier, SessionRecord, UserRecord
from app.services.auth_service import get_current_user
from app.services.deliverable_catalog import (
    DeliverableCatalogResponse,
    DeliverableDetailResponse,
    DeliverableGovernanceEntry,
    DeliverableGovernanceOverview,
    DeliverableGovernanceResponse,
    DeliverableGovernanceUpdate,
    DeliverableType,
    DeliverablePromptResponse,
    DeliverablePromptUpdate,
    DeliverablePromptValidationRequest,
    DeliverablePromptValidationResponse,
    build_deliverable_catalog_response,
    build_deliverable_detail_response,
    build_deliverable_governance_overview,
    deliverable_governance_entry,
    get_deliverable_prompt,
    get_registry_entry,
    list_registry_entries,
    seed_deliverable_governance_defaults,
    update_deliverable_prompt,
    upsert_deliverable_governance,
    validate_deliverable_prompt,
)
from app.services.runtime_access_control import ensure_platform_admin
from app.services.stage5_service import (
    FEATURE_FLAG_DELIVERABLE_CATALOG,
    FEATURE_FLAG_DELIVERABLE_GOVERNANCE_ADMIN,
    is_feature_flag_enabled,
)
from app.services.workspace_access import WorkspaceAccessContext, get_current_workspace_context


router = APIRouter(prefix="/v3", tags=["deliverable-catalog-v3"])


def _admin_scope_workspace_id(scope: Literal["platform", "workspace"], workspace_context: WorkspaceAccessContext):
    return workspace_context.workspace.id if scope == "workspace" else None


def _ensure_deliverable_catalog_enabled(db: Session, workspace_context: WorkspaceAccessContext) -> None:
    seed_deliverable_governance_defaults(db)
    if not is_feature_flag_enabled(
        db,
        FEATURE_FLAG_DELIVERABLE_CATALOG,
        workspace_id=workspace_context.workspace.id,
        default_if_missing=True,
    ):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Deliverable catalog feature flag is disabled")


def _ensure_deliverable_governance_admin_enabled(db: Session, workspace_context: WorkspaceAccessContext) -> None:
    _ensure_deliverable_catalog_enabled(db, workspace_context)
    if not is_feature_flag_enabled(
        db,
        FEATURE_FLAG_DELIVERABLE_GOVERNANCE_ADMIN,
        workspace_id=workspace_context.workspace.id,
        default_if_missing=True,
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Deliverable governance admin feature flag is disabled",
        )


def _ensure_session_belongs_to_workspace(
    db: Session,
    *,
    session_id: UUID | None,
    workspace_context: WorkspaceAccessContext,
) -> UUID | None:
    if session_id is None:
        return None
    record = db.exec(
        select(SessionRecord).where(
            SessionRecord.id == session_id,
            SessionRecord.workspace_id == workspace_context.workspace.id,
            SessionRecord.deleted_at.is_(None),
        )
    ).first()
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found in current workspace")
    return record.id


@router.get("/deliverables/catalog", response_model=DeliverableCatalogResponse)
def get_deliverable_catalog_v3(
    tier: CommercialTier = Query(default=CommercialTier.blueprint),
    current_stage: str = Query(default="discover"),
    session_id: UUID | None = Query(default=None),
    include_inactive: bool = Query(default=False),
    db: Session = Depends(get_session),
    _: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> DeliverableCatalogResponse:
    _ensure_deliverable_catalog_enabled(db, workspace_context)
    resolved_session_id = _ensure_session_belongs_to_workspace(
        db,
        session_id=session_id,
        workspace_context=workspace_context,
    )
    return build_deliverable_catalog_response(
        db,
        workspace_id=workspace_context.workspace.id,
        session_id=resolved_session_id,
        role=workspace_context.membership.role,
        tier=tier,
        current_stage=current_stage,
        include_inactive=include_inactive,
    )


@router.get("/deliverables/stage/{stage_key}", response_model=DeliverableCatalogResponse)
def get_stage_deliverables_v3(
    stage_key: str,
    tier: CommercialTier = Query(default=CommercialTier.blueprint),
    current_stage: str | None = Query(default=None),
    session_id: UUID | None = Query(default=None),
    include_inactive: bool = Query(default=False),
    db: Session = Depends(get_session),
    _: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> DeliverableCatalogResponse:
    _ensure_deliverable_catalog_enabled(db, workspace_context)
    resolved_session_id = _ensure_session_belongs_to_workspace(
        db,
        session_id=session_id,
        workspace_context=workspace_context,
    )
    return build_deliverable_catalog_response(
        db,
        workspace_id=workspace_context.workspace.id,
        session_id=resolved_session_id,
        role=workspace_context.membership.role,
        tier=tier,
        current_stage=current_stage or stage_key,
        stage_filter=stage_key,
        include_inactive=include_inactive,
    )


@router.get("/deliverables/{deliverable_key}", response_model=DeliverableDetailResponse)
def get_deliverable_detail_v3(
    deliverable_key: str,
    tier: CommercialTier = Query(default=CommercialTier.blueprint),
    current_stage: str = Query(default="discover"),
    session_id: UUID | None = Query(default=None),
    db: Session = Depends(get_session),
    _: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> DeliverableDetailResponse:
    _ensure_deliverable_catalog_enabled(db, workspace_context)
    resolved_session_id = _ensure_session_belongs_to_workspace(
        db,
        session_id=session_id,
        workspace_context=workspace_context,
    )
    detail = build_deliverable_detail_response(
        db,
        deliverable_key=deliverable_key,
        workspace_id=workspace_context.workspace.id,
        session_id=resolved_session_id,
        role=workspace_context.membership.role,
        tier=tier,
        current_stage=current_stage,
    )
    if detail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Deliverable not found")
    return detail


@router.get("/admin/deliverable-governance", response_model=DeliverableGovernanceResponse)
def list_deliverable_governance_v3(
    scope: Literal["platform", "workspace"] = Query(default="platform"),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> DeliverableGovernanceResponse:
    try:
        ensure_platform_admin(db, current_user)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    _ensure_deliverable_governance_admin_enabled(db, workspace_context)
    workspace_id = _admin_scope_workspace_id(scope, workspace_context)
    return DeliverableGovernanceResponse(
        entries=[
            deliverable_governance_entry(db, entry, workspace_id=workspace_id)
            for entry in list_registry_entries(include_inactive=True)
        ]
    )


@router.get("/admin/deliverable-governance/overview", response_model=DeliverableGovernanceOverview)
def get_deliverable_governance_overview_v3(
    scope: Literal["platform", "workspace"] = Query(default="platform"),
    tier: CommercialTier = Query(default=CommercialTier.blueprint),
    current_stage: str = Query(default="discover"),
    product: Literal["blueprint", "blueprint_pro", "acp"] | None = Query(default=None),
    stage: str | None = Query(default=None),
    deliverable_type: DeliverableType | None = Query(default=None, alias="type"),
    quality_state: Literal["unknown", "passed", "warning", "failed", "stale"] | None = Query(default=None),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> DeliverableGovernanceOverview:
    try:
        ensure_platform_admin(db, current_user)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    _ensure_deliverable_governance_admin_enabled(db, workspace_context)
    return build_deliverable_governance_overview(
        db,
        workspace_id=_admin_scope_workspace_id(scope, workspace_context),
        tier=tier,
        current_stage=current_stage,
        role=workspace_context.membership.role,
        product=product,
        stage=stage,
        deliverable_type=deliverable_type,
        quality_state=quality_state,
    )


@router.patch("/admin/deliverable-governance/{deliverable_key}", response_model=DeliverableGovernanceEntry)
def update_deliverable_governance_v3(
    deliverable_key: str,
    payload: DeliverableGovernanceUpdate,
    scope: Literal["platform", "workspace"] = Query(default="platform"),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> DeliverableGovernanceEntry:
    try:
        ensure_platform_admin(db, current_user)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    _ensure_deliverable_governance_admin_enabled(db, workspace_context)
    entry = get_registry_entry(deliverable_key)
    if entry is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Deliverable not found")
    result = upsert_deliverable_governance(
        db,
        entry,
        payload,
        workspace_id=_admin_scope_workspace_id(scope, workspace_context),
        actor_user_id=current_user.id,
    )
    db.commit()
    return result


@router.get("/admin/deliverable-governance/{deliverable_key}/prompt", response_model=DeliverablePromptResponse)
def get_deliverable_prompt_v3(
    deliverable_key: str,
    scope: Literal["platform", "workspace"] = Query(default="platform"),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> DeliverablePromptResponse:
    try:
        ensure_platform_admin(db, current_user)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    _ensure_deliverable_governance_admin_enabled(db, workspace_context)
    entry = get_registry_entry(deliverable_key)
    if entry is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Deliverable not found")
    return get_deliverable_prompt(
        db,
        entry,
        workspace_id=_admin_scope_workspace_id(scope, workspace_context),
    )


@router.patch("/admin/deliverable-governance/{deliverable_key}/prompt", response_model=DeliverablePromptResponse)
def update_deliverable_prompt_v3(
    deliverable_key: str,
    payload: DeliverablePromptUpdate,
    scope: Literal["platform", "workspace"] = Query(default="platform"),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> DeliverablePromptResponse:
    try:
        ensure_platform_admin(db, current_user)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    _ensure_deliverable_governance_admin_enabled(db, workspace_context)
    entry = get_registry_entry(deliverable_key)
    if entry is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Deliverable not found")
    try:
        result = update_deliverable_prompt(
            db,
            entry,
            payload,
            workspace_id=_admin_scope_workspace_id(scope, workspace_context),
            actor_user_id=current_user.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    db.commit()
    return result


@router.post(
    "/admin/deliverable-governance/{deliverable_key}/prompt/validate",
    response_model=DeliverablePromptValidationResponse,
)
def validate_deliverable_prompt_v3(
    deliverable_key: str,
    payload: DeliverablePromptValidationRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> DeliverablePromptValidationResponse:
    try:
        ensure_platform_admin(db, current_user)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    _ensure_deliverable_governance_admin_enabled(db, workspace_context)
    entry = get_registry_entry(deliverable_key)
    if entry is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Deliverable not found")
    return validate_deliverable_prompt(entry, payload)
