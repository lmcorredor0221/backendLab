from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session

from app.db import get_session
from app.models import (
    LLMProviderKey,
    LLMRuntimeSettings,
    LLMRuntimeSettingsUpdateRequest,
    PlatformRuntimeProviderResponse,
    PlatformRuntimeProviderUpdateRequest,
    RuntimeSettingsAuditListResponse,
    UserRecord,
)
from app.services.auth_service import get_current_user
from app.services.llm_runtime.platform_runtime_service import (
    list_platform_runtime_audit,
    list_platform_runtime_providers,
    load_platform_runtime_defaults_for_admin,
    update_platform_runtime_defaults,
    update_platform_runtime_provider,
    validate_runtime_update_request_against_platform_registry,
)
from app.services.runtime_access_control import ensure_platform_admin


router = APIRouter(prefix="/platform/runtime", tags=["platform-runtime"])


@router.get("/providers", response_model=list[PlatformRuntimeProviderResponse])
def list_platform_runtime_providers_route(
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> list[PlatformRuntimeProviderResponse]:
    try:
        ensure_platform_admin(db, current_user)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    return list_platform_runtime_providers(db)


@router.patch("/providers/{provider_key}", response_model=PlatformRuntimeProviderResponse)
def update_platform_runtime_provider_route(
    provider_key: LLMProviderKey,
    payload: PlatformRuntimeProviderUpdateRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> PlatformRuntimeProviderResponse:
    try:
        ensure_platform_admin(db, current_user)
        return update_platform_runtime_provider(
            db,
            provider_key,
            payload,
            actor_user_id=current_user.id,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/defaults", response_model=LLMRuntimeSettings)
def get_platform_runtime_defaults_route(
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> LLMRuntimeSettings:
    try:
        ensure_platform_admin(db, current_user)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    return load_platform_runtime_defaults_for_admin(db)


@router.patch("/defaults", response_model=LLMRuntimeSettings)
def update_platform_runtime_defaults_route(
    payload: LLMRuntimeSettingsUpdateRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> LLMRuntimeSettings:
    try:
        ensure_platform_admin(db, current_user)
        validate_runtime_update_request_against_platform_registry(db, payload)
        return update_platform_runtime_defaults(
            db,
            payload,
            actor_user_id=current_user.id,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/audit", response_model=RuntimeSettingsAuditListResponse)
def list_platform_runtime_audit_route(
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> RuntimeSettingsAuditListResponse:
    try:
        ensure_platform_admin(db, current_user)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    return list_platform_runtime_audit(db, limit=limit)
