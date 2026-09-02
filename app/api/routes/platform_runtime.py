from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session

from app.db import get_session
from app.models import (
    LLMProviderKey,
    LLMRuntimeSettings,
    LLMRuntimeSettingsUpdateRequest,
    PlatformProviderSecretResponse,
    PlatformRuntimeProviderResponse,
    PlatformRuntimeProviderUpdateRequest,
    RuntimePropagationRequest,
    RuntimePropagationRunResponse,
    RuntimeSettingsAuditListResponse,
    UserRecord,
    WorkspaceProviderSecretUpsertRequest,
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
from app.services.llm_runtime.runtime_secrets_service import (
    build_platform_provider_secret_view,
    delete_platform_provider_secret,
    rotate_platform_provider_secret,
    upsert_platform_provider_secret,
)
from app.services.runtime_access_control import ensure_platform_admin
from app.services.runtime_propagation_service import propagate_platform_runtime_settings


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


@router.get("/secrets/{provider_key}", response_model=PlatformProviderSecretResponse)
def get_platform_provider_secret_route(
    provider_key: LLMProviderKey,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> PlatformProviderSecretResponse:
    try:
        ensure_platform_admin(db, current_user)
        return build_platform_provider_secret_view(db, provider_key)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc


@router.post("/secrets/{provider_key}", response_model=PlatformProviderSecretResponse)
def upsert_platform_provider_secret_route(
    provider_key: LLMProviderKey,
    payload: WorkspaceProviderSecretUpsertRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> PlatformProviderSecretResponse:
    try:
        ensure_platform_admin(db, current_user)
        return upsert_platform_provider_secret(
            db,
            provider_key,
            payload,
            actor_user_id=current_user.id,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/secrets/{provider_key}/rotate", response_model=PlatformProviderSecretResponse)
def rotate_platform_provider_secret_route(
    provider_key: LLMProviderKey,
    payload: WorkspaceProviderSecretUpsertRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> PlatformProviderSecretResponse:
    try:
        ensure_platform_admin(db, current_user)
        return rotate_platform_provider_secret(
            db,
            provider_key,
            payload,
            actor_user_id=current_user.id,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.delete("/secrets/{provider_key}", response_model=PlatformProviderSecretResponse)
def delete_platform_provider_secret_route(
    provider_key: LLMProviderKey,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> PlatformProviderSecretResponse:
    try:
        ensure_platform_admin(db, current_user)
        return delete_platform_provider_secret(
            db,
            provider_key,
            actor_user_id=current_user.id,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/defaults/propagate", response_model=RuntimePropagationRunResponse)
def propagate_platform_runtime_defaults_route(
    payload: RuntimePropagationRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> RuntimePropagationRunResponse:
    try:
        ensure_platform_admin(db, current_user)
        validate_runtime_update_request_against_platform_registry(db, payload.payload)
        return propagate_platform_runtime_settings(
            db,
            payload=payload,
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
