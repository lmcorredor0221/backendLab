from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from app.db import get_session
from app.models import (
    LLMProviderKey,
    LLMRuntimeSettings,
    LLMRuntimeSettingsUpdateRequest,
    UserRecord,
    WorkspaceProviderSecretResponse,
    WorkspaceRuntimeHealthResponse,
    WorkspaceProviderSecretUpsertRequest,
)
from app.services.auth_service import get_current_user
from app.services.llm_runtime.platform_runtime_service import validate_runtime_update_request_against_platform_registry
from app.services.llm_runtime.runtime_health_service import build_workspace_runtime_health
from app.services.llm_runtime.runtime_secrets_service import (
    annotate_runtime_settings_with_workspace_secrets,
    delete_workspace_provider_secret,
    rotate_workspace_provider_secret,
    upsert_workspace_provider_secret,
)
from app.services.llm_runtime.runtime_settings_service import (
    load_effective_runtime_settings,
    persist_workspace_runtime_settings,
    reset_workspace_runtime_settings,
)
from app.services.memory_rollout import build_memory_rollout_summary
from app.services.runtime_access_control import ensure_platform_admin
from app.services.workspace_access import WorkspaceAccessContext, get_current_workspace_context
from app.services.workspace_bootstrap import apply_workspace_bootstrap


def require_platform_runtime_admin(
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> None:
    try:
        ensure_platform_admin(db, current_user)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc


router = APIRouter(
    prefix="/runtime",
    tags=["runtime-settings"],
    dependencies=[Depends(require_platform_runtime_admin)],
)


def _runtime_settings_response(
    *,
    db: Session,
    workspace_id,
    runtime_settings: LLMRuntimeSettings,
) -> LLMRuntimeSettings:
    runtime_with_secrets = annotate_runtime_settings_with_workspace_secrets(
        db,
        workspace_id,
        runtime_settings,
    )
    return runtime_with_secrets.model_copy(
        update={
            "memory_rollout": build_memory_rollout_summary(
                runtime_with_secrets,
                session=db,
                workspace_id=workspace_id,
            )
        }
    )


@router.get("/llm", response_model=LLMRuntimeSettings)
def get_llm_runtime_settings_route(
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> LLMRuntimeSettings:
    _ = current_user
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    runtime_settings = load_effective_runtime_settings(db, workspace_context.workspace.id)
    return _runtime_settings_response(
        db=db,
        workspace_id=workspace_context.workspace.id,
        runtime_settings=runtime_settings,
    )


@router.patch("/llm", response_model=LLMRuntimeSettings)
def update_llm_runtime_settings_route(
    payload: LLMRuntimeSettingsUpdateRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> LLMRuntimeSettings:
    try:
        validate_runtime_update_request_against_platform_registry(db, payload)
    except (PermissionError, ValueError) as exc:
        status_code = status.HTTP_403_FORBIDDEN if isinstance(exc, PermissionError) else status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    runtime_settings = persist_workspace_runtime_settings(
        db,
        workspace_context.workspace.id,
        payload,
        actor_user_id=current_user.id,
    )
    return _runtime_settings_response(
        db=db,
        workspace_id=workspace_context.workspace.id,
        runtime_settings=runtime_settings,
    )


@router.delete("/llm", response_model=LLMRuntimeSettings)
def reset_llm_runtime_settings_route(
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> LLMRuntimeSettings:
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    runtime_settings = reset_workspace_runtime_settings(
        db,
        workspace_context.workspace.id,
        actor_user_id=current_user.id,
        mirror_legacy_runtime=False,
    )
    return _runtime_settings_response(
        db=db,
        workspace_id=workspace_context.workspace.id,
        runtime_settings=runtime_settings,
    )


@router.post("/llm/secrets/{provider_key}", response_model=WorkspaceProviderSecretResponse)
def upsert_workspace_provider_secret_route(
    provider_key: LLMProviderKey,
    payload: WorkspaceProviderSecretUpsertRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> WorkspaceProviderSecretResponse:
    try:
        apply_workspace_bootstrap(db, workspace_context.workspace.id)
        return upsert_workspace_provider_secret(
            db,
            workspace_context.workspace.id,
            provider_key,
            payload,
            actor_user_id=current_user.id,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/llm/secrets/{provider_key}/rotate", response_model=WorkspaceProviderSecretResponse)
def rotate_workspace_provider_secret_route(
    provider_key: LLMProviderKey,
    payload: WorkspaceProviderSecretUpsertRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> WorkspaceProviderSecretResponse:
    try:
        apply_workspace_bootstrap(db, workspace_context.workspace.id)
        return rotate_workspace_provider_secret(
            db,
            workspace_context.workspace.id,
            provider_key,
            payload,
            actor_user_id=current_user.id,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.delete("/llm/secrets/{provider_key}", response_model=WorkspaceProviderSecretResponse)
def delete_workspace_provider_secret_route(
    provider_key: LLMProviderKey,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> WorkspaceProviderSecretResponse:
    try:
        apply_workspace_bootstrap(db, workspace_context.workspace.id)
        return delete_workspace_provider_secret(
            db,
            workspace_context.workspace.id,
            provider_key,
            actor_user_id=current_user.id,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/llm/health", response_model=WorkspaceRuntimeHealthResponse)
def get_workspace_runtime_health_route(
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
):
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    return build_workspace_runtime_health(
        db,
        workspace_context.workspace.id,
        mode="health",
    )


@router.post("/llm/test", response_model=WorkspaceRuntimeHealthResponse)
def test_workspace_runtime_route(
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
):
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    return build_workspace_runtime_health(
        db,
        workspace_context.workspace.id,
        mode="test",
    )
