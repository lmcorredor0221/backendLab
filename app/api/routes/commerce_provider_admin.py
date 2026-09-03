from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session, select

from app.db import get_session
from app.models import (
    CommerceProviderConfigRecord,
    CommerceProviderCheckoutRecord,
    CommerceProviderCheckoutRecordResponse,
    CommerceProviderCredentialUpsertRequest,
    CommerceProviderDefinitionResponse,
    CommerceProviderProductMappingResponse,
    CommerceProviderProductMappingUpsertRequest,
    CommerceProviderReadinessResponse,
    CommerceProviderStatusResponse,
    CommerceProviderTestConnectionResponse,
    CommerceProviderWebhookEventRecord,
    CommerceProviderWebhookEventResponse,
    UserRecord,
    utc_now,
)
from app.services.auth_service import get_current_user
from app.services.commerce_provider_mappings import (
    list_commerce_provider_mappings,
    upsert_commerce_provider_mapping,
)
from app.services.commerce_provider_readiness import (
    build_commerce_provider_readiness,
    list_commerce_providers,
)
from app.services.commerce_provider_secrets import (
    build_commerce_provider_status,
    load_commerce_provider_secret,
    upsert_commerce_provider_credentials,
)
from app.services.commerce_provider_utils import normalize_commerce_provider_environment
from app.services.hotmart.secrets import test_hotmart_connection
from app.services.rebill.client import RebillClient, RebillClientConfig
from app.services.runtime_access_control import ensure_platform_admin
from app.services.workspace_access import WorkspaceAccessContext, get_current_workspace_context
from app.services.workspace_bootstrap import apply_workspace_bootstrap, resolve_platform_admin_template_workspace_id


router = APIRouter(prefix="/admin/commerce/providers", tags=["commerce-provider-admin"])


@router.get("", response_model=list[CommerceProviderDefinitionResponse])
def list_commerce_providers_route(
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> list[CommerceProviderDefinitionResponse]:
    _ensure_platform_admin_or_403(db, current_user)
    return list_commerce_providers()


@router.get("/{provider_key}/status", response_model=CommerceProviderStatusResponse)
def get_commerce_provider_status_route(
    provider_key: str,
    environment: str = Query(default="sandbox"),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> CommerceProviderStatusResponse:
    _ensure_platform_admin_or_403(db, current_user)
    workspace_id = _commerce_platform_workspace_id(db, current_user, workspace_context)
    return build_commerce_provider_status(
        db,
        workspace_id=workspace_id,
        provider_key=provider_key,
        environment=environment,
    )


@router.post("/{provider_key}/credentials", response_model=CommerceProviderStatusResponse)
def upsert_commerce_provider_credentials_route(
    provider_key: str,
    payload: CommerceProviderCredentialUpsertRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> CommerceProviderStatusResponse:
    _ensure_platform_admin_or_403(db, current_user)
    workspace_id = _commerce_platform_workspace_id(db, current_user, workspace_context)
    apply_workspace_bootstrap(db, workspace_id)
    try:
        response = upsert_commerce_provider_credentials(
            db,
            workspace_id=workspace_id,
            provider_key=provider_key,
            payload=payload,
            actor_user_id=current_user.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    db.commit()
    return response


@router.post("/{provider_key}/test-connection", response_model=CommerceProviderTestConnectionResponse)
def test_commerce_provider_connection_route(
    provider_key: str,
    environment: str = Query(default="sandbox"),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> CommerceProviderTestConnectionResponse:
    _ensure_platform_admin_or_403(db, current_user)
    workspace_id = _commerce_platform_workspace_id(db, current_user, workspace_context)
    apply_workspace_bootstrap(db, workspace_id)
    env = normalize_commerce_provider_environment(environment)
    if provider_key.strip().lower() == "hotmart":
        hotmart_response = test_hotmart_connection(db, workspace_id=workspace_id, environment=env)
        db.commit()
        return CommerceProviderTestConnectionResponse(
            workspace_id=workspace_id,
            provider_key="hotmart",
            environment=env,  # type: ignore[arg-type]
            reachable=hotmart_response.reachable,
            status=hotmart_response.status,
            message=hotmart_response.message,
            http_status=hotmart_response.http_status,
            checked_at=hotmart_response.checked_at,
        )
    if provider_key.strip().lower() == "sandbox":
        return CommerceProviderTestConnectionResponse(
            workspace_id=workspace_id,
            provider_key="sandbox",
            environment=env,  # type: ignore[arg-type]
            reachable=True,
            status="connected",
            message="Sandbox provider is always available locally.",
            checked_at=utc_now(),
        )
    status_response = build_commerce_provider_status(
        db,
        workspace_id=workspace_id,
        provider_key=provider_key,
        environment=env,
    )
    secret_key = load_commerce_provider_secret(
        db,
        workspace_id=workspace_id,
        provider_key=provider_key,
        environment=env,
        secret_kind="secret_key",
    )
    checked_at = utc_now()
    if not secret_key:
        return CommerceProviderTestConnectionResponse(
            workspace_id=workspace_id,
            provider_key=provider_key.strip().lower(),
            environment=env,  # type: ignore[arg-type]
            reachable=False,
            status="missing_credentials",
            message="Secret key is required to validate the provider connection.",
            checked_at=checked_at,
        )
    client = RebillClient(
        RebillClientConfig(
            api_base_url=status_response.api_base_url,
            timeout_seconds=30,
        )
    )
    reachable, message, http_status = client.test_connection(secret_key=secret_key)
    config = db.exec(
        select(CommerceProviderConfigRecord).where(
            CommerceProviderConfigRecord.workspace_id == workspace_id,
            CommerceProviderConfigRecord.provider_key == provider_key.strip().lower(),
            CommerceProviderConfigRecord.environment == env,
        )
    ).first()
    if config is not None:
        config.last_checked_at = checked_at
        config.last_health_status = "connected" if reachable else "connection_failed"
        config.last_health_message = message
        config.status = "connected" if reachable else "connection_failed"
        config.updated_at = checked_at
        db.add(config)
        db.commit()
    return CommerceProviderTestConnectionResponse(
        workspace_id=workspace_id,
        provider_key=provider_key.strip().lower(),
        environment=env,  # type: ignore[arg-type]
        reachable=reachable,
        status="connected" if reachable else "connection_failed",
        message=message,
        http_status=http_status,
        checked_at=checked_at,
    )


@router.get("/{provider_key}/mappings", response_model=list[CommerceProviderProductMappingResponse])
def list_commerce_provider_mappings_route(
    provider_key: str,
    environment: str = Query(default="sandbox"),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> list[CommerceProviderProductMappingResponse]:
    _ensure_platform_admin_or_403(db, current_user)
    workspace_id = _commerce_platform_workspace_id(db, current_user, workspace_context)
    return list_commerce_provider_mappings(
        db,
        workspace_id=workspace_id,
        provider_key=provider_key,
        environment=environment,
    )


@router.post("/{provider_key}/mappings", response_model=CommerceProviderProductMappingResponse)
def upsert_commerce_provider_mapping_route(
    provider_key: str,
    payload: CommerceProviderProductMappingUpsertRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> CommerceProviderProductMappingResponse:
    _ensure_platform_admin_or_403(db, current_user)
    workspace_id = _commerce_platform_workspace_id(db, current_user, workspace_context)
    apply_workspace_bootstrap(db, workspace_id)
    try:
        response = upsert_commerce_provider_mapping(
            db,
            workspace_id=workspace_id,
            provider_key=provider_key,
            payload=payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    db.commit()
    return response


@router.get("/{provider_key}/checkouts", response_model=list[CommerceProviderCheckoutRecordResponse])
def list_commerce_provider_checkouts_route(
    provider_key: str,
    environment: str = Query(default="sandbox"),
    limit: int = Query(default=100, ge=1, le=200),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> list[CommerceProviderCheckoutRecordResponse]:
    _ensure_platform_admin_or_403(db, current_user)
    workspace_id = _commerce_platform_workspace_id(db, current_user, workspace_context)
    env = normalize_commerce_provider_environment(environment)
    records = db.exec(
        select(CommerceProviderCheckoutRecord)
        .where(
            CommerceProviderCheckoutRecord.workspace_id == workspace_id,
            CommerceProviderCheckoutRecord.provider_key == provider_key.strip().lower(),
            CommerceProviderCheckoutRecord.environment == env,
        )
        .order_by(CommerceProviderCheckoutRecord.created_at.desc())
        .limit(limit)
    ).all()
    return [_serialize_checkout_record(record) for record in records]


@router.get("/{provider_key}/webhook-events", response_model=list[CommerceProviderWebhookEventResponse])
def list_commerce_provider_webhook_events_route(
    provider_key: str,
    environment: str = Query(default="sandbox"),
    limit: int = Query(default=100, ge=1, le=200),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> list[CommerceProviderWebhookEventResponse]:
    _ensure_platform_admin_or_403(db, current_user)
    workspace_id = _commerce_platform_workspace_id(db, current_user, workspace_context)
    env = normalize_commerce_provider_environment(environment)
    records = db.exec(
        select(CommerceProviderWebhookEventRecord)
        .where(
            CommerceProviderWebhookEventRecord.workspace_id == workspace_id,
            CommerceProviderWebhookEventRecord.provider_key == provider_key.strip().lower(),
            CommerceProviderWebhookEventRecord.environment == env,
        )
        .order_by(CommerceProviderWebhookEventRecord.created_at.desc())
        .limit(limit)
    ).all()
    return [_serialize_webhook_event(record) for record in records]


@router.get("/{provider_key}/readiness", response_model=CommerceProviderReadinessResponse)
def get_commerce_provider_readiness_route(
    provider_key: str,
    environment: str = Query(default="sandbox"),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> CommerceProviderReadinessResponse:
    _ensure_platform_admin_or_403(db, current_user)
    workspace_id = _commerce_platform_workspace_id(db, current_user, workspace_context)
    return build_commerce_provider_readiness(
        db,
        workspace_id=workspace_id,
        provider_key=provider_key,
        environment=environment,
    )


def _ensure_platform_admin_or_403(db: Session, current_user: UserRecord) -> None:
    try:
        ensure_platform_admin(db, current_user)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc


def _commerce_platform_workspace_id(
    db: Session,
    current_user: UserRecord,
    workspace_context: WorkspaceAccessContext,
) -> UUID:
    platform_workspace_id = resolve_platform_admin_template_workspace_id(db)
    if platform_workspace_id is not None:
        return platform_workspace_id
    if current_user.default_workspace_id is not None:
        return current_user.default_workspace_id
    return workspace_context.workspace.id


def _serialize_checkout_record(record: CommerceProviderCheckoutRecord) -> CommerceProviderCheckoutRecordResponse:
    return CommerceProviderCheckoutRecordResponse(
        id=record.id,
        workspace_id=record.workspace_id,
        provider_key=record.provider_key,
        environment=record.environment,  # type: ignore[arg-type]
        order_id=record.order_id,
        checkout_ref=record.checkout_ref,
        provider_checkout_id=record.provider_checkout_id,
        provider_payment_link_id=record.provider_payment_link_id,
        provider_customer_id=record.provider_customer_id,
        checkout_url=record.checkout_url,
        status=record.status,
        amount_cents=record.amount_cents,
        currency=record.currency,
        metadata=dict(record.metadata_payload or {}),
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _serialize_webhook_event(record: CommerceProviderWebhookEventRecord) -> CommerceProviderWebhookEventResponse:
    return CommerceProviderWebhookEventResponse(
        id=record.id,
        provider_key=record.provider_key,
        environment=record.environment,  # type: ignore[arg-type]
        event_id=record.event_id,
        event_type=record.event_type,
        provider_resource_id=record.provider_resource_id,
        workspace_id=record.workspace_id,
        order_id=record.order_id,
        payment_id=record.payment_id,
        signature_validated=record.signature_validated,
        processing_status=record.processing_status,
        retries=record.retries,
        error_code=record.error_code,
        error_message=record.error_message,
        processed_at=record.processed_at,
        created_at=record.created_at,
    )
