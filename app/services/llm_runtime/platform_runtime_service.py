from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlmodel import Session, select

from app.models import (
    LLMProviderKey,
    LLMRuntimeSettingsUpdateRequest,
    PlatformRuntimeProviderRecord,
    PlatformRuntimeProviderResponse,
    PlatformRuntimeProviderUpdateRequest,
    RuntimeGovernanceScopeType,
    RuntimeSettingsAuditEntry,
    RuntimeSettingsAuditListResponse,
    RuntimeSettingsAuditRecord,
    UserRecord,
    utc_now,
)
from app.services.llm_runtime.runtime_settings_service import (
    load_platform_runtime_defaults,
    persist_platform_runtime_defaults,
)
from app.services.runtime_governance_bootstrap import backfill_platform_runtime_governance


def _normalize_string_list(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        token = item.strip()
        if not token:
            continue
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(token)
    return normalized


def _provider_response(record: PlatformRuntimeProviderRecord) -> PlatformRuntimeProviderResponse:
    return PlatformRuntimeProviderResponse(
        provider_key=record.provider_key,
        label=record.label,
        is_enabled=record.is_enabled,
        allowed_models=list(record.allowed_models),
        default_models=dict(record.default_models),
        allowed_auth_modes=list(record.allowed_auth_modes),
        supports_workspace_secrets=record.supports_workspace_secrets,
        supports_platform_managed_credentials=record.supports_platform_managed_credentials,
        release_stage=record.release_stage,
        health_policy=dict(record.health_policy),
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _resolve_actor_email(session: Session, actor_user_id: UUID | None) -> str:
    if actor_user_id is None:
        return ""
    actor = session.get(UserRecord, actor_user_id)
    if actor is None:
        return ""
    return actor.email


def _record_platform_audit(
    session: Session,
    *,
    scope_id: str,
    change_type: str,
    before_payload: dict[str, Any],
    after_payload: dict[str, Any],
    actor_user_id: UUID | None,
) -> None:
    session.add(
        RuntimeSettingsAuditRecord(
            scope_type=RuntimeGovernanceScopeType.platform,
            scope_id=scope_id,
            change_type=change_type,
            before_payload_redacted=before_payload,
            after_payload_redacted=after_payload,
            actor_user_id=actor_user_id,
            actor_email=_resolve_actor_email(session, actor_user_id),
        )
    )


def list_platform_runtime_providers(session: Session) -> list[PlatformRuntimeProviderResponse]:
    backfill_platform_runtime_governance(session)
    rows = session.exec(
        select(PlatformRuntimeProviderRecord).order_by(PlatformRuntimeProviderRecord.provider_key.asc())
    ).all()
    return [_provider_response(item) for item in rows]


def update_platform_runtime_provider(
    session: Session,
    provider_key: LLMProviderKey,
    payload: PlatformRuntimeProviderUpdateRequest,
    *,
    actor_user_id: UUID | None = None,
) -> PlatformRuntimeProviderResponse:
    backfill_platform_runtime_governance(session)
    record = session.exec(
        select(PlatformRuntimeProviderRecord).where(PlatformRuntimeProviderRecord.provider_key == provider_key)
    ).first()
    if record is None:
        raise LookupError(f"No existe configuracion de plataforma para {provider_key.value}.")

    before_payload = _provider_response(record).model_dump(mode="json")

    if payload.label is not None:
        record.label = payload.label.strip() or record.label
    if payload.is_enabled is not None:
        record.is_enabled = payload.is_enabled
    normalized_models = _normalize_string_list(payload.allowed_models)
    if normalized_models is not None:
        record.allowed_models = normalized_models
    if payload.default_models is not None:
        record.default_models = dict(payload.default_models)
    normalized_auth_modes = _normalize_string_list(payload.allowed_auth_modes)
    if normalized_auth_modes is not None:
        record.allowed_auth_modes = normalized_auth_modes
    if payload.supports_workspace_secrets is not None:
        record.supports_workspace_secrets = payload.supports_workspace_secrets
    if payload.supports_platform_managed_credentials is not None:
        record.supports_platform_managed_credentials = payload.supports_platform_managed_credentials
    if payload.release_stage is not None:
        record.release_stage = payload.release_stage
    if payload.health_policy is not None:
        record.health_policy = dict(payload.health_policy)
    record.updated_at = utc_now()
    session.add(record)
    session.flush()

    after_payload = _provider_response(record).model_dump(mode="json")
    _record_platform_audit(
        session,
        scope_id=f"provider:{provider_key.value}",
        change_type="platform_runtime_provider_updated",
        before_payload=before_payload,
        after_payload=after_payload,
        actor_user_id=actor_user_id,
    )
    session.commit()
    session.refresh(record)
    return _provider_response(record)


def load_platform_runtime_defaults_for_admin(session: Session):
    return load_platform_runtime_defaults(session)


def update_platform_runtime_defaults(
    session: Session,
    payload: LLMRuntimeSettingsUpdateRequest,
    *,
    actor_user_id: UUID | None = None,
):
    return persist_platform_runtime_defaults(
        session,
        payload,
        actor_user_id=actor_user_id,
        mirror_legacy_runtime=False,
    )


def list_platform_runtime_audit(session: Session, *, limit: int = 50) -> RuntimeSettingsAuditListResponse:
    backfill_platform_runtime_governance(session)
    rows = session.exec(
        select(RuntimeSettingsAuditRecord)
        .where(RuntimeSettingsAuditRecord.scope_type == RuntimeGovernanceScopeType.platform)
        .order_by(RuntimeSettingsAuditRecord.created_at.desc())
        .limit(max(1, min(limit, 200)))
    ).all()
    return RuntimeSettingsAuditListResponse(
        items=[
            RuntimeSettingsAuditEntry(
                id=item.id,
                scope_type=item.scope_type,
                scope_id=item.scope_id,
                change_type=item.change_type,
                before_payload_redacted=dict(item.before_payload_redacted),
                after_payload_redacted=dict(item.after_payload_redacted),
                actor_user_id=item.actor_user_id,
                actor_email=item.actor_email,
                created_at=item.created_at,
            )
            for item in rows
        ]
    )


def validate_runtime_update_request_against_platform_registry(
    session: Session,
    payload: LLMRuntimeSettingsUpdateRequest,
) -> None:
    backfill_platform_runtime_governance(session)
    provider = session.exec(
        select(PlatformRuntimeProviderRecord).where(PlatformRuntimeProviderRecord.provider_key == payload.active_provider)
    ).first()
    if provider is None:
        raise ValueError(f"El provider {payload.active_provider.value} no existe en el catalogo de plataforma.")
    if not provider.is_enabled:
        raise ValueError(f"El provider {payload.active_provider.value} esta deshabilitado por plataforma.")
    if not provider.supports_workspace_secrets and not provider.supports_platform_managed_credentials:
        return
    if payload.uses_platform_credentials is False and not provider.supports_workspace_secrets:
        raise ValueError(f"El provider {payload.active_provider.value} no soporta credenciales por workspace.")
    if payload.uses_platform_credentials is True and not provider.supports_platform_managed_credentials:
        raise ValueError(f"El provider {payload.active_provider.value} no soporta credenciales gestionadas por plataforma.")
