from __future__ import annotations

import base64
import hashlib
from uuid import UUID

from cryptography.fernet import Fernet, InvalidToken
from sqlmodel import Session, select

try:
    from openai import OpenAI as OpenAISDK
except ImportError:  # pragma: no cover - optional dependency in some test environments
    OpenAISDK = None  # type: ignore[assignment]

from app.core.config import get_settings
from app.models import (
    LLMProviderKey,
    LLMRuntimeSettings,
    PlatformRuntimeProviderRecord,
    RuntimeGovernanceScopeType,
    RuntimeSecretStatus,
    RuntimeSettingsAuditRecord,
    UserRecord,
    WorkspaceProviderSecretRecord,
    WorkspaceProviderSecretResponse,
    WorkspaceProviderSecretUpsertRequest,
    WorkspaceRuntimeSettingsRecord,
    utc_now,
)
from app.services.llm_runtime.runtime_settings_service import (
    load_effective_runtime_settings,
    load_workspace_runtime_settings,
    persist_workspace_runtime_credential_mode,
)
from app.services.openai_builder import build_provider_options
from app.services.runtime_governance_bootstrap import backfill_platform_runtime_governance


def _resolve_actor_email(session: Session, actor_user_id: UUID | None) -> str:
    if actor_user_id is None:
        return ""
    actor = session.get(UserRecord, actor_user_id)
    if actor is None:
        return ""
    return actor.email


def _fernet() -> Fernet:
    settings = get_settings()
    if settings.runtime_secrets_master_key.strip():
        return Fernet(settings.runtime_secrets_master_key.strip().encode("utf-8"))
    fallback_material = (
        f"{settings.database_url}|{settings.local_admin_email}|{settings.local_admin_password}".encode("utf-8")
    )
    derived_key = base64.urlsafe_b64encode(hashlib.sha256(fallback_material).digest())
    return Fernet(derived_key)


def _encrypt_secret_value(secret_value: str) -> str:
    return _fernet().encrypt(secret_value.encode("utf-8")).decode("utf-8")


def _decrypt_secret_value(secret_ciphertext: str) -> str:
    try:
        return _fernet().decrypt(secret_ciphertext.encode("utf-8")).decode("utf-8")
    except InvalidToken as exc:
        raise ValueError("Workspace provider secret could not be decrypted with the configured master key.") from exc


def _provider_record(session: Session, provider_key: LLMProviderKey) -> PlatformRuntimeProviderRecord | None:
    return session.exec(
        select(PlatformRuntimeProviderRecord).where(PlatformRuntimeProviderRecord.provider_key == provider_key)
    ).first()


def _supports_workspace_secrets(session: Session, provider_key: LLMProviderKey) -> bool:
    provider_record = _provider_record(session, provider_key)
    if provider_record is None:
        return provider_key in {LLMProviderKey.openai, LLMProviderKey.deepseek}
    return provider_record.supports_workspace_secrets


def _workspace_secret_record(
    session: Session,
    workspace_id: UUID,
    provider_key: LLMProviderKey,
    *,
    secret_kind: str = "api_key",
) -> WorkspaceProviderSecretRecord | None:
    return session.exec(
        select(WorkspaceProviderSecretRecord).where(
            WorkspaceProviderSecretRecord.workspace_id == workspace_id,
            WorkspaceProviderSecretRecord.provider_key == provider_key,
            WorkspaceProviderSecretRecord.secret_kind == secret_kind,
        )
    ).first()


def _workspace_secret_records(session: Session, workspace_id: UUID) -> dict[LLMProviderKey, WorkspaceProviderSecretRecord]:
    rows = session.exec(
        select(WorkspaceProviderSecretRecord).where(WorkspaceProviderSecretRecord.workspace_id == workspace_id)
    ).all()
    return {item.provider_key: item for item in rows}


def resolve_workspace_provider_secret_value(
    session: Session,
    workspace_id: UUID,
    provider_key: LLMProviderKey,
    *,
    secret_kind: str = "api_key",
) -> str | None:
    record = _workspace_secret_record(
        session,
        workspace_id,
        provider_key,
        secret_kind=secret_kind,
    )
    if not _record_is_configured(record) or record is None:
        return None
    if record.secret_ciphertext:
        return _decrypt_secret_value(record.secret_ciphertext)
    if record.secret_ref:
        return None
    return None


def _record_is_configured(record: WorkspaceProviderSecretRecord | None) -> bool:
    if record is None:
        return False
    if record.status != RuntimeSecretStatus.configured:
        return False
    return bool(record.secret_ciphertext or record.secret_ref)


def _storage_mode(record: WorkspaceProviderSecretRecord | None) -> str:
    if record is None:
        return "none"
    if record.secret_ciphertext and record.secret_ref:
        return "ciphertext+reference"
    if record.secret_ref:
        return "reference"
    if record.secret_ciphertext:
        return "ciphertext"
    return "none"


def _workspace_secret_runtime_state(
    record: WorkspaceProviderSecretRecord | None,
) -> tuple[bool, str, str]:
    if not _record_is_configured(record) or record is None:
        return False, "workspace_missing", "No existe un secreto configurado para este workspace."
    if record.secret_ciphertext:
        if OpenAISDK is None:
            return False, "sdk_missing", "El SDK OpenAI-compatible no esta instalado en el backend."
        try:
            secret_value = _decrypt_secret_value(record.secret_ciphertext)
        except ValueError as exc:
            return False, "workspace_invalid", str(exc)
        if not secret_value.strip():
            return False, "workspace_invalid", "El secreto cifrado del workspace esta vacio."
        return True, "workspace_ready", "Se resolvio el secreto cifrado del workspace."
    if record.secret_ref:
        return (
            False,
            "workspace_reference_pending",
            "Hay una referencia externa configurada, pero el runtime todavia no resuelve secret_ref para este provider.",
        )
    return False, "workspace_missing", "No existe material secreto utilizable para este workspace."


def _build_secret_response(
    session: Session,
    *,
    workspace_id: UUID,
    provider_key: LLMProviderKey,
    runtime_settings: LLMRuntimeSettings,
    uses_platform_credentials: bool,
    record: WorkspaceProviderSecretRecord | None,
    supports_workspace_secrets: bool,
) -> WorkspaceProviderSecretResponse:
    active_for_runtime = runtime_settings.active_provider == provider_key and not uses_platform_credentials
    workspace_configured = _record_is_configured(record)
    last_rotated_at = record.last_rotated_at if record is not None else None
    updated_at = record.updated_at if record is not None else None
    secret_kind = record.secret_kind if record is not None else "api_key"

    if provider_key == LLMProviderKey.openai:
        platform_configured = runtime_settings.openai.api_key_configured
    elif provider_key == LLMProviderKey.deepseek:
        platform_configured = runtime_settings.deepseek.api_key_configured
    else:
        platform_configured = False

    if not supports_workspace_secrets:
        health_status = "local_runtime_ready" if runtime_settings.codex_local.available else "local_runtime_missing"
        return WorkspaceProviderSecretResponse(
            provider_key=provider_key,
            workspace_id=workspace_id,
            secret_kind=secret_kind,
            configured=False,
            uses_platform_credentials=uses_platform_credentials,
            secret_source="local_runtime",
            status=RuntimeSecretStatus.not_configured,
            health_status=health_status,
            last_rotated_at=last_rotated_at,
            updated_at=updated_at,
            active_for_runtime=False,
            storage_mode=_storage_mode(record),
            supports_workspace_secrets=False,
        )

    if active_for_runtime:
        secret_source = "workspace_managed"
        configured = workspace_configured
        _, health_status, _ = _workspace_secret_runtime_state(record)
        status = record.status if record is not None else RuntimeSecretStatus.not_configured
    elif workspace_configured:
        secret_source = "workspace_staged"
        configured = True
        _, health_status, _ = _workspace_secret_runtime_state(record)
        status = record.status
    else:
        secret_source = "platform_managed"
        configured = platform_configured
        health_status = "platform_ready" if platform_configured else "platform_missing"
        status = RuntimeSecretStatus.configured if platform_configured else RuntimeSecretStatus.not_configured

    return WorkspaceProviderSecretResponse(
        provider_key=provider_key,
        workspace_id=workspace_id,
        secret_kind=secret_kind,
        configured=configured,
        uses_platform_credentials=uses_platform_credentials,
        secret_source=secret_source,
        status=status,
        health_status=health_status,
        last_rotated_at=last_rotated_at,
        updated_at=updated_at,
        active_for_runtime=active_for_runtime,
        storage_mode=_storage_mode(record),
        supports_workspace_secrets=True,
    )


def _record_secret_audit(
    session: Session,
    *,
    workspace_id: UUID,
    change_type: str,
    before_payload: dict[str, object],
    after_payload: dict[str, object],
    actor_user_id: UUID | None,
) -> None:
    session.add(
        RuntimeSettingsAuditRecord(
            scope_type=RuntimeGovernanceScopeType.workspace,
            scope_id=str(workspace_id),
            change_type=change_type,
            before_payload_redacted=before_payload,
            after_payload_redacted=after_payload,
            actor_user_id=actor_user_id,
            actor_email=_resolve_actor_email(session, actor_user_id),
        )
    )


def _annotate_provider_secret_state(
    session: Session,
    record: WorkspaceProviderSecretRecord | None,
    runtime_settings: LLMRuntimeSettings,
    *,
    secret_view: WorkspaceProviderSecretResponse,
) -> LLMRuntimeSettings:
    if secret_view.secret_source in {"workspace_managed", "workspace_staged"}:
        available, _, status_note = _workspace_secret_runtime_state(record)
    else:
        available = None
        status_note = ""

    if secret_view.provider_key == LLMProviderKey.openai:
        openai_config = runtime_settings.openai.model_copy(
            update={
                "api_key_configured": secret_view.configured,
                "available": runtime_settings.openai.available if available is None else available,
                "secret_source": secret_view.secret_source,
                "last_rotated_at": secret_view.last_rotated_at,
                "health_status": secret_view.health_status,
                "status_note": runtime_settings.openai.status_note if not status_note else status_note,
            }
        )
        updated_settings = runtime_settings.model_copy(update={"openai": openai_config})
    elif secret_view.provider_key == LLMProviderKey.deepseek:
        deepseek_config = runtime_settings.deepseek.model_copy(
            update={
                "api_key_configured": secret_view.configured,
                "available": runtime_settings.deepseek.available if available is None else available,
                "secret_source": secret_view.secret_source,
                "last_rotated_at": secret_view.last_rotated_at,
                "health_status": secret_view.health_status,
                "status_note": runtime_settings.deepseek.status_note if not status_note else status_note,
            }
        )
        updated_settings = runtime_settings.model_copy(update={"deepseek": deepseek_config})
    else:
        codex_config = runtime_settings.codex_local.model_copy(
            update={
                "secret_source": secret_view.secret_source,
                "last_rotated_at": secret_view.last_rotated_at,
                "health_status": secret_view.health_status,
            }
        )
        updated_settings = runtime_settings.model_copy(update={"codex_local": codex_config})
    return updated_settings


def annotate_runtime_settings_with_workspace_secrets(
    session: Session,
    workspace_id: UUID,
    runtime_settings: LLMRuntimeSettings,
) -> LLMRuntimeSettings:
    backfill_platform_runtime_governance(session)
    runtime_record = load_workspace_runtime_settings(session, workspace_id)
    uses_platform_credentials = True if runtime_record is None else runtime_record.uses_platform_credentials
    secret_records = _workspace_secret_records(session, workspace_id)

    updated_settings = runtime_settings.model_copy(update={"uses_platform_credentials": uses_platform_credentials})
    for provider_key in LLMProviderKey:
        supports_workspace_secrets = _supports_workspace_secrets(session, provider_key)
        record = secret_records.get(provider_key)
        secret_view = _build_secret_response(
            session,
            workspace_id=workspace_id,
            provider_key=provider_key,
            runtime_settings=updated_settings,
            uses_platform_credentials=uses_platform_credentials,
            record=record,
            supports_workspace_secrets=supports_workspace_secrets,
        )
        updated_settings = _annotate_provider_secret_state(session, record, updated_settings, secret_view=secret_view)
    return updated_settings.model_copy(update={"provider_options": build_provider_options(updated_settings)})


def build_workspace_provider_secret_view(
    session: Session,
    workspace_id: UUID,
    provider_key: LLMProviderKey,
) -> WorkspaceProviderSecretResponse:
    backfill_platform_runtime_governance(session)
    runtime_settings = load_effective_runtime_settings(session, workspace_id)
    runtime_record = load_workspace_runtime_settings(session, workspace_id)
    uses_platform_credentials = True if runtime_record is None else runtime_record.uses_platform_credentials
    record = _workspace_secret_record(session, workspace_id, provider_key)
    supports_workspace_secrets = _supports_workspace_secrets(session, provider_key)
    return _build_secret_response(
        session,
        workspace_id=workspace_id,
        provider_key=provider_key,
        runtime_settings=runtime_settings,
        uses_platform_credentials=uses_platform_credentials,
        record=record,
        supports_workspace_secrets=supports_workspace_secrets,
    )


def upsert_workspace_provider_secret(
    session: Session,
    workspace_id: UUID,
    provider_key: LLMProviderKey,
    payload: WorkspaceProviderSecretUpsertRequest,
    *,
    actor_user_id: UUID | None = None,
    rotate: bool = False,
) -> WorkspaceProviderSecretResponse:
    backfill_platform_runtime_governance(session)
    if not _supports_workspace_secrets(session, provider_key):
        raise ValueError(f"El provider {provider_key.value} no soporta secretos por workspace.")

    before_view = build_workspace_provider_secret_view(session, workspace_id, provider_key)
    record = _workspace_secret_record(session, workspace_id, provider_key)
    now = utc_now()
    if record is None:
        record = WorkspaceProviderSecretRecord(
            workspace_id=workspace_id,
            provider_key=provider_key,
            secret_kind=payload.secret_kind.strip() or "api_key",
            created_at=now,
        )

    record.secret_kind = payload.secret_kind.strip() or record.secret_kind or "api_key"
    secret_value = payload.secret_value.strip()
    secret_ref = payload.secret_ref.strip()
    if secret_value:
        record.secret_ciphertext = _encrypt_secret_value(secret_value)
        record.secret_ref = ""
    else:
        record.secret_ciphertext = ""
        record.secret_ref = secret_ref
    record.status = RuntimeSecretStatus.configured
    record.last_rotated_at = now if rotate or before_view.configured else now
    record.updated_by_user_id = actor_user_id
    record.updated_at = now
    session.add(record)
    session.commit()

    if payload.activate_for_runtime:
        persist_workspace_runtime_credential_mode(
            session,
            workspace_id,
            uses_platform_credentials=False,
            actor_user_id=actor_user_id,
            mirror_legacy_runtime=False,
        )

    after_view = build_workspace_provider_secret_view(session, workspace_id, provider_key)
    _record_secret_audit(
        session,
        workspace_id=workspace_id,
        change_type="workspace_provider_secret_rotated" if rotate else "workspace_provider_secret_upserted",
        before_payload=before_view.model_dump(mode="json"),
        after_payload=after_view.model_dump(mode="json"),
        actor_user_id=actor_user_id,
    )
    session.commit()
    return after_view


def rotate_workspace_provider_secret(
    session: Session,
    workspace_id: UUID,
    provider_key: LLMProviderKey,
    payload: WorkspaceProviderSecretUpsertRequest,
    *,
    actor_user_id: UUID | None = None,
) -> WorkspaceProviderSecretResponse:
    return upsert_workspace_provider_secret(
        session,
        workspace_id,
        provider_key,
        payload,
        actor_user_id=actor_user_id,
        rotate=True,
    )


def delete_workspace_provider_secret(
    session: Session,
    workspace_id: UUID,
    provider_key: LLMProviderKey,
    *,
    actor_user_id: UUID | None = None,
) -> WorkspaceProviderSecretResponse:
    backfill_platform_runtime_governance(session)
    if not _supports_workspace_secrets(session, provider_key):
        raise ValueError(f"El provider {provider_key.value} no soporta secretos por workspace.")

    runtime_settings = load_effective_runtime_settings(session, workspace_id)
    before_view = build_workspace_provider_secret_view(session, workspace_id, provider_key)
    record = _workspace_secret_record(session, workspace_id, provider_key)
    if record is None:
        raise LookupError("No existe un secreto configurado para ese provider.")

    session.delete(record)
    session.commit()

    if runtime_settings.active_provider == provider_key and not runtime_settings.uses_platform_credentials:
        persist_workspace_runtime_credential_mode(
            session,
            workspace_id,
            uses_platform_credentials=True,
            actor_user_id=actor_user_id,
            mirror_legacy_runtime=False,
        )

    after_view = build_workspace_provider_secret_view(session, workspace_id, provider_key)
    _record_secret_audit(
        session,
        workspace_id=workspace_id,
        change_type="workspace_provider_secret_deleted",
        before_payload=before_view.model_dump(mode="json"),
        after_payload=after_view.model_dump(mode="json"),
        actor_user_id=actor_user_id,
    )
    session.commit()
    return after_view
