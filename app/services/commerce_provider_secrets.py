from __future__ import annotations

import base64
import hashlib
from typing import Any
from uuid import UUID

from cryptography.fernet import Fernet, InvalidToken
from sqlmodel import Session, select

from app.core.config import get_settings
from app.models import (
    CommerceProviderConfigRecord,
    CommerceProviderCredentialUpsertRequest,
    CommerceProviderSecretRecord,
    CommerceProviderSecretStatusResponse,
    CommerceProviderStatusResponse,
    utc_now,
)
from app.services.commerce_provider_registry import get_commerce_provider_registry
from app.services.commerce_provider_utils import (
    normalize_commerce_provider_environment,
    normalize_commerce_provider_key,
)


COMMERCE_PROVIDER_SECRET_KINDS: dict[str, tuple[str, ...]] = {
    "sandbox": (),
    "hotmart": (),
    "rebill": ("secret_key", "public_key", "webhook_signing_secret", "webhook_url_secret"),
}


def _fernet() -> Fernet:
    settings = get_settings()
    if settings.runtime_secrets_master_key.strip():
        return Fernet(settings.runtime_secrets_master_key.strip().encode("utf-8"))
    fallback_material = (
        f"{settings.database_url}|{settings.local_admin_email}|{settings.local_admin_password}|commerce_provider".encode(
            "utf-8"
        )
    )
    derived_key = base64.urlsafe_b64encode(hashlib.sha256(fallback_material).digest())
    return Fernet(derived_key)


def _encrypt_secret_value(secret_value: str) -> str:
    return _fernet().encrypt(secret_value.encode("utf-8")).decode("utf-8")


def _decrypt_secret_value(secret_ciphertext: str) -> str:
    try:
        return _fernet().decrypt(secret_ciphertext.encode("utf-8")).decode("utf-8")
    except InvalidToken as exc:
        raise ValueError("Commerce provider credential could not be decrypted with the configured master key.") from exc


def _provider_secret_kinds(provider_key: str) -> tuple[str, ...]:
    return COMMERCE_PROVIDER_SECRET_KINDS.get(provider_key, ("secret_key", "webhook_signing_secret", "webhook_url_secret"))


def _config_record(
    session: Session,
    *,
    workspace_id: UUID,
    provider_key: str,
    environment: str,
) -> CommerceProviderConfigRecord | None:
    return session.exec(
        select(CommerceProviderConfigRecord).where(
            CommerceProviderConfigRecord.workspace_id == workspace_id,
            CommerceProviderConfigRecord.provider_key == provider_key,
            CommerceProviderConfigRecord.environment == environment,
        )
    ).first()


def _secret_record(
    session: Session,
    *,
    workspace_id: UUID,
    provider_key: str,
    environment: str,
    secret_kind: str,
) -> CommerceProviderSecretRecord | None:
    return session.exec(
        select(CommerceProviderSecretRecord).where(
            CommerceProviderSecretRecord.workspace_id == workspace_id,
            CommerceProviderSecretRecord.provider_key == provider_key,
            CommerceProviderSecretRecord.environment == environment,
            CommerceProviderSecretRecord.secret_kind == secret_kind,
        )
    ).first()


def _secret_records(
    session: Session,
    *,
    workspace_id: UUID,
    provider_key: str,
    environment: str,
) -> dict[str, CommerceProviderSecretRecord]:
    records = session.exec(
        select(CommerceProviderSecretRecord).where(
            CommerceProviderSecretRecord.workspace_id == workspace_id,
            CommerceProviderSecretRecord.provider_key == provider_key,
            CommerceProviderSecretRecord.environment == environment,
        )
    ).all()
    return {record.secret_kind: record for record in records}


def _settings_secret_value(provider_key: str, environment: str, secret_kind: str) -> str:
    settings = get_settings()
    if provider_key != "rebill":
        return ""
    if normalize_commerce_provider_environment(settings.rebill_environment) != environment:
        return ""
    mapping = {
        "secret_key": settings.rebill_secret_key,
        "public_key": settings.rebill_public_key,
        "webhook_signing_secret": settings.rebill_webhook_signing_secret,
        "webhook_url_secret": settings.rebill_webhook_url_secret,
    }
    return str(mapping.get(secret_kind, "") or "").strip()


def _settings_enabled(provider_key: str, environment: str) -> bool:
    settings = get_settings()
    if provider_key == "sandbox":
        return True
    if provider_key == "rebill" and normalize_commerce_provider_environment(settings.rebill_environment) == environment:
        return bool(settings.rebill_enabled)
    return False


def _settings_api_base_url(provider_key: str) -> str:
    settings = get_settings()
    if provider_key == "rebill":
        return settings.rebill_api_base_url.rstrip("/")
    return ""


def _settings_webhook_public_url(provider_key: str) -> str:
    settings = get_settings()
    if provider_key == "rebill":
        return settings.rebill_webhook_public_url
    return ""


def _is_configured(record: CommerceProviderSecretRecord | None) -> bool:
    return bool(record and record.configured and record.status == "configured" and (record.secret_ciphertext or record.secret_ref))


def load_commerce_provider_secret(
    session: Session,
    *,
    workspace_id: UUID,
    provider_key: str,
    environment: str = "sandbox",
    secret_kind: str,
) -> str:
    provider = normalize_commerce_provider_key(provider_key)
    env = normalize_commerce_provider_environment(environment)
    env_value = _settings_secret_value(provider, env, secret_kind)
    if env_value:
        return env_value
    record = _secret_record(
        session,
        workspace_id=workspace_id,
        provider_key=provider,
        environment=env,
        secret_kind=secret_kind,
    )
    if not _is_configured(record) or record is None or record.secret_ref:
        return ""
    return _decrypt_secret_value(record.secret_ciphertext)


def _secret_status(
    session: Session,
    *,
    workspace_id: UUID,
    provider_key: str,
    environment: str,
    secret_kind: str,
) -> CommerceProviderSecretStatusResponse:
    record = _secret_record(
        session,
        workspace_id=workspace_id,
        provider_key=provider_key,
        environment=environment,
        secret_kind=secret_kind,
    )
    env_configured = bool(_settings_secret_value(provider_key, environment, secret_kind))
    configured = bool(_is_configured(record) or env_configured)
    if record is not None and record.secret_ref:
        storage_mode = "reference"
    elif record is not None and record.secret_ciphertext:
        storage_mode = "ciphertext"
    elif env_configured:
        storage_mode = "environment"
    else:
        storage_mode = "none"
    return CommerceProviderSecretStatusResponse(
        secret_kind=secret_kind,
        configured=configured,
        status="configured" if configured else "not_configured",
        storage_mode=storage_mode,
        last_rotated_at=record.last_rotated_at if record is not None else None,
    )


def _computed_status(secret_statuses: list[CommerceProviderSecretStatusResponse], *, provider_key: str) -> str:
    required = {"rebill": {"secret_key"}}.get(provider_key, set())
    if not required:
        return "configured" if provider_key == "sandbox" else "not_configured"
    configured = {item.secret_kind for item in secret_statuses if item.configured}
    if required.issubset(configured):
        return "configured"
    if configured:
        return "partial_configured"
    return "not_configured"


def build_commerce_provider_status(
    session: Session,
    *,
    workspace_id: UUID,
    provider_key: str,
    environment: str = "sandbox",
) -> CommerceProviderStatusResponse:
    provider = normalize_commerce_provider_key(provider_key)
    env = normalize_commerce_provider_environment(environment)
    definition = get_commerce_provider_registry().require_definition(provider)
    if provider == "hotmart":
        from app.services.hotmart.secrets import build_hotmart_status

        hotmart_status = build_hotmart_status(session, workspace_id=workspace_id, environment=env)
        return CommerceProviderStatusResponse(
            workspace_id=workspace_id,
            provider_key=provider,
            environment=env,  # type: ignore[arg-type]
            enabled=hotmart_status.enabled,
            status=hotmart_status.status,
            api_base_url=hotmart_status.api_base_url,
            webhook_public_url=hotmart_status.webhook_public_url,
            capabilities=list(definition.capabilities),
            secret_statuses=[
                CommerceProviderSecretStatusResponse(
                    secret_kind="client_id",
                    configured=hotmart_status.client_id_configured,
                    status="configured" if hotmart_status.client_id_configured else "not_configured",
                    storage_mode=hotmart_status.storage_mode,
                    last_rotated_at=None,
                ),
                CommerceProviderSecretStatusResponse(
                    secret_kind="client_secret",
                    configured=hotmart_status.client_secret_configured,
                    status="configured" if hotmart_status.client_secret_configured else "not_configured",
                    storage_mode=hotmart_status.storage_mode,
                    last_rotated_at=None,
                ),
                CommerceProviderSecretStatusResponse(
                    secret_kind="basic_token",
                    configured=hotmart_status.basic_token_configured,
                    status="configured" if hotmart_status.basic_token_configured else "not_configured",
                    storage_mode=hotmart_status.storage_mode,
                    last_rotated_at=None,
                ),
                CommerceProviderSecretStatusResponse(
                    secret_kind="hottok",
                    configured=hotmart_status.hottok_configured,
                    status="configured" if hotmart_status.hottok_configured else "not_configured",
                    storage_mode=hotmart_status.storage_mode,
                    last_rotated_at=None,
                ),
            ],
            last_health_check_at=hotmart_status.last_health_check_at,
            last_health_status=hotmart_status.last_health_status,
            last_health_message=hotmart_status.last_health_message,
            updated_at=hotmart_status.updated_at,
        )
    config = _config_record(session, workspace_id=workspace_id, provider_key=provider, environment=env)
    secret_statuses = [
        _secret_status(
            session,
            workspace_id=workspace_id,
            provider_key=provider,
            environment=env,
            secret_kind=secret_kind,
        )
        for secret_kind in _provider_secret_kinds(provider)
    ]
    computed_status = _computed_status(secret_statuses, provider_key=provider)
    enabled = config.enabled if config is not None else _settings_enabled(provider, env)
    return CommerceProviderStatusResponse(
        workspace_id=workspace_id,
        provider_key=provider,
        environment=env,  # type: ignore[arg-type]
        enabled=enabled,
        status=config.status if config is not None and config.status != "not_configured" else computed_status,
        api_base_url=(config.api_base_url if config is not None and config.api_base_url else _settings_api_base_url(provider)),
        webhook_public_url=(
            config.webhook_public_url if config is not None and config.webhook_public_url else _settings_webhook_public_url(provider)
        ),
        capabilities=list(definition.capabilities),
        secret_statuses=secret_statuses,
        last_health_check_at=config.last_checked_at if config is not None else None,
        last_health_status=config.last_health_status if config is not None else "",
        last_health_message=config.last_health_message if config is not None else "",
        updated_at=config.updated_at if config is not None else None,
    )


def upsert_commerce_provider_credentials(
    session: Session,
    *,
    workspace_id: UUID,
    provider_key: str,
    payload: CommerceProviderCredentialUpsertRequest,
    actor_user_id: UUID | None = None,
) -> CommerceProviderStatusResponse:
    provider = normalize_commerce_provider_key(provider_key)
    env = normalize_commerce_provider_environment(payload.environment)
    definition = get_commerce_provider_registry().require_definition(provider)
    config = _config_record(session, workspace_id=workspace_id, provider_key=provider, environment=env)
    if config is None:
        config = CommerceProviderConfigRecord(
            workspace_id=workspace_id,
            provider_key=provider,
            environment=env,
            created_by_user_id=actor_user_id,
        )
    config.enabled = payload.enabled
    config.api_base_url = payload.api_base_url.strip().rstrip("/") or _settings_api_base_url(provider)
    config.webhook_public_url = payload.webhook_public_url.strip() or _settings_webhook_public_url(provider)
    config.capabilities = {capability: True for capability in definition.capabilities}
    config.updated_by_user_id = actor_user_id
    config.updated_at = utc_now()
    session.add(config)
    session.flush()

    allowed_secret_kinds = set(_provider_secret_kinds(provider))
    for secret_kind, secret_value in payload.secrets.items():
        normalized_kind = secret_kind.strip().lower()
        if not normalized_kind or normalized_kind not in allowed_secret_kinds:
            raise ValueError(f"Unsupported secret kind for {provider}: {secret_kind}")
        if not str(secret_value or "").strip():
            continue
        record = _secret_record(
            session,
            workspace_id=workspace_id,
            provider_key=provider,
            environment=env,
            secret_kind=normalized_kind,
        )
        if record is None:
            record = CommerceProviderSecretRecord(
                workspace_id=workspace_id,
                provider_key=provider,
                environment=env,
                secret_kind=normalized_kind,
            )
        record.secret_ciphertext = _encrypt_secret_value(str(secret_value).strip())
        record.secret_ref = ""
        record.configured = True
        record.status = "configured"
        record.last_rotated_at = utc_now()
        record.updated_by_user_id = actor_user_id
        record.updated_at = utc_now()
        session.add(record)

    session.flush()
    status = build_commerce_provider_status(session, workspace_id=workspace_id, provider_key=provider, environment=env)
    config.status = status.status
    config.updated_at = utc_now()
    session.add(config)
    session.flush()
    return build_commerce_provider_status(session, workspace_id=workspace_id, provider_key=provider, environment=env)
