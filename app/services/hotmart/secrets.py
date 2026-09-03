from __future__ import annotations

import base64
import hashlib
from typing import Any
from uuid import UUID

import httpx
from cryptography.fernet import Fernet, InvalidToken
from sqlmodel import Session, select

from app.core.config import get_settings
from app.models import (
    HotmartCredentialUpsertRequest,
    HotmartIntegrationConfigRecord,
    HotmartIntegrationSecretRecord,
    HotmartIntegrationStatusResponse,
    HotmartTestConnectionResponse,
    utc_now,
)
from app.services.hotmart.auth import (
    HotmartCredentials,
    default_hotmart_api_base_url,
    default_hotmart_auth_base_url,
    normalize_hotmart_environment,
)
from app.services.hotmart.client import HotmartClient, HotmartClientConfig


HOTMART_SECRET_KINDS = ("client_id", "client_secret", "basic_token", "hottok")
HOTMART_OAUTH_SECRET_KINDS = ("client_id", "client_secret", "basic_token")
LEGACY_SANDBOX_AUTH_BASE_URL = "https://sandbox.hotmart.com"


def _fernet() -> Fernet:
    settings = get_settings()
    if settings.runtime_secrets_master_key.strip():
        return Fernet(settings.runtime_secrets_master_key.strip().encode("utf-8"))
    fallback_material = (
        f"{settings.database_url}|{settings.local_admin_email}|{settings.local_admin_password}|hotmart".encode("utf-8")
    )
    derived_key = base64.urlsafe_b64encode(hashlib.sha256(fallback_material).digest())
    return Fernet(derived_key)


def _encrypt_secret_value(secret_value: str) -> str:
    return _fernet().encrypt(secret_value.encode("utf-8")).decode("utf-8")


def _decrypt_secret_value(secret_ciphertext: str) -> str:
    try:
        return _fernet().decrypt(secret_ciphertext.encode("utf-8")).decode("utf-8")
    except InvalidToken as exc:
        raise ValueError("Hotmart credential could not be decrypted with the configured master key.") from exc


def _resolved_api_base_url(environment: str, explicit_value: str = "") -> str:
    return (explicit_value or get_settings().hotmart_api_base_url or default_hotmart_api_base_url(environment)).rstrip("/")


def _resolved_auth_base_url(environment: str, explicit_value: str = "") -> str:
    env = normalize_hotmart_environment(environment)
    configured_value = (explicit_value or get_settings().hotmart_auth_base_url).strip().rstrip("/")
    if env == "sandbox" and configured_value == LEGACY_SANDBOX_AUTH_BASE_URL:
        configured_value = ""
    return (configured_value or default_hotmart_auth_base_url(env)).rstrip("/")


def _resolved_webhook_public_url(explicit_value: str = "") -> str:
    return explicit_value or get_settings().hotmart_webhook_public_url


def _config_record(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str,
) -> HotmartIntegrationConfigRecord | None:
    return session.exec(
        select(HotmartIntegrationConfigRecord).where(
            HotmartIntegrationConfigRecord.workspace_id == workspace_id,
            HotmartIntegrationConfigRecord.environment == environment,
        )
    ).first()


def _secret_record(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str,
    secret_kind: str,
) -> HotmartIntegrationSecretRecord | None:
    return session.exec(
        select(HotmartIntegrationSecretRecord).where(
            HotmartIntegrationSecretRecord.workspace_id == workspace_id,
            HotmartIntegrationSecretRecord.environment == environment,
            HotmartIntegrationSecretRecord.secret_kind == secret_kind,
        )
    ).first()


def _secret_records(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str,
) -> dict[str, HotmartIntegrationSecretRecord]:
    rows = session.exec(
        select(HotmartIntegrationSecretRecord).where(
            HotmartIntegrationSecretRecord.workspace_id == workspace_id,
            HotmartIntegrationSecretRecord.environment == environment,
        )
    ).all()
    return {record.secret_kind: record for record in rows}


def _is_configured(record: HotmartIntegrationSecretRecord | None) -> bool:
    return bool(record and record.status == "configured" and (record.secret_ciphertext or record.secret_ref))


def _configured_flags(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str,
) -> dict[str, bool]:
    records = _secret_records(session, workspace_id=workspace_id, environment=environment)
    return {secret_kind: _is_configured(records.get(secret_kind)) for secret_kind in HOTMART_SECRET_KINDS}


def _settings_credentials_configured(environment: str) -> dict[str, bool]:
    settings = get_settings()
    if normalize_hotmart_environment(settings.hotmart_environment) != environment:
        return {secret_kind: False for secret_kind in HOTMART_SECRET_KINDS}
    return {
        "client_id": bool(settings.hotmart_client_id.strip()),
        "client_secret": bool(settings.hotmart_client_secret.strip()),
        "basic_token": bool(settings.hotmart_basic_token.strip()),
        "hottok": bool(settings.hotmart_hottok.strip()),
    }


def _settings_hotmart_credentials(environment: str) -> HotmartCredentials | None:
    settings = get_settings()
    if normalize_hotmart_environment(settings.hotmart_environment) != environment:
        return None
    if not (
        settings.hotmart_client_id.strip()
        and settings.hotmart_client_secret.strip()
        and settings.hotmart_basic_token.strip()
    ):
        return None
    return HotmartCredentials(
        client_id=settings.hotmart_client_id.strip(),
        client_secret=settings.hotmart_client_secret.strip(),
        basic_token=settings.hotmart_basic_token.strip(),
    )


def _settings_hotmart_hottok(environment: str) -> str:
    settings = get_settings()
    if normalize_hotmart_environment(settings.hotmart_environment) != environment:
        return ""
    return settings.hotmart_hottok.strip()


def _secret_fingerprint(secret_value: str) -> dict[str, Any]:
    value = (secret_value or "").strip()
    return {
        "present": bool(value),
        "length": len(value),
        "sha256_prefix": hashlib.sha256(value.encode("utf-8")).hexdigest()[:12] if value else "",
    }


def _merge_configured_flags(
    record_flags: dict[str, bool],
    settings_flags: dict[str, bool],
) -> dict[str, bool]:
    return {secret_kind: bool(record_flags.get(secret_kind) or settings_flags.get(secret_kind)) for secret_kind in HOTMART_SECRET_KINDS}


def _resolved_configured_flags(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str,
) -> dict[str, bool]:
    return _merge_configured_flags(
        _configured_flags(session, workspace_id=workspace_id, environment=environment),
        _settings_credentials_configured(environment),
    )


def _status_from_configured_flags(flags: dict[str, bool]) -> str:
    oauth_ready = all(flags[kind] for kind in HOTMART_OAUTH_SECRET_KINDS)
    return "configured" if oauth_ready else ("partial_configured" if any(flags.values()) else "not_configured")


def _storage_mode(records: dict[str, HotmartIntegrationSecretRecord], *, environment: str) -> str:
    configured = [record for record in records.values() if _is_configured(record)]
    if not configured:
        settings = get_settings()
        if normalize_hotmart_environment(settings.hotmart_environment) != environment:
            return "none"
        if any(
            (
                settings.hotmart_client_id.strip(),
                settings.hotmart_client_secret.strip(),
                settings.hotmart_basic_token.strip(),
                settings.hotmart_hottok.strip(),
            )
        ):
            return "environment"
        return "none"
    if any(record.secret_ref for record in configured):
        return "reference"
    return "ciphertext"


def _sync_configured_flags(session: Session, config: HotmartIntegrationConfigRecord) -> None:
    flags = _resolved_configured_flags(session, workspace_id=config.workspace_id, environment=config.environment)
    config.client_id_configured = flags["client_id"]
    config.client_secret_configured = flags["client_secret"]
    config.basic_token_configured = flags["basic_token"]
    config.hottok_configured = flags["hottok"]
    config.status = _status_from_configured_flags(flags)
    config.updated_at = utc_now()
    session.add(config)


def build_hotmart_status(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str = "sandbox",
) -> HotmartIntegrationStatusResponse:
    env = normalize_hotmart_environment(environment)
    config = _config_record(session, workspace_id=workspace_id, environment=env)
    records = _secret_records(session, workspace_id=workspace_id, environment=env)
    flags = _resolved_configured_flags(session, workspace_id=workspace_id, environment=env)
    computed_status = _status_from_configured_flags(flags)

    if config is None:
        return HotmartIntegrationStatusResponse(
            workspace_id=workspace_id,
            environment=env,  # type: ignore[arg-type]
            enabled=bool(get_settings().hotmart_enabled),
            status=computed_status,
            client_id_configured=flags["client_id"],
            client_secret_configured=flags["client_secret"],
            basic_token_configured=flags["basic_token"],
            hottok_configured=flags["hottok"],
            api_base_url=_resolved_api_base_url(env),
            auth_base_url=_resolved_auth_base_url(env),
            webhook_public_url=_resolved_webhook_public_url(),
            storage_mode=_storage_mode(records, environment=env),
        )

    return HotmartIntegrationStatusResponse(
        workspace_id=workspace_id,
        environment=env,  # type: ignore[arg-type]
        enabled=config.enabled,
        status=computed_status,
        client_id_configured=flags["client_id"],
        client_secret_configured=flags["client_secret"],
        basic_token_configured=flags["basic_token"],
        hottok_configured=flags["hottok"],
        api_base_url=_resolved_api_base_url(env, config.api_base_url),
        auth_base_url=_resolved_auth_base_url(env, config.auth_base_url),
        webhook_public_url=config.webhook_public_url or _resolved_webhook_public_url(),
        last_health_check_at=config.last_health_check_at,
        last_health_status=config.last_health_status,
        last_health_message=config.last_health_message,
        last_sync_at=config.last_sync_at,
        storage_mode=_storage_mode(records, environment=env),
        updated_at=config.updated_at,
    )


def upsert_hotmart_credentials(
    session: Session,
    *,
    workspace_id: UUID,
    payload: HotmartCredentialUpsertRequest,
    actor_user_id: UUID | None = None,
) -> HotmartIntegrationStatusResponse:
    env = normalize_hotmart_environment(payload.environment)
    config = _config_record(session, workspace_id=workspace_id, environment=env)
    if config is None:
        config = HotmartIntegrationConfigRecord(
            workspace_id=workspace_id,
            environment=env,
            created_by_user_id=actor_user_id,
        )
    config.enabled = payload.enabled
    config.api_base_url = _resolved_api_base_url(env, payload.api_base_url)
    config.auth_base_url = _resolved_auth_base_url(env, payload.auth_base_url)
    config.webhook_public_url = _resolved_webhook_public_url(payload.webhook_public_url)
    config.updated_by_user_id = actor_user_id
    config.updated_at = utc_now()
    session.add(config)
    session.flush()

    incoming_values = {
        "client_id": payload.client_id,
        "client_secret": payload.client_secret,
        "basic_token": payload.basic_token,
        "hottok": payload.hottok,
    }
    for secret_kind, secret_value in incoming_values.items():
        if not secret_value.strip():
            continue
        record = _secret_record(
            session,
            workspace_id=workspace_id,
            environment=env,
            secret_kind=secret_kind,
        )
        if record is None:
            record = HotmartIntegrationSecretRecord(
                workspace_id=workspace_id,
                environment=env,
                secret_kind=secret_kind,
            )
        record.secret_ciphertext = _encrypt_secret_value(secret_value.strip())
        record.secret_ref = ""
        record.status = "configured"
        record.last_rotated_at = utc_now()
        record.updated_at = utc_now()
        session.add(record)

    session.flush()
    _sync_configured_flags(session, config)
    session.flush()
    return build_hotmart_status(session, workspace_id=workspace_id, environment=env)


def load_hotmart_credentials(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str = "sandbox",
) -> HotmartCredentials | None:
    env = normalize_hotmart_environment(environment)
    records = _secret_records(session, workspace_id=workspace_id, environment=env)
    values: dict[str, str] = {}
    for secret_kind in HOTMART_OAUTH_SECRET_KINDS:
        record = records.get(secret_kind)
        if not _is_configured(record) or record is None:
            return _settings_hotmart_credentials(env)
        if record.secret_ref:
            return None
        try:
            values[secret_kind] = _decrypt_secret_value(record.secret_ciphertext)
        except ValueError:
            # Production can retain encrypted DB credentials created with an older
            # master key. If the runtime also provides matching ENV credentials,
            # recover without blocking Hotmart callbacks or admin checks.
            return _settings_hotmart_credentials(env)
    return HotmartCredentials(
        client_id=values["client_id"],
        client_secret=values["client_secret"],
        basic_token=values["basic_token"],
    )


def load_hotmart_hottok(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str = "sandbox",
) -> str:
    value, _diagnostics = resolve_hotmart_hottok(
        session,
        workspace_id=workspace_id,
        environment=environment,
    )
    return value


def resolve_hotmart_hottok(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str = "sandbox",
) -> tuple[str, dict[str, Any]]:
    env = normalize_hotmart_environment(environment)
    env_token = _settings_hotmart_hottok(env)
    diagnostics: dict[str, Any] = {
        "environment": env,
        "workspace_id": str(workspace_id),
        "source": "missing",
        "db_secret_configured": False,
        "db_secret_ref_configured": False,
        "db_decrypt_failed": False,
        "env_candidate_configured": bool(env_token),
        "expected_present": False,
        "expected_length": 0,
        "expected_sha256_prefix": "",
    }

    def finish(source: str, token: str) -> tuple[str, dict[str, Any]]:
        fingerprint = _secret_fingerprint(token)
        diagnostics["source"] = source
        diagnostics["expected_present"] = fingerprint["present"]
        diagnostics["expected_length"] = fingerprint["length"]
        diagnostics["expected_sha256_prefix"] = fingerprint["sha256_prefix"]
        return token, diagnostics

    record = _secret_record(session, workspace_id=workspace_id, environment=env, secret_kind="hottok")
    diagnostics["db_secret_configured"] = _is_configured(record)
    diagnostics["db_secret_ref_configured"] = bool(record and record.secret_ref)
    if env_token:
        return finish("env_primary", env_token)

    if _is_configured(record) and record is not None and not record.secret_ref:
        try:
            return finish("db_ciphertext", _decrypt_secret_value(record.secret_ciphertext))
        except ValueError:
            # Fall back to the runtime ENV token when DB ciphertext cannot be
            # decrypted with this deployment's current master key.
            diagnostics["db_decrypt_failed"] = True
            source = "env_fallback_after_decrypt_error" if env_token else "missing_after_decrypt_error"
            return finish(source, env_token)
    if record is not None and record.secret_ref:
        return finish("external_secret_ref_unresolved", env_token)
    return finish("env" if env_token else "missing", env_token)


def test_hotmart_connection(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str = "sandbox",
    transport: httpx.BaseTransport | None = None,
) -> HotmartTestConnectionResponse:
    env = normalize_hotmart_environment(environment)
    config = _config_record(session, workspace_id=workspace_id, environment=env)
    checked_at = utc_now()
    if config is None:
        flags = _settings_credentials_configured(env)
        oauth_ready = all(flags[kind] for kind in HOTMART_OAUTH_SECRET_KINDS)
        if not oauth_ready:
            return HotmartTestConnectionResponse(
                workspace_id=workspace_id,
                environment=env,  # type: ignore[arg-type]
                reachable=False,
                status="not_configured",
                message="Hotmart integration is not configured for this workspace.",
                checked_at=checked_at,
            )
        config = HotmartIntegrationConfigRecord(
            workspace_id=workspace_id,
            environment=env,
            enabled=bool(get_settings().hotmart_enabled),
            status="configured",
            api_base_url=_resolved_api_base_url(env),
            auth_base_url=_resolved_auth_base_url(env),
            webhook_public_url=_resolved_webhook_public_url(),
        )
        session.add(config)
        session.flush()
        _sync_configured_flags(session, config)
        session.flush()

    credentials = load_hotmart_credentials(session, workspace_id=workspace_id, environment=env)
    if credentials is None:
        config.last_health_check_at = checked_at
        config.last_health_status = "missing_credentials"
        config.last_health_message = "Hotmart OAuth credentials are incomplete or stored as external references."
        config.status = "missing_credentials"
        config.updated_at = checked_at
        session.add(config)
        session.flush()
        return HotmartTestConnectionResponse(
            workspace_id=workspace_id,
            environment=env,  # type: ignore[arg-type]
            reachable=False,
            status="missing_credentials",
            message=config.last_health_message,
            checked_at=checked_at,
        )

    client = HotmartClient(
        HotmartClientConfig(
            environment=env,
            api_base_url=_resolved_api_base_url(env, config.api_base_url),
            auth_base_url=_resolved_auth_base_url(env, config.auth_base_url),
            timeout_seconds=get_settings().hotmart_request_timeout_seconds,
        ),
        transport=transport,
    )
    result = client.test_connection(credentials)
    config.last_health_check_at = checked_at
    config.last_health_status = result.status
    config.last_health_message = result.message
    config.status = "connected" if result.reachable else "connection_failed"
    config.updated_at = checked_at
    session.add(config)
    session.flush()
    return HotmartTestConnectionResponse(
        workspace_id=workspace_id,
        environment=env,  # type: ignore[arg-type]
        reachable=result.reachable,
        status=result.status,
        message=result.message,
        token_expires_in=result.token_expires_in,
        http_status=result.http_status,
        rate_limit_remaining=result.rate_limit_remaining,
        checked_at=checked_at,
    )
