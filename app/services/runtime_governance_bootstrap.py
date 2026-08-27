from __future__ import annotations

from threading import Lock
from typing import Any

from sqlmodel import Session, select

from app.core.config import get_settings
from app.models import (
    AgentExecutionBackend,
    CodexAuthMode,
    CodexLocalProviderConfigUpdate,
    AntigravityProviderConfigUpdate,
    DeepSeekProviderConfigUpdate,
    KnowledgeAccessBackend,
    LLMProviderKey,
    LLMRuntimeSettings,
    OpenAIProviderConfigUpdate,
    PlatformRole,
    PlatformRoleAssignmentRecord,
    PlatformRuntimeDefaultsRecord,
    PlatformRuntimeProviderRecord,
    RuntimeProviderReleaseStage,
    UserRecord,
    utc_now,
)
from app.services.openai_builder import load_llm_runtime_settings


_RUNTIME_GOVERNANCE_CACHE_LOCK = Lock()
_RUNTIME_GOVERNANCE_CACHE: set[tuple[int, str]] = set()


def _normalize_model_list(*values: str) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = value.strip()
        if not token:
            continue
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(token)
    return normalized


def _provider_registry_seed_payloads(runtime_settings: LLMRuntimeSettings) -> list[dict[str, Any]]:
    return [
        {
            "provider_key": LLMProviderKey.openai,
            "label": "OpenAI",
            "is_enabled": True,
            "allowed_models": _normalize_model_list(
                runtime_settings.openai.fast_model,
                runtime_settings.openai.reasoning_model,
            ),
            "default_models": OpenAIProviderConfigUpdate(
                fast_model=runtime_settings.openai.fast_model,
                reasoning_model=runtime_settings.openai.reasoning_model,
                reasoning_effort=runtime_settings.openai.reasoning_effort,
            ).model_dump(mode="json"),
            "allowed_auth_modes": [CodexAuthMode.api_key.value],
            "supports_workspace_secrets": True,
            "supports_platform_managed_credentials": True,
            "release_stage": RuntimeProviderReleaseStage.general_availability,
            "health_policy": {
                "configured_flag": "openai.api_key_configured",
                "availability_flag": "openai.available",
                "status_note_field": "openai.status_note",
            },
        },
        {
            "provider_key": LLMProviderKey.deepseek,
            "label": "DeepSeek",
            "is_enabled": True,
            "allowed_models": _normalize_model_list(
                runtime_settings.deepseek.fast_model,
                runtime_settings.deepseek.reasoning_model,
            ),
            "default_models": DeepSeekProviderConfigUpdate(
                base_url=runtime_settings.deepseek.base_url,
                fast_model=runtime_settings.deepseek.fast_model,
                reasoning_model=runtime_settings.deepseek.reasoning_model,
                reasoning_effort=runtime_settings.deepseek.reasoning_effort,
            ).model_dump(mode="json"),
            "allowed_auth_modes": [CodexAuthMode.api_key.value],
            "supports_workspace_secrets": True,
            "supports_platform_managed_credentials": True,
            "release_stage": RuntimeProviderReleaseStage.general_availability,
            "health_policy": {
                "configured_flag": "deepseek.api_key_configured",
                "availability_flag": "deepseek.available",
                "status_note_field": "deepseek.status_note",
            },
        },
        {
            "provider_key": LLMProviderKey.codex_local,
            "label": "Codex local",
            "is_enabled": True,
            "allowed_models": _normalize_model_list(
                runtime_settings.codex_local.model,
                *runtime_settings.codex_local.fallback_models,
            ),
            "default_models": CodexLocalProviderConfigUpdate(
                command=runtime_settings.codex_local.command,
                model=runtime_settings.codex_local.model,
                profile=runtime_settings.codex_local.profile,
                cost_policy=runtime_settings.codex_local.cost_policy,
                timeout_ms=runtime_settings.codex_local.timeout_ms,
                max_concurrency=runtime_settings.codex_local.max_concurrency,
                runner_id=runtime_settings.codex_local.runner_id,
                auth_mode=runtime_settings.codex_local.auth_mode,
                fallback_models=runtime_settings.codex_local.fallback_models,
                primary_agents=runtime_settings.codex_local.primary_agents,
                shadow_agents=runtime_settings.codex_local.shadow_agents,
                staged_agents=runtime_settings.codex_local.staged_agents,
            ).model_dump(mode="json"),
            "allowed_auth_modes": [item.value for item in CodexAuthMode],
            "supports_workspace_secrets": False,
            "supports_platform_managed_credentials": False,
            "release_stage": RuntimeProviderReleaseStage.preview,
            "health_policy": {
                "configured_flag": "codex_local.command",
                "availability_flag": "codex_local.available",
                "status_note_field": "codex_local.status_note",
            },
        },
        {
            "provider_key": LLMProviderKey.antigravity_cli,
            "label": "Antigravity CLI",
            "is_enabled": True,
            "allowed_models": _normalize_model_list(
                runtime_settings.antigravity.model,
                *runtime_settings.antigravity.fallback_models,
            ),
            "default_models": AntigravityProviderConfigUpdate(
                executable=runtime_settings.antigravity.executable,
                model=runtime_settings.antigravity.model,
                effort=runtime_settings.antigravity.effort,
                timeout_ms=runtime_settings.antigravity.timeout_ms,
                max_concurrency=runtime_settings.antigravity.max_concurrency,
                runner_id=runtime_settings.antigravity.runner_id,
                auth_mode=runtime_settings.antigravity.auth_mode,
                fallback_models=runtime_settings.antigravity.fallback_models,
                primary_agents=runtime_settings.antigravity.primary_agents,
                shadow_agents=runtime_settings.antigravity.shadow_agents,
                staged_agents=runtime_settings.antigravity.staged_agents,
            ).model_dump(mode="json"),
            "allowed_auth_modes": ["auto", "api_key", "session"],
            "supports_workspace_secrets": False,
            "supports_platform_managed_credentials": False,
            "release_stage": RuntimeProviderReleaseStage.preview,
            "health_policy": {
                "configured_flag": "antigravity.executable",
                "availability_flag": "antigravity.available",
                "status_note_field": "antigravity.status_note",
            },
        },
    ]


def _platform_defaults_payload(runtime_settings: LLMRuntimeSettings) -> dict[str, Any]:
    return {
        "active_provider_default": runtime_settings.active_provider,
        "agent_execution_backend_default": AgentExecutionBackend(runtime_settings.agent_execution_backend.value),
        "knowledge_access_backend_default": KnowledgeAccessBackend(runtime_settings.knowledge_access_backend.value),
        "per_provider_defaults": {
            "openai": OpenAIProviderConfigUpdate(
                fast_model=runtime_settings.openai.fast_model,
                reasoning_model=runtime_settings.openai.reasoning_model,
                reasoning_effort=runtime_settings.openai.reasoning_effort,
            ).model_dump(mode="json"),
            "deepseek": DeepSeekProviderConfigUpdate(
                base_url=runtime_settings.deepseek.base_url,
                fast_model=runtime_settings.deepseek.fast_model,
                reasoning_model=runtime_settings.deepseek.reasoning_model,
                reasoning_effort=runtime_settings.deepseek.reasoning_effort,
            ).model_dump(mode="json"),
            "codex_local": CodexLocalProviderConfigUpdate(
                command=runtime_settings.codex_local.command,
                model=runtime_settings.codex_local.model,
                profile=runtime_settings.codex_local.profile,
                cost_policy=runtime_settings.codex_local.cost_policy,
                timeout_ms=runtime_settings.codex_local.timeout_ms,
                max_concurrency=runtime_settings.codex_local.max_concurrency,
                runner_id=runtime_settings.codex_local.runner_id,
                auth_mode=runtime_settings.codex_local.auth_mode,
                fallback_models=runtime_settings.codex_local.fallback_models,
                primary_agents=runtime_settings.codex_local.primary_agents,
                shadow_agents=runtime_settings.codex_local.shadow_agents,
                staged_agents=runtime_settings.codex_local.staged_agents,
            ).model_dump(mode="json"),
            "antigravity": AntigravityProviderConfigUpdate(
                executable=runtime_settings.antigravity.executable,
                model=runtime_settings.antigravity.model,
                effort=runtime_settings.antigravity.effort,
                timeout_ms=runtime_settings.antigravity.timeout_ms,
                max_concurrency=runtime_settings.antigravity.max_concurrency,
                runner_id=runtime_settings.antigravity.runner_id,
                auth_mode=runtime_settings.antigravity.auth_mode,
                fallback_models=runtime_settings.antigravity.fallback_models,
                primary_agents=runtime_settings.antigravity.primary_agents,
                shadow_agents=runtime_settings.antigravity.shadow_agents,
                staged_agents=runtime_settings.antigravity.staged_agents,
            ).model_dump(mode="json"),
        },
        "is_active": True,
        "version": 1,
        "updated_by_user_id": None,
        "updated_at": utc_now(),
    }


def seed_platform_runtime_governance(
    session: Session,
    *,
    runtime_settings: LLMRuntimeSettings | None = None,
    seed_defaults: bool = False,
) -> dict[str, Any]:
    runtime_settings = runtime_settings or load_llm_runtime_settings()

    existing_provider_rows = session.exec(select(PlatformRuntimeProviderRecord)).all()
    existing_by_key = {item.provider_key: item for item in existing_provider_rows}
    provider_created = 0
    for payload in _provider_registry_seed_payloads(runtime_settings):
        existing = existing_by_key.get(payload["provider_key"])
        if existing is None:
            session.add(PlatformRuntimeProviderRecord(**payload))
            provider_created += 1

    defaults_created = False
    active_defaults = session.exec(
        select(PlatformRuntimeDefaultsRecord)
        .where(PlatformRuntimeDefaultsRecord.is_active == True)  # noqa: E712
        .order_by(PlatformRuntimeDefaultsRecord.version.desc())
    ).first()
    if seed_defaults and active_defaults is None:
        session.add(PlatformRuntimeDefaultsRecord(**_platform_defaults_payload(runtime_settings)))
        defaults_created = True

    if provider_created or defaults_created:
        session.commit()

    return {
        "defaults_created": defaults_created,
        "provider_created": provider_created,
        "provider_existing": len(existing_provider_rows),
    }


def ensure_platform_admin_role(session: Session, user: UserRecord) -> None:
    assignment = session.exec(
        select(PlatformRoleAssignmentRecord).where(
            PlatformRoleAssignmentRecord.user_id == user.id,
            PlatformRoleAssignmentRecord.role == PlatformRole.platform_admin,
        )
    ).first()
    if assignment is None:
        session.add(
            PlatformRoleAssignmentRecord(
                user_id=user.id,
                role=PlatformRole.platform_admin,
            )
        )
        session.commit()
        return

    if not assignment.is_active:
        assignment.is_active = True
        assignment.updated_at = utc_now()
        session.add(assignment)
        session.commit()


def sync_configured_platform_admin(session: Session) -> None:
    configured_email = get_settings().local_admin_email.strip().lower()
    configured_user = session.exec(select(UserRecord).where(UserRecord.email == configured_email)).first()
    if configured_user is None:
        return

    ensure_platform_admin_role(session, configured_user)

    stale_assignments = list(
        session.exec(
            select(PlatformRoleAssignmentRecord).where(
                PlatformRoleAssignmentRecord.role == PlatformRole.platform_admin,
                PlatformRoleAssignmentRecord.is_active == True,  # noqa: E712
                PlatformRoleAssignmentRecord.user_id != configured_user.id,
            )
        ).all()
    )
    if not stale_assignments:
        return

    for assignment in stale_assignments:
        assignment.is_active = False
        assignment.updated_at = utc_now()
        session.add(assignment)
    session.commit()


def _runtime_governance_cache_key(session: Session) -> tuple[int, str]:
    bind = session.get_bind()
    engine = getattr(bind, "engine", bind)
    return id(engine), get_settings().local_admin_email.strip().lower()


def _runtime_governance_seed_present(session: Session) -> bool:
    provider_exists = session.exec(select(PlatformRuntimeProviderRecord.id)).first() is not None
    if not provider_exists:
        return False

    configured_email = get_settings().local_admin_email.strip().lower()
    if not configured_email:
        return True

    configured_user_id = session.exec(select(UserRecord.id).where(UserRecord.email == configured_email)).first()
    if configured_user_id is None:
        return True

    assignment_exists = session.exec(
        select(PlatformRoleAssignmentRecord.id).where(
            PlatformRoleAssignmentRecord.user_id == configured_user_id,
            PlatformRoleAssignmentRecord.role == PlatformRole.platform_admin,
            PlatformRoleAssignmentRecord.is_active == True,  # noqa: E712
        )
    ).first()
    return assignment_exists is not None


def backfill_platform_runtime_governance(session: Session) -> None:
    cache_key = _runtime_governance_cache_key(session)
    with _RUNTIME_GOVERNANCE_CACHE_LOCK:
        if cache_key in _RUNTIME_GOVERNANCE_CACHE:
            return

    if _runtime_governance_seed_present(session):
        with _RUNTIME_GOVERNANCE_CACHE_LOCK:
            _RUNTIME_GOVERNANCE_CACHE.add(cache_key)
        return

    seed_platform_runtime_governance(session, seed_defaults=False)
    sync_configured_platform_admin(session)
    with _RUNTIME_GOVERNANCE_CACHE_LOCK:
        _RUNTIME_GOVERNANCE_CACHE.add(cache_key)
