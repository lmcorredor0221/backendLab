from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlmodel import Session, select

from app.models import (
    CodexLocalProviderConfigUpdate,
    AntigravityProviderConfigUpdate,
    DeepSeekProviderConfigUpdate,
    LLMRuntimeSettings,
    LLMRuntimeSettingsUpdateRequest,
    OpenAIProviderConfigUpdate,
    PlatformRuntimeDefaultsRecord,
    RuntimeGovernanceScopeType,
    RuntimeSettingsAuditRecord,
    SessionRecord,
    UserRecord,
    WorkspaceRuntimeSettingsRecord,
    utc_now,
)
from app.services.openai_builder import (
    load_llm_runtime_settings,
    persist_llm_runtime_settings,
    resolve_runtime_settings_payload,
)
from app.services.runtime_governance_bootstrap import backfill_platform_runtime_governance


PLATFORM_RUNTIME_SCOPE_ID = "platform-runtime-defaults"
RUNTIME_ORIGIN_DEFAULT = "default"
RUNTIME_ORIGIN_OVERRIDE = "override"


def _merge_dicts(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _effective_runtime_base_payload(runtime_settings: LLMRuntimeSettings) -> dict[str, Any]:
    return runtime_settings.model_dump(
        mode="json",
        exclude={
            "provider_options",
            "memory_rollout",
        },
    )


def _runtime_update_request_from_runtime(runtime_settings: LLMRuntimeSettings) -> LLMRuntimeSettingsUpdateRequest:
    return LLMRuntimeSettingsUpdateRequest(
        active_provider=runtime_settings.active_provider,
        agent_execution_backend=runtime_settings.agent_execution_backend,
        knowledge_access_backend=runtime_settings.knowledge_access_backend,
        uses_platform_credentials=runtime_settings.uses_platform_credentials,
        openai=OpenAIProviderConfigUpdate(
            fast_model=runtime_settings.openai.fast_model,
            reasoning_model=runtime_settings.openai.reasoning_model,
            reasoning_effort=runtime_settings.openai.reasoning_effort,
        ),
        deepseek=DeepSeekProviderConfigUpdate(
            base_url=runtime_settings.deepseek.base_url,
            fast_model=runtime_settings.deepseek.fast_model,
            reasoning_model=runtime_settings.deepseek.reasoning_model,
            reasoning_effort=runtime_settings.deepseek.reasoning_effort,
        ),
        codex_local=CodexLocalProviderConfigUpdate(
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
        ),
        antigravity=AntigravityProviderConfigUpdate(
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
        ),
    )


def _editable_runtime_payload(runtime_settings: LLMRuntimeSettings) -> dict[str, Any]:
    return _runtime_update_request_from_runtime(runtime_settings).model_dump(mode="json", exclude_none=True)


def _mark_runtime_field_origins(
    payload: dict[str, Any],
    *,
    origin: str,
    origins: dict[str, str],
    prefix: str = "",
) -> None:
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            _mark_runtime_field_origins(value, origin=origin, origins=origins, prefix=path)
            continue
        origins[path] = origin


def _build_runtime_field_origins(
    *,
    platform_runtime: LLMRuntimeSettings,
    workspace_record: WorkspaceRuntimeSettingsRecord | None = None,
) -> dict[str, str]:
    origins: dict[str, str] = {}
    _mark_runtime_field_origins(
        _editable_runtime_payload(platform_runtime),
        origin=RUNTIME_ORIGIN_DEFAULT,
        origins=origins,
    )
    if workspace_record is None:
        return origins

    for key in (
        "active_provider",
        "agent_execution_backend",
        "knowledge_access_backend",
        "uses_platform_credentials",
    ):
        origins[key] = RUNTIME_ORIGIN_OVERRIDE

    provider_overrides = workspace_record.provider_overrides if isinstance(workspace_record.provider_overrides, dict) else {}
    _mark_runtime_field_origins(
        provider_overrides,
        origin=RUNTIME_ORIGIN_OVERRIDE,
        origins=origins,
    )
    return origins


def _annotate_runtime_field_origins(
    runtime_settings: LLMRuntimeSettings,
    *,
    platform_runtime: LLMRuntimeSettings,
    workspace_record: WorkspaceRuntimeSettingsRecord | None = None,
) -> LLMRuntimeSettings:
    return runtime_settings.model_copy(
        update={
            "field_origins": _build_runtime_field_origins(
                platform_runtime=platform_runtime,
                workspace_record=workspace_record,
            )
        }
    )


def _provider_overrides_from_update(payload: LLMRuntimeSettingsUpdateRequest) -> dict[str, Any]:
    return {
        "openai": OpenAIProviderConfigUpdate(
            fast_model=payload.openai.fast_model,
            reasoning_model=payload.openai.reasoning_model,
            reasoning_effort=payload.openai.reasoning_effort,
        ).model_dump(mode="json"),
        "deepseek": DeepSeekProviderConfigUpdate(
            base_url=payload.deepseek.base_url,
            fast_model=payload.deepseek.fast_model,
            reasoning_model=payload.deepseek.reasoning_model,
            reasoning_effort=payload.deepseek.reasoning_effort,
        ).model_dump(mode="json"),
        "codex_local": CodexLocalProviderConfigUpdate(
            command=payload.codex_local.command,
            model=payload.codex_local.model,
            profile=payload.codex_local.profile,
            cost_policy=payload.codex_local.cost_policy,
            timeout_ms=payload.codex_local.timeout_ms,
            max_concurrency=payload.codex_local.max_concurrency,
            runner_id=payload.codex_local.runner_id,
            auth_mode=payload.codex_local.auth_mode,
            fallback_models=payload.codex_local.fallback_models,
            primary_agents=payload.codex_local.primary_agents,
            shadow_agents=payload.codex_local.shadow_agents,
            staged_agents=payload.codex_local.staged_agents,
        ).model_dump(mode="json"),
    }


def _resolve_actor_email(session: Session, actor_user_id: UUID | None) -> str:
    if actor_user_id is None:
        return ""
    actor = session.get(UserRecord, actor_user_id)
    if actor is None:
        return ""
    return actor.email


def _active_platform_defaults_record(session: Session) -> PlatformRuntimeDefaultsRecord | None:
    return session.exec(
        select(PlatformRuntimeDefaultsRecord)
        .where(PlatformRuntimeDefaultsRecord.is_active == True)  # noqa: E712
        .order_by(PlatformRuntimeDefaultsRecord.version.desc())
    ).first()


def _active_workspace_runtime_record(session: Session, workspace_id: UUID) -> WorkspaceRuntimeSettingsRecord | None:
    return session.exec(
        select(WorkspaceRuntimeSettingsRecord)
        .where(
            WorkspaceRuntimeSettingsRecord.workspace_id == workspace_id,
            WorkspaceRuntimeSettingsRecord.is_active == True,  # noqa: E712
        )
        .order_by(WorkspaceRuntimeSettingsRecord.version.desc())
    ).first()


def _build_platform_runtime_payload(
    *,
    bootstrap_runtime: LLMRuntimeSettings,
    defaults_record: PlatformRuntimeDefaultsRecord,
) -> dict[str, Any]:
    payload = _effective_runtime_base_payload(bootstrap_runtime)
    payload["active_provider"] = defaults_record.active_provider_default.value
    payload["agent_execution_backend"] = defaults_record.agent_execution_backend_default.value
    payload["knowledge_access_backend"] = defaults_record.knowledge_access_backend_default.value
    payload["uses_platform_credentials"] = True
    payload["updated_at"] = defaults_record.updated_at.isoformat()
    provider_defaults = defaults_record.per_provider_defaults if isinstance(defaults_record.per_provider_defaults, dict) else {}
    return _merge_dicts(payload, provider_defaults)


def _build_workspace_runtime_payload(
    *,
    platform_runtime: LLMRuntimeSettings,
    workspace_record: WorkspaceRuntimeSettingsRecord,
) -> dict[str, Any]:
    payload = _effective_runtime_base_payload(platform_runtime)
    payload["active_provider"] = workspace_record.active_provider.value
    payload["agent_execution_backend"] = workspace_record.agent_execution_backend.value
    payload["knowledge_access_backend"] = workspace_record.knowledge_access_backend.value
    payload["uses_platform_credentials"] = workspace_record.uses_platform_credentials
    payload["updated_at"] = workspace_record.updated_at.isoformat()
    provider_overrides = workspace_record.provider_overrides if isinstance(workspace_record.provider_overrides, dict) else {}
    return _merge_dicts(payload, provider_overrides)


def _record_runtime_audit(
    session: Session,
    *,
    scope_type: RuntimeGovernanceScopeType,
    scope_id: str,
    change_type: str,
    before_payload: dict[str, Any],
    after_payload: dict[str, Any],
    actor_user_id: UUID | None,
) -> None:
    session.add(
        RuntimeSettingsAuditRecord(
            scope_type=scope_type,
            scope_id=scope_id,
            change_type=change_type,
            before_payload_redacted=before_payload,
            after_payload_redacted=after_payload,
            actor_user_id=actor_user_id,
            actor_email=_resolve_actor_email(session, actor_user_id),
        )
    )


def build_redacted_runtime_view(runtime_settings: LLMRuntimeSettings) -> LLMRuntimeSettings:
    redacted_payload = runtime_settings.model_dump(mode="json")
    return LLMRuntimeSettings.model_validate(redacted_payload)


def load_platform_runtime_defaults(session: Session) -> LLMRuntimeSettings:
    backfill_platform_runtime_governance(session)
    defaults_record = _active_platform_defaults_record(session)
    bootstrap_runtime = load_llm_runtime_settings()
    if defaults_record is None:
        platform_runtime = build_redacted_runtime_view(bootstrap_runtime)
        return _annotate_runtime_field_origins(
            platform_runtime,
            platform_runtime=platform_runtime,
        )
    payload = _build_platform_runtime_payload(
        bootstrap_runtime=bootstrap_runtime,
        defaults_record=defaults_record,
    )
    platform_runtime = build_redacted_runtime_view(resolve_runtime_settings_payload(payload))
    return _annotate_runtime_field_origins(
        platform_runtime,
        platform_runtime=platform_runtime,
    )


def load_workspace_runtime_settings(session: Session, workspace_id: UUID) -> WorkspaceRuntimeSettingsRecord | None:
    backfill_platform_runtime_governance(session)
    return _active_workspace_runtime_record(session, workspace_id)


def _load_effective_runtime_settings_base(session: Session, workspace_id: UUID) -> LLMRuntimeSettings:
    platform_runtime = load_platform_runtime_defaults(session)
    workspace_runtime = load_workspace_runtime_settings(session, workspace_id)
    if workspace_runtime is None:
        effective_runtime = build_redacted_runtime_view(platform_runtime.model_copy(update={"uses_platform_credentials": True}))
        return _annotate_runtime_field_origins(
            effective_runtime,
            platform_runtime=platform_runtime,
        )
    payload = _build_workspace_runtime_payload(
        platform_runtime=platform_runtime,
        workspace_record=workspace_runtime,
    )
    effective_runtime = build_redacted_runtime_view(resolve_runtime_settings_payload(payload))
    return _annotate_runtime_field_origins(
        effective_runtime,
        platform_runtime=platform_runtime,
        workspace_record=workspace_runtime,
    )


def load_effective_runtime_settings(
    session: Session,
    workspace_id: UUID,
    *,
    annotate_workspace_secrets: bool = True,
) -> LLMRuntimeSettings:
    effective_runtime = _load_effective_runtime_settings_base(session, workspace_id)
    if not annotate_workspace_secrets:
        return effective_runtime

    from app.services.llm_runtime.runtime_secrets_service import annotate_runtime_settings_with_workspace_secrets

    return annotate_runtime_settings_with_workspace_secrets(
        session,
        workspace_id,
        effective_runtime,
    )


def load_llm_execution_runtime_settings(session: Session, workspace_id: UUID | None = None) -> LLMRuntimeSettings:
    if workspace_id is None:
        return load_platform_runtime_defaults(session)
    return load_effective_runtime_settings(session, workspace_id)


def load_effective_runtime_settings_for_session(session: Session, session_id: UUID) -> LLMRuntimeSettings:
    record = session.get(SessionRecord, session_id)
    if record is None:
        return load_platform_runtime_defaults(session)
    return load_effective_runtime_settings(session, record.workspace_id)


def persist_platform_runtime_defaults(
    session: Session,
    payload: LLMRuntimeSettingsUpdateRequest,
    *,
    actor_user_id: UUID | None = None,
    mirror_legacy_runtime: bool = False,
) -> LLMRuntimeSettings:
    backfill_platform_runtime_governance(session)
    before_runtime = load_platform_runtime_defaults(session)
    active_record = _active_platform_defaults_record(session)
    next_version = 1 if active_record is None else active_record.version + 1
    if active_record is not None:
        active_record.is_active = False
        active_record.updated_at = utc_now()
        session.add(active_record)

    now = utc_now()
    session.add(
        PlatformRuntimeDefaultsRecord(
            active_provider_default=payload.active_provider,
            agent_execution_backend_default=payload.agent_execution_backend,
            knowledge_access_backend_default=payload.knowledge_access_backend,
            per_provider_defaults=_provider_overrides_from_update(payload),
            is_active=True,
            version=next_version,
            updated_by_user_id=actor_user_id,
            created_at=now,
            updated_at=now,
        )
    )
    session.flush()

    after_runtime = load_platform_runtime_defaults(session)
    _record_runtime_audit(
        session,
        scope_type=RuntimeGovernanceScopeType.platform,
        scope_id=PLATFORM_RUNTIME_SCOPE_ID,
        change_type="platform_runtime_defaults_updated",
        before_payload=before_runtime.model_dump(mode="json"),
        after_payload=after_runtime.model_dump(mode="json"),
        actor_user_id=actor_user_id,
    )
    session.commit()
    if mirror_legacy_runtime:
        persist_llm_runtime_settings(payload)
    return after_runtime


def persist_workspace_runtime_settings(
    session: Session,
    workspace_id: UUID,
    payload: LLMRuntimeSettingsUpdateRequest,
    *,
    actor_user_id: UUID | None = None,
    mirror_legacy_runtime: bool = False,
) -> LLMRuntimeSettings:
    backfill_platform_runtime_governance(session)
    before_runtime = load_effective_runtime_settings(session, workspace_id)
    active_record = _active_workspace_runtime_record(session, workspace_id)
    next_version = 1 if active_record is None else active_record.version + 1
    uses_platform_credentials = True if active_record is None else active_record.uses_platform_credentials
    if payload.uses_platform_credentials is not None:
        uses_platform_credentials = payload.uses_platform_credentials
    if active_record is not None:
        active_record.is_active = False
        active_record.updated_at = utc_now()
        session.add(active_record)

    now = utc_now()
    session.add(
        WorkspaceRuntimeSettingsRecord(
            workspace_id=workspace_id,
            active_provider=payload.active_provider,
            agent_execution_backend=payload.agent_execution_backend,
            knowledge_access_backend=payload.knowledge_access_backend,
            provider_overrides=_provider_overrides_from_update(payload),
            uses_platform_credentials=uses_platform_credentials,
            is_active=True,
            version=next_version,
            updated_by_user_id=actor_user_id,
            created_at=now,
            updated_at=now,
        )
    )
    session.flush()

    after_runtime = load_effective_runtime_settings(session, workspace_id)
    _record_runtime_audit(
        session,
        scope_type=RuntimeGovernanceScopeType.workspace,
        scope_id=str(workspace_id),
        change_type="workspace_runtime_settings_updated",
        before_payload=before_runtime.model_dump(mode="json"),
        after_payload=after_runtime.model_dump(mode="json"),
        actor_user_id=actor_user_id,
    )
    session.commit()
    if mirror_legacy_runtime:
        persist_llm_runtime_settings(payload)
    return after_runtime


def persist_workspace_runtime_credential_mode(
    session: Session,
    workspace_id: UUID,
    *,
    uses_platform_credentials: bool,
    actor_user_id: UUID | None = None,
    mirror_legacy_runtime: bool = False,
) -> LLMRuntimeSettings:
    current_runtime = load_effective_runtime_settings(session, workspace_id)
    payload = _runtime_update_request_from_runtime(current_runtime).model_copy(
        update={"uses_platform_credentials": uses_platform_credentials}
    )
    return persist_workspace_runtime_settings(
        session,
        workspace_id,
        payload,
        actor_user_id=actor_user_id,
        mirror_legacy_runtime=mirror_legacy_runtime,
    )


def reset_workspace_runtime_settings(
    session: Session,
    workspace_id: UUID,
    *,
    actor_user_id: UUID | None = None,
    mirror_legacy_runtime: bool = False,
) -> LLMRuntimeSettings:
    backfill_platform_runtime_governance(session)
    active_record = _active_workspace_runtime_record(session, workspace_id)
    current_runtime = load_effective_runtime_settings(session, workspace_id)
    platform_runtime = load_platform_runtime_defaults(session)

    if active_record is None:
        return build_redacted_runtime_view(platform_runtime)

    active_record.is_active = False
    active_record.updated_at = utc_now()
    session.add(active_record)
    session.flush()

    reset_runtime = load_effective_runtime_settings(session, workspace_id)
    _record_runtime_audit(
        session,
        scope_type=RuntimeGovernanceScopeType.workspace,
        scope_id=str(workspace_id),
        change_type="workspace_runtime_settings_reset_to_platform_defaults",
        before_payload=current_runtime.model_dump(mode="json"),
        after_payload=reset_runtime.model_dump(mode="json"),
        actor_user_id=actor_user_id,
    )
    session.commit()

    if mirror_legacy_runtime:
        persist_llm_runtime_settings(_runtime_update_request_from_runtime(platform_runtime))

    return reset_runtime
