from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.core.config import get_settings
from app.models import (
    LLMRuntimeSettingsUpdateRequest,
    RuntimeGovernanceScopeType,
    RuntimeSettingsAuditRecord,
    UserRecord,
    WorkspaceRecord,
)
from app.services.auth_service import hash_password
from app.services.llm_runtime.runtime_settings_service import (
    load_effective_runtime_settings,
    load_llm_execution_runtime_settings,
    load_platform_runtime_defaults,
    persist_platform_runtime_defaults,
    persist_workspace_runtime_settings,
)


def _build_engine():
    return create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def _seed_user_and_workspace(session: Session, *, email: str, workspace_name: str) -> tuple[UserRecord, WorkspaceRecord]:
    user = UserRecord(
        email=email,
        full_name=workspace_name,
        password_hash=hash_password("LeanBuilder123!"),
    )
    session.add(user)
    session.commit()
    session.refresh(user)

    workspace = WorkspaceRecord(
        name=workspace_name,
        slug=workspace_name.lower().replace(" ", "-"),
        created_by_user_id=user.id,
    )
    session.add(workspace)
    session.commit()
    session.refresh(workspace)
    return user, workspace


def test_load_effective_runtime_settings_uses_platform_defaults_without_workspace_override() -> None:
    legacy_payload = {
        "active_provider": "openai",
        "agent_execution_backend": "provider_native",
        "knowledge_access_backend": "workspace_staged",
        "openai": {
            "fast_model": "gpt-5.4-mini",
            "reasoning_model": "gpt-5.5",
            "reasoning_effort": "low",
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com",
            "fast_model": "deepseek-v4-flash",
            "reasoning_model": "deepseek-v4-pro",
            "reasoning_effort": "high",
        },
        "codex_local": {
            "command": "codex",
            "model": "gpt-5.5",
            "profile": "default-profile",
            "cost_policy": "hybrid",
            "timeout_ms": 150000,
            "max_concurrency": 1,
            "runner_id": "local",
            "auth_mode": "auto",
            "fallback_models": [],
            "primary_agents": [],
            "shadow_agents": [],
            "staged_agents": [],
        },
    }
    settings = get_settings()
    original_path = settings.llm_config_path

    with TemporaryDirectory(prefix="lean-builder-runtime-service-") as runtime_dir:
        runtime_path = Path(runtime_dir) / "llm_settings.json"
        runtime_path.write_text(json.dumps(legacy_payload, ensure_ascii=True, indent=2), encoding="utf-8")
        settings.llm_config_path = runtime_path
        try:
            engine = _build_engine()
            SQLModel.metadata.create_all(engine)
            with Session(engine) as session:
                _, workspace = _seed_user_and_workspace(
                    session,
                    email="workspace-a@leanbuilder.local",
                    workspace_name="Workspace A",
                )
                platform_runtime = load_platform_runtime_defaults(session)
                workspace_runtime = load_effective_runtime_settings(session, workspace.id)
        finally:
            settings.llm_config_path = original_path

    assert platform_runtime.active_provider == workspace_runtime.active_provider
    assert platform_runtime.agent_execution_backend == workspace_runtime.agent_execution_backend
    assert platform_runtime.knowledge_access_backend == workspace_runtime.knowledge_access_backend
    assert platform_runtime.deepseek.fast_model == workspace_runtime.deepseek.fast_model


def test_persist_workspace_runtime_settings_isolated_by_workspace_and_audited() -> None:
    settings = get_settings()
    original_path = settings.llm_config_path

    with TemporaryDirectory(prefix="lean-builder-runtime-service-") as runtime_dir:
        runtime_path = Path(runtime_dir) / "llm_settings.json"
        runtime_path.write_text(json.dumps({"active_provider": "openai"}, ensure_ascii=True), encoding="utf-8")
        settings.llm_config_path = runtime_path
        try:
            engine = _build_engine()
            SQLModel.metadata.create_all(engine)
            with Session(engine) as session:
                actor, workspace_a = _seed_user_and_workspace(
                    session,
                    email="workspace-admin@leanbuilder.local",
                    workspace_name="Workspace A",
                )
                _, workspace_b = _seed_user_and_workspace(
                    session,
                    email="workspace-editor@leanbuilder.local",
                    workspace_name="Workspace B",
                )

                updated_runtime = persist_workspace_runtime_settings(
                    session,
                    workspace_a.id,
                    payload=LLMRuntimeSettingsUpdateRequest.model_validate(
                        {
                            "active_provider": "deepseek",
                            "agent_execution_backend": "codex_cli",
                            "knowledge_access_backend": "workspace_staged",
                            "openai": {
                                "fast_model": "gpt-5.4-mini",
                                "reasoning_model": "gpt-5.5",
                                "reasoning_effort": "low",
                            },
                            "deepseek": {
                                "base_url": "https://api.deepseek.com",
                                "fast_model": "deepseek-v4-flash",
                                "reasoning_model": "deepseek-v4-pro",
                                "reasoning_effort": "max",
                            },
                            "codex_local": {
                                "command": "codex",
                                "model": "gpt-5.5",
                                "profile": "workspace-a-profile",
                                "cost_policy": "hybrid",
                                "timeout_ms": 180000,
                                "max_concurrency": 2,
                                "runner_id": "workspace-a",
                                "auth_mode": "auto",
                                "fallback_models": ["gpt-5.5-mini"],
                                "primary_agents": ["normalize_discovery"],
                                "shadow_agents": ["synthesize_blueprint_narrative"],
                                "staged_agents": ["evaluate_readiness"],
                            },
                        }
                    ),
                    actor_user_id=actor.id,
                    mirror_legacy_runtime=False,
                )
                untouched_runtime = load_effective_runtime_settings(session, workspace_b.id)
                audit_rows = session.exec(
                    select(RuntimeSettingsAuditRecord).where(
                        RuntimeSettingsAuditRecord.scope_type == RuntimeGovernanceScopeType.workspace,
                        RuntimeSettingsAuditRecord.scope_id == str(workspace_a.id),
                    )
                ).all()
        finally:
            settings.llm_config_path = original_path

    assert updated_runtime.active_provider.value == "deepseek"
    assert updated_runtime.agent_execution_backend.value == "codex_cli"
    assert updated_runtime.deepseek.reasoning_effort == "max"
    assert updated_runtime.codex_local.profile == "workspace-a-profile"
    assert untouched_runtime.active_provider.value == "openai"
    assert len(audit_rows) == 1
    assert audit_rows[0].actor_email == "workspace-admin@leanbuilder.local"
    assert audit_rows[0].after_payload_redacted["active_provider"] == "deepseek"


def test_load_llm_execution_runtime_settings_follows_workspace_override() -> None:
    settings = get_settings()
    original_path = settings.llm_config_path

    with TemporaryDirectory(prefix="lean-builder-runtime-service-") as runtime_dir:
        runtime_path = Path(runtime_dir) / "llm_settings.json"
        runtime_path.write_text(json.dumps({"active_provider": "openai"}, ensure_ascii=True), encoding="utf-8")
        settings.llm_config_path = runtime_path
        try:
            engine = _build_engine()
            SQLModel.metadata.create_all(engine)
            with Session(engine) as session:
                actor, workspace = _seed_user_and_workspace(
                    session,
                    email="platform-admin@leanbuilder.local",
                    workspace_name="Platform Workspace",
                )

                persist_platform_runtime_defaults(
                    session,
                    payload=LLMRuntimeSettingsUpdateRequest.model_validate(
                        {
                            "active_provider": "deepseek",
                            "agent_execution_backend": "provider_native",
                            "knowledge_access_backend": "inline_context",
                            "openai": {
                                "fast_model": "gpt-5.4-mini",
                                "reasoning_model": "gpt-5.5",
                                "reasoning_effort": "low",
                            },
                            "deepseek": {
                                "base_url": "https://api.deepseek.com",
                                "fast_model": "deepseek-v4-flash",
                                "reasoning_model": "deepseek-v4-pro",
                                "reasoning_effort": "max",
                            },
                            "codex_local": {
                                "command": "codex",
                                "model": "gpt-5.5",
                                "profile": "platform-profile",
                                "cost_policy": "hybrid",
                                "timeout_ms": 180000,
                                "max_concurrency": 2,
                                "runner_id": "platform",
                                "auth_mode": "auto",
                                "fallback_models": ["gpt-5.5-mini"],
                                "primary_agents": ["normalize_discovery"],
                                "shadow_agents": ["synthesize_blueprint_narrative"],
                                "staged_agents": ["evaluate_readiness"],
                            },
                        }
                    ),
                    actor_user_id=actor.id,
                    mirror_legacy_runtime=False,
                )
                persist_workspace_runtime_settings(
                    session,
                    workspace.id,
                    payload=LLMRuntimeSettingsUpdateRequest.model_validate(
                        {
                            "active_provider": "openai",
                            "uses_platform_credentials": False,
                            "agent_execution_backend": "provider_native",
                            "knowledge_access_backend": "workspace_staged",
                            "openai": {
                                "fast_model": "gpt-5.4-mini",
                                "reasoning_model": "gpt-5.5",
                                "reasoning_effort": "low",
                            },
                            "deepseek": {
                                "base_url": "https://api.deepseek.com",
                                "fast_model": "deepseek-v4-flash",
                                "reasoning_model": "deepseek-v4-pro",
                                "reasoning_effort": "max",
                            },
                            "codex_local": {
                                "command": "codex",
                                "model": "gpt-5.5",
                                "profile": "workspace-profile",
                                "cost_policy": "hybrid",
                                "timeout_ms": 180000,
                                "max_concurrency": 2,
                                "runner_id": "workspace",
                                "auth_mode": "auto",
                                "fallback_models": ["gpt-5.5-mini"],
                                "primary_agents": ["normalize_discovery"],
                                "shadow_agents": ["synthesize_blueprint_narrative"],
                                "staged_agents": ["evaluate_readiness"],
                            },
                        }
                    ),
                    actor_user_id=actor.id,
                    mirror_legacy_runtime=False,
                )
                workspace_runtime = load_effective_runtime_settings(session, workspace.id)
                execution_runtime = load_llm_execution_runtime_settings(session, workspace.id)
        finally:
            settings.llm_config_path = original_path

    assert workspace_runtime.active_provider.value == "openai"
    assert workspace_runtime.uses_platform_credentials is False
    assert execution_runtime.active_provider.value == "openai"
    assert execution_runtime.uses_platform_credentials is False
    assert execution_runtime.knowledge_access_backend.value == "workspace_staged"


def test_persist_platform_runtime_defaults_updates_workspace_fallbacks() -> None:
    settings = get_settings()
    original_path = settings.llm_config_path

    with TemporaryDirectory(prefix="lean-builder-runtime-service-") as runtime_dir:
        runtime_path = Path(runtime_dir) / "llm_settings.json"
        runtime_path.write_text(json.dumps({"active_provider": "openai"}, ensure_ascii=True), encoding="utf-8")
        settings.llm_config_path = runtime_path
        try:
            engine = _build_engine()
            SQLModel.metadata.create_all(engine)
            with Session(engine) as session:
                actor, workspace = _seed_user_and_workspace(
                    session,
                    email="platform-admin@leanbuilder.local",
                    workspace_name="Platform Workspace",
                )
                updated_defaults = persist_platform_runtime_defaults(
                    session,
                    payload=LLMRuntimeSettingsUpdateRequest.model_validate(
                        {
                            "active_provider": "codex_local",
                            "agent_execution_backend": "shadow_codex_cli",
                            "knowledge_access_backend": "hybrid",
                            "openai": {
                                "fast_model": "gpt-5.4-mini",
                                "reasoning_model": "gpt-5.5",
                                "reasoning_effort": "low",
                            },
                            "deepseek": {
                                "base_url": "https://api.deepseek.com",
                                "fast_model": "deepseek-v4-flash",
                                "reasoning_model": "deepseek-v4-pro",
                                "reasoning_effort": "high",
                            },
                            "codex_local": {
                                "command": "codex",
                                "model": "gpt-5.5",
                                "profile": "platform-default",
                                "cost_policy": "hybrid",
                                "timeout_ms": 210000,
                                "max_concurrency": 2,
                                "runner_id": "platform",
                                "auth_mode": "chatgpt_session",
                                "fallback_models": ["gpt-5.5-mini"],
                                "primary_agents": ["normalize_discovery"],
                                "shadow_agents": ["synthesize_blueprint_narrative"],
                                "staged_agents": ["evaluate_readiness"],
                            },
                        }
                    ),
                    actor_user_id=actor.id,
                    mirror_legacy_runtime=False,
                )
                workspace_runtime = load_effective_runtime_settings(session, workspace.id)
        finally:
            settings.llm_config_path = original_path

    assert updated_defaults.active_provider.value == "codex_local"
    assert updated_defaults.agent_execution_backend.value == "shadow_codex_cli"
    assert workspace_runtime.active_provider.value == "codex_local"
    assert workspace_runtime.codex_local.profile == "platform-default"
