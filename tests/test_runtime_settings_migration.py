from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.core.config import get_settings
from app.models import PlatformRuntimeDefaultsRecord, SchemaMigrationRecord, WorkspaceRecord
from app.services.llm_runtime.settings_migration import (
    MIGRATION_KEY_RT7,
    apply_runtime_llm_multitenant_migration,
    inspect_runtime_settings_migration,
)


def _build_engine():
    return create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def test_runtime_settings_migration_normalizes_legacy_payload_without_writing() -> None:
    legacy_payload = {
        "active_provider": "codex_local",
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
            "profile": "legacy-profile",
            "cost_policy": "hybrid",
        },
        "compatibility_mode": "backward_compatible",
        "updated_at": "2026-07-15T21:23:09.766295",
    }

    with TemporaryDirectory(prefix="codex-runtime-migration-") as runtime_dir:
        runtime_path = Path(runtime_dir) / "llm_settings.json"
        runtime_path.write_text(json.dumps(legacy_payload, ensure_ascii=True, indent=2), encoding="utf-8")

        inspection = inspect_runtime_settings_migration(runtime_path)

    normalized = inspection["normalized_payload"]
    assert inspection["changed"] is True
    assert normalized["active_provider"] == "codex_local"
    assert normalized["agent_execution_backend"] == "provider_native"
    assert normalized["knowledge_access_backend"] == "workspace_staged"
    assert normalized["codex_local"]["runner_id"] == "local"
    assert normalized["codex_local"]["auth_mode"] == "auto"
    assert normalized["codex_local"]["fallback_models"] == []
    assert normalized["updated_at"] == legacy_payload["updated_at"]


def test_runtime_llm_multitenant_migration_backfills_platform_defaults_and_records_schema_state() -> None:
    legacy_payload = {
        "active_provider": "deepseek",
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
            "profile": "bridge",
            "cost_policy": "hybrid",
            "timeout_ms": 120000,
            "max_concurrency": 2,
            "runner_id": "codex-dev",
            "auth_mode": "auto",
            "fallback_models": ["gpt-5.5-mini"],
            "primary_agents": [],
            "shadow_agents": [],
            "staged_agents": [],
        },
        "compatibility_mode": "backward_compatible",
        "updated_at": "2026-07-20T10:00:00",
    }
    settings = get_settings()
    original_path = settings.llm_config_path

    with TemporaryDirectory(prefix="lean-builder-runtime-rt7-") as runtime_dir:
        runtime_path = Path(runtime_dir) / "llm_settings.json"
        runtime_path.write_text(json.dumps(legacy_payload, ensure_ascii=True, indent=2), encoding="utf-8")
        settings.llm_config_path = runtime_path
        try:
            engine = _build_engine()
            SQLModel.metadata.create_all(engine)
            with Session(engine) as session:
                workspace = WorkspaceRecord(name="Workspace A", slug="workspace-a")
                session.add(workspace)
                session.commit()

                summary = apply_runtime_llm_multitenant_migration(session, config_path=runtime_path)
                defaults_row = session.exec(
                    select(PlatformRuntimeDefaultsRecord).where(PlatformRuntimeDefaultsRecord.is_active == True)  # noqa: E712
                ).first()
                migration_row = session.exec(
                    select(SchemaMigrationRecord).where(SchemaMigrationRecord.migration_key == MIGRATION_KEY_RT7)
                ).first()
        finally:
            settings.llm_config_path = original_path

    assert summary.platform_defaults_seeded is True
    assert summary.workspace_settings_seeded == 0
    assert defaults_row is not None
    assert defaults_row.active_provider_default.value == "deepseek"
    assert defaults_row.per_provider_defaults["codex_local"]["runner_id"] == "codex-dev"
    assert migration_row is not None


def test_runtime_llm_multitenant_migration_is_idempotent_after_first_run() -> None:
    settings = get_settings()
    original_path = settings.llm_config_path

    with TemporaryDirectory(prefix="lean-builder-runtime-rt7-idempotent-") as runtime_dir:
        runtime_path = Path(runtime_dir) / "llm_settings.json"
        runtime_path.write_text(json.dumps({"active_provider": "openai"}, ensure_ascii=True), encoding="utf-8")
        settings.llm_config_path = runtime_path
        try:
            engine = _build_engine()
            SQLModel.metadata.create_all(engine)
            with Session(engine) as session:
                first = apply_runtime_llm_multitenant_migration(session, config_path=runtime_path)
                second = apply_runtime_llm_multitenant_migration(session, config_path=runtime_path)
        finally:
            settings.llm_config_path = original_path

    assert first.already_recorded is False
    assert second.already_recorded is True
    assert second.platform_defaults_preexisting is True
