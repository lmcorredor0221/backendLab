from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from app.core.config import get_settings
from app.services.openai_builder import load_llm_runtime_settings


def test_load_runtime_settings_backfills_new_codex_fields_from_legacy_payload() -> None:
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

    settings = get_settings()
    original_path = settings.llm_config_path

    with TemporaryDirectory(prefix="lean-builder-runtime-settings-") as runtime_dir:
        runtime_path = Path(runtime_dir) / "llm_settings.json"
        runtime_path.write_text(json.dumps(legacy_payload, ensure_ascii=True, indent=2), encoding="utf-8")
        settings.llm_config_path = runtime_path
        try:
            resolved = load_llm_runtime_settings()
        finally:
            settings.llm_config_path = original_path

    assert resolved.active_provider.value == "codex_local"
    assert resolved.agent_execution_backend.value == "provider_native"
    assert resolved.knowledge_access_backend.value == "workspace_staged"
    assert resolved.codex_local.command == "codex"
    assert resolved.codex_local.model == "gpt-5.5"
    assert resolved.codex_local.profile == "legacy-profile"
    assert resolved.codex_local.timeout_ms >= 1000
    assert resolved.codex_local.max_concurrency >= 1
    assert resolved.codex_local.runner_id == "local"
    assert resolved.codex_local.auth_mode.value == "auto"
    assert resolved.codex_local.fallback_models == []


def test_load_runtime_settings_ignores_legacy_file_when_fallback_is_disabled() -> None:
    legacy_payload = {
        "active_provider": "deepseek",
        "openai": {
            "fast_model": "legacy-fast",
            "reasoning_model": "legacy-reasoning",
            "reasoning_effort": "medium",
        },
    }

    settings = get_settings()
    original_path = settings.llm_config_path
    original_fallback = settings.runtime_legacy_file_fallback_enabled

    with TemporaryDirectory(prefix="lean-builder-runtime-settings-no-fallback-") as runtime_dir:
        runtime_path = Path(runtime_dir) / "llm_settings.json"
        runtime_path.write_text(json.dumps(legacy_payload, ensure_ascii=True, indent=2), encoding="utf-8")
        settings.llm_config_path = runtime_path
        settings.runtime_legacy_file_fallback_enabled = False
        try:
            resolved = load_llm_runtime_settings()
        finally:
            settings.llm_config_path = original_path
            settings.runtime_legacy_file_fallback_enabled = original_fallback

    assert resolved.active_provider.value != "deepseek"
    assert resolved.openai.fast_model != "legacy-fast"
