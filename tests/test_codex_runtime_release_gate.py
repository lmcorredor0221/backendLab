from __future__ import annotations

from app.models import LLMRuntimeSettings
from app.services.llm_runtime.release_gate import detect_rollout_stage, evaluate_release_gate


def _runtime_settings(**overrides) -> LLMRuntimeSettings:
    payload = {
        "active_provider": "openai",
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
            "reasoning_effort": "high",
        },
        "codex_local": {
            "command": "codex",
            "model": "gpt-5.5",
            "profile": "",
            "cost_policy": "hybrid",
            "timeout_ms": 150000,
            "max_concurrency": 1,
            "runner_id": "local",
            "auth_mode": "auto",
            "fallback_models": ["gpt-5.5-mini"],
            "primary_agents": [],
            "shadow_agents": [],
            "staged_agents": [],
        },
        "compatibility_mode": "backward_compatible",
    }
    payload.update(overrides)
    return LLMRuntimeSettings.model_validate(payload)


def test_detect_rollout_stage_prefers_primary_when_backend_or_provider_promotes_codex() -> None:
    assert (
        detect_rollout_stage(
            _runtime_settings(
                agent_execution_backend="codex_cli",
                codex_local={
                    "command": "codex",
                    "model": "gpt-5.5",
                    "profile": "",
                    "cost_policy": "hybrid",
                    "timeout_ms": 150000,
                    "max_concurrency": 1,
                    "runner_id": "local",
                    "auth_mode": "auto",
                    "fallback_models": [],
                    "primary_agents": ["normalize_discovery"],
                    "shadow_agents": [],
                    "staged_agents": [],
                },
            )
        )
        == "primary"
    )
    assert (
        detect_rollout_stage(
            _runtime_settings(
                agent_execution_backend="shadow_codex_cli",
                codex_local={
                    "command": "codex",
                    "model": "gpt-5.5",
                    "profile": "",
                    "cost_policy": "hybrid",
                    "timeout_ms": 150000,
                    "max_concurrency": 1,
                    "runner_id": "local",
                    "auth_mode": "auto",
                    "fallback_models": [],
                    "primary_agents": [],
                    "shadow_agents": ["build_canvas"],
                    "staged_agents": [],
                },
            )
        )
        == "shadow"
    )


def test_release_gate_marks_disabled_stage_active_when_runtime_is_hardened() -> None:
    settings = _runtime_settings()
    runtime_status = {
        "smoke_ready": True,
        "last_known_result": {
            "status": "succeeded",
        },
    }
    checks = {
        "migration": {"ok": True},
        "smoke": {"ok": True},
        "queue": {"ok": True},
        "fallback": {"ok": True},
        "timeout": {"ok": True},
        "walkthrough": {"ok": True},
    }

    result = evaluate_release_gate(settings, runtime_status, checks)

    assert result["current_stage"] == "disabled"
    assert result["overall_ok"] is True
    assert result["transitions"][0]["status"] == "ready"
    assert result["transitions"][1]["status"] == "blocked"
    assert result["stages"][0]["status"] == "active"
