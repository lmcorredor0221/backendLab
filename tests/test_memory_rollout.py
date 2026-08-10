from __future__ import annotations

import json
from pathlib import Path

from app.models import LLMRuntimeSettings
from app.services.memory_rollout import (
    build_memory_rollout_summary,
    expected_monitoring_stages,
    resolve_effective_stage_backend,
)


def _runtime_settings(knowledge_access_backend: str = "workspace_staged") -> LLMRuntimeSettings:
    return LLMRuntimeSettings.model_validate(
        {
            "active_provider": "openai",
            "agent_execution_backend": "provider_native",
            "knowledge_access_backend": knowledge_access_backend,
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
                "fallback_models": [],
                "primary_agents": [],
                "shadow_agents": [],
                "staged_agents": [],
            },
            "compatibility_mode": "backward_compatible",
        }
    )


def test_memory_rollout_summary_promotes_workspace_staged_when_manifest_is_ready(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "knowledge-memory"
    runtime_root.mkdir(parents=True, exist_ok=True)
    (runtime_root / "knowledge-corpus-manifest.json").write_text(
        json.dumps({"corpus_hash": "abc123"}, ensure_ascii=True),
        encoding="utf-8",
    )

    summary = build_memory_rollout_summary(
        _runtime_settings(),
        runtime_root=runtime_root,
    )

    assert summary.status == "ready"
    assert summary.manifest_ready is True
    assert summary.requested_backend == "workspace_staged"
    assert summary.effective_default_backend == "workspace_staged"
    assert all(item.enabled for item in summary.phases)
    assert [item.stage_key for item in summary.stages] == [
        "define",
        "design",
        "tools",
        "memory",
        "evaluate",
        "build",
    ]
    assert all(item.effective_backend == "workspace_staged" for item in summary.stages)
    assert [key for key, _ in expected_monitoring_stages(summary)] == [
        "define",
        "design",
        "tools",
        "memory",
        "evaluate",
        "build",
    ]


def test_memory_rollout_backend_falls_back_to_inline_when_manifest_or_phase_is_missing() -> None:
    assert (
        resolve_effective_stage_backend(
            "workspace_staged",
            stage_key="define",
            manifest_ready=False,
        )
        == "inline_context"
    )
    assert (
        resolve_effective_stage_backend(
            "hybrid",
            stage_key="build",
            manifest_ready=True,
            extended_enabled=False,
        )
        == "inline_context"
    )
