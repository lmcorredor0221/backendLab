from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from app.models import (
    AntigravityProviderConfig,
    LLMProviderKey,
    LLMRuntimeSettings,
)
from app.services.llm_runtime.antigravity_cli.execution_service import (
    AgyExecutionService,
    resolve_agy_executable,
)
from app.services.llm_runtime.antigravity_cli.fallback_policy import (
    AgyFallbackPolicy,
)
from app.services.llm_runtime.antigravity_cli.provider_facade import (
    AntigravityLocalBuilderService,
)
from app.services.llm_runtime.antigravity_cli.runtime_types import (
    AgyRuntimeErrorCode,
)
from app.services.llm_runtime.antigravity_cli.workspace_builder import (
    AgyRunWorkspace,
)


def test_resolve_executable_from_env_var(tmp_path):
    fake_exe = tmp_path / "fake_agy.exe"
    fake_exe.write_text("binary", encoding="utf-8")
    with patch.dict(os.environ, {"ANTIGRAVITY_EXECUTABLE": str(fake_exe)}):
        result = resolve_agy_executable()
        assert result == str(fake_exe)


def test_resolve_executable_from_path():
    with patch("shutil.which", return_value="/usr/bin/agy"):
        result = resolve_agy_executable()
        assert result == "/usr/bin/agy"


def test_resolve_executable_fallback():
    with patch.dict(os.environ, {"ANTIGRAVITY_EXECUTABLE": ""}, clear=True), \
         patch("shutil.which", return_value=None):
        result = resolve_agy_executable(configured="agy")
        assert result is None or result == "agy"


def test_build_execution_args_with_model(tmp_path):
    settings = LLMRuntimeSettings(
        antigravity=AntigravityProviderConfig(
            executable="agy",
            model="gemini-3.6-flash",
            effort="high",
        )
    )
    svc = AgyExecutionService(settings)
    ws = AgyRunWorkspace(
        run_id="test-123",
        task_kind="test_kind",
        root_dir=tmp_path,
        prompt_path=tmp_path / "prompt.md",
        output_path=tmp_path / "output.md",
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
    )
    args = svc.build_execution_args(workspace=ws, model="gemini-3.6-pro", enable_web_search=True)
    assert "--print" in args
    assert str(tmp_path) in args
    assert "--dangerously-skip-permissions" in args
    assert "--model" in args
    assert "gemini-3.6-pro" in args
    assert "--effort" in args
    assert "high" in args


def test_build_execution_args_without_model(tmp_path):
    settings = LLMRuntimeSettings(
        antigravity=AntigravityProviderConfig(
            executable="agy",
            model="",
        )
    )
    svc = AgyExecutionService(settings)
    ws = AgyRunWorkspace(
        run_id="test-123",
        task_kind="test_kind",
        root_dir=tmp_path,
        prompt_path=tmp_path / "prompt.md",
        output_path=tmp_path / "output.md",
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
    )
    args = svc.build_execution_args(workspace=ws)
    assert "--model" not in args


def test_fallback_policy_capacity_error():
    policy = AgyFallbackPolicy(model="gemini-3.6-flash", fallback_models=["gemini-3.6-pro"])
    decision = policy.classify_failure(stdout="", stderr="Resource exhausted: rate limit reached for model")
    assert decision.retryable is True
    assert decision.error_code == AgyRuntimeErrorCode.quota_exceeded


def test_fallback_policy_auth_error():
    policy = AgyFallbackPolicy(model="gemini-3.6-flash", fallback_models=[])
    decision = policy.classify_failure(stdout="", stderr="Unauthenticated: invalid api key provided")
    assert decision.retryable is False
    assert decision.error_code == AgyRuntimeErrorCode.auth_error


def test_fallback_policy_generic_error():
    policy = AgyFallbackPolicy(model="gemini-3.6-flash", fallback_models=[])
    decision = policy.classify_failure(stdout="", stderr="Some unhandled internal exception")
    assert decision.retryable is False
    assert decision.error_code == AgyRuntimeErrorCode.execution_failed


def test_resolve_auth_mode_from_env():
    settings = LLMRuntimeSettings()
    svc = AgyExecutionService(settings)
    with patch.dict(os.environ, {"ANTIGRAVITY_API_KEY": "test-key"}):
        mode, is_avail = svc.resolve_auth_mode()
        assert mode in {"auto", "api_key"}
        assert is_avail is True


def test_resolve_auth_mode_from_credentials_file(tmp_path):
    settings = LLMRuntimeSettings()
    svc = AgyExecutionService(settings)
    with patch.dict(os.environ, {"ANTIGRAVITY_API_KEY": ""}), \
         patch.object(svc, "resolve_agy_home", return_value=tmp_path):
        creds = tmp_path / "credentials.json"
        creds.write_text('{"token": "xyz"}', encoding="utf-8")
        mode, is_avail = svc.resolve_auth_mode()
        assert mode == "session"
        assert is_avail is True


def test_get_runtime_status_no_executable():
    settings = LLMRuntimeSettings()
    svc = AgyExecutionService(settings)
    with patch("app.services.llm_runtime.antigravity_cli.execution_service.resolve_agy_executable", return_value=None):
        status = svc.get_runtime_status()
        assert status["provider"] == "antigravity_cli"
        assert status["smoke_ready"] is False
        assert len(status["smoke_blocking_reasons"]) > 0


def test_build_attempt_sequence_with_fallbacks():
    policy = AgyFallbackPolicy(model="gemini-3.6-flash", fallback_models=["gemini-3.6-pro", "gemini-3.6-flash"])
    seq = policy.build_attempt_sequence(primary_model="gemini-3.6-flash")
    assert seq == ["gemini-3.6-flash", "gemini-3.6-pro"]


def test_provider_facade_unavailable():
    settings = LLMRuntimeSettings(
        active_provider=LLMProviderKey.antigravity_cli,
        antigravity=AntigravityProviderConfig(executable_found=False),
    )
    facade = AntigravityLocalBuilderService(settings)
    assert facade.can_attempt() is True
    assert facade.is_available() is False

    from app.models import DiscoveryInput
    res = facade.normalize_discovery(DiscoveryInput())
    assert res.artifact is None
