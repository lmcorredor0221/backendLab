from app.core.config import (
    Settings,
    runtime_legacy_file_fallback_enabled,
    runtime_legacy_file_write_through_enabled,
)


def test_default_cors_origins_cover_localhost_and_loopback() -> None:
    settings = Settings()

    assert "http://localhost:3200" in settings.cors_origins
    assert "http://127.0.0.1:3200" in settings.cors_origins
    assert settings.deepseek_base_url == "https://api.deepseek.com"
    assert settings.deepseek_model_fast == "deepseek-v4-flash"
    assert settings.deepseek_model_reasoning == "deepseek-v4-pro"
    assert settings.codex_exec_timeout_ms >= 1000
    assert settings.codex_exec_max_concurrency >= 1
    assert settings.agent_execution_backend == "provider_native"
    assert settings.knowledge_access_backend == "workspace_staged"
    assert runtime_legacy_file_fallback_enabled(settings) is True
    assert runtime_legacy_file_write_through_enabled(settings) is False
