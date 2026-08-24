from app.core.config import (
    Settings,
    get_settings,
    knowledge_repo_autosync_enabled,
    runtime_bootstrap_enabled,
    runtime_legacy_file_fallback_enabled,
    runtime_legacy_file_write_through_enabled,
    should_auto_create_schema,
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


def test_local_only_defaults_for_schema_and_repo_autosync() -> None:
    settings = get_settings()
    original_database_url = settings.database_url
    original_schema_mode = settings.schema_management_mode
    original_autosync_override = settings.knowledge_repo_autosync_enabled
    original_runtime_bootstrap_override = settings.runtime_bootstrap_enabled

    try:
        settings.schema_management_mode = "create_all_local"
        settings.knowledge_repo_autosync_enabled = None
        settings.runtime_bootstrap_enabled = None

        settings.database_url = "sqlite:///tmp.db"
        assert should_auto_create_schema(settings) is True
        assert knowledge_repo_autosync_enabled(settings) is True
        assert runtime_bootstrap_enabled(settings) is True

        settings.database_url = "postgresql+psycopg://user:secret@127.0.0.1:5432/LAB"
        assert should_auto_create_schema(settings) is True
        assert knowledge_repo_autosync_enabled(settings) is True
        assert runtime_bootstrap_enabled(settings) is True

        settings.database_url = "postgresql+psycopg://user:secret@aws-0-us-east-2.pooler.supabase.com:5432/postgres"
        assert should_auto_create_schema(settings) is False
        assert knowledge_repo_autosync_enabled(settings) is False
        assert runtime_bootstrap_enabled(settings) is False
    finally:
        settings.database_url = original_database_url
        settings.schema_management_mode = original_schema_mode
        settings.knowledge_repo_autosync_enabled = original_autosync_override
        settings.runtime_bootstrap_enabled = original_runtime_bootstrap_override
