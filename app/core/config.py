from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url


BACKEND_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    app_name: str = "Lean Agent Builder API"
    app_debug: bool = True
    api_v1_prefix: str = "/api/v1"
    database_url: str = "postgresql+psycopg://lean_builder:lean_builder@localhost:5432/lean_builder"
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3200", "http://127.0.0.1:3200"]
    )
    llm_provider: str = "openai"
    llm_mode: str = "hybrid"
    llm_config_path: Path = Path(__file__).resolve().parents[2] / "runtime" / "llm_settings.json"
    runtime_legacy_file_fallback_enabled: bool | None = None
    runtime_legacy_file_write_through_enabled: bool | None = None
    openai_api_key: str | None = None
    openai_model_fast: str = "gpt-5.4-mini"
    openai_model_reasoning: str = "gpt-5.5"
    openai_reasoning_effort: str = "low"
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model_fast: str = "deepseek-v4-flash"
    deepseek_model_reasoning: str = "deepseek-v4-pro"
    deepseek_reasoning_effort: str = "high"
    codex_cli_command: str = "codex"
    codex_model: str = "gpt-5.5"
    codex_profile: str = ""
    codex_executable: str = ""
    codex_exec_model: str = ""
    codex_exec_timeout_ms: int = 150000
    codex_exec_max_concurrency: int = 1
    codex_runner_id: str = "local"
    codex_auth_mode: str = "auto"
    codex_exec_fallback_models: str = ""
    agent_execution_backend: str = "provider_native"
    knowledge_access_backend: str = "workspace_staged"
    codex_primary_agents: str = ""
    codex_shadow_agents: str = ""
    codex_staged_agents: str = ""
    knowledge_docs_root: Path = Path(__file__).resolve().parents[3] / "Docs"
    knowledge_repo_autosync_enabled: bool | None = None
    runtime_secrets_master_key: str = ""
    local_admin_email: str = "lmcorredor@leanagentbuilder.com"
    local_admin_password: str = "LeanBuilder123!"
    local_admin_name: str = "Lean Builder Admin"
    auth_token_ttl_hours: int = 12
    allow_demo_tier_upgrade: bool | None = None
    commerce_public_base_url: str = "http://localhost:3200"
    commerce_checkout_provider: str = "sandbox"
    hotmart_enabled: bool = False
    hotmart_environment: str = "sandbox"
    hotmart_api_base_url: str = ""
    hotmart_auth_base_url: str = ""
    hotmart_webhook_public_url: str = ""
    hotmart_client_id: str = ""
    hotmart_client_secret: str = ""
    hotmart_basic_token: str = ""
    hotmart_hottok: str = ""
    hotmart_request_timeout_seconds: int = 30
    hotmart_max_retries: int = 3
    hotmart_sync_page_size: int = 50
    hotmart_payment_link_create_path: str = "/payments/api/v1/payment-links"
    hotmart_payment_link_list_path: str = "/payments/api/v1/payment-links"
    hotmart_coupon_create_path_template: str = "/products/api/v1/product/{product_id}/coupon"
    hotmart_coupon_list_path_template: str = "/products/api/v1/coupon/product/{product_id}"
    hotmart_coupon_delete_path_template: str = "/products/api/v1/coupon/{coupon_id}"
    hotmart_products_list_path: str = "/products/api/v1/products"
    hotmart_product_offers_path_template: str = "/products/api/v1/products/{product_id}/offers"
    hotmart_product_plans_path_template: str = "/products/api/v1/products/{product_id}/plans"
    hotmart_sales_history_path: str = "/payments/api/v1/sales/history"
    hotmart_subscriptions_list_path: str = "/payments/api/v1/subscriptions"
    hotmart_club_modules_path: str = "/club/api/v1/modules"
    hotmart_club_pages_path_template: str = "/club/api/v1/modules/{module_id}/pages"
    hotmart_club_students_path: str = "/club/api/v1/users"
    hotmart_club_progress_path_template: str = "/club/api/v1/users/{user_id}/lessons"
    schema_management_mode: str = "create_all_local"

    model_config = SettingsConfigDict(
        env_file=BACKEND_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def split_cors_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, list):
            return value
        return [item.strip() for item in value.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


def runtime_legacy_file_fallback_enabled(settings: Settings | None = None) -> bool:
    resolved = settings or get_settings()
    if resolved.runtime_legacy_file_fallback_enabled is not None:
        return resolved.runtime_legacy_file_fallback_enabled
    return resolved.app_debug


def runtime_legacy_file_write_through_enabled(settings: Settings | None = None) -> bool:
    resolved = settings or get_settings()
    if resolved.runtime_legacy_file_write_through_enabled is not None:
        return resolved.runtime_legacy_file_write_through_enabled
    return False


def allow_demo_tier_upgrade(settings: Settings | None = None) -> bool:
    resolved = settings or get_settings()
    if resolved.allow_demo_tier_upgrade is not None:
        return resolved.allow_demo_tier_upgrade
    return resolved.app_debug


def _database_is_local(settings: Settings) -> bool:
    parsed = make_url(settings.database_url)
    if parsed.drivername.startswith("sqlite"):
        return True
    host = (parsed.host or "").strip().lower()
    return host in {"127.0.0.1", "localhost"}


def should_auto_create_schema(settings: Settings | None = None) -> bool:
    resolved = settings or get_settings()
    mode = resolved.schema_management_mode.strip().lower()
    if mode == "alembic":
        return False
    if mode == "create_all_local":
        return _database_is_local(resolved)
    return True


def knowledge_repo_autosync_enabled(settings: Settings | None = None) -> bool:
    resolved = settings or get_settings()
    if resolved.knowledge_repo_autosync_enabled is not None:
        return resolved.knowledge_repo_autosync_enabled
    return _database_is_local(resolved)
