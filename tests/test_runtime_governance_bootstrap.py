from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.core.config import get_settings
from app.models import (
    PlatformRole,
    PlatformRoleAssignmentRecord,
    PlatformRuntimeDefaultsRecord,
    PlatformRuntimeProviderRecord,
    UserRecord,
)
from app.services.auth_service import hash_password
from app.services.runtime_governance_bootstrap import (
    backfill_platform_runtime_governance,
    ensure_platform_admin_role,
    seed_platform_runtime_governance,
)


def _build_engine():
    return create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def test_seed_platform_runtime_governance_creates_provider_registry_and_platform_defaults() -> None:
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
            "profile": "local-default",
            "cost_policy": "hybrid",
            "timeout_ms": 120000,
            "max_concurrency": 2,
            "runner_id": "codex-dev",
            "auth_mode": "auto",
            "fallback_models": ["gpt-5.5-mini"],
            "primary_agents": ["builder"],
            "shadow_agents": ["validator"],
            "staged_agents": ["designer"],
        },
        "compatibility_mode": "backward_compatible",
        "updated_at": "2026-07-20T10:00:00",
    }
    settings = get_settings()
    original_path = settings.llm_config_path

    with TemporaryDirectory(prefix="lean-builder-runtime-governance-") as runtime_dir:
        runtime_path = Path(runtime_dir) / "llm_settings.json"
        runtime_path.write_text(json.dumps(legacy_payload, ensure_ascii=True, indent=2), encoding="utf-8")
        settings.llm_config_path = runtime_path
        try:
            engine = _build_engine()
            SQLModel.metadata.create_all(engine)
            with Session(engine) as session:
                seed_platform_runtime_governance(session, seed_defaults=True)

                provider_rows = session.exec(
                    select(PlatformRuntimeProviderRecord).order_by(PlatformRuntimeProviderRecord.provider_key.asc())
                ).all()
                defaults_row = session.exec(
                    select(PlatformRuntimeDefaultsRecord).where(PlatformRuntimeDefaultsRecord.is_active == True)  # noqa: E712
                ).first()
        finally:
            settings.llm_config_path = original_path

    assert defaults_row is not None
    assert defaults_row.active_provider_default.value == "deepseek"
    assert defaults_row.agent_execution_backend_default.value == "provider_native"
    assert defaults_row.knowledge_access_backend_default.value == "workspace_staged"
    assert defaults_row.per_provider_defaults["codex_local"]["runner_id"] == "codex-dev"

    provider_keys = [item.provider_key.value for item in provider_rows]
    assert provider_keys == ["antigravity_cli", "codex_local", "deepseek", "openai"]

    codex_provider = next(item for item in provider_rows if item.provider_key.value == "codex_local")
    assert codex_provider.release_stage.value == "preview"
    assert codex_provider.supports_platform_managed_credentials is False
    assert "gpt-5.5" in codex_provider.allowed_models


def test_backfill_platform_runtime_governance_assigns_platform_admin_to_local_admin() -> None:
    settings = get_settings()
    original_path = settings.llm_config_path

    with TemporaryDirectory(prefix="lean-builder-runtime-admin-") as runtime_dir:
        runtime_path = Path(runtime_dir) / "llm_settings.json"
        runtime_path.write_text(json.dumps({"active_provider": "openai"}, ensure_ascii=True), encoding="utf-8")
        settings.llm_config_path = runtime_path
        try:
            engine = _build_engine()
            SQLModel.metadata.create_all(engine)
            with Session(engine) as session:
                admin = UserRecord(
                    email=settings.local_admin_email,
                    full_name=settings.local_admin_name,
                    password_hash=hash_password(settings.local_admin_password),
                )
                session.add(admin)
                session.commit()
                session.refresh(admin)

                backfill_platform_runtime_governance(session)

                assignment = session.exec(
                    select(PlatformRoleAssignmentRecord).where(
                        PlatformRoleAssignmentRecord.user_id == admin.id,
                        PlatformRoleAssignmentRecord.role == PlatformRole.platform_admin,
                    )
                ).first()
        finally:
            settings.llm_config_path = original_path

    assert assignment is not None
    assert assignment.is_active is True


def test_ensure_platform_admin_role_is_idempotent() -> None:
    engine = _build_engine()
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        user = UserRecord(
            email="platform-owner@leanbuilder.local",
            full_name="Platform Owner",
            password_hash=hash_password("LeanBuilder123!"),
        )
        session.add(user)
        session.commit()
        session.refresh(user)

        ensure_platform_admin_role(session, user)
        ensure_platform_admin_role(session, user)

        assignments = session.exec(
            select(PlatformRoleAssignmentRecord).where(
                PlatformRoleAssignmentRecord.user_id == user.id,
                PlatformRoleAssignmentRecord.role == PlatformRole.platform_admin,
            )
        ).all()

    assert len(assignments) == 1
