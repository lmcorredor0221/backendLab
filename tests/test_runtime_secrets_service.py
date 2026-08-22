from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.core.config import get_settings
from app.models import (
    LLMProviderKey,
    RuntimeSettingsAuditRecord,
    RuntimeSecretStatus,
    UserRecord,
    WorkspaceProviderSecretRecord,
    WorkspaceRecord,
    WorkspaceProviderSecretUpsertRequest,
)
from app.services.auth_service import hash_password
from app.services.llm_runtime.runtime_secrets_service import (
    annotate_runtime_settings_with_workspace_secrets,
    delete_workspace_provider_secret,
    resolve_workspace_provider_secret_value,
    upsert_workspace_provider_secret,
)
from app.services.llm_runtime.runtime_settings_service import load_effective_runtime_settings


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


def test_upsert_workspace_provider_secret_encrypts_storage_and_redacts_runtime_view() -> None:
    settings = get_settings()
    original_path = settings.llm_config_path

    with TemporaryDirectory(prefix="lean-builder-runtime-secrets-") as runtime_dir:
        runtime_path = Path(runtime_dir) / "llm_settings.json"
        runtime_path.write_text(json.dumps({"active_provider": "openai"}, ensure_ascii=True), encoding="utf-8")
        settings.llm_config_path = runtime_path
        try:
            engine = _build_engine()
            SQLModel.metadata.create_all(engine)
            with Session(engine) as session:
                actor, workspace = _seed_user_and_workspace(
                    session,
                    email="runtime-owner@leanbuilder.local",
                    workspace_name="Workspace Secrets",
                )

                secret_view = upsert_workspace_provider_secret(
                    session,
                    workspace.id,
                    LLMProviderKey.openai,
                    WorkspaceProviderSecretUpsertRequest(
                        secret_value="sk-workspace-alpha",
                        activate_for_runtime=True,
                    ),
                    actor_user_id=actor.id,
                )
                stored_secret = session.exec(
                    select(WorkspaceProviderSecretRecord).where(
                        WorkspaceProviderSecretRecord.workspace_id == workspace.id,
                        WorkspaceProviderSecretRecord.provider_key == LLMProviderKey.openai,
                    )
                ).first()
                audit_rows = session.exec(
                    select(RuntimeSettingsAuditRecord).where(
                        RuntimeSettingsAuditRecord.scope_id == str(workspace.id)
                    )
                ).all()
                runtime_view = annotate_runtime_settings_with_workspace_secrets(
                    session,
                    workspace.id,
                    load_effective_runtime_settings(session, workspace.id),
                )
                resolved_secret = resolve_workspace_provider_secret_value(
                    session,
                    workspace.id,
                    LLMProviderKey.openai,
                )
        finally:
            settings.llm_config_path = original_path

    assert stored_secret is not None
    assert stored_secret.status == RuntimeSecretStatus.configured
    assert stored_secret.secret_ciphertext
    assert stored_secret.secret_ciphertext != "sk-workspace-alpha"
    assert stored_secret.secret_ref == ""
    assert secret_view.secret_source == "workspace_managed"
    assert secret_view.storage_mode == "ciphertext"
    assert runtime_view.uses_platform_credentials is False
    assert runtime_view.openai.api_key_configured is True
    assert runtime_view.openai.secret_source == "workspace_managed"
    assert runtime_view.openai.health_status == "workspace_ready"
    assert resolved_secret == "sk-workspace-alpha"
    assert "sk-workspace-alpha" not in runtime_view.model_dump_json()
    assert any(item.change_type == "workspace_provider_secret_upserted" for item in audit_rows)
    assert all("sk-workspace-alpha" not in json.dumps(item.after_payload_redacted) for item in audit_rows)


def test_workspace_secrets_are_isolated_and_delete_reverts_to_platform_mode() -> None:
    settings = get_settings()
    original_path = settings.llm_config_path

    with TemporaryDirectory(prefix="lean-builder-runtime-secrets-") as runtime_dir:
        runtime_path = Path(runtime_dir) / "llm_settings.json"
        runtime_path.write_text(json.dumps({"active_provider": "openai"}, ensure_ascii=True), encoding="utf-8")
        settings.llm_config_path = runtime_path
        try:
            engine = _build_engine()
            SQLModel.metadata.create_all(engine)
            with Session(engine) as session:
                actor, workspace_a = _seed_user_and_workspace(
                    session,
                    email="workspace-a@leanbuilder.local",
                    workspace_name="Workspace A",
                )
                _, workspace_b = _seed_user_and_workspace(
                    session,
                    email="workspace-b@leanbuilder.local",
                    workspace_name="Workspace B",
                )

                upsert_workspace_provider_secret(
                    session,
                    workspace_a.id,
                    LLMProviderKey.deepseek,
                    WorkspaceProviderSecretUpsertRequest(
                        secret_ref="vault://deepseek/workspace-a",
                        activate_for_runtime=False,
                    ),
                    actor_user_id=actor.id,
                )
                runtime_a = annotate_runtime_settings_with_workspace_secrets(
                    session,
                    workspace_a.id,
                    load_effective_runtime_settings(session, workspace_a.id),
                )
                runtime_b = annotate_runtime_settings_with_workspace_secrets(
                    session,
                    workspace_b.id,
                    load_effective_runtime_settings(session, workspace_b.id),
                )
                delete_view = delete_workspace_provider_secret(
                    session,
                    workspace_a.id,
                    LLMProviderKey.deepseek,
                    actor_user_id=actor.id,
                )
                runtime_after_delete = annotate_runtime_settings_with_workspace_secrets(
                    session,
                    workspace_a.id,
                    load_effective_runtime_settings(session, workspace_a.id),
                )
                audit_rows = session.exec(
                    select(RuntimeSettingsAuditRecord).where(
                        RuntimeSettingsAuditRecord.scope_id == str(workspace_a.id)
                    )
                ).all()
        finally:
            settings.llm_config_path = original_path

    assert runtime_a.deepseek.secret_source == "workspace_staged"
    assert runtime_a.deepseek.api_key_configured is True
    assert runtime_b.deepseek.secret_source == "platform_managed"
    assert runtime_b.deepseek.api_key_configured is False
    assert delete_view.secret_source == "platform_managed"
    assert delete_view.configured is False
    assert delete_view.uses_platform_credentials is True
    assert runtime_after_delete.uses_platform_credentials is True
    assert runtime_after_delete.deepseek.secret_source == "platform_managed"
    assert any(item.change_type == "workspace_provider_secret_upserted" for item in audit_rows)
    assert any(item.change_type == "workspace_provider_secret_deleted" for item in audit_rows)
    assert all("vault://deepseek/workspace-a" not in json.dumps(item.after_payload_redacted) for item in audit_rows)
