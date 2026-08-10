from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from sqlmodel import Session, select

from app.core.config import (
    get_settings,
    runtime_legacy_file_fallback_enabled,
    runtime_legacy_file_write_through_enabled,
)
from app.models import (
    CodexLocalProviderConfigUpdate,
    DeepSeekProviderConfigUpdate,
    LLMRuntimeSettings,
    OpenAIProviderConfigUpdate,
    PlatformRuntimeDefaultsRecord,
    RuntimeGovernanceScopeType,
    RuntimeSettingsAuditRecord,
    SchemaMigrationRecord,
    WorkspaceRecord,
    WorkspaceRuntimeSettingsRecord,
    utc_now,
)
from app.services.openai_builder import load_llm_runtime_settings
from app.services.runtime_governance_bootstrap import seed_platform_runtime_governance


MIGRATION_KEY_RT7 = "2026-07-20-rt7-runtime-llm-multitenant-backfill"
PLATFORM_RUNTIME_SCOPE_ID = "platform-runtime-defaults"
WorkspaceMigrationMode = Literal["inherit_defaults", "seed_overrides"]


@dataclass
class RuntimeLLMMultitenantMigrationSummary:
    migration_key: str = MIGRATION_KEY_RT7
    already_recorded: bool = False
    config_path: str = ""
    config_exists: bool = False
    config_changed_by_normalization: bool = False
    legacy_file_fallback_enabled: bool = False
    legacy_file_write_through_enabled: bool = False
    provider_registry_seeded: int = 0
    provider_registry_preexisting: int = 0
    platform_defaults_seeded: bool = False
    platform_defaults_preexisting: bool = False
    workspace_mode: WorkspaceMigrationMode = "inherit_defaults"
    workspace_candidates: int = 0
    workspace_settings_seeded: int = 0
    workspace_settings_preexisting: int = 0
    audit_entries_created: int = 0
    normalized_payload: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    report_generated_at: str = field(default_factory=lambda: utc_now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_persistable_runtime_settings_payload(
    resolved: LLMRuntimeSettings,
    *,
    updated_at: str | None = None,
) -> dict[str, Any]:
    return {
        "active_provider": resolved.active_provider.value,
        "agent_execution_backend": resolved.agent_execution_backend.value,
        "knowledge_access_backend": resolved.knowledge_access_backend.value,
        "uses_platform_credentials": resolved.uses_platform_credentials,
        "openai": {
            "fast_model": resolved.openai.fast_model,
            "reasoning_model": resolved.openai.reasoning_model,
            "reasoning_effort": resolved.openai.reasoning_effort,
        },
        "deepseek": {
            "base_url": resolved.deepseek.base_url,
            "fast_model": resolved.deepseek.fast_model,
            "reasoning_model": resolved.deepseek.reasoning_model,
            "reasoning_effort": resolved.deepseek.reasoning_effort,
        },
        "codex_local": {
            "command": resolved.codex_local.command,
            "model": resolved.codex_local.model,
            "profile": resolved.codex_local.profile,
            "cost_policy": resolved.codex_local.cost_policy.value,
            "timeout_ms": resolved.codex_local.timeout_ms,
            "max_concurrency": resolved.codex_local.max_concurrency,
            "runner_id": resolved.codex_local.runner_id,
            "auth_mode": resolved.codex_local.auth_mode.value,
            "fallback_models": list(resolved.codex_local.fallback_models),
            "primary_agents": list(resolved.codex_local.primary_agents),
            "shadow_agents": list(resolved.codex_local.shadow_agents),
            "staged_agents": list(resolved.codex_local.staged_agents),
        },
        "compatibility_mode": resolved.compatibility_mode,
        "updated_at": updated_at or utc_now().isoformat(),
    }


def _provider_defaults_payload(resolved: LLMRuntimeSettings) -> dict[str, Any]:
    return {
        "openai": OpenAIProviderConfigUpdate(
            fast_model=resolved.openai.fast_model,
            reasoning_model=resolved.openai.reasoning_model,
            reasoning_effort=resolved.openai.reasoning_effort,
        ).model_dump(mode="json"),
        "deepseek": DeepSeekProviderConfigUpdate(
            base_url=resolved.deepseek.base_url,
            fast_model=resolved.deepseek.fast_model,
            reasoning_model=resolved.deepseek.reasoning_model,
            reasoning_effort=resolved.deepseek.reasoning_effort,
        ).model_dump(mode="json"),
        "codex_local": CodexLocalProviderConfigUpdate(
            command=resolved.codex_local.command,
            model=resolved.codex_local.model,
            profile=resolved.codex_local.profile,
            cost_policy=resolved.codex_local.cost_policy,
            timeout_ms=resolved.codex_local.timeout_ms,
            max_concurrency=resolved.codex_local.max_concurrency,
            runner_id=resolved.codex_local.runner_id,
            auth_mode=resolved.codex_local.auth_mode,
            fallback_models=list(resolved.codex_local.fallback_models),
            primary_agents=list(resolved.codex_local.primary_agents),
            shadow_agents=list(resolved.codex_local.shadow_agents),
            staged_agents=list(resolved.codex_local.staged_agents),
        ).model_dump(mode="json"),
    }


def _read_raw_runtime_payload(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_resolved_runtime_settings(config_path: Path) -> LLMRuntimeSettings:
    settings = get_settings()
    original_path = settings.llm_config_path
    original_fallback_flag = settings.runtime_legacy_file_fallback_enabled
    settings.llm_config_path = config_path
    settings.runtime_legacy_file_fallback_enabled = True
    try:
        return load_llm_runtime_settings()
    finally:
        settings.llm_config_path = original_path
        settings.runtime_legacy_file_fallback_enabled = original_fallback_flag


def _active_platform_defaults_record(session: Session) -> PlatformRuntimeDefaultsRecord | None:
    return session.exec(
        select(PlatformRuntimeDefaultsRecord)
        .where(PlatformRuntimeDefaultsRecord.is_active == True)  # noqa: E712
        .order_by(PlatformRuntimeDefaultsRecord.version.desc())
    ).first()


def _next_platform_defaults_version(session: Session) -> int:
    latest = session.exec(select(PlatformRuntimeDefaultsRecord).order_by(PlatformRuntimeDefaultsRecord.version.desc())).first()
    return 1 if latest is None else latest.version + 1


def _active_workspace_runtime_ids(session: Session) -> set[str]:
    rows = session.exec(
        select(WorkspaceRuntimeSettingsRecord).where(WorkspaceRuntimeSettingsRecord.is_active == True)  # noqa: E712
    ).all()
    return {str(item.workspace_id) for item in rows}


def inspect_runtime_settings_migration(config_path: Path) -> dict[str, Any]:
    original_payload = _read_raw_runtime_payload(config_path)
    resolved = _load_resolved_runtime_settings(config_path)
    normalized_payload = build_persistable_runtime_settings_payload(
        resolved,
        updated_at=original_payload.get("updated_at") if isinstance(original_payload, dict) else None,
    )
    return {
        "changed": normalized_payload != original_payload,
        "config_path": str(config_path),
        "normalized_payload": normalized_payload,
        "original_payload": original_payload,
    }


def migrate_runtime_settings_file(
    config_path: Path,
    *,
    backup_suffix: str = ".bak",
) -> dict[str, Any]:
    inspection = inspect_runtime_settings_migration(config_path)
    if not inspection["changed"]:
        return {
            **inspection,
            "backup_path": None,
            "written": False,
            "written_payload": inspection["normalized_payload"],
        }

    backup_path = config_path.with_suffix(f"{config_path.suffix}{backup_suffix}")
    if config_path.exists():
        backup_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    resolved = _load_resolved_runtime_settings(config_path)
    written_payload = build_persistable_runtime_settings_payload(resolved)
    config_path.write_text(
        json.dumps(written_payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return {
        **inspection,
        "backup_path": str(backup_path),
        "written": True,
        "written_payload": written_payload,
    }


def write_runtime_llm_multitenant_migration_report(
    summary: RuntimeLLMMultitenantMigrationSummary,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary.to_dict(), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return output_path


def apply_runtime_llm_multitenant_migration(
    session: Session,
    *,
    config_path: Path | None = None,
    workspace_mode: WorkspaceMigrationMode = "inherit_defaults",
    report_path: Path | None = None,
) -> RuntimeLLMMultitenantMigrationSummary:
    if workspace_mode not in {"inherit_defaults", "seed_overrides"}:
        raise ValueError("workspace_mode debe ser 'inherit_defaults' o 'seed_overrides'.")

    resolved_config_path = Path(config_path or get_settings().llm_config_path).resolve()
    inspection = inspect_runtime_settings_migration(resolved_config_path)
    resolved_runtime = _load_resolved_runtime_settings(resolved_config_path)
    normalized_payload = inspection["normalized_payload"]
    summary = RuntimeLLMMultitenantMigrationSummary(
        config_path=str(resolved_config_path),
        config_exists=resolved_config_path.exists(),
        config_changed_by_normalization=bool(inspection["changed"]),
        legacy_file_fallback_enabled=runtime_legacy_file_fallback_enabled(),
        legacy_file_write_through_enabled=runtime_legacy_file_write_through_enabled(),
        normalized_payload=normalized_payload,
        workspace_mode=workspace_mode,
    )

    provider_summary = seed_platform_runtime_governance(
        session,
        runtime_settings=resolved_runtime,
        seed_defaults=False,
    )
    summary.provider_registry_seeded = int(provider_summary["provider_created"])
    summary.provider_registry_preexisting = int(provider_summary["provider_existing"])

    dirty = False
    migration_record = session.exec(
        select(SchemaMigrationRecord).where(SchemaMigrationRecord.migration_key == MIGRATION_KEY_RT7)
    ).first()
    if migration_record is not None:
        summary.already_recorded = True

    active_defaults = _active_platform_defaults_record(session)
    if active_defaults is None:
        now = utc_now()
        session.add(
            PlatformRuntimeDefaultsRecord(
                active_provider_default=resolved_runtime.active_provider,
                agent_execution_backend_default=resolved_runtime.agent_execution_backend,
                knowledge_access_backend_default=resolved_runtime.knowledge_access_backend,
                per_provider_defaults=_provider_defaults_payload(resolved_runtime),
                is_active=True,
                version=_next_platform_defaults_version(session),
                updated_by_user_id=None,
                created_at=now,
                updated_at=now,
            )
        )
        session.add(
            RuntimeSettingsAuditRecord(
                scope_type=RuntimeGovernanceScopeType.platform,
                scope_id=PLATFORM_RUNTIME_SCOPE_ID,
                change_type="runtime_legacy_file_backfill",
                before_payload_redacted={},
                after_payload_redacted=normalized_payload,
                actor_user_id=None,
                actor_email="system",
            )
        )
        summary.platform_defaults_seeded = True
        summary.audit_entries_created += 1
        dirty = True
    else:
        summary.platform_defaults_preexisting = True

    workspaces = session.exec(select(WorkspaceRecord).order_by(WorkspaceRecord.created_at.asc())).all()
    active_workspace_runtime_ids = _active_workspace_runtime_ids(session)
    summary.workspace_candidates = len(workspaces)
    summary.workspace_settings_preexisting = len(active_workspace_runtime_ids)

    if workspace_mode == "seed_overrides":
        now = utc_now()
        provider_overrides = _provider_defaults_payload(resolved_runtime)
        for workspace in workspaces:
            if str(workspace.id) in active_workspace_runtime_ids:
                continue
            session.add(
                WorkspaceRuntimeSettingsRecord(
                    workspace_id=workspace.id,
                    active_provider=resolved_runtime.active_provider,
                    agent_execution_backend=resolved_runtime.agent_execution_backend,
                    knowledge_access_backend=resolved_runtime.knowledge_access_backend,
                    provider_overrides=provider_overrides,
                    uses_platform_credentials=resolved_runtime.uses_platform_credentials,
                    is_active=True,
                    version=1,
                    updated_by_user_id=None,
                    created_at=now,
                    updated_at=now,
                )
            )
            session.add(
                RuntimeSettingsAuditRecord(
                    scope_type=RuntimeGovernanceScopeType.workspace,
                    scope_id=str(workspace.id),
                    change_type="runtime_legacy_workspace_seed",
                    before_payload_redacted={},
                    after_payload_redacted=normalized_payload,
                    actor_user_id=None,
                    actor_email="system",
                )
            )
            summary.workspace_settings_seeded += 1
            summary.audit_entries_created += 1
            dirty = True
    else:
        summary.notes.append(
            "Los workspaces existentes heredaran defaults de plataforma sin crear overrides adicionales."
        )

    if migration_record is None:
        session.add(
            SchemaMigrationRecord(
                migration_key=MIGRATION_KEY_RT7,
                description=(
                    "Migra runtime LLM legado a platform defaults/workspace-compatible settings "
                    "y deja el archivo como insumo de compatibilidad controlada."
                ),
            )
        )
        dirty = True

    if summary.legacy_file_fallback_enabled:
        summary.notes.append(
            "El fallback al archivo legado sigue habilitado como compatibilidad controlada para desarrollo."
        )
    else:
        summary.notes.append(
            "El fallback al archivo legado esta deshabilitado; la base de datos gobierna el runtime operativo."
        )

    if summary.legacy_file_write_through_enabled:
        summary.notes.append(
            "El write-through al archivo legado sigue habilitado por configuracion; revisar antes de produccion."
        )
    else:
        summary.notes.append(
            "El write-through al archivo legado queda deshabilitado por defecto en el runtime multitenant."
        )

    if summary.config_changed_by_normalization:
        summary.notes.append(
            "El archivo legado requiere normalizacion; ejecutar la migracion de archivo si se desea conservarlo legible."
        )

    if dirty:
        session.commit()

    if report_path is not None:
        write_runtime_llm_multitenant_migration_report(summary, report_path)

    return summary
