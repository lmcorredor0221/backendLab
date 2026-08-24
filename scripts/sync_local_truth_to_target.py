from __future__ import annotations

import argparse
import copy
import json
import os
import socket
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy.engine import make_url
from sqlmodel import Session, create_engine, select

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from alembic import command
from alembic.config import Config

from app.core.config import get_settings
from app.db import bootstrap_application_data
from app.models import (
    AntigravityProviderConfigUpdate,
    CommercialPackageCatalogRecord,
    CommercialQuotaProductConfigRecord,
    CodexLocalProviderConfigUpdate,
    DeepSeekProviderConfigUpdate,
    KnowledgeScope,
    LLMProviderKey,
    LLMRuntimeSettingsUpdateRequest,
    OpenAIProviderConfigUpdate,
    PlatformRuntimeDefaultsRecord,
    PlatformRuntimeProviderRecord,
    ProductCatalogRecord,
    ProductPriceRecord,
    RuntimeCatalogEntryRecord,
    RuntimeFeatureFlagRecord,
    RuntimeSecretStatus,
    WorkspaceProviderSecretRecord,
    WorkspaceRecord,
    WorkflowTemplateRecord,
    GovernancePolicyRecord,
)
from app.services.knowledge_memory import KnowledgeMemoryService
from app.services.llm_runtime.runtime_secrets_service import (
    _encrypt_secret_value,
    resolve_workspace_provider_secret_value,
)
from app.services.llm_runtime.runtime_settings_service import (
    load_effective_runtime_settings,
    load_platform_runtime_defaults,
    load_workspace_runtime_settings,
    persist_platform_runtime_defaults,
    persist_workspace_runtime_settings,
    reset_workspace_runtime_settings,
)
from app.services.workspace_bootstrap import apply_workspace_bootstrap


@dataclass
class TableSyncSummary:
    created: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped: int = 0
    extras: int = 0
    notes: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sincroniza desde la base local hacia una base objetivo los seeds, parametros de runtime y version "
            "de esquema gobernados por el repositorio."
        ),
    )
    parser.add_argument(
        "--target-database-url",
        default=os.environ.get("TARGET_DATABASE_URL", "").strip(),
        help="Cadena de conexion de la base objetivo.",
    )
    parser.add_argument(
        "--source-database-url",
        default=get_settings().database_url,
        help="Cadena de conexion de la base fuente local.",
    )
    parser.add_argument(
        "--workspace-slug",
        action="append",
        default=[],
        help="Limita la sincronizacion workspace-scoped a los slugs indicados. Se puede repetir.",
    )
    parser.add_argument(
        "--workspace-map",
        action="append",
        default=[],
        help=(
            "Mapea un workspace fuente hacia un workspace destino con el formato "
            "'source_slug=target_slug'. Se puede repetir para propagar un mismo source "
            "a varios targets."
        ),
    )
    parser.add_argument(
        "--skip-alembic",
        action="store_true",
        help="Omite el upgrade Alembic de la base objetivo.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calcula y reporta cambios sin escribir en la base objetivo.",
    )
    parser.add_argument(
        "--force-knowledge",
        action="store_true",
        help="Fuerza la resincronizacion del corpus de knowledge en la base objetivo.",
    )
    parser.add_argument(
        "--sync-workspace-secrets",
        action="store_true",
        help="Reencripta y sincroniza secretos por workspace para OpenAI y DeepSeek.",
    )
    parser.add_argument(
        "--target-runtime-secrets-master-key",
        default=os.environ.get("TARGET_RUNTIME_SECRETS_MASTER_KEY", "").strip(),
        help="Master key del target para reencriptar workspace_provider_secrets.",
    )
    parser.add_argument(
        "--target-local-admin-email",
        default=os.environ.get("TARGET_LOCAL_ADMIN_EMAIL", "").strip(),
        help="Email admin del target cuando la encriptacion depende del fallback por entorno.",
    )
    parser.add_argument(
        "--target-local-admin-password",
        default=os.environ.get("TARGET_LOCAL_ADMIN_PASSWORD", "").strip(),
        help="Password admin del target cuando la encriptacion depende del fallback por entorno.",
    )
    parser.add_argument(
        "--report",
        default="",
        help="Ruta opcional de reporte JSON.",
    )
    return parser.parse_args()


def _payload(record: Any, field_names: list[str], *, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {field_name: copy.deepcopy(getattr(record, field_name)) for field_name in field_names}
    if overrides:
        payload.update(overrides)
    return payload


def _apply_fields(target_record: Any, source_record: Any, field_names: list[str], *, overrides: dict[str, Any] | None = None) -> bool:
    changed = False
    payload = _payload(source_record, field_names, overrides=overrides)
    for field_name, value in payload.items():
        if getattr(target_record, field_name) != value:
            setattr(target_record, field_name, value)
            changed = True
    return changed


def _descriptor(database_url: str) -> dict[str, Any]:
    parsed = make_url(database_url)
    return {
        "drivername": parsed.drivername,
        "host": parsed.host or "",
        "port": parsed.port or "",
        "database": parsed.database or "",
    }


def _normalized_database_identity(database_url: str) -> tuple[str, str, str, str]:
    descriptor = _descriptor(database_url)
    return (
        descriptor["drivername"],
        descriptor["host"],
        str(descriptor["port"]),
        descriptor["database"],
    )


def _database_url_with_hostaddr(database_url: str) -> str:
    parsed = make_url(database_url)
    if not parsed.host or parsed.query.get("hostaddr"):
        return database_url
    try:
        hostaddr = socket.gethostbyname(parsed.host)
    except OSError:
        return database_url
    return parsed.update_query_dict({"hostaddr": hostaddr}).render_as_string(hide_password=False)


@contextmanager
def _temporary_settings_env(**overrides: str | None):
    original = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()


def _runtime_update_request(runtime_settings) -> LLMRuntimeSettingsUpdateRequest:
    return LLMRuntimeSettingsUpdateRequest(
        active_provider=runtime_settings.active_provider,
        agent_execution_backend=runtime_settings.agent_execution_backend,
        knowledge_access_backend=runtime_settings.knowledge_access_backend,
        uses_platform_credentials=runtime_settings.uses_platform_credentials,
        openai=OpenAIProviderConfigUpdate(
            fast_model=runtime_settings.openai.fast_model,
            reasoning_model=runtime_settings.openai.reasoning_model,
            reasoning_effort=runtime_settings.openai.reasoning_effort,
        ),
        deepseek=DeepSeekProviderConfigUpdate(
            base_url=runtime_settings.deepseek.base_url,
            fast_model=runtime_settings.deepseek.fast_model,
            reasoning_model=runtime_settings.deepseek.reasoning_model,
            reasoning_effort=runtime_settings.deepseek.reasoning_effort,
        ),
        codex_local=CodexLocalProviderConfigUpdate(
            command=runtime_settings.codex_local.command,
            model=runtime_settings.codex_local.model,
            profile=runtime_settings.codex_local.profile,
            cost_policy=runtime_settings.codex_local.cost_policy,
            timeout_ms=runtime_settings.codex_local.timeout_ms,
            max_concurrency=runtime_settings.codex_local.max_concurrency,
            runner_id=runtime_settings.codex_local.runner_id,
            auth_mode=runtime_settings.codex_local.auth_mode,
            fallback_models=list(runtime_settings.codex_local.fallback_models),
            primary_agents=list(runtime_settings.codex_local.primary_agents),
            shadow_agents=list(runtime_settings.codex_local.shadow_agents),
            staged_agents=list(runtime_settings.codex_local.staged_agents),
        ),
        antigravity_cli=AntigravityProviderConfigUpdate(
            executable=runtime_settings.antigravity.executable,
            model=runtime_settings.antigravity.model,
            effort=runtime_settings.antigravity.effort,
            timeout_ms=runtime_settings.antigravity.timeout_ms,
            max_concurrency=runtime_settings.antigravity.max_concurrency,
            runner_id=runtime_settings.antigravity.runner_id,
            auth_mode=runtime_settings.antigravity.auth_mode,
            fallback_models=list(runtime_settings.antigravity.fallback_models),
            primary_agents=list(runtime_settings.antigravity.primary_agents),
            shadow_agents=list(runtime_settings.antigravity.shadow_agents),
            staged_agents=list(runtime_settings.antigravity.staged_agents),
        ),
    )


def _runtime_signature(runtime_settings) -> dict[str, Any]:
    payload = _runtime_update_request(runtime_settings).model_dump(mode="json", by_alias=True, exclude_none=True)
    return payload


def _validate_secret_sync_context(args: argparse.Namespace) -> None:
    if not args.sync_workspace_secrets:
        return
    if args.target_runtime_secrets_master_key:
        return
    if args.target_local_admin_email and args.target_local_admin_password:
        return
    raise ValueError(
        "Para sincronizar workspace secrets debes proveer --target-runtime-secrets-master-key o "
        "--target-local-admin-email junto con --target-local-admin-password."
    )


def _parse_workspace_maps(items: list[str]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for raw_item in items:
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Workspace map invalido '{raw_item}'. Usa el formato source_slug=target_slug."
            )
        source_slug, target_slug = item.split("=", 1)
        normalized_source = source_slug.strip().lower()
        normalized_target = target_slug.strip().lower()
        if not normalized_source or not normalized_target:
            raise ValueError(
                f"Workspace map invalido '{raw_item}'. Source y target no pueden estar vacios."
            )
        mapping.setdefault(normalized_source, [])
        if normalized_target not in mapping[normalized_source]:
            mapping[normalized_source].append(normalized_target)
    return mapping


def _upgrade_target_schema(target_database_url: str) -> None:
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    try:
        with _temporary_settings_env(DATABASE_URL=_database_url_with_hostaddr(target_database_url)):
            command.upgrade(config, "head")
    except Exception as exc:
        raise RuntimeError(
            "No se pudo llevar la base objetivo a Alembic head. Si el target sigue sin alembic_version pero ya "
            "tiene tablas legacy, ejecuta primero backend/scripts/prod_schema_alignment_20260824.sql."
        ) from exc


def _sync_platform_runtime_providers(
    source_session: Session,
    target_session: Session,
    *,
    dry_run: bool,
) -> TableSyncSummary:
    fields = [
        "label",
        "is_enabled",
        "allowed_models",
        "default_models",
        "allowed_auth_modes",
        "supports_workspace_secrets",
        "supports_platform_managed_credentials",
        "release_stage",
        "health_policy",
        "updated_at",
    ]
    summary = TableSyncSummary()
    source_rows = list(source_session.exec(select(PlatformRuntimeProviderRecord)).all())
    target_rows = {
        row.provider_key: row
        for row in target_session.exec(select(PlatformRuntimeProviderRecord)).all()
    }

    for source_row in source_rows:
        target_row = target_rows.get(source_row.provider_key)
        if target_row is None:
            summary.created += 1
            if not dry_run:
                target_session.add(
                    PlatformRuntimeProviderRecord(
                        id=source_row.id,
                        provider_key=source_row.provider_key,
                        **_payload(source_row, fields),
                    )
                )
            continue
        if _apply_fields(target_row, source_row, fields):
            summary.updated += 1
        else:
            summary.unchanged += 1

    summary.extras = max(len(target_rows) - len(source_rows), 0)
    if not dry_run:
        target_session.commit()
    return summary


def _sync_platform_runtime_defaults(
    source_session: Session,
    target_session: Session,
    *,
    dry_run: bool,
) -> TableSyncSummary:
    summary = TableSyncSummary()
    source_runtime = load_platform_runtime_defaults(source_session)
    target_runtime = load_platform_runtime_defaults(target_session)
    if _runtime_signature(source_runtime) == _runtime_signature(target_runtime):
        summary.unchanged = 1
        return summary

    summary.updated = 1
    if not dry_run:
        persist_platform_runtime_defaults(
            target_session,
            _runtime_update_request(source_runtime),
            actor_user_id=None,
            mirror_legacy_runtime=False,
        )
    return summary


def _sync_runtime_catalog_entries(
    source_session: Session,
    target_session: Session,
    *,
    dry_run: bool,
) -> TableSyncSummary:
    fields = ["label", "version", "order_index", "is_active", "payload", "updated_at"]
    summary = TableSyncSummary()
    source_rows = list(source_session.exec(select(RuntimeCatalogEntryRecord)).all())
    target_rows = {
        (row.catalog_key, row.item_key): row
        for row in target_session.exec(select(RuntimeCatalogEntryRecord)).all()
    }

    for source_row in source_rows:
        key = (source_row.catalog_key, source_row.item_key)
        target_row = target_rows.get(key)
        if target_row is None:
            summary.created += 1
            if not dry_run:
                target_session.add(
                    RuntimeCatalogEntryRecord(
                        id=source_row.id,
                        catalog_key=source_row.catalog_key,
                        item_key=source_row.item_key,
                        **_payload(source_row, fields),
                    )
                )
            continue
        if _apply_fields(target_row, source_row, fields):
            summary.updated += 1
        else:
            summary.unchanged += 1

    summary.extras = max(len(target_rows) - len(source_rows), 0)
    if not dry_run:
        target_session.commit()
    return summary


def _sync_product_catalog(
    source_session: Session,
    target_session: Session,
    *,
    dry_run: bool,
) -> TableSyncSummary:
    fields = [
        "product_key",
        "tier",
        "product_type",
        "status",
        "name",
        "description",
        "scope",
        "benefits",
        "exclusions",
        "capabilities",
        "metadata_payload",
        "version",
        "is_active",
        "created_at",
        "updated_at",
    ]
    summary = TableSyncSummary()
    source_rows = list(source_session.exec(select(ProductCatalogRecord)).all())
    target_rows = {
        (row.product_key, row.version): row
        for row in target_session.exec(select(ProductCatalogRecord)).all()
    }

    for source_row in source_rows:
        key = (source_row.product_key, source_row.version)
        target_row = target_rows.get(key)
        if target_row is None:
            summary.created += 1
            if not dry_run:
                target_session.add(ProductCatalogRecord(id=source_row.id, **_payload(source_row, fields)))
            continue
        if _apply_fields(target_row, source_row, fields):
            summary.updated += 1
        else:
            summary.unchanged += 1

    summary.extras = max(len(target_rows) - len(source_rows), 0)
    if not dry_run:
        target_session.commit()
    return summary


def _sync_product_prices(
    source_session: Session,
    target_session: Session,
    *,
    dry_run: bool,
) -> TableSyncSummary:
    fields = [
        "product_key",
        "price_code",
        "currency",
        "unit_amount_cents",
        "unit_amount_usd_cents",
        "billing_period",
        "status",
        "version",
        "metadata_payload",
        "created_at",
        "updated_at",
    ]
    summary = TableSyncSummary()
    source_rows = list(source_session.exec(select(ProductPriceRecord)).all())
    target_rows = {
        row.price_code: row
        for row in target_session.exec(select(ProductPriceRecord)).all()
    }

    for source_row in source_rows:
        target_row = target_rows.get(source_row.price_code)
        if target_row is None:
            summary.created += 1
            if not dry_run:
                target_session.add(ProductPriceRecord(id=source_row.id, **_payload(source_row, fields)))
            continue
        if _apply_fields(target_row, source_row, fields):
            summary.updated += 1
        else:
            summary.unchanged += 1

    summary.extras = max(len(target_rows) - len(source_rows), 0)
    if not dry_run:
        target_session.commit()
    return summary


def _sync_quota_product_configs(
    source_session: Session,
    target_session: Session,
    *,
    dry_run: bool,
) -> TableSyncSummary:
    fields = [
        "product_key",
        "display_name",
        "enabled",
        "initial_free_units",
        "consumption_priority",
        "checkout_required_on_zero_balance",
        "fifo_auto_approval_enabled",
        "default_blocked_request_ttl_hours",
        "default_checkout_ttl_minutes",
        "debt_enabled",
        "allow_manual_override_without_charge",
        "allow_courtesy",
        "allow_debt_pending",
        "catalog_priority_strategy",
        "sync_retry_limit",
        "duplicate_conflict_visibility",
        "metadata_payload",
        "created_at",
        "updated_at",
    ]
    summary = TableSyncSummary()
    source_rows = list(source_session.exec(select(CommercialQuotaProductConfigRecord)).all())
    target_rows = {
        row.product_key: row
        for row in target_session.exec(select(CommercialQuotaProductConfigRecord)).all()
    }

    for source_row in source_rows:
        target_row = target_rows.get(source_row.product_key)
        if target_row is None:
            summary.created += 1
            if not dry_run:
                target_session.add(
                    CommercialQuotaProductConfigRecord(id=source_row.id, **_payload(source_row, fields))
                )
            continue
        if _apply_fields(target_row, source_row, fields):
            summary.updated += 1
        else:
            summary.unchanged += 1

    summary.extras = max(len(target_rows) - len(source_rows), 0)
    if not dry_run:
        target_session.commit()
    return summary


def _sync_package_catalog(
    source_session: Session,
    target_session: Session,
    *,
    dry_run: bool,
) -> TableSyncSummary:
    fields = [
        "package_code",
        "display_name",
        "product_key",
        "package_type",
        "enabled",
        "granted_units",
        "granted_units_blueprint_pro",
        "granted_units_acp",
        "validity_days",
        "billing_cycle",
        "renewal_policy",
        "recommendation_priority",
        "hotmart_environment",
        "hotmart_product_id",
        "hotmart_product_ucode",
        "offer_code",
        "plan_code",
        "checkout_currency_mode",
        "hotmart_price_strategy",
        "metadata_payload",
        "created_at",
        "updated_at",
    ]
    summary = TableSyncSummary()
    source_rows = list(source_session.exec(select(CommercialPackageCatalogRecord)).all())
    target_rows = {
        row.package_code: row
        for row in target_session.exec(select(CommercialPackageCatalogRecord)).all()
    }

    for source_row in source_rows:
        target_row = target_rows.get(source_row.package_code)
        if target_row is None:
            summary.created += 1
            if not dry_run:
                target_session.add(
                    CommercialPackageCatalogRecord(id=source_row.id, **_payload(source_row, fields))
                )
            continue
        if _apply_fields(target_row, source_row, fields):
            summary.updated += 1
        else:
            summary.unchanged += 1

    summary.extras = max(len(target_rows) - len(source_rows), 0)
    if not dry_run:
        target_session.commit()
    return summary


def _sync_runtime_feature_flags(
    source_session: Session,
    target_session: Session,
    *,
    source_workspace_id,
    target_workspace_id,
    dry_run: bool,
) -> TableSyncSummary:
    fields = ["enabled", "description", "stage_hint", "updated_at"]
    summary = TableSyncSummary()
    source_rows = list(
        source_session.exec(
            select(RuntimeFeatureFlagRecord).where(RuntimeFeatureFlagRecord.workspace_id == source_workspace_id)
        ).all()
    )
    target_rows = {
        row.flag_key: row
        for row in target_session.exec(
            select(RuntimeFeatureFlagRecord).where(RuntimeFeatureFlagRecord.workspace_id == target_workspace_id)
        ).all()
    }

    for source_row in source_rows:
        target_row = target_rows.get(source_row.flag_key)
        if target_row is None:
            summary.created += 1
            if not dry_run:
                target_session.add(
                    RuntimeFeatureFlagRecord(
                        id=source_row.id,
                        workspace_id=target_workspace_id,
                        flag_key=source_row.flag_key,
                        **_payload(source_row, fields),
                    )
                )
            continue
        if _apply_fields(target_row, source_row, fields):
            summary.updated += 1
        else:
            summary.unchanged += 1

    summary.extras = max(len(target_rows) - len(source_rows), 0)
    if not dry_run:
        target_session.commit()
    return summary


def _sync_workflow_templates(
    source_session: Session,
    target_session: Session,
    *,
    source_workspace_id,
    target_workspace_id,
    dry_run: bool,
) -> TableSyncSummary:
    fields = [
        "label",
        "summary",
        "architecture_scope",
        "supports_approvals",
        "supports_handoffs",
        "workflow_profile",
        "governance_hints",
        "is_active",
        "updated_at",
    ]
    summary = TableSyncSummary()
    source_rows = list(
        source_session.exec(
            select(WorkflowTemplateRecord).where(WorkflowTemplateRecord.workspace_id == source_workspace_id)
        ).all()
    )
    target_rows = {
        row.template_key: row
        for row in target_session.exec(
            select(WorkflowTemplateRecord).where(WorkflowTemplateRecord.workspace_id == target_workspace_id)
        ).all()
    }

    for source_row in source_rows:
        target_row = target_rows.get(source_row.template_key)
        if target_row is None:
            summary.created += 1
            if not dry_run:
                target_session.add(
                    WorkflowTemplateRecord(
                        id=source_row.id,
                        workspace_id=target_workspace_id,
                        template_key=source_row.template_key,
                        **_payload(source_row, fields),
                    )
                )
            continue
        if _apply_fields(target_row, source_row, fields):
            summary.updated += 1
        else:
            summary.unchanged += 1

    summary.extras = max(len(target_rows) - len(source_rows), 0)
    if not dry_run:
        target_session.commit()
    return summary


def _sync_governance_policies(
    source_session: Session,
    target_session: Session,
    *,
    source_workspace_id,
    target_workspace_id,
    dry_run: bool,
) -> TableSyncSummary:
    fields = ["label", "summary", "scope", "is_active", "policy_payload", "updated_at"]
    summary = TableSyncSummary()
    source_rows = list(
        source_session.exec(
            select(GovernancePolicyRecord).where(GovernancePolicyRecord.workspace_id == source_workspace_id)
        ).all()
    )
    target_rows = {
        row.policy_key: row
        for row in target_session.exec(
            select(GovernancePolicyRecord).where(GovernancePolicyRecord.workspace_id == target_workspace_id)
        ).all()
    }

    for source_row in source_rows:
        target_row = target_rows.get(source_row.policy_key)
        if target_row is None:
            summary.created += 1
            if not dry_run:
                target_session.add(
                    GovernancePolicyRecord(
                        id=source_row.id,
                        workspace_id=target_workspace_id,
                        policy_key=source_row.policy_key,
                        **_payload(source_row, fields),
                    )
                )
            continue
        if _apply_fields(target_row, source_row, fields):
            summary.updated += 1
        else:
            summary.unchanged += 1

    summary.extras = max(len(target_rows) - len(source_rows), 0)
    if not dry_run:
        target_session.commit()
    return summary


def _sync_workspace_runtime_settings(
    source_session: Session,
    target_session: Session,
    *,
    source_workspace_id,
    target_workspace_id,
    dry_run: bool,
) -> TableSyncSummary:
    summary = TableSyncSummary()
    source_record = load_workspace_runtime_settings(source_session, source_workspace_id)
    target_record = load_workspace_runtime_settings(target_session, target_workspace_id)
    if source_record is None:
        if target_record is None:
            summary.unchanged = 1
            return summary
        summary.updated = 1
        if not dry_run:
            reset_workspace_runtime_settings(
                target_session,
                target_workspace_id,
                actor_user_id=None,
                mirror_legacy_runtime=False,
            )
        return summary

    source_runtime = load_effective_runtime_settings(source_session, source_workspace_id)
    target_runtime = load_effective_runtime_settings(target_session, target_workspace_id)
    if _runtime_signature(source_runtime) == _runtime_signature(target_runtime):
        summary.unchanged = 1
        return summary

    summary.updated = 1
    if not dry_run:
        persist_workspace_runtime_settings(
            target_session,
            target_workspace_id,
            _runtime_update_request(source_runtime),
            actor_user_id=None,
            mirror_legacy_runtime=False,
        )
    return summary


def _encrypt_secret_for_target(secret_value: str, *, target_database_url: str, args: argparse.Namespace) -> str:
    overrides: dict[str, str | None] = {
        "DATABASE_URL": target_database_url,
    }
    if args.target_runtime_secrets_master_key:
        overrides["RUNTIME_SECRETS_MASTER_KEY"] = args.target_runtime_secrets_master_key
    else:
        overrides["RUNTIME_SECRETS_MASTER_KEY"] = None
        overrides["LOCAL_ADMIN_EMAIL"] = args.target_local_admin_email
        overrides["LOCAL_ADMIN_PASSWORD"] = args.target_local_admin_password
    with _temporary_settings_env(**overrides):
        return _encrypt_secret_value(secret_value)


def _sync_workspace_secrets(
    source_session: Session,
    target_session: Session,
    *,
    source_workspace_id,
    target_workspace_id,
    target_database_url: str,
    args: argparse.Namespace,
    dry_run: bool,
) -> TableSyncSummary:
    summary = TableSyncSummary()
    source_rows = list(
        source_session.exec(
            select(WorkspaceProviderSecretRecord).where(WorkspaceProviderSecretRecord.workspace_id == source_workspace_id)
        ).all()
    )
    if not source_rows:
        summary.unchanged = 1
        return summary

    target_rows = {
        (row.provider_key, row.secret_kind): row
        for row in target_session.exec(
            select(WorkspaceProviderSecretRecord).where(WorkspaceProviderSecretRecord.workspace_id == target_workspace_id)
        ).all()
    }

    for source_row in source_rows:
        key = (source_row.provider_key, source_row.secret_kind)
        target_row = target_rows.get(key)
        secret_ciphertext = ""
        secret_ref = source_row.secret_ref
        if not secret_ref:
            secret_value = resolve_workspace_provider_secret_value(
                source_session,
                source_workspace_id,
                source_row.provider_key,
                secret_kind=source_row.secret_kind,
            )
            if not secret_value:
                summary.skipped += 1
                summary.notes.append(
                    f"No se pudo desencriptar el secreto {source_row.provider_key.value}:{source_row.secret_kind} "
                    f"del workspace {source_workspace_id}."
                )
                continue
            secret_ciphertext = _encrypt_secret_for_target(
                secret_value,
                target_database_url=target_database_url,
                args=args,
            )

        if target_row is None:
            summary.created += 1
            if not dry_run:
                target_session.add(
                    WorkspaceProviderSecretRecord(
                        workspace_id=target_workspace_id,
                        provider_key=source_row.provider_key,
                        secret_kind=source_row.secret_kind,
                        secret_ciphertext=secret_ciphertext,
                        secret_ref=secret_ref,
                        status=RuntimeSecretStatus.configured,
                        last_rotated_at=source_row.last_rotated_at,
                        updated_by_user_id=None,
                        created_at=source_row.created_at,
                        updated_at=source_row.updated_at,
                    )
                )
            continue

        changed = False
        for field_name, value in {
            "secret_ciphertext": secret_ciphertext,
            "secret_ref": secret_ref,
            "status": RuntimeSecretStatus.configured,
            "last_rotated_at": source_row.last_rotated_at,
            "updated_at": source_row.updated_at,
            "updated_by_user_id": None,
        }.items():
            if getattr(target_row, field_name) != value:
                setattr(target_row, field_name, value)
                changed = True
        if changed:
            summary.updated += 1
        else:
            summary.unchanged += 1

    summary.extras = max(len(target_rows) - len(source_rows), 0)
    if not dry_run:
        target_session.commit()
    return summary


def _knowledge_sync(
    target_session: Session,
    *,
    dry_run: bool,
    force: bool,
) -> dict[str, Any]:
    if dry_run:
        return {
            "planned": True,
            "scope": KnowledgeScope.platform.value,
            "force": force,
        }
    report = KnowledgeMemoryService().sync_docs_corpus(
        target_session,
        scope=KnowledgeScope.platform,
        force=force,
    )
    return {
        "planned": False,
        "scope": report.scope.value,
        "document_count": report.document_count,
        "changed_document_count": report.changed_document_count,
        "filesystem_manifest_path": report.filesystem_manifest_path,
    }


def main() -> int:
    args = parse_args()
    if not args.target_database_url:
        raise SystemExit("Debes indicar --target-database-url o definir TARGET_DATABASE_URL.")
    _validate_secret_sync_context(args)
    target_database_url = _database_url_with_hostaddr(args.target_database_url)

    source_identity = _normalized_database_identity(args.source_database_url)
    target_identity = _normalized_database_identity(target_database_url)
    if source_identity == target_identity:
        raise SystemExit("La base fuente y la base objetivo resuelven a la misma identidad. Aborto por seguridad.")

    if not args.dry_run and not args.skip_alembic:
        _upgrade_target_schema(target_database_url)

    source_engine = create_engine(args.source_database_url, pool_pre_ping=True)
    target_engine = create_engine(target_database_url, pool_pre_ping=True)

    workspace_filter = {item.strip().lower() for item in args.workspace_slug if item.strip()}
    workspace_maps = _parse_workspace_maps(args.workspace_map)
    summary: dict[str, Any] = {
        "ok": True,
        "dry_run": args.dry_run,
        "source": _descriptor(args.source_database_url),
        "target": _descriptor(target_database_url),
        "steps": {},
        "warnings": [],
    }

    with Session(source_engine) as source_session, Session(target_engine) as target_session:
        if not args.dry_run:
            bootstrap_application_data(target_session)
            target_session.commit()
        summary["steps"]["target_bootstrap"] = {
            "executed": not args.dry_run,
            "skip_alembic": args.skip_alembic,
        }

        summary["steps"]["platform_runtime_providers"] = asdict(
            _sync_platform_runtime_providers(source_session, target_session, dry_run=args.dry_run)
        )
        summary["steps"]["platform_runtime_defaults"] = asdict(
            _sync_platform_runtime_defaults(source_session, target_session, dry_run=args.dry_run)
        )
        summary["steps"]["runtime_catalog_entries"] = asdict(
            _sync_runtime_catalog_entries(source_session, target_session, dry_run=args.dry_run)
        )
        summary["steps"]["commercial_seed"] = {
            "product_catalog": asdict(_sync_product_catalog(source_session, target_session, dry_run=args.dry_run)),
            "product_prices": asdict(_sync_product_prices(source_session, target_session, dry_run=args.dry_run)),
            "quota_product_configs": asdict(
                _sync_quota_product_configs(source_session, target_session, dry_run=args.dry_run)
            ),
            "package_catalog": asdict(_sync_package_catalog(source_session, target_session, dry_run=args.dry_run)),
        }
        summary["steps"]["knowledge"] = _knowledge_sync(
            target_session,
            dry_run=args.dry_run,
            force=args.force_knowledge,
        )

        source_workspaces = list(
            source_session.exec(
                select(WorkspaceRecord).where(WorkspaceRecord.is_active == True).order_by(WorkspaceRecord.slug.asc())  # noqa: E712
            ).all()
        )
        if workspace_filter:
            source_workspaces = [row for row in source_workspaces if row.slug.strip().lower() in workspace_filter]

        target_workspaces_by_slug = {
            row.slug.strip().lower(): row
            for row in target_session.exec(
                select(WorkspaceRecord).where(WorkspaceRecord.is_active == True)  # noqa: E712
            ).all()
        }

        workspace_steps: list[dict[str, Any]] = []
        for source_workspace in source_workspaces:
            normalized_source_slug = source_workspace.slug.strip().lower()
            mapped_target_slugs = workspace_maps.get(normalized_source_slug, [normalized_source_slug])
            resolved_targets = [
                target_workspaces_by_slug[target_slug]
                for target_slug in mapped_target_slugs
                if target_slug in target_workspaces_by_slug
            ]
            if not resolved_targets:
                workspace_steps.append(
                    {
                        "workspace_slug": source_workspace.slug,
                        "workspace_name": source_workspace.name,
                        "status": "skipped_missing_target_workspace",
                    }
                )
                continue

            source_runtime = load_effective_runtime_settings(source_session, source_workspace.id)
            for target_workspace in resolved_targets:
                if not args.dry_run:
                    apply_workspace_bootstrap(target_session, target_workspace.id)

                runtime_summary = _sync_workspace_runtime_settings(
                    source_session,
                    target_session,
                    source_workspace_id=source_workspace.id,
                    target_workspace_id=target_workspace.id,
                    dry_run=args.dry_run,
                )
                feature_flag_summary = _sync_runtime_feature_flags(
                    source_session,
                    target_session,
                    source_workspace_id=source_workspace.id,
                    target_workspace_id=target_workspace.id,
                    dry_run=args.dry_run,
                )
                workflow_summary = _sync_workflow_templates(
                    source_session,
                    target_session,
                    source_workspace_id=source_workspace.id,
                    target_workspace_id=target_workspace.id,
                    dry_run=args.dry_run,
                )
                governance_summary = _sync_governance_policies(
                    source_session,
                    target_session,
                    source_workspace_id=source_workspace.id,
                    target_workspace_id=target_workspace.id,
                    dry_run=args.dry_run,
                )

                workspace_step: dict[str, Any] = {
                    "workspace_slug": source_workspace.slug,
                    "workspace_name": source_workspace.name,
                    "target_workspace_slug": target_workspace.slug,
                    "target_workspace_name": target_workspace.name,
                    "target_workspace_id": str(target_workspace.id),
                    "runtime_settings": asdict(runtime_summary),
                    "feature_flags": asdict(feature_flag_summary),
                    "workflow_templates": asdict(workflow_summary),
                    "governance_policies": asdict(governance_summary),
                }

                if args.sync_workspace_secrets:
                    secret_summary = _sync_workspace_secrets(
                        source_session,
                        target_session,
                        source_workspace_id=source_workspace.id,
                        target_workspace_id=target_workspace.id,
                        target_database_url=target_database_url,
                        args=args,
                        dry_run=args.dry_run,
                    )
                    workspace_step["workspace_secrets"] = asdict(secret_summary)
                elif (
                    not source_runtime.uses_platform_credentials
                    and source_runtime.active_provider in {LLMProviderKey.openai, LLMProviderKey.deepseek}
                ):
                    warning = (
                        f"Workspace {source_workspace.slug} usa credenciales propias para "
                        f"{source_runtime.active_provider.value}; rota o resincroniza el secreto en el target "
                        f"{target_workspace.slug}."
                    )
                    workspace_step["workspace_secrets"] = {
                        "created": 0,
                        "updated": 0,
                        "unchanged": 0,
                        "skipped": 1,
                        "extras": 0,
                        "notes": [warning],
                    }
                    summary["warnings"].append(warning)

                workspace_steps.append(workspace_step)

        summary["steps"]["workspace_scope"] = workspace_steps

    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        summary["report_path"] = str(report_path)
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
