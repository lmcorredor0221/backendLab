from collections.abc import Generator

from sqlalchemy import inspect, text
from sqlmodel import Session, SQLModel, create_engine

from app.core.config import (
    get_settings,
    knowledge_repo_autosync_enabled,
    runtime_bootstrap_enabled,
    should_auto_create_schema,
)
from app.services.alembic_runtime_guard import assert_alembic_head_applied
from app.services.knowledge_memory import KnowledgeMemoryService
from app.services.knowledge_memory_migration import apply_knowledge_memory_governance_migration
from app.diagnostics import LEGACY_AUTONOMY_LEVEL_MAP, LEGACY_CASE_TYPE_MAP
from app.services.session_migration import apply_session_contract_migration


settings = get_settings()
engine = create_engine(settings.database_url, echo=settings.app_debug, pool_pre_ping=True)


def _json_default_literal(dialect_name: str) -> str:
    if dialect_name == "postgresql":
        return "'{}'::json"
    return "'{}'"


def _uuid_column_ddl(dialect_name: str) -> str:
    if dialect_name == "postgresql":
        return "UUID"
    return "TEXT"


def ensure_runtime_schema() -> None:
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    json_default = _json_default_literal(engine.dialect.name)
    uuid_default = _uuid_column_ddl(engine.dialect.name)
    required_columns = {
        "sessions": {
            "workspace_id": uuid_default,
            "commercial_tier": "TEXT NOT NULL DEFAULT 'blueprint'",
            "selected_workflow_template_key": "TEXT NOT NULL DEFAULT ''",
            "suggested_title": "TEXT",
            "title_source": "TEXT NOT NULL DEFAULT 'migrated'",
            "row_version": "INTEGER NOT NULL DEFAULT 1",
            "archived_at": "TIMESTAMP",
            "archived_by_user_id": uuid_default,
            "deleted_at": "TIMESTAMP",
            "deleted_by_user_id": uuid_default,
        },
        "users": {
            "default_workspace_id": uuid_default,
        },
        "runtime_feature_flags": {
            "workspace_id": uuid_default,
        },
        "workflow_templates": {
            "workspace_id": uuid_default,
        },
        "governance_policies": {
            "workspace_id": uuid_default,
        },
        "opportunities": {
            "operational_baseline": f"JSON NOT NULL DEFAULT {json_default}",
            "mvp_definition": f"JSON NOT NULL DEFAULT {json_default}",
        },
        "canvases": {
            "agent_profile": f"JSON NOT NULL DEFAULT {json_default}",
        },
        "blueprints": {
            "delivery_package": f"JSON NOT NULL DEFAULT {json_default}",
            "knowledge_profile": f"JSON NOT NULL DEFAULT {json_default}",
        },
    }

    with engine.begin() as connection:
        for table_name, columns in required_columns.items():
            if table_name not in existing_tables:
                continue
            current_columns = {column["name"] for column in inspector.get_columns(table_name)}
            for column_name, ddl in columns.items():
                if column_name in current_columns:
                    continue
                connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {ddl}"))

        if "opportunities" in existing_tables:
            for legacy_value, canonical_value in LEGACY_AUTONOMY_LEVEL_MAP.items():
                if legacy_value == canonical_value:
                    continue
                connection.execute(
                    text(
                        "UPDATE opportunities SET autonomy_level = :canonical_value "
                        "WHERE autonomy_level = :legacy_value"
                    ),
                    {
                        "canonical_value": canonical_value,
                        "legacy_value": legacy_value,
                    },
                )
            for legacy_value, canonical_value in LEGACY_CASE_TYPE_MAP.items():
                if legacy_value == canonical_value:
                    continue
                connection.execute(
                    text(
                        "UPDATE opportunities SET case_type = :canonical_value "
                        "WHERE case_type = :legacy_value"
                    ),
                    {
                        "canonical_value": canonical_value,
                        "legacy_value": legacy_value,
                    },
                )


def bootstrap_application_data(session: Session) -> None:
    from app.services.deliverable_catalog import persistence as _deliverable_catalog_persistence  # noqa: F401
    from app.services.product_processing import persistence as _product_processing_persistence  # noqa: F401
    from app.services.journey_stage_migration import JourneyStageMigrationService
    from app.services.llm_runtime.settings_migration import apply_runtime_llm_multitenant_migration
    from app.services.runtime_governance_bootstrap import backfill_platform_runtime_governance
    from app.services.workspace_access import backfill_user_default_workspaces
    from app.services.workspace_schema_migration import (
        apply_workspace_scoped_legacy_backfill,
        apply_workspace_uuid_column_migration,
    )

    apply_workspace_uuid_column_migration(session)
    backfill_user_default_workspaces(session)
    apply_workspace_scoped_legacy_backfill(session)
    backfill_platform_runtime_governance(session)
    apply_runtime_llm_multitenant_migration(session)
    apply_session_contract_migration(session)
    JourneyStageMigrationService().apply(session)
    apply_knowledge_memory_governance_migration(session)
    from app.services.commerce_service import backfill_legacy_entitlements, ensure_commercial_seed

    ensure_commercial_seed(session)
    backfill_legacy_entitlements(session)
    if knowledge_repo_autosync_enabled(settings):
        KnowledgeMemoryService().ensure_repo_docs_ingested(session)


def create_db_and_tables() -> None:
    if should_auto_create_schema(settings):
        SQLModel.metadata.create_all(engine)
        ensure_runtime_schema()
    else:
        assert_alembic_head_applied(engine)
    if not runtime_bootstrap_enabled(settings):
        return
    with Session(engine) as session:
        bootstrap_application_data(session)


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session
