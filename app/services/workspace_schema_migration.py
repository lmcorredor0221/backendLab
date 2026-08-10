from __future__ import annotations

from dataclasses import asdict, dataclass, field
from uuid import UUID, uuid4

from sqlalchemy import inspect, text
from sqlmodel import Session, select

from app.models import (
    GovernancePolicyRecord,
    RuntimeFeatureFlagRecord,
    SchemaMigrationRecord,
    SessionRecord,
    UserRecord,
    WorkflowTemplateRecord,
    WorkspaceRecord,
    utc_now,
)


MIGRATION_KEY_WORKSPACE_UUID_COLUMNS = "2026-07-20-workspace-uuid-column-normalization"
MIGRATION_KEY_WORKSPACE_SCOPED_BACKFILL = "2026-07-20-workspace-scoped-legacy-backfill"
WORKSPACE_UUID_COLUMNS: tuple[tuple[str, str], ...] = (
    ("users", "default_workspace_id"),
    ("sessions", "workspace_id"),
    ("runtime_feature_flags", "workspace_id"),
    ("workflow_templates", "workspace_id"),
    ("governance_policies", "workspace_id"),
)


@dataclass
class WorkspaceUuidColumnMigrationSummary:
    migration_key: str = MIGRATION_KEY_WORKSPACE_UUID_COLUMNS
    already_recorded: bool = False
    skipped: bool = False
    converted_columns: list[str] = field(default_factory=list)
    already_uuid_columns: list[str] = field(default_factory=list)
    blocked_columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class WorkspaceScopedBackfillMigrationSummary:
    migration_key: str = MIGRATION_KEY_WORKSPACE_SCOPED_BACKFILL
    already_recorded: bool = False
    skipped: bool = False
    legacy_sessions_updated: int = 0
    legacy_feature_flags_cloned: int = 0
    legacy_feature_flags_deleted: int = 0
    legacy_workflow_templates_cloned: int = 0
    legacy_workflow_templates_deleted: int = 0
    legacy_governance_policies_cloned: int = 0
    legacy_governance_policies_deleted: int = 0
    dropped_indexes: list[str] = field(default_factory=list)
    created_indexes: list[str] = field(default_factory=list)
    created_constraints: list[str] = field(default_factory=list)
    enforced_not_null: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _column_data_type(session: Session, *, table_name: str, column_name: str) -> str | None:
    row = session.execute(
        text(
            """
            select data_type
            from information_schema.columns
            where table_schema = 'public'
              and table_name = :table_name
              and column_name = :column_name
            """
        ),
        {
            "table_name": table_name,
            "column_name": column_name,
        },
    ).first()
    if row is None:
        return None
    return str(row[0])


def _invalid_uuid_count(session: Session, *, table_name: str, column_name: str) -> int:
    row = session.execute(
        text(
            f"""
            select count(*) as invalid_count
            from {table_name}
            where {column_name} is not null
              and {column_name} <> ''
              and {column_name} !~* '^[0-9a-f]{{8}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{12}}$'
            """
        )
    ).first()
    if row is None:
        return 0
    return int(row[0])


def _coerce_uuid(value: UUID | str | None) -> UUID | None:
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            return UUID(candidate)
        except ValueError:
            return None
    return None


def _constraint_exists(session: Session, *, table_name: str, constraint_name: str) -> bool:
    row = session.execute(
        text(
            """
            select 1
            from pg_constraint c
            join pg_class t on c.conrelid = t.oid
            join pg_namespace n on t.relnamespace = n.oid
            where n.nspname = 'public'
              and t.relname = :table_name
              and c.conname = :constraint_name
            """
        ),
        {
            "table_name": table_name,
            "constraint_name": constraint_name,
        },
    ).first()
    return row is not None


def _index_definition(session: Session, *, table_name: str, index_name: str) -> str | None:
    row = session.execute(
        text(
            """
            select indexdef
            from pg_indexes
            where schemaname = 'public'
              and tablename = :table_name
              and indexname = :index_name
            """
        ),
        {
            "table_name": table_name,
            "index_name": index_name,
        },
    ).first()
    if row is None:
        return None
    return str(row[0])


def _drop_index_if_present(session: Session, *, index_name: str, summary: WorkspaceScopedBackfillMigrationSummary) -> None:
    session.execute(text(f"DROP INDEX IF EXISTS {index_name}"))
    summary.dropped_indexes.append(index_name)


def _create_index_if_missing(
    session: Session,
    *,
    table_name: str,
    index_name: str,
    create_sql: str,
    summary: WorkspaceScopedBackfillMigrationSummary,
) -> None:
    if _index_definition(session, table_name=table_name, index_name=index_name) is not None:
        return
    session.execute(text(create_sql))
    summary.created_indexes.append(index_name)


def _create_constraint_if_missing(
    session: Session,
    *,
    table_name: str,
    constraint_name: str,
    create_sql: str,
    summary: WorkspaceScopedBackfillMigrationSummary,
) -> None:
    if _constraint_exists(session, table_name=table_name, constraint_name=constraint_name):
        return
    session.execute(text(create_sql))
    summary.created_constraints.append(constraint_name)


def _null_count(session: Session, *, table_name: str, column_name: str) -> int:
    row = session.execute(text(f"select count(*) from {table_name} where {column_name} is null")).first()
    if row is None:
        return 0
    return int(row[0])


def _set_not_null_if_safe(
    session: Session,
    *,
    table_name: str,
    column_name: str,
    summary: WorkspaceScopedBackfillMigrationSummary,
) -> None:
    if _null_count(session, table_name=table_name, column_name=column_name) > 0:
        summary.notes.append(f"{table_name}.{column_name} conserva valores nulos y no se marco NOT NULL.")
        return
    session.execute(text(f"ALTER TABLE {table_name} ALTER COLUMN {column_name} SET NOT NULL"))
    summary.enforced_not_null.append(f"{table_name}.{column_name}")


def _prepare_workspace_scoped_indexes(session: Session, summary: WorkspaceScopedBackfillMigrationSummary) -> None:
    legacy_unique_indexes = (
        ("runtime_feature_flags", "ix_runtime_feature_flags_flag_key"),
        ("workflow_templates", "ix_workflow_templates_template_key"),
        ("governance_policies", "ix_governance_policies_policy_key"),
    )
    for table_name, index_name in legacy_unique_indexes:
        definition = _index_definition(session, table_name=table_name, index_name=index_name)
        if definition and "CREATE UNIQUE INDEX" in definition.upper():
            _drop_index_if_present(session, index_name=index_name, summary=summary)


def _clone_legacy_feature_flags(session: Session, summary: WorkspaceScopedBackfillMigrationSummary) -> None:
    workspaces = session.exec(select(WorkspaceRecord).where(WorkspaceRecord.is_active == True)).all()  # noqa: E712
    legacy_rows = session.exec(
        select(RuntimeFeatureFlagRecord).where(RuntimeFeatureFlagRecord.workspace_id == None)  # noqa: E711
    ).all()
    if not legacy_rows or not workspaces:
        return

    existing_keys = {
        (str(item.workspace_id), item.flag_key)
        for item in session.exec(
            select(RuntimeFeatureFlagRecord).where(RuntimeFeatureFlagRecord.workspace_id != None)  # noqa: E711
        ).all()
    }
    for workspace in workspaces:
        for row in legacy_rows:
            tuple_key = (str(workspace.id), row.flag_key)
            if tuple_key in existing_keys:
                continue
            session.add(
                RuntimeFeatureFlagRecord(
                    id=uuid4(),
                    workspace_id=workspace.id,
                    flag_key=row.flag_key,
                    enabled=row.enabled,
                    description=row.description,
                    stage_hint=row.stage_hint,
                    updated_at=row.updated_at or utc_now(),
                )
            )
            existing_keys.add(tuple_key)
            summary.legacy_feature_flags_cloned += 1
    for row in legacy_rows:
        session.delete(row)
        summary.legacy_feature_flags_deleted += 1


def _clone_legacy_workflow_templates(session: Session, summary: WorkspaceScopedBackfillMigrationSummary) -> None:
    workspaces = session.exec(select(WorkspaceRecord).where(WorkspaceRecord.is_active == True)).all()  # noqa: E712
    legacy_rows = session.exec(
        select(WorkflowTemplateRecord).where(WorkflowTemplateRecord.workspace_id == None)  # noqa: E711
    ).all()
    if not legacy_rows or not workspaces:
        return

    existing_keys = {
        (str(item.workspace_id), item.template_key)
        for item in session.exec(
            select(WorkflowTemplateRecord).where(WorkflowTemplateRecord.workspace_id != None)  # noqa: E711
        ).all()
    }
    for workspace in workspaces:
        for row in legacy_rows:
            tuple_key = (str(workspace.id), row.template_key)
            if tuple_key in existing_keys:
                continue
            session.add(
                WorkflowTemplateRecord(
                    id=uuid4(),
                    workspace_id=workspace.id,
                    template_key=row.template_key,
                    label=row.label,
                    summary=row.summary,
                    architecture_scope=list(row.architecture_scope),
                    supports_approvals=row.supports_approvals,
                    supports_handoffs=row.supports_handoffs,
                    workflow_profile=dict(row.workflow_profile),
                    governance_hints=list(row.governance_hints),
                    is_active=row.is_active,
                    updated_at=row.updated_at or utc_now(),
                )
            )
            existing_keys.add(tuple_key)
            summary.legacy_workflow_templates_cloned += 1
    for row in legacy_rows:
        session.delete(row)
        summary.legacy_workflow_templates_deleted += 1


def _clone_legacy_governance_policies(session: Session, summary: WorkspaceScopedBackfillMigrationSummary) -> None:
    workspaces = session.exec(select(WorkspaceRecord).where(WorkspaceRecord.is_active == True)).all()  # noqa: E712
    legacy_rows = session.exec(
        select(GovernancePolicyRecord).where(GovernancePolicyRecord.workspace_id == None)  # noqa: E711
    ).all()
    if not legacy_rows or not workspaces:
        return

    existing_keys = {
        (str(item.workspace_id), item.policy_key)
        for item in session.exec(
            select(GovernancePolicyRecord).where(GovernancePolicyRecord.workspace_id != None)  # noqa: E711
        ).all()
    }
    for workspace in workspaces:
        for row in legacy_rows:
            tuple_key = (str(workspace.id), row.policy_key)
            if tuple_key in existing_keys:
                continue
            session.add(
                GovernancePolicyRecord(
                    id=uuid4(),
                    workspace_id=workspace.id,
                    policy_key=row.policy_key,
                    label=row.label,
                    summary=row.summary,
                    scope=row.scope,
                    is_active=row.is_active,
                    policy_payload=dict(row.policy_payload),
                    updated_at=row.updated_at or utc_now(),
                )
            )
            existing_keys.add(tuple_key)
            summary.legacy_governance_policies_cloned += 1
    for row in legacy_rows:
        session.delete(row)
        summary.legacy_governance_policies_deleted += 1


def _backfill_legacy_sessions(session: Session, summary: WorkspaceScopedBackfillMigrationSummary) -> None:
    records = session.exec(select(SessionRecord).where(SessionRecord.workspace_id == None)).all()  # noqa: E711
    for record in records:
        user = session.get(UserRecord, record.user_id)
        workspace_id = _coerce_uuid(getattr(user, "default_workspace_id", None)) if user is not None else None
        if workspace_id is None:
            summary.notes.append(f"sessions.{record.id} no pudo backfillearse porque el usuario no tiene default_workspace_id.")
            continue
        record.workspace_id = workspace_id
        session.add(record)
        summary.legacy_sessions_updated += 1


def _finalize_workspace_scoped_indexes(session: Session, summary: WorkspaceScopedBackfillMigrationSummary) -> None:
    index_statements = (
        (
            "runtime_feature_flags",
            "ix_runtime_feature_flags_flag_key",
            "CREATE INDEX ix_runtime_feature_flags_flag_key ON runtime_feature_flags (flag_key)",
        ),
        (
            "runtime_feature_flags",
            "ix_runtime_feature_flags_workspace_id",
            "CREATE INDEX ix_runtime_feature_flags_workspace_id ON runtime_feature_flags (workspace_id)",
        ),
        (
            "workflow_templates",
            "ix_workflow_templates_template_key",
            "CREATE INDEX ix_workflow_templates_template_key ON workflow_templates (template_key)",
        ),
        (
            "workflow_templates",
            "ix_workflow_templates_workspace_id",
            "CREATE INDEX ix_workflow_templates_workspace_id ON workflow_templates (workspace_id)",
        ),
        (
            "governance_policies",
            "ix_governance_policies_policy_key",
            "CREATE INDEX ix_governance_policies_policy_key ON governance_policies (policy_key)",
        ),
        (
            "governance_policies",
            "ix_governance_policies_workspace_id",
            "CREATE INDEX ix_governance_policies_workspace_id ON governance_policies (workspace_id)",
        ),
        (
            "sessions",
            "ix_sessions_workspace_id",
            "CREATE INDEX ix_sessions_workspace_id ON sessions (workspace_id)",
        ),
    )
    for table_name, index_name, create_sql in index_statements:
        _create_index_if_missing(
            session,
            table_name=table_name,
            index_name=index_name,
            create_sql=create_sql,
            summary=summary,
        )

    constraint_statements = (
        (
            "runtime_feature_flags",
            "uq_runtime_feature_flag_workspace",
            "ALTER TABLE runtime_feature_flags ADD CONSTRAINT uq_runtime_feature_flag_workspace UNIQUE (workspace_id, flag_key)",
        ),
        (
            "workflow_templates",
            "uq_workflow_template_workspace",
            "ALTER TABLE workflow_templates ADD CONSTRAINT uq_workflow_template_workspace UNIQUE (workspace_id, template_key)",
        ),
        (
            "governance_policies",
            "uq_governance_policy_workspace",
            "ALTER TABLE governance_policies ADD CONSTRAINT uq_governance_policy_workspace UNIQUE (workspace_id, policy_key)",
        ),
    )
    for table_name, constraint_name, create_sql in constraint_statements:
        _create_constraint_if_missing(
            session,
            table_name=table_name,
            constraint_name=constraint_name,
            create_sql=create_sql,
            summary=summary,
        )

    _set_not_null_if_safe(session, table_name="runtime_feature_flags", column_name="workspace_id", summary=summary)
    _set_not_null_if_safe(session, table_name="workflow_templates", column_name="workspace_id", summary=summary)
    _set_not_null_if_safe(session, table_name="governance_policies", column_name="workspace_id", summary=summary)
    _set_not_null_if_safe(session, table_name="sessions", column_name="workspace_id", summary=summary)


def apply_workspace_uuid_column_migration(session: Session) -> WorkspaceUuidColumnMigrationSummary:
    summary = WorkspaceUuidColumnMigrationSummary()
    bind = session.get_bind()
    if bind is None or bind.dialect.name != "postgresql":
        summary.skipped = True
        summary.notes.append("La migracion de columnas workspace_id/default_workspace_id solo aplica a PostgreSQL.")
        return summary

    migration_record = session.exec(
        select(SchemaMigrationRecord).where(
            SchemaMigrationRecord.migration_key == MIGRATION_KEY_WORKSPACE_UUID_COLUMNS
        )
    ).first()
    if migration_record is not None:
        summary.already_recorded = True

    inspector = inspect(bind)
    existing_tables = set(inspector.get_table_names())
    dirty = False

    for table_name, column_name in WORKSPACE_UUID_COLUMNS:
        if table_name not in existing_tables:
            summary.notes.append(f"{table_name}.{column_name} no existe en la base actual.")
            continue

        qualified_name = f"{table_name}.{column_name}"
        data_type = _column_data_type(session, table_name=table_name, column_name=column_name)
        if data_type is None:
            summary.notes.append(f"No fue posible determinar el tipo de {qualified_name}.")
            continue
        if data_type == "uuid":
            summary.already_uuid_columns.append(qualified_name)
            continue
        if data_type not in {"text", "character varying"}:
            summary.blocked_columns.append(qualified_name)
            summary.notes.append(f"{qualified_name} usa tipo inesperado '{data_type}' y no se altero automaticamente.")
            continue

        invalid_count = _invalid_uuid_count(session, table_name=table_name, column_name=column_name)
        if invalid_count > 0:
            summary.blocked_columns.append(qualified_name)
            summary.notes.append(
                f"{qualified_name} contiene {invalid_count} valores no convertibles a UUID; requiere limpieza manual."
            )
            continue

        session.exec(
            text(
                f"""
                ALTER TABLE {table_name}
                ALTER COLUMN {column_name}
                TYPE UUID
                USING NULLIF({column_name}, '')::uuid
                """
            )
        )
        summary.converted_columns.append(qualified_name)
        dirty = True

    if not summary.blocked_columns and migration_record is None:
        session.add(
            SchemaMigrationRecord(
                migration_key=MIGRATION_KEY_WORKSPACE_UUID_COLUMNS,
                description=(
                    "Convierte columnas heredadas workspace_id/default_workspace_id de TEXT a UUID en PostgreSQL "
                    "para restaurar la consistencia multitenant."
                ),
            )
        )
        dirty = True

    if dirty:
        session.commit()

    return summary


def apply_workspace_scoped_legacy_backfill(session: Session) -> WorkspaceScopedBackfillMigrationSummary:
    summary = WorkspaceScopedBackfillMigrationSummary()
    bind = session.get_bind()
    if bind is None or bind.dialect.name != "postgresql":
        summary.skipped = True
        summary.notes.append("El backfill de legado single-tenant a workspace-scope solo aplica a PostgreSQL.")
        return summary

    migration_record = session.exec(
        select(SchemaMigrationRecord).where(
            SchemaMigrationRecord.migration_key == MIGRATION_KEY_WORKSPACE_SCOPED_BACKFILL
        )
    ).first()
    if migration_record is not None:
        summary.already_recorded = True

    _prepare_workspace_scoped_indexes(session, summary)
    _backfill_legacy_sessions(session, summary)
    _clone_legacy_feature_flags(session, summary)
    _clone_legacy_workflow_templates(session, summary)
    _clone_legacy_governance_policies(session, summary)
    session.flush()
    _finalize_workspace_scoped_indexes(session, summary)

    if migration_record is None:
        session.add(
            SchemaMigrationRecord(
                migration_key=MIGRATION_KEY_WORKSPACE_SCOPED_BACKFILL,
                description=(
                    "Backfillea datos legado single-tenant con workspace_id nulo, corrige indices globales "
                    "y restituye constraints multitenant por workspace."
                ),
            )
        )
    session.commit()
    return summary
