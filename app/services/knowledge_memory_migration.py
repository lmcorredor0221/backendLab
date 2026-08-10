from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from sqlalchemy import inspect, text
from sqlmodel import Session, select

from app.models import (
    KnowledgeDocumentRecord,
    KnowledgeDocumentStatus,
    KnowledgeScope,
    KnowledgeSectionRecord,
    KnowledgeVisibility,
    SchemaMigrationRecord,
    utc_now,
)


MIGRATION_KEY_CI2 = "2026-07-22-ci2-knowledge-memory-governance"


def _uuid_ddl(dialect_name: str) -> str:
    return "UUID" if dialect_name == "postgresql" else "TEXT"


def _json_default(dialect_name: str, value: str) -> str:
    if dialect_name == "postgresql":
        return f"'{value}'::json"
    return f"'{value}'"


@dataclass
class KnowledgeMemoryGovernanceMigrationSummary:
    migration_key: str = MIGRATION_KEY_CI2
    already_recorded: bool = False
    columns_added: list[str] = field(default_factory=list)
    indexes_created: list[str] = field(default_factory=list)
    documents_backfilled: int = 0
    sections_backfilled: int = 0
    runs_backfilled: int = 0
    notes: list[str] = field(default_factory=list)
    report_generated_at: str = field(default_factory=lambda: utc_now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_knowledge_memory_governance_migration(session: Session) -> KnowledgeMemoryGovernanceMigrationSummary:
    bind = session.get_bind()
    inspector = inspect(bind)
    existing_tables = set(inspector.get_table_names())
    summary = KnowledgeMemoryGovernanceMigrationSummary()

    migration_record = session.exec(
        select(SchemaMigrationRecord).where(SchemaMigrationRecord.migration_key == MIGRATION_KEY_CI2)
    ).first()
    if migration_record is not None:
        summary.already_recorded = True

    if not {"knowledge_ingestion_runs", "knowledge_documents", "knowledge_sections"}.issubset(existing_tables):
        summary.notes.append("Las tablas de knowledge aun no existen; create_all las materializara antes del siguiente ciclo.")
        return summary

    dialect_name = bind.dialect.name
    required_columns = {
        "knowledge_ingestion_runs": {
            "scope": "TEXT NOT NULL DEFAULT 'platform'",
            "workspace_id": _uuid_ddl(dialect_name),
            "session_id": _uuid_ddl(dialect_name),
        },
        "knowledge_documents": {
            "scope": "TEXT NOT NULL DEFAULT 'platform'",
            "workspace_id": _uuid_ddl(dialect_name),
            "session_id": _uuid_ddl(dialect_name),
            "visibility": "TEXT NOT NULL DEFAULT 'platform'",
            "status": "TEXT NOT NULL DEFAULT 'approved'",
            "stage_affinity_text": "TEXT NOT NULL DEFAULT ''",
            "agent_affinity_text": "TEXT NOT NULL DEFAULT ''",
            "approved_by_user_id": _uuid_ddl(dialect_name),
            "approved_at": "TIMESTAMP",
            "effective_from": "TIMESTAMP",
            "expires_at": "TIMESTAMP",
            "supersedes_document_id": _uuid_ddl(dialect_name),
        },
        "knowledge_sections": {
            "source_root": "TEXT NOT NULL DEFAULT 'Docs'",
            "scope": "TEXT NOT NULL DEFAULT 'platform'",
            "workspace_id": _uuid_ddl(dialect_name),
            "session_id": _uuid_ddl(dialect_name),
            "visibility": "TEXT NOT NULL DEFAULT 'platform'",
            "status": "TEXT NOT NULL DEFAULT 'approved'",
            "authority_level": "TEXT NOT NULL DEFAULT ''",
            "memory_usage": "TEXT NOT NULL DEFAULT ''",
            "stage_affinity": f"JSON NOT NULL DEFAULT {_json_default(dialect_name, '[]')}",
            "stage_affinity_text": "TEXT NOT NULL DEFAULT ''",
            "agent_affinity": f"JSON NOT NULL DEFAULT {_json_default(dialect_name, '[]')}",
            "agent_affinity_text": "TEXT NOT NULL DEFAULT ''",
            "document_version_number": "INTEGER NOT NULL DEFAULT 1",
            "approved_by_user_id": _uuid_ddl(dialect_name),
            "approved_at": "TIMESTAMP",
            "effective_from": "TIMESTAMP",
            "expires_at": "TIMESTAMP",
        },
    }

    with bind.begin() as connection:
        for table_name, columns in required_columns.items():
            current_columns = {column["name"] for column in inspect(bind).get_columns(table_name)}
            for column_name, ddl in columns.items():
                if column_name in current_columns:
                    continue
                connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {ddl}"))
                summary.columns_added.append(f"{table_name}.{column_name}")

        index_statements = {
            "uq_knowledge_documents_scope_relative": (
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_knowledge_documents_scope_relative "
                "ON knowledge_documents("
                "scope, COALESCE(CAST(workspace_id AS TEXT), ''), COALESCE(CAST(session_id AS TEXT), ''), relative_path)"
            ),
            "uq_knowledge_sections_scope_key": (
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_knowledge_sections_scope_key "
                "ON knowledge_sections("
                "scope, COALESCE(CAST(workspace_id AS TEXT), ''), COALESCE(CAST(session_id AS TEXT), ''), section_key)"
            ),
            "ix_knowledge_sections_scope_visibility_status": (
                "CREATE INDEX IF NOT EXISTS ix_knowledge_sections_scope_visibility_status "
                "ON knowledge_sections(scope, workspace_id, session_id, visibility, status, authority_level, memory_usage)"
            ),
            "ix_knowledge_documents_scope_status": (
                "CREATE INDEX IF NOT EXISTS ix_knowledge_documents_scope_status "
                "ON knowledge_documents(scope, workspace_id, session_id, visibility, status)"
            ),
        }
        for index_name, statement in index_statements.items():
            connection.execute(text(statement))
            summary.indexes_created.append(index_name)

    now = utc_now()
    documents = session.exec(select(KnowledgeDocumentRecord)).all()
    for record in documents:
        if not str(record.scope or "").strip():
            record.scope = KnowledgeScope.platform
        if not str(record.visibility or "").strip():
            record.visibility = KnowledgeVisibility.platform
        if not str(record.status or "").strip():
            record.status = KnowledgeDocumentStatus.approved
        if not str(record.stage_affinity_text or "").strip():
            normalized_stage = [str(item).strip().lower() for item in record.stage_affinity if str(item).strip()]
            record.stage_affinity_text = "|" + "|".join(dict.fromkeys(normalized_stage)) + "|" if normalized_stage else ""
        if not str(record.agent_affinity_text or "").strip():
            normalized_agent = [str(item).strip().lower() for item in record.agent_affinity if str(item).strip()]
            record.agent_affinity_text = "|" + "|".join(dict.fromkeys(normalized_agent)) + "|" if normalized_agent else ""
        if record.effective_from is None:
            record.effective_from = record.updated_at or now
        if record.status != KnowledgeDocumentStatus.expired and record.expires_at is not None and record.expires_at <= now:
            record.status = KnowledgeDocumentStatus.expired
        session.add(record)
        summary.documents_backfilled += 1

    documents_by_id = {item.id: item for item in documents}
    sections = session.exec(select(KnowledgeSectionRecord)).all()
    for section in sections:
        document = documents_by_id.get(section.document_id)
        if document is None:
            continue
        section.source_root = document.source_root
        section.scope = document.scope
        section.workspace_id = document.workspace_id
        section.session_id = document.session_id
        section.visibility = document.visibility
        section.status = document.status
        section.authority_level = document.authority_level
        section.memory_usage = document.memory_usage
        section.stage_affinity = list(document.stage_affinity)
        section.stage_affinity_text = document.stage_affinity_text
        section.agent_affinity = list(document.agent_affinity)
        section.agent_affinity_text = document.agent_affinity_text
        section.document_version_number = document.version_number
        section.approved_by_user_id = document.approved_by_user_id
        section.approved_at = document.approved_at
        section.effective_from = document.effective_from
        section.expires_at = document.expires_at
        session.add(section)
        summary.sections_backfilled += 1

    runs_update = session.exec(
        text(
            "UPDATE knowledge_ingestion_runs "
            "SET scope = COALESCE(NULLIF(scope, ''), 'platform') "
            "WHERE scope IS NULL OR scope = ''"
        )
    )
    session.exec(
        text(
            "UPDATE knowledge_documents "
            "SET visibility = COALESCE(NULLIF(visibility, ''), 'platform'), "
            "status = COALESCE(NULLIF(status, ''), 'approved') "
            "WHERE visibility IS NULL OR visibility = '' OR status IS NULL OR status = ''"
        )
    )

    if migration_record is None:
        session.add(
            SchemaMigrationRecord(
                migration_key=MIGRATION_KEY_CI2,
                description=(
                    "Anade scope, gobierno y filtros multitenant al corpus de knowledge, "
                    "sincronizando documentos y secciones para retrieval gobernado."
                ),
            )
        )

    session.commit()
    summary.runs_backfilled = max(getattr(runs_update, "rowcount", 0), 0)
    return summary
