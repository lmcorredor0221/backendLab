from __future__ import annotations

from sqlalchemy.dialects import postgresql
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine
from sqlalchemy import text

from app.services.knowledge_memory_migration import (
    _build_blank_normalization_statement,
    _normalize_blank_scalar_column,
)


def _build_engine():
    return create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def test_blank_normalization_statement_uses_text_cast_for_postgres() -> None:
    statement = _build_blank_normalization_statement("knowledge_ingestion_runs", "scope", "platform")
    compiled = str(statement.compile(dialect=postgresql.dialect()))

    assert "SET scope = 'platform'" in compiled
    assert "CAST(scope AS TEXT) = ''" in compiled
    assert "scope = ''" not in compiled


def test_blank_normalization_updates_blank_and_null_rows_in_sqlite() -> None:
    engine = _build_engine()

    with Session(engine) as session:
        session.exec(text("CREATE TABLE knowledge_ingestion_runs (id INTEGER PRIMARY KEY, scope TEXT)"))
        session.exec(text("INSERT INTO knowledge_ingestion_runs (id, scope) VALUES (1, '')"))
        session.exec(text("INSERT INTO knowledge_ingestion_runs (id, scope) VALUES (2, NULL)"))
        session.exec(text("INSERT INTO knowledge_ingestion_runs (id, scope) VALUES (3, 'workspace')"))
        session.commit()

        updated = _normalize_blank_scalar_column(
            session,
            table_name="knowledge_ingestion_runs",
            column_name="scope",
            replacement="platform",
        )
        session.commit()

        rows = session.exec(text("SELECT scope FROM knowledge_ingestion_runs ORDER BY id")).all()

    assert updated == 2
    assert [row[0] for row in rows] == ["platform", "platform", "workspace"]
