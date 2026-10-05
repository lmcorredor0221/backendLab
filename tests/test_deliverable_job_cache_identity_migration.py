from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy import text

from app.core.config import get_settings


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_REVISION = "20260925_0032"
TARGET_REVISION = "20260926_0033"
TABLE_NAME = "deliverable_generation_jobs_v1"


@contextmanager
def configured_sqlite_database() -> Iterator[tuple[Config, Path]]:
    previous_url = os.environ.get("DATABASE_URL")
    with TemporaryDirectory(prefix="deliverable-cache-migration-") as temp_dir:
        db_path = Path(temp_dir) / "migration.db"
        os.environ["DATABASE_URL"] = f"sqlite:///{db_path.as_posix()}"
        get_settings.cache_clear()
        config = Config(str(BACKEND_ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(BACKEND_ROOT / "alembic"))
        try:
            yield config, db_path
        finally:
            if previous_url is None:
                os.environ.pop("DATABASE_URL", None)
            else:
                os.environ["DATABASE_URL"] = previous_url
            get_settings.cache_clear()


def _engine_for(db_path: Path) -> sa.Engine:
    return sa.create_engine(f"sqlite:///{db_path.as_posix()}")


def _create_historic_job_table(db_path: Path) -> None:
    engine = _engine_for(db_path)
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    CREATE TABLE deliverable_generation_jobs_v1 (
                        id VARCHAR(36) PRIMARY KEY,
                        workspace_id VARCHAR(36) NOT NULL,
                        session_id VARCHAR(36) NOT NULL,
                        deliverable_key VARCHAR NOT NULL,
                        status VARCHAR NOT NULL
                    )
                    """
                )
            )
    finally:
        engine.dispose()


def test_deliverable_job_cache_identity_migration_is_additive_and_reversible() -> None:
    with configured_sqlite_database() as (config, db_path):
        _create_historic_job_table(db_path)

        command.stamp(config, PREVIOUS_REVISION)
        command.upgrade(config, TARGET_REVISION)

        engine = _engine_for(db_path)
        try:
            inspector = sa.inspect(engine)
            columns = {column["name"] for column in inspector.get_columns(TABLE_NAME)}
            indexes = {index["name"] for index in inspector.get_indexes(TABLE_NAME)}

            assert {"input_fingerprint", "builder_version", "generation_profile_version"}.issubset(columns)
            assert "ix_deliverable_generation_jobs_v1_input_fingerprint" in indexes
            assert "ix_deliverable_generation_jobs_v1_builder_version" in indexes
            assert "ix_deliverable_generation_jobs_v1_cache_identity" in indexes
        finally:
            engine.dispose()

        command.downgrade(config, PREVIOUS_REVISION)

        engine = _engine_for(db_path)
        try:
            inspector = sa.inspect(engine)
            columns = {column["name"] for column in inspector.get_columns(TABLE_NAME)}
            indexes = {index["name"] for index in inspector.get_indexes(TABLE_NAME)}

            assert "input_fingerprint" not in columns
            assert "builder_version" not in columns
            assert "generation_profile_version" not in columns
            assert "ix_deliverable_generation_jobs_v1_cache_identity" not in indexes
        finally:
            engine.dispose()
