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
from sqlmodel import SQLModel, create_engine

from app.core.config import get_settings
from app.models import StageOperationRecord  # noqa: F401
from app.services.product_processing.persistence import ProductBuildRunRecord, ProductBuildStepRecord  # noqa: F401


BACKEND_ROOT = Path(__file__).resolve().parents[1]
A07_REVISION = "20260816_0014"
PREVIOUS_REVISION = "20260813_0012"


EXPECTED_COLUMNS = {
    "product_build_runs_v1": {
        "id",
        "workspace_id",
        "session_id",
        "product_key",
        "product_mode",
        "entitlement_tier",
        "access_state",
        "lifecycle",
        "progress_percent",
        "completed_units",
        "total_units",
        "blocked_units",
        "idempotency_key",
        "checkpoint_payload",
        "error_payload",
        "created_by_user_id",
        "created_at",
        "started_at",
        "completed_at",
        "requires_attention_at",
        "updated_at",
    },
    "product_build_steps_v1": {
        "id",
        "run_id",
        "workspace_id",
        "session_id",
        "step_key",
        "stage_key",
        "deliverable_key",
        "job_id",
        "dependency_key",
        "status",
        "sequence",
        "progress_percent",
        "checkpoint_payload",
        "error_payload",
        "started_at",
        "completed_at",
        "updated_at",
    },
    "stage_operations": {
        "id",
        "workspace_id",
        "session_id",
        "user_id",
        "stage_key",
        "action",
        "idempotency_key",
        "attempt_count",
        "status",
        "current_step",
        "detail",
        "request_payload",
        "steps",
        "result_artifact_id",
        "error_message",
        "technical_detail",
        "cancel_requested_at",
        "heartbeat_at",
        "expires_at",
        "created_at",
        "updated_at",
        "completed_at",
    },
}

EXPECTED_INDEXES = {
    "product_build_runs_v1": {
        "ix_product_build_runs_v1_workspace_id",
        "ix_product_build_runs_v1_session_id",
        "ix_product_build_runs_v1_product_key",
        "ix_product_build_runs_v1_product_mode",
        "ix_product_build_runs_v1_entitlement_tier",
        "ix_product_build_runs_v1_access_state",
        "ix_product_build_runs_v1_lifecycle",
        "ix_product_build_runs_v1_idempotency_key",
        "ix_product_build_runs_v1_created_by_user_id",
    },
    "product_build_steps_v1": {
        "ix_product_build_steps_v1_run_id",
        "ix_product_build_steps_v1_workspace_id",
        "ix_product_build_steps_v1_session_id",
        "ix_product_build_steps_v1_step_key",
        "ix_product_build_steps_v1_stage_key",
        "ix_product_build_steps_v1_deliverable_key",
        "ix_product_build_steps_v1_job_id",
        "ix_product_build_steps_v1_dependency_key",
        "ix_product_build_steps_v1_status",
    },
    "stage_operations": {
        "ix_stage_operations_workspace_id",
        "ix_stage_operations_session_id",
        "ix_stage_operations_user_id",
        "ix_stage_operations_stage_key",
        "ix_stage_operations_action",
        "ix_stage_operations_idempotency_key",
        "ix_stage_operations_status",
        "ix_stage_operations_result_artifact_id",
        "ix_stage_operations_cancel_requested_at",
        "ix_stage_operations_heartbeat_at",
        "ix_stage_operations_expires_at",
        "uq_stage_operations_workspace_session_action_idempotency",
    },
}


@contextmanager
def configured_sqlite_database() -> Iterator[tuple[Config, Path]]:
    previous_url = os.environ.get("DATABASE_URL")
    with TemporaryDirectory(prefix="a02-product-build-migration-") as temp_dir:
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


def _engine_for(db_path: Path):
    return sa.create_engine(f"sqlite:///{db_path.as_posix()}")


def _assert_a02_schema(db_path: Path) -> None:
    engine = _engine_for(db_path)
    try:
        inspector = sa.inspect(engine)
        for table_name, expected_columns in EXPECTED_COLUMNS.items():
            assert inspector.has_table(table_name)
            assert {column["name"] for column in inspector.get_columns(table_name)} == expected_columns
            index_names = {index["name"] for index in inspector.get_indexes(table_name)}
            assert EXPECTED_INDEXES[table_name].issubset(index_names)

        run_unique_names = {
            constraint["name"] for constraint in inspector.get_unique_constraints("product_build_runs_v1")
        }
        run_unique_names.update(
            index["name"] for index in inspector.get_indexes("product_build_runs_v1") if index.get("unique")
        )
        step_unique_names = {
            constraint["name"] for constraint in inspector.get_unique_constraints("product_build_steps_v1")
        }
        step_unique_names.update(
            index["name"] for index in inspector.get_indexes("product_build_steps_v1") if index.get("unique")
        )

        assert "uq_product_build_run_workspace_idempotency_v1" in run_unique_names
        assert "uq_product_build_step_run_key_v1" in step_unique_names
    finally:
        engine.dispose()


def test_a02_upgrade_downgrade_upgrade_from_0012() -> None:
    with configured_sqlite_database() as (config, db_path):
        command.stamp(config, PREVIOUS_REVISION)
        command.upgrade(config, "head")
        _assert_a02_schema(db_path)

        engine = _engine_for(db_path)
        try:
            with engine.connect() as connection:
                current = connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
            assert current == A07_REVISION

            command.downgrade(config, PREVIOUS_REVISION)
            inspector = sa.inspect(engine)
            assert not inspector.has_table("stage_operations")
            assert not inspector.has_table("product_build_steps_v1")
            assert not inspector.has_table("product_build_runs_v1")
        finally:
            engine.dispose()

        command.upgrade(config, "head")
        _assert_a02_schema(db_path)


def test_a02_upgrade_is_idempotent_when_create_all_already_created_tables() -> None:
    with configured_sqlite_database() as (config, db_path):
        engine = create_engine(f"sqlite:///{db_path.as_posix()}")
        try:
            SQLModel.metadata.create_all(engine)
        finally:
            engine.dispose()

        command.stamp(config, PREVIOUS_REVISION)
        command.upgrade(config, "head")
        _assert_a02_schema(db_path)


def test_a02_upgrade_preserves_rows_in_incomplete_product_build_run_table() -> None:
    with configured_sqlite_database() as (config, db_path):
        engine = _engine_for(db_path)
        try:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        """
                        CREATE TABLE product_build_runs_v1 (
                            id VARCHAR(36) PRIMARY KEY,
                            workspace_id VARCHAR(36) NOT NULL,
                            session_id VARCHAR(36) NOT NULL,
                            product_key VARCHAR NOT NULL,
                            product_mode VARCHAR NOT NULL,
                            idempotency_key VARCHAR NOT NULL
                        )
                        """
                    )
                )
                connection.execute(
                    text(
                        """
                        INSERT INTO product_build_runs_v1
                            (id, workspace_id, session_id, product_key, product_mode, idempotency_key)
                        VALUES
                            ('run-1', 'workspace-1', 'session-1', 'blueprint_basic', 'basic_free', 'basic:session-1')
                        """
                    )
                )
        finally:
            engine.dispose()

        command.stamp(config, PREVIOUS_REVISION)
        command.upgrade(config, "head")
        _assert_a02_schema(db_path)

        with engine.connect() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT id, entitlement_tier, access_state, lifecycle, progress_percent
                    FROM product_build_runs_v1
                    WHERE id = 'run-1'
                    """
                )
            ).mappings().one()
        engine.dispose()

        assert row["id"] == "run-1"
        assert row["entitlement_tier"] == "blueprint"
        assert row["access_state"] == "preview"
        assert row["lifecycle"] == "ready_to_start"
        assert row["progress_percent"] == 0
