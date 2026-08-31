from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from sqlalchemy import event
from sqlmodel import SQLModel, Session, create_engine, select

from app.models import RuntimeFeatureFlagRecord, WorkspaceRecord
from app.services.stage5_service import FEATURE_FLAG_DESIGN_INTELLIGENCE, FEATURE_FLAG_STAGE_ANSWER_INFERENCE
from app.services.workspace_bootstrap import DEFAULT_FEATURE_FLAGS, seed_runtime_feature_flags


def test_default_feature_flags_include_design_intelligence_rollout_switch() -> None:
    flags = {item["key"]: item for item in DEFAULT_FEATURE_FLAGS}

    assert FEATURE_FLAG_DESIGN_INTELLIGENCE in flags
    assert flags[FEATURE_FLAG_DESIGN_INTELLIGENCE]["enabled"] is True
    assert flags[FEATURE_FLAG_DESIGN_INTELLIGENCE]["stage_hint"] == "di141"
    assert FEATURE_FLAG_STAGE_ANSWER_INFERENCE in flags
    assert flags[FEATURE_FLAG_STAGE_ANSWER_INFERENCE]["enabled"] is False
    assert flags[FEATURE_FLAG_STAGE_ANSWER_INFERENCE]["stage_hint"] == "iai148"


def test_seed_runtime_feature_flags_recovers_from_concurrent_insert_race() -> None:
    with TemporaryDirectory(prefix="lean-builder-bootstrap-race-") as tmp_dir:
        database_path = Path(tmp_dir) / "bootstrap-race.db"
        engine = create_engine(
            f"sqlite:///{database_path.as_posix()}",
            connect_args={"check_same_thread": False},
        )
        try:
            SQLModel.metadata.create_all(engine)
            workspace_id = uuid4()

            with Session(engine) as session:
                session.add(
                    WorkspaceRecord(
                        id=workspace_id,
                        name="Race Workspace",
                        slug=f"race-{workspace_id.hex[:10]}",
                    )
                )
                session.commit()

            injected = {"done": False}

            def inject_concurrent_seed(conn, cursor, statement, parameters, context, executemany) -> None:  # noqa: ANN001
                _ = conn, cursor, parameters, context, executemany
                if injected["done"] or not statement.lstrip().lower().startswith("insert into runtime_feature_flags"):
                    return
                injected["done"] = True
                with Session(engine) as concurrent_session:
                    seed_runtime_feature_flags(concurrent_session, workspace_id=workspace_id)

            event.listen(engine, "before_cursor_execute", inject_concurrent_seed)
            try:
                with Session(engine) as session:
                    seed_runtime_feature_flags(session, workspace_id=workspace_id)
            finally:
                event.remove(engine, "before_cursor_execute", inject_concurrent_seed)

            with Session(engine) as session:
                rows = session.exec(
                    select(RuntimeFeatureFlagRecord).where(RuntimeFeatureFlagRecord.workspace_id == workspace_id)
                ).all()
        finally:
            engine.dispose()

    assert injected["done"] is True
    assert len(rows) == len(DEFAULT_FEATURE_FLAGS)
    assert {row.flag_key for row in rows} == {item["key"] for item in DEFAULT_FEATURE_FLAGS}
