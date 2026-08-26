from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.models import (
    ArtifactStatus,
    SessionRecord,
    SessionStage,
    SkillRunArtifactRecord,
    SkillRunRecord,
    SubagentRunRecord,
    UserRecord,
    WorkspaceRecord,
)
from scripts import import_session_snapshot


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def test_import_snapshot_includes_skill_runs_and_subagent_runs(tmp_path, monkeypatch) -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(
        engine,
        tables=[
            UserRecord.__table__,
            WorkspaceRecord.__table__,
            SessionRecord.__table__,
            SkillRunRecord.__table__,
            SkillRunArtifactRecord.__table__,
            SubagentRunRecord.__table__,
        ],
    )
    monkeypatch.setattr(import_session_snapshot, "engine", engine)

    source_user = UserRecord(
        id=uuid4(),
        email="owner@example.com",
        full_name="Prod Owner",
        password_hash="prod-hash",
    )
    source_workspace = WorkspaceRecord(
        id=uuid4(),
        name="Lean Workspace",
        slug="lean-workspace-runtime",
        created_by_user_id=source_user.id,
    )
    source_session = SessionRecord(
        id=uuid4(),
        user_id=source_user.id,
        workspace_id=source_workspace.id,
        title="Proyecto runtime",
    )
    source_skill_run = SkillRunRecord(
        id=uuid4(),
        session_id=source_session.id,
        skill_key="recommend_memory_architecture",
        stage=SessionStage.build_blueprint,
        source_action="recommend_memory",
        status=ArtifactStatus.failed,
        duration_ms=2100,
        result_summary="No se pudo generar memoria automaticamente",
    )
    source_skill_artifact = SkillRunArtifactRecord(
        id=uuid4(),
        skill_run_id=source_skill_run.id,
        artifact_role="output",
        artifact_kind="memory.trace",
        payload={"summary": "runtime_error"},
    )
    source_subagent_run = SubagentRunRecord(
        id=uuid4(),
        session_id=source_session.id,
        blueprint_version_number=2,
        run_kind="memory_recovery",
        title="Memory recovery",
        status=ArtifactStatus.failed,
        summary="Subagente quedo bloqueado",
        input_payload={"stage": "memory"},
        output_payload={"status": "failed"},
    )

    _write_rows(tmp_path / "users.json", [source_user.model_dump(mode="json")])
    _write_rows(tmp_path / "workspaces.json", [source_workspace.model_dump(mode="json")])
    _write_rows(tmp_path / "sessions.json", [source_session.model_dump(mode="json")])
    _write_rows(tmp_path / "skill_runs.json", [source_skill_run.model_dump(mode="json")])
    _write_rows(tmp_path / "skill_run_artifacts.json", [source_skill_artifact.model_dump(mode="json")])
    _write_rows(tmp_path / "subagent_runs.json", [source_subagent_run.model_dump(mode="json")])

    summary = import_session_snapshot.import_snapshot(tmp_path)

    assert summary["skill_runs.json"] == 1
    assert summary["skill_run_artifacts.json"] == 1
    assert summary["subagent_runs.json"] == 1

    with Session(engine) as session:
        skill_runs = session.exec(select(SkillRunRecord)).all()
        skill_run_artifacts = session.exec(select(SkillRunArtifactRecord)).all()
        subagent_runs = session.exec(select(SubagentRunRecord)).all()

    assert len(skill_runs) == 1
    assert skill_runs[0].skill_key == "recommend_memory_architecture"
    assert skill_runs[0].source_action == "recommend_memory"
    assert skill_runs[0].status == ArtifactStatus.failed
    assert len(skill_run_artifacts) == 1
    assert skill_run_artifacts[0].skill_run_id == source_skill_run.id
    assert skill_run_artifacts[0].payload == {"summary": "runtime_error"}
    assert len(subagent_runs) == 1
    assert subagent_runs[0].run_kind == "memory_recovery"
    assert subagent_runs[0].status == ArtifactStatus.failed
