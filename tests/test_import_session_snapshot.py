from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.models import (
    LLMProviderKey,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceRecord,
    WorkspaceRole,
    WorkspaceRuntimeSettingsRecord,
)
from scripts import import_session_snapshot


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def test_import_snapshot_reuses_existing_workspace_runtime_settings_identity(tmp_path, monkeypatch) -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(
        engine,
        tables=[
            UserRecord.__table__,
            WorkspaceRecord.__table__,
            WorkspaceRuntimeSettingsRecord.__table__,
        ],
    )
    monkeypatch.setattr(import_session_snapshot, "engine", engine)

    with Session(engine) as session:
        local_user = UserRecord(email="owner@example.com", full_name="Local Owner", password_hash="hash")
        session.add(local_user)
        session.commit()
        session.refresh(local_user)

        local_workspace = WorkspaceRecord(
            name="Lean Workspace",
            slug="lean-workspace",
            created_by_user_id=local_user.id,
        )
        session.add(local_workspace)
        session.commit()
        session.refresh(local_workspace)

        existing_runtime = WorkspaceRuntimeSettingsRecord(
            workspace_id=local_workspace.id,
            version=2,
            active_provider=LLMProviderKey.openai,
            is_active=True,
        )
        session.add(existing_runtime)
        session.commit()
        session.refresh(existing_runtime)
        existing_runtime_id = existing_runtime.id
        local_workspace_id = local_workspace.id

    source_user = UserRecord(
        id=uuid4(),
        email="owner@example.com",
        full_name="Prod Owner",
        password_hash="prod-hash",
    )
    source_workspace = WorkspaceRecord(
        id=uuid4(),
        name="Lean Workspace",
        slug="lean-workspace",
        created_by_user_id=source_user.id,
    )
    source_runtime_v2 = WorkspaceRuntimeSettingsRecord(
        id=uuid4(),
        workspace_id=source_workspace.id,
        version=2,
        active_provider=LLMProviderKey.deepseek,
        is_active=False,
    )
    source_runtime_v3 = WorkspaceRuntimeSettingsRecord(
        id=uuid4(),
        workspace_id=source_workspace.id,
        version=3,
        active_provider=LLMProviderKey.deepseek,
        is_active=True,
    )

    _write_rows(tmp_path / "users.json", [source_user.model_dump(mode="json")])
    _write_rows(tmp_path / "workspaces.json", [source_workspace.model_dump(mode="json")])
    _write_rows(
        tmp_path / "workspace_runtime_settings.json",
        [
            source_runtime_v2.model_dump(mode="json"),
            source_runtime_v3.model_dump(mode="json"),
        ],
    )

    summary = import_session_snapshot.import_snapshot(tmp_path)

    assert summary["workspace_runtime_settings.json"] == 2

    with Session(engine) as session:
        runtime_rows = session.exec(
            select(WorkspaceRuntimeSettingsRecord)
            .where(WorkspaceRuntimeSettingsRecord.workspace_id == local_workspace_id)
            .order_by(WorkspaceRuntimeSettingsRecord.version)
        ).all()

    assert [(row.version, row.active_provider, row.is_active) for row in runtime_rows] == [
        (2, "deepseek", False),
        (3, "deepseek", True),
    ]
    assert runtime_rows[0].id == existing_runtime_id
    assert runtime_rows[1].id == source_runtime_v3.id


def test_import_snapshot_reuses_existing_workspace_membership_identity(tmp_path, monkeypatch) -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(
        engine,
        tables=[
            UserRecord.__table__,
            WorkspaceRecord.__table__,
            WorkspaceMembershipRecord.__table__,
        ],
    )
    monkeypatch.setattr(import_session_snapshot, "engine", engine)

    with Session(engine) as session:
        local_user = UserRecord(email="owner@example.com", full_name="Local Owner", password_hash="hash")
        session.add(local_user)
        session.commit()
        session.refresh(local_user)

        local_workspace = WorkspaceRecord(
            name="Lean Workspace",
            slug="lean-workspace",
            created_by_user_id=local_user.id,
        )
        session.add(local_workspace)
        session.commit()
        session.refresh(local_workspace)

        existing_membership = WorkspaceMembershipRecord(
            workspace_id=local_workspace.id,
            user_id=local_user.id,
            role=WorkspaceRole.viewer,
            is_active=False,
        )
        session.add(existing_membership)
        session.commit()
        session.refresh(existing_membership)
        existing_membership_id = existing_membership.id
        local_workspace_id = local_workspace.id
        local_user_id = local_user.id

    source_user = UserRecord(
        id=uuid4(),
        email="owner@example.com",
        full_name="Prod Owner",
        password_hash="prod-hash",
    )
    source_workspace = WorkspaceRecord(
        id=uuid4(),
        name="Lean Workspace",
        slug="lean-workspace",
        created_by_user_id=source_user.id,
    )
    source_membership = WorkspaceMembershipRecord(
        id=uuid4(),
        workspace_id=source_workspace.id,
        user_id=source_user.id,
        role=WorkspaceRole.owner,
        is_active=True,
    )

    _write_rows(tmp_path / "users.json", [source_user.model_dump(mode="json")])
    _write_rows(tmp_path / "workspaces.json", [source_workspace.model_dump(mode="json")])
    _write_rows(tmp_path / "workspace_memberships.json", [source_membership.model_dump(mode="json")])

    summary = import_session_snapshot.import_snapshot(tmp_path)

    assert summary["workspace_memberships.json"] == 1

    with Session(engine) as session:
        memberships = session.exec(
            select(WorkspaceMembershipRecord).where(
                WorkspaceMembershipRecord.workspace_id == local_workspace_id,
                WorkspaceMembershipRecord.user_id == local_user_id,
            )
        ).all()

    assert len(memberships) == 1
    assert memberships[0].id == existing_membership_id
    assert memberships[0].role == "owner"
    assert memberships[0].is_active is True
