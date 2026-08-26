from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
from uuid import UUID

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sqlmodel import Session, select

from app.db import engine
from app.models import (
    ExecutionLogRecord,
    RuntimeSettingsAuditRecord,
    SessionRecord,
    SkillRunArtifactRecord,
    SkillRunRecord,
    SubagentRunRecord,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceProviderSecretRecord,
    WorkspaceRecord,
    WorkspaceRuntimeSettingsRecord,
    ArtifactRegistryRecord,
    JourneyStageArtifactRecord,
    StageOperationRecord,
    LLMUsageLedgerRecord,
)
from app.services.diagram_center.persistence import DiagramGenerationJobRecord, DiagramVersionRecord


DEFAULT_SNAPSHOT_DIR = REPO_ROOT / "runtime" / "prod-session-84e2cdc7-5352-40d1-bd06-7795021d4b2d"


TABLE_IMPORT_ORDER: list[tuple[str, type[Any]]] = [
    ("users.json", UserRecord),
    ("workspaces.json", WorkspaceRecord),
    ("workspace_memberships.json", WorkspaceMembershipRecord),
    ("sessions.json", SessionRecord),
    ("workspace_runtime_settings.json", WorkspaceRuntimeSettingsRecord),
    ("workspace_provider_secrets.json", WorkspaceProviderSecretRecord),
    ("runtime_settings_audit.json", RuntimeSettingsAuditRecord),
    ("execution_logs.json", ExecutionLogRecord),
    ("artifact_records.json", ArtifactRegistryRecord),
    ("journey_stage_artifacts.json", JourneyStageArtifactRecord),
    ("stage_operations.json", StageOperationRecord),
    ("subagent_runs.json", SubagentRunRecord),
    ("skill_runs.json", SkillRunRecord),
    ("skill_run_artifacts.json", SkillRunArtifactRecord),
    ("diagram_generation_jobs_v3.json", DiagramGenerationJobRecord),
    ("diagram_versions_v3.json", DiagramVersionRecord),
    ("llm_usage_ledger.json", LLMUsageLedgerRecord),
]

DEFERRED_SELF_REFERENCE_FIELDS: dict[str, tuple[str, ...]] = {
    "journey_stage_artifacts.json": ("based_on_artifact_id", "superseded_by_artifact_id"),
}

USER_ID_FIELDS = {
    "user_id",
    "created_by_user_id",
    "updated_by_user_id",
    "approved_by_user_id",
    "archived_by_user_id",
    "deleted_by_user_id",
    "requested_by_user_id",
    "actor_user_id",
    "buyer_user_id",
}

WORKSPACE_ID_FIELDS = {
    "workspace_id",
    "default_workspace_id",
}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def _load_dataset(snapshot_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {filename: _load_rows(snapshot_dir / filename) for filename, _ in TABLE_IMPORT_ORDER}


def _build_user_id_map(session: Session, dataset: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in dataset.get("users.json", []):
        source_id = str(row.get("id") or "").strip()
        email = str(row.get("email") or "").strip().lower()
        if not source_id or not email:
            continue
        existing = session.exec(select(UserRecord).where(UserRecord.email == email)).first()
        if existing is not None:
            mapping[source_id] = str(existing.id)
    return mapping


def _build_workspace_id_map(session: Session, dataset: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in dataset.get("workspaces.json", []):
        source_id = str(row.get("id") or "").strip()
        slug = str(row.get("slug") or "").strip().lower()
        if not source_id or not slug:
            continue
        existing = session.exec(select(WorkspaceRecord).where(WorkspaceRecord.slug == slug)).first()
        if existing is not None:
            mapping[source_id] = str(existing.id)
    return mapping


def _build_workspace_membership_id_map(
    session: Session,
    dataset: dict[str, list[dict[str, Any]]],
    *,
    user_id_map: dict[str, str],
    workspace_id_map: dict[str, str],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in dataset.get("workspace_memberships.json", []):
        source_id = str(row.get("id") or "").strip()
        workspace_id = _coerce_uuid(
            workspace_id_map.get(str(row.get("workspace_id") or "").strip(), str(row.get("workspace_id") or "").strip())
        )
        user_id = _coerce_uuid(user_id_map.get(str(row.get("user_id") or "").strip(), str(row.get("user_id") or "").strip()))
        if not source_id or not workspace_id or not user_id:
            continue
        existing = session.exec(
            select(WorkspaceMembershipRecord).where(
                WorkspaceMembershipRecord.workspace_id == workspace_id,
                WorkspaceMembershipRecord.user_id == user_id,
            )
        ).first()
        if existing is not None:
            mapping[source_id] = str(existing.id)
    return mapping


def _build_workspace_runtime_settings_id_map(
    session: Session,
    dataset: dict[str, list[dict[str, Any]]],
    *,
    workspace_id_map: dict[str, str],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in dataset.get("workspace_runtime_settings.json", []):
        source_id = str(row.get("id") or "").strip()
        workspace_id = _coerce_uuid(
            workspace_id_map.get(str(row.get("workspace_id") or "").strip(), str(row.get("workspace_id") or "").strip())
        )
        version = row.get("version")
        if not source_id or not workspace_id or version is None:
            continue
        existing = session.exec(
            select(WorkspaceRuntimeSettingsRecord).where(
                WorkspaceRuntimeSettingsRecord.workspace_id == workspace_id,
                WorkspaceRuntimeSettingsRecord.version == version,
            )
        ).first()
        if existing is not None:
            mapping[source_id] = str(existing.id)
    return mapping


def _build_workspace_provider_secret_id_map(
    session: Session,
    dataset: dict[str, list[dict[str, Any]]],
    *,
    workspace_id_map: dict[str, str],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in dataset.get("workspace_provider_secrets.json", []):
        source_id = str(row.get("id") or "").strip()
        workspace_id = _coerce_uuid(
            workspace_id_map.get(str(row.get("workspace_id") or "").strip(), str(row.get("workspace_id") or "").strip())
        )
        provider_key = str(row.get("provider_key") or "").strip()
        secret_kind = str(row.get("secret_kind") or "").strip()
        if not source_id or not workspace_id or not provider_key or not secret_kind:
            continue
        existing = session.exec(
            select(WorkspaceProviderSecretRecord).where(
                WorkspaceProviderSecretRecord.workspace_id == workspace_id,
                WorkspaceProviderSecretRecord.provider_key == provider_key,
                WorkspaceProviderSecretRecord.secret_kind == secret_kind,
            )
        ).first()
        if existing is not None:
            mapping[source_id] = str(existing.id)
    return mapping


def _coerce_uuid(value: Any) -> UUID | None:
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    token = str(value).strip()
    if not token:
        return None
    return UUID(token)


def _remap_row_ids(
    filename: str,
    row: dict[str, Any],
    *,
    user_id_map: dict[str, str],
    workspace_id_map: dict[str, str],
    workspace_membership_id_map: dict[str, str],
    workspace_runtime_settings_id_map: dict[str, str],
    workspace_provider_secret_id_map: dict[str, str],
) -> dict[str, Any]:
    rewritten = dict(row)
    source_row_id = str(rewritten.get("id") or "").strip()
    if filename == "users.json" and source_row_id in user_id_map:
        rewritten["id"] = user_id_map[source_row_id]
    if filename == "workspaces.json" and source_row_id in workspace_id_map:
        rewritten["id"] = workspace_id_map[source_row_id]
    if filename == "workspace_memberships.json" and source_row_id in workspace_membership_id_map:
        rewritten["id"] = workspace_membership_id_map[source_row_id]
    if filename == "workspace_runtime_settings.json" and source_row_id in workspace_runtime_settings_id_map:
        rewritten["id"] = workspace_runtime_settings_id_map[source_row_id]
    if filename == "workspace_provider_secrets.json" and source_row_id in workspace_provider_secret_id_map:
        rewritten["id"] = workspace_provider_secret_id_map[source_row_id]

    for key, value in list(rewritten.items()):
        if value is None:
            continue
        token = str(value)
        if key in USER_ID_FIELDS and token in user_id_map:
            rewritten[key] = user_id_map[token]
        elif key in WORKSPACE_ID_FIELDS and token in workspace_id_map:
            rewritten[key] = workspace_id_map[token]

    if filename == "runtime_settings_audit.json":
        scope_id = str(rewritten.get("scope_id") or "").strip()
        if scope_id in workspace_id_map:
            rewritten["scope_id"] = workspace_id_map[scope_id]
    return rewritten


def _without_fields(row: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    rewritten = dict(row)
    for field in fields:
        if field in rewritten:
            rewritten[field] = None
    return rewritten


def import_snapshot(snapshot_dir: Path) -> dict[str, int]:
    summary: dict[str, int] = {}
    with Session(engine) as session:
        dataset = _load_dataset(snapshot_dir)
        user_id_map = _build_user_id_map(session, dataset)
        workspace_id_map = _build_workspace_id_map(session, dataset)
        workspace_membership_id_map = _build_workspace_membership_id_map(
            session,
            dataset,
            user_id_map=user_id_map,
            workspace_id_map=workspace_id_map,
        )
        workspace_runtime_settings_id_map = _build_workspace_runtime_settings_id_map(
            session,
            dataset,
            workspace_id_map=workspace_id_map,
        )
        workspace_provider_secret_id_map = _build_workspace_provider_secret_id_map(
            session,
            dataset,
            workspace_id_map=workspace_id_map,
        )
        for filename, model_cls in TABLE_IMPORT_ORDER:
            rows = dataset.get(filename, [])
            imported = 0
            deferred_fields = DEFERRED_SELF_REFERENCE_FIELDS.get(filename, ())
            with session.no_autoflush:
                for row in rows:
                    rewritten = _remap_row_ids(
                        filename,
                        row,
                        user_id_map=user_id_map,
                        workspace_id_map=workspace_id_map,
                        workspace_membership_id_map=workspace_membership_id_map,
                        workspace_runtime_settings_id_map=workspace_runtime_settings_id_map,
                        workspace_provider_secret_id_map=workspace_provider_secret_id_map,
                    )
                    if deferred_fields:
                        rewritten = _without_fields(rewritten, deferred_fields)
                    session.merge(model_cls.model_validate(rewritten))
                    imported += 1
            session.commit()
            if deferred_fields and rows:
                with session.no_autoflush:
                    for row in rows:
                        rewritten = _remap_row_ids(
                            filename,
                            row,
                            user_id_map=user_id_map,
                            workspace_id_map=workspace_id_map,
                            workspace_membership_id_map=workspace_membership_id_map,
                            workspace_runtime_settings_id_map=workspace_runtime_settings_id_map,
                            workspace_provider_secret_id_map=workspace_provider_secret_id_map,
                        )
                        session.merge(model_cls.model_validate(rewritten))
                session.commit()
            summary[filename] = imported
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Importa un snapshot JSON de una sesion productiva a la base local.")
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=DEFAULT_SNAPSHOT_DIR,
        help="Directorio que contiene los archivos *.json exportados desde produccion.",
    )
    args = parser.parse_args()
    snapshot_dir = args.snapshot_dir.resolve()
    if not snapshot_dir.exists():
        raise SystemExit(f"Snapshot directory not found: {snapshot_dir}")
    summary = import_snapshot(snapshot_dir)
    print(json.dumps({"snapshot_dir": str(snapshot_dir), "imported": summary}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
