from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from uuid import UUID

from dotenv import dotenv_values
from sqlalchemy import MetaData, Table, create_engine, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOTENV = REPO_ROOT / ".env"

if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from app.models import UserRecord, WorkspaceMembershipRecord, WorkspaceRecord, WorkspaceRuntimeSettingsRecord
from app.models import GovernancePolicyRecord, RuntimeFeatureFlagRecord, WorkflowTemplateRecord


PREFERRED_TABLE_ORDER = [
    "users",
    "workspaces",
    "workspace_memberships",
    "sessions",
    "workspace_runtime_settings",
    "runtime_settings_audit",
]


def _database_url() -> str:
    explicit = os.environ.get("DATABASE_URL", "").strip()
    if explicit:
        return explicit
    env_values = dotenv_values(DEFAULT_DOTENV)
    candidate = str(env_values.get("DATABASE_URL") or "").strip()
    if not candidate:
        raise SystemExit("No se encontro DATABASE_URL en el entorno ni en backend/.env")
    return candidate


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def _load_snapshot(snapshot_dir: Path) -> dict[str, list[dict[str, Any]]]:
    dataset: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(snapshot_dir.glob("*.json")):
        if path.name == "snapshot_manifest.json" or path.name.endswith(".raw.json"):
            continue
        dataset[path.stem] = _load_rows(path)
    return dataset


def _coerce_uuid(value: Any) -> UUID | None:
    if value in (None, ""):
        return None
    if isinstance(value, UUID):
        return value
    return UUID(str(value))


def _build_user_id_map(session: Session, dataset: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in dataset.get("users", []):
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
    for row in dataset.get("workspaces", []):
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
    for row in dataset.get("workspace_memberships", []):
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
    for row in dataset.get("workspace_runtime_settings", []):
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


def _build_runtime_feature_flag_id_map(
    session: Session,
    dataset: dict[str, list[dict[str, Any]]],
    *,
    workspace_id_map: dict[str, str],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in dataset.get("runtime_feature_flags", []):
        source_id = str(row.get("id") or "").strip()
        workspace_id = _coerce_uuid(
            workspace_id_map.get(str(row.get("workspace_id") or "").strip(), str(row.get("workspace_id") or "").strip())
        )
        flag_key = str(row.get("flag_key") or "").strip()
        if not source_id or not workspace_id or not flag_key:
            continue
        existing = session.exec(
            select(RuntimeFeatureFlagRecord).where(
                RuntimeFeatureFlagRecord.workspace_id == workspace_id,
                RuntimeFeatureFlagRecord.flag_key == flag_key,
            )
        ).first()
        if existing is not None:
            mapping[source_id] = str(existing.id)
    return mapping


def _build_governance_policy_id_map(
    session: Session,
    dataset: dict[str, list[dict[str, Any]]],
    *,
    workspace_id_map: dict[str, str],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in dataset.get("governance_policies", []):
        source_id = str(row.get("id") or "").strip()
        workspace_id = _coerce_uuid(
            workspace_id_map.get(str(row.get("workspace_id") or "").strip(), str(row.get("workspace_id") or "").strip())
        )
        policy_key = str(row.get("policy_key") or "").strip()
        if not source_id or not workspace_id or not policy_key:
            continue
        existing = session.exec(
            select(GovernancePolicyRecord).where(
                GovernancePolicyRecord.workspace_id == workspace_id,
                GovernancePolicyRecord.policy_key == policy_key,
            )
        ).first()
        if existing is not None:
            mapping[source_id] = str(existing.id)
    return mapping


def _build_workflow_template_id_map(
    session: Session,
    dataset: dict[str, list[dict[str, Any]]],
    *,
    workspace_id_map: dict[str, str],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in dataset.get("workflow_templates", []):
        source_id = str(row.get("id") or "").strip()
        workspace_id = _coerce_uuid(
            workspace_id_map.get(str(row.get("workspace_id") or "").strip(), str(row.get("workspace_id") or "").strip())
        )
        template_key = str(row.get("template_key") or "").strip()
        if not source_id or not workspace_id or not template_key:
            continue
        existing = session.exec(
            select(WorkflowTemplateRecord).where(
                WorkflowTemplateRecord.workspace_id == workspace_id,
                WorkflowTemplateRecord.template_key == template_key,
            )
        ).first()
        if existing is not None:
            mapping[source_id] = str(existing.id)
    return mapping


def _is_user_id_field(field_name: str) -> bool:
    return field_name == "user_id" or field_name.endswith("_user_id")


def _is_workspace_id_field(field_name: str) -> bool:
    return field_name == "workspace_id" or field_name.endswith("_workspace_id")


def _rewrite_row(
    table_name: str,
    row: dict[str, Any],
    *,
    user_id_map: dict[str, str],
    workspace_id_map: dict[str, str],
    workspace_membership_id_map: dict[str, str],
    workspace_runtime_settings_id_map: dict[str, str],
    runtime_feature_flag_id_map: dict[str, str],
    governance_policy_id_map: dict[str, str],
    workflow_template_id_map: dict[str, str],
) -> dict[str, Any]:
    rewritten = dict(row)
    row_id = str(rewritten.get("id") or "").strip()

    if table_name == "users" and row_id in user_id_map:
        rewritten["id"] = user_id_map[row_id]
    if table_name == "workspaces" and row_id in workspace_id_map:
        rewritten["id"] = workspace_id_map[row_id]
    if table_name == "workspace_memberships" and row_id in workspace_membership_id_map:
        rewritten["id"] = workspace_membership_id_map[row_id]
    if table_name == "workspace_runtime_settings" and row_id in workspace_runtime_settings_id_map:
        rewritten["id"] = workspace_runtime_settings_id_map[row_id]
    if table_name == "runtime_feature_flags" and row_id in runtime_feature_flag_id_map:
        rewritten["id"] = runtime_feature_flag_id_map[row_id]
    if table_name == "governance_policies" and row_id in governance_policy_id_map:
        rewritten["id"] = governance_policy_id_map[row_id]
    if table_name == "workflow_templates" and row_id in workflow_template_id_map:
        rewritten["id"] = workflow_template_id_map[row_id]

    for key, value in list(rewritten.items()):
        if value in (None, ""):
            continue
        token = str(value).strip()
        if _is_user_id_field(key) and token in user_id_map:
            rewritten[key] = user_id_map[token]
        elif _is_workspace_id_field(key) and token in workspace_id_map:
            rewritten[key] = workspace_id_map[token]

    if table_name == "runtime_settings_audit":
        scope_id = str(rewritten.get("scope_id") or "").strip()
        if scope_id in workspace_id_map:
            rewritten["scope_id"] = workspace_id_map[scope_id]

    return rewritten


def _preferred_sort_key(table_name: str) -> tuple[int, str]:
    try:
        return (PREFERRED_TABLE_ORDER.index(table_name), table_name)
    except ValueError:
        return (len(PREFERRED_TABLE_ORDER), table_name)


def _upsert_row(connection, table: Table, row: dict[str, Any]) -> None:
    primary_key_columns = [column.name for column in table.primary_key.columns]
    if not primary_key_columns:
        raise RuntimeError(f"La tabla {table.name} no tiene primary key")
    statement = pg_insert(table).values(**row)
    update_payload = {
        column.name: statement.excluded[column.name]
        for column in table.columns
        if column.name not in primary_key_columns
    }
    statement = statement.on_conflict_do_update(
        index_elements=[table.c[name] for name in primary_key_columns],
        set_=update_payload,
    )
    connection.execute(statement)


def import_snapshot(snapshot_dir: Path) -> dict[str, Any]:
    dataset = _load_snapshot(snapshot_dir)
    database_url = _database_url()
    engine = create_engine(database_url, pool_pre_ping=True)

    with Session(engine) as session:
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
        runtime_feature_flag_id_map = _build_runtime_feature_flag_id_map(
            session,
            dataset,
            workspace_id_map=workspace_id_map,
        )
        governance_policy_id_map = _build_governance_policy_id_map(
            session,
            dataset,
            workspace_id_map=workspace_id_map,
        )
        workflow_template_id_map = _build_workflow_template_id_map(
            session,
            dataset,
            workspace_id_map=workspace_id_map,
        )

    metadata = MetaData()
    target_tables = sorted(dataset.keys(), key=_preferred_sort_key)
    metadata.reflect(bind=engine, only=[name for name in target_tables if dataset.get(name)])

    pending: dict[str, list[dict[str, Any]]] = {}
    for table_name, rows in dataset.items():
        rewritten_rows = [
            _rewrite_row(
                table_name,
                row,
                user_id_map=user_id_map,
                workspace_id_map=workspace_id_map,
                workspace_membership_id_map=workspace_membership_id_map,
                workspace_runtime_settings_id_map=workspace_runtime_settings_id_map,
                runtime_feature_flag_id_map=runtime_feature_flag_id_map,
                governance_policy_id_map=governance_policy_id_map,
                workflow_template_id_map=workflow_template_id_map,
            )
            for row in rows
        ]
        pending[table_name] = rewritten_rows

    imported_counts = {table_name: 0 for table_name in target_tables}
    passes: list[dict[str, Any]] = []

    for pass_number in range(1, 9):
        progress = False
        pass_report: dict[str, Any] = {"pass": pass_number, "tables": {}}
        for table_name in target_tables:
            rows = pending.get(table_name, [])
            if not rows:
                pass_report["tables"][table_name] = {"imported": 0, "remaining": 0}
                continue
            table = metadata.tables.get(table_name)
            if table is None:
                pass_report["tables"][table_name] = {"imported": 0, "remaining": len(rows), "skipped": "table_missing_locally"}
                continue

            remaining: list[dict[str, Any]] = []
            imported_now = 0
            last_error = ""
            for row in rows:
                try:
                    with engine.begin() as connection:
                        _upsert_row(connection, table, row)
                    imported_now += 1
                except IntegrityError as exc:
                    remaining.append(row)
                    last_error = str(exc)
            if imported_now:
                progress = True
            imported_counts[table_name] += imported_now
            pending[table_name] = remaining
            payload: dict[str, Any] = {"imported": imported_now, "remaining": len(remaining)}
            if last_error:
                payload["last_error"] = last_error
            pass_report["tables"][table_name] = payload
        passes.append(pass_report)
        if all(not rows for rows in pending.values()):
            break
        if not progress:
            break

    unresolved = {table_name: len(rows) for table_name, rows in pending.items() if rows}
    return {
        "snapshot_dir": str(snapshot_dir),
        "database_url_host": engine.url.host or "",
        "database_name": engine.url.database or "",
        "imported": imported_counts,
        "unresolved": unresolved,
        "passes": passes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Importa un snapshot materializado desde SQL Editor a la base local.")
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    args = parser.parse_args()

    snapshot_dir = args.snapshot_dir.resolve()
    if not snapshot_dir.exists():
        raise SystemExit(f"Snapshot directory not found: {snapshot_dir}")

    result = import_snapshot(snapshot_dir)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    if result["unresolved"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
