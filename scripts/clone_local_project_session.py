from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import MetaData, Table, and_, create_engine, select, update

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.config import get_settings


EXCLUDED_SESSION_TABLES = {
    # Project clones must not duplicate or mutate the global/shared knowledge base.
    "knowledge_documents",
    "knowledge_sections",
}

WORKSPACE_UNIQUE_IDEMPOTENCY_TABLES = {
    "deliverable_generation_jobs_v1",
    "diagram_generation_jobs_v3",
    "product_build_runs_v1",
}

SELF_REFERENCE_FIELDS = {
    "journey_stage_artifacts": {"based_on_artifact_id", "superseded_by_artifact_id"},
}


def _json_default(value: Any) -> str:
    if isinstance(value, (datetime, UUID)):
        return str(value)
    return repr(value)


def _row_dict(row: Any) -> dict[str, Any]:
    return dict(row._mapping)


def _pk_id(table: Table) -> Any | None:
    if "id" not in table.c:
        return None
    pk_cols = list(table.primary_key.columns)
    if len(pk_cols) == 1 and pk_cols[0].name == "id":
        return table.c.id
    return None


def _stringify(value: Any) -> str:
    if isinstance(value, UUID):
        return str(value)
    return str(value)


def _remap_value(value: Any, replacements: dict[str, str]) -> Any:
    if value is None:
        return None
    if isinstance(value, UUID):
        mapped = replacements.get(str(value))
        return UUID(mapped) if mapped else value
    if isinstance(value, dict):
        return {key: _remap_value(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_remap_value(item, replacements) for item in value]
    if isinstance(value, tuple):
        return tuple(_remap_value(item, replacements) for item in value)
    if isinstance(value, str):
        mapped = replacements.get(value)
        if mapped is not None:
            return mapped
        rewritten = value
        for source, target in replacements.items():
            if source in rewritten:
                rewritten = rewritten.replace(source, target)
        return rewritten
    return value


def _select_rows_by_ids(connection: Any, table: Table, ids: set[str]) -> list[dict[str, Any]]:
    pk = _pk_id(table)
    if pk is None or not ids:
        return []
    rows = connection.execute(select(table).where(pk.in_(list(ids)))).all()
    return [_row_dict(row) for row in rows]


def _select_rows_for_source(connection: Any, table: Table, source_session_id: UUID) -> list[dict[str, Any]]:
    if "session_id" not in table.c:
        return []
    rows = connection.execute(select(table).where(table.c.session_id == source_session_id)).all()
    return [_row_dict(row) for row in rows]


def _collect_rows(
    connection: Any,
    metadata: MetaData,
    *,
    source_session_id: UUID,
) -> dict[str, list[dict[str, Any]]]:
    rows_by_table: dict[str, list[dict[str, Any]]] = {}
    sessions = metadata.tables["sessions"]
    source_session = connection.execute(select(sessions).where(sessions.c.id == source_session_id)).first()
    if source_session is None:
        raise RuntimeError(f"No existe la sesion fuente local {source_session_id}")
    rows_by_table["sessions"] = [_row_dict(source_session)]

    for table_name, table in sorted(metadata.tables.items()):
        if table_name == "sessions" or table_name in EXCLUDED_SESSION_TABLES:
            continue
        if "session_id" not in table.c:
            continue
        rows = _select_rows_for_source(connection, table, source_session_id)
        if rows:
            rows_by_table[table_name] = rows

    # Include dependent child records that do not carry session_id, for example
    # skill_run_artifacts -> skill_runs. Iterate to catch second-level children too.
    selected_ids: dict[str, set[str]] = defaultdict(set)
    for table_name, rows in rows_by_table.items():
        table = metadata.tables[table_name]
        if _pk_id(table) is None:
            continue
        selected_ids[table_name].update(str(row["id"]) for row in rows if row.get("id") is not None)

    changed = True
    while changed:
        changed = False
        for table_name, table in sorted(metadata.tables.items()):
            if table_name in rows_by_table or table_name in EXCLUDED_SESSION_TABLES:
                continue
            pk = _pk_id(table)
            if pk is None:
                continue
            matched_ids: set[str] = set()
            for fk in table.foreign_keys:
                parent_table = fk.column.table.name
                parent_ids = selected_ids.get(parent_table)
                if not parent_ids:
                    continue
                rows = connection.execute(
                    select(table.c.id).where(fk.parent.in_(list(parent_ids)))
                ).all()
                matched_ids.update(str(row[0]) for row in rows if row[0] is not None)
            if not matched_ids:
                continue
            rows = _select_rows_by_ids(connection, table, matched_ids)
            if rows:
                rows_by_table[table_name] = rows
                selected_ids[table_name].update(matched_ids)
                changed = True

    return rows_by_table


def _topological_order(metadata: MetaData, table_names: set[str]) -> list[str]:
    dependencies: dict[str, set[str]] = {name: set() for name in table_names}
    dependents: dict[str, set[str]] = {name: set() for name in table_names}
    for table_name in table_names:
        table = metadata.tables[table_name]
        for fk in table.foreign_keys:
            parent_table = fk.column.table.name
            if parent_table == table_name or parent_table not in table_names:
                continue
            dependencies[table_name].add(parent_table)
            dependents[parent_table].add(table_name)

    ready = deque(sorted(name for name, deps in dependencies.items() if not deps))
    ordered: list[str] = []
    while ready:
        name = ready.popleft()
        ordered.append(name)
        for child in sorted(dependents[name]):
            dependencies[child].discard(name)
            if not dependencies[child]:
                ready.append(child)

    if len(ordered) != len(table_names):
        remaining = sorted(table_names.difference(ordered))
        ordered.extend(remaining)
    if "sessions" in ordered:
        ordered.remove("sessions")
        ordered.insert(0, "sessions")
    return ordered


def _prepare_replacements(
    metadata: MetaData,
    rows_by_table: dict[str, list[dict[str, Any]]],
    *,
    source_session_id: UUID,
    target_session_id: UUID,
) -> dict[str, str]:
    replacements = {str(source_session_id): str(target_session_id)}
    for table_name, rows in rows_by_table.items():
        table = metadata.tables[table_name]
        if _pk_id(table) is None:
            continue
        for row in rows:
            source_id = row.get("id")
            if source_id is None:
                continue
            if table_name == "sessions":
                replacements[str(source_id)] = str(target_session_id)
            else:
                replacements[str(source_id)] = str(uuid4())
    return replacements


def _rewrite_row(
    table_name: str,
    row: dict[str, Any],
    *,
    replacements: dict[str, str],
    target_session_id: UUID,
    title_suffix: str,
    null_self_refs: bool,
) -> dict[str, Any]:
    rewritten = copy.deepcopy(row)
    for key, value in list(rewritten.items()):
        rewritten[key] = _remap_value(value, replacements)

    if "id" in rewritten and str(row.get("id")) in replacements:
        rewritten["id"] = UUID(replacements[str(row["id"])])
    if "session_id" in rewritten:
        rewritten["session_id"] = target_session_id

    if table_name in WORKSPACE_UNIQUE_IDEMPOTENCY_TABLES and rewritten.get("idempotency_key"):
        rewritten["idempotency_key"] = f"{rewritten['idempotency_key']}::clone::{str(target_session_id)[:8]}"

    if table_name == "sessions":
        original_title = str(row.get("title") or "Proyecto sin titulo").strip()
        rewritten["id"] = target_session_id
        rewritten["title"] = f"{original_title} {title_suffix}".strip()
        rewritten["suggested_title"] = rewritten["title"]
        rewritten["title_source"] = "migrated"
        rewritten["row_version"] = int(row.get("row_version") or 1) + 1
        rewritten["archived_at"] = None
        rewritten["archived_by_user_id"] = None
        rewritten["deleted_at"] = None
        rewritten["deleted_by_user_id"] = None

    if null_self_refs:
        for field in SELF_REFERENCE_FIELDS.get(table_name, set()):
            if field in rewritten:
                rewritten[field] = None
    return rewritten


def _update_self_references(
    connection: Any,
    table: Table,
    table_name: str,
    rows: list[dict[str, Any]],
    *,
    replacements: dict[str, str],
) -> None:
    fields = SELF_REFERENCE_FIELDS.get(table_name, set())
    if not fields:
        return
    for row in rows:
        values: dict[str, Any] = {}
        for field in fields:
            source_value = row.get(field)
            if source_value is None:
                continue
            mapped = replacements.get(str(source_value))
            if mapped:
                values[field] = UUID(mapped)
        if not values:
            continue
        target_id = UUID(replacements[str(row["id"])])
        connection.execute(update(table).where(table.c.id == target_id).values(**values))


def clone_session(
    *,
    source_session_id: UUID,
    target_session_id: UUID,
    title_suffix: str,
    execute: bool,
) -> dict[str, Any]:
    settings = get_settings()
    engine = create_engine(settings.database_url, echo=False, pool_pre_ping=True)
    metadata = MetaData()
    metadata.reflect(bind=engine)

    with engine.begin() as connection:
        rows_by_table = _collect_rows(connection, metadata, source_session_id=source_session_id)
        replacements = _prepare_replacements(
            metadata,
            rows_by_table,
            source_session_id=source_session_id,
            target_session_id=target_session_id,
        )
        ordered_tables = _topological_order(metadata, set(rows_by_table))
        counts = {table_name: len(rows_by_table[table_name]) for table_name in ordered_tables}

        if execute:
            for table_name in ordered_tables:
                table = metadata.tables[table_name]
                rows = rows_by_table[table_name]
                rewritten_rows = [
                    _rewrite_row(
                        table_name,
                        row,
                        replacements=replacements,
                        target_session_id=target_session_id,
                        title_suffix=title_suffix,
                        null_self_refs=table_name in SELF_REFERENCE_FIELDS,
                    )
                    for row in rows
                ]
                if rewritten_rows:
                    connection.execute(table.insert(), rewritten_rows)
            for table_name in ordered_tables:
                _update_self_references(
                    connection,
                    metadata.tables[table_name],
                    table_name,
                    rows_by_table[table_name],
                    replacements=replacements,
                )

    return {
        "database_host": settings.database_url.split("@")[-1].split("/")[0] if "@" in settings.database_url else "local",
        "source_session_id": str(source_session_id),
        "target_session_id": str(target_session_id),
        "execute": execute,
        "title_suffix": title_suffix,
        "tables": counts,
        "total_rows": sum(counts.values()),
        "excluded_session_tables": sorted(EXCLUDED_SESSION_TABLES),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Clona una sesion/proyecto local para pruebas E2E reproducibles.")
    parser.add_argument("--source-session-id", required=True, type=UUID)
    parser.add_argument("--target-session-id", type=UUID, default=None)
    parser.add_argument("--title-suffix", default="[Pivote calidad ACP146]")
    parser.add_argument("--execute", action="store_true", help="Inserta la copia. Sin este flag solo muestra dry-run.")
    args = parser.parse_args()

    result = clone_session(
        source_session_id=args.source_session_id,
        target_session_id=args.target_session_id or uuid4(),
        title_suffix=args.title_suffix,
        execute=args.execute,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
