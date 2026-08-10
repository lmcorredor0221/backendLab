from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, inspect, text


BACKEND_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_VERSIONS_DIR = BACKEND_ROOT / "alembic" / "versions"


def _literal_assignment(tree: ast.Module, name: str) -> Any:
    for node in tree.body:
        value_node: ast.AST | None = None
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    value_node = node.value
                    break
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            value_node = node.value
        if value_node is None:
            continue
        try:
            return ast.literal_eval(value_node)
        except (ValueError, SyntaxError):
            return None
    return None


def _tuple_values(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, (tuple, list, set)):
        return {str(item) for item in value if item}
    return {str(value)}


def resolve_expected_alembic_heads(versions_dir: Path = ALEMBIC_VERSIONS_DIR) -> tuple[str, ...]:
    revisions: set[str] = set()
    down_revisions: set[str] = set()
    for path in versions_dir.glob("*.py"):
        if path.name.startswith("__"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        revision = _literal_assignment(tree, "revision")
        if isinstance(revision, str) and revision:
            revisions.add(revision)
        down_revisions.update(_tuple_values(_literal_assignment(tree, "down_revision")))
    heads = sorted(revisions - down_revisions)
    if not heads:
        raise RuntimeError(f"No Alembic head revision could be resolved from {versions_dir}")
    return tuple(heads)


def assert_alembic_head_applied(engine: Engine) -> None:
    inspector = inspect(engine)
    if "alembic_version" not in set(inspector.get_table_names()):
        raise RuntimeError(
            "schema_management_mode=alembic requires an initialized database. "
            "Run `alembic upgrade head` before starting the API."
        )

    with engine.connect() as connection:
        applied = {
            str(row[0])
            for row in connection.execute(text("SELECT version_num FROM alembic_version")).all()
            if row[0]
        }
    expected = set(resolve_expected_alembic_heads())
    if applied != expected:
        raise RuntimeError(
            "schema_management_mode=alembic detected an out-of-date database. "
            f"Expected Alembic head(s) {sorted(expected)}, found {sorted(applied)}. "
            "Run `alembic upgrade head` or rollback intentionally before starting the API."
        )
