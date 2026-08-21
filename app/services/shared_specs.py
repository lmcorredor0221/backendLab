from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Mapping


SHARED_SPECS_ENV_VAR = "LEAN_SHARED_SPECS_DIR"


def backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def shared_specs_candidates(
    *,
    backend_repo_root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[Path, ...]:
    repo_root = backend_repo_root or backend_root()
    environment = env or os.environ
    candidates: list[Path] = []

    override = str(environment.get(SHARED_SPECS_ENV_VAR, "")).strip()
    if override:
        candidates.append(Path(override).expanduser())

    candidates.extend(
        [
            repo_root / "shared_specs",
            repo_root.parent / "shared_specs",
        ]
    )
    return tuple(_unique_paths(candidates))


def find_shared_specs_dir(
    *,
    backend_repo_root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    candidates = shared_specs_candidates(backend_repo_root=backend_repo_root, env=env)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"shared_specs directory not found. Checked: {checked}")


@lru_cache(maxsize=1)
def resolve_shared_specs_dir() -> Path:
    return find_shared_specs_dir()


def resolve_shared_spec_path(*parts: str) -> Path:
    path = resolve_shared_specs_dir().joinpath(*parts)
    if not path.is_file():
        raise FileNotFoundError(f"shared_specs file not found: {path}")
    return path
