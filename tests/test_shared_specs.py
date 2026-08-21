from __future__ import annotations

from pathlib import Path

import pytest

from app.services.shared_specs import (
    SHARED_SPECS_ENV_VAR,
    find_shared_specs_dir,
    resolve_shared_spec_path,
    resolve_shared_specs_dir,
)


def test_find_shared_specs_dir_prefers_backend_repo_copy(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    backend_root = workspace_root / "backend"
    backend_specs = backend_root / "shared_specs"
    workspace_specs = workspace_root / "shared_specs"
    backend_specs.mkdir(parents=True)
    workspace_specs.mkdir(parents=True)

    assert find_shared_specs_dir(backend_repo_root=backend_root, env={}) == backend_specs


def test_find_shared_specs_dir_falls_back_to_workspace_parent(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    backend_root = workspace_root / "backend"
    workspace_specs = workspace_root / "shared_specs"
    backend_root.mkdir(parents=True)
    workspace_specs.mkdir(parents=True)

    assert find_shared_specs_dir(backend_repo_root=backend_root, env={}) == workspace_specs


def test_find_shared_specs_dir_uses_env_override(tmp_path: Path) -> None:
    backend_root = tmp_path / "workspace" / "backend"
    override_specs = tmp_path / "override-shared-specs"
    backend_root.mkdir(parents=True)
    override_specs.mkdir(parents=True)

    assert find_shared_specs_dir(
        backend_repo_root=backend_root,
        env={SHARED_SPECS_ENV_VAR: str(override_specs)},
    ) == override_specs


def test_resolve_shared_spec_path_raises_for_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend_root = tmp_path / "backend"
    specs_root = backend_root / "shared_specs"
    specs_root.mkdir(parents=True)

    monkeypatch.setattr("app.services.shared_specs.backend_root", lambda: backend_root)
    resolve_shared_specs_dir.cache_clear()
    with pytest.raises(FileNotFoundError, match="missing.json"):
        resolve_shared_spec_path("missing.json")
    resolve_shared_specs_dir.cache_clear()
