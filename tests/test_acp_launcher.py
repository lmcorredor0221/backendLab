from __future__ import annotations

import json
import os
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest

from app.services.acp_generator import generate_acp_preview
from app.services.acp_zip_export import build_acp_zip
from tests.api_testkit import build_test_client
from tests.canonical_fixture_builder import build_full_session_snapshot


@pytest.fixture(scope="module")
def acp_launcher_preview():
    with pytest.MonkeyPatch.context() as monkeypatch:
        with build_test_client(monkeypatch) as client:
            snapshot = build_full_session_snapshot(
                client,
                "03-agent-with-knowledge-rag",
                "Caso ACP launcher portable",
            )
            return generate_acp_preview(snapshot)


def test_acp_preview_includes_framework_neutral_launcher_and_adapters(acp_launcher_preview) -> None:
    paths = {item.path for item in acp_launcher_preview.files}

    assert "ACP/launcher/launch-manifest.json" in paths
    assert "ACP/launcher/acp-launcher.py" in paths
    assert "ACP/launcher/start-acp.ps1" in paths
    assert "ACP/launcher/start-acp.bat" in paths
    assert "ACP/launcher/start-acp.sh" in paths
    assert "ACP/launcher/README.md" in paths
    assert "ACP/adapters/adapter-registry.json" in paths
    assert "ACP/adapters/framework-neutral-build-plan.md" in paths
    assert "ACP/adapters/codex-cli.md" in paths
    assert "ACP/adapters/claude-code.md" in paths
    assert "ACP/adapters/cursor.md" in paths
    assert "ACP/adapters/github-copilot.md" in paths
    assert "ACP/adapters/openai-agents-sdk.md" in paths


def test_acp_launcher_runs_outside_lean_and_writes_report(acp_launcher_preview, tmp_path: Path) -> None:
    zip_bytes = build_acp_zip(acp_launcher_preview)
    with ZipFile(BytesIO(zip_bytes)) as archive:
        archive.extractall(tmp_path)

    launcher_path = tmp_path / "ACP" / "launcher" / "acp-launcher.py"
    result = subprocess.run(
        [sys.executable, str(launcher_path), "--dry-run", "--no-open"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report_path = tmp_path / "ACP" / "launcher" / "launch-report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["launcher_version"] == "acp-launcher.v1"
    assert report["package_state"]["exists"] is True
    assert report["package_state"]["missing_files"] == []
    assert report["safety"] == {
        "installs_dependencies": False,
        "requires_lean_backend": False,
        "runs_build": False,
        "runs_destructive_commands": False,
    }
    assert report["detected_tools"]
    assert report["detected_prerequisites"]
    assert report["next_steps"]


def test_acp_launcher_guides_user_when_no_agentic_tool_is_detected(acp_launcher_preview, tmp_path: Path) -> None:
    zip_bytes = build_acp_zip(acp_launcher_preview)
    with ZipFile(BytesIO(zip_bytes)) as archive:
        archive.extractall(tmp_path)

    launcher_path = tmp_path / "ACP" / "launcher" / "acp-launcher.py"
    env = {**os.environ, "PATH": ""}
    result = subprocess.run(
        [sys.executable, str(launcher_path), "--dry-run", "--no-open"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "ACP" / "launcher" / "launch-report.json").read_text(encoding="utf-8"))
    assert report["recommendation"] is None
    assert all(item["available"] is False for item in report["detected_tools"])
    assert any("No se detecto herramienta agentica" in step for step in report["next_steps"])
