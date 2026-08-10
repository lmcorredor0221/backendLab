from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings
from app.services.llm_runtime.codex_cli.execution_service import CodexExecutionService
from app.services.llm_runtime.release_gate import evaluate_release_gate
from app.services.llm_runtime.settings_migration import inspect_runtime_settings_migration
from app.services.openai_builder import load_llm_runtime_settings


OUTPUT_ROOT = BACKEND_ROOT / "runtime" / "codex-runtime-release-gate"


def now_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def run_script(python_path: Path, script_path: Path, output_dir: Path, step_name: str) -> dict[str, Any]:
    completed = subprocess.run(
        [str(python_path), str(script_path)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    stdout_path = output_dir / f"{step_name}.stdout.log"
    stderr_path = output_dir / f"{step_name}.stderr.log"
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")

    parsed_payload: Any = None
    stdout_lines = [line.strip() for line in (completed.stdout or "").splitlines() if line.strip()]
    if stdout_lines:
        candidate = stdout_lines[-1]
        try:
            parsed_payload = json.loads(candidate)
        except json.JSONDecodeError:
            parsed_payload = None

    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "payload": parsed_payload,
    }


def build_markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Codex Runtime Release Gate",
        "",
        f"- Fecha UTC: {summary['generated_at']}",
        f"- Runtime config: `{summary['config_path']}`",
        f"- Evidence dir: `{summary['evidence_dir']}`",
        f"- Current rollout stage: `{summary['release_gate']['current_stage']}`",
        f"- Overall OK: `{summary['release_gate']['overall_ok']}`",
        "",
        "## Checks",
        "",
    ]
    for step_name, payload in summary["checks"].items():
        lines.append(f"- {step_name}: `{'ok' if payload['ok'] else 'failed'}`")
    lines.extend(["", "## Promotion Path", ""])
    for stage in summary["release_gate"]["stages"]:
        lines.append(f"- {stage['stage']}: `{stage['status']}`")
        lines.append(f"  - {stage['summary']}")
    lines.extend(["", "## Transition Gates", ""])
    for transition in summary["release_gate"]["transitions"]:
        lines.append(f"- {transition['from']} -> {transition['to']}: `{transition['status']}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    output_dir = OUTPUT_ROOT / now_slug()
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    config_path = settings.llm_config_path
    python_path = BACKEND_ROOT / ".venv" / "Scripts" / "python.exe"
    runtime_settings = load_llm_runtime_settings()
    runtime_status_before_checks = CodexExecutionService(runtime_settings, repo_root=REPO_ROOT).get_runtime_status()

    migration = inspect_runtime_settings_migration(config_path)
    checks = {
        "migration": {
            "ok": not bool(migration["changed"]),
            "changed": bool(migration["changed"]),
            "config_path": str(config_path),
        },
        "smoke": run_script(python_path, BACKEND_ROOT / "scripts" / "run_codex_runtime_smoke.py", output_dir, "smoke"),
        "queue": run_script(
            python_path,
            BACKEND_ROOT / "scripts" / "run_codex_execution_queue_regression.py",
            output_dir,
            "queue",
        ),
        "fallback": run_script(
            python_path,
            BACKEND_ROOT / "scripts" / "run_codex_model_fallback_regression.py",
            output_dir,
            "fallback",
        ),
        "timeout": run_script(
            python_path,
            BACKEND_ROOT / "scripts" / "run_llm_runtime_timeout_regression.py",
            output_dir,
            "timeout",
        ),
        "walkthrough": run_script(
            python_path,
            BACKEND_ROOT / "scripts" / "run_codex_builder_provider_walkthrough.py",
            output_dir,
            "walkthrough",
        ),
    }
    runtime_status = CodexExecutionService(runtime_settings, repo_root=REPO_ROOT).get_runtime_status()
    release_gate = evaluate_release_gate(runtime_settings, runtime_status, checks)
    summary = {
        "ok": release_gate["overall_ok"],
        "generated_at": datetime.now(UTC).isoformat(),
        "config_path": str(config_path),
        "evidence_dir": str(output_dir),
        "runtime_status_before_checks": runtime_status_before_checks,
        "runtime_status": runtime_status,
        "checks": checks,
        "release_gate": release_gate,
    }

    write_json(output_dir / "runtime-status-before-checks.json", runtime_status_before_checks)
    write_json(output_dir / "runtime-status.json", runtime_status)
    write_json(output_dir / "migration-inspection.json", migration)
    write_json(output_dir / "summary.json", summary)
    write_text(output_dir / "summary.md", build_markdown_summary(summary))

    latest_dir = OUTPUT_ROOT / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    write_json(latest_dir / "summary.json", summary)
    write_text(latest_dir / "summary.md", build_markdown_summary(summary))
    write_text(latest_dir / "latest-run.txt", str(output_dir))

    print(json.dumps(summary, ensure_ascii=True))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
