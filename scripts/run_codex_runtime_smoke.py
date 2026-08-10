from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.llm_runtime.codex_cli.execution_service import CodexExecutionService
from app.services.openai_builder import load_llm_runtime_settings


EXPECTED_TOKEN = "CODEX_RUNTIME_SMOKE_OK"


class SmokeOutput(BaseModel):
    token: str


def latest_runtime_smoke_workspace() -> str:
    runtime_root = BACKEND_ROOT / "runtime" / "codex-workspaces"
    candidates = sorted(
        (item for item in runtime_root.iterdir() if item.is_dir() and "runtime-smoke" in item.name),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else ""


def main() -> int:
    settings = load_llm_runtime_settings()
    service = CodexExecutionService(settings, repo_root=REPO_ROOT)
    result = service.execute_structured_prompt(
        task_kind="runtime_smoke",
        prompt=(
            "Return only valid JSON matching the schema. "
            f"Set token to '{EXPECTED_TOKEN}'."
        ),
        output_model=SmokeOutput,
    )
    if result.token != EXPECTED_TOKEN:
        raise RuntimeError(f"Smoke token inesperado: {result.token}")
    print(
        json.dumps(
            {
                "ok": True,
                "token": result.token,
                "workspace": latest_runtime_smoke_workspace(),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
