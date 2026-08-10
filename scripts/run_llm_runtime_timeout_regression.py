from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.llm_runtime.codex_cli.execution_service import CodexExecutionService
from app.services.llm_runtime.codex_cli.runtime_types import CodexProcessResult
from app.services.openai_builder import load_llm_runtime_settings


class TimeoutOutput(BaseModel):
    status: str


class TimeoutProbeExecutionService(CodexExecutionService):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.captured_timeouts: list[float] = []

    def run_process(
        self,
        *,
        command: list[str],
        workdir: Path,
        stdin_text: str,
        timeout_seconds: float,
    ) -> CodexProcessResult:
        self.captured_timeouts.append(timeout_seconds)
        return CodexProcessResult(
            command=command,
            workdir=workdir,
            stdout='{"status":"ok"}',
            stderr="",
            returncode=0,
        )


def main() -> int:
    settings = load_llm_runtime_settings().model_copy(deep=True)
    settings.codex_local.runner_id = "timeout-regression"
    service = TimeoutProbeExecutionService(settings, repo_root=REPO_ROOT)
    original = os.environ.get("F04_RUN_TIMEOUT_MS")
    os.environ["F04_RUN_TIMEOUT_MS"] = "2345"
    try:
        env_result = service.execute_structured_prompt(
            task_kind="f04",
            prompt="Return only valid JSON matching the schema. Set status to 'ok'.",
            output_model=TimeoutOutput,
        )
        override_result = service.execute_structured_prompt(
            task_kind="f04",
            prompt="Return only valid JSON matching the schema. Set status to 'ok'.",
            output_model=TimeoutOutput,
            timeout_ms=3456,
        )
    finally:
        if original is None:
            os.environ.pop("F04_RUN_TIMEOUT_MS", None)
        else:
            os.environ["F04_RUN_TIMEOUT_MS"] = original
    if env_result.status != "ok" or override_result.status != "ok":
        raise RuntimeError("Las corridas de timeout regression no devolvieron 'ok'.")
    if [round(value, 3) for value in service.captured_timeouts] != [2.345, 3.456]:
        raise RuntimeError(f"Timeouts propagados inesperados: {service.captured_timeouts}")
    print(
        json.dumps(
            {
                "ok": True,
                "captured_timeouts": [round(value, 3) for value in service.captured_timeouts],
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
