from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from time import sleep

from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.llm_runtime.codex_cli.execution_service import CodexExecutionService
from app.services.llm_runtime.codex_cli.runtime_types import CodexProcessResult
from app.services.openai_builder import load_llm_runtime_settings


class QueueOutput(BaseModel):
    status: str


class QueueProbeExecutionService(CodexExecutionService):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._lock = Lock()
        self.active = 0
        self.max_seen = 0

    def run_process(
        self,
        *,
        command: list[str],
        workdir: Path,
        stdin_text: str,
        timeout_seconds: float,
    ) -> CodexProcessResult:
        with self._lock:
            self.active += 1
            self.max_seen = max(self.max_seen, self.active)
        sleep(0.1)
        with self._lock:
            self.active -= 1
        return CodexProcessResult(
            command=command,
            workdir=workdir,
            stdout='{"status":"ok"}',
            stderr="",
            returncode=0,
        )


def run_case(*, max_concurrency: int, worker_count: int) -> int:
    settings = load_llm_runtime_settings().model_copy(deep=True)
    settings.codex_local.max_concurrency = max_concurrency
    settings.codex_local.runner_id = f"queue-regression-{max_concurrency}"
    service = QueueProbeExecutionService(settings, repo_root=REPO_ROOT)

    def worker(_: int) -> str:
        result = service.execute_structured_prompt(
            task_kind=f"queue_regression_{max_concurrency}",
            prompt="Return only valid JSON matching the schema. Set status to 'ok'.",
            output_model=QueueOutput,
        )
        return result.status

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        statuses = list(executor.map(worker, range(worker_count)))
    if any(status != "ok" for status in statuses):
        raise RuntimeError(f"Una corrida devolvio estado inesperado: {statuses}")
    return service.max_seen


def main() -> int:
    serialized_max = run_case(max_concurrency=1, worker_count=3)
    parallel_max = run_case(max_concurrency=2, worker_count=4)
    if serialized_max != 1:
        raise RuntimeError(f"La cola serial debio ver maximo 1 corrida simultanea y vio {serialized_max}.")
    if parallel_max > 2 or parallel_max < 2:
        raise RuntimeError(f"La cola con max_concurrency=2 debio ver exactamente 2 y vio {parallel_max}.")
    print(
        json.dumps(
            {
                "ok": True,
                "serialized_max": serialized_max,
                "parallel_max": parallel_max,
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
