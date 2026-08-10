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
from app.services.llm_runtime.codex_cli.runtime_types import CodexProcessResult
from app.services.openai_builder import load_llm_runtime_settings


class FallbackOutput(BaseModel):
    status: str


class FallbackProbeExecutionService(CodexExecutionService):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seen_models: list[str] = []

    def run_process(
        self,
        *,
        command: list[str],
        workdir: Path,
        stdin_text: str,
        timeout_seconds: float,
    ) -> CodexProcessResult:
        model = command[command.index("--model") + 1]
        self.seen_models.append(model)
        if len(self.seen_models) == 1:
            return CodexProcessResult(
                command=command,
                workdir=workdir,
                stdout="",
                stderr="selected model is at capacity",
                returncode=1,
            )
        return CodexProcessResult(
            command=command,
            workdir=workdir,
            stdout='{"status":"fallback-ok"}',
            stderr="",
            returncode=0,
        )


def latest_fallback_workspace() -> Path:
    runtime_root = BACKEND_ROOT / "runtime" / "codex-workspaces"
    candidates = sorted(
        (item for item in runtime_root.iterdir() if item.is_dir() and "fallback-regression" in item.name),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError("No se encontro workspace de fallback regression.")
    return candidates[0]


def main() -> int:
    settings = load_llm_runtime_settings().model_copy(deep=True)
    settings.codex_local.fallback_models = ["gpt-5.5-mini"]
    settings.codex_local.runner_id = "fallback-regression"
    service = FallbackProbeExecutionService(settings, repo_root=REPO_ROOT)
    result = service.execute_structured_prompt(
        task_kind="fallback_regression",
        prompt="Return only valid JSON matching the schema. Set status to 'fallback-ok'.",
        output_model=FallbackOutput,
    )
    workspace = latest_fallback_workspace()
    invocation = json.loads((workspace / "invocation.json").read_text(encoding="utf-8"))
    if result.status != "fallback-ok":
        raise RuntimeError(f"Resultado inesperado de fallback: {result.status}")
    if service.seen_models != [settings.codex_local.model, "gpt-5.5-mini"]:
        raise RuntimeError(f"Secuencia de modelos inesperada: {service.seen_models}")
    if invocation["selected_model"] != "gpt-5.5-mini":
        raise RuntimeError(f"El modelo final esperado era gpt-5.5-mini y fue {invocation['selected_model']}")
    print(
        json.dumps(
            {
                "ok": True,
                "attempted_models": invocation["attempted_models"],
                "selected_model": invocation["selected_model"],
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
