from __future__ import annotations

import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class AgyRunWorkspace:
    """
    Directorio efimero de trabajo para un run de Antigravity CLI.

    Estructura:
      <runtime_root>/<run_id>/
        prompt.md          – prompt enviado a agy via stdin
        output.md          – archivo de salida escrito por agy (--output)
        stdout.log         – stdout capturado del proceso
        stderr.log         – stderr capturado del proceso
        invocation.json    – auditoria del run
    """

    run_id: str
    task_kind: str
    root_dir: Path
    prompt_path: Path
    output_path: Path
    stdout_path: Path
    stderr_path: Path


class AgyPromptWorkspaceBuilder:
    """
    Construye y limpia directorios de trabajo para runs de Antigravity CLI.

    Sigue el mismo patron de context manager que CodexPromptWorkspaceBuilder:
    el workspace existe durante el bloque `with` y puede limpiarse al salir.
    """

    def __init__(self, *, runtime_root: Path) -> None:
        self.runtime_root = runtime_root

    @contextmanager
    def build(self, *, task_kind: str) -> Iterator[AgyRunWorkspace]:
        run_id = str(uuid.uuid4())
        root_dir = self.runtime_root / run_id
        root_dir.mkdir(parents=True, exist_ok=True)

        workspace = AgyRunWorkspace(
            run_id=run_id,
            task_kind=task_kind,
            root_dir=root_dir,
            prompt_path=root_dir / "prompt.md",
            output_path=root_dir / "output.md",
            stdout_path=root_dir / "stdout.log",
            stderr_path=root_dir / "stderr.log",
        )

        # Inicializar archivos de log vacios
        workspace.stdout_path.write_text("", encoding="utf-8")
        workspace.stderr_path.write_text("", encoding="utf-8")
        workspace.output_path.write_text("", encoding="utf-8")

        yield workspace
