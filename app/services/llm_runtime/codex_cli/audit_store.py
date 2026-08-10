from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from app.services.llm_runtime.codex_cli.runtime_types import CodexExecutionAuditRecord, CodexPromptWorkspace


class CodexAuditStore:
    def __init__(self, *, audit_log_path: Path | None = None) -> None:
        self.audit_log_path = audit_log_path

    def persist(
        self,
        *,
        workspace: CodexPromptWorkspace,
        record: CodexExecutionAuditRecord,
        append_to_log: bool = False,
    ) -> None:
        payload = self._normalize(record)
        workspace.invocation_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        if append_to_log:
            audit_log_path = self.audit_log_path or (workspace.root_dir.parent / "runtime-audit.jsonl")
            audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            with audit_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True))
                handle.write("\n")

    def _normalize(self, value: Any) -> Any:
        if is_dataclass(value):
            return self._normalize(asdict(value))
        if isinstance(value, dict):
            return {str(key): self._normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._normalize(item) for item in value]
        if isinstance(value, tuple):
            return [self._normalize(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Enum):
            return value.value
        return value
