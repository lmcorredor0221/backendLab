from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class AgyRuntimeErrorCode(StrEnum):
    config_error = "config_error"
    binary_not_found = "binary_not_found"
    auth_error = "auth_error"
    timeout = "timeout"
    parse_error = "parse_error"
    output_missing = "output_missing"
    fallback_exhausted = "fallback_exhausted"
    execution_failed = "execution_failed"
    queue_rejected = "queue_rejected"
    quota_exceeded = "quota_exceeded"
    rate_limited = "rate_limited"
    invalid_schema = "invalid_schema"


@dataclass(frozen=True)
class AgyProcessResult:
    """Resultado bruto de un proceso agy."""

    command: list[str]
    workdir: Path
    stdout: str
    stderr: str
    returncode: int


@dataclass(frozen=True)
class AgyAttemptRecord:
    """Registro de un intento de ejecucion dentro del loop de fallback."""

    attempt_number: int
    model: str
    status: str
    started_at: str
    finished_at: str
    duration_ms: int
    returncode: int | None
    error_code: str | None = None
    error_message: str | None = None
    payload_source: str | None = None
    retryable: bool = False
    output_size_bytes: int = 0


@dataclass(frozen=True)
class AgyExecutionMetrics:
    """Metricas de una ejecucion completada."""

    duration_ms: int
    queue_wait_ms: int
    output_size_bytes: int
    stdout_bytes: int
    stderr_bytes: int
    exit_code: int | None


@dataclass(frozen=True)
class AgyExecutionAuditRecord:
    """Registro de auditoria JSONL de una ejecucion Antigravity CLI."""

    run_id: str
    task_kind: str
    runner_id: str
    status: str
    requested_model: str
    selected_model: str | None
    attempted_models: list[str]
    fallback_used: bool
    effort: str
    workspace_root: str
    workdir: str
    prompt_path: str
    output_path: str
    stdout_path: str
    stderr_path: str
    started_at: str
    finished_at: str | None
    returncode: int | None
    payload_source: str | None
    command: list[str]
    error: str | None
    error_code: str | None
    recoverable: bool
    attempts: list[AgyAttemptRecord] = field(default_factory=list)
    metrics: AgyExecutionMetrics | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AgyExecutionError(RuntimeError):
    """Error de ejecucion del runtime Antigravity CLI."""

    def __init__(
        self,
        message: str,
        *,
        code: AgyRuntimeErrorCode = AgyRuntimeErrorCode.execution_failed,
        recoverable: bool = False,
        result: AgyProcessResult | None = None,
        attempted_models: list[str] | None = None,
        selected_model: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.recoverable = recoverable
        self.result = result
        self.attempted_models = attempted_models or []
        self.selected_model = selected_model
        self.detail = detail or {}
