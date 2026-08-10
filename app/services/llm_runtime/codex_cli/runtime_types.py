from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CodexPromptWorkspace:
    run_id: str
    task_kind: str
    root_dir: Path
    agents_path: Path
    schema_path: Path
    prompt_path: Path
    read_order_path: Path
    knowledge_manifest_path: Path
    required_knowledge_dir: Path
    candidate_knowledge_dir: Path
    output_dir: Path
    last_message_path: Path
    structured_output_path: Path
    stdout_path: Path
    stderr_path: Path
    invocation_path: Path


@dataclass(frozen=True)
class CodexProcessResult:
    command: list[str]
    workdir: Path
    stdout: str
    stderr: str
    returncode: int


class CodexRuntimeErrorCode(StrEnum):
    config_error = "config_error"
    binary_not_found = "binary_not_found"
    auth_error = "auth_error"
    timeout = "timeout"
    parse_error = "parse_error"
    output_missing = "output_missing"
    fallback_exhausted = "fallback_exhausted"
    execution_failed = "execution_failed"
    queue_rejected = "queue_rejected"
    model_capacity = "model_capacity"
    invalid_schema = "invalid_schema"


@dataclass(frozen=True)
class CodexAttemptRecord:
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
class CodexExecutionMetrics:
    duration_ms: int
    queue_wait_ms: int
    output_size_bytes: int
    stdout_bytes: int
    stderr_bytes: int
    exit_code: int | None


@dataclass(frozen=True)
class CodexExecutionAuditRecord:
    run_id: str
    task_kind: str
    runner_id: str
    status: str
    requested_model: str
    selected_model: str | None
    attempted_models: list[str]
    fallback_used: bool
    profile: str
    workspace_root: str
    workdir: str
    prompt_path: str
    schema_path: str
    last_message_path: str
    structured_output_path: str
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
    attempts: list[CodexAttemptRecord] = field(default_factory=list)
    metrics: CodexExecutionMetrics | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class CodexExecutionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: CodexRuntimeErrorCode = CodexRuntimeErrorCode.execution_failed,
        recoverable: bool = False,
        result: CodexProcessResult | None = None,
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
