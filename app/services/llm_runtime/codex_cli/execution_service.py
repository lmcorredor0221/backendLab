from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from app.core.config import get_settings
from app.models import LLMRuntimeSettings
from app.services.llm_runtime.codex_cli.audit_store import CodexAuditStore
from app.services.llm_runtime.codex_cli.context_assembler import (
    CodexContextAssembler,
    CodexContextAssembly,
    CodexContextRequest,
)
from app.services.llm_runtime.codex_cli.fallback_policy import CodexFallbackPolicy
from app.services.llm_runtime.codex_cli.queue_service import CodexExecutionQueueService
from app.services.llm_runtime.codex_cli.runtime_types import (
    CodexAttemptRecord,
    CodexExecutionAuditRecord,
    CodexExecutionError,
    CodexExecutionMetrics,
    CodexProcessResult,
    CodexPromptWorkspace,
    CodexRuntimeErrorCode,
)
from app.services.llm_runtime.codex_cli.workspace_builder import CodexPromptWorkspaceBuilder


def resolve_codex_executable_path(command: str) -> str | None:
    normalized = command.strip()
    if not normalized:
        return None
    candidate = Path(os.path.expandvars(os.path.expanduser(normalized)))
    bundled = _resolve_bundled_codex_executable(normalized)
    if candidate.exists():
        if bundled and _is_windowsapps_codex_alias(candidate):
            return bundled
        return str(candidate)
    executable = shutil.which(normalized)
    if executable:
        if bundled and _is_windowsapps_codex_alias(executable):
            return bundled
        return executable
    return bundled


def _is_windowsapps_codex_alias(path: str | Path) -> bool:
    normalized = str(path).replace("/", "\\").lower()
    return "\\windowsapps\\" in normalized and Path(path).name.lower() in {"codex", "codex.exe", "codex.cmd"}


def _resolve_bundled_codex_executable(command: str) -> str | None:
    command_name = Path(command).name.lower()
    if command_name not in {"codex", "codex.exe", "codex.cmd"}:
        return None

    extension_roots = [
        Path.home() / ".vscode" / "extensions",
        Path.home() / ".vscode-insiders" / "extensions",
    ]
    user_profile = os.getenv("USERPROFILE", "").strip()
    if user_profile:
        profile_root = Path(user_profile)
        extension_roots.extend(
            [
                profile_root / ".vscode" / "extensions",
                profile_root / ".vscode-insiders" / "extensions",
            ]
        )

    seen_roots: set[Path] = set()
    for extension_root in extension_roots:
        if extension_root in seen_roots or not extension_root.exists():
            continue
        seen_roots.add(extension_root)
        candidates = sorted(
            list(extension_root.glob("openai.chatgpt-*/bin/**/codex.exe"))
            + list(extension_root.glob("openai.chatgpt-*/bin/**/codex.cmd")),
            key=lambda path: str(path).lower(),
            reverse=True,
        )
        for candidate_path in candidates:
            if candidate_path.exists():
                return str(candidate_path)

    return None


class CodexExecutionService:
    def __init__(
        self,
        runtime_settings: LLMRuntimeSettings,
        *,
        repo_root: Path | None = None,
        workspace_builder: CodexPromptWorkspaceBuilder | None = None,
        fallback_policy: CodexFallbackPolicy | None = None,
        queue_service: CodexExecutionQueueService | None = None,
        audit_store: CodexAuditStore | None = None,
    ) -> None:
        self.runtime_settings = runtime_settings
        self.repo_root = repo_root or Path(__file__).resolve().parents[5]
        settings = get_settings()
        self.workspace_builder = workspace_builder or CodexPromptWorkspaceBuilder(
            runtime_root=settings.llm_config_path.parent / "codex-workspaces"
        )
        self.context_assembler = CodexContextAssembler(repo_root=self.repo_root)
        self.fallback_policy = fallback_policy or CodexFallbackPolicy(runtime_settings)
        self.queue_service = queue_service or CodexExecutionQueueService()
        self.audit_store = audit_store or CodexAuditStore()

    def resolve_executable(self) -> str:
        return (
            resolve_codex_executable_path(self.runtime_settings.codex_local.command)
            or self.runtime_settings.codex_local.command
        )

    def resolve_timeout_ms(self, *, task_kind: str, timeout_ms: int | None = None) -> int:
        env_key = self._build_task_timeout_env_key(task_kind)
        env_value = os.getenv(env_key, "").strip()
        if env_value:
            try:
                return max(1_000, int(env_value))
            except ValueError:
                pass
        if timeout_ms is not None:
            return max(1_000, int(timeout_ms))
        return max(1_000, self.runtime_settings.codex_local.timeout_ms)

    def resolve_max_concurrency(self) -> int:
        return max(1, self.runtime_settings.codex_local.max_concurrency)

    def _build_task_timeout_env_key(self, task_kind: str) -> str:
        normalized = "".join(character if character.isalnum() else "_" for character in task_kind.upper())
        while "__" in normalized:
            normalized = normalized.replace("__", "_")
        return f"{normalized.strip('_') or 'DEFAULT'}_RUN_TIMEOUT_MS"

    def resolve_runtime_root(self) -> Path:
        runtime_root = getattr(self.workspace_builder, "runtime_root", None)
        if isinstance(runtime_root, Path):
            return runtime_root
        return get_settings().llm_config_path.parent / "codex-workspaces"

    def resolve_codex_home_path(self) -> Path:
        configured = os.getenv("CODEX_HOME", "").strip()
        return Path(configured) if configured else (Path.home() / ".codex")

    def resolve_auth_mode(self) -> tuple[str, bool]:
        configured_mode = self.runtime_settings.codex_local.auth_mode.value
        has_access_token = bool(os.getenv("CODEX_ACCESS_TOKEN") or os.getenv("CODEX_API_KEY"))
        has_session = (self.resolve_codex_home_path() / "auth.json").exists()
        if configured_mode == "chatgpt_access_token":
            return configured_mode, has_access_token
        if configured_mode == "chatgpt_session":
            return configured_mode, has_session
        if has_access_token:
            return "chatgpt_access_token", True
        if has_session:
            return "chatgpt_session", True
        return "unknown", False

    def resolve_version(self) -> str | None:
        executable = resolve_codex_executable_path(self.runtime_settings.codex_local.command)
        if not executable:
            return None
        try:
            completed = subprocess.run(
                [executable, "--version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=15,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
            )
        except OSError:
            return None
        output = (completed.stdout or completed.stderr or "").strip()
        return output.splitlines()[0].strip() if output else None

    def build_smoke_command(self) -> str:
        python_path = self.repo_root / "backend" / ".venv" / "Scripts" / "python.exe"
        script_path = self.repo_root / "backend" / "scripts" / "run_codex_runtime_smoke.py"
        if python_path.exists():
            return f"{python_path} {script_path}"
        return f"python {script_path}"

    def read_last_known_result(self) -> dict[str, Any] | None:
        audit_log_path = self.resolve_runtime_root() / "runtime-audit.jsonl"
        if not audit_log_path.exists():
            return None
        lines = [line.strip() for line in audit_log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            return None
        payload = json.loads(lines[-1])
        metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
        return {
            "run_id": payload.get("run_id"),
            "task_kind": payload.get("task_kind"),
            "status": payload.get("status"),
            "finished_at": payload.get("finished_at"),
            "selected_model": payload.get("selected_model"),
            "attempted_models": payload.get("attempted_models", []),
            "fallback_used": payload.get("fallback_used"),
            "error_code": payload.get("error_code"),
            "recoverable": payload.get("recoverable"),
            "exit_code": payload.get("returncode"),
            "duration_ms": metrics.get("duration_ms"),
            "queue_wait_ms": metrics.get("queue_wait_ms"),
            "workspace_root": payload.get("workspace_root"),
        }

    def get_runtime_status(self) -> dict[str, Any]:
        executable = resolve_codex_executable_path(self.runtime_settings.codex_local.command)
        auth_mode, auth_detected = self.resolve_auth_mode()
        version = self.resolve_version()
        configured_model = self.runtime_settings.codex_local.model.strip() or None
        configured_fallbacks = list(self.runtime_settings.codex_local.fallback_models)
        last_known_result = self.read_last_known_result()
        smoke_blocking_reasons: list[str] = []
        if not executable:
            smoke_blocking_reasons.append("No se pudo resolver el ejecutable Codex configurado.")
        if not configured_model:
            smoke_blocking_reasons.append("No hay modelo default configurado para codex_local.")
        if not auth_detected:
            smoke_blocking_reasons.append("No se detecto autenticacion utilizable para Codex.")
        smoke_ready = not smoke_blocking_reasons
        status = "healthy" if smoke_ready else "degraded"
        recommendation = (
            f"Ejecuta {self.build_smoke_command()} antes de promover corridas reales."
            if smoke_ready
            else "Corrige los bloqueos de readiness y luego ejecuta el smoke del runtime Codex."
        )
        return {
            "status": status,
            "provider": "codex_local",
            "active_provider": self.runtime_settings.active_provider.value,
            "selected_as_active_provider": self.runtime_settings.active_provider.value == "codex_local",
            "available": bool(executable and configured_model),
            "executable": executable or self.runtime_settings.codex_local.command,
            "version": version,
            "implementation_backend": "codex_exec_wrapper",
            "implementation_detail": "Codex CLI staged workspace runtime",
            "auth_mode": auth_mode,
            "auth_detected": auth_detected,
            "smoke_ready": smoke_ready,
            "smoke_blocking_reasons": smoke_blocking_reasons,
            "codex_home_path": str(self.resolve_codex_home_path()),
            "runner_id": self.runtime_settings.codex_local.runner_id,
            "configured_models": {
                "default": configured_model,
                "primary": configured_model,
                "shadow": configured_model,
                "diagnostic": configured_model,
            },
            "configured_fallback_models": {
                "default": configured_fallbacks,
                "primary": configured_fallbacks,
                "shadow": configured_fallbacks,
                "diagnostic": configured_fallbacks,
            },
            "timeout_ms": self.runtime_settings.codex_local.timeout_ms,
            "max_concurrency": self.runtime_settings.codex_local.max_concurrency,
            "smoke_command": self.build_smoke_command(),
            "recommended_check": recommendation,
            "last_known_result": last_known_result,
            "last_error": None if not last_known_result or last_known_result.get("status") == "succeeded" else last_known_result,
        }

    def build_execution_args(self, *, workspace: CodexPromptWorkspace, model: str | None = None) -> list[str]:
        resolved_model = (model or self.runtime_settings.codex_local.model).strip() or "session_default"
        reasoning_effort = self._resolve_codex_reasoning_effort()
        command = [
            self.resolve_executable(),
            "exec",
            "-C",
            str(workspace.root_dir),
            "--sandbox",
            "workspace-write",
            "--color",
            "never",
            "--skip-git-repo-check",
            "--model",
            resolved_model,
            "--output-schema",
            str(workspace.schema_path),
            "--output-last-message",
            str(workspace.last_message_path),
        ]
        if reasoning_effort:
            command.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
        if self.runtime_settings.codex_local.profile:
            command.extend(["--profile", self.runtime_settings.codex_local.profile])
        command.append("-")
        return command

    def _resolve_codex_reasoning_effort(self) -> str:
        configured = self.runtime_settings.openai.reasoning_effort.strip().lower()
        if configured in {"low", "medium", "high", "xhigh"}:
            return configured
        if configured == "max":
            return "high"
        return "low"

    def run_process(
        self,
        *,
        command: list[str],
        workdir: Path,
        stdin_text: str,
        timeout_seconds: float,
    ) -> CodexProcessResult:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            child = subprocess.Popen(
                command,
                cwd=workdir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                creationflags=creationflags,
            )
        except FileNotFoundError as exc:
            raise CodexExecutionError(
                "No se encontro el ejecutable Codex configurado.",
                code=CodexRuntimeErrorCode.binary_not_found,
                detail={"command": command[0], "workdir": str(workdir)},
            ) from exc
        try:
            stdout, stderr = child.communicate(input=stdin_text, timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            self._terminate_process_tree(child)
            stdout, stderr = child.communicate()
            result = CodexProcessResult(
                command=command,
                workdir=workdir,
                stdout=stdout or "",
                stderr=stderr or "",
                returncode=child.returncode if child.returncode is not None else -1,
            )
            raise CodexExecutionError(
                "Codex excedio el timeout configurado para esta tarea.",
                code=CodexRuntimeErrorCode.timeout,
                recoverable=True,
                result=result,
                detail={"timeout_seconds": timeout_seconds, "workdir": str(workdir)},
            ) from exc
        return CodexProcessResult(
            command=command,
            workdir=workdir,
            stdout=stdout or "",
            stderr=stderr or "",
            returncode=child.returncode if child.returncode is not None else -1,
        )

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _terminate_process_tree(self, child: subprocess.Popen[str]) -> None:
        if child.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(child.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            child.kill()

    def _duration_ms(self, started_at: str, finished_at: str) -> int:
        started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        finished = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
        return max(0, int((finished - started).total_seconds() * 1000))

    def _reset_attempt_output_files(self, workspace: CodexPromptWorkspace) -> None:
        workspace.last_message_path.write_text("", encoding="utf-8")

    def _append_attempt_logs(
        self,
        *,
        workspace: CodexPromptWorkspace,
        attempt_number: int,
        model: str,
        result: CodexProcessResult,
    ) -> None:
        stdout_header = (
            f"=== attempt {attempt_number} | model {model} | returncode {result.returncode} | stdout ===\n"
        )
        stderr_header = (
            f"=== attempt {attempt_number} | model {model} | returncode {result.returncode} | stderr ===\n"
        )
        with workspace.stdout_path.open("a", encoding="utf-8") as handle:
            handle.write(stdout_header)
            handle.write(result.stdout)
            handle.write("\n")
        with workspace.stderr_path.open("a", encoding="utf-8") as handle:
            handle.write(stderr_header)
            handle.write(result.stderr)
            handle.write("\n")

    def _resolve_payload_text(
        self,
        *,
        workspace: CodexPromptWorkspace,
        result: CodexProcessResult,
    ) -> tuple[str, str | None]:
        raw_last_message = workspace.last_message_path.read_text(encoding="utf-8").strip()
        stdout_payload = result.stdout.strip()
        if stdout_payload:
            return stdout_payload, "stdout"
        if raw_last_message:
            return raw_last_message, "output/last_message.md"
        return "", None

    def _build_attempt_record(
        self,
        *,
        attempt_number: int,
        model: str,
        status: str,
        started_at: str,
        finished_at: str,
        returncode: int | None = None,
        error_code: CodexRuntimeErrorCode | None = None,
        error_message: str | None = None,
        payload_source: str | None = None,
        retryable: bool = False,
        output_size_bytes: int = 0,
    ) -> CodexAttemptRecord:
        return CodexAttemptRecord(
            attempt_number=attempt_number,
            model=model,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=self._duration_ms(started_at, finished_at),
            returncode=returncode,
            error_code=error_code.value if error_code else None,
            error_message=error_message,
            payload_source=payload_source,
            retryable=retryable,
            output_size_bytes=output_size_bytes,
        )

    def _build_audit_record(
        self,
        *,
        workspace: CodexPromptWorkspace,
        status: str,
        requested_model: str,
        selected_model: str | None,
        attempted_models: list[str],
        fallback_used: bool,
        started_at: str,
        finished_at: str | None,
        returncode: int | None,
        payload_source: str | None,
        command: list[str],
        error: str | None,
        error_code: CodexRuntimeErrorCode | None,
        recoverable: bool,
        attempts: list[CodexAttemptRecord],
        queue_wait_ms: int,
        metadata: dict[str, object] | None = None,
    ) -> CodexExecutionAuditRecord:
        metrics = None
        if finished_at is not None:
            metrics = CodexExecutionMetrics(
                duration_ms=self._duration_ms(started_at, finished_at),
                queue_wait_ms=queue_wait_ms,
                output_size_bytes=len(workspace.structured_output_path.read_text(encoding="utf-8").encode("utf-8"))
                if workspace.structured_output_path.read_text(encoding="utf-8").strip()
                else 0,
                stdout_bytes=len(workspace.stdout_path.read_bytes()),
                stderr_bytes=len(workspace.stderr_path.read_bytes()),
                exit_code=returncode,
            )
        return CodexExecutionAuditRecord(
            run_id=workspace.run_id,
            task_kind=workspace.task_kind,
            runner_id=self.runtime_settings.codex_local.runner_id,
            status=status,
            requested_model=requested_model,
            selected_model=selected_model,
            attempted_models=attempted_models,
            fallback_used=fallback_used,
            profile=self.runtime_settings.codex_local.profile,
            workspace_root=str(workspace.root_dir),
            workdir=str(workspace.root_dir),
            prompt_path=str(workspace.prompt_path),
            schema_path=str(workspace.schema_path),
            last_message_path=str(workspace.last_message_path),
            structured_output_path=str(workspace.structured_output_path),
            stdout_path=str(workspace.stdout_path),
            stderr_path=str(workspace.stderr_path),
            started_at=started_at,
            finished_at=finished_at,
            returncode=returncode,
            payload_source=payload_source,
            command=command,
            error=error,
            error_code=error_code.value if error_code else None,
            recoverable=recoverable,
            attempts=attempts,
            metrics=metrics,
            metadata=dict(metadata or {}),
        )

    def _assemble_context(
        self,
        *,
        task_kind: str,
        context_request: CodexContextRequest | None,
    ) -> CodexContextAssembly | None:
        if context_request is None:
            return None
        return self.context_assembler.assemble(task_kind=task_kind, request=context_request)

    def _compose_prompt(
        self,
        *,
        prompt: str,
        context_assembly: CodexContextAssembly | None,
    ) -> str:
        if context_assembly is None:
            return prompt
        return f"{context_assembly.prompt_preamble}\n\nTask:\n{prompt}".strip()

    def execute_structured_prompt(
        self,
        *,
        task_kind: str,
        prompt: str,
        output_model: type[BaseModel],
        timeout_ms: int | None = None,
        model_override: str | None = None,
        context_request: CodexContextRequest | None = None,
    ) -> BaseModel:
        context_assembly = self._assemble_context(task_kind=task_kind, context_request=context_request)
        resolved_prompt = self._compose_prompt(prompt=prompt, context_assembly=context_assembly)
        base_metadata = {
            "attempt_count": 0,
            "context": context_assembly.metadata_payload() if context_assembly is not None else {},
        }
        with self.workspace_builder.build(
            output_model=output_model,
            task_kind=task_kind,
            knowledge_access_backend=self.runtime_settings.knowledge_access_backend.value,
            context_assembly=context_assembly,
        ) as workspace:
            workspace.prompt_path.write_text(resolved_prompt, encoding="utf-8")
            started_at = self._utc_now()
            timeout_budget_ms = self.resolve_timeout_ms(task_kind=task_kind, timeout_ms=timeout_ms)
            attempts: list[CodexAttemptRecord] = []
            attempted_models: list[str] = []
            selected_model: str | None = None
            fallback_used = False
            payload_source: str | None = None
            final_command: list[str] = [self.resolve_executable(), "exec"]
            final_returncode: int | None = None
            final_error: CodexExecutionError | None = None
            queue_wait_ms = 0
            try:
                with self.queue_service.with_execution_slot(
                    runner_id=self.runtime_settings.codex_local.runner_id,
                    max_concurrency=self.resolve_max_concurrency(),
                    timeout_ms=timeout_budget_ms,
                ) as queue_slot:
                    queue_wait_ms = queue_slot.wait_ms
                    attempt_sequence = self.fallback_policy.build_attempt_sequence(primary_model=model_override)
                    for attempt_number, model in enumerate(attempt_sequence, start=1):
                        attempted_models.append(model)
                        self._reset_attempt_output_files(workspace)
                        command = self.build_execution_args(workspace=workspace, model=model)
                        final_command = command
                        running_audit = self._build_audit_record(
                            workspace=workspace,
                            status="running",
                            requested_model=attempt_sequence[0],
                            selected_model=model,
                            attempted_models=attempted_models,
                            fallback_used=attempt_number > 1,
                            started_at=started_at,
                            finished_at=None,
                            returncode=None,
                            payload_source=None,
                            command=command,
                            error=None,
                            error_code=None,
                            recoverable=False,
                            attempts=attempts,
                            queue_wait_ms=queue_wait_ms,
                            metadata={
                                **base_metadata,
                                "attempt_count": len(attempt_sequence),
                                "timeout_ms": timeout_budget_ms,
                                "task_timeout_env_key": self._build_task_timeout_env_key(task_kind),
                            },
                        )
                        self.audit_store.persist(workspace=workspace, record=running_audit)
                        attempt_started_at = self._utc_now()
                        try:
                            result = self.run_process(
                                command=command,
                                workdir=workspace.root_dir,
                                stdin_text=resolved_prompt,
                                timeout_seconds=timeout_budget_ms / 1000,
                            )
                        except CodexExecutionError as exc:
                            attempt_finished_at = self._utc_now()
                            attempts.append(
                                self._build_attempt_record(
                                    attempt_number=attempt_number,
                                    model=model,
                                    status="failed",
                                    started_at=attempt_started_at,
                                    finished_at=attempt_finished_at,
                                    returncode=exc.result.returncode if exc.result else None,
                                    error_code=exc.code,
                                    error_message=str(exc),
                                    retryable=exc.recoverable,
                                )
                            )
                            final_error = CodexExecutionError(
                                str(exc),
                                code=exc.code,
                                recoverable=exc.recoverable,
                                result=exc.result,
                                attempted_models=attempted_models,
                                selected_model=model,
                                detail=exc.detail,
                            )
                            final_returncode = exc.result.returncode if exc.result else None
                            selected_model = model
                            payload_source = None
                            break
                        self._append_attempt_logs(
                            workspace=workspace,
                            attempt_number=attempt_number,
                            model=model,
                            result=result,
                        )
                        payload_text, payload_source = self._resolve_payload_text(workspace=workspace, result=result)
                        if not workspace.last_message_path.read_text(encoding="utf-8").strip() and payload_text:
                            workspace.last_message_path.write_text(payload_text, encoding="utf-8")
                        attempt_finished_at = self._utc_now()
                        output_size_bytes = len(payload_text.encode("utf-8")) if payload_text else 0
                        if result.returncode != 0:
                            decision = self.fallback_policy.classify_failure(stdout=result.stdout, stderr=result.stderr)
                            attempts.append(
                                self._build_attempt_record(
                                    attempt_number=attempt_number,
                                    model=model,
                                    status="failed",
                                    started_at=attempt_started_at,
                                    finished_at=attempt_finished_at,
                                    returncode=result.returncode,
                                    error_code=decision.error_code,
                                    error_message=decision.reason,
                                    payload_source=payload_source,
                                    retryable=decision.retryable,
                                    output_size_bytes=output_size_bytes,
                                )
                            )
                            if decision.retryable and attempt_number < len(attempt_sequence):
                                fallback_used = True
                                continue
                            error_code = (
                                CodexRuntimeErrorCode.fallback_exhausted if decision.retryable else decision.error_code
                            )
                            message = (
                                "Se agotaron los modelos fallback configurados para Codex."
                                if error_code == CodexRuntimeErrorCode.fallback_exhausted
                                else decision.reason
                            )
                            final_error = CodexExecutionError(
                                message,
                                code=error_code,
                                recoverable=decision.retryable,
                                result=result,
                                attempted_models=attempted_models,
                                selected_model=model,
                                detail={"payload_source": payload_source},
                            )
                            final_returncode = result.returncode
                            selected_model = model
                            break
                        if not payload_text:
                            attempts.append(
                                self._build_attempt_record(
                                    attempt_number=attempt_number,
                                    model=model,
                                    status="failed",
                                    started_at=attempt_started_at,
                                    finished_at=attempt_finished_at,
                                    returncode=result.returncode,
                                    error_code=CodexRuntimeErrorCode.output_missing,
                                    error_message="Codex no produjo salida estructurada ni last_message.",
                                )
                            )
                            final_error = CodexExecutionError(
                                "Codex no produjo salida estructurada ni last_message.",
                                code=CodexRuntimeErrorCode.output_missing,
                                result=result,
                                attempted_models=attempted_models,
                                selected_model=model,
                            )
                            final_returncode = result.returncode
                            selected_model = model
                            break
                        try:
                            payload = json.loads(payload_text)
                        except json.JSONDecodeError as exc:  # pragma: no cover - covered by tests and smoke
                            attempts.append(
                                self._build_attempt_record(
                                    attempt_number=attempt_number,
                                    model=model,
                                    status="failed",
                                    started_at=attempt_started_at,
                                    finished_at=attempt_finished_at,
                                    returncode=result.returncode,
                                    error_code=CodexRuntimeErrorCode.parse_error,
                                    error_message="Codex no devolvio un JSON valido.",
                                    payload_source=payload_source,
                                    output_size_bytes=output_size_bytes,
                                )
                            )
                            final_error = CodexExecutionError(
                                "Codex no devolvio un JSON valido.",
                                code=CodexRuntimeErrorCode.parse_error,
                                result=result,
                                attempted_models=attempted_models,
                                selected_model=model,
                                detail={"payload_source": payload_source},
                            )
                            final_returncode = result.returncode
                            selected_model = model
                            break
                        try:
                            validated = output_model.model_validate(payload)
                        except ValidationError as exc:
                            attempts.append(
                                self._build_attempt_record(
                                    attempt_number=attempt_number,
                                    model=model,
                                    status="failed",
                                    started_at=attempt_started_at,
                                    finished_at=attempt_finished_at,
                                    returncode=result.returncode,
                                    error_code=CodexRuntimeErrorCode.parse_error,
                                    error_message="Codex devolvio un JSON que no cumple el schema esperado.",
                                    payload_source=payload_source,
                                    output_size_bytes=output_size_bytes,
                                )
                            )
                            final_error = CodexExecutionError(
                                "Codex devolvio un JSON que no cumple el schema esperado.",
                                code=CodexRuntimeErrorCode.parse_error,
                                result=result,
                                attempted_models=attempted_models,
                                selected_model=model,
                                detail={"validation_errors": exc.errors(), "payload_source": payload_source},
                            )
                            final_returncode = result.returncode
                            selected_model = model
                            break
                        workspace.structured_output_path.write_text(
                            json.dumps(payload, ensure_ascii=True, indent=2),
                            encoding="utf-8",
                        )
                        attempts.append(
                            self._build_attempt_record(
                                attempt_number=attempt_number,
                                model=model,
                                status="succeeded",
                                started_at=attempt_started_at,
                                finished_at=attempt_finished_at,
                                returncode=result.returncode,
                                payload_source=payload_source,
                                output_size_bytes=output_size_bytes,
                            )
                        )
                        selected_model = model
                        fallback_used = attempt_number > 1
                        final_returncode = result.returncode
                        finished_at = self._utc_now()
                        success_audit = self._build_audit_record(
                            workspace=workspace,
                            status="succeeded",
                            requested_model=attempt_sequence[0],
                            selected_model=selected_model,
                            attempted_models=attempted_models,
                            fallback_used=fallback_used,
                            started_at=started_at,
                            finished_at=finished_at,
                            returncode=final_returncode,
                            payload_source=payload_source,
                            command=command,
                            error=None,
                            error_code=None,
                            recoverable=False,
                            attempts=attempts,
                            queue_wait_ms=queue_wait_ms,
                            metadata={
                                **base_metadata,
                                "attempt_count": len(attempt_sequence),
                                "timeout_ms": timeout_budget_ms,
                                "task_timeout_env_key": self._build_task_timeout_env_key(task_kind),
                            },
                        )
                        self.audit_store.persist(workspace=workspace, record=success_audit, append_to_log=True)
                        return validated
            except CodexExecutionError as exc:
                if exc.code != CodexRuntimeErrorCode.queue_rejected:
                    raise
                queue_wait_ms = int(exc.detail.get("queue_wait_ms", 0)) if exc.detail else 0
                finished_at = self._utc_now()
                queue_audit = self._build_audit_record(
                    workspace=workspace,
                    status="failed",
                    requested_model=(model_override or self.runtime_settings.codex_local.model),
                    selected_model=None,
                    attempted_models=[],
                    fallback_used=False,
                    started_at=started_at,
                    finished_at=finished_at,
                    returncode=None,
                    payload_source=None,
                    command=final_command,
                    error=str(exc),
                    error_code=exc.code,
                    recoverable=exc.recoverable,
                    attempts=[],
                    queue_wait_ms=queue_wait_ms,
                    metadata={
                        **base_metadata,
                        "attempt_count": 0,
                        "timeout_ms": timeout_budget_ms,
                        "task_timeout_env_key": self._build_task_timeout_env_key(task_kind),
                    },
                )
                self.audit_store.persist(workspace=workspace, record=queue_audit, append_to_log=True)
                raise
            finished_at = self._utc_now()
            failed_audit = self._build_audit_record(
                workspace=workspace,
                status="failed",
                requested_model=(model_override or self.runtime_settings.codex_local.model),
                selected_model=selected_model,
                attempted_models=attempted_models,
                fallback_used=fallback_used,
                started_at=started_at,
                finished_at=finished_at,
                returncode=final_returncode,
                payload_source=payload_source,
                command=final_command,
                error=str(final_error) if final_error else "La ejecucion Codex fallo sin detalle clasificado.",
                error_code=final_error.code if final_error else CodexRuntimeErrorCode.execution_failed,
                recoverable=final_error.recoverable if final_error else False,
                attempts=attempts,
                queue_wait_ms=queue_wait_ms,
                metadata={
                    **base_metadata,
                    "attempt_count": len(attempted_models),
                    "timeout_ms": timeout_budget_ms,
                    "task_timeout_env_key": self._build_task_timeout_env_key(task_kind),
                },
            )
            self.audit_store.persist(workspace=workspace, record=failed_audit, append_to_log=True)
            if final_error is not None:
                raise final_error
            raise CodexExecutionError(
                "La ejecucion Codex termino sin resultado y sin error clasificado.",
                code=CodexRuntimeErrorCode.execution_failed,
                attempted_models=attempted_models,
                selected_model=selected_model,
            )
