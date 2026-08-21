from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def _extract_json_payload(raw: str) -> Any:
    text = raw.strip()
    if not text:
        raise ValueError("Salida vacia recibida de Antigravity CLI")
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return json.loads(match.group(1).strip())
    start_brace = text.find("{")
    start_bracket = text.find("[")
    if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
        end_brace = text.rfind("}")
        if end_brace != -1:
            return json.loads(text[start_brace : end_brace + 1])
    elif start_bracket != -1:
        end_bracket = text.rfind("]")
        if end_bracket != -1:
            return json.loads(text[start_bracket : end_bracket + 1])
    return json.loads(text)


from pydantic import BaseModel, ValidationError

from app.core.config import get_settings
from app.models import AntigravityProviderConfig, LLMRuntimeSettings
from app.services.llm_runtime.antigravity_cli.audit_store import AgyAuditStore
from app.services.llm_runtime.antigravity_cli.fallback_policy import AgyFallbackPolicy
from app.services.llm_runtime.antigravity_cli.queue_service import AgyExecutionQueueService
from app.services.llm_runtime.antigravity_cli.runtime_types import (
    AgyAttemptRecord,
    AgyExecutionAuditRecord,
    AgyExecutionError,
    AgyExecutionMetrics,
    AgyProcessResult,
    AgyRuntimeErrorCode,
)
from app.services.llm_runtime.antigravity_cli.workspace_builder import AgyPromptWorkspaceBuilder, AgyRunWorkspace


# ---------------------------------------------------------------------------
# Deteccion del ejecutable agy
# ---------------------------------------------------------------------------

def resolve_agy_executable(configured: str | None = None) -> str | None:
    """
    Resolucion del ejecutable agy en 4 pasos (en orden de prioridad):

    1. Variable de entorno ANTIGRAVITY_EXECUTABLE (ruta explicita)
    2. Valor configurado en runtime_settings.antigravity.executable
    3. PATH del sistema: busca 'agy' primero, luego 'antigravity'
    4. Ruta conocida de instalacion en Windows: %LOCALAPPDATA%\\agy\\bin\\agy.exe
    """
    # 1. Variable de entorno tiene maxima prioridad
    env_path = os.getenv("ANTIGRAVITY_EXECUTABLE", "").strip()
    if env_path:
        candidate = Path(os.path.expandvars(os.path.expanduser(env_path)))
        if candidate.exists():
            return str(candidate)

    # 2. Configuracion del tenant / runtime_settings
    if configured and configured.strip() and configured.strip() != "agy":
        candidate = Path(os.path.expandvars(os.path.expanduser(configured.strip())))
        if candidate.exists():
            return str(candidate)
        found = shutil.which(configured.strip())
        if found:
            return found

    # 3. PATH del sistema
    for name in ("agy", "antigravity"):
        found = shutil.which(name)
        if found:
            return found

    # 4. Ruta conocida de instalacion en Windows
    if os.name == "nt":
        local_app_data = os.getenv("LOCALAPPDATA", "").strip()
        if local_app_data:
            win_path = Path(local_app_data) / "agy" / "bin" / "agy.exe"
            if win_path.exists():
                return str(win_path)

    return None


# ---------------------------------------------------------------------------
# Servicio principal de ejecucion
# ---------------------------------------------------------------------------

class AgyExecutionService:
    """
    Servicio de ejecucion del proveedor Antigravity CLI.

    Implementa el mismo algoritmo que CodexExecutionService:
    - Resolucion del ejecutable
    - Control de concurrencia con AgyExecutionQueueService
    - Loop de intentos con fallback de modelos via AgyFallbackPolicy
    - Auditoria con AgyAuditStore
    - Manejo de timeouts y terminacion del proceso

    El metodo principal es execute_structured_prompt(), que devuelve
    una instancia validada del output_model recibido.
    """

    def __init__(
        self,
        runtime_settings: LLMRuntimeSettings,
        *,
        workspace_builder: AgyPromptWorkspaceBuilder | None = None,
        fallback_policy: AgyFallbackPolicy | None = None,
        queue_service: AgyExecutionQueueService | None = None,
        audit_store: AgyAuditStore | None = None,
    ) -> None:
        self.runtime_settings = runtime_settings
        self._agy_cfg: AntigravityProviderConfig = runtime_settings.antigravity

        settings = get_settings()
        runtime_root = settings.llm_config_path.parent / "agy-workspaces"
        self.workspace_builder = workspace_builder or AgyPromptWorkspaceBuilder(runtime_root=runtime_root)

        self.fallback_policy = fallback_policy or AgyFallbackPolicy(
            model=self._agy_cfg.model,
            fallback_models=list(self._agy_cfg.fallback_models),
        )
        self.queue_service = queue_service or AgyExecutionQueueService()
        self.audit_store = audit_store or AgyAuditStore()

    # ------------------------------------------------------------------
    # Resolucion de configuracion
    # ------------------------------------------------------------------

    def resolve_executable(self) -> str:
        found = resolve_agy_executable(self._agy_cfg.executable)
        return found or self._agy_cfg.executable

    def resolve_max_concurrency(self) -> int:
        env_val = os.getenv("ANTIGRAVITY_EXEC_MAX_CONCURRENCY", "").strip()
        if env_val:
            try:
                return max(1, int(env_val))
            except ValueError:
                pass
        return max(1, self._agy_cfg.max_concurrency)

    def resolve_timeout_ms(self, *, timeout_ms: int | None = None) -> int:
        if timeout_ms is not None:
            return max(1_000, int(timeout_ms))
        env_val = os.getenv("ANTIGRAVITY_EXEC_TIMEOUT_MS", "").strip()
        if env_val:
            try:
                return max(1_000, int(env_val))
            except ValueError:
                pass
        return max(1_000, self._agy_cfg.timeout_ms)

    def resolve_agy_home(self) -> Path:
        env_home = os.getenv("ANTIGRAVITY_HOME", "").strip()
        return Path(env_home) if env_home else (Path.home() / ".antigravity")

    def resolve_auth_mode(self) -> tuple[str, bool]:
        """
        Detecta el modo de autenticacion activo en orden de prioridad:
        1. auth_mode configurado explicitamente en runtime_settings
        2. Variable de entorno ANTIGRAVITY_API_KEY
        3. Archivos de sesion/credenciales ~/.antigravity/credentials.json o ~/.gemini
        4. Deteccion del ejecutable agy autenticado en la plataforma
        5. Fallback: 'unknown'
        """
        configured_mode = (self._agy_cfg.auth_mode or "auto").strip()

        has_api_key = bool(os.getenv("ANTIGRAVITY_API_KEY", "").strip())
        has_credentials = (
            (self.resolve_agy_home() / "credentials.json").exists()
            or (Path.home() / ".gemini" / "oauth_creds.json").exists()
            or (Path.home() / ".gemini" / "google_accounts.json").exists()
            or (Path.home() / ".antigravity" / "argv.json").exists()
        )
        has_executable = resolve_agy_executable(self._agy_cfg.executable) is not None

        if configured_mode not in {"", "auto", "unknown"}:
            is_available = (
                has_api_key
                or has_credentials
                or (configured_mode in {"platform", "session", "auto"} and has_executable)
            )
            return configured_mode, is_available

        # Deteccion automatica
        if has_api_key:
            return "api_key", True

        if has_credentials or has_executable:
            return "session", True

        return "unknown", False

    def resolve_version(self) -> str | None:
        executable = resolve_agy_executable(self._agy_cfg.executable)
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

    def get_runtime_status(self) -> dict[str, Any]:
        """Equivalente a CodexExecutionService.get_runtime_status()."""
        executable = resolve_agy_executable(self._agy_cfg.executable)
        auth_mode, auth_detected = self.resolve_auth_mode()
        version = self.resolve_version()
        configured_model = self._agy_cfg.model.strip() or None
        configured_fallbacks = list(self._agy_cfg.fallback_models)

        smoke_blocking_reasons: list[str] = []
        if not executable:
            smoke_blocking_reasons.append("No se pudo resolver el ejecutable agy en el entorno actual.")
        if not configured_model:
            smoke_blocking_reasons.append("No hay modelo default configurado para antigravity_cli.")
        if not auth_detected:
            smoke_blocking_reasons.append("No se detecto autenticacion utilizable para Antigravity CLI.")

        smoke_ready = not smoke_blocking_reasons
        status = "healthy" if smoke_ready else "degraded"

        return {
            "status": status,
            "provider": "antigravity_cli",
            "available": bool(executable and configured_model),
            "executable": executable or self._agy_cfg.executable,
            "version": version,
            "implementation_backend": "agy_cli_wrapper",
            "implementation_detail": "Antigravity CLI (agy) staged workspace runtime",
            "auth_mode": auth_mode,
            "auth_detected": auth_detected,
            "smoke_ready": smoke_ready,
            "smoke_blocking_reasons": smoke_blocking_reasons,
            "agy_home_path": str(self.resolve_agy_home()),
            "runner_id": self._agy_cfg.runner_id,
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
            "timeout_ms": self._agy_cfg.timeout_ms,
            "max_concurrency": self._agy_cfg.max_concurrency,
        }

    def read_last_known_result(self) -> dict[str, Any] | None:
        """Lee el ultimo resultado del log de auditoria JSONL."""
        settings = get_settings()
        audit_log_path = settings.llm_config_path.parent / "agy-workspaces" / "runtime-audit-agy.jsonl"
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
        }

    # ------------------------------------------------------------------
    # Construccion de argumentos CLI
    # ------------------------------------------------------------------

    def build_execution_args(
        self,
        *,
        workspace: AgyRunWorkspace,
        prompt: str = "",
        model: str | None = None,
        enable_web_search: bool = False,
    ) -> list[str]:
        """
        Construye la lista de argumentos para invocar agy en modo print estructurado.
        Si el prompt es extenso (> 3500 caracteres), se referencia el archivo prompt.md
        en el workspace para evitar límites de longitud en la línea de comandos de Windows (WinError 206).
        """
        executable = self.resolve_executable()
        effective_model = (model if model is not None else self._agy_cfg.model).strip() or None
        effort = (self._agy_cfg.effort or "high").strip() or "high"

        # Escribir siempre el prompt completo en el archivo prompt.md del workspace
        workspace.prompt_path.write_text(prompt, encoding="utf-8")

        if len(prompt) > 3500:
            cli_prompt = (
                f"Lee detalladamente el archivo {workspace.prompt_path.name} dentro del workspace "
                "y genera exclusivamente el JSON solicitado que cumpla con el schema especificado en dicho archivo."
            )
        else:
            cli_prompt = prompt

        args = [
            executable,
            "--dangerously-skip-permissions",
            "--add-dir",
            str(workspace.root_dir),
            "--print",
            cli_prompt,
        ]

        if effective_model:
            args.extend(["--model", effective_model, "--effort", effort])

        return args

    # ------------------------------------------------------------------
    # Ejecucion del proceso
    # ------------------------------------------------------------------

    def run_process(
        self,
        *,
        command: list[str],
        workdir: Path,
        stdin_text: str,
        timeout_seconds: float,
    ) -> AgyProcessResult:
        """
        Ejecuta el proceso agy con manejo de timeout y terminacion del arbol de procesos.
        El prompt se envia por stdin; la salida se captura de stdout/stderr.
        """
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
                errors="replace",
                creationflags=creationflags,
            )
        except FileNotFoundError as exc:
            raise AgyExecutionError(
                f"No se encontro el ejecutable agy o fallo al crearse el proceso: {exc}",
                code=AgyRuntimeErrorCode.binary_not_found,
                detail={"command": command[0], "workdir": str(workdir)},
            ) from exc

        try:
            stdout, stderr = child.communicate(input=stdin_text, timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            self._terminate_process_tree(child)
            stdout, stderr = child.communicate()
            result = AgyProcessResult(
                command=command,
                workdir=workdir,
                stdout=stdout or "",
                stderr=stderr or "",
                returncode=child.returncode if child.returncode is not None else -1,
            )
            raise AgyExecutionError(
                "Antigravity CLI excedio el timeout configurado para esta tarea.",
                code=AgyRuntimeErrorCode.timeout,
                recoverable=True,
                result=result,
                detail={"timeout_seconds": timeout_seconds, "workdir": str(workdir)},
            ) from exc

        return AgyProcessResult(
            command=command,
            workdir=workdir,
            stdout=stdout or "",
            stderr=stderr or "",
            returncode=child.returncode if child.returncode is not None else -1,
        )

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

    # ------------------------------------------------------------------
    # Helpers de tiempo y logs
    # ------------------------------------------------------------------

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _duration_ms(self, started_at: str, finished_at: str) -> int:
        started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        finished = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
        return max(0, int((finished - started).total_seconds() * 1000))

    def _append_attempt_logs(
        self,
        *,
        workspace: AgyRunWorkspace,
        attempt_number: int,
        model: str,
        result: AgyProcessResult,
    ) -> None:
        header_stdout = f"=== attempt {attempt_number} | model {model} | returncode {result.returncode} | stdout ===\n"
        header_stderr = f"=== attempt {attempt_number} | model {model} | returncode {result.returncode} | stderr ===\n"
        with workspace.stdout_path.open("a", encoding="utf-8") as fh:
            fh.write(header_stdout)
            fh.write(result.stdout)
            fh.write("\n")
        with workspace.stderr_path.open("a", encoding="utf-8") as fh:
            fh.write(header_stderr)
            fh.write(result.stderr)
            fh.write("\n")

    def _build_attempt_record(
        self,
        *,
        attempt_number: int,
        model: str,
        status: str,
        started_at: str,
        finished_at: str,
        returncode: int | None = None,
        error_code: AgyRuntimeErrorCode | None = None,
        error_message: str | None = None,
        payload_source: str | None = None,
        retryable: bool = False,
        output_size_bytes: int = 0,
    ) -> AgyAttemptRecord:
        return AgyAttemptRecord(
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
        workspace: AgyRunWorkspace,
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
        error_code: AgyRuntimeErrorCode | None,
        recoverable: bool,
        attempts: list[AgyAttemptRecord],
        queue_wait_ms: int,
        metadata: dict[str, object] | None = None,
    ) -> AgyExecutionAuditRecord:
        metrics = None
        if finished_at is not None:
            output_text = workspace.output_path.read_text(encoding="utf-8").strip() if workspace.output_path.exists() else ""
            metrics = AgyExecutionMetrics(
                duration_ms=self._duration_ms(started_at, finished_at),
                queue_wait_ms=queue_wait_ms,
                output_size_bytes=len(output_text.encode("utf-8")) if output_text else 0,
                stdout_bytes=workspace.stdout_path.stat().st_size if workspace.stdout_path.exists() else 0,
                stderr_bytes=workspace.stderr_path.stat().st_size if workspace.stderr_path.exists() else 0,
                exit_code=returncode,
            )
        return AgyExecutionAuditRecord(
            run_id=workspace.run_id,
            task_kind=workspace.task_kind,
            runner_id=self._agy_cfg.runner_id,
            status=status,
            requested_model=requested_model,
            selected_model=selected_model,
            attempted_models=attempted_models,
            fallback_used=fallback_used,
            effort=self._agy_cfg.effort,
            workspace_root=str(workspace.root_dir),
            workdir=str(workspace.root_dir),
            prompt_path=str(workspace.prompt_path),
            output_path=str(workspace.output_path),
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

    # ------------------------------------------------------------------
    # Ejecucion principal con output estructurado
    # ------------------------------------------------------------------

    def execute_structured_prompt(
        self,
        *,
        task_kind: str,
        prompt: str,
        output_model: type[BaseModel],
        timeout_ms: int | None = None,
        model_override: str | None = None,
        enable_web_search: bool = False,
    ) -> BaseModel:
        """
        Ejecuta un prompt contra el CLI de Antigravity y retorna el resultado
        validado contra output_model.

        Implementa el mismo loop de intentos que CodexExecutionService:
        1. Adquirir slot de concurrencia (AgyExecutionQueueService)
        2. Construir secuencia de modelos a intentar (AgyFallbackPolicy)
        3. Loop: spawn agy → leer output → validar JSON → si falla y es retriable → siguiente modelo
        4. Registrar auditoria en cada intento (AgyAuditStore)
        """
        with self.workspace_builder.build(task_kind=task_kind) as workspace:
            workspace.prompt_path.write_text(prompt, encoding="utf-8")

            started_at = self._utc_now()
            timeout_budget_ms = self.resolve_timeout_ms(timeout_ms=timeout_ms)
            attempts: list[AgyAttemptRecord] = []
            attempted_models: list[str] = []
            selected_model: str | None = None
            fallback_used = False
            payload_source: str | None = None
            final_command: list[str] = [self.resolve_executable(), "run"]
            final_returncode: int | None = None
            final_error: AgyExecutionError | None = None
            queue_wait_ms = 0

            try:
                with self.queue_service.with_execution_slot(
                    runner_id=self._agy_cfg.runner_id,
                    max_concurrency=self.resolve_max_concurrency(),
                    timeout_ms=timeout_budget_ms,
                ) as queue_slot:
                    queue_wait_ms = queue_slot.wait_ms
                    attempt_sequence = self.fallback_policy.build_attempt_sequence(primary_model=model_override)

                    for attempt_number, model in enumerate(attempt_sequence, start=1):
                        attempted_models.append(model)

                        # Limpiar archivo de salida entre intentos
                        workspace.output_path.write_text("", encoding="utf-8")

                        command = self.build_execution_args(
                            workspace=workspace,
                            prompt=prompt,
                            model=model,
                            enable_web_search=enable_web_search,
                        )
                        final_command = command

                        # Auditoria: estado 'running'
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
                            metadata={"timeout_ms": timeout_budget_ms, "attempt_count": len(attempt_sequence)},
                        )
                        self.audit_store.persist(run_dir=workspace.root_dir, record=running_audit)

                        attempt_started_at = self._utc_now()

                        try:
                            result = self.run_process(
                                command=command,
                                workdir=workspace.root_dir,
                                stdin_text=prompt,
                                timeout_seconds=timeout_budget_ms / 1000,
                            )
                        except AgyExecutionError as exc:
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
                            final_error = AgyExecutionError(
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
                            break

                        self._append_attempt_logs(
                            workspace=workspace,
                            attempt_number=attempt_number,
                            model=model,
                            result=result,
                        )

                        attempt_finished_at = self._utc_now()

                        # Leer salida: preferir el archivo --output de agy; si no, stdout
                        output_text = workspace.output_path.read_text(encoding="utf-8").strip()
                        if not output_text:
                            output_text = result.stdout.strip()
                            payload_source = "stdout" if output_text else None
                        else:
                            payload_source = "output_file"

                        output_size_bytes = len(output_text.encode("utf-8")) if output_text else 0

                        if result.returncode != 0:
                            decision = self.fallback_policy.classify_failure(
                                stdout=result.stdout, stderr=result.stderr
                            )
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
                                AgyRuntimeErrorCode.fallback_exhausted
                                if decision.retryable
                                else decision.error_code
                            )
                            message = (
                                "Se agotaron los modelos fallback configurados para Antigravity CLI."
                                if error_code == AgyRuntimeErrorCode.fallback_exhausted
                                else decision.reason
                            )
                            final_error = AgyExecutionError(
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

                        if not output_text:
                            attempts.append(
                                self._build_attempt_record(
                                    attempt_number=attempt_number,
                                    model=model,
                                    status="failed",
                                    started_at=attempt_started_at,
                                    finished_at=attempt_finished_at,
                                    returncode=result.returncode,
                                    error_code=AgyRuntimeErrorCode.output_missing,
                                    error_message="agy no produjo contenido en el archivo de salida ni en stdout.",
                                )
                            )
                            final_error = AgyExecutionError(
                                "agy no produjo contenido en el archivo de salida ni en stdout.",
                                code=AgyRuntimeErrorCode.output_missing,
                                result=result,
                                attempted_models=attempted_models,
                                selected_model=model,
                            )
                            final_returncode = result.returncode
                            selected_model = model
                            break

                        # Parsear y validar JSON con extractor tolerante
                        try:
                            payload = _extract_json_payload(output_text)
                        except (json.JSONDecodeError, ValueError, Exception) as exc:
                            attempts.append(
                                self._build_attempt_record(
                                    attempt_number=attempt_number,
                                    model=model,
                                    status="failed",
                                    started_at=attempt_started_at,
                                    finished_at=attempt_finished_at,
                                    returncode=result.returncode,
                                    error_code=AgyRuntimeErrorCode.parse_error,
                                    error_message="agy no devolvio un JSON valido.",
                                    payload_source=payload_source,
                                    output_size_bytes=output_size_bytes,
                                )
                            )
                            final_error = AgyExecutionError(
                                "agy no devolvio un JSON valido.",
                                code=AgyRuntimeErrorCode.parse_error,
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
                                    error_code=AgyRuntimeErrorCode.parse_error,
                                    error_message="agy devolvio un JSON que no cumple el schema esperado.",
                                    payload_source=payload_source,
                                    output_size_bytes=output_size_bytes,
                                )
                            )
                            final_error = AgyExecutionError(
                                "agy devolvio un JSON que no cumple el schema esperado.",
                                code=AgyRuntimeErrorCode.parse_error,
                                result=result,
                                attempted_models=attempted_models,
                                selected_model=model,
                                detail={"validation_errors": exc.errors(), "payload_source": payload_source},
                            )
                            final_returncode = result.returncode
                            selected_model = model
                            break

                        # Exito
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
                        )
                        self.audit_store.persist(run_dir=workspace.root_dir, record=success_audit, append_to_log=True)
                        return validated

            except AgyExecutionError:
                raise

            except Exception as exc:
                final_error = AgyExecutionError(
                    f"Error inesperado en el runtime Antigravity CLI: {exc}",
                    code=AgyRuntimeErrorCode.execution_failed,
                    recoverable=False,
                )

            # Si llegamos aqui, todos los intentos fallaron
            finished_at = self._utc_now()
            failure_audit = self._build_audit_record(
                workspace=workspace,
                status="failed",
                requested_model=(self.fallback_policy.build_attempt_sequence(primary_model=model_override) or [""])[0],
                selected_model=selected_model,
                attempted_models=attempted_models,
                fallback_used=fallback_used,
                started_at=started_at,
                finished_at=finished_at,
                returncode=final_returncode,
                payload_source=payload_source,
                command=final_command,
                error=str(final_error) if final_error else None,
                error_code=final_error.code if final_error else AgyRuntimeErrorCode.execution_failed,
                recoverable=final_error.recoverable if final_error else False,
                attempts=attempts,
                queue_wait_ms=queue_wait_ms,
            )
            self.audit_store.persist(run_dir=workspace.root_dir, record=failure_audit, append_to_log=True)

            if final_error is not None:
                raise final_error
            raise AgyExecutionError(
                "Todos los modelos fallback de Antigravity CLI fallaron.",
                code=AgyRuntimeErrorCode.fallback_exhausted,
            )
