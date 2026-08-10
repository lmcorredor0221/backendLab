from __future__ import annotations

from dataclasses import dataclass

from app.models import LLMRuntimeSettings
from app.services.llm_runtime.codex_cli.runtime_types import CodexRuntimeErrorCode

_CAPACITY_PATTERNS = (
    "selected model is at capacity",
    "model is at capacity",
    "model capacity",
    "try a different model",
    "at capacity",
)
_AUTH_PATTERNS = (
    "login required",
    "authentication",
    "unauthorized",
    "forbidden",
    "access token",
    "api key",
    "codex login",
)


@dataclass(frozen=True)
class CodexFailureDecision:
    error_code: CodexRuntimeErrorCode
    retryable: bool
    reason: str


class CodexFallbackPolicy:
    def __init__(self, runtime_settings: LLMRuntimeSettings) -> None:
        self.runtime_settings = runtime_settings

    def build_attempt_sequence(self, *, primary_model: str | None = None) -> list[str]:
        candidates = [primary_model or self.runtime_settings.codex_local.model]
        candidates.extend(self.runtime_settings.codex_local.fallback_models)
        sequence: list[str] = []
        for candidate in candidates:
            normalized = str(candidate or "").strip()
            if normalized and normalized not in sequence:
                sequence.append(normalized)
        return sequence or ["session_default"]

    def classify_failure(self, *, stdout: str, stderr: str) -> CodexFailureDecision:
        text = "\n".join(part for part in (stderr, stdout) if part).lower()
        if "invalid_json_schema" in text or "response_format" in text:
            return CodexFailureDecision(
                error_code=CodexRuntimeErrorCode.invalid_schema,
                retryable=False,
                reason="El schema enviado a Codex es invalido para response_format.",
            )
        if any(pattern in text for pattern in _AUTH_PATTERNS):
            return CodexFailureDecision(
                error_code=CodexRuntimeErrorCode.auth_error,
                retryable=False,
                reason="Codex no tiene autenticacion utilizable en el entorno actual.",
            )
        if any(pattern in text for pattern in _CAPACITY_PATTERNS):
            return CodexFailureDecision(
                error_code=CodexRuntimeErrorCode.model_capacity,
                retryable=True,
                reason="El modelo reporto capacidad agotada y admite fallback.",
            )
        return CodexFailureDecision(
            error_code=CodexRuntimeErrorCode.execution_failed,
            retryable=False,
            reason="La ejecucion fallo sin una condicion recuperable conocida.",
        )
