from __future__ import annotations

from dataclasses import dataclass

from app.services.llm_runtime.antigravity_cli.runtime_types import AgyExecutionError, AgyRuntimeErrorCode

# Patrones de error propios de Antigravity CLI / agy
_QUOTA_PATTERNS = (
    "quota exceeded",
    "resource exhausted",
    "too many requests",
    "rate limit",
    "ratelimit",
    "limit exceeded",
)

_AUTH_PATTERNS = (
    "unauthenticated",
    "unauthorized",
    "forbidden",
    "authentication required",
    "invalid api key",
    "api key",
    "credentials",
    "login required",
    "permission denied",
)

_CAPACITY_PATTERNS = (
    "model is at capacity",
    "model overloaded",
    "overloaded",
    "at capacity",
    "try again later",
    "service unavailable",
)


@dataclass(frozen=True)
class AgyFailureDecision:
    error_code: AgyRuntimeErrorCode
    retryable: bool
    reason: str


class AgyFallbackPolicy:
    """
    Politica de seleccion de modelos y clasificacion de errores para el proveedor
    Antigravity CLI.

    Sigue el mismo contrato que CodexFallbackPolicy:
      - build_attempt_sequence(): construye la lista [modelo_primario, ...fallbacks]
      - classify_failure(): dado stdout/stderr, retorna si el error es recuperable
    """

    def __init__(self, *, model: str, fallback_models: list[str]) -> None:
        self._model = model
        self._fallback_models = list(fallback_models)

    def build_attempt_sequence(self, *, primary_model: str | None = None) -> list[str]:
        """
        Construye la secuencia ordenada de modelos a intentar.
        El modelo solicitado tiene prioridad; luego se agregan los fallbacks
        eliminando duplicados y preservando el orden.
        """
        candidates = [primary_model or self._model]
        candidates.extend(self._fallback_models)
        sequence: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = str(candidate or "").strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                sequence.append(normalized)
        return sequence or ["gemini-3.6-flash"]

    def classify_failure(self, *, stdout: str, stderr: str) -> AgyFailureDecision:
        """Clasifica la causa de un fallo y determina si es recuperable."""
        text = "\n".join(part for part in (stderr, stdout) if part).lower()

        if any(pattern in text for pattern in _AUTH_PATTERNS):
            return AgyFailureDecision(
                error_code=AgyRuntimeErrorCode.auth_error,
                retryable=False,
                reason="El proveedor Antigravity CLI no tiene autenticacion valida en el entorno actual.",
            )

        if any(pattern in text for pattern in _QUOTA_PATTERNS):
            return AgyFailureDecision(
                error_code=AgyRuntimeErrorCode.quota_exceeded,
                retryable=True,
                reason="El modelo reporto cuota agotada o rate limit; se puede reintentar con otro modelo.",
            )

        if any(pattern in text for pattern in _CAPACITY_PATTERNS):
            return AgyFailureDecision(
                error_code=AgyRuntimeErrorCode.quota_exceeded,
                retryable=True,
                reason="El modelo reporto sobrecarga de capacidad; admite fallback a otro modelo.",
            )

        return AgyFailureDecision(
            error_code=AgyRuntimeErrorCode.execution_failed,
            retryable=False,
            reason="La ejecucion de agy fallo sin una condicion recuperable conocida.",
        )
