from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from uuid import UUID

from app.services.agentic_runtime.contracts import BuilderActionRequest, BuilderAgentState


class BuilderLoopGuardViolation(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class BuilderLoopGuardConfig:
    max_iterations: int = 8
    max_total_ms: int = 120_000
    max_llm_calls: int = 5
    max_token_usage: int = 30_000
    repeated_action_limit: int = 2


@dataclass
class BuilderLoopGuardState:
    started_monotonic: float = field(default_factory=monotonic)
    action_history: list[str] = field(default_factory=list)


def build_idempotency_key(run_id: UUID, action_key: str, iteration: int) -> str:
    return f"react:{run_id}:{action_key}:{iteration}"


class BuilderLoopGuards:
    def __init__(self, config: BuilderLoopGuardConfig | None = None):
        self.config = config or BuilderLoopGuardConfig()

    def before_action(
        self,
        *,
        state: BuilderAgentState,
        action: BuilderActionRequest,
        runtime: BuilderLoopGuardState,
    ) -> None:
        if state.iteration >= self.config.max_iterations:
            raise BuilderLoopGuardViolation("max_iterations", "El loop alcanzo el maximo de iteraciones permitido.")
        if (monotonic() - runtime.started_monotonic) * 1000 >= self.config.max_total_ms:
            raise BuilderLoopGuardViolation("timeout", "El loop alcanzo el timeout total permitido.")
        if state.llm_calls >= self.config.max_llm_calls and action.key in {"invoke_capability", "invoke_critique"}:
            raise BuilderLoopGuardViolation("max_llm_calls", "El loop alcanzo el maximo de llamadas LLM permitido.")
        if state.token_usage >= self.config.max_token_usage:
            raise BuilderLoopGuardViolation("token_budget", "El loop alcanzo el presupuesto de tokens permitido.")
        recent_count = sum(1 for item in runtime.action_history[-self.config.repeated_action_limit :] if item == action.key)
        if recent_count >= self.config.repeated_action_limit:
            raise BuilderLoopGuardViolation("repeated_action", f"La accion '{action.key}' se repitio sin progreso suficiente.")

    def record_action(self, runtime: BuilderLoopGuardState, action: BuilderActionRequest) -> None:
        runtime.action_history.append(action.key)
