from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from threading import BoundedSemaphore, Lock
from time import monotonic
from typing import Iterator

from app.services.llm_runtime.codex_cli.runtime_types import (
    CodexExecutionError,
    CodexRuntimeErrorCode,
)


@dataclass(frozen=True)
class CodexExecutionSlot:
    runner_id: str
    max_concurrency: int
    wait_ms: int


class CodexExecutionQueueService:
    _registry_lock = Lock()
    _semaphore_registry: dict[tuple[str, int], BoundedSemaphore] = {}

    def _resolve_semaphore(self, *, runner_id: str, max_concurrency: int) -> BoundedSemaphore:
        key = (runner_id, max(1, max_concurrency))
        with self._registry_lock:
            semaphore = self._semaphore_registry.get(key)
            if semaphore is None:
                semaphore = BoundedSemaphore(key[1])
                self._semaphore_registry[key] = semaphore
            return semaphore

    @contextmanager
    def with_execution_slot(
        self,
        *,
        runner_id: str,
        max_concurrency: int,
        timeout_ms: int,
    ) -> Iterator[CodexExecutionSlot]:
        semaphore = self._resolve_semaphore(runner_id=runner_id, max_concurrency=max_concurrency)
        started = monotonic()
        acquired = semaphore.acquire(timeout=max(1.0, timeout_ms / 1000))
        wait_ms = int((monotonic() - started) * 1000)
        if not acquired:
            raise CodexExecutionError(
                "La cola del runtime Codex supero el tiempo de espera antes de obtener un slot.",
                code=CodexRuntimeErrorCode.queue_rejected,
                recoverable=True,
                detail={
                    "runner_id": runner_id,
                    "max_concurrency": max(1, max_concurrency),
                    "queue_wait_ms": wait_ms,
                },
            )
        try:
            yield CodexExecutionSlot(
                runner_id=runner_id,
                max_concurrency=max(1, max_concurrency),
                wait_ms=wait_ms,
            )
        finally:
            semaphore.release()
