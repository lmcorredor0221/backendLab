from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from threading import BoundedSemaphore, Lock
from time import monotonic
from typing import Iterator

from app.services.llm_runtime.antigravity_cli.runtime_types import AgyExecutionError, AgyRuntimeErrorCode


@dataclass(frozen=True)
class AgyExecutionSlot:
    runner_id: str
    max_concurrency: int
    wait_ms: int


class AgyExecutionQueueService:
    """
    Control de concurrencia para el runtime Antigravity CLI.

    Implementa el mismo mecanismo de BoundedSemaphore que CodexExecutionQueueService:
    - Un registry global evita crear semaforos duplicados para el mismo runner_id.
    - El timeout de adquisicion se mapea desde el timeout_ms de la ejecucion.
    """

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
    ) -> Iterator[AgyExecutionSlot]:
        semaphore = self._resolve_semaphore(runner_id=runner_id, max_concurrency=max_concurrency)
        started = monotonic()
        acquired = semaphore.acquire(timeout=max(1.0, timeout_ms / 1000))
        wait_ms = int((monotonic() - started) * 1000)
        if not acquired:
            raise AgyExecutionError(
                "La cola del runtime Antigravity CLI supero el tiempo de espera antes de obtener un slot.",
                code=AgyRuntimeErrorCode.queue_rejected,
                recoverable=True,
                detail={
                    "runner_id": runner_id,
                    "max_concurrency": max(1, max_concurrency),
                    "queue_wait_ms": wait_ms,
                },
            )
        try:
            yield AgyExecutionSlot(
                runner_id=runner_id,
                max_concurrency=max(1, max_concurrency),
                wait_ms=wait_ms,
            )
        finally:
            semaphore.release()
