from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import sleep

from app.services.llm_runtime.codex_cli.queue_service import CodexExecutionQueueService
from app.services.llm_runtime.codex_cli.runtime_types import CodexExecutionError, CodexRuntimeErrorCode


def test_queue_service_limits_concurrency_to_one() -> None:
    service = CodexExecutionQueueService()
    lock = Lock()
    active = 0
    max_seen = 0

    def worker() -> None:
        nonlocal active, max_seen
        with service.with_execution_slot(runner_id="local", max_concurrency=1, timeout_ms=1000):
            with lock:
                active += 1
                max_seen = max(max_seen, active)
            sleep(0.05)
            with lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(lambda _: worker(), range(3)))

    assert max_seen == 1


def test_queue_service_allows_two_parallel_slots_when_configured() -> None:
    service = CodexExecutionQueueService()
    lock = Lock()
    active = 0
    max_seen = 0

    def worker() -> None:
        nonlocal active, max_seen
        with service.with_execution_slot(runner_id="parallel", max_concurrency=2, timeout_ms=1000):
            with lock:
                active += 1
                max_seen = max(max_seen, active)
            sleep(0.05)
            with lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _: worker(), range(4)))

    assert 1 < max_seen <= 2


def test_queue_service_raises_queue_rejected_when_wait_expires() -> None:
    service = CodexExecutionQueueService()

    with service.with_execution_slot(runner_id="locked", max_concurrency=1, timeout_ms=1000):
        try:
            with service.with_execution_slot(runner_id="locked", max_concurrency=1, timeout_ms=20):
                raise AssertionError("No debio adquirir un segundo slot.")
        except CodexExecutionError as exc:
            assert exc.code == CodexRuntimeErrorCode.queue_rejected
