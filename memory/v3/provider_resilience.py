"""Resilience helpers for local subconscious providers."""

from __future__ import annotations

import asyncio
import logging
from time import time
from typing import Awaitable, Callable, Optional, TypeVar

import aiosqlite

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SQLiteResilienceController:
    """Wrap retry and circuit-breaker behavior for SQLite-backed providers."""

    def __init__(self, *, bank_id: str, max_retries: int, retry_delay_ms: int) -> None:
        self.bank_id = bank_id
        self.max_retries = max_retries
        self.retry_delay_ms = retry_delay_ms
        self.failure_count = 0
        self.circuit_open = False
        self.last_failure_time: Optional[float] = None

    async def execute(self, operation: Callable[[], Awaitable[T]]) -> T:
        self._check_circuit_breaker()
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                result = await operation()
                self.failure_count = 0
                return result
            except (aiosqlite.Error, asyncio.TimeoutError) as exc:
                last_error = exc
                self._record_retry_failure()
                if attempt < self.max_retries:
                    await self._sleep_before_retry(attempt + 1)
                    continue
                logger.error(
                    "DB failed after %s attempts: %s",
                    self.max_retries + 1,
                    exc,
                )
                raise
        raise last_error if last_error else RuntimeError("Unexpected retry failure")

    def reset(self) -> None:
        self.failure_count = 0
        self.circuit_open = False
        self.last_failure_time = None

    def _check_circuit_breaker(self) -> None:
        if not self.circuit_open:
            return
        if not self.last_failure_time:
            raise RuntimeError(f"Circuit breaker open for {self.bank_id}")
        elapsed = time() - self.last_failure_time
        if elapsed < 60:
            raise RuntimeError(
                f"Circuit breaker open for {self.bank_id}, retry in {60 - elapsed:.0f}s"
            )
        self.circuit_open = False
        self.failure_count = 0
        logger.info("Circuit breaker reset for %s", self.bank_id)

    def _record_retry_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count < 5:
            return
        self.circuit_open = True
        self.last_failure_time = time()
        logger.error(
            "Circuit breaker opened for %s after %s failures",
            self.bank_id,
            self.failure_count,
        )

    async def _sleep_before_retry(self, attempt_number: int) -> None:
        delay = self.retry_delay_ms / 1000
        logger.warning(
            "DB failed: %s/%s. Retry in %ss...",
            attempt_number,
            self.max_retries + 1,
            delay,
        )
        await asyncio.sleep(delay)
