"""
Shared rate limiter and semantic cache for NVIDIA NIM API calls.

Provides per-operation-type token bucket rate limiting and an LRU
semantic cache for cacheable operations like embeddings.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Any, ClassVar

logger = logging.getLogger("foresight_nvidia.rate_limiter")


class TokenBucket:
    """Token bucket rate limiter with configurable burst support.

    Thread-safe via asyncio.Lock. Tokens accrue continuously at
    *rate_per_second* up to *burst*.  Acquire blocks until a token
    is available.
    """

    def __init__(self, rate_per_minute: float, burst: int | None = None) -> None:
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute must be positive")
        self.rate_per_second = rate_per_minute / 60.0
        self.burst = burst if burst is not None else max(1, int(rate_per_minute * 0.1))
        self.tokens = float(self.burst)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """Return the wait time in seconds before the caller should proceed.

        If the returned value is 0 the caller can proceed immediately.
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(float(self.burst), self.tokens + elapsed * self.rate_per_second)
            self.last_refill = now

            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return 0.0

            wait_time = (1.0 - self.tokens) / self.rate_per_second
            self.tokens = 0.0
            self.last_refill = now + wait_time
            return wait_time

    async def wait(self) -> None:
        """Block until a token is available."""
        wait_time = await self.acquire()
        if wait_time > 0:
            logger.debug("Rate limit wait: %.2fs for token bucket", wait_time)
            await asyncio.sleep(wait_time)


class NvidiaRateLimiter:
    """Rate limiter for NVIDIA NIM API calls with per-operation-type limits.

    Default limits:
      - generation (chat completions) : 60 req/min, burst 10
      - embedding                   : 120 req/min, burst 20
      - crisis_detection            : 30 req/min, burst 5

    Override any via constructor kwargs::

        limiter = NvidiaRateLimiter(generation=(30, 5))
    """

    DEFAULTS: ClassVar[dict[str, tuple[float, int | None]]] = {
        "generation": (60, 10),
        "embedding": (120, 20),
        "crisis_detection": (30, 5),
    }

    def __init__(self, **overrides: tuple[float, int | None]) -> None:
        merged = {**self.DEFAULTS, **overrides}
        self._buckets: dict[str, TokenBucket] = {op: TokenBucket(rate, burst) for op, (rate, burst) in merged.items()}

    async def wait(self, op_type: str = "generation") -> None:
        """Block until a token is available for *op_type*."""
        bucket = self._buckets.get(op_type)
        if bucket is None:
            logger.warning(
                "Unknown operation type '%s', falling back to 'generation' limits",
                op_type,
            )
            bucket = self._buckets["generation"]
        await bucket.wait()


class SemanticCache:
    """LRU cache with TTL, designed for semantically cacheable operations.

    Typical use-case: embedding generation where identical input text
    should always produce the same vector without an API round-trip.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300.0) -> None:
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

    def get(self, key: str) -> Any | None:
        """Return cached value or None if missing / expired."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        timestamp, value = entry
        if time.monotonic() - timestamp > self.ttl_seconds:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return value

    def set(self, key: str, value: Any) -> None:
        """Store *value* under *key*, evicting the LRU entry if at capacity."""
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (time.monotonic(), value)
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)
