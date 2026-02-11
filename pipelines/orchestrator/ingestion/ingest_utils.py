"""Clean ingestion utilities used by connectors and tests.

Exports:
- read_with_retry(path_obj, retry_options=None) -> bytes
- RateLimiter(capacity=10, refill_rate=1.0) with acquire(blocking=True, timeout=None)

This module is intentionally minimal and stable; connectors rely on its API.
"""
from __future__ import annotations

import secrets  # Added for secure random generation
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _default_sleep(seconds: float) -> None:
    time.sleep(seconds)


def read_with_retry(
    path_obj: Any,
    retry_options: dict | None = None,
    sleep_func: Callable[[float], None] = _default_sleep,
) -> bytes:
    """Read bytes from a Path-like object with retries and exponential backoff.

    path_obj may be a Path, str, callable returning bytes, or an object with
    a read_bytes() method.

    retry_options keys: retries (int), backoff_factor (float), max_backoff (float),
    jitter (float), exceptions (tuple).
    """
    opts = {
        "retries": 3,
        "backoff_factor": 0.2,
        "max_backoff": 5.0,
        "jitter": 0.0,
        "exceptions": (Exception,),
    }
    if retry_options:
        opts.update(retry_options)

    attempts = 0
    last_exc = None
    while attempts < int(opts["retries"]):
        try:
            if isinstance(path_obj, (str, Path)):
                return Path(path_obj).read_bytes()
            if callable(path_obj):
                return path_obj()
            return path_obj.read_bytes()
        except tuple(opts.get("exceptions", (Exception,))) as e:
            last_exc = e
            attempts += 1
            if attempts >= int(opts["retries"]):
                break
            backoff = min(opts["max_backoff"], opts["backoff_factor"] * (2 ** (attempts - 1)))
            # Use cryptographically secure random for jitter
            jitter = secrets.SystemRandom().random() * float(opts.get("jitter", 0.0))
            sleep_func(backoff + jitter)
    # Re-raise the last exception for visibility to callers
    raise last_exc


class RateLimiter:
    """Token-bucket rate limiter (single-process, not thread-safe).

    Simple implementation sufficient for connectors running in a single thread.
    """

    def __init__(self, capacity: int = 10, refill_rate: float = 1.0, time_func: Callable[[], float] = time.monotonic):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if refill_rate <= 0:
            raise ValueError("refill_rate must be > 0")
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)
        self._tokens = float(capacity)
        self._last = time_func()
        self._time = time_func

    def _refill(self) -> None:
        now = self._time()
        if now <= self._last:
            return
        delta = now - self._last
        self._tokens = min(self.capacity, self._tokens + delta * self.refill_rate)
        self._last = now

    def acquire(self, blocking: bool = True, timeout: float | None = None, sleep_func: Callable[[float], None] = _default_sleep) -> bool:
        """Attempt to consume a token. If blocking=True, wait up to `timeout` seconds.

        Returns True if a token was acquired, False otherwise.
        """
        start = self._time()
        while True:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            if not blocking:
                return False
            if timeout is not None and (self._time() - start) >= float(timeout):
                return False
            sleep_func(0.01)


__all__ = ["RateLimiter", "read_with_retry"]
