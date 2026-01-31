#!/usr/bin/env python3
import time
from pathlib import Path

from ai.pipelines.orchestrator.ingest_utils import RateLimiter, read_with_retry


def smoke_read_with_retry():
    p = Path("/tmp") / "smoke_test_file.bin"
    p.write_bytes(b"hello-smoke")
    data = read_with_retry(p, retry_options={"retries": 2, "backoff_factor": 0.0, "jitter": False})
    assert data == b"hello-smoke"


def smoke_rate_limiter():
    rl = RateLimiter(rate=2.0, burst=2)
    assert rl.try_acquire()
    assert rl.try_acquire()
    assert not rl.try_acquire()
    time.sleep(0.6)
    assert rl.try_acquire()


if __name__ == "__main__":
    smoke_read_with_retry()
    smoke_rate_limiter()
