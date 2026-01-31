import time
from pathlib import Path

from ai.pipelines.orchestrator.ingest_utils import RateLimiter, read_with_retry


def test_rate_limiter_basic():
    rl = RateLimiter(capacity=2, refill_rate=1.0)
    assert rl.acquire(timeout=0.1) is True
    assert rl.acquire(timeout=0.1) is True
    # now tokens exhausted; immediate acquire should block and return False quickly
    start = time.monotonic()
    ok = rl.acquire(timeout=0.2)
    elapsed = time.monotonic() - start
    assert ok is False or (ok is True and elapsed >= 0.9)


def test_read_with_retry(tmp_path: Path):
    p = tmp_path / "t1.bin"
    data = b"hello"
    p.write_bytes(data)
    out = read_with_retry(p, retry_options={"retries": 1, "backoff_factor": 0.01})
    assert out == data

    # non-existent file should raise after retries
    missing = tmp_path / "missing.bin"
    try:
        read_with_retry(missing, retry_options={"retries": 1, "backoff_factor": 0.01})
        raised = False
    except Exception:
        raised = True
    assert raised is True
import pytest

from ai.pipelines.orchestrator.ingest_utils import retry


def test_retry_succeeds_after_transient(monkeypatch):
    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise OSError("transient")
        return "ok"

    wrapped = retry(retries=5, backoff_factor=0.0, jitter=False, sleep_func=lambda s: None)(flaky)
    assert wrapped() == "ok"
    assert calls["count"] == 3


def test_retry_gives_up(monkeypatch):
    def always_fail():
        raise ValueError("fatal")

    wrapped = retry(retries=2, backoff_factor=0.0, jitter=False, sleep_func=lambda s: None, exceptions=(ValueError,))(always_fail)
    with pytest.raises(ValueError):
        wrapped()


def test_read_with_retry(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello")
    result = read_with_retry(p, retry_options={"retries": 2, "backoff_factor": 0.0, "jitter": False})
    assert result == b"hello"


def test_rate_limiter_basic():
    # use a fake time so test runs fast and is deterministic
    t = {"now": 0.0}

    def time_func():
        return t["now"]

    sleep_calls = []

    def sleep_func(s):
        # advance time as if sleeping
        t["now"] += s
        sleep_calls.append(s)

    rl = RateLimiter(capacity=2, refill_rate=1.0, time_func=time_func)
    assert rl.acquire(blocking=False) is True
    assert rl.acquire(blocking=False) is True
    # now empty
    assert rl.acquire(blocking=False) is False
    # after 1 second one token becomes available
    assert rl.acquire(blocking=True, timeout=2, sleep_func=sleep_func) is True
