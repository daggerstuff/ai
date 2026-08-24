"""Unit tests for tier-aware rate limiter and HTTP driver dispatch."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from ai.utils.common.rate_limiter import (
    DEFAULT_TIER_LIMITS,
    PROVIDER_ENV_TIER,
    TierAwareRateLimiter,
    TierResolver,
    TokenBucket,
    default_rate_limiter,
    reset_default_rate_limiter,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "FIREWORKS_API_TIER",
        "NVIDIA_API_TIER",
        "OPENAI_TIER",
        "RATELIMIT_DISABLED",
        "FIREWORKS_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    reset_default_rate_limiter()


def test_token_bucket_capacity_refill() -> None:
    bucket = TokenBucket(refill_per_second=10.0, capacity=5.0)
    assert bucket.try_consume(5.0) is True
    assert bucket.try_consume(1.0) is False
    time.sleep(0.15)
    assert bucket.try_consume(1.0) is True


def test_token_bucket_wait_for_timeout() -> None:
    bucket = TokenBucket(refill_per_second=0.01, capacity=1.0)
    assert bucket.try_consume(1.0) is True
    assert bucket.wait_for(1.0, timeout_seconds=0.05) is False


def test_resolver_env_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIREWORKS_API_TIER", "pro")
    resolver = TierResolver()
    tier, limits = resolver.resolve("fireworks")
    assert tier == "pro"
    assert limits is DEFAULT_TIER_LIMITS["pro"]


def test_resolver_env_unknown_tier_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIREWORKS_API_TIER", "mystery")
    resolver = TierResolver()
    tier, _ = resolver.resolve("fireworks")
    assert tier == "starter"


def test_resolver_static_lookup_unused_for_non_fireworks() -> None:
    resolver = TierResolver()
    with patch.object(resolver, "_fetch_fireworks_account_tier") as probe:
        tier, _ = resolver.resolve("openai")
        probe.assert_not_called()
    assert tier == "starter"


def test_resolver_account_endpoint_called_when_no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {"account": {"tier": "Pro"}}
    with patch("httpx.get", return_value=fake_response) as mock_get:
        resolver = TierResolver()
        tier, _ = resolver.resolve("fireworks")
    mock_get.assert_called_once()
    assert tier == "pro"


def test_resolver_account_endpoint_caches_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {"tier": "developer"}
    with patch("httpx.get", return_value=fake_response) as mock_get:
        resolver = TierResolver()
        resolver.resolve("fireworks")
        resolver.resolve("fireworks")
    assert mock_get.call_count == 1


def test_resolver_account_endpoint_failure_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    with patch("httpx.get", side_effect=RuntimeError("network down")):
        resolver = TierResolver()
        tier, _ = resolver.resolve("fireworks")
    assert tier == "starter"


def test_limiter_disabled_env_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RATELIMIT_DISABLED", "1")
    limiter = TierAwareRateLimiter()
    assert limiter.disabled() is True
    assert limiter.acquire(provider="fireworks", model="anything") is True


def test_limiter_concurrency_cap_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIREWORKS_API_TIER", "free")
    limiter = TierAwareRateLimiter()
    key = ("fireworks", "m")
    assert limiter.acquire(provider="fireworks", model="m") is True
    assert limiter._concurrency[key] == 1
    limiter._in_flight[key] = 1
    assert limiter.acquire(provider="fireworks", model="m", timeout_seconds=0.05) is False


def test_limiter_initializes_bucket_lazily() -> None:
    limiter = TierAwareRateLimiter()
    assert limiter.acquire(provider="nvidia", model="meta/llama-3.1-405b-instruct") is True


def test_default_singleton_reset_replaces_instance() -> None:
    first = default_rate_limiter()
    reset_default_rate_limiter()
    second = default_rate_limiter()
    assert first is not second


def test_provider_env_keys_present() -> None:
    assert set(PROVIDER_ENV_TIER) == {"fireworks", "nvidia", "openai"}
    assert PROVIDER_ENV_TIER["fireworks"] == "FIREWORKS_API_TIER"
