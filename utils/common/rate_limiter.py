"""Tier-aware token-bucket rate limiter for LLM providers.

Source-of-truth for tier resolution (in precedence order):
  1. Env override: ``FIREWORKS_API_TIER`` / ``NVIDIA_API_TIER`` / ``OPENAI_TIER``.
  2. On Fireworks only, when env var absent: GET ``/v1/accounts/me`` with TTL cache
     (default 1h). Failure falls through to static ``starter`` tier.
  3. NVIDIA NIM and OpenAI use static tier tables only.

Tier constants follow the Fireworks serverless rate-limits doc
(``docs.fireworks.ai/serverless/rate-limits``): starter tier begins at
3.6 M Total Prompt TPM, 900 K Uncached Prompt TPM, 36 K Generated TPM.
Concurrency bounds here mirror the published per-(account, model) scoping.

The ``RATELIMIT_DISABLED=1`` env var drops the limiter entirely so callers
can A/B measure without code changes.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TierLimit:
    """Tokens-per-minute budget and concurrency per tier."""

    prompt_tpm: int
    generated_tpm: int
    concurrency: int


# Tier constants are conservative estimates so a minimum-tier account
# cannot be silently throttled. Caps follow the Fireworks starter-tier base:
# 3.6M prompt TPM / 36K generated TPM. Higher tiers scale per the rate-limits
# doc. ``free`` and ``developer`` are included for completeness even though
# Fireworks' published "starter" is the lowest adaptive tier.
DEFAULT_TIER_LIMITS: dict[str, TierLimit] = {
    "free": TierLimit(prompt_tpm=60_000, generated_tpm=1_000, concurrency=1),
    "starter": TierLimit(prompt_tpm=3_600_000, generated_tpm=36_000, concurrency=8),
    "developer": TierLimit(prompt_tpm=1_500_000, generated_tpm=15_000, concurrency=16),
    "pro": TierLimit(prompt_tpm=4_500_000, generated_tpm=45_000, concurrency=32),
    "business": TierLimit(prompt_tpm=12_000_000, generated_tpm=120_000, concurrency=64),
}


PROVIDER_ENV_TIER: dict[str, str] = {
    "fireworks": "FIREWORKS_API_TIER",
    "nvidia": "NVIDIA_API_TIER",
    "openai": "OPENAI_TIER",
}


def _fallback_tier() -> tuple[str, TierLimit]:
    """Default to a conservative starter-tier footprint if a tier is unknown."""

    return "starter", DEFAULT_TIER_LIMITS["starter"]


class TierResolver:
    """Resolve a provider's tier via env override, /v1/accounts/me probe, or static table."""

    def __init__(self, ttl_seconds: int = 3600, timeout_seconds: float = 5.0):
        self._ttl = ttl_seconds
        self._timeout = timeout_seconds
        self._cache: dict[str, tuple[float, str | None]] = {}
        self._lock = threading.Lock()

    def resolve(self, provider: str) -> tuple[str, TierLimit]:
        env_var = PROVIDER_ENV_TIER.get(provider)
        if env_var:
            env = os.environ.get(env_var)
            if env:
                tier = env.strip().lower()
                limits = DEFAULT_TIER_LIMITS.get(tier)
                if limits is not None:
                    return tier, limits
                logger.warning("Unknown tier %r for %s; falling back to starter", tier, provider)

        if provider == "fireworks":
            tier = self._fetch_fireworks_account_tier()
            if tier:
                limits = DEFAULT_TIER_LIMITS.get(tier)
                if limits is not None:
                    return tier, limits
                logger.warning("Fireworks-reported tier %r unknown; falling back to starter", tier)

        return _fallback_tier()

    def _fetch_fireworks_account_tier(self) -> str | None:
        cache_key = "fireworks_account_tier"
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached and (now - cached[0]) < self._ttl:
                return cached[1]

        api_key = os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            with self._lock:
                self._cache[cache_key] = (now, None)
            return None

        try:
            response = httpx.get(
                "https://api.fireworks.ai/v1/accounts/me",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=self._timeout,
            )
            if response.status_code == 200:
                payload = response.json()
                tier = payload.get("account", {}).get("tier") or payload.get("tier") or payload.get("spending_tier")
                tier_name = str(tier).strip().lower() if tier else "starter"
                with self._lock:
                    self._cache[cache_key] = (now, tier_name)
                return tier_name
            logger.warning(
                "Fireworks /v1/accounts/me returned %s (tier unknown, using starter)",
                response.status_code,
            )
        except Exception as exc:  # network probe, all error types collapse to "no tier"
            logger.warning("Fireworks /v1/accounts/me lookup failed: %s", exc)

        with self._lock:
            self._cache[cache_key] = (now, None)
        return None


class TokenBucket:
    """Lock-protected token bucket sized to a per-second refill rate."""

    def __init__(self, refill_per_second: float, capacity: float):
        self._rate = max(refill_per_second, 0.0)
        self._capacity = max(capacity, 1.0)
        self._tokens = self._capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def try_consume(self, tokens: float) -> bool:
        """Return True if tokens were consumed; False if bucket ran dry."""

        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            self._last = now
            if self._tokens < tokens:
                return False
            self._tokens -= tokens
            return True

    def wait_for(self, tokens: float, timeout_seconds: float = 30.0, poll_seconds: float = 0.05) -> bool:
        """Block (sleep-poll) until the bucket has enough tokens or we time out."""

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self.try_consume(tokens):
                return True
            time.sleep(poll_seconds)
        return False


class TierAwareRateLimiter:
    """Bundle a tier resolver and per-(provider, model) buckets.

    Buckets are lazily created on first use so cold paths do not pay for
    probing Fireworks. Set ``RATELIMIT_DISABLED=1`` to short-circuit acquisition.
    """

    def __init__(self, resolver: TierResolver | None = None):
        self._resolver = resolver or TierResolver()
        self._buckets: dict[tuple[str, str], TokenBucket] = {}
        self._concurrency: dict[tuple[str, str], int] = {}
        self._in_flight: dict[tuple[str, str], int] = {}
        self._lock = threading.Lock()

    @staticmethod
    def disabled() -> bool:
        return os.environ.get("RATELIMIT_DISABLED", "").strip() == "1"

    def acquire(
        self,
        provider: str,
        model: str,
        estimated_tokens: int = 1000,
        timeout_seconds: float = 30.0,
    ) -> bool:
        if self.disabled():
            return True

        key = (provider, model)
        # Phase 1: ensure bucket is initialised WITHOUT doing blocking HTTP
        # inside the global lock. (cubic #1)
        with self._lock:
            bucket = self._buckets.get(key)
            concurrency = self._concurrency.get(key)
        if bucket is None or concurrency is None:
            tier_name, limits = self._resolver.resolve(provider)
            new_bucket, new_concurrency = self._build_bucket(limits)
            with self._lock:
                bucket = self._buckets.setdefault(key, new_bucket)
                concurrency = self._concurrency.setdefault(key, new_concurrency)
                self._in_flight.setdefault(key, 0)
                if bucket is new_bucket:
                    logger.info(
                        "Rate limiter initialized for %s/%s tier=%s prompt_tpm=%s generated_tpm=%s concurrency=%s",
                        provider,
                        model,
                        tier_name,
                        limits.prompt_tpm,
                        limits.generated_tpm,
                        concurrency,
                    )

        # Phase 2: wait for the token bucket (does NOT hold the concurrency slot)
        cost = max(estimated_tokens / 1000.0, 1.0)
        acquired = bucket.wait_for(cost, timeout_seconds=timeout_seconds)
        if not acquired:
            return False

        # Phase 3: claim a concurrency slot right before the provider call.
        # The slot now brackets only the in-flight call (cubic #2).
        with self._lock:
            if self._in_flight[key] >= self._concurrency[key]:
                return False
            self._in_flight[key] += 1

        # Caller MUST invoke release_in_flight() after the provider call.
        return True

    def release_in_flight(self, provider: str, model: str) -> None:
        """Release a concurrency slot acquired via :meth:`acquire`.

        The caller MUST invoke this after the provider call returns so the
        per-tier concurrency cap reflects actual in-flight requests, not
        queued ones.
        """
        if self.disabled():
            return
        key = (provider, model)
        with self._lock:
            current = self._in_flight.get(key, 0)
            self._in_flight[key] = max(current - 1, 0)

    @staticmethod
    def _build_bucket(limits: TierLimit) -> tuple[TokenBucket, int]:
        # Refill rate sized to the generated-TPM cap; the limiter is conservative
        # (under-promises, never over) so callers cannot race past Fireworks'
        # published starter-tier ceiling.
        refill = max(limits.generated_tpm / 60_000.0, 0.0)  # cubic #3: no min floor
        capacity = max(float(limits.concurrency) * 4.0, 4.0)
        return TokenBucket(refill_per_second=refill, capacity=capacity), int(limits.concurrency)


# Module-level singleton so callers do not need to thread an instance through.
_default_limiter: TierAwareRateLimiter | None = None
_default_lock = threading.Lock()


def default_rate_limiter() -> TierAwareRateLimiter:
    global _default_limiter
    with _default_lock:
        if _default_limiter is None:
            _default_limiter = TierAwareRateLimiter()
        return _default_limiter


def reset_default_rate_limiter() -> None:
    """Test helper: drop the cached singleton so tests can swap the resolver."""

    global _default_limiter
    with _default_lock:
        _default_limiter = None
