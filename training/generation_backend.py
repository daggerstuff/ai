"""Single env-configurable generation backend for all training generators.

One OpenAI-compatible client resolves to Cloudflare Workers AI, 9Router, or a
local vLLM endpoint from environment variables. It also enforces the permanent
never-Llama rule and carries the Colab credit-burn guard (Moderate policy).

Backend resolution
------------------
``NF_BACKEND`` selects the provider:

* ``cloudflare`` (default) — Workers AI OpenAI-compatible endpoint.
* ``9router`` — OpenAI-compatible gateway at ``NINEROUTER_URL``.
* ``vllm`` — local/remote OpenAI-compatible server at ``VLLM_URL``.

``NF_MODEL`` selects the model ID; any Llama-family ID is rejected at the config
layer (permanent rule: allowed families are GLM, Qwen, Mistral).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import aiohttp

try:
    import weave as _weave
except ImportError:  # Weave is optional; tracing only activates when installed
    _weave = None

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = float(os.environ.get("NF_REQUEST_TIMEOUT", "120"))
_HTTP_TOO_MANY_REQUESTS = 429
_WEAVE_STATE = {"initialized": False}


class RateLimitError(Exception):
    """Raised when the endpoint returns HTTP 429."""


class EmptyContentError(RuntimeError):
    """Raised when a 200 response carries no usable assistant ``content``.

    Happens when a reasoning model exhausts its budget on ``reasoning_content``
    and returns ``finish_reason="length"`` with an empty answer. Retryable.
    """


class GenerationLimitExceededError(Exception):
    """Raised when the Moderate guard's hourly or hard ceiling is exceeded."""


# Permanent rule: never Llama. Allowed families: GLM, Qwen, Mistral.
_LLAMA_MARKERS = ("llama",)


def is_llama_model(model: str) -> bool:
    """Return True if a model ID belongs to the forbidden Llama family."""
    lowered = model.lower()
    return any(marker in lowered for marker in _LLAMA_MARKERS)


@dataclass(frozen=True)
class BackendConfig:
    """Resolved endpoint + auth + model for the OpenAI-compatible client."""

    name: str
    url: str
    model: str
    auth_header: str | None = None


def _cloudflare_api_token(env: dict[str, str]) -> str:
    # Prefer the dedicated Workers AI key first. The generic CLOUDFLARE_API_TOKEN
    # can be repurposed (e.g. an R2 API token), which would 401 on Workers AI.
    # The dedicated key lives under _KEY in .env, not _TOKEN — both are honored.
    return (
        env.get("CLOUDFLARE_WORKERS_AI_API_KEY")
        or env.get("CLOUDFLARE_AI_API_KEY")
        or env.get("CF_AIG_TOKEN")
        or env.get("CLOUDFLARE_WORKERS_AI_API")
        or env.get("CLOUDFLARE_WORKERS_AI_API_TOKEN")
        or env.get("CLOUDFLARE_AUTH_TOKEN")
        or env.get("CLOUDFLARE_API_TOKEN")
        or env.get("CLOUDFLARE_TOKEN")
        or "dummy"
    )


def resolve_backend(env: dict[str, str] | None = None) -> BackendConfig:
    """Resolve the OpenAI-compatible generation backend from environment.

    Reads ``NF_BACKEND`` (cloudflare|9router|vllm) and ``NF_MODEL``. Raises
    ``ValueError`` on an unknown backend, a missing required URL, or a
    never-Llama model ID.
    """
    env = dict(os.environ if env is None else env)
    backend = env.get("NF_BACKEND", "cloudflare").strip().lower()
    model = env.get("NF_MODEL", "@cf/deepseek-ai/deepseek-v4-pro-0813")

    if is_llama_model(model):
        raise ValueError(
            f"never-Llama rule violated: NF_MODEL={model!r}. "
            "Allowed families: GLM, Qwen, Mistral."
        )

    if backend == "cloudflare":
        account_id = env.get("CLOUDFLARE_ACCOUNT_ID", "")
        url = (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{account_id}/ai/v1/chat/completions"
        )
        return BackendConfig(
            name="cloudflare",
            url=url,
            model=model,
            auth_header=f"Bearer {_cloudflare_api_token(env)}",
        )

    if backend == "9router":
        base = env.get("NINEROUTER_URL", "").rstrip("/")
        if not base:
            raise ValueError("NF_BACKEND=9router requires NINEROUTER_URL to be set")
        token = env.get("NINEROUTER_API_KEY") or env.get("NINEROUTER_KEY") or ""
        return BackendConfig(
            name="9router",
            url=f"{base}/v1/chat/completions",
            model=model,
            auth_header=f"Bearer {token}" if token else None,
        )

    if backend == "vllm":
        base = env.get("VLLM_URL", "http://localhost:8000").rstrip("/")
        token = env.get("VLLM_API_KEY") or ""
        return BackendConfig(
            name="vllm",
            url=f"{base}/v1/chat/completions",
            model=model,
            auth_header=f"Bearer {token}" if token else None,
        )

    raise ValueError(f"unknown NF_BACKEND={backend!r}; expected cloudflare|9router|vllm")


class ModerateGuard:
    """Colab credit-burn guard: 10k records/hour + 50k hard ceiling.

    Moderate policy (resolved decision 2): bulk generation runs Colab-first but
    must auto-kill if throughput exceeds 10k records/hour or if the run ever
    exceeds the 50k hard ceiling. ``record`` raises ``GenerationLimitExceededError``
    when either limit is crossed; the caller lets it propagate to terminate the
    run and checkpoint/resume picks up where it stopped on the next invocation.
    """

    HOURLY_LIMIT = 10_000
    HARD_CEILING = 50_000

    def __init__(
        self,
        *,
        hourly_limit: int = HOURLY_LIMIT,
        hard_ceiling: int = HARD_CEILING,
        window_seconds: float = 3600.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.hourly_limit = hourly_limit
        self.hard_ceiling = hard_ceiling
        self.window_seconds = window_seconds
        self._clock = clock or time.monotonic
        self._total = 0
        self._window: list[float] = []

    @property
    def total(self) -> int:
        return self._total

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._window and self._window[0] < cutoff:
            self._window.pop(0)

    def record(self, count: int = 1) -> None:
        if count <= 0:
            return
        now = self._clock()
        self._prune(now)
        self._total += count
        self._window.extend([now] * count)
        if self._total > self.hard_ceiling:
            raise GenerationLimitExceededError(
                f"hard ceiling exceeded: {self._total} > {self.hard_ceiling} records"
            )
        if len(self._window) > self.hourly_limit:
            raise GenerationLimitExceededError(
                f"hourly rate exceeded: {len(self._window)} records in "
                f"{self.window_seconds:.0f}s (limit {self.hourly_limit})"
            )

    def record_many(self, records: list[Any]) -> None:
        self.record(len(records))


def init_weave(project: str | None = None) -> None:
    """Init Weave tracing once, when weave + W&B credentials are available.

    No-op when ``weave`` is not installed or ``WANDB_API_KEY`` is unset. Idempotent:
    after the first successful init the module state short-circuits further calls.
    """
    if _weave is None or _WEAVE_STATE["initialized"]:
        return
    if not os.getenv("WANDB_API_KEY"):
        return
    project = project or os.getenv("WANDB_PROJECT", "pixelated-empathy-kan28")
    try:
        _weave.init(project)
        _WEAVE_STATE["initialized"] = True
    except Exception:
        logger.warning("weave.init failed; generation tracing disabled", exc_info=True)


def _log_generation_call(
    *,
    backend: str,
    model: str,
    metrics: dict[str, int | float],
) -> dict[str, Any]:
    """Record one generation call (Weave op when weave is installed)."""
    return {"backend": backend, "model": model, **metrics}


if _weave is not None:
    _log_generation_call = _weave.op()(_log_generation_call)


async def chat_completion(
    session: aiohttp.ClientSession,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    token_counter: dict | None = None,
    max_retries: int = 3,
    max_tokens: int | None = None,
) -> str:
    """OpenAI-compatible chat completion with 429 + transient-error handling.

    Mirrors ``nightmare_fuel_generator._chat_completion`` semantics: 429 raises
    ``RateLimitError`` immediately (callers rate-limit/backoff), transient
    transport errors retry with exponential backoff, and token usage is folded
    into ``token_counter`` when present.

    ``max_tokens`` bounds the completion length; when omitted it falls back to
    ``NF_MAX_TOKENS`` (default 4096). A 4-turn clinical dialogue spends a large
    budget before the final ``content`` lands, so the cap must be generous or
    the model finishes with ``finish_reason="length"`` and an empty answer.
    """
    backend = resolve_backend()
    init_weave()
    started = time.monotonic()
    if max_tokens is None:
        max_tokens = int(os.environ.get("NF_MAX_TOKENS", "4096"))
    payload = {
        "model": backend.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    if backend.auth_header:
        headers["Authorization"] = backend.auth_header
    last_error: BaseException | None = None
    for attempt in range(max_retries):
        try:
            async with session.post(
                backend.url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS),
            ) as response:
                if response.status == _HTTP_TOO_MANY_REQUESTS:
                    raise RateLimitError("HTTP 429: rate limit exceeded")
                response.raise_for_status()
                data = await response.json()
            if token_counter is not None and "usage" in data:
                usage = data["usage"]
                token_counter["prompt_tokens"] = token_counter.get("prompt_tokens", 0) + usage.get("prompt_tokens", 0)
                token_counter["completion_tokens"] = token_counter.get("completion_tokens", 0) + usage.get(
                    "completion_tokens", 0
                )
                token_counter["total_tokens"] = (
                    token_counter.get("total_tokens", 0)
                    + usage.get("prompt_tokens", 0)
                    + usage.get("completion_tokens", 0)
                )
            content = data["choices"][0]["message"].get("content") or ""
            if not content.strip():
                raise EmptyContentError(
                    "endpoint returned an empty assistant message "
                    f"(finish_reason={data['choices'][0].get('finish_reason')!r})"
                )
            usage = data.get("usage", {})
            _log_generation_call(
                backend=backend.name,
                model=backend.model,
                metrics={
                    "input_chars": sum(len(m.get("content", "")) for m in messages),
                    "output_chars": len(content or ""),
                    "latency_ms": (time.monotonic() - started) * 1000,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                },
            )
            return content
        except RateLimitError:
            raise
        except (EmptyContentError, TimeoutError, asyncio.CancelledError, aiohttp.ClientError) as e:
            last_error = e
            wait = min(2**attempt * 5, 30)
            logger.warning(
                "generation_backend request failed (attempt %d/%d): %s — retrying in %ds",
                attempt + 1,
                max_retries,
                type(e).__name__,
                wait,
            )
            await asyncio.sleep(wait)
    if last_error is None:
        last_error = RuntimeError("chat completion failed without a recorded error")
    raise last_error
