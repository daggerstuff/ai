"""Tests for the shared env-configurable generation backend + Moderate guard."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from training import generation_backend as gb


def _mock_response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.status = 200
    response.raise_for_status = MagicMock()
    response.json = AsyncMock(return_value=payload)
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)
    return response


def _vllm_backend() -> gb.BackendConfig:
    return gb.BackendConfig(
        name="vllm",
        url="http://localhost:8000/v1/chat/completions",
        model="@cf/zai-org/glm-5.3",
    )


class TestResolveBackend:
    def test_default_is_cloudflare(self):
        cfg = gb.resolve_backend({})
        assert cfg.name == "cloudflare"
        assert "/ai/v1/chat/completions" in cfg.url
        assert cfg.model == "@cf/zai-org/glm-5.3"

    def test_9router_requires_ninerouter_url(self):
        with pytest.raises(ValueError, match="NINEROUTER_URL"):
            gb.resolve_backend({"NF_BACKEND": "9router"})

    def test_9router_resolves_with_url(self):
        cfg = gb.resolve_backend(
            {
                "NF_BACKEND": "9router",
                "NINEROUTER_URL": "http://localhost:8787",
                "NINEROUTER_API_KEY": "k",
                "NF_MODEL": "qwen/qwen2.5-7b-instruct",
            }
        )
        assert cfg.name == "9router"
        assert cfg.url == "http://localhost:8787/v1/chat/completions"
        assert cfg.auth_header == "Bearer k"

    def test_vllm_resolves_default_url(self):
        cfg = gb.resolve_backend({"NF_BACKEND": "vllm"})
        assert cfg.url == "http://localhost:8000/v1/chat/completions"
        assert cfg.auth_header is None

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="unknown NF_BACKEND"):
            gb.resolve_backend({"NF_BACKEND": "openai"})

    def test_never_llama_rejected(self):
        with pytest.raises(ValueError, match="never-Llama"):
            gb.resolve_backend({"NF_MODEL": "meta/llama-3.1-8b-instruct"})

    def test_non_llama_families_allowed(self):
        for model in (
            "qwen/qwen2.5-7b-instruct",
            "mistralai/mistral-7b-instruct-v0.3",
            "@cf/zai-org/glm-5.3",
        ):
            cfg = gb.resolve_backend({"NF_MODEL": model, "NF_BACKEND": "vllm"})
            assert cfg.model == model


class TestModerateGuard:
    def test_defaults_are_moderate_policy(self):
        assert gb.ModerateGuard.HOURLY_LIMIT == 10_000
        assert gb.ModerateGuard.HARD_CEILING == 50_000

    def test_records_within_limits(self):
        guard = gb.ModerateGuard(hourly_limit=10, hard_ceiling=10)
        for _ in range(10):
            guard.record()
        assert guard.total == 10

    def test_hard_ceiling_triggers(self):
        guard = gb.ModerateGuard(hourly_limit=1000, hard_ceiling=2)
        guard.record()
        guard.record()
        with pytest.raises(gb.GenerationLimitExceededError, match="hard ceiling"):
            guard.record()

    def test_hourly_rate_triggers(self):
        guard = gb.ModerateGuard(hourly_limit=3, hard_ceiling=100)
        for _ in range(3):
            guard.record()
        with pytest.raises(gb.GenerationLimitExceededError, match="hourly rate"):
            guard.record()

    def test_hourly_window_prunes_old_records(self):
        ticks = iter([0.0, 0.0, 0.0, 4000.0, 4000.0])
        guard = gb.ModerateGuard(hourly_limit=3, hard_ceiling=100, clock=lambda: next(ticks))
        for _ in range(3):
            guard.record()
        # Window advanced past 3600s: the first three are pruned, no overrun.
        guard.record()
        guard.record()
        assert guard.total == 5

    def test_record_many_counts_all(self):
        guard = gb.ModerateGuard(hourly_limit=1000, hard_ceiling=10)
        guard.record_many([object()] * 5)
        assert guard.total == 5


class TestChatCompletion:
    @pytest.mark.asyncio
    async def test_returns_content_and_counts_tokens(self, monkeypatch):
        monkeypatch.setattr(gb, "resolve_backend", lambda: _vllm_backend())
        payload = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        session = MagicMock()
        session.post = MagicMock(return_value=_mock_response(payload))
        counter: dict = {}
        result = await gb.chat_completion(
            session,
            [{"role": "user", "content": "ping"}],
            temperature=0.5,
            token_counter=counter,
        )
        assert result == "ok"
        assert counter["prompt_tokens"] == 10
        assert counter["completion_tokens"] == 5
        assert counter["total_tokens"] == 15

    @pytest.mark.asyncio
    async def test_429_raises_rate_limit_error(self, monkeypatch):
        monkeypatch.setattr(gb, "resolve_backend", lambda: _vllm_backend())
        response = MagicMock()
        response.status = 429
        response.raise_for_status = MagicMock()
        response.json = AsyncMock(return_value={})
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=False)
        session = MagicMock()
        session.post = MagicMock(return_value=response)
        with pytest.raises(gb.RateLimitError):
            await gb.chat_completion(
                session,
                [{"role": "user", "content": "ping"}],
                temperature=0.5,
            )


class TestObservability:
    def test_log_generation_call_returns_dict(self):
        result = gb._log_generation_call(
            backend="vllm",
            model="@cf/zai-org/glm-5.3",
            metrics={
                "input_chars": 10,
                "output_chars": 5,
                "latency_ms": 1.0,
                "prompt_tokens": 2,
                "completion_tokens": 3,
            },
        )
        assert result["backend"] == "vllm"
        assert result["model"] == "@cf/zai-org/glm-5.3"
        assert result["output_chars"] == 5

    def test_init_weave_is_safe_noop(self, monkeypatch):
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        # No-op regardless of weave availability; must never raise.
        assert gb.init_weave() is None