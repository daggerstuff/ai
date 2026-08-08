"""Tests for nightmare_fuel_generator async rewrite + PIX-4235 checkpointing."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from training import nightmare_fuel_generator as nfg


def _mock_response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = AsyncMock(return_value=payload)
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)
    return response


class TestParseTranscript:
    def test_parse_transcript_splits_roles(self):
        transcript = (
            "Patient: I don't want to talk.\n"
            "Therapist: I hear that this feels difficult.\n"
            "Patient: You don't understand me."
        )
        parsed = nfg._parse_transcript(transcript, "scenario")
        assert parsed["scenario"] == "scenario"
        assert parsed["messages"] == [
            {"role": "user", "content": "I don't want to talk."},
            {"role": "assistant", "content": "I hear that this feels difficult."},
            {"role": "user", "content": "You don't understand me."},
        ]


class TestAsyncGeneration:
    @pytest.mark.asyncio
    async def test_generate_nightmare_scenario_async(self):
        session = MagicMock()
        session.post = MagicMock(
            return_value=_mock_response(
                {"choices": [{"message": {"content": "A resistant patient with comorbidities."}}]}
            )
        )

        scenario = await nfg.generate_nightmare_scenario_async(session)
        assert scenario == "A resistant patient with comorbidities."
        session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_simulate_therapy_session_async(self):
        session = MagicMock()
        session.post = MagicMock(
            return_value=_mock_response(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "Patient: Leave me alone.\n"
                                    "Therapist: I can stay with you while we figure out next steps."
                                )
                            }
                        }
                    ]
                }
            )
        )

        result = await nfg.simulate_therapy_session_async(session, "scenario")
        assert result["scenario"] == "scenario"
        assert len(result["messages"]) == 2

    @pytest.mark.asyncio
    async def test_generate_cases_async_runs_concurrently(self):
        call_count = 0
        in_flight = 0
        max_in_flight = 0

        async def fake_scenario(*_args, **_kwargs):
            nonlocal call_count, in_flight, max_in_flight
            call_count += 1
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return f"scenario-{call_count}"

        async def fake_session(_session, scenario, **_kw):
            return {
                "scenario": scenario,
                "messages": [
                    {"role": "user", "content": "Patient line"},
                    {"role": "assistant", "content": "Therapist line"},
                ],
            }

        with (
            patch.object(nfg, "generate_nightmare_scenario_async", side_effect=fake_scenario),
            patch.object(nfg, "simulate_therapy_session_async", side_effect=fake_session),
        ):
            cases = await nfg.generate_cases_async(num_cases=3, concurrency=3)

        assert len(cases) == 3
        assert call_count == 3
        assert max_in_flight > 1, "concurrency not achieved: tasks ran sequentially"


class TestSyncWrappers:
    def test_generate_nightmare_scenario_sync_wrapper(self):
        with patch.object(
            nfg,
            "generate_nightmare_scenario_async",
            new=AsyncMock(return_value="sync scenario"),
        ):
            assert nfg.generate_nightmare_scenario() == "sync scenario"

    def test_main_runs_async_pipeline(self, tmp_path):
        """main() should call main_async which drives the pipeline.

        We patch sys.argv to provide argparse defaults and patch the async
        entrypoint so we do not exercise the network or the disk pipeline.
        """

        with (
            patch("sys.argv", ["nightmare_fuel_generator"]),
            patch.object(nfg, "main_async", new=AsyncMock()) as main_async_mock,
        ):
            nfg.main()

        main_async_mock.assert_awaited_once()


class TestBatchController:
    def test_initial_batch_size_within_bounds(self):
        ctrl = nfg.BatchController(
            initial_batch_size=8,
            min_batch_size=2,
            max_batch_size=32,
            target_tokens_per_batch=4096,
        )
        assert ctrl.current_batch_size == 8

    def test_clamps_initial_to_min(self):
        ctrl = nfg.BatchController(
            initial_batch_size=1,
            min_batch_size=4,
            max_batch_size=32,
            target_tokens_per_batch=4096,
        )
        assert ctrl.current_batch_size == 4

    def test_clamps_initial_to_max(self):
        ctrl = nfg.BatchController(
            initial_batch_size=99,
            min_batch_size=2,
            max_batch_size=32,
            target_tokens_per_batch=4096,
        )
        assert ctrl.current_batch_size == 32

    def test_grow_on_fast_low_token_batches(self):
        ctrl = nfg.BatchController(
            initial_batch_size=4,
            min_batch_size=2,
            max_batch_size=32,
            target_tokens_per_batch=4096,
        )
        # Fast + under token budget -> grow.
        ctrl.record_batch(duration_seconds=0.2, tokens_used=512, rate_limited=False)
        ctrl.adjust()
        assert ctrl.current_batch_size > 4

    def test_shrink_on_slow_batches(self):
        ctrl = nfg.BatchController(
            initial_batch_size=16,
            min_batch_size=2,
            max_batch_size=32,
            target_tokens_per_batch=4096,
        )
        # Slow per-slot: 20s over 16 concurrent slots ~ 1.25s/slot > 1s threshold -> shrink.
        ctrl.record_batch(duration_seconds=20.0, tokens_used=1024, rate_limited=False)
        ctrl.adjust()
        assert ctrl.current_batch_size < 16

    def test_shrink_on_token_overrun(self):
        ctrl = nfg.BatchController(
            initial_batch_size=16,
            min_batch_size=2,
            max_batch_size=32,
            target_tokens_per_batch=2048,
        )
        ctrl.record_batch(duration_seconds=0.2, tokens_used=4096, rate_limited=False)
        ctrl.adjust()
        assert ctrl.current_batch_size < 16

    def test_rate_limit_triggers_backoff_and_shrink(self):
        ctrl = nfg.BatchController(
            initial_batch_size=16,
            min_batch_size=2,
            max_batch_size=32,
            target_tokens_per_batch=4096,
        )
        ctrl.record_batch(duration_seconds=0.5, tokens_used=512, rate_limited=True)
        ctrl.adjust()
        assert ctrl.current_batch_size < 16
        assert ctrl.backoff_delay() > 0.0

    def test_backoff_clears_after_successful_batch(self):
        ctrl = nfg.BatchController(
            initial_batch_size=16,
            min_batch_size=2,
            max_batch_size=32,
            target_tokens_per_batch=4096,
        )
        ctrl.record_batch(duration_seconds=0.5, tokens_used=512, rate_limited=True)
        ctrl.adjust()
        assert ctrl.backoff_delay() > 0.0
        ctrl.record_batch(duration_seconds=0.5, tokens_used=512, rate_limited=False)
        ctrl.adjust()
        assert ctrl.backoff_delay() == 0.0

    def test_exponential_backoff_caps_at_max(self):
        ctrl = nfg.BatchController(
            initial_batch_size=4,
            min_batch_size=2,
            max_batch_size=32,
            target_tokens_per_batch=4096,
            backoff_base=2.0,
            backoff_max=10.0,
        )
        for _ in range(5):
            ctrl.record_batch(duration_seconds=0.5, tokens_used=512, rate_limited=True)
            ctrl.adjust()
        assert ctrl.backoff_delay() <= 10.0


def _mock_429_response() -> MagicMock:
    """Build a mock response that raises HTTP 429 via raise_for_status()."""
    import aiohttp

    response = MagicMock()
    response.status = 429
    response.raise_for_status = MagicMock(
        side_effect=aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=429,
            message="Too Many Requests",
        )
    )
    response.json = AsyncMock(return_value={})
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)
    return response


class TestChatCompletion429:
    """PIX-4234: _chat_completion raises RateLimitError on HTTP 429."""

    @pytest.mark.asyncio
    async def test_raises_rate_limit_error_on_429(self):
        session = MagicMock()
        session.post = MagicMock(return_value=_mock_429_response())
        with pytest.raises(nfg.RateLimitError):
            await nfg._chat_completion(
                session,
                [{"role": "user", "content": "ping"}],
                temperature=0.5,
            )

    @pytest.mark.asyncio
    async def test_token_counter_records_usage(self):
        payload = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        session = MagicMock()
        session.post = MagicMock(return_value=_mock_response(payload))
        counter: dict = {}
        result = await nfg._chat_completion(
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
    async def test_token_counter_omitted_keeps_legacy_behavior(self):
        """When token_counter not passed, behavior unchanged."""
        payload = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        session = MagicMock()
        session.post = MagicMock(return_value=_mock_response(payload))
        result = await nfg._chat_completion(
            session,
            [{"role": "user", "content": "ping"}],
            temperature=0.5,
        )
        assert result == "ok"


class TestGenerateCasesBatched:
    """PIX-4234: generate_cases_async uses BatchController for dynamic sizing."""

    @pytest.mark.asyncio
    async def test_concurrency_kwarg_seeds_initial_batch_size(self):
        """concurrency kwarg must seed BatchController so first batch is >= concurrency."""
        # 6 cases, concurrency=4 -> BatchController should start at 4 -> all 6 dispatched
        # across 2 batches.  Both batches must see concurrent in_flight.
        call_count = 0
        in_flight = 0
        max_in_flight = 0

        async def fake_scenario(*_args, **_kwargs):
            nonlocal call_count, in_flight, max_in_flight
            call_count += 1
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return f"scenario-{call_count}"

        async def fake_session(_session, scenario, **_kw):
            return {
                "scenario": scenario,
                "messages": [
                    {"role": "user", "content": "Patient line"},
                    {"role": "assistant", "content": "Therapist line"},
                ],
            }

        with (
            patch.object(nfg, "generate_nightmare_scenario_async", side_effect=fake_scenario),
            patch.object(nfg, "simulate_therapy_session_async", side_effect=fake_session),
        ):
            cases = await nfg.generate_cases_async(num_cases=6, concurrency=4)

        assert len(cases) == 6
        assert call_count == 6
        assert max_in_flight > 1, "batch sizing not achieving concurrency"

    @pytest.mark.asyncio
    async def test_rate_limit_errors_dropped_but_pipeline_continues(self):
        """When _generate_case raises RateLimitError, batch drops it but continues."""
        # 5 cases; first 2 raise RateLimitError, rest succeed.
        call_count = 0

        async def fake_scenario(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise nfg.RateLimitError("simulated 429")
            await asyncio.sleep(0.01)
            return f"scenario-{call_count}"

        async def fake_session(_session, scenario, **_kw):
            return {
                "scenario": scenario,
                "messages": [
                    {"role": "user", "content": "p"},
                    {"role": "assistant", "content": "t"},
                ],
            }

        with (
            patch.object(nfg, "generate_nightmare_scenario_async", side_effect=fake_scenario),
            patch.object(nfg, "simulate_therapy_session_async", side_effect=fake_session),
        ):
            cases = await nfg.generate_cases_async(num_cases=5, concurrency=5)

        # First 2 raised RateLimitError -> dropped; last 3 survived.
        assert len(cases) == 3
        assert call_count == 5

    @pytest.mark.asyncio
    async def test_backoff_delay_applied_after_rate_limit(self):
        """After a rate-limited batch, next batch must await the backoff delay."""
        call_count = 0
        backoff_observed = 0.0

        async def fake_scenario(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise nfg.RateLimitError("simulated 429")
            await asyncio.sleep(0.01)
            return f"scenario-{call_count}"

        async def fake_session(_session, scenario, **_kw):
            return {
                "scenario": scenario,
                "messages": [
                    {"role": "user", "content": "p"},
                    {"role": "assistant", "content": "t"},
                ],
            }

        async def fake_sleep(delay, *_a, **_kw):
            nonlocal backoff_observed
            backoff_observed = max(backoff_observed, delay)

        with (
            patch.object(nfg, "generate_nightmare_scenario_async", side_effect=fake_scenario),
            patch.object(nfg, "simulate_therapy_session_async", side_effect=fake_session),
            patch.object(nfg.asyncio, "sleep", side_effect=fake_sleep),
        ):
            await nfg.generate_cases_async(
                num_cases=4,
                concurrency=2,
                backoff_base=2.0,
                backoff_max=10.0,
            )

        assert backoff_observed > 0, "expected backoff delay after rate-limited batch"
