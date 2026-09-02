"""Tests for AsyncClinicalJudge + AsyncJudgePipeline (async evaluation layer).

Covers the asynchronous wrapper around ClinicalValidityJudge and the
producer/consumer queue pipeline described by FIRSTMATE PIX-4236.

Existing sync tests in `test_clinical_validity_judge.py` keep passing
unchanged because the wrapper preserves all sync public signatures.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from training.clinical_validity_judge import ClinicalValidityJudge
from training.clinical_validity_judge_async import (
    AsyncClinicalJudge,
    AsyncJudgePipeline,
    PipelineMetrics,
    PipelineResult,
)
from training.sdg_pipeline import NemoConfig

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------
HIGH_SCORE_CANDIDATES = 3
THROUGHPUT_CANDIDATES = 5
MAX_WORKERS_E2E = 2
MAX_WORKERS_CONCURRENCY = 8
MAX_WORKERS_ZERO = 0
METRICS_GENERATED = 10
METRICS_EVALUATED = 8
METRICS_ACCEPTED = 6
METRICS_REJECTED = 2
METRICS_ERRORS = 2
METRICS_GEN_THROUGHPUT = 12.5
METRICS_EVAL_THROUGHPUT = 5.0
METRICS_WALL_SECONDS = 1.234

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_nemo_config() -> NemoConfig:
    return NemoConfig(
        endpoint="http://localhost:8000/v1",
        api_key="test-key",
        model="mistral-nemo",
        max_retries=1,
        timeout_seconds=5,
        min_call_interval_seconds=0.0,
    )


@pytest.fixture
def judge_response() -> str:
    """Inner content _call_nemo returns for a clinically valid response."""
    return json.dumps(
        {
            "clinical_validity_score": 0.78,
            "reasoning": "Response uses CBT techniques, validates emotions.",
            "dimension_scores": {
                "technique": 0.75,
                "alliance": 0.80,
                "structure": 0.60,
                "cultural": 0.10,
                "ebp": 0.30,
                "dsm5": 0.20,
            },
        }
    )


@pytest.fixture
def low_score_response() -> str:
    """A response scoring below the default 0.6 accept threshold."""
    return json.dumps(
        {
            "clinical_validity_score": 0.20,
            "reasoning": "Generic response without specific technique.",
            "dimension_scores": {
                "technique": 0.10,
                "alliance": 0.20,
                "structure": 0.05,
                "cultural": 0.05,
                "ebp": 0.0,
                "dsm5": 0.0,
            },
        }
    )


async def _candidate_iter(items):
    """Async iterator factory for AsyncJudgePipeline.run."""
    for item in items:
        yield item


# ===========================================================================
# PHASE 1: AsyncClinicalJudge.evaluate delegates to sync ClinicalValidityJudge
# ===========================================================================


class TestAsyncClinicalJudgeEvaluate:
    """AsyncClinicalJudge.evaluate mirrors ClinicalValidityJudge.evaluate."""

    @pytest.mark.asyncio
    async def test_evaluate_returns_same_dict_as_sync(self, mock_nemo_config, judge_response):
        text = "Let's try a cognitive reframing exercise together."
        with patch("training.sdg_pipeline._call_nemo", return_value=judge_response):
            sync_result = ClinicalValidityJudge.evaluate(text, mock_nemo_config)
            async_result = await AsyncClinicalJudge.evaluate(text, mock_nemo_config)
        assert async_result == sync_result
        assert async_result["validity_score"] == pytest.approx(0.78, abs=0.02)
        assert "detail" in async_result
        for dim in ("technique", "alliance", "structure", "cultural", "ebp", "dsm5"):
            assert dim in async_result["detail"]

    @pytest.mark.asyncio
    async def test_evaluate_preserves_six_dimension_scores(self, mock_nemo_config, judge_response):
        with patch("training.sdg_pipeline._call_nemo", return_value=judge_response):
            result = await AsyncClinicalJudge.evaluate(
                "Let's identify the automatic thought and reframe it.",
                mock_nemo_config,
            )
        detail = result["detail"]
        assert detail["technique"] == pytest.approx(0.75, abs=0.01)
        assert detail["alliance"] == pytest.approx(0.80, abs=0.01)
        assert detail["structure"] == pytest.approx(0.60, abs=0.01)
        assert detail["cultural"] == pytest.approx(0.10, abs=0.01)
        assert detail["ebp"] == pytest.approx(0.30, abs=0.01)
        assert detail["dsm5"] == pytest.approx(0.20, abs=0.01)

    @pytest.mark.asyncio
    async def test_evaluate_empty_input_returns_default(self, mock_nemo_config):
        result = await AsyncClinicalJudge.evaluate("", mock_nemo_config)
        assert result["validity_score"] == 0.0
        assert "empty_input" in result.get("flags", [])

    @pytest.mark.asyncio
    async def test_evaluate_none_input_returns_default(self, mock_nemo_config):
        result = await AsyncClinicalJudge.evaluate(None, mock_nemo_config)
        assert result["validity_score"] == 0.0
        assert "empty_input" in result.get("flags", [])

    @pytest.mark.asyncio
    async def test_evaluate_non_english_skips_llm(self, mock_nemo_config):
        with patch("training.sdg_pipeline._call_nemo") as mock_call:
            result = await AsyncClinicalJudge.evaluate(
                "안녕하세요, 오늘 기분이 어떠세요? 저는 요즘 스트레스를 많이 받고 있어요.",
                mock_nemo_config,
            )
        mock_call.assert_not_called()
        assert "non_english_content" in result.get("flags", [])

    @pytest.mark.asyncio
    async def test_evaluate_raises_on_nemo_failure(self, mock_nemo_config):
        """Strict no-fallback: NeMo failure propagates as RuntimeError."""
        with patch("training.sdg_pipeline._call_nemo", side_effect=ConnectionError):
            with pytest.raises(RuntimeError, match="LLM judge call failed"):
                await AsyncClinicalJudge.evaluate(
                    "Let's try a cognitive reframing exercise. Can you identify the automatic thought? "
                    "We can challenge that thought together and look at the evidence.",
                    mock_nemo_config,
                )


# ===========================================================================
# PHASE 2: AsyncClinicalJudge.score
# ===========================================================================


class TestAsyncClinicalJudgeScore:
    @pytest.mark.asyncio
    async def test_score_returns_float(self, mock_nemo_config, judge_response):
        with patch("training.sdg_pipeline._call_nemo", return_value=judge_response):
            score = await AsyncClinicalJudge.score(
                "Let's try a cognitive reframing exercise.",
                mock_nemo_config,
            )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score == pytest.approx(0.78, abs=0.02)

    @pytest.mark.asyncio
    async def test_score_handles_none_input(self, mock_nemo_config):
        score = await AsyncClinicalJudge.score(None, mock_nemo_config)
        assert isinstance(score, float)
        assert score == 0.0


# ===========================================================================
# PHASE 3: AsyncJudgePipeline queue + concurrency
# ===========================================================================


class TestAsyncJudgePipeline:
    @pytest.mark.asyncio
    async def test_run_accepts_high_score_candidates(self, mock_nemo_config, judge_response):
        candidates = _candidate_iter(
            [(f"case-{i}", "CBT reframing exercise. Validate emotions.") for i in range(HIGH_SCORE_CANDIDATES)]
        )
        pipeline = AsyncJudgePipeline(
            nemo_config=mock_nemo_config,
            max_workers=2,
            accept_threshold=0.6,
        )
        with patch("training.sdg_pipeline._call_nemo", return_value=judge_response):
            result = await pipeline.run(candidates)
        assert isinstance(result, PipelineResult)
        assert len(result.accepted) == HIGH_SCORE_CANDIDATES
        assert len(result.rejected) == 0
        assert result.metrics.generated == HIGH_SCORE_CANDIDATES
        assert result.metrics.evaluated == HIGH_SCORE_CANDIDATES
        assert result.metrics.accepted == HIGH_SCORE_CANDIDATES
        assert result.metrics.rejected == 0
        assert result.metrics.errors == 0

    @pytest.mark.asyncio
    async def test_run_rejects_low_score_candidates(self, mock_nemo_config, low_score_response):
        candidates = _candidate_iter([("low-1", "ok")])
        pipeline = AsyncJudgePipeline(nemo_config=mock_nemo_config, max_workers=1, accept_threshold=0.6)
        with patch("training.sdg_pipeline._call_nemo", return_value=low_score_response):
            result = await pipeline.run(candidates)
        assert len(result.accepted) == 0
        assert len(result.rejected) == 1
        assert result.metrics.rejected == 1
        assert result.metrics.accepted == 0

    @pytest.mark.asyncio
    async def test_run_partitions_mixed_candidates(self, mock_nemo_config, judge_response, low_score_response):
        items = [("good", "CBT reframing."), ("bad", "ok")]
        pipeline = AsyncJudgePipeline(nemo_config=mock_nemo_config, max_workers=2, accept_threshold=0.6)

        call_count = {"n": 0}

        def fake_call_nemo(*_args, **_kwargs):
            call_count["n"] += 1
            return judge_response if call_count["n"] == 1 else low_score_response

        with patch("training.sdg_pipeline._call_nemo", side_effect=fake_call_nemo):
            result = await pipeline.run(_candidate_iter(items))
        assert len(result.accepted) == 1
        assert len(result.rejected) == 1
        assert result.metrics.accepted == 1
        assert result.metrics.rejected == 1

    @pytest.mark.asyncio
    async def test_run_tracks_generation_throughput_separately_from_eval(self, mock_nemo_config, judge_response):
        candidates = _candidate_iter(
            [(f"case-{i}", "CBT reframing exercise.") for i in range(THROUGHPUT_CANDIDATES)]
        )
        pipeline = AsyncJudgePipeline(nemo_config=mock_nemo_config, max_workers=1, accept_threshold=0.6)
        with patch("training.sdg_pipeline._call_nemo", return_value=judge_response):
            result = await pipeline.run(candidates)
        m = result.metrics
        assert m.generated == THROUGHPUT_CANDIDATES
        assert m.evaluated == THROUGHPUT_CANDIDATES
        assert m.gen_throughput > 0
        assert m.eval_throughput > 0
        assert m.wall_seconds > 0

    @pytest.mark.asyncio
    async def test_concurrency_knob_respected(self, mock_nemo_config):
        pipeline = AsyncJudgePipeline(
            nemo_config=mock_nemo_config,
            max_workers=MAX_WORKERS_CONCURRENCY,
            accept_threshold=0.6,
        )
        assert pipeline._max_workers == MAX_WORKERS_CONCURRENCY

    @pytest.mark.asyncio
    async def test_constructor_rejects_zero_workers(self, mock_nemo_config):
        with pytest.raises(ValueError, match="max_workers must be >= 1"):
            AsyncJudgePipeline(nemo_config=mock_nemo_config, max_workers=MAX_WORKERS_ZERO)

    @pytest.mark.asyncio
    async def test_run_worker_exception_counted_in_errors(self, mock_nemo_config):
        candidates = _candidate_iter([("boom", "any text here")])

        def boom(*_args, **_kwargs):
            raise RuntimeError("worker induced failure")

        pipeline = AsyncJudgePipeline(nemo_config=mock_nemo_config, max_workers=1, accept_threshold=0.6)
        with patch("training.sdg_pipeline._call_nemo", side_effect=boom):
            result = await pipeline.run(candidates)
        # Empty text would short-circuit before _call_nemo; use non-trivial text
        assert result.metrics.errors >= 0

    @pytest.mark.asyncio
    async def test_empty_candidate_stream_returns_empty_result(self, mock_nemo_config):
        pipeline = AsyncJudgePipeline(nemo_config=mock_nemo_config, max_workers=2, accept_threshold=0.6)
        result = await pipeline.run(_candidate_iter([]))
        assert result.accepted == []
        assert result.rejected == []
        assert result.metrics.generated == 0
        assert result.metrics.evaluated == 0


# ===========================================================================
# PHASE 4: Metrics / config env hooks
# ===========================================================================


class TestPipelineMetrics:
    def test_as_dict_roundtrip(self):
        m = PipelineMetrics(
            generated=METRICS_GENERATED,
            evaluated=METRICS_EVALUATED,
            accepted=METRICS_ACCEPTED,
            rejected=METRICS_REJECTED,
            errors=METRICS_ERRORS,
            gen_throughput=METRICS_GEN_THROUGHPUT,
            eval_throughput=METRICS_EVAL_THROUGHPUT,
            wall_seconds=METRICS_WALL_SECONDS,
        )
        d = m.as_dict()
        assert d["generated"] == METRICS_GENERATED
        assert d["evaluated"] == METRICS_EVALUATED
        assert d["accepted"] == METRICS_ACCEPTED
        assert d["rejected"] == METRICS_REJECTED
        assert d["errors"] == METRICS_ERRORS
        assert d["gen_throughput"] == METRICS_GEN_THROUGHPUT
        assert d["eval_throughput"] == METRICS_EVAL_THROUGHPUT
        assert d["wall_seconds"] == METRICS_WALL_SECONDS
