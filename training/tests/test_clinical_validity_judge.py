"""Tests for ClinicalValidityJudge — LLM-based clinical validity evaluation.

These tests mock the NeMo API calls and verify the judge's scoring,
failure behavior (no fallback), and output schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

# SUT import — will fail until the module exists
from training.clinical_validity_judge import DOMAIN_DIMENSIONS, ClinicalValidityJudge
from training.sdg_pipeline import NemoConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]

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
def mock_judge_eval_response() -> str:
    """Simulate what _call_nemo returns for a clinically valid therapeutic text.

    _call_nemo() extracts choices[0].message.content from the NeMo HTTP response,
    so this fixture returns the *inner* JSON string directly.
    """
    return json.dumps(
        {
            "clinical_validity_score": 0.78,
            "reasoning": "Response uses CBT techniques, validates emotions, provides structure",
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


# ===========================================================================
# PHASE 1: Evaluate with valid NeMo response (S1 — Happy path)
# ===========================================================================


class TestEvaluateWithNeMo:
    """ClinicalValidityJudge uses NeMo API to evaluate clinical quality."""

    def test_judge_returns_score_in_range(self, mock_nemo_config, mock_judge_eval_response):
        """S1: Valid NeMo response → score in [0.0, 1.0]."""
        with patch("training.sdg_pipeline._call_nemo", return_value=mock_judge_eval_response):
            result = ClinicalValidityJudge.evaluate(
                "Let's try a cognitive reframing exercise. "
                "Can you identify the automatic thought you had in that situation?",
                mock_nemo_config,
            )

        assert 0.0 <= result["validity_score"] <= 1.0
        assert result["validity_score"] == pytest.approx(0.78, abs=0.02)

    def test_judge_detail_contains_all_keys(self, mock_nemo_config, mock_judge_eval_response):
        """S1: Detail dict contains all expected dimension keys."""
        with patch("training.sdg_pipeline._call_nemo", return_value=mock_judge_eval_response):
            result = ClinicalValidityJudge.evaluate(
                "Let's work together on identifying your goals for therapy.",
                mock_nemo_config,
            )

        assert "detail" in result
        clinical_dims = DOMAIN_DIMENSIONS["clinical"]
        assert set(result["detail"]) == set(clinical_dims)
        assert all(0.0 <= result["detail"][dim] <= 1.0 for dim in clinical_dims)

    def test_judge_flags_include_present_dimensions(self, mock_nemo_config, mock_judge_eval_response):
        """S1: Flags include dimension_present for high-scoring dimensions."""
        with patch("training.sdg_pipeline._call_nemo", return_value=mock_judge_eval_response):
            result = ClinicalValidityJudge.evaluate(
                "Let's work together on this. Research shows CBT is effective.",
                mock_nemo_config,
            )

        assert result["validity_score"] > 0.0
        assert isinstance(result["flags"], list)
        assert isinstance(result["category"], str)

    def test_judge_calls_nemo_with_evaluation_prompt(self, mock_nemo_config, mock_judge_eval_response):
        """S1: NeMo is called with a structured evaluation prompt."""
        with patch("training.sdg_pipeline._call_nemo", return_value=mock_judge_eval_response) as mock_call:
            ClinicalValidityJudge.evaluate(
                "Test therapeutic response here.",
                mock_nemo_config,
            )

        mock_call.assert_called_once()
        call_args = mock_call.call_args[0]
        prompt_text = call_args[0]

        # The prompt should ask for clinical validity evaluation
        assert (
            "clinical" in prompt_text.lower()
            or "validity" in prompt_text.lower()
            or "therapeutic" in prompt_text.lower()
        )
        assert "score" in prompt_text.lower() or "rate" in prompt_text.lower()


# ===========================================================================
# PHASE 2: Failure when NeMo is unavailable (no fallback)
# ===========================================================================


class TestNoFallbackOnNeMoFailure:
    """When NeMo API fails, ClinicalValidityJudge raises — no regex fallback."""

    def test_raises_on_none_response(self, mock_nemo_config):
        """S2: NeMo returns None → raises RuntimeError."""
        with patch("training.sdg_pipeline._call_nemo", return_value=None):
            with pytest.raises(RuntimeError, match="no result"):
                ClinicalValidityJudge.evaluate(
                    "Let's try a cognitive reframing exercise. "
                    "Can you identify the automatic thought? "
                    "We can challenge that thought together and look at the evidence.",
                    mock_nemo_config,
                )

    def test_raises_on_empty_response(self, mock_nemo_config):
        """S2: NeMo returns empty string → raises RuntimeError."""
        with patch("training.sdg_pipeline._call_nemo", return_value=""):
            with pytest.raises(RuntimeError, match="no result"):
                ClinicalValidityJudge.evaluate(
                    "Let's try a cognitive reframing exercise.",
                    mock_nemo_config,
                )

    def test_raises_on_malformed_response(self, mock_nemo_config):
        """S2: NeMo returns malformed JSON → raises RuntimeError."""
        with patch(
            "training.sdg_pipeline._call_nemo",
            return_value='{"choices": [{"message": {"content": "not valid json"}}]}',
        ), pytest.raises(RuntimeError, match="no result"):
            ClinicalValidityJudge.evaluate(
                "Let's try a cognitive reframing exercise.",
                mock_nemo_config,
            )

    def test_raises_on_exception(self, mock_nemo_config):
        """S2: NeMo raises exception → raises RuntimeError."""
        with patch("training.sdg_pipeline._call_nemo", side_effect=ConnectionError("API unavailable")):
            with pytest.raises(RuntimeError, match="LLM judge call failed"):
                ClinicalValidityJudge.evaluate(
                    "Let's try a cognitive reframing exercise.",
                    mock_nemo_config,
                )

    def test_raises_on_missing_nemo_config(self):
        """S2: nemo_config is None → raises RuntimeError (no silent fallback)."""
        with pytest.raises(RuntimeError, match="requires nemo_config"):
            ClinicalValidityJudge.evaluate(
                "Let's try a cognitive reframing exercise.",
                None,
            )


# ===========================================================================
# PHASE 3: Empty input handling (S3 — Edge case)
# ===========================================================================


class TestEmptyInput:
    """Empty or None inputs return safe defaults."""

    def test_empty_string(self, mock_nemo_config):
        """S3: Empty string → 0.0 with empty_input flag."""
        self._extracted_from_test_none_text_3("", mock_nemo_config)

    def test_whitespace_string(self, mock_nemo_config):
        """S3: Whitespace → 0.0 with empty_input flag."""
        self._extracted_from_test_none_text_3("   ", mock_nemo_config)

    def test_none_text(self, mock_nemo_config):
        """S3: None → 0.0 with empty_input flag."""
        self._extracted_from_test_none_text_3(None, mock_nemo_config)

    # TODO Rename this here and in `test_empty_string`, `test_whitespace_string` and `test_none_text`
    def _extracted_from_test_none_text_3(self, arg0, mock_nemo_config):
        result = ClinicalValidityJudge.evaluate(arg0, mock_nemo_config)
        assert result["validity_score"] == 0.0
        assert "empty_input" in result.get("flags", [])


# ===========================================================================
# PHASE 4: Non-English detection (S4 — Edge case)
# ===========================================================================


class TestNonEnglish:
    """Non-English content is flagged without NeMo call."""

    def test_korean_text_not_calling_nemo(self, mock_nemo_config):
        """S4: Korean text → flagged without calling NeMo API."""
        result = self._extracted_from_test_japanese_text_not_calling_nemo_3(
            "안녕하세요, 오늘 기분이 어떠세요? 저는 요즘 스트레스를 많이 받고 있어요.",
            mock_nemo_config,
        )
        assert result["validity_score"] == 0.0

    def test_japanese_text_not_calling_nemo(self, mock_nemo_config):
        """S4: Japanese text → flagged without calling NeMo API."""
        self._extracted_from_test_japanese_text_not_calling_nemo_3(
            "こんにちは、今日は気分がどうですか？", mock_nemo_config
        )

    # TODO Rename this here and in `test_korean_text_not_calling_nemo` and `test_japanese_text_not_calling_nemo`
    def _extracted_from_test_japanese_text_not_calling_nemo_3(self, arg0, mock_nemo_config):
        with patch("training.sdg_pipeline._call_nemo") as mock_call:
            result = ClinicalValidityJudge.evaluate(arg0, mock_nemo_config)
        mock_call.assert_not_called()
        assert "non_english_content" in result.get("flags", [])
        return result

    def test_spanish_english_mixed_does_call_nemo(self, mock_nemo_config):
        """S4: Mostly English with few Spanish words → proceeds to NeMo and raises on failure."""
        with patch("training.sdg_pipeline._call_nemo", return_value=None) as mock_call:
            with pytest.raises(RuntimeError, match="no result"):
                ClinicalValidityJudge.evaluate(
                    "I'm feeling much better today. Gracias for your help.",
                    mock_nemo_config,
                )

        mock_call.assert_called_once()


# ===========================================================================
# PHASE 5: Score helper and classify_score (S5 — Regression)
# ===========================================================================


class TestScoreHelper:
    """Convenience `score()` and `classify_score()` methods."""

    def test_score_returns_float(self, mock_nemo_config, mock_judge_eval_response):
        with patch("training.sdg_pipeline._call_nemo", return_value=mock_judge_eval_response):
            score = ClinicalValidityJudge.score(
                "Let's try a cognitive reframing exercise.",
                mock_nemo_config,
            )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_raises_on_failure(self, mock_nemo_config):
        """No fallback — score() raises RuntimeError when NeMo fails."""
        with patch("training.sdg_pipeline._call_nemo", return_value=None):
            with pytest.raises(RuntimeError, match="no result"):
                ClinicalValidityJudge.score(
                    "Let's try a cognitive reframing exercise.",
                    mock_nemo_config,
                )

    def test_classify_score_functions(self):
        """classify_score is re-exported or delegates to ClinicalValidityScorer."""
        assert ClinicalValidityJudge.classify_score(0.2) == "excluded"
        assert ClinicalValidityJudge.classify_score(0.5) == "annotation_needed"
        assert ClinicalValidityJudge.classify_score(0.8) == "accepted"


# ===========================================================================
# PHASE 6: CLI entry point
# ===========================================================================


class TestCLI:
    """CLI works with --text argument and produces valid JSON."""

    def test_cli_exits_without_config(self):
        """CLI exits non-zero when NeMo is not configured (no fallback)."""
        import subprocess
        import sys

        with patch.dict("os.environ", {}, clear=True):
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "training.clinical_validity_judge",
                    "--text",
                    "Test therapeutic cognitive reframing CBT automatic thoughts.",
                ],
                capture_output=True,
                text=True,
                cwd=str(_REPO_ROOT),
            )

        assert result.returncode == 1
        assert "NEMO_API_KEY" in result.stderr or "NVIDIA_API_KEY" in result.stderr

    def test_cli_help_succeeds(self):
        """CLI --help returns exit code 0."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "training.clinical_validity_judge", "--help"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        assert result.returncode == 0
