"""Tests for the mental health evaluation suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    from hypothesis import given, settings, strategies as st
except ImportError:
    given = None
    settings = None
    st = None

from training.mental_health_eval import (
    _compute_metrics,
    _load_dataset,
    build_parser,
    run_eval,
)

EXPECTED_SAMPLE_COUNT = 2
EXPECTED_RESPONSE_LENGTH_MEAN = 4.0


class TestComputeMetrics:
    def test_basic_metrics(self):
        prompts = ["Hello", "How are you"]
        responses = ["Hi there", "I'm doing well"]
        metrics = _compute_metrics(prompts, responses)
        assert "response_length_mean" in metrics
        assert "response_length_p95" in metrics
        assert metrics["response_length_mean"] > 0
        assert metrics["response_length_p95"] > 0

    def test_empty_inputs(self):
        metrics = _compute_metrics([], [])
        assert metrics["response_length_mean"] == 0
        assert metrics["response_length_p95"] == 0


class TestLoadDataset:
    def test_loads_valid_jsonl(self, tmp_path: Path):
        path = tmp_path / "eval.jsonl"
        path.write_text(
            '{"prompt": "Hello", "response": "World"}\n{"prompt": "How", "response": "Good"}\n',
            encoding="utf-8",
        )
        samples = _load_dataset(path)
        assert len(samples) == EXPECTED_SAMPLE_COUNT

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            _load_dataset(tmp_path / "nonexistent.jsonl")

    def test_invalid_json_skipped(self, tmp_path: Path):
        path = tmp_path / "eval.jsonl"
        path.write_text(
            '{"prompt": "Hello", "response": "World"}\nnot json\n{"prompt": "How", "response": "Good"}\n',
            encoding="utf-8",
        )
        samples = _load_dataset(path)
        assert len(samples) == EXPECTED_SAMPLE_COUNT


class TestRunEval:
    def test_eval_report_has_required_fields(self, tmp_path: Path):
        dataset_path = tmp_path / "eval.jsonl"
        dataset_path.write_text(
            '{"prompt": "Hello", "response": "World"}\n{"prompt": "How", "response": "Good"}\n',
            encoding="utf-8",
        )
        report_path = tmp_path / "report.json"
        args = build_parser().parse_args(
            [
                "--dataset",
                str(dataset_path),
                "--output",
                str(report_path),
            ]
        )
        run_eval(args)
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert "metrics" in report
        assert "samples" in report
        assert len(report["samples"]) == EXPECTED_SAMPLE_COUNT

    def test_eval_computes_correct_mean(self, tmp_path: Path):
        dataset_path = tmp_path / "eval.jsonl"
        dataset_path.write_text(
            '{"prompt": "A", "response": "BB"}\n{"prompt": "C", "response": "DDDD"}\n',
            encoding="utf-8",
        )
        report_path = tmp_path / "report.json"
        args = build_parser().parse_args(
            [
                "--dataset",
                str(dataset_path),
                "--output",
                str(report_path),
            ]
        )
        run_eval(args)
        report = json.loads(report_path.read_text())
        # Response lengths: 2 and 4, mean = 3.0
        tolerance = 0.01
        assert abs(report["metrics"]["response_length_mean"] - 3.0) < tolerance
        # P95 of [2, 4] = 4
        expected_p95 = 4
        assert report["metrics"]["response_length_p95"] == expected_p95


# ---------------------------------------------------------------------------
# Hypothesis property tests
# ---------------------------------------------------------------------------

if st is not None:

    @given(
        st.lists(
            st.tuples(
                st.text(min_size=1, max_size=50),
                st.text(min_size=1, max_size=200),
            ),
            min_size=1,
            max_size=100,
        )
    )
    @settings(max_examples=100)
    def test_metrics_non_negative(pairs):
        prompts = [p for p, r in pairs]
        responses = [r for p, r in pairs]
        metrics = _compute_metrics(prompts, responses)
        assert metrics["response_length_mean"] >= 0.0
        assert metrics["response_length_p95"] >= 0.0

    @given(
        st.lists(
            st.tuples(
                st.text(min_size=1, max_size=50),
                st.text(min_size=1, max_size=200),
            ),
            min_size=1,
            max_size=100,
        )
    )
    @settings(max_examples=100)
    def test_p95_at_least_mean(pairs):
        prompts = [p for p, r in pairs]
        responses = [r for p, r in pairs]
        metrics = _compute_metrics(prompts, responses)
        # P95 should always be >= mean
        assert metrics["response_length_p95"] >= metrics["response_length_mean"]

else:

    @pytest.mark.skip(reason="hypothesis not installed")
    def test_hypothesis_not_installed():
        raise AssertionError("Skipped")
