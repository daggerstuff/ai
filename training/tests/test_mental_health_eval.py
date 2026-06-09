"""Tests for the mental health evaluation suite (Prop 13)."""
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
from training.mental_health_eval import (_compute_metrics, _load_dataset, build_parser, run_eval)
EXPECTED_SAMPLE_COUNT = 2


class TestComputeMetrics:
    def test_basic_metrics(self):
        samples = [
            {"prompt": "Hello", "response": "I hear you."},
            {"prompt": "Sad", "response": "Here for you."},
        ]
        m = _compute_metrics(samples)
        assert m["response_length_mean"] > 0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Empty dataset"):
            _compute_metrics([])


class TestLoadDataset:
    def test_valid(self, tmp_path: Path):
        p = tmp_path / "x.jsonl"
        p.write_text(
            '{"prompt": "a", "response": "b"}\n{"prompt": "c", "response": "d"}\n',
            encoding="utf-8",
        )
        assert len(_load_dataset(p)) == EXPECTED_SAMPLE_COUNT

    def test_missing_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            _load_dataset(tmp_path / "nope.jsonl")

    def test_skip_bad(self, tmp_path: Path):
        p = tmp_path / "x.jsonl"
        p.write_text(
            '{"prompt": "a", "response": "b"}\nnot json\n{"prompt": "c", "response": "d"}\n',
            encoding="utf-8",
        )
        assert len(_load_dataset(p)) == EXPECTED_SAMPLE_COUNT


class TestRunEval:
    def test_report_has_metrics(self, tmp_path: Path):
        p = tmp_path / "e.jsonl"
        p.write_text(
            '{"prompt": "hi", "response": "world"}\n', encoding="utf-8"
        )
        out = tmp_path / "out"
        out.mkdir()
        args = build_parser().parse_args(
            [
                "--eval_dataset", str(p),
                "--checkpoint", "ckpt1",
                "--output_dir", str(out),
            ]
        )
        run_eval(args)
        r = json.loads((out / "eval_report.json").read_text())
        assert "metrics" in r
        assert r["checkpoint"] == "ckpt1"


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
    @settings(max_examples=20)
    def test_metrics_non_negative(pairs):
        samples = [{"prompt": p, "response": r} for p, r in pairs]
        m = _compute_metrics(samples)
        assert m["response_length_mean"] >= 0
        assert m["response_length_p95"] >= 0

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
    @settings(max_examples=20)
    def test_p95_at_least_mean(pairs):
        samples = [{"prompt": p, "response": r} for p, r in pairs]
        m = _compute_metrics(samples)
        assert m["response_length_p95"] >= m["response_length_mean"]

else:
    @pytest.mark.skip(reason="hypothesis not installed")
    def test_hypothesis_not_installed():
        raise AssertionError("Skipped")
