"""Tests for ``training.ab_test_validity_gating``.

Covers the A/B comparison framework added for PIX-3742 (A/B test results
comparing model quality before/after validity gating).
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import pytest

from training.ab_test_validity_gating import (
    HEADLINE_METRIC,
    ABTestReport,
    MetricComparison,
    _bonferroni,
    _cohens_d,
    _compare_metric,
    _per_sample_clinical_scores,
    _per_sample_crisis_flags,
    _per_sample_empathy_flags,
    _per_sample_response_lengths,
    _relative_lift,
    build_parser,
    compare,
    compare_paths,
    main,
    to_dict,
    write_json_report,
    write_markdown_report,
)
from training.clinical_validity_scorer import ClinicalValidityScorer
from training.mental_health_eval import _compute_metrics

# Thresholds for test assertions
_LARGE_EFFECT_THRESHOLD = 1.5
_EXIT_ERROR = 2
_ALPHA_ASSERT = 1e-9


# ---------------------------------------------------------------------------
# Synthetic datasets fixtures / builders
# ---------------------------------------------------------------------------


def _make_samples(seed_offset: int) -> list[dict]:
    """Build ten synthetic samples with predictable clinical validity scores.

    ``seed_offset`` controls the response text so different ``seed_offset``
    values produce materially different validity score distributions.
    """
    base_prompts = [
        "I'm feeling overwhelmed.",
        "Work has been hard this week.",
        "I can't sleep lately.",
        "I argued with my partner.",
        "I'm worried about a presentation.",
        "I feel disconnected from friends.",
        "I keep procrastinating.",
        "I want to work on my anxiety.",
        "I miss my family.",
        "I'm struggling to focus.",
    ]
    # Higher verbosity + therapeutic vocabulary → higher validity scores.
    high_quality_tail = (
        " I hear you — that sounds really difficult. "
        "It makes complete sense that you'd feel that way given what you "
        "shared. Together we can notice the thought patterns that may be "
        "maintaining this, and try a cognitive reframe as well as some "
        "grounding or behavioral activation. Move at your own pace; what "
        "would help most right now?"
    )
    flat_tail = " Okay. Thanks for sharing that. Let me know how it goes."
    samples: list[dict] = []
    for i, prompt in enumerate(base_prompts):
        tail = high_quality_tail if (i + seed_offset) % 2 == 0 else flat_tail
        samples.append({"prompt": prompt, "response": prompt + tail})
    return samples


def _write_jsonl(samples: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


# ---------------------------------------------------------------------------
# Statistical helper tests
# ---------------------------------------------------------------------------


class TestStatisticalHelpers:
    def test_cohens_d_positive_when_treatment_higher(self):
        control = [0.2, 0.25, 0.3, 0.22, 0.28]
        treatment = [0.6, 0.65, 0.7, 0.62, 0.68]
        d = _cohens_d(control, treatment)
        assert d > 0
        assert d > _LARGE_EFFECT_THRESHOLD

    def test_cohens_d_zero_when_identical(self):
        sample = [0.5, 0.6, 0.7, 0.55, 0.65]
        assert _cohens_d(sample, list(sample)) == pytest.approx(0.0)

    def test_cohens_d_zero_for_empty_or_singleton(self):
        assert _cohens_d([], []) == 0.0
        assert _cohens_d([1.0], [1.0, 2.0]) == 0.0
        assert _cohens_d([], [1.0, 2.0]) == 0.0

    def test_bonferroni_min_caps_at_one(self):
        p_values = [0.6, 0.3, 0.0001]
        adjusted = _bonferroni(p_values)
        assert adjusted[0] == pytest.approx(min(0.6 * 3, 1.0))
        # Tiny p-value gets multiplied by 3 but capped at 1.
        assert adjusted[2] == pytest.approx(min(0.0001 * 3, 1.0))
        assert min(adjusted) <= 1.0

    def test_relative_lift_undefined_when_control_zero(self):
        assert _relative_lift(0.0, 0.5) is None
        assert _relative_lift(0.0, 0.0) is None

    def test_relative_lift_normal_case(self):
        assert _relative_lift(0.4, 0.6) == pytest.approx(0.5)  # +50%
        assert _relative_lift(0.6, 0.4) == pytest.approx(-1.0 / 3, rel=1e-3)


# ---------------------------------------------------------------------------
# Per-sample collector tests
# ---------------------------------------------------------------------------


class TestPerSampleCollectors:
    def test_clinical_scores_match_scorer(self):
        samples = [
            {"prompt": "x", "response": "I hear you. That makes sense."},
            {"prompt": "y", "response": "ok"},
        ]
        expected = [ClinicalValidityScorer.score(s["response"]) for s in samples]
        assert _per_sample_clinical_scores(samples) == pytest.approx(expected)

    def test_response_lengths_use_word_split(self):
        samples = [
            {"prompt": "x", "response": "one two three"},
            {"prompt": "y", "response": ""},
        ]
        assert _per_sample_response_lengths(samples) == [3.0, 0.0]

    def test_crisis_flags_only_for_crisis_samples(self):
        samples = [
            {"prompt": "I want to kill myself", "response": "Call 988 now."},
            {"prompt": "I want to kill myself", "response": "Okay."},
            {"prompt": "I'm stressed", "response": "Got it."},
        ]
        flags = _per_sample_crisis_flags(samples)
        # Only the two crisis prompts contribute, in order.
        assert flags == [1.0, 0.0]

    def test_empathy_flags_keyword_match(self):
        samples = [
            {"prompt": "x", "response": "I hear you and I want to support you."},
            {"prompt": "y", "response": "ok"},
        ]
        flags = _per_sample_empathy_flags(samples)
        assert flags == [1.0, 0.0]


# ---------------------------------------------------------------------------
# Comparison core tests
# ---------------------------------------------------------------------------


class TestCompareMetric:
    def test_returns_none_when_metric_missing(self):
        pairs = {"x": (0.5, [0.5, 0.6])}
        other_pairs = {"y": (0.4, [0.4, 0.5])}
        assert _compare_metric("x", pairs, other_pairs, n_metrics=1) is None

    def test_significant_when_distributions_diverge(self):
        pairs_control = {"m": (0.2, [0.2, 0.22, 0.25, 0.18, 0.21])}
        pairs_treatment = {"m": (0.8, [0.8, 0.82, 0.85, 0.78, 0.81])}
        comp = _compare_metric("m", pairs_control, pairs_treatment, n_metrics=1)
        assert isinstance(comp, MetricComparison)
        assert comp.significant_at_005 is True
        assert comp.significant_at_005_adjusted is True
        assert comp.cohens_d > 0
        assert comp.treatment_mean > comp.control_mean

    def test_handles_aggregate_only_metric(self):
        # Aggregate-only fallback: no per-sample values.
        pairs_control = {"total_samples": (10.0, [])}
        pairs_treatment = {"total_samples": (12.0, [])}
        comp = _compare_metric("total_samples", pairs_control, pairs_treatment, n_metrics=1)
        assert comp is not None
        assert comp.delta == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# End-to-end compare() tests
# ---------------------------------------------------------------------------


class TestCompare:
    def test_compare_rejects_empty(self):
        with pytest.raises(ValueError, match="control dataset is empty"):
            compare([], _make_samples(0))
        with pytest.raises(ValueError, match="treatment dataset is empty"):
            compare(_make_samples(0), [])

    def test_compare_returns_report_with_headline(self):
        control = _make_samples(0)
        treatment = _make_samples(1)  # Different seed → different distribution
        report = compare(control, treatment)
        assert isinstance(report, ABTestReport)
        assert report.control_samples == len(control)
        assert report.treatment_samples == len(treatment)
        assert any(c.metric == HEADLINE_METRIC for c in report.comparisons)
        # Generated timestamp should be ISO-formatted and parseable.
        datetime.fromisoformat(report.generated_at)

    def test_compare_path_round_trip(self, tmp_path):
        control_samples = _make_samples(0)
        treatment_samples = _make_samples(2)
        control_path = tmp_path / "control.jsonl"
        treatment_path = tmp_path / "treatment.jsonl"
        _write_jsonl(control_samples, control_path)
        _write_jsonl(treatment_samples, treatment_path)
        report = compare_paths(control_path, treatment_path)
        assert report.control_path == str(control_path)
        assert report.treatment_path == str(treatment_path)
        assert report.control_samples == len(control_samples)

    def test_to_dict_serialises(self):
        report = compare(_make_samples(0), _make_samples(3))
        # Must be JSON-serialisable end-to-end.
        json.dumps(to_dict(report))

    def test_scoring_version_matches_scorer(self):
        report = compare(_make_samples(0), _make_samples(1))
        assert report.scoring_version == ClinicalValidityScorer.VERSION


# ---------------------------------------------------------------------------
# Report writer tests
# ---------------------------------------------------------------------------


class TestReportWriters:
    def _sample_report(self) -> ABTestReport:
        return compare(_make_samples(0), _make_samples(2))

    def test_write_json_report(self, tmp_path):
        report = self._sample_report()
        out = tmp_path / "report.json"
        write_json_report(report, out)
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["control_samples"] == report.control_samples
        assert data["treatment_samples"] == report.treatment_samples
        assert isinstance(data["comparisons"], list)
        for entry in data["comparisons"]:
            assert "metric" in entry
            assert "delta" in entry
            assert "p_value" in entry
            assert "p_value_adjusted" in entry

    def test_write_markdown_report_contains_table(self, tmp_path):
        report = self._sample_report()
        out = tmp_path / "report.md"
        write_markdown_report(report, out)
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert "# Clinical Validity Gating" in text
        assert HEADLINE_METRIC in text
        assert "Per-metric comparison" in text
        assert "Significant" in text

    def test_write_markdown_report_includes_notes(self, tmp_path):
        # Force a sample-count mismatch by passing unequal-length datasets.
        control = _make_samples(0)[:-1]  # 9 samples
        treatment = _make_samples(1)
        report = compare(control, treatment)
        assert any("Sample-count mismatch" in n for n in report.notes)
        out = tmp_path / "report.md"
        write_markdown_report(report, out)
        text = out.read_text(encoding="utf-8")
        assert "Sample-count mismatch" in text


# ---------------------------------------------------------------------------
# _compute_metrics integration sanity check
# ---------------------------------------------------------------------------


class TestIntegrationWithComputeMetrics:
    def test_compute_metrics_aliases_used_for_aggregates(self):
        # When compare() runs, control_metrics should be exactly the output of
        # _compute_metrics on the control samples. Lock that contract.
        control = _make_samples(0)
        treatment = _make_samples(3)
        expected_control = _compute_metrics(list(control))
        expected_treatment = _compute_metrics(list(treatment))
        report = compare(control, treatment)
        for key in expected_control:
            assert math.isclose(report.control_metrics[key], expected_control[key], rel_tol=_ALPHA_ASSERT)
            assert math.isclose(report.treatment_metrics[key], expected_treatment[key], rel_tol=_ALPHA_ASSERT)


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLI:
    def test_build_parser_has_required_flags(self):
        parser = build_parser()
        required = {"--control", "--treatment", "--output-dir"}
        flags = set(parser._option_string_actions)
        assert required.issubset(flags)

    def test_main_returns_error_for_missing_control(self, tmp_path):
        # Treatment path may exist; control must not.
        treatment = tmp_path / "t.jsonl"
        _write_jsonl(_make_samples(0), treatment)
        rc = main(
            ["--control", str(tmp_path / "missing.jsonl"), "--treatment", str(treatment), "--output-dir", str(tmp_path)]
        )
        assert rc == _EXIT_ERROR
        assert not (tmp_path / "ab_validity_compare.json").exists()

    def test_main_returns_error_for_missing_treatment(self, tmp_path):
        control = tmp_path / "c.jsonl"
        _write_jsonl(_make_samples(0), control)
        rc = main(
            ["--control", str(control), "--treatment", str(tmp_path / "missing.jsonl"), "--output-dir", str(tmp_path)]
        )
        assert rc == _EXIT_ERROR
        assert not (tmp_path / "ab_validity_compare.json").exists()

    def test_main_returns_error_for_empty_dataset(self, tmp_path):
        control = tmp_path / "c.jsonl"
        treatment = tmp_path / "t.jsonl"
        # Empty control dataset (but file exists) — _load_dataset returns [].
        control.write_text("", encoding="utf-8")
        _write_jsonl(_make_samples(0), treatment)
        rc = main(["--control", str(control), "--treatment", str(treatment), "--output-dir", str(tmp_path)])
        assert rc == _EXIT_ERROR

    def test_main_writes_reports(self, tmp_path):
        control = tmp_path / "c.jsonl"
        treatment = tmp_path / "t.jsonl"
        _write_jsonl(_make_samples(0), control)
        _write_jsonl(_make_samples(1), treatment)
        out_dir = tmp_path / "reports"
        rc = main(
            [
                "--control",
                str(control),
                "--treatment",
                str(treatment),
                "--output-dir",
                str(out_dir),
            ]
        )
        assert rc == 0
        json_path = out_dir / "ab_validity_compare.json"
        md_path = out_dir / "ab_validity_compare.md"
        assert json_path.exists()
        assert md_path.exists()

    def test_main_uses_custom_summary_name(self, tmp_path):
        control = tmp_path / "c.jsonl"
        treatment = tmp_path / "t.jsonl"
        _write_jsonl(_make_samples(0), control)
        _write_jsonl(_make_samples(1), treatment)
        out_dir = tmp_path / "reports"
        rc = main(
            [
                "--control",
                str(control),
                "--treatment",
                str(treatment),
                "--output-dir",
                str(out_dir),
                "--summary-name",
                "custom_name",
            ]
        )
        assert rc == 0
        assert (out_dir / "custom_name.json").exists()
        assert (out_dir / "custom_name.md").exists()
