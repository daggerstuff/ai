"""Tests for feedback_to_metrics_bridge.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai.qa.reports.feedback_to_metrics_bridge import (
    DEFAULT_METRIC_SCORES,
    PATTERN_TYPE_TO_METRIC,
    SEVERITY_PENALTY,
    FeedbackMetricsMapping,
    _compute_metric_scores,
    _enrich_with_interventions,
    _enrich_with_upstream,
    _finalize_reasons,
    _load_feedback_report,
    transform_feedback_to_metrics,
)


@pytest.fixture
def minimal_feedback_report(tmp_path: Path) -> Path:
    report = tmp_path / "feedback_report.json"
    report.write_text(
        json.dumps(
            {
                "failure_patterns": [
                    {
                        "pattern_id": "pattern_memory_recall_low",
                        "pattern_type": "memory_deficiency",
                        "description": "Model fails to recall relevant memories",
                        "severity": "medium",
                        "frequency": 0.3,
                        "metrics_impacted": ["memory_recall_recall"],
                    },
                    {
                        "pattern_id": "pattern_context_drift",
                        "pattern_type": "context_alignment",
                        "description": "Responses drift from context",
                        "severity": "high",
                        "frequency": 0.15,
                        "metrics_impacted": ["context_relevance"],
                    },
                ],
                "upstream_mappings": [
                    {
                        "failure_pattern": {
                            "pattern_id": "pattern_memory_recall_low",
                            "pattern_type": "memory_deficiency",
                        },
                        "upstream_domain": "acquisition",
                        "confidence": 0.66,
                        "root_cause_hypothesis": "Source data lacks quality pairs",
                        "evidence": ["acquisition_logs"],
                    },
                ],
                "interventions": [
                    {
                        "intervention_id": "intervention_1",
                        "intervention_type": "priority_change",
                        "title": "Adjust source priorities",
                        "upstream_domain": "acquisition",
                        "priority": "medium",
                        "expected_impact": "Improve recall by 10-20%",
                        "related_patterns": ["pattern_memory_recall_low"],
                    },
                ],
                "summary": {"critical_issues": 0, "high_priority_issues": 0, "recommended_actions": 1},
            }
        )
    )
    return report


class TestTransformFeedbackToMetrics:
    def test_transform_returns_mapping(self, minimal_feedback_report: Path):
        result = transform_feedback_to_metrics(minimal_feedback_report)
        assert isinstance(result, FeedbackMetricsMapping)
        assert result.pattern_count == 2
        assert result.intervention_count == 1
        assert result.upstream_count == 1

    def test_transform_metrics_contain_expected_keys(self, minimal_feedback_report: Path):
        result = transform_feedback_to_metrics(minimal_feedback_report)
        # memory_deficiency → clinical_reasoning_accuracy
        assert "clinical_reasoning_accuracy" in result.metrics
        # context_alignment → empathy_score
        assert "empathy_score" in result.metrics

    def test_transform_scores_penalized_by_frequency(self, minimal_feedback_report: Path):
        result = transform_feedback_to_metrics(minimal_feedback_report)
        # memory_deficiency: baseline=0.90, freq=0.3, severity=medium(1.0)
        # score = 0.90 - 0.3*1.0 = 0.60
        assert result.metrics["clinical_reasoning_accuracy"] == pytest.approx(0.60, abs=0.01)

    def test_transform_high_severity_penalty(self, minimal_feedback_report: Path):
        result = transform_feedback_to_metrics(minimal_feedback_report)
        # context_alignment: baseline=0.85, freq=0.15, severity=high(1.5)
        # score = 0.85 - 0.15*1.5 = 0.85 - 0.225 = 0.625
        assert result.metrics["empathy_score"] == pytest.approx(0.625, abs=0.01)

    def test_transform_reasons_contain_upstream(self, minimal_feedback_report: Path):
        result = transform_feedback_to_metrics(minimal_feedback_report)
        # clinical_reasoning_accuracy should have upstream reason.
        assert "clinical_reasoning_accuracy" in result.reasons
        assert "upstream(acquisition" in result.reasons["clinical_reasoning_accuracy"]

    def test_transform_to_dict(self, minimal_feedback_report: Path):
        result = transform_feedback_to_metrics(minimal_feedback_report)
        d = result.to_dict()
        assert d["metrics"] == result.metrics
        assert d["reasons"] == result.reasons
        assert d["pattern_count"] == 2


class TestComputeMetricScores:
    def test_unknown_pattern_type_ignored(self):
        patterns = [
            {"pattern_type": "unknown_type", "frequency": 0.5, "severity": "high", "description": ""},
        ]
        metrics, _reasons = _compute_metric_scores(patterns)
        # Should remain at defaults.
        assert metrics["clinical_reasoning_accuracy"] == pytest.approx(0.90, abs=0.01)

    def test_validation_gap_accumulates(self):
        patterns = [
            {"pattern_type": "generation_quality", "frequency": 0.3, "severity": "low", "description": ""},
            {"pattern_type": "generation_quality", "frequency": 0.2, "severity": "low", "description": ""},
        ]
        metrics, _ = _compute_metric_scores(patterns)
        # generation_quality → validation_gap, accumulates: 10.0 + 0.3*100 + 0.2*100 = 60.0
        assert metrics["validation_gap"] == pytest.approx(60.0, abs=0.01)

    def test_score_clamped_to_zero(self):
        patterns = [
            {"pattern_type": "memory_deficiency", "frequency": 1.0, "severity": "critical", "description": ""},
        ]
        metrics, _ = _compute_metric_scores(patterns)
        # baseline=0.90, penalty=1.0*2.0=2.0 → max(0.0, 0.90-2.0) = 0.0
        assert metrics["clinical_reasoning_accuracy"] == 0.0


class TestEnrichWithUpstream:
    def test_upstream_appends_to_reasons(self):
        reasons = {"clinical_reasoning_accuracy": ["existing reason"]}
        upstream = [
            {
                "failure_pattern": {"pattern_type": "memory_deficiency"},
                "upstream_domain": "acquisition",
                "confidence": 0.7,
                "root_cause_hypothesis": "Poor source data",
            },
        ]
        _enrich_with_upstream(upstream, reasons)
        assert len(reasons["clinical_reasoning_accuracy"]) == 2
        assert "upstream(acquisition" in reasons["clinical_reasoning_accuracy"][1]

    def test_upstream_unknown_pattern_ignored(self):
        reasons: dict[str, list[str]] = {}
        upstream = [
            {
                "failure_pattern": {"pattern_type": "nonexistent"},
                "upstream_domain": "curation",
                "confidence": 0.5,
                "root_cause_hypothesis": "test",
            },
        ]
        _enrich_with_upstream(upstream, reasons)
        assert reasons == {}


class TestEnrichWithInterventions:
    def test_intervention_appends_to_domain_metrics(self):
        reasons: dict[str, list[str]] = {"clinical_reasoning_accuracy": [], "validation_gap": []}
        interventions = [
            {
                "title": "Adjust priorities",
                "priority": "medium",
                "upstream_domain": "acquisition",
                "expected_impact": "Improve by 10%",
                "related_patterns": ["pattern_x"],
            },
        ]
        _enrich_with_interventions(interventions, reasons)
        assert any("intervention[acquisition]" in r for r in reasons["clinical_reasoning_accuracy"])

    def test_intervention_unknown_domain_ignored(self):
        reasons: dict[str, list[str]] = {}
        interventions = [
            {
                "title": "Test",
                "priority": "low",
                "upstream_domain": "unknown_domain",
                "expected_impact": "",
                "related_patterns": [],
            },
        ]
        _enrich_with_interventions(interventions, reasons)
        assert reasons == {}


class TestFinalizeReasons:
    def test_collapses_lists_to_strings(self):
        reasons = {"metric_a": ["reason1", "reason2"], "metric_b": []}
        result = _finalize_reasons(reasons)
        assert result == {"metric_a": "reason1; reason2"}
        assert "metric_b" not in result

    def test_empty_reasons_excluded(self):
        reasons = {"metric_a": [], "metric_b": ["only"]}
        result = _finalize_reasons(reasons)
        assert result == {"metric_b": "only"}


class TestLoadFeedbackReport:
    def test_missing_file_raises_error(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            _load_feedback_report(tmp_path / "nonexistent.json")

    def test_missing_keys_raises_error(self, tmp_path: Path):
        report = tmp_path / "bad.json"
        report.write_text(json.dumps({"only_one_key": True}))
        with pytest.raises(ValueError, match="missing required keys"):
            _load_feedback_report(report)

    def test_valid_report_returns_data(self, tmp_path: Path):
        report = tmp_path / "valid.json"
        report.write_text(
            json.dumps(
                {
                    "failure_patterns": [],
                    "interventions": [],
                    "upstream_mappings": [],
                }
            )
        )
        data = _load_feedback_report(report)
        assert data["failure_patterns"] == []


class TestConstants:
    def test_pattern_type_mapping_covers_all_severities(self):
        for pattern_type in PATTERN_TYPE_TO_METRIC:
            assert pattern_type in PATTERN_TYPE_TO_METRIC

    def test_severity_penalty_values(self):
        assert SEVERITY_PENALTY["critical"] == 2.0
        assert SEVERITY_PENALTY["high"] == 1.5
        assert SEVERITY_PENALTY["medium"] == 1.0
        assert SEVERITY_PENALTY["low"] == 0.5

    def test_default_metric_scores_present(self):
        assert "clinical_reasoning_accuracy" in DEFAULT_METRIC_SCORES
        assert "safety_score" in DEFAULT_METRIC_SCORES
        assert "validation_gap" in DEFAULT_METRIC_SCORES
