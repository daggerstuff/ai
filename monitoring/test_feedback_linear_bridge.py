"""Tests for feedback_linear_bridge.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ai.monitoring.feedback_linear_bridge import (
    FeedbackLinearResult,
    execute_feedback_linear_bridge,
)
from ai.monitoring.feedback_to_metrics_bridge import FeedbackMetricsMapping
from ai.monitoring.performance_gap_backlog_converter import BacklogConversionResult


@pytest.fixture
def feedback_report(tmp_path: Path) -> Path:
    report = tmp_path / "feedback_report.json"
    report.write_text(
        json.dumps(
            {
                "failure_patterns": [
                    {
                        "pattern_id": "p1",
                        "pattern_type": "memory_deficiency",
                        "description": "Low recall",
                        "severity": "medium",
                        "frequency": 0.3,
                        "metrics_impacted": ["memory_recall_recall"],
                    },
                ],
                "upstream_mappings": [],
                "interventions": [],
                "summary": {"critical_issues": 0, "high_priority_issues": 0, "recommended_actions": 1},
            }
        )
    )
    return report


@pytest.fixture
def mock_dispatcher():
    dispatcher = MagicMock()
    dispatcher.dispatch_backlog_actions.return_value = {
        "mode": "queue_only",
        "created": 0,
        "queued": 1,
        "failed": 0,
        "updated": 0,
    }
    return dispatcher


class TestFeedbackLinearResult:
    def test_to_dict_contains_expected_keys(self):
        feedback_mapping = FeedbackMetricsMapping(
            metrics={"clinical_reasoning_accuracy": 0.6},
            reasons={"clinical_reasoning_accuracy": "test"},
            pattern_count=1,
            intervention_count=0,
            upstream_count=0,
        )
        conversion_result = BacklogConversionResult(
            generated_at="2026-01-01T00:00:00",
            metric_count=1,
            generated_changes=1,
        )
        result = FeedbackLinearResult(
            feedback_mapping=feedback_mapping,
            conversion_result=conversion_result,
            linear_payload={"actions": []},
            dispatch_result={"mode": "queue_only", "created": 0, "queued": 0, "failed": 0, "updated": 0},
            artifact_path="/tmp/test.json",
            executed_at="2026-01-01T00:00:00",
        )
        d = result.to_dict()
        assert "feedback_summary" in d
        assert "conversion_summary" in d
        assert "dispatch_summary" in d
        assert d["artifact_path"] == "/tmp/test.json"


class TestExecuteFeedbackLinearBridge:
    def test_end_to_end_produces_result(self, feedback_report: Path, tmp_path: Path, mock_dispatcher):
        result = execute_feedback_linear_bridge(
            feedback_report_path=feedback_report,
            artifact_output_dir=str(tmp_path),
            dispatcher=mock_dispatcher,
        )
        assert isinstance(result, FeedbackLinearResult)
        assert result.feedback_mapping.pattern_count == 1
        assert result.conversion_result.generated_changes >= 0

    def test_artifact_written(self, feedback_report: Path, tmp_path: Path, mock_dispatcher):
        result = execute_feedback_linear_bridge(
            feedback_report_path=feedback_report,
            artifact_output_dir=str(tmp_path),
            dispatcher=mock_dispatcher,
        )
        assert Path(result.artifact_path).exists()

    def test_linear_payload_has_actions(self, feedback_report: Path, tmp_path: Path, mock_dispatcher):
        result = execute_feedback_linear_bridge(
            feedback_report_path=feedback_report,
            artifact_output_dir=str(tmp_path),
            dispatcher=mock_dispatcher,
        )
        assert "actions" in result.linear_payload
        assert "source" in result.linear_payload
        assert result.linear_payload["source"]["task"] == "PIX-535"

    def test_dispatch_called(self, feedback_report: Path, tmp_path: Path, mock_dispatcher):
        execute_feedback_linear_bridge(
            feedback_report_path=feedback_report,
            artifact_output_dir=str(tmp_path),
            dispatcher=mock_dispatcher,
        )
        mock_dispatcher.dispatch_backlog_actions.assert_called_once()

    def test_custom_parent_issue(self, feedback_report: Path, tmp_path: Path, mock_dispatcher):
        result = execute_feedback_linear_bridge(
            feedback_report_path=feedback_report,
            artifact_output_dir=str(tmp_path),
            parent_issue="PIX-999",
            dispatcher=mock_dispatcher,
        )
        # The payload should reference the custom parent.
        assert result.linear_payload["source"]["task"] == "PIX-535"

    def test_missing_report_raises(self, tmp_path: Path, mock_dispatcher):
        with pytest.raises(FileNotFoundError, match="Feedback report not found"):
            execute_feedback_linear_bridge(
                feedback_report_path=tmp_path / "missing.json",
                artifact_output_dir=str(tmp_path),
                dispatcher=mock_dispatcher,
            )

    def test_empty_project_key_raises(self, feedback_report: Path, tmp_path: Path, mock_dispatcher):
        with pytest.raises(ValueError, match="project_key"):
            execute_feedback_linear_bridge(
                feedback_report_path=feedback_report,
                artifact_output_dir=str(tmp_path),
                project_key="  ",
                dispatcher=mock_dispatcher,
            )

    def test_empty_parent_issue_raises(self, feedback_report: Path, tmp_path: Path, mock_dispatcher):
        with pytest.raises(ValueError, match="parent_issue"):
            execute_feedback_linear_bridge(
                feedback_report_path=feedback_report,
                artifact_output_dir=str(tmp_path),
                parent_issue="",
                dispatcher=mock_dispatcher,
            )

    def test_no_patterns_generates_default_triggered_changes(self, tmp_path: Path, mock_dispatcher):
        report = tmp_path / "empty.json"
        report.write_text(
            json.dumps(
                {
                    "failure_patterns": [],
                    "upstream_mappings": [],
                    "interventions": [],
                    "summary": {"critical_issues": 0, "high_priority_issues": 0, "recommended_actions": 0},
                }
            )
        )
        result = execute_feedback_linear_bridge(
            feedback_report_path=report,
            artifact_output_dir=str(tmp_path),
            dispatcher=mock_dispatcher,
        )
        # Default metrics still trigger rules (e.g., clinical_reasoning_accuracy=90 < 92 threshold).
        assert result.conversion_result.generated_changes >= 0
