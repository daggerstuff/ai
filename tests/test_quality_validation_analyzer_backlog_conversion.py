#!/usr/bin/env python3
"""Tests for connecting validation analysis to backlog conversion."""

import json
from pathlib import Path

from ai.monitoring.linear_backlog_dispatcher import LinearBacklogDispatcher
from ai.monitoring.quality_validation_analyzer import QualityValidationAnalyzer, ValidationAnalysis


def test_quality_validation_to_backlog_changes():
    analyzer = QualityValidationAnalyzer()
    analyses = {
        "therapeutic_accuracy": ValidationAnalysis(
            metric="therapeutic_accuracy",
            total_validations=100,
            passed_validations=72,
            failed_validations=28,
            pass_rate=72.0,
            average_score=0.68,
            score_distribution={"excellent": 40, "good": 30, "fair": 20, "poor": 10},
            failure_patterns=["Low confidence in failed validations"],
            recommendations=["Urgent"],
        ),
        "clinical_compliance": ValidationAnalysis(
            metric="clinical_compliance",
            total_validations=100,
            passed_validations=70,
            failed_validations=30,
            pass_rate=70.0,
            average_score=0.71,
            score_distribution={"excellent": 45, "good": 25, "fair": 20, "poor": 10},
            failure_patterns=["Review adherence to clinical guidelines"],
            recommendations=["Strengthen clinical checks"],
        ),
        "emotional_authenticity": ValidationAnalysis(
            metric="emotional_authenticity",
            total_validations=80,
            passed_validations=74,
            failed_validations=6,
            pass_rate=74.0,
            average_score=0.72,
            score_distribution={"excellent": 20, "good": 20, "fair": 20, "poor": 20},
            failure_patterns=["Tone pattern variance"],
            recommendations=["Improve emotional intelligence"],
        ),
        "safety_score": ValidationAnalysis(
            metric="safety_score",
            total_validations=200,
            passed_validations=89,
            failed_validations=11,
            pass_rate=89.0,
            average_score=0.74,
            score_distribution={"excellent": 70, "good": 50, "fair": 50, "poor": 30},
            failure_patterns=["Crisis coverage gaps"],
            recommendations=["Critical response"],
        ),
    }

    result = analyzer.convert_analysis_to_backlog_actions(analyses)

    titles = [change.title for change in result.changes]
    assert "Prioritize clinical conversation sources" in titles
    assert "Shift review attention to clinical compliance failures" in titles
    assert "Shift dataset weighting toward empathetic style coverage" in titles
    assert "Escalate crisis detection and harm review capacity" in titles
    assert result.generated_changes >= 4


def test_quality_validation_report_includes_backlog_conversion(monkeypatch, tmp_path):
    monkeypatch.delenv("LINEAR_API_KEY", raising=False)
    monkeypatch.delenv("LINEAR_API_KEY_TEST", raising=False)
    analyzer = QualityValidationAnalyzer()
    analyzer.output_dir = tmp_path / "validation_out"
    analyzer.output_dir.mkdir(parents=True, exist_ok=True)
    analyzer.linear_artifact_dir = analyzer.output_dir / "linear_backlog_artifacts"
    analyzer.linear_dispatcher = LinearBacklogDispatcher(
        queue_path=str(analyzer.linear_artifact_dir / "linear_backlog_queue.jsonl"),
        linear_api_key_env="LINEAR_API_KEY",
    )

    analyses = {
        "clinical_compliance": ValidationAnalysis(
            metric="clinical_compliance",
            total_validations=40,
            passed_validations=20,
            failed_validations=20,
            pass_rate=50.0,
            average_score=0.42,
            score_distribution={"excellent": 3, "good": 7, "fair": 10, "poor": 20},
            failure_patterns=["Consistency mismatch"],
            recommendations=["Improve clinical checks"],
        )
    }

    visualizations = {}
    report_path = analyzer.export_validation_report(analyses, visualizations)
    report = json.loads((tmp_path / "validation_out" / Path(report_path).name).read_text())

    conversion = report["backlog_conversion"]
    assert conversion["generated_changes"] >= 2
    assert any(item["area"] == "curation_rules" for item in conversion["changes"])
    assert any(item["area"] == "review_focus" for item in conversion["changes"])
    artifact_path = Path(report["linear_backlog_artifact"])
    assert artifact_path.exists()
    artifact_payload = json.loads(artifact_path.read_text())
    assert artifact_payload["source"]["task"] == "PIX-535"
    assert len(artifact_payload["actions"]) >= 2
    dispatch = report["linear_backlog_dispatch"]
    assert dispatch["attempted"] >= 2
    assert dispatch["mode"] == "queue_only"
    assert dispatch["queued"] >= 2
    assert "items" in dispatch
