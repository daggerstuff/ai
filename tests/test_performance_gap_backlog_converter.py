#!/usr/bin/env python3
"""Tests for performance-gap-to-backlog rule conversion."""

from ai.monitoring.performance_gap_backlog_converter import (
    PerformanceGapBacklogConverter,
    RulePriority,
)


def test_converts_critical_clinical_reasoning_gap():
    converter = PerformanceGapBacklogConverter()
    result = converter.convert(
        {
            "clinical_reasoning_accuracy": 72.0,
            "clinical_compliance": 72.0,
            "safety_score": 88.0,
        }
    )

    titles = {change.title for change in result.changes}
    assert "Prioritize clinical conversation sources" in titles
    assert "Shift review attention to clinical compliance failures" in titles
    assert "Escalate crisis detection and harm review capacity" in titles

    priorities = [change.priority for change in result.changes]
    assert RulePriority.CRITICAL in priorities


def test_medium_gap_uses_empathy_alias():
    converter = PerformanceGapBacklogConverter()
    result = converter.convert_from_validation_analysis(
        {
            "emotional_authenticity": {"pass_rate": 74.0},
            "therapeutic_accuracy": {"pass_rate": 70.0},
        }
    )

    aliases = [change.summary for change in result.changes]
    assert any("Empathy score below threshold" in text for text in aliases)

    # Clinical reasoning should be mapped from therapeutic_accuracy in this payload.
    assert any("clinical_reasoning_accuracy" in change.evidence["metric"] for change in result.changes)


def test_no_changes_when_above_threshold():
    """All metrics above thresholds → no backlog changes generated."""
    converter = PerformanceGapBacklogConverter()
    result = converter.convert(
        {
            "clinical_reasoning_accuracy": 95.0,
            "clinical_compliance": 90.0,
            "safety_score": 98.0,
            "empathy_score": 90.0,
            "validation_gap": 10.0,
        }
    )

    assert result.generated_changes == 0
    assert result.changes == []


def test_validation_gap_triggers_high_priority():
    """validation_gap > 30 triggers HIGH priority pipeline-allocation change."""
    converter = PerformanceGapBacklogConverter()
    result = converter.convert({"validation_gap": 45.0})

    assert result.generated_changes == 1
    change = result.changes[0]
    assert change.priority == RulePriority.HIGH
    assert change.area == "pipeline_allocation"
    # Title references quality checks / pipeline cycles / bottleneck
    assert any(kw in change.title.lower() for kw in ["pipeline", "bottleneck", "quality", "validation"])


def test_clinical_reasoning_high_priority_rule():
    """Clinical reasoning between 85-92% triggers HIGH (not CRITICAL) priority."""
    converter = PerformanceGapBacklogConverter()
    result = converter.convert({"clinical_reasoning_accuracy": 88.0})

    titles = {change.title for change in result.changes}
    # 88% is above CRITICAL threshold (85%) but below HIGH threshold (92%)
    assert "Prioritize clinical conversation sources" not in titles
    assert "Tighten curation filters for clinical rationale" in titles
    priorities = {change.priority for change in result.changes}
    assert RulePriority.HIGH in priorities
    assert RulePriority.CRITICAL not in priorities


def test_safety_score_triggers_critical():
    """safety_score < 90 triggers CRITICAL priority review-focus change."""
    converter = PerformanceGapBacklogConverter()
    result = converter.convert({"safety_score": 85.0})

    assert result.generated_changes == 1
    change = result.changes[0]
    assert change.priority == RulePriority.CRITICAL
    assert change.area == "review_focus"
    assert "crisis detection" in change.title.lower()


def test_multiple_rules_fire_simultaneously():
    """Multiple metric gaps produce multiple distinct backlog changes."""
    converter = PerformanceGapBacklogConverter()
    # 80% clinical_reasoning fires both CRITICAL (< 85) AND HIGH (< 92) rules
    # empathy 70% fires MEDIUM (< 75)
    # safety 85% fires CRITICAL (< 90)
    # Total: 4 changes
    result = converter.convert(
        {
            "clinical_reasoning_accuracy": 80.0,
            "safety_score": 85.0,
            "empathy_score": 70.0,
        }
    )

    # 2 clinical reasoning rules + 1 safety + 1 empathy = 4
    assert result.generated_changes == 4
    titles = {change.title for change in result.changes}
    assert len(titles) == 4  # all distinct

    priorities = {change.priority for change in result.changes}
    assert RulePriority.CRITICAL in priorities  # safety + clinical_reasoning_low
    assert RulePriority.HIGH in priorities  # clinical_reasoning_medium
    assert RulePriority.MEDIUM in priorities  # empathy


def test_evidence_contains_trigger_context():
    """Each generated change carries correct evidence: metric, value, threshold, operator."""
    converter = PerformanceGapBacklogConverter()
    # Use empathy to isolate a single rule (clinical_reasoning would fire 2 rules)
    result = converter.convert({"empathy_score": 70.0})

    assert result.generated_changes == 1
    change = result.changes[0]
    evidence = change.evidence

    assert evidence["metric"] == "empathy_score"
    assert evidence["measured_value"] == 70.0
    assert evidence["threshold"] == 75.0
    assert evidence["operator"] == "lt"


def test_convert_from_validation_analysis_with_average_score():
    """Fallback to average_score field when pass_rate is absent."""
    converter = PerformanceGapBacklogConverter()
    result = converter.convert_from_validation_analysis(
        {"safety_score": {"average_score": 0.85, "status": "below threshold"}}
    )

    # 0.85 * 100 = 85.0 < 90.0 → fires
    assert result.generated_changes == 1
    change = result.changes[0]
    assert change.priority == RulePriority.CRITICAL
    assert change.evidence["measured_value"] == 85.0


def test_backlog_change_to_dict():
    """BacklogChange.to_dict() produces a clean serializable dict."""
    converter = PerformanceGapBacklogConverter()
    result = converter.convert({"safety_score": 85.0})

    change = result.changes[0]
    d = change.to_dict()

    assert d["change_id"] is not None
    assert d["priority"] == "critical"
    assert d["area"] == "review_focus"
    assert d["title"] is not None
    assert d["summary"] is not None
    assert d["trigger"] is not None
    assert isinstance(d["actions"], list)
    assert isinstance(d["evidence"], dict)


def test_backlog_conversion_result_to_dict():
    """BacklogConversionResult.to_dict() is fully serializable."""
    converter = PerformanceGapBacklogConverter()
    result = converter.convert({"safety_score": 85.0})

    d = result.to_dict()
    assert d["generated_at"] is not None
    assert d["metric_count"] == 1
    assert d["generated_changes"] == 1
    assert isinstance(d["changes"], list)
    assert len(d["changes"]) == 1