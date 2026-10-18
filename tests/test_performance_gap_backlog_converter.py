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
