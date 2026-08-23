#!/usr/bin/env python3
"""Tests for Linear backlog artifact generation."""

from ai.monitoring.linear_backlog_action_builder import (
    build_linear_backlog_payload,
    write_linear_backlog_artifact,
)
from ai.monitoring.performance_gap_backlog_converter import (
    BacklogChange,
    BacklogConversionResult,
    RulePriority,
)


def test_build_linear_backlog_payload_includes_mcp_hint_and_actions(tmp_path):
    result = BacklogConversionResult(
        generated_at="2026-05-13T00:00:00+00:00",
        metric_count=1,
        generated_changes=1,
        changes=[
            BacklogChange(
                change_id="rule:metric:abc",
                priority=RulePriority.CRITICAL,
                area="review_focus",
                title="Escalate review for safety failures",
                summary="Safety score below target",
                trigger="safety_score=88.0 (< 90.0)",
                actions=["Add safety triage lane", "Require dual-pass review"],
                suggested_sop="Create temporary high risk queue",
                expected_impact="Lower safety incident risk",
                evidence={"metric": "safety_score", "measured_value": 88.0},
            )
        ],
    )

    payload = build_linear_backlog_payload(result, project_key="MOD", include_mcp_instructions=True)
    assert payload["source"]["task"] == "PIX-535"
    assert payload["source"]["change_count"] == 1

    action = payload["actions"][0]
    assert action["title"] == "Escalate review for safety failures"
    assert action["priority"] == "urgent"
    assert "project_key" in action
    assert action["mcp_payload"] is not None
    assert action["mcp_payload"]["tool"] == "mcp__linear__create_issue"
    assert action["mcp_payload"]["variables"]["priority"] == "urgent"


def test_write_linear_backlog_artifact_roundtrip(tmp_path):
    payload = {"foo": "bar", "actions": []}
    output = tmp_path / "artifact.json"
    path = write_linear_backlog_artifact(payload, output)
    assert path == str(output)
    loaded = output.read_text()
    assert "foo" in loaded
