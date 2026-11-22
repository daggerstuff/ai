#!/usr/bin/env python3
"""Build Linear-ready backlog action artifacts from gap-to-backlog changes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .performance_gap_backlog_converter import BacklogConversionResult


@dataclass(frozen=True)
class LinearBacklogAction:
    """Linear-friendly representation of a conversion change."""

    operation: str
    title: str
    description: str
    priority: str
    labels: list[str]
    area: str
    change_id: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "labels": self.labels,
            "area": self.area,
            "change_id": self.change_id,
            "evidence": self.evidence,
        }


def _priority_to_linear_priority(priority: str) -> str:
    """Map internal priorities to Linear import format."""
    normalized = priority.lower()
    if normalized == "critical":
        return "urgent"
    if normalized == "high":
        return "high"
    if normalized == "medium":
        return "medium"
    return "low"


def build_linear_backlog_payload(
    conversion_result: BacklogConversionResult,
    *,
    project_key: str = "PIX",
    default_parent_issue: str = "PIX-535",
    include_mcp_instructions: bool = True,
) -> dict[str, Any]:
    """Create a Linear draft payload from backlog conversion output."""
    actions = []

    for change in conversion_result.changes:
        action_body = (
            f"{change.summary}\\n\\n"
            f"Trigger: {change.trigger}\\n"
            f"SOP: {change.suggested_sop}\\n"
            f"Expected Impact: {change.expected_impact}\\n"
            f"Evidence: {change.evidence}"
        )
        labels = ["pix-535", "performance-gap", change.area, change.priority.value]
        description = (
            f"{action_body}\\n\\nExecution checklist:\\n" + "\\n".join(f"- {item}" for item in change.actions) + "\\n"
        )

        actions.append(
            {
                "operation": "create",
                "title": change.title,
                "description": description,
                "priority": _priority_to_linear_priority(change.priority.value),
                "labels": labels,
                "area": change.area,
                "project_key": project_key,
                "parent_issue": default_parent_issue,
                "change_id": change.change_id,
                "evidence": change.evidence,
                "mcp_payload": (
                    {
                        "tool": "mcp__linear__create_issue",
                        "variables": {
                            "title": change.title,
                            "description": description,
                            "projectId": project_key,
                            "labelIds": labels,
                            "priority": _priority_to_linear_priority(change.priority.value),
                            "parentId": default_parent_issue,
                        },
                    }
                    if include_mcp_instructions
                    else None
                ),
            }
        )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": {
            "task": "PIX-535",
            "task_label": "performance-gap-to-backlog conversion",
            "metric_count": conversion_result.metric_count,
            "change_count": conversion_result.generated_changes,
        },
        "actions": actions,
    }


def write_linear_backlog_artifact(payload: dict[str, Any], output_path: str | Path) -> str:
    """Write payload to disk and return path."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")
    return str(output_file)


__all__ = [
    "LinearBacklogAction",
    "build_linear_backlog_payload",
    "write_linear_backlog_artifact",
]
