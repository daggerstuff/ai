#!/usr/bin/env python3
"""End-to-end bridge: feedback_report.json → Linear issue creation.

Wires together:
  1. FeedbackMetricsMapping (feedback_to_metrics_bridge)
  2. PerformanceGapBacklogConverter.convert()
  3. build_linear_backlog_payload()
  4. LinearBacklogDispatcher.dispatch_backlog_actions()

This is the operational bridge that PIX-537 requires to make evaluation
feedback produce actual Linear issues instead of orphaned markdown files.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .feedback_to_metrics_bridge import (
    FeedbackMetricsMapping,
    transform_feedback_to_metrics,
)
from .linear_backlog_action_builder import (
    build_linear_backlog_payload,
    write_linear_backlog_artifact,
)
from .linear_backlog_dispatcher import LinearBacklogDispatcher
from .performance_gap_backlog_converter import (
    BacklogConversionResult,
    PerformanceGapBacklogConverter,
)


@dataclass(frozen=True)
class FeedbackLinearResult:
    """Complete result of the feedback → Linear pipeline."""

    feedback_mapping: FeedbackMetricsMapping
    conversion_result: BacklogConversionResult
    linear_payload: dict[str, Any]
    dispatch_result: dict[str, Any]
    artifact_path: str
    executed_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "executed_at": self.executed_at,
            "feedback_summary": {
                "pattern_count": self.feedback_mapping.pattern_count,
                "intervention_count": self.feedback_mapping.intervention_count,
                "metrics_count": len(self.feedback_mapping.metrics),
            },
            "conversion_summary": {
                "changes_generated": self.conversion_result.generated_changes,
                "metric_count": self.conversion_result.metric_count,
            },
            "dispatch_summary": {
                "mode": self.dispatch_result.get("mode"),
                "created": self.dispatch_result.get("created"),
                "queued": self.dispatch_result.get("queued"),
                "failed": self.dispatch_result.get("failed"),
                "updated": self.dispatch_result.get("updated"),
            },
            "artifact_path": self.artifact_path,
        }


def execute_feedback_linear_bridge(
    feedback_report_path: str | Path,
    *,
    artifact_output_dir: str | Path = "monitoring/linear_backlog_artifacts",
    project_key: str = "PIX",
    parent_issue: str = "PIX-535",
    dispatcher: LinearBacklogDispatcher | None = None,
) -> FeedbackLinearResult:
    """Run the full feedback → Linear pipeline.

    Args:
        feedback_report_path: Path to feedback_report.json.
        artifact_output_dir: Where to write Linear backlog artifacts.
        project_key: Linear project key.
        parent_issue: Parent issue ID for created issues.
        dispatcher: Optional pre-configured dispatcher (creates default if None).

    Returns:
        FeedbackLinearResult with all intermediate and final results.

    Raises:
        FileNotFoundError: If feedback_report_path does not exist.
        ValueError: If project_key or parent_issue is empty.
    """
    report_path = Path(feedback_report_path)
    if not report_path.is_file():
        raise FileNotFoundError(f"Feedback report not found: {report_path}")
    if not project_key.strip():
        raise ValueError("project_key must be a non-empty Linear project key")
    if not parent_issue.strip():
        raise ValueError("parent_issue must be a non-empty Linear issue identifier")

    executed_at = datetime.now(UTC).isoformat()

    # Step 1: Transform feedback report into metrics dict.
    feedback_mapping = transform_feedback_to_metrics(report_path)

    # Step 2: Convert metrics into backlog actions.
    converter = PerformanceGapBacklogConverter()
    conversion_result = converter.convert(
        metrics=feedback_mapping.metrics,
        reasons=feedback_mapping.reasons,
    )

    # Step 3: Build Linear-ready payload.
    linear_payload = build_linear_backlog_payload(
        conversion_result,
        project_key=project_key,
        default_parent_issue=parent_issue,
    )

    # Step 4: Write artifact and dispatch.
    artifact_dir = Path(artifact_output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = write_linear_backlog_artifact(linear_payload, artifact_dir / "feedback_linear_payload.json")

    if dispatcher is None:
        dispatcher = LinearBacklogDispatcher(
            queue_path=str(artifact_dir / "pending_actions.jsonl"),
        )

    dispatch_result = dispatcher.dispatch_backlog_actions(linear_payload)

    return FeedbackLinearResult(
        feedback_mapping=feedback_mapping,
        conversion_result=conversion_result,
        linear_payload=linear_payload,
        dispatch_result=dispatch_result,
        artifact_path=artifact_path,
        executed_at=executed_at,
    )


def _main() -> None:
    """CLI entry point for ad-hoc execution."""
    parser = argparse.ArgumentParser(description="Execute feedback → Linear bridge pipeline")
    parser.add_argument(
        "--report",
        default="ai/lab/evals/feedback_output/feedback_report.json",
        help="Path to feedback_report.json",
    )
    parser.add_argument(
        "--output-dir",
        default="monitoring/linear_backlog_artifacts",
        help="Directory for Linear artifacts",
    )
    parser.add_argument(
        "--parent-issue",
        default="PIX-535",
        help="Parent issue ID for created Linear issues",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build payload but skip GraphQL dispatch",
    )
    args = parser.parse_args()

    dispatcher = None
    if args.dry_run:
        dispatcher = LinearBacklogDispatcher(
            queue_path=str(Path(args.output_dir) / "pending_actions.jsonl"),
        )
        # Override credentials check to force queue-only mode.
        dispatcher.linear_api_key = ""

    result = execute_feedback_linear_bridge(
        feedback_report_path=args.report,
        artifact_output_dir=args.output_dir,
        parent_issue=args.parent_issue,
        dispatcher=dispatcher,
    )

    sys.stdout.write(json.dumps(result.to_dict(), indent=2) + "\n")


if __name__ == "__main__":
    _main()


__all__ = [
    "FeedbackLinearResult",
    "execute_feedback_linear_bridge",
]
