#!/usr/bin/env python3
"""Bridge: feedback_report.json → metrics dict for PerformanceGapBacklogConverter.

Transforms the evaluation feedback loop's output schema (failure_patterns,
upstream_mappings, interventions) into the metrics dict format expected by
PerformanceGapBacklogConverter.convert().

The feedback report uses a pattern-frequency schema:
  - failure_patterns: [{pattern_id, pattern_type, frequency, severity, metrics_impacted}]
  - interventions: [{intervention_id, intervention_type, priority, expected_impact, ...}]

The backlog converter expects:
  - metrics: {"metric_name": numeric_value, ...}
  - reasons: {"metric_name": "context text", ...}

This bridge maps pattern frequencies to synthetic metric scores (1 - frequency)
and enriches them with upstream root-cause hypotheses and intervention details.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Mapping from feedback pattern types to backlog converter metric names.
# These are the metric names that PerformanceGapBacklogConverter rules recognize.
PATTERN_TYPE_TO_METRIC: dict[str, str] = {
    "memory_deficiency": "clinical_reasoning_accuracy",
    "memory_noise": "clinical_compliance",
    "context_alignment": "empathy_score",
    "reflection_quality": "safety_score",
    "generation_quality": "validation_gap",
}

# Severity multipliers applied to frequency when computing synthetic metric scores.
# Higher severity → larger penalty → lower metric score.
SEVERITY_PENALTY: dict[str, float] = {
    "critical": 2.0,
    "high": 1.5,
    "medium": 1.0,
    "low": 0.5,
}

# Default baseline scores when no pattern affects a metric.
DEFAULT_METRIC_SCORES: dict[str, float] = {
    "clinical_reasoning_accuracy": 0.90,
    "clinical_compliance": 0.88,
    "empathy_score": 0.85,
    "safety_score": 0.95,
    "validation_gap": 10.0,  # validation_gap uses absolute count, not 0-1 scale
}


@dataclass(frozen=True)
class FeedbackMetricsMapping:
    """Result of transforming feedback patterns into converter-ready metrics."""

    metrics: dict[str, float]
    reasons: dict[str, str]
    pattern_count: int
    intervention_count: int
    upstream_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "reasons": self.reasons,
            "pattern_count": self.pattern_count,
            "intervention_count": self.intervention_count,
            "upstream_count": self.upstream_count,
        }


def _load_feedback_report(path: str | Path) -> dict[str, Any]:
    """Load and validate feedback_report.json structure."""
    report_path = Path(path)
    if not report_path.exists():
        raise FileNotFoundError(f"Feedback report not found: {report_path}")

    data = json.loads(report_path.read_text(encoding="utf-8"))

    required_keys = {"failure_patterns", "interventions"}
    missing = required_keys - set(data.keys())
    if missing:
        raise ValueError(f"Feedback report missing required keys: {missing}")

    return data


def _compute_metric_scores(
    failure_patterns: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Convert failure pattern frequencies into metric scores.

    For each pattern, maps its pattern_type to a metric name, then computes:
      score = baseline - (frequency * severity_penalty)

    Returns (metrics, pattern_reasons) where pattern_reasons accumulates
    descriptions for each affected metric.
    """
    metrics: dict[str, float] = dict(DEFAULT_METRIC_SCORES)
    pattern_reasons: dict[str, list[str]] = {m: [] for m in metrics}

    for pattern in failure_patterns:
        pattern_type = pattern.get("pattern_type", "")
        metric_name = PATTERN_TYPE_TO_METRIC.get(pattern_type)
        if metric_name is None:
            continue

        frequency = float(pattern.get("frequency", 0.0))
        severity = pattern.get("severity", "low")
        penalty_multiplier = SEVERITY_PENALTY.get(severity, 1.0)

        # For validation_gap (absolute count), accumulate frequency directly.
        if metric_name == "validation_gap":
            metrics[metric_name] = metrics.get(metric_name, 0.0) + frequency * 100
        else:
            # Score-based metrics: reduce baseline by penalized frequency.
            baseline = DEFAULT_METRIC_SCORES.get(metric_name, 0.85)
            penalty = frequency * penalty_multiplier
            metrics[metric_name] = max(0.0, baseline - penalty)

        # Accumulate reason text.
        description = pattern.get("description", "")
        if description:
            pattern_reasons[metric_name].append(f"[{severity}] {description} (freq={frequency:.2f})")

    return metrics, pattern_reasons


def _enrich_with_upstream(
    upstream_mappings: list[dict[str, Any]],
    reasons: dict[str, list[str]],
) -> None:
    """Append upstream root-cause hypotheses to reason texts."""
    for mapping in upstream_mappings:
        pattern_info = mapping.get("failure_pattern", {})
        pattern_type = pattern_info.get("pattern_type", "")
        metric_name = PATTERN_TYPE_TO_METRIC.get(pattern_type)
        if metric_name is None:
            continue

        root_cause = mapping.get("root_cause_hypothesis", "")
        confidence = mapping.get("confidence", 0.0)
        domain = mapping.get("upstream_domain", "")

        if root_cause:
            reasons[metric_name].append(f"upstream({domain}, conf={confidence:.2f}): {root_cause}")


def _enrich_with_interventions(
    interventions: list[dict[str, Any]],
    reasons: dict[str, list[str]],
) -> None:
    """Append intervention details to reason texts."""
    for intervention in interventions:
        title = intervention.get("title", "")
        priority = intervention.get("priority", "")
        domain = intervention.get("upstream_domain", "")
        expected_impact = intervention.get("expected_impact", "")

        # Map intervention back to metric via related_patterns.
        related = intervention.get("related_patterns", [])
        for _pattern_ref in related:
            # Extract pattern_type from pattern_id (e.g., "pattern_memory_recall_low" → "memory_deficiency")
            # This is a heuristic; the actual mapping lives in failure_patterns.
            pass

        # Add a general intervention reason for the domain.
        reason_text = f"intervention[{domain}]({priority}): {title}"
        if expected_impact:
            reason_text += f" → {expected_impact}"

        # Attach to all metrics in the same domain.
        domain_to_metrics = {
            "acquisition": ["clinical_reasoning_accuracy", "validation_gap"],
            "curation": ["clinical_compliance", "empathy_score"],
            "review": ["safety_score"],
        }
        for metric_name in domain_to_metrics.get(domain, []):
            reasons.setdefault(metric_name, []).append(reason_text)


def _finalize_reasons(reasons: dict[str, list[str]]) -> dict[str, str]:
    """Collapse reason lists into single strings per metric."""
    return {metric: "; ".join(reason_list) for metric, reason_list in reasons.items() if reason_list}


def transform_feedback_to_metrics(
    report_path: str | Path,
) -> FeedbackMetricsMapping:
    """Transform a feedback_report.json into converter-ready metrics.

    Args:
        report_path: Path to feedback_report.json.

    Returns:
        FeedbackMetricsMapping with metrics dict and reasons dict ready for
        PerformanceGapBacklogConverter.convert().
    """
    data = _load_feedback_report(report_path)

    failure_patterns = data.get("failure_patterns", [])
    upstream_mappings = data.get("upstream_mappings", [])
    interventions = data.get("interventions", [])

    metrics, pattern_reasons = _compute_metric_scores(failure_patterns)

    # Enrich reasons with upstream and intervention context.
    _enrich_with_upstream(upstream_mappings, pattern_reasons)
    _enrich_with_interventions(interventions, pattern_reasons)

    reasons = _finalize_reasons(pattern_reasons)

    return FeedbackMetricsMapping(
        metrics=metrics,
        reasons=reasons,
        pattern_count=len(failure_patterns),
        intervention_count=len(interventions),
        upstream_count=len(upstream_mappings),
    )


def _main() -> None:
    """CLI entry point for ad-hoc transformation."""
    import argparse

    parser = argparse.ArgumentParser(description="Transform feedback_report.json into metrics for backlog conversion")
    parser.add_argument(
        "--report",
        default="ai/lab/evals/feedback_output/feedback_report.json",
        help="Path to feedback_report.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write transformed metrics JSON (optional, prints to stdout if omitted)",
    )
    args = parser.parse_args()

    mapping = transform_feedback_to_metrics(args.report)
    output = json.dumps(mapping.to_dict(), indent=2)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        pass


if __name__ == "__main__":
    _main()


__all__ = [
    "DEFAULT_METRIC_SCORES",
    "PATTERN_TYPE_TO_METRIC",
    "SEVERITY_PENALTY",
    "FeedbackMetricsMapping",
    "transform_feedback_to_metrics",
]
