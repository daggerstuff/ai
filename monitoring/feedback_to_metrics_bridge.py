"""Bridge: feedback_report.json -> metrics dict for PerformanceGapBacklogConverter.

The evaluation feedback loop reports *failure patterns* (a pattern type, how
often it fires, how severe it is). The backlog converter consumes *metric
scores* plus a human-readable reason per metric. This module translates between
the two, and carries the upstream root-cause hypotheses and the proposed
interventions across so the backlog entry explains itself.

Pattern frequencies are turned into synthetic scores:

    score = baseline - (frequency * severity_penalty)

except ``validation_gap``, which is an absolute count and accumulates
``frequency * 100`` per matching pattern instead.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PATTERN_TYPE_TO_METRIC: dict[str, str] = {
    "memory_deficiency": "clinical_reasoning_accuracy",
    "memory_noise": "clinical_compliance",
    "context_alignment": "empathy_score",
    "reflection_quality": "safety_score",
    "generation_quality": "validation_gap",
}

SEVERITY_PENALTY: dict[str, float] = {
    "critical": 2.0,
    "high": 1.5,
    "medium": 1.0,
    "low": 0.5,
}

DEFAULT_METRIC_SCORES: dict[str, float] = {
    "clinical_reasoning_accuracy": 0.90,
    "clinical_compliance": 0.88,
    "empathy_score": 0.85,
    "safety_score": 0.95,
    "validation_gap": 10.0,
}

DOMAIN_TO_METRICS: dict[str, list[str]] = {
    "acquisition": ["clinical_reasoning_accuracy", "validation_gap"],
    "curation": ["clinical_compliance", "empathy_score"],
    "review": ["safety_score"],
}

VALIDATION_GAP_SCALE = 100.0
CLI_ARGUMENT_COUNT = 2
REQUIRED_REPORT_KEYS = frozenset({"failure_patterns", "interventions"})


@dataclass(frozen=True)
class FeedbackMetricsMapping:
    """Result of transforming feedback patterns into converter-ready metrics."""

    metrics: dict[str, float]
    reasons: dict[str, str]
    pattern_count: int
    intervention_count: int
    upstream_count: int

    def to_dict(self) -> dict[str, Any]:
        """Serialise the mapping for a backlog entry."""
        return {
            "metrics": self.metrics,
            "reasons": self.reasons,
            "pattern_count": self.pattern_count,
            "intervention_count": self.intervention_count,
            "upstream_count": self.upstream_count,
        }


def _load_feedback_report(path: str | Path) -> dict[str, Any]:
    """Load a feedback report, rejecting anything without the required sections."""
    report_path = Path(path)
    if not report_path.exists():
        raise FileNotFoundError(f"Feedback report not found: {report_path}")

    data = json.loads(report_path.read_text(encoding="utf-8"))
    missing = REQUIRED_REPORT_KEYS - set(data)
    if missing:
        raise ValueError(f"Feedback report missing required keys: {sorted(missing)}")
    return data


def _compute_metric_scores(
    failure_patterns: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Turn pattern frequencies into metric scores plus per-metric reason text."""
    metrics: dict[str, float] = dict(DEFAULT_METRIC_SCORES)
    pattern_reasons: dict[str, list[str]] = {metric: [] for metric in metrics}

    for pattern in failure_patterns:
        metric_name = PATTERN_TYPE_TO_METRIC.get(pattern.get("pattern_type", ""))
        if metric_name is None:
            continue

        frequency = float(pattern.get("frequency", 0.0))
        severity = pattern.get("severity", "low")
        penalty_multiplier = SEVERITY_PENALTY.get(severity, 1.0)

        if metric_name == "validation_gap":
            metrics[metric_name] = metrics.get(metric_name, 0.0) + frequency * VALIDATION_GAP_SCALE
        else:
            baseline = DEFAULT_METRIC_SCORES.get(metric_name, 0.85)
            metrics[metric_name] = max(0.0, baseline - frequency * penalty_multiplier)

        description = pattern.get("description", "")
        if description:
            pattern_reasons[metric_name].append(f"[{severity}] {description} (freq={frequency:.2f})")

    return metrics, pattern_reasons


def _enrich_with_upstream(
    upstream_mappings: list[dict[str, Any]],
    reasons: dict[str, list[str]],
) -> None:
    """Append each upstream root-cause hypothesis to its metric's reasons."""
    for mapping in upstream_mappings:
        pattern_type = mapping.get("failure_pattern", {}).get("pattern_type", "")
        metric_name = PATTERN_TYPE_TO_METRIC.get(pattern_type)
        if metric_name is None:
            continue

        root_cause = mapping.get("root_cause_hypothesis", "")
        if not root_cause:
            continue

        confidence = float(mapping.get("confidence", 0.0))
        domain = mapping.get("upstream_domain", "")
        reasons.setdefault(metric_name, []).append(f"upstream({domain}, conf={confidence:.2f}): {root_cause}")


def _enrich_with_interventions(
    interventions: list[dict[str, Any]],
    reasons: dict[str, list[str]],
) -> None:
    """Append each intervention to every metric of its upstream domain."""
    for intervention in interventions:
        domain = intervention.get("upstream_domain", "")
        targets = DOMAIN_TO_METRICS.get(domain, [])
        if not targets:
            continue

        title = intervention.get("title", "")
        priority = intervention.get("priority", "")
        reason_text = f"intervention[{domain}]({priority}): {title}"
        expected_impact = intervention.get("expected_impact", "")
        if expected_impact:
            reason_text += f" -> {expected_impact}"

        for metric_name in targets:
            reasons.setdefault(metric_name, []).append(reason_text)


def _finalize_reasons(reasons: dict[str, list[str]]) -> dict[str, str]:
    """Collapse each reason list into one string, dropping metrics with none."""
    return {metric: "; ".join(entries) for metric, entries in reasons.items() if entries}


def transform_feedback_to_metrics(report_path: str | Path) -> FeedbackMetricsMapping:
    """Transform a ``feedback_report.json`` into converter-ready metrics."""
    data = _load_feedback_report(report_path)

    failure_patterns = data.get("failure_patterns", [])
    upstream_mappings = data.get("upstream_mappings", [])
    interventions = data.get("interventions", [])

    metrics, pattern_reasons = _compute_metric_scores(failure_patterns)
    _enrich_with_upstream(upstream_mappings, pattern_reasons)
    _enrich_with_interventions(interventions, pattern_reasons)

    return FeedbackMetricsMapping(
        metrics=metrics,
        reasons=_finalize_reasons(pattern_reasons),
        pattern_count=len(failure_patterns),
        intervention_count=len(interventions),
        upstream_count=len(upstream_mappings),
    )


def _main() -> None:
    """Print the transformed mapping for one report path."""
    if len(sys.argv) != CLI_ARGUMENT_COUNT:
        print("usage: feedback_to_metrics_bridge.py <feedback_report.json>")
        raise SystemExit(2)

    mapping = transform_feedback_to_metrics(sys.argv[1])
    print(json.dumps(mapping.to_dict(), indent=2))


if __name__ == "__main__":
    _main()
