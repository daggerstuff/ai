#!/usr/bin/env python3
"""Performance gap -> backlog conversion rules.

This module codifies how evaluation gaps are converted into concrete pipeline
actions for source acquisition, curation, and review focus. It is designed to be
the first execution layer after quality-performance reporting so gaps become
operational work, not only diagnostics.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class RulePriority(StrEnum):
    """Backlog rule urgency levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class BacklogChange:
    """A concrete action that should be applied to the training data workflow."""

    change_id: str
    priority: RulePriority
    area: str
    title: str
    summary: str
    trigger: str
    actions: list[str]
    suggested_sop: str
    expected_impact: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_id": self.change_id,
            "priority": self.priority.value,
            "area": self.area,
            "title": self.title,
            "summary": self.summary,
            "trigger": self.trigger,
            "actions": self.actions,
            "suggested_sop": self.suggested_sop,
            "expected_impact": self.expected_impact,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class ConversionRule:
    """Rule that maps one metric gap to one or more backlog actions."""

    rule_id: str
    metric: str
    operator: str
    threshold: float
    priority: RulePriority
    area: str
    title: str
    summary_template: str
    actions: list[str]
    suggested_sop: str
    expected_impact: str


@dataclass
class BacklogConversionResult:
    """Output package returned by the conversion engine."""

    generated_at: str
    metric_count: int
    generated_changes: int
    changes: list[BacklogChange] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "metric_count": self.metric_count,
            "generated_changes": self.generated_changes,
            "changes": [change.to_dict() for change in self.changes],
        }


def _to_percent(value: float) -> float:
    """Convert score-like inputs into percentage scale for consistent comparisons."""
    if 0.0 <= value <= 1.0:
        return value * 100.0
    return value


def _build_change_id(rule_id: str, metric: str, trigger_value: float) -> str:
    """Create stable IDs for generated backlog changes."""
    hash_input = f"{rule_id}:{metric}:{round(trigger_value, 4)}".encode()
    digest = hashlib.sha256(hash_input).hexdigest()[:10]
    return f"{rule_id}:{metric}:{digest}"


class PerformanceGapBacklogConverter:
    """Translate measured performance gaps into concrete backlog changes."""

    def __init__(self) -> None:
        self.rules: list[ConversionRule] = self._default_rules()

    @staticmethod
    def _default_rules() -> list[ConversionRule]:
        return [
            ConversionRule(
                rule_id="clinical_reasoning_low",
                metric="clinical_reasoning_accuracy",
                operator="lt",
                threshold=85.0,
                priority=RulePriority.CRITICAL,
                area="acquisition",
                title="Prioritize clinical conversation sources",
                summary_template=(
                    "Clinical reasoning accuracy is below 85%; prioritize high-"
                    "quality clinical sources and add targeted source selection."
                ),
                actions=[
                    "Raise acquisition priority for sources with clinical-therapy tags",
                    "Route source selection to increase clinical reasoning coverage by +20%",
                    "Add review gate requiring clinical rationale validation before merge",
                ],
                suggested_sop=(
                    "Adjust source queue weighting to double-source clinical "
                    "channels, then run a 2-cycle pilot before broad rollout."
                ),
                expected_impact="Increase clinical reasoning coverage and reduce "
                "clinical failures by 10-20% in next validation batch",
            ),
            ConversionRule(
                rule_id="clinical_reasoning_medium",
                metric="clinical_reasoning_accuracy",
                operator="lt",
                threshold=92.0,
                priority=RulePriority.HIGH,
                area="curation_rules",
                title="Tighten curation filters for clinical rationale",
                summary_template=(
                    "Clinical reasoning is degrading; adjust curation rules to catch "
                    "weak rationale before training ingestion."
                ),
                actions=[
                    "Introduce curation rule: require clear intervention/plan structure",
                    "Add minimum clinical anchor requirement (grounded recommendation or safety signpost)",
                    "Increase curator sampling rate for borderline clinical responses",
                ],
                suggested_sop=(
                    "Apply rule changes to 50% of incoming clinical batches for two days, "
                    "then compare post-curation gap metrics."
                ),
                expected_impact="Improve signal quality of clinical content and stabilize reasoning pass rates",
            ),
            ConversionRule(
                rule_id="clinical_compliance_low",
                metric="clinical_compliance",
                operator="lt",
                threshold=80.0,
                priority=RulePriority.CRITICAL,
                area="review_focus",
                title="Shift review attention to clinical compliance failures",
                summary_template=(
                    "Clinical compliance below 80% requires focused review focus and "
                    "dataset balancing before continuing scale-up."
                ),
                actions=[
                    "Assign dedicated reviewer pass on top 20 failing clinical patterns",
                    "Hold non-clinical/adjacent samples until compliance review complete",
                    "Create compliance-first sampling lane in queue",
                ],
                suggested_sop=(
                    "Pause global rollout and process a 10K sample set through human review "
                    "for compliance-tagged failures."
                ),
                expected_impact="Reduce compliance-related rejects and improve safety alignment",
            ),
            ConversionRule(
                rule_id="empathy_gap_low",
                metric="empathy_score",
                operator="lt",
                threshold=75.0,
                priority=RulePriority.MEDIUM,
                area="curation_rules",
                title="Shift dataset weighting toward empathetic style coverage",
                summary_template=(
                    "Empathy score below threshold indicates low emotional attunement in curated samples."
                ),
                actions=[
                    "Add empathetic response pattern checks to pre-ingestion curation",
                    "Lower tolerance for abrupt/flat emotional tone in selected channels",
                    "Add source enrichment task for supportive-response-heavy datasets",
                ],
                suggested_sop=(
                    "Run a curation rule A/B test for 3k records and compare empathy pass rates before full rollout."
                ),
                expected_impact="Increase emotional alignment and reduce low-empathy failure clusters",
            ),
            ConversionRule(
                rule_id="safety_score_low",
                metric="safety_score",
                operator="lt",
                threshold=90.0,
                priority=RulePriority.CRITICAL,
                area="review_focus",
                title="Escalate crisis detection and harm review capacity",
                summary_template=("Safety score under threshold requires immediate reviewer and rule escalation."),
                actions=[
                    "Increase review rate for crisis-related samples by +30%",
                    "Add dedicated safety escalation checklist for high-risk channels",
                    "Restrict ambiguous crisis phrasing sources from automatic intake",
                ],
                suggested_sop=(
                    "Create a temporary high-risk queue and require dual-pass verification before approval."
                ),
                expected_impact="Lower risk of unsafe output and reduce safety exceptions post-deploy",
            ),
            ConversionRule(
                rule_id="processing_validation_gap",
                metric="validation_gap",
                operator="gt",
                threshold=30.0,
                priority=RulePriority.HIGH,
                area="pipeline_allocation",
                title="Increase pipeline cycles for bottlenecked quality checks",
                summary_template=(
                    "Large validation backlog indicates throughput is not keeping pace with quality feedback volume."
                ),
                actions=[
                    "Raise throughput on quality gates for critical metrics first",
                    "Re-order pipeline stages: safety and clinical checks before low-priority transforms",
                    "Add temporary capacity for validation workers",
                ],
                suggested_sop=(
                    "Add two additional validation workers for one sprint; compare queue age and fail-rate lag."
                ),
                expected_impact="Reduce validation lag and make feedback loops actionable sooner",
            ),
        ]

    def _passes(self, metric_value: float, operator: str, threshold: float) -> bool:
        if operator == "lt":
            return metric_value < threshold
        if operator == "gt":
            return metric_value > threshold
        return False

    def convert(self, metrics: dict[str, float], reasons: dict[str, str] | None = None) -> BacklogConversionResult:
        """Convert metric gaps into backlog actions.

        Args:
            metrics: Mapping of metric name -> numeric metric value.
            reasons: Optional mapping of metric -> context text used for evidence.
        """
        reasons = reasons or {}
        generated: list[BacklogChange] = []
        seen: set[str] = set()

        for rule in self.rules:
            if rule.metric not in metrics:
                continue

            raw_value = float(metrics[rule.metric])
            metric_value = _to_percent(raw_value)
            if not self._passes(metric_value, rule.operator, rule.threshold):
                continue

            change_id = _build_change_id(rule.rule_id, rule.metric, metric_value)
            if change_id in seen:
                continue
            seen.add(change_id)

            generated.append(
                BacklogChange(
                    change_id=change_id,
                    priority=rule.priority,
                    area=rule.area,
                    title=rule.title,
                    summary=rule.summary_template,
                    trigger=(f"{rule.metric}={metric_value:.2f} ({rule.operator} {rule.threshold})"),
                    actions=rule.actions,
                    suggested_sop=rule.suggested_sop,
                    expected_impact=rule.expected_impact,
                    evidence={
                        "metric": rule.metric,
                        "measured_value": metric_value,
                        "threshold": rule.threshold,
                        "operator": rule.operator,
                        "notes": reasons.get(rule.metric, ""),
                    },
                )
            )

        return BacklogConversionResult(
            generated_at=datetime.now(UTC).isoformat(),
            metric_count=len(metrics),
            generated_changes=len(generated),
            changes=generated,
        )

    def convert_from_validation_analysis(self, analyses: dict[str, Any]) -> BacklogConversionResult:
        """Convert quality gap analysis payloads into backlog changes.

        The payload can be produced by QualityValidationAnalyzer or any caller with
        similar structure:
          {"clinical_compliance": {"pass_rate": 72.0}, ...}
        """
        metrics: dict[str, float] = {}
        reasons: dict[str, str] = {}

        # Clinical compliance is historically critical for this project and mapped
        # from legacy "clinical_compliance" metric.
        for metric, payload in analyses.items():
            if not isinstance(payload, dict):
                continue
            pass_rate = payload.get("pass_rate")
            if pass_rate is not None:
                metrics[metric] = float(pass_rate)
                reasons[metric] = payload.get("failure_patterns", [])
                if isinstance(reasons[metric], list):
                    reasons[metric] = "; ".join(str(item) for item in reasons[metric])
            elif "average_score" in payload:
                # Fallback for callers that send only score.
                metrics[metric] = _to_percent(float(payload["average_score"]))
                reasons[metric] = payload.get("status", "")

        # Add synthetic bridge metric used for throughput-style planning.
        failures = [
            metric_payload.get("failed_validations", 0)
            for metric_payload in analyses.values()
            if isinstance(metric_payload, dict)
        ]
        if failures:
            metrics.setdefault("validation_gap", sum(failures))
            reasons["validation_gap"] = f"Sum failed_validations={sum(failures)}"

        # Special alias mapping from legacy metric names to new rule names:
        # clinical_reasoning_accuracy maps from therapeutic_accuracy / clinical_compliance
        if "clinical_reasoning_accuracy" not in metrics:
            if "therapeutic_accuracy" in metrics:
                metrics["clinical_reasoning_accuracy"] = metrics["therapeutic_accuracy"]
                reasons["clinical_reasoning_accuracy"] = (
                    "Mapped from therapeutic_accuracy for clinical reasoning parity"
                )
            elif "clinical_compliance" in metrics:
                metrics["clinical_reasoning_accuracy"] = metrics["clinical_compliance"]
                reasons["clinical_reasoning_accuracy"] = "Mapped from clinical_compliance for reasoning proxy"
        if "empathy_score" not in metrics and "emotional_authenticity" in metrics:
            metrics["empathy_score"] = metrics["emotional_authenticity"]
            reasons["empathy_score"] = "Mapped from emotional_authenticity for empathetic response checks"

        return self.convert(metrics, reasons)

    def export(self, result: BacklogConversionResult, output_path: str | Path) -> str:
        """Persist a conversion report to disk and return the path."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        return str(output_file)


def _main() -> None:
    """Simple manual invocation for ad-hoc conversion payloads."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert performance gaps into backlog updates")
    parser.add_argument(
        "--metrics-json",
        required=True,
        help="JSON string with metric_name -> value map",
    )
    parser.add_argument(
        "--output",
        default="/tmp/performance_gap_backlog_plan.json",
        help="Path for persisted backlog conversion report",
    )
    args = parser.parse_args()

    raw_metrics = json.loads(args.metrics_json)
    if not isinstance(raw_metrics, dict):
        raise ValueError("--metrics-json must be an object")

    converter = PerformanceGapBacklogConverter()
    result = converter.convert({k: float(v) for k, v in raw_metrics.items() if v is not None})
    converter.export(result, args.output)


if __name__ == "__main__":
    _main()


__all__ = [
    "BacklogChange",
    "BacklogConversionResult",
    "ConversionRule",
    "PerformanceGapBacklogConverter",
    "RulePriority",
]
