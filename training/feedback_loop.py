#!/usr/bin/env python3
"""
Evaluation-to-Data Feedback Loop

Closes the loop between evaluation findings and upstream dataset pipeline actions.
This module:
1. Parses evaluation results and identifies failure patterns
2. Maps failures to likely upstream causes
3. Generates concrete dataset interventions
4. Creates actionable backlog items for upstream pipelines

Usage:
    python -m ai.training.feedback_loop \
        --evaluation-report ./evaluation_report.json \
        --output-dir ./feedback/actions
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class UpstreamDomain(StrEnum):
    """Upstream domains that can be affected by feedback."""
    ACQUISITION = "acquisition"  # PIX-188
    CURATION = "curation"  # PIX-247
    PRIVACY = "privacy"  # PIX-248
    REVIEW = "review"  # PIX-250


class InterventionType(StrEnum):
    """Types of interventions that can be generated."""
    RULE_CHANGE = "rule_change"  # Modify curation/validation rules
    THRESHOLD_CHANGE = "threshold_change"  # Adjust quality thresholds
    PRIORITY_CHANGE = "priority_change"  # Change data source priorities
    REVIEW_FOCUS = "review_focus"  # Add human review focus area
    VALIDATION_GATE = "validation_gate"  # Add new validation check
    DATASET_FILTER = "dataset_filter"  # Filter specific data patterns


@dataclass
class FailurePattern:
    """Identified failure pattern from evaluation."""

    pattern_id: str
    pattern_type: str
    description: str
    affected_examples: list[str]
    severity: str  # critical, high, medium, low
    frequency: float  # 0.0-1.0
    metrics_impacted: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "affected_examples": self.affected_examples,
            "severity": self.severity,
            "frequency": self.frequency,
            "metrics_impacted": self.metrics_impacted,
        }


@dataclass
class UpstreamMapping:
    """Mapping of failure to upstream cause."""

    failure_pattern: FailurePattern
    likely_upstream_domain: UpstreamDomain
    confidence: float  # 0.0-1.0
    root_cause_hypothesis: str
    evidence: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_pattern": self.failure_pattern.to_dict(),
            "upstream_domain": self.likely_upstream_domain.value,
            "confidence": self.confidence,
            "root_cause_hypothesis": self.root_cause_hypothesis,
            "evidence": self.evidence,
        }


@dataclass
class DatasetIntervention:
    """Concrete intervention to improve dataset quality."""

    intervention_id: str
    intervention_type: InterventionType
    title: str
    description: str
    upstream_domain: UpstreamDomain
    priority: str  # critical, high, medium, low
    expected_impact: str
    implementation_details: dict[str, Any]
    validation_criteria: list[str]
    related_patterns: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "intervention_id": self.intervention_id,
            "intervention_type": self.intervention_type.value,
            "title": self.title,
            "description": self.description,
            "upstream_domain": self.upstream_domain.value,
            "priority": self.priority,
            "expected_impact": self.expected_impact,
            "implementation_details": self.implementation_details,
            "validation_criteria": self.validation_criteria,
            "related_patterns": self.related_patterns,
        }


@dataclass
class FeedbackReport:
    """Complete feedback report with mappings and interventions."""

    evaluation_source: str
    generated_at: str
    total_evaluated: int
    overall_score: float

    # Analysis results
    failure_patterns: list[FailurePattern] = field(default_factory=list)
    upstream_mappings: list[UpstreamMapping] = field(default_factory=list)
    interventions: list[DatasetIntervention] = field(default_factory=list)

    # Summary
    critical_issues: int = 0
    high_priority_issues: int = 0
    recommended_actions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_source": self.evaluation_source,
            "generated_at": self.generated_at,
            "total_evaluated": self.total_evaluated,
            "overall_score": self.overall_score,
            "failure_patterns": [p.to_dict() for p in self.failure_patterns],
            "upstream_mappings": [m.to_dict() for m in self.upstream_mappings],
            "interventions": [i.to_dict() for i in self.interventions],
            "summary": {
                "critical_issues": self.critical_issues,
                "high_priority_issues": self.high_priority_issues,
                "recommended_actions": self.recommended_actions,
            },
        }


class EvaluationParser:
    """Parses evaluation results and extracts failure patterns."""

    # Failure pattern templates
    PATTERN_TEMPLATES = {
        "memory_recall_low": {
            "type": "memory_deficiency",
            "description": "Model fails to recall relevant memories in context",
            "threshold": 0.6,
            "metric": "memory_recall_recall",
        },
        "memory_irrelevant": {
            "type": "memory_noise",
            "description": "Model retrieves irrelevant or low-value memories",
            "threshold": 0.5,
            "metric": "memory_recall_precision",
        },
        "context_drift": {
            "type": "context_alignment",
            "description": "Generated responses drift from conversation context",
            "threshold": 0.6,
            "metric": "context_relevance",
        },
        "reflection_absent": {
            "type": "reflection_quality",
            "description": "Lacks reflective or insightful content",
            "threshold": 0.5,
            "metric": "reflection_quality",
        },
        "generation_incoherent": {
            "type": "generation_quality",
            "description": "Generated text is incoherent or low quality",
            "threshold": 0.6,
            "metric": "generation_quality",
        },
    }

    def __init__(self):
        self.patterns: list[FailurePattern] = []

    def parse(
        self,
        evaluation_results: dict[str, Any],
    ) -> list[FailurePattern]:
        """
        Parse evaluation results and identify failure patterns.

        Args:
            evaluation_results: Evaluation report data

        Returns:
            List of identified failure patterns
        """
        self.patterns = []

        # Extract metrics
        memory_metrics = evaluation_results.get("memory_metrics", {})
        quality_metrics = evaluation_results.get("quality_metrics", {})

        # Check each pattern template
        self._check_pattern(
            "memory_recall_low",
            memory_metrics.get("avg_recall_recall", 1.0),
            evaluation_results,
        )

        self._check_pattern(
            "memory_irrelevant",
            memory_metrics.get("avg_recall_precision", 1.0),
            evaluation_results,
        )

        self._check_pattern(
            "context_drift",
            quality_metrics.get("avg_context_relevance", 1.0),
            evaluation_results,
        )

        self._check_pattern(
            "reflection_absent",
            quality_metrics.get("avg_reflection_quality", 1.0),
            evaluation_results,
        )

        self._check_pattern(
            "generation_incoherent",
            quality_metrics.get("avg_generation_quality", 1.0),
            evaluation_results,
        )

        return self.patterns

    def _check_pattern(
        self,
        pattern_key: str,
        actual_value: float,
        evaluation_results: dict[str, Any],
    ) -> None:
        """Check if a failure pattern is present."""
        template = self.PATTERN_TEMPLATES.get(pattern_key)
        if not template:
            return

        threshold = template["threshold"]

        # If below threshold, pattern is present
        if actual_value < threshold:
            pattern = FailurePattern(
                pattern_id=f"pattern_{pattern_key}",
                pattern_type=template["type"],
                description=template["description"],
                affected_examples=self._extract_affected_examples(
                    pattern_key, evaluation_results
                ),
                severity=self._determine_severity(actual_value, threshold),
                frequency=(threshold - actual_value) / threshold,
                metrics_impacted=[template["metric"]],
            )
            self.patterns.append(pattern)

    def _extract_affected_examples(
        self,
        pattern_key: str,
        evaluation_results: dict[str, Any],
    ) -> list[str]:
        """Extract example IDs affected by this pattern."""
        # In a full implementation, this would parse detailed results
        # For now, return placeholder
        return []

    def _determine_severity(
        self,
        actual: float,
        threshold: float,
    ) -> str:
        """Determine severity based on gap from threshold."""
        if actual < threshold * 0.5:
            return "critical"
        if actual < threshold * 0.7:
            return "high"
        if actual < threshold * 0.9:
            return "medium"
        return "low"


class UpstreamCauseMapper:
    """Maps failure patterns to likely upstream causes."""

    # Mapping rules
    MAPPING_RULES = {
        "memory_deficiency": {
            "domain": UpstreamDomain.ACQUISITION,
            "hypothesis": "Source data lacks high-quality memory-context pairs",
            "evidence_sources": ["acquisition_logs", "source_metadata"],
        },
        "memory_noise": {
            "domain": UpstreamDomain.CURATION,
            "hypothesis": "Curation rules allow low-relevance memories through",
            "evidence_sources": ["curation_rules", "dedup_logs"],
        },
        "context_alignment": {
            "domain": UpstreamDomain.CURATION,
            "hypothesis": "Context normalization insufficient or misaligned",
            "evidence_sources": ["normalization_config", "context_samples"],
        },
        "reflection_quality": {
            "domain": UpstreamDomain.REVIEW,
            "hypothesis": "Human review needed for reflection quality gate",
            "evidence_sources": ["review_guidelines", "quality_samples"],
        },
        "generation_quality": {
            "domain": UpstreamDomain.ACQUISITION,
            "hypothesis": "Training data quality floor too low",
            "evidence_sources": ["quality_thresholds", "source_rankings"],
        },
        "privacy_concern": {
            "domain": UpstreamDomain.PRIVACY,
            "hypothesis": "Privacy gate missed PII or sensitive patterns",
            "evidence_sources": ["privacy_audit", "pii_patterns"],
        },
    }

    def map(
        self,
        failure_patterns: list[FailurePattern],
    ) -> list[UpstreamMapping]:
        """
        Map failure patterns to upstream causes.

        Args:
            failure_patterns: List of identified failure patterns

        Returns:
            List of upstream mappings with confidence scores
        """
        mappings = []

        for pattern in failure_patterns:
            rule = self.MAPPING_RULES.get(pattern.pattern_type, {})

            mapping = UpstreamMapping(
                failure_pattern=pattern,
                likely_upstream_domain=rule.get(
                    "domain", UpstreamDomain.CURATION
                ),
                confidence=self._calculate_confidence(pattern),
                root_cause_hypothesis=rule.get(
                    "hypothesis", "Unknown upstream cause"
                ),
                evidence=rule.get("evidence_sources", []),
            )
            mappings.append(mapping)

        return mappings

    def _calculate_confidence(self, pattern: FailurePattern) -> float:
        """Calculate confidence in upstream mapping."""
        # Higher confidence for severe, frequent patterns
        severity_weight = {
            "critical": 1.0,
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4,
        }

        base_confidence = severity_weight.get(pattern.severity, 0.5)
        frequency_boost = pattern.frequency * 0.2

        return min(1.0, base_confidence + frequency_boost)


class InterventionGenerator:
    """Generates concrete interventions from upstream mappings."""

    # Intervention templates
    INTERVENTION_TEMPLATES = {
        UpstreamDomain.ACQUISITION: [
            {
                "type": InterventionType.PRIORITY_CHANGE,
                "title": "Adjust source data priorities",
                "description_template": "Increase priority of high-{quality_type} sources",
            },
            {
                "type": InterventionType.THRESHOLD_CHANGE,
                "title": "Update quality thresholds",
                "description_template": "Raise {metric} threshold from {current} to {target}",
            },
        ],
        UpstreamDomain.CURATION: [
            {
                "type": InterventionType.RULE_CHANGE,
                "title": "Update curation rules",
                "description_template": "Add curation rule for {pattern_type}",
            },
            {
                "type": InterventionType.DATASET_FILTER,
                "title": "Add dataset filter",
                "description_template": "Filter {pattern_type} patterns from dataset",
            },
        ],
        UpstreamDomain.PRIVACY: [
            {
                "type": InterventionType.VALIDATION_GATE,
                "title": "Add privacy validation gate",
                "description_template": "Add gate for {pattern_type} detection",
            },
        ],
        UpstreamDomain.REVIEW: [
            {
                "type": InterventionType.REVIEW_FOCUS,
                "title": "Add human review focus area",
                "description_template": "Focus human review on {pattern_type}",
            },
        ],
    }

    def generate(
        self,
        mappings: list[UpstreamMapping],
    ) -> list[DatasetIntervention]:
        """
        Generate interventions from upstream mappings.

        Args:
            mappings: List of upstream mappings

        Returns:
            List of dataset interventions
        """
        interventions = []

        for mapping in mappings:
            domain = mapping.likely_upstream_domain
            templates = self.INTERVENTION_TEMPLATES.get(domain, [])

            for template in templates[:2]:  # Top 2 interventions per mapping
                intervention = self._create_intervention(
                    template, mapping, domain
                )
                interventions.append(intervention)

        return interventions

    def _create_intervention(
        self,
        template: dict[str, Any],
        mapping: UpstreamMapping,
        domain: UpstreamDomain,
    ) -> DatasetIntervention:
        """Create a single intervention."""
        pattern = mapping.failure_pattern

        # Fill in template
        description = template["description_template"].format(
            pattern_type=pattern.pattern_type,
            quality_type=pattern.pattern_type.split("_")[-1],
            metric=pattern.metrics_impacted[0] if pattern.metrics_impacted else "quality",
            current=f"{pattern.frequency:.2f}",
            target=f"{pattern.frequency + 0.2:.2f}",
        )

        return DatasetIntervention(
            intervention_id=f"intervention_{mapping.failure_pattern.pattern_id}",
            intervention_type=template["type"],
            title=template["title"],
            description=description,
            upstream_domain=domain,
            priority=mapping.failure_pattern.severity,
            expected_impact=f"Improve {pattern.metrics_impacted[0]} by 10-20%",
            implementation_details={
                "root_cause": mapping.root_cause_hypothesis,
                "confidence": mapping.confidence,
                "evidence": mapping.evidence,
            },
            validation_criteria=[
                f"{pattern.metrics_impacted[0]} improves by >10%",
                "Pattern frequency reduces by >50%",
            ],
            related_patterns=[pattern.pattern_id],
        )


class FeedbackLoop:
    """
    Main feedback loop orchestrator.

    Connects evaluation results to upstream actions.
    """

    def __init__(self):
        self.parser = EvaluationParser()
        self.mapper = UpstreamCauseMapper()
        self.generator = InterventionGenerator()

    def run(
        self,
        evaluation_report_path: str | Path,
        output_dir: str | Path,
    ) -> FeedbackReport:
        """
        Run complete feedback loop.

        Args:
            evaluation_report_path: Path to evaluation report JSON
            output_dir: Directory for output files

        Returns:
            Complete feedback report
        """
        evaluation_report_path = Path(evaluation_report_path)
        output_dir = Path(output_dir)

        # Load evaluation results
        with open(evaluation_report_path, encoding="utf-8") as f:
            evaluation_results = json.load(f)

        # Parse failure patterns
        failure_patterns = self.parser.parse(evaluation_results)

        # Map to upstream causes
        upstream_mappings = self.mapper.map(failure_patterns)

        # Generate interventions
        interventions = self.generator.generate(upstream_mappings)

        # Create report
        overall_score = evaluation_results.get("overall_score", 0.0)
        report = FeedbackReport(
            evaluation_source=str(evaluation_report_path),
            generated_at=datetime.now(UTC).isoformat(),
            total_evaluated=evaluation_results.get("evaluated_examples", 0),
            overall_score=overall_score,
            failure_patterns=failure_patterns,
            upstream_mappings=upstream_mappings,
            interventions=interventions,
            critical_issues=sum(
                1 for p in failure_patterns if p.severity == "critical"
            ),
            high_priority_issues=sum(
                1 for p in failure_patterns if p.severity == "high"
            ),
            recommended_actions=len(interventions),
        )

        # Save report
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "feedback_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        # Generate Linear issues for interventions
        self._create_linear_issues(interventions, output_dir)

        logger.info(
            f"Feedback report saved to {report_path} "
            f"with {len(interventions)} interventions"
        )

        return report

    def _create_linear_issues(
        self,
        interventions: list[DatasetIntervention],
        output_dir: Path,
    ) -> None:
        """Create Linear issue templates for interventions."""
        issues_path = output_dir / "linear_issues"
        issues_path.mkdir(exist_ok=True)

        for intervention in interventions:
            issue_file = issues_path / f"{intervention.intervention_id}.md"

            content = self._format_linear_issue(intervention)
            with open(issue_file, "w", encoding="utf-8") as f:
                f.write(content)

    def _format_linear_issue(
        self,
        intervention: DatasetIntervention,
    ) -> str:
        """Format intervention as Linear issue template."""
        return f"""# {intervention.title}

**Type**: {intervention.intervention_type.value}
**Upstream Domain**: {intervention.upstream_domain.value}
**Priority**: {intervention.priority}

## Description
{intervention.description}

## Root Cause
{intervention.implementation_details.get('root_cause', 'Unknown')}

## Expected Impact
{intervention.expected_impact}

## Implementation Details
{json.dumps(intervention.implementation_details, indent=2)}

## Validation Criteria
{chr(10).join(f'- [ ] {c}' for c in intervention.validation_criteria)}

## Related Patterns
{', '.join(intervention.related_patterns)}

---
*Generated by feedback loop automation*
"""


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run evaluation-to-data feedback loop"
    )

    parser.add_argument(
        "--evaluation-report",
        type=str,
        required=True,
        help="Path to evaluation report JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./feedback/actions",
        help="Output directory for feedback actions",
    )

    args = parser.parse_args()

    loop = FeedbackLoop()
    report = loop.run(args.evaluation_report, args.output_dir)

    # Print summary

    if report.interventions:
        for _i, _intervention in enumerate(report.interventions[:5], 1):
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
