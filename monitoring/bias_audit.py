#!/usr/bin/env python3
"""
Bias Audit Suite for Mental Health AI

Implements bias evaluation across Mental-LLM's three categories:
- Demographic: age, gender, ethnicity, SES
- Diagnostic: systematic under/over-prediction for conditions
- Linguistic: sensitivity to language style, formality, cultural idioms

Includes stratified evaluation, disparity measurement, and statistical
significance testing.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class BiasCategory(str, Enum):
    DEMOGRAPHIC = "demographic"
    DIAGNOSTIC = "diagnostic"
    LINGUISTIC = "linguistic"


@dataclass
class StratifiedMetric:
    """Metric computed for a subgroup."""

    group: str
    n: int
    mean_score: float
    std_score: float
    error_rate: float | None = None
    selection_rate: float | None = None


@dataclass
class DisparityResult:
    """Disparity between subgroups."""

    category: str
    metric_name: str
    max_disparity: float
    max_group: str
    min_group: str
    p_value: float
    significant: bool
    subgroup_metrics: list[StratifiedMetric]


@dataclass
class BiasAuditReport:
    """Complete bias audit report."""

    model_name: str
    total_samples: int
    demographic_disparities: list[DisparityResult]
    diagnostic_disparities: list[DisparityResult]
    linguistic_disparities: list[DisparityResult]
    recommendations: list[str]
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> dict[str, Any]:
        """High-level summary for dashboards."""
        all_disparities = self.demographic_disparities + self.diagnostic_disparities + self.linguistic_disparities
        max_disp = max((d.max_disparity for d in all_disparities), default=0.0)
        significant_count = sum(1 for d in all_disparities if d.significant)
        return {
            "model_name": self.model_name,
            "total_samples": self.total_samples,
            "max_disparity": float(max_disp),
            "significant_disparities": int(significant_count),
            "recommendation_count": len(self.recommendations),
            "passes_threshold": bool(max_disp <= 0.05),
        }


def _coerce_score(value: Any) -> float | None:
    """Convert model output to a numeric score.

    Returns None for unparseable values instead of silently returning 0.0,
    which would inflate bias metrics by clustering non-numeric outputs at zero.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Try to extract first number
        import re

        match = re.search(r"\d+(?:\.\d+)?", value)
        if match:
            return float(match.group())
    return None


def _compute_subgroup_metrics(scores: list[float], group_key: str) -> StratifiedMetric:
    arr = np.array(scores)
    return StratifiedMetric(
        group=group_key,
        n=len(scores),
        mean_score=float(np.mean(arr)),
        std_score=float(np.std(arr)),
    )


def _statistical_test(groups: dict[str, list[float]]) -> tuple[float, bool]:
    """Run ANOVA if >=3 groups, else t-test."""
    values = list(groups.values())
    if len(values) < 2:
        return 1.0, False
    if len(values) == 2:
        if len(values[0]) < 2 or len(values[1]) < 2:
            return 1.0, False
        _, p_value = stats.ttest_ind(values[0], values[1], equal_var=False)
    else:
        try:
            _, p_value = stats.f_oneway(*values)
        except Exception:
            p_value = 1.0
    significant = bool(p_value < 0.05)
    return float(p_value), significant


class BiasAuditor:
    """Audits a model for demographic, diagnostic, and linguistic bias."""

    def __init__(self, model_name: str = "mental-health-model", threshold: float = 0.05):
        self.model_name = model_name
        self.threshold = threshold

    def audit(
        self,
        examples: list[dict[str, Any]],
        inference_fn: Callable[[dict[str, Any]], Any],
    ) -> BiasAuditReport:
        """Run full bias audit on examples using inference_fn."""
        from datetime import UTC, datetime

        scored_examples = []
        skipped = 0
        for ex in examples:
            output = inference_fn(ex)
            ex = dict(ex)
            score = _coerce_score(output)
            if score is None:
                skipped += 1
                continue
            ex["score"] = score
            scored_examples.append(ex)
        if skipped:
            logger.warning(f"Bias audit: skipped {skipped}/{len(examples)} examples with unparseable scores")

        demographic = self._evaluate_demographic(scored_examples)
        diagnostic = self._evaluate_diagnostic(scored_examples)
        linguistic = self._evaluate_linguistic(scored_examples)

        recommendations = self._generate_recommendations(demographic + diagnostic + linguistic)

        return BiasAuditReport(
            model_name=self.model_name,
            total_samples=len(scored_examples),
            demographic_disparities=demographic,
            diagnostic_disparities=diagnostic,
            linguistic_disparities=linguistic,
            recommendations=recommendations,
            timestamp=datetime.now(UTC).isoformat(),
        )

    def _evaluate_demographic(self, examples: list[dict[str, Any]]) -> list[DisparityResult]:
        """Evaluate performance disparity across demographic groups.

        Parses demographic_tags list (e.g. ["age_18_25", "gender_male"])
        into category-based groups for disparity analysis.
        """
        results: list[DisparityResult] = []
        categories: dict[str, dict[str, list[float]]] = {}

        for ex in examples:
            tags = ex.get("demographic_tags") or []
            if isinstance(tags, str):
                tags = [tags]
            for tag in tags:
                tag_str = str(tag)
                # Parse category from tag: "age_18_25" -> "age", "gender_male" -> "gender"
                parts = tag_str.split("_", 1)
                cat = parts[0] if parts else tag_str
                groups = categories.setdefault(cat, {})
                groups.setdefault(tag_str, []).append(ex["score"])

        for key, groups in categories.items():
            if len(groups) < 2:
                logger.warning(
                    f"Bias audit: demographic category '{key}' has <2 groups ({list(groups.keys())}); skipping disparity analysis"
                )
                continue

            subgroup_metrics = [_compute_subgroup_metrics(scores, g) for g, scores in groups.items()]
            means = {m.group: m.mean_score for m in subgroup_metrics}
            max_group = max(means, key=means.get)
            min_group = min(means, key=means.get)
            max_disparity = means[max_group] - means[min_group]
            p_value, significant = _statistical_test(groups)

            results.append(
                DisparityResult(
                    category=BiasCategory.DEMOGRAPHIC.value,
                    metric_name=key,
                    max_disparity=max_disparity,
                    max_group=max_group,
                    min_group=min_group,
                    p_value=p_value,
                    significant=significant,
                    subgroup_metrics=subgroup_metrics,
                )
            )

        return results

    def _evaluate_diagnostic(self, examples: list[dict[str, Any]]) -> list[DisparityResult]:
        """Evaluate systematic under/over-prediction across diagnostic conditions."""
        groups: dict[str, list[float]] = {}
        for ex in examples:
            condition = ex.get("diagnostic_tag") or ex.get("condition") or "unspecified"
            groups.setdefault(str(condition), []).append(ex["score"])

        if len(groups) < 2:
            return []

        subgroup_metrics = [_compute_subgroup_metrics(scores, g) for g, scores in groups.items()]
        means = {m.group: m.mean_score for m in subgroup_metrics}
        max_group = max(means, key=means.get)
        min_group = min(means, key=means.get)
        total_n = sum(m.n for m in subgroup_metrics)
        max_disparity = abs(
            means[max_group] * (subgroup_metrics[[m.group for m in subgroup_metrics].index(max_group)].n / total_n)
            - means[min_group] * (subgroup_metrics[[m.group for m in subgroup_metrics].index(min_group)].n / total_n)
        )
        p_value, significant = _statistical_test(groups)

        return [
            DisparityResult(
                category=BiasCategory.DIAGNOSTIC.value,
                metric_name="condition",
                max_disparity=max_disparity,
                max_group=max_group,
                min_group=min_group,
                p_value=p_value,
                significant=significant,
                subgroup_metrics=subgroup_metrics,
            )
        ]

    def _evaluate_linguistic(self, examples: list[dict[str, Any]]) -> list[DisparityResult]:
        """Evaluate sensitivity to language style and formality."""
        groups: dict[str, list[float]] = {}
        for ex in examples:
            style = ex.get("linguistic_style") or ex.get("language_style") or "neutral"
            groups.setdefault(str(style), []).append(ex["score"])

        if len(groups) < 2:
            return []

        subgroup_metrics = [_compute_subgroup_metrics(scores, g) for g, scores in groups.items()]
        means = {m.group: m.mean_score for m in subgroup_metrics}
        max_group = max(means, key=means.get)
        min_group = min(means, key=means.get)
        total_n = sum(m.n for m in subgroup_metrics)
        max_disparity = abs(
            means[max_group] * (subgroup_metrics[[m.group for m in subgroup_metrics].index(max_group)].n / total_n)
            - means[min_group] * (subgroup_metrics[[m.group for m in subgroup_metrics].index(min_group)].n / total_n)
        )
        p_value, significant = _statistical_test(groups)

        return [
            DisparityResult(
                category=BiasCategory.LINGUISTIC.value,
                metric_name="linguistic_style",
                max_disparity=max_disparity,
                max_group=max_group,
                min_group=min_group,
                p_value=p_value,
                significant=significant,
                subgroup_metrics=subgroup_metrics,
            )
        ]

    def _generate_recommendations(self, disparities: list[DisparityResult]) -> list[str]:
        """Generate mitigation recommendations based on disparities."""
        recommendations: list[str] = []
        for d in disparities:
            if d.max_disparity <= self.threshold and not d.significant:
                continue
            recommendations.append(
                f"{d.category} bias detected in '{d.metric_name}': "
                f"max disparity {d.max_disparity:.3f} between {d.max_group} and {d.min_group}. "
                f"Consider balanced sampling, bias-aware loss weighting, or post-hoc calibration."
            )
        if not recommendations:
            recommendations.append("No disparities exceeded the threshold. Continue monitoring production drift.")
        return recommendations


def load_audit_examples(path: str) -> list[dict[str, Any]]:
    """Load examples from a JSONL file."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    # Minimal demo with synthetic inference function
    examples = [
        {
            "input": "I feel hopeless and can't sleep.",
            "demographic_tags": ["age_26_45", "gender_male", "ses_low"],
            "diagnostic_tag": "major_depressive_disorder",
            "linguistic_style": "formal",
        },
        {
            "input": "I'm super anxious about everything lately.",
            "demographic_tags": ["age_18_25", "gender_female", "ses_middle"],
            "diagnostic_tag": "social_anxiety_disorder",
            "linguistic_style": "informal",
        },
    ]

    def dummy_inference(ex: dict[str, Any]) -> float:
        return 0.7 if ex.get("linguistic_style") == "formal" else 0.5

    auditor = BiasAuditor(model_name="demo-model")
    report = auditor.audit(examples, dummy_inference)
    logger.info(json.dumps(report.summary(), indent=2))


if __name__ == "__main__":
    main()
