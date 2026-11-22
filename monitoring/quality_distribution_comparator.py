"""Group comparisons and correlations for the quality distribution analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .quality_distribution_analyzer import QualityDistributionAnalyzer
from .quality_metrics_store import COMPONENT_COLUMNS

MIN_GROUP_SAMPLES = 2
SIGNIFICANCE_ALPHA = 0.05
STRONG_CORRELATION = 0.7


@dataclass(frozen=True)
class GroupDistribution:
    """One group's share of the distribution."""

    name: str
    count: int
    mean: float
    std_dev: float
    median: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class DistributionComparison:
    """Groups plus the tests that separate them."""

    dimension: str
    metric: str
    groups: list[GroupDistribution] = field(default_factory=list)
    statistical_tests: list[dict[str, Any]] = field(default_factory=list)
    variance_test: dict[str, Any] = field(default_factory=dict)


class QualityDistributionComparator:
    """Compare distributions across tiers, components and metric pairs."""

    def __init__(self, analyzer: QualityDistributionAnalyzer | None = None) -> None:
        self.analyzer = analyzer or QualityDistributionAnalyzer()

    def _group(self, name: str, series: pd.Series) -> GroupDistribution:
        values = series.dropna().to_numpy(dtype=float)
        if values.size == 0:
            return GroupDistribution(name, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return GroupDistribution(
            name=name,
            count=int(values.size),
            mean=float(values.mean()),
            std_dev=float(values.std(ddof=1)) if values.size > 1 else 0.0,
            median=float(np.median(values)),
            minimum=float(values.min()),
            maximum=float(values.max()),
        )

    def compare_across_tiers(self, frame: pd.DataFrame, metric: str = "overall_quality") -> DistributionComparison:
        """Group one metric by tier and test every pair."""
        groups = [
            self._group(str(tier), series) for tier, series in frame.groupby("tier")[metric] if series.dropna().size > 0
        ]
        tests: list[dict[str, Any]] = []
        for group_a, group_b in combinations(groups, 2):
            sample_a = frame.loc[frame["tier"] == group_a.name, metric].dropna().to_numpy(dtype=float)
            sample_b = frame.loc[frame["tier"] == group_b.name, metric].dropna().to_numpy(dtype=float)
            if sample_a.size < MIN_GROUP_SAMPLES or sample_b.size < MIN_GROUP_SAMPLES:
                continue
            welch = scipy_stats.ttest_ind(sample_a, sample_b, equal_var=False, nan_policy="omit")
            levene = scipy_stats.levene(sample_a, sample_b)
            tests.append(
                {
                    "group_a": group_a.name,
                    "group_b": group_b.name,
                    "difference": float(sample_a.mean() - sample_b.mean()),
                    "welch_t": float(welch.statistic),
                    "p_value": float(welch.pvalue) if np.isfinite(welch.pvalue) else 1.0,
                    "significant": bool(np.isfinite(welch.pvalue) and welch.pvalue < SIGNIFICANCE_ALPHA),
                    "levene_statistic": float(levene.statistic),
                    "levene_p_value": float(levene.pvalue) if np.isfinite(levene.pvalue) else 1.0,
                }
            )

        variance_test: dict[str, Any] = {}
        if len(groups) >= MIN_GROUP_SAMPLES:
            samples = [
                frame.loc[frame["tier"] == group.name, metric].dropna().to_numpy(dtype=float) for group in groups
            ]
            samples = [sample for sample in samples if sample.size >= MIN_GROUP_SAMPLES]
            if len(samples) >= MIN_GROUP_SAMPLES:
                bartlett = scipy_stats.bartlett(*samples)
                variance_test = {
                    "test": "bartlett",
                    "statistic": float(bartlett.statistic),
                    "p_value": float(bartlett.pvalue) if np.isfinite(bartlett.pvalue) else 1.0,
                    "equal_variance": bool(np.isfinite(bartlett.pvalue) and bartlett.pvalue >= SIGNIFICANCE_ALPHA),
                }

        return DistributionComparison(
            dimension="tier", metric=metric, groups=groups, statistical_tests=tests, variance_test=variance_test
        )

    def compare_across_components(self, frame: pd.DataFrame) -> DistributionComparison:
        """Compare the seven quality components against each other."""
        groups = [
            self._group(component, frame[component])
            for component in COMPONENT_COLUMNS
            if component in frame.columns and frame[component].dropna().size > 0
        ]
        samples = [
            frame[group.name].dropna().to_numpy(dtype=float)
            for group in groups
            if frame[group.name].dropna().size >= MIN_GROUP_SAMPLES
        ]
        tests: list[dict[str, Any]] = []
        if len(samples) >= MIN_GROUP_SAMPLES:
            anova = scipy_stats.f_oneway(*samples)
            kruskal = scipy_stats.kruskal(*samples)
            tests.append(
                {
                    "test": "one_way_anova",
                    "statistic": float(anova.statistic),
                    "p_value": float(anova.pvalue) if np.isfinite(anova.pvalue) else 1.0,
                    "significant": bool(np.isfinite(anova.pvalue) and anova.pvalue < SIGNIFICANCE_ALPHA),
                }
            )
            tests.append(
                {
                    "test": "kruskal_wallis",
                    "statistic": float(kruskal.statistic),
                    "p_value": float(kruskal.pvalue) if np.isfinite(kruskal.pvalue) else 1.0,
                    "significant": bool(np.isfinite(kruskal.pvalue) and kruskal.pvalue < SIGNIFICANCE_ALPHA),
                }
            )
        return DistributionComparison(
            dimension="component", metric="overall_quality", groups=groups, statistical_tests=tests
        )

    def calculate_correlation_analysis(self, frame: pd.DataFrame, min_abs_correlation: float = 0.3) -> dict[str, Any]:
        """Correlate every metric pair and surface the strongest relationships."""
        columns = [column for column in (*COMPONENT_COLUMNS, "overall_quality") if column in frame.columns]
        if len(columns) < MIN_GROUP_SAMPLES:
            return {"correlation_matrix": {}, "strongest_correlations": []}

        matrix = frame[columns].corr(method="pearson")
        strongest: list[dict[str, Any]] = []
        for index, column_a in enumerate(columns):
            for column_b in columns[index + 1 :]:
                value = matrix.loc[column_a, column_b]
                if not np.isfinite(value) or abs(value) < min_abs_correlation:
                    continue
                strongest.append(
                    {
                        "metric_a": column_a,
                        "metric_b": column_b,
                        "correlation": float(value),
                        "strength": "strong" if abs(value) >= STRONG_CORRELATION else "moderate",
                        "direction": "positive" if value > 0 else "negative",
                    }
                )
        strongest.sort(key=lambda entry: abs(entry["correlation"]), reverse=True)
        return {
            "correlation_matrix": {
                row: {column: float(matrix.loc[row, column]) for column in columns} for row in columns
            },
            "strongest_correlations": strongest,
        }
