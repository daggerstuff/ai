"""Statistical comparison of quality scores across tiers, datasets and components.

The comparator answers three questions the quality reports ask: do tiers differ,
do datasets differ, and which component carries the most weight. Every answer is
paired with a real test rather than a bare mean difference: Welch's t-test for
the p-value, Cohen's d for effect size, and a bootstrap-free 95% interval on the
difference of means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .quality_metrics_store import COMPONENT_COLUMNS, SCORE_COLUMNS, load_quality_frame

DEFAULT_TARGET_QUALITY = 0.80
SIGNIFICANCE_LEVEL = 0.05


@dataclass(frozen=True)
class GroupStatistics:
    """Descriptive statistics for one group of scores."""

    label: str
    count: int
    mean: float
    std: float
    median: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class TierComparison:
    """Pairwise difference between two tiers on one metric."""

    metric: str
    tier_a: str
    tier_b: str
    stats_a: GroupStatistics
    stats_b: GroupStatistics
    difference: float
    effect_size: float
    p_value: float
    significant: bool
    confidence_interval: tuple[float, float]


@dataclass(frozen=True)
class DatasetComparison:
    """Pairwise difference between two datasets on one metric."""

    metric: str
    dataset_a: str
    dataset_b: str
    stats_a: GroupStatistics
    stats_b: GroupStatistics
    difference: float
    effect_size: float
    statistical_tests: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ComponentComparison:
    """How one quality component behaves across the whole frame."""

    component: str
    mean: float
    std: float
    median: float
    rank: int
    contribution: float


@dataclass(frozen=True)
class BenchmarkAnalysis:
    """One group's distance from the clinical target."""

    group_type: str
    group: str
    benchmark_name: str
    benchmark_value: float
    observed_mean: float
    performance_gap: float
    gap_ratio: float
    status: str


def _group_statistics(series: pd.Series, label: str) -> GroupStatistics:
    values = series.dropna().to_numpy(dtype=float)
    if values.size == 0:
        return GroupStatistics(label, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return GroupStatistics(
        label=label,
        count=int(values.size),
        mean=float(values.mean()),
        std=float(values.std(ddof=1)) if values.size > 1 else 0.0,
        median=float(np.median(values)),
        minimum=float(values.min()),
        maximum=float(values.max()),
    )


def _cohens_d(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """Pooled-standard-deviation effect size; 0.0 when a group is degenerate."""
    if sample_a.size < 2 or sample_b.size < 2:
        return 0.0
    pooled_variance = ((sample_a.size - 1) * sample_a.var(ddof=1) + (sample_b.size - 1) * sample_b.var(ddof=1)) / (
        sample_a.size + sample_b.size - 2
    )
    if pooled_variance <= 0:
        return 0.0
    return float((sample_a.mean() - sample_b.mean()) / np.sqrt(pooled_variance))


def _difference_interval(sample_a: np.ndarray, sample_b: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """Welch interval on the difference of means; (0, 0) when undefined."""
    if sample_a.size < 2 or sample_b.size < 2:
        return (0.0, 0.0)
    difference = sample_a.mean() - sample_b.mean()
    variance_a = sample_a.var(ddof=1) / sample_a.size
    variance_b = sample_b.var(ddof=1) / sample_b.size
    standard_error = np.sqrt(variance_a + variance_b)
    if standard_error == 0:
        return (float(difference), float(difference))
    numerator = (variance_a + variance_b) ** 2
    denominator = (variance_a**2) / (sample_a.size - 1) + (variance_b**2) / (sample_b.size - 1)
    degrees_of_freedom = numerator / denominator if denominator else sample_a.size + sample_b.size - 2
    critical = scipy_stats.t.ppf((1 + confidence) / 2, degrees_of_freedom)
    margin = critical * standard_error
    return (float(difference - margin), float(difference + margin))


def _p_value(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    if sample_a.size < 2 or sample_b.size < 2:
        return 1.0
    result = scipy_stats.ttest_ind(sample_a, sample_b, equal_var=False, nan_policy="omit")
    statistic = float(result.pvalue)
    return statistic if np.isfinite(statistic) else 1.0


class QualityComparator:
    """Compare quality scores between tiers and datasets, and rank components."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else None

    def load_comparison_data(self, days_back: int | None = 30) -> pd.DataFrame:
        """Load the conversations-to-metrics frame used by every comparison."""
        if self.db_path is None:
            raise ValueError("QualityComparator needs a db_path to load data")
        return load_quality_frame(self.db_path, days_back=days_back)

    def compare_tiers(self, frame: pd.DataFrame, metric: str = "overall_quality") -> list[TierComparison]:
        """Pairwise tier comparison on one metric."""
        comparisons: list[TierComparison] = []
        tiers = sorted(frame["tier"].dropna().unique())
        for tier_a, tier_b in combinations(tiers, 2):
            sample_a = frame.loc[frame["tier"] == tier_a, metric].dropna().to_numpy(dtype=float)
            sample_b = frame.loc[frame["tier"] == tier_b, metric].dropna().to_numpy(dtype=float)
            if sample_a.size == 0 or sample_b.size == 0:
                continue
            p_value = _p_value(sample_a, sample_b)
            comparisons.append(
                TierComparison(
                    metric=metric,
                    tier_a=str(tier_a),
                    tier_b=str(tier_b),
                    stats_a=_group_statistics(pd.Series(sample_a), str(tier_a)),
                    stats_b=_group_statistics(pd.Series(sample_b), str(tier_b)),
                    difference=float(sample_a.mean() - sample_b.mean()),
                    effect_size=_cohens_d(sample_a, sample_b),
                    p_value=p_value,
                    significant=p_value < SIGNIFICANCE_LEVEL,
                    confidence_interval=_difference_interval(sample_a, sample_b),
                )
            )
        return comparisons

    def compare_datasets(self, frame: pd.DataFrame, metric: str = "overall_quality") -> list[DatasetComparison]:
        """Pairwise dataset comparison on one metric, with both tests attached."""
        comparisons: list[DatasetComparison] = []
        datasets = sorted(frame["dataset_name"].dropna().unique())
        for dataset_a, dataset_b in combinations(datasets, 2):
            sample_a = frame.loc[frame["dataset_name"] == dataset_a, metric].dropna().to_numpy(dtype=float)
            sample_b = frame.loc[frame["dataset_name"] == dataset_b, metric].dropna().to_numpy(dtype=float)
            if sample_a.size == 0 or sample_b.size == 0:
                continue
            welch = scipy_stats.ttest_ind(sample_a, sample_b, equal_var=False, nan_policy="omit")
            mann_whitney = scipy_stats.mannwhitneyu(sample_a, sample_b, alternative="two-sided")
            comparisons.append(
                DatasetComparison(
                    metric=metric,
                    dataset_a=str(dataset_a),
                    dataset_b=str(dataset_b),
                    stats_a=_group_statistics(pd.Series(sample_a), str(dataset_a)),
                    stats_b=_group_statistics(pd.Series(sample_b), str(dataset_b)),
                    difference=float(sample_a.mean() - sample_b.mean()),
                    effect_size=_cohens_d(sample_a, sample_b),
                    statistical_tests={
                        "welch_t": {
                            "statistic": float(welch.statistic),
                            "p_value": float(welch.pvalue) if np.isfinite(welch.pvalue) else 1.0,
                        },
                        "mann_whitney_u": {
                            "statistic": float(mann_whitney.statistic),
                            "p_value": float(mann_whitney.pvalue) if np.isfinite(mann_whitney.pvalue) else 1.0,
                        },
                    },
                )
            )
        return comparisons

    def compare_components(self, frame: pd.DataFrame) -> list[ComponentComparison]:
        """Rank the quality components by mean score and share of the total."""
        means = {column: float(frame[column].dropna().mean()) for column in COMPONENT_COLUMNS if column in frame}
        means = {column: value for column, value in means.items() if np.isfinite(value)}
        if not means:
            return []
        total = sum(means.values())
        ordered = sorted(means.items(), key=lambda item: item[1], reverse=True)
        comparisons: list[ComponentComparison] = []
        for rank, (component, mean) in enumerate(ordered, start=1):
            series = frame[component].dropna()
            comparisons.append(
                ComponentComparison(
                    component=component,
                    mean=mean,
                    std=float(series.std(ddof=1)) if series.size > 1 else 0.0,
                    median=float(series.median()),
                    rank=rank,
                    contribution=float(mean / total) if total else 0.0,
                )
            )
        return comparisons

    def perform_benchmark_analysis(
        self, frame: pd.DataFrame, target: float = DEFAULT_TARGET_QUALITY
    ) -> list[BenchmarkAnalysis]:
        """Measure each tier and dataset against the clinical target."""
        analyses: list[BenchmarkAnalysis] = []
        for group_type, column in (("tier", "tier"), ("dataset", "dataset_name")):
            for group, series in frame.groupby(column)["overall_quality"]:
                observed = float(series.dropna().mean())
                if not np.isfinite(observed):
                    continue
                gap = observed - target
                analyses.append(
                    BenchmarkAnalysis(
                        group_type=group_type,
                        group=str(group),
                        benchmark_name="clinical_target",
                        benchmark_value=target,
                        observed_mean=observed,
                        performance_gap=gap,
                        gap_ratio=float(gap / target) if target else 0.0,
                        status="meets_target" if gap >= 0 else "below_target",
                    )
                )
        return analyses

    def calculate_performance_rankings(self, frame: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
        """Rank tiers, datasets and components by mean overall quality."""

        def _rank(column: str, metric: str = "overall_quality") -> list[dict[str, Any]]:
            grouped = frame.groupby(column)[metric].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
            return [
                {
                    "rank": rank,
                    "name": str(name),
                    "mean": float(row["mean"]),
                    "std": float(row["std"]) if np.isfinite(row["std"]) else 0.0,
                    "count": int(row["count"]),
                }
                for rank, (name, row) in enumerate(grouped.iterrows(), start=1)
            ]

        component_rankings = [
            {"rank": comparison.rank, "name": comparison.component, "mean": comparison.mean, "std": comparison.std}
            for comparison in self.compare_components(frame)
        ]
        return {
            "tiers": _rank("tier"),
            "datasets": _rank("dataset_name"),
            "components": component_rankings,
        }

    def score_columns(self) -> tuple[str, ...]:
        """The metrics a comparison can be run against."""
        return SCORE_COLUMNS
