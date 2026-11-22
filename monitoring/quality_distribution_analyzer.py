"""Distribution analysis for quality scores.

Descriptive statistics alone hide the shape of a metric: whether it is normal,
whether it is skewed, and where the tails sit. This module reports the full
picture and runs the three normality tests the quality reports cite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .quality_metrics_store import load_quality_frame

OUTLIER_IQR_FACTOR = 1.5


@dataclass(frozen=True)
class DistributionStatistics:
    """Descriptive shape of one score series."""

    count: int
    mean: float
    std_dev: float
    min_value: float
    max_value: float
    q1: float
    median: float
    q3: float
    skewness: float
    kurtosis: float
    iqr: float


@dataclass(frozen=True)
class NormalityTest:
    """One normality test's verdict."""

    test_name: str
    statistic: float
    p_value: float
    is_normal: bool
    interpretation: str


@dataclass(frozen=True)
class OutlierRecord:
    """A single value outside the IQR fence."""

    index: int
    value: float
    direction: str
    distance_from_median: float


@dataclass(frozen=True)
class DistributionAnalysis:
    """Everything the distribution report needs for one metric."""

    metric: str
    sample_size: int
    statistics: DistributionStatistics
    distribution_type: str = "unknown"
    normality_tests: list[NormalityTest] = field(default_factory=list)
    histogram: dict[str, int] = field(default_factory=dict)
    outliers: list[OutlierRecord] = field(default_factory=list)


def _classify(statistics: DistributionStatistics, normality_tests: list[NormalityTest]) -> str:
    """Name the shape implied by the skew, kurtosis and normality verdicts."""
    if statistics.count == 0:
        return "empty"
    normal_verdicts = [test.is_normal for test in normality_tests]
    if normal_verdicts and all(normal_verdicts) and abs(statistics.skewness) < 0.5:
        return "normal"
    if statistics.skewness >= 0.5:
        return "skewed_right"
    if statistics.skewness <= -0.5:
        return "skewed_left"
    if statistics.kurtosis <= -1.0:
        return "flat"
    if statistics.kurtosis >= 1.0:
        return "heavy_tailed"
    return "non_normal"


def _statistics(values: np.ndarray) -> DistributionStatistics:
    quartiles = np.percentile(values, [25, 50, 75])
    return DistributionStatistics(
        count=int(values.size),
        mean=float(values.mean()),
        std_dev=float(values.std(ddof=1)) if values.size > 1 else 0.0,
        min_value=float(values.min()),
        max_value=float(values.max()),
        q1=float(quartiles[0]),
        median=float(quartiles[1]),
        q3=float(quartiles[2]),
        skewness=float(scipy_stats.skew(values)) if values.size > 2 else 0.0,
        kurtosis=float(scipy_stats.kurtosis(values)) if values.size > 3 else 0.0,
        iqr=float(quartiles[2] - quartiles[0]),
    )


def _normality_tests(values: np.ndarray) -> list[NormalityTest]:
    tests: list[NormalityTest] = []
    if values.size < 3:
        return tests

    shapiro = scipy_stats.shapiro(values)
    tests.append(
        NormalityTest(
            test_name="shapiro_wilk",
            statistic=float(shapiro.statistic),
            p_value=float(shapiro.pvalue),
            is_normal=bool(shapiro.pvalue >= 0.05),
            interpretation="Shapiro-Wilk rejects normality at p < 0.05"
            if shapiro.pvalue < 0.05
            else "Shapiro-Wilk finds no evidence against normality",
        )
    )

    if values.size >= 8:
        dagostino = scipy_stats.normaltest(values)
        tests.append(
            NormalityTest(
                test_name="dagostino_pearson",
                statistic=float(dagostino.statistic),
                p_value=float(dagostino.pvalue),
                is_normal=bool(dagostino.pvalue >= 0.05),
                interpretation="Skew and kurtosis differ from a normal distribution"
                if dagostino.pvalue < 0.05
                else "Skew and kurtosis match a normal distribution",
            )
        )

    standardised = (values - values.mean()) / values.std(ddof=1) if values.std(ddof=1) else values * 0
    kolmogorov = scipy_stats.kstest(standardised, "norm")
    tests.append(
        NormalityTest(
            test_name="kolmogorov_smirnov",
            statistic=float(kolmogorov.statistic),
            p_value=float(kolmogorov.pvalue),
            is_normal=bool(kolmogorov.pvalue >= 0.05),
            interpretation="Empirical CDF departs from the normal CDF"
            if kolmogorov.pvalue < 0.05
            else "Empirical CDF matches the normal CDF",
        )
    )
    return tests


class QualityDistributionAnalyzer:
    """Describe how quality scores are distributed."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else None

    def load_quality_data(self, days_back: int | None = 30) -> pd.DataFrame:
        """Load the conversations-to-metrics frame."""
        if self.db_path is None:
            raise ValueError("QualityDistributionAnalyzer needs a db_path to load data")
        return load_quality_frame(self.db_path, days_back=days_back)

    def analyze_quality_distribution(
        self, values: pd.Series, metric: str = "overall_quality", bins: int = 20
    ) -> DistributionAnalysis:
        """Describe one score series, including its normality verdicts."""
        cleaned = pd.Series(values).dropna().to_numpy(dtype=float)
        if cleaned.size == 0:
            empty = DistributionStatistics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            return DistributionAnalysis(metric=metric, sample_size=0, statistics=empty)

        counts, edges = np.histogram(cleaned, bins=bins, range=(0.0, 1.0))
        histogram = {f"{edges[index]:.2f}-{edges[index + 1]:.2f}": int(count) for index, count in enumerate(counts)}
        statistics = _statistics(cleaned)
        normality_tests = _normality_tests(cleaned)
        return DistributionAnalysis(
            metric=metric,
            sample_size=int(cleaned.size),
            statistics=statistics,
            distribution_type=_classify(statistics, normality_tests),
            normality_tests=normality_tests,
            histogram=histogram,
            outliers=self.detect_outliers(cleaned),
        )

    def detect_outliers(
        self, values: pd.Series | np.ndarray, factor: float = OUTLIER_IQR_FACTOR
    ) -> list[OutlierRecord]:
        """Flag values outside the median +/- factor * IQR fence."""
        cleaned = pd.Series(values).dropna()
        if cleaned.size < 4:
            return []
        q1, median, q3 = cleaned.quantile([0.25, 0.5, 0.75]).to_numpy(dtype=float)
        spread = q3 - q1
        if spread == 0:
            return []
        lower = q1 - factor * spread
        upper = q3 + factor * spread
        outliers: list[OutlierRecord] = []
        for index, value in cleaned.items():
            if value < lower or value > upper:
                outliers.append(
                    OutlierRecord(
                        index=int(index) if isinstance(index, (int, np.integer)) else -1,
                        value=float(value),
                        direction="low" if value < lower else "high",
                        distance_from_median=float(abs(value - median)),
                    )
                )
        return outliers
