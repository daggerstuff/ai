"""Trend analysis over historical quality scores.

Reads the same conversations-to-metrics frame the comparison modules use and
answers four questions: which way quality is moving, where it is heading,
which periods break the pattern, and whether there is a repeating seasonal
shape. Regressions are ordinary least squares; predictions carry a 95%
prediction interval rather than a bare point estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .quality_metrics_store import load_quality_frame

DEFAULT_FREQUENCY = "W"
MIN_PERIODS_FOR_TREND = 3
SIGNIFICANCE_ALPHA = 0.05
MIN_OBSERVATIONS_PER_PERIOD = 2


@dataclass(frozen=True)
class TrendPoint:
    """One aggregated period in the series."""

    period: str
    mean: float
    count: int


@dataclass(frozen=True)
class TrendAnalysis:
    """Direction and strength of the overall movement."""

    direction: str
    slope: float
    r_squared: float
    p_value: float
    significant: bool
    start_mean: float
    end_mean: float
    change: float
    points: list[TrendPoint] = field(default_factory=list)


@dataclass(frozen=True)
class Prediction:
    """A forecast period with its interval."""

    period: str
    predicted: float
    lower: float
    upper: float


@dataclass(frozen=True)
class Anomaly:
    """A period that departs from the fitted trend."""

    period: str
    value: float
    expected: float
    residual: float
    z_score: float
    direction: str


@dataclass(frozen=True)
class SeasonalPattern:
    """A repeating shape, grouped by weekday or calendar month."""

    period_type: str
    pattern: dict[str, float]
    strength: float
    peak: str
    trough: str


def _aggregate(frame: pd.DataFrame, freq: str, metric: str) -> pd.Series:
    series = frame.set_index("created_at")[metric].dropna().sort_index()
    if series.empty:
        return series
    return series.resample(freq).mean().dropna()


class QualityTrendAnalyzer:
    """Analyse how quality scores move over time."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else None

    def load_historical_data(self, days_back: int | None = 180, metric: str = "overall_quality") -> pd.DataFrame:
        """Load the historical frame, keeping only rows with the metric present."""
        if self.db_path is None:
            raise ValueError("QualityTrendAnalyzer needs a db_path to load data")
        frame = load_quality_frame(self.db_path, days_back=days_back)
        return frame.dropna(subset=[metric])

    def analyze_overall_trend(
        self, frame: pd.DataFrame, freq: str = DEFAULT_FREQUENCY, metric: str = "overall_quality"
    ) -> TrendAnalysis:
        """Fit a least-squares line through the period means."""
        series = _aggregate(frame, freq, metric)
        if series.empty:
            return TrendAnalysis("insufficient_data", 0.0, 0.0, 1.0, False, 0.0, 0.0, 0.0, [])

        points = [TrendPoint(str(period.date()), float(value), 1) for period, value in series.items()]
        if series.size < MIN_PERIODS_FOR_TREND:
            change = float(series.iloc[-1] - series.iloc[0]) if series.size > 1 else 0.0
            return TrendAnalysis(
                direction="insufficient_data",
                slope=0.0,
                r_squared=0.0,
                p_value=1.0,
                significant=False,
                start_mean=float(series.iloc[0]),
                end_mean=float(series.iloc[-1]),
                change=change,
                points=points,
            )

        positions = np.arange(series.size, dtype=float)
        values = series.to_numpy(dtype=float)
        regression = scipy_stats.linregress(positions, values)
        slope = float(regression.slope)
        p_value = float(regression.pvalue) if np.isfinite(regression.pvalue) else 1.0
        direction = "improving" if slope > 0 else "declining" if slope < 0 else "stable"
        return TrendAnalysis(
            direction=direction,
            slope=slope,
            r_squared=float(regression.rvalue**2),
            p_value=p_value,
            significant=p_value < SIGNIFICANCE_ALPHA,
            start_mean=float(values[0]),
            end_mean=float(values[-1]),
            change=float(values[-1] - values[0]),
            points=points,
        )

    def generate_predictions(
        self,
        frame: pd.DataFrame,
        periods: int = 4,
        freq: str = DEFAULT_FREQUENCY,
        metric: str = "overall_quality",
        confidence: float = 0.95,
    ) -> list[Prediction]:
        """Extrapolate the fitted line and attach a prediction interval."""
        series = _aggregate(frame, freq, metric)
        if series.size < MIN_PERIODS_FOR_TREND:
            return []

        positions = np.arange(series.size, dtype=float)
        values = series.to_numpy(dtype=float)
        regression = scipy_stats.linregress(positions, values)

        residuals = values - (regression.intercept + regression.slope * positions)
        degrees_of_freedom = max(series.size - 2, 1)
        residual_error = float(np.sqrt(np.sum(residuals**2) / degrees_of_freedom))
        critical = float(scipy_stats.t.ppf((1 + confidence) / 2, degrees_of_freedom))
        mean_position = float(positions.mean())
        spread = float(np.sum((positions - mean_position) ** 2))

        step = series.index.to_series().diff().dropna()
        offset = step.median() if not step.empty else pd.Timedelta(days=7)
        predictions: list[Prediction] = []
        for horizon in range(1, periods + 1):
            position = float(series.size - 1 + horizon)
            predicted = float(regression.intercept + regression.slope * position)
            standard_error = residual_error * np.sqrt(1 + 1 / series.size + ((position - mean_position) ** 2) / spread)
            margin = critical * standard_error
            predictions.append(
                Prediction(
                    period=str((series.index[-1] + offset * horizon).date()),
                    predicted=float(np.clip(predicted, 0.0, 1.0)),
                    lower=float(np.clip(predicted - margin, 0.0, 1.0)),
                    upper=float(np.clip(predicted + margin, 0.0, 1.0)),
                )
            )
        return predictions

    def detect_anomalies(
        self,
        frame: pd.DataFrame,
        threshold: float = 3.0,
        freq: str = DEFAULT_FREQUENCY,
        metric: str = "overall_quality",
    ) -> list[Anomaly]:
        """Flag periods whose residual from the fitted line exceeds ``threshold`` sigma."""
        series = _aggregate(frame, freq, metric)
        if series.size < MIN_PERIODS_FOR_TREND:
            return []

        positions = np.arange(series.size, dtype=float)
        values = series.to_numpy(dtype=float)
        regression = scipy_stats.linregress(positions, values)
        expected = regression.intercept + regression.slope * positions
        residuals = values - expected
        deviation = float(residuals.std(ddof=1)) if residuals.size > 1 else 0.0
        if deviation == 0:
            return []

        anomalies: list[Anomaly] = []
        for index, (period, value) in enumerate(series.items()):
            z_score = float(residuals[index] / deviation)
            if abs(z_score) < threshold:
                continue
            anomalies.append(
                Anomaly(
                    period=str(period.date()),
                    value=float(value),
                    expected=float(expected[index]),
                    residual=float(residuals[index]),
                    z_score=z_score,
                    direction="above" if z_score > 0 else "below",
                )
            )
        return anomalies

    def detect_seasonal_patterns(
        self, frame: pd.DataFrame, metric: str = "overall_quality", min_strength: float = 0.1
    ) -> list[SeasonalPattern]:
        """Compare weekday and calendar-month means against the overall mean."""
        patterns: list[SeasonalPattern] = []
        if metric not in frame.columns:
            return patterns
        overall = float(frame[metric].dropna().mean())
        if not np.isfinite(overall) or overall == 0:
            return patterns

        for period_type, extractor in (
            ("weekday", lambda index: index.day_name()),
            ("month", lambda index: index.month_name()),
        ):
            working = frame.dropna(subset=[metric, "created_at"]).copy()
            if working.empty:
                continue
            working["period"] = extractor(working["created_at"].dt)
            grouped = working.groupby("period")[metric].agg(["mean", "count"])
            grouped = grouped[grouped["count"] >= MIN_OBSERVATIONS_PER_PERIOD]
            if grouped.empty:
                continue
            means = {str(name): float(row["mean"]) for name, row in grouped.iterrows()}
            deviations = {name: value - overall for name, value in means.items()}
            strength = max(abs(value) for value in deviations.values())
            if strength < min_strength:
                continue
            peak = max(means, key=means.get)
            trough = min(means, key=means.get)
            patterns.append(
                SeasonalPattern(
                    period_type=period_type,
                    pattern=means,
                    strength=float(strength),
                    peak=peak,
                    trough=trough,
                )
            )
        return patterns

    def summarize(self, frame: pd.DataFrame, metric: str = "overall_quality") -> dict[str, Any]:
        """Compact numeric summary used by the reporter's executive summary."""
        series = frame[metric].dropna()
        if series.empty:
            return {"count": 0}
        return {
            "count": int(series.size),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std(ddof=1)) if series.size > 1 else 0.0,
            "first": float(series.iloc[0]),
            "last": float(series.iloc[-1]),
        }
