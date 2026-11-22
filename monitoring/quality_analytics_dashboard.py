"""Quality analytics dashboard.

Aggregates the quality metrics of every validated conversation into the figures
a reviewer needs: how quality is distributed across the four bands, how each
tier performs, where the trend is heading, which conversations sit outside the
normal spread, and what to do about it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .quality_metrics_store import load_quality_frame

QUALITY_THRESHOLDS: dict[str, float] = {
    "excellent": 0.8,
    "good": 0.6,
    "fair": 0.4,
    "poor": 0.0,
}

QUALITY_LEVELS: tuple[str, ...] = ("Poor", "Fair", "Good", "Excellent")

COLOR_SCHEMES: dict[str, Any] = {
    "quality_levels": ["#7f7f88", "#9a9aa3", "#e8e6df", "#55d6e0"],
    "tiers": ["#55d6e0", "#e8e6df", "#9a9aa3", "#7f7f88", "#3a3a42"],
    "trends": {
        "improving": "#55d6e0",
        "stable": "#9a9aa3",
        "declining": "#e8e6df",
    },
    "anomalies": {"marker": "#e8e6df", "baseline": "#3a3a42"},
}

OUTLIER_IQR_FACTOR = 1.5
MIN_SAMPLES_FOR_ANOMALIES = 4
MIN_TREND_POINTS = 2
TREND_DELTA = 0.02
DEFAULT_CACHE_DURATION = 300
NO_DATA_MESSAGE = "No quality data available for analysis"


@dataclass
class QualityAnalytics:
    """One computed analytics snapshot."""

    total_conversations: int = 0
    average_quality: float = 0.0
    quality_distribution: dict[str, int] = field(default_factory=dict)
    tier_performance: dict[str, float] = field(default_factory=dict)
    trend_data: list[dict[str, Any]] = field(default_factory=list)
    anomalies: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_conversations": self.total_conversations,
            "average_quality": self.average_quality,
            "quality_distribution": dict(self.quality_distribution),
            "tier_performance": dict(self.tier_performance),
            "trend_data": list(self.trend_data),
            "anomalies": list(self.anomalies),
            "recommendations": list(self.recommendations),
        }


def classify_quality(score: float, thresholds: dict[str, float] | None = None) -> str:
    """Place one score in a quality band."""
    bands = thresholds or QUALITY_THRESHOLDS
    for level in reversed(QUALITY_LEVELS):
        if score >= bands[level.lower()]:
            return level
    return "Poor"


class QualityAnalyticsDashboard:
    """Load quality metrics and turn them into analytics plus charts."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        cache_duration: int = DEFAULT_CACHE_DURATION,
    ) -> None:
        self.db_path = db_path if db_path is None or isinstance(db_path, str) else str(db_path)
        self.cache_duration = cache_duration
        self.quality_thresholds = dict(QUALITY_THRESHOLDS)
        self.color_schemes = {
            key: list(value) if isinstance(value, list) else dict(value) for key, value in COLOR_SCHEMES.items()
        }
        self._cache: pd.DataFrame | None = None
        self._last_cache_time: float = 0.0

    def load_quality_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """Load the conversation/metric frame, honouring the cache window.

        The dashboard reports at day granularity, so ``created_at`` is normalised
        to midnight of the conversation's day. Every chart and every band count
        below is a daily aggregate, and a time-of-day component would make
        date-range filtering disagree with the daily trend.
        """
        if (
            not force_refresh
            and self._cache is not None
            and (time.time() - self._last_cache_time) < self.cache_duration
        ):
            return self._cache

        if self.db_path is None:
            frame = pd.DataFrame()
        else:
            frame = load_quality_frame(self.db_path, days_back=None)
            if not frame.empty:
                frame["created_at"] = frame["created_at"].dt.normalize()

        self._cache = frame
        self._last_cache_time = time.time()
        return frame

    def calculate_quality_analytics(self, frame: pd.DataFrame) -> QualityAnalytics:
        """Compute the distribution, tier means, trend, anomalies and advice."""
        if frame is None or frame.empty or "overall_quality" not in frame.columns:
            return QualityAnalytics(recommendations=[NO_DATA_MESSAGE])

        scores = frame["overall_quality"].dropna()
        if scores.empty:
            return QualityAnalytics(recommendations=[NO_DATA_MESSAGE])

        distribution = dict.fromkeys(QUALITY_LEVELS, 0)
        for score in scores:
            distribution[classify_quality(float(score), self.quality_thresholds)] += 1

        tier_performance: dict[str, float] = {}
        if "tier" in frame.columns:
            for tier, series in frame.groupby("tier")["overall_quality"]:
                values = series.dropna()
                if not values.empty:
                    tier_performance[str(tier)] = float(values.mean())

        trend_data = self._trend_data(frame)
        anomalies = self._detect_anomalies(frame)

        analytics = QualityAnalytics(
            total_conversations=int(scores.size),
            average_quality=float(scores.mean()),
            quality_distribution=distribution,
            tier_performance=tier_performance,
            trend_data=trend_data,
            anomalies=anomalies,
        )
        analytics.recommendations = self._recommendations(analytics)
        return analytics

    def _trend_data(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        if "created_at" not in frame.columns:
            return []
        dated = frame.dropna(subset=["created_at", "overall_quality"]).copy()
        if dated.empty:
            return []
        dated["_day"] = pd.to_datetime(dated["created_at"], utc=True).dt.date
        daily = dated.groupby("_day")["overall_quality"].agg(["mean", "count"]).sort_index()
        return [
            {"date": day.isoformat(), "quality": float(row["mean"]), "count": int(row["count"])}
            for day, row in daily.iterrows()
        ]

    def _detect_anomalies(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        scores = frame["overall_quality"].dropna()
        if scores.size < MIN_SAMPLES_FOR_ANOMALIES:
            return []
        q1, _, q3 = scores.quantile([0.25, 0.5, 0.75]).to_numpy(dtype=float)
        spread = q3 - q1
        if spread == 0:
            return []
        lower = q1 - OUTLIER_IQR_FACTOR * spread
        upper = q3 + OUTLIER_IQR_FACTOR * spread

        anomalies: list[dict[str, Any]] = []
        identifier_column = "id" if "id" in frame.columns else "conversation_id"
        for _, row in frame.iterrows():
            score = row.get("overall_quality")
            if score is None or pd.isna(score):
                continue
            if lower <= float(score) <= upper:
                continue
            created = row.get("created_at")
            raw_id = row.get(identifier_column) if identifier_column in frame.columns else None
            anomalies.append(
                {
                    "id": int(raw_id) if raw_id is not None and pd.notna(raw_id) else -1,
                    "dataset": str(row.get("dataset_name", "")),
                    "tier": str(row.get("tier", "")),
                    "quality": float(score),
                    "date": pd.Timestamp(created).isoformat() if pd.notna(created) else "",
                    "direction": "low" if float(score) < lower else "high",
                }
            )
        anomalies.sort(key=lambda anomaly: anomaly["quality"])
        return anomalies

    def _recommendations(self, analytics: QualityAnalytics) -> list[str]:
        items: list[str] = []
        total = analytics.total_conversations
        poor = analytics.quality_distribution.get("Poor", 0)
        excellent = analytics.quality_distribution.get("Excellent", 0)

        if poor:
            items.append(
                f"{poor} of {total} conversation(s) scored below "
                f"{self.quality_thresholds['fair']:.2f}; review them before the next training run."
            )
        if excellent:
            items.append(
                f"{excellent} of {total} conversation(s) reached the excellent band "
                f"({self.quality_thresholds['excellent']:.2f}+); use them as few-shot exemplars."
            )
        weakest = sorted(analytics.tier_performance.items(), key=lambda item: item[1])[:1]
        if weakest and weakest[0][1] < self.quality_thresholds["good"]:
            tier, score = weakest[0]
            items.append(
                f"Tier '{tier}' averages {score:.3f}, under the good threshold "
                f"{self.quality_thresholds['good']:.2f}; it needs the most attention."
            )
        if analytics.anomalies:
            items.append(
                f"{len(analytics.anomalies)} conversation(s) fall outside the interquartile fence; "
                "inspect them for label or generation defects."
            )
        if len(analytics.trend_data) >= MIN_TREND_POINTS:
            first = analytics.trend_data[0]["quality"]
            last = analytics.trend_data[-1]["quality"]
            delta = last - first
            if delta > TREND_DELTA:
                items.append(f"Daily mean quality rose {delta:+.3f} across the window; the trend is improving.")
            elif delta < -TREND_DELTA:
                items.append(f"Daily mean quality fell {delta:+.3f} across the window; investigate the regression.")
        if not items:
            items.append(f"Quality is stable at {analytics.average_quality:.3f}; no action is required this cycle.")
        return items

    def create_quality_overview_chart(self, analytics: QualityAnalytics) -> Any:
        """Band distribution plus tier means as one two-row figure."""

        figure = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Quality band distribution", "Mean quality by tier"),
        )
        levels = [level for level in QUALITY_LEVELS if level in analytics.quality_distribution]
        figure.add_trace(
            go.Bar(
                x=levels or ["No data"],
                y=[analytics.quality_distribution.get(level, 0) for level in levels] or [0],
                marker_color=self.color_schemes["quality_levels"][: max(len(levels), 1)],
                name="conversations",
            ),
            row=1,
            col=1,
        )
        tiers = list(analytics.tier_performance)
        figure.add_trace(
            go.Bar(
                x=tiers or ["No data"],
                y=[analytics.tier_performance[tier] for tier in tiers] or [0],
                marker_color=self.color_schemes["tiers"][: max(len(tiers), 1)],
                name="mean quality",
            ),
            row=1,
            col=2,
        )
        figure.update_yaxes(range=[0, 1], row=1, col=2)
        figure.update_layout(
            title=f"Quality overview - {analytics.total_conversations} conversations "
            f"(mean {analytics.average_quality:.3f})",
            showlegend=False,
        )
        return figure

    def create_detailed_quality_heatmap(self, frame: pd.DataFrame) -> Any:
        """Component-by-tier heatmap of mean scores."""

        components = [
            "therapeutic_accuracy",
            "conversation_coherence",
            "emotional_authenticity",
            "clinical_compliance",
            "personality_consistency",
            "language_quality",
            "safety_score",
        ]
        present = [component for component in components if component in frame.columns]
        if frame is None or frame.empty or not present or "tier" not in frame.columns:
            figure = go.Figure()
            figure.update_layout(title="Component quality heatmap - no data")
            return figure

        grouped = frame.groupby("tier")[present].mean()
        figure = go.Figure(
            go.Heatmap(
                z=grouped.to_numpy(dtype=float).tolist(),
                x=[component.replace("_", " ") for component in present],
                y=[str(tier) for tier in grouped.index],
                zmin=0,
                zmax=1,
                colorscale=[[0.0, "#0b0b0c"], [0.5, "#3a3a42"], [1.0, "#e8e6df"]],
            )
        )
        figure.update_layout(title="Mean component score by tier")
        return figure

    def create_anomaly_detection_chart(self, analytics: QualityAnalytics) -> Any:
        """Trend line with the anomalous conversations marked."""

        figure = go.Figure()
        if not analytics.trend_data and not analytics.anomalies:
            figure.update_layout(title="Anomaly detection - no data")
            return figure

        if analytics.trend_data:
            figure.add_trace(
                go.Scatter(
                    x=[point["date"] for point in analytics.trend_data],
                    y=[point["quality"] for point in analytics.trend_data],
                    mode="lines+markers",
                    name="daily mean",
                    line={"color": self.color_schemes["trends"]["stable"]},
                )
            )
        if analytics.anomalies:
            figure.add_trace(
                go.Scatter(
                    x=[anomaly["date"] for anomaly in analytics.anomalies],
                    y=[anomaly["quality"] for anomaly in analytics.anomalies],
                    mode="markers",
                    name="anomaly",
                    marker={
                        "size": 12,
                        "symbol": "x",
                        "color": self.color_schemes["anomalies"]["marker"],
                    },
                    text=[f"{anomaly['tier']} / {anomaly['dataset']}" for anomaly in analytics.anomalies],
                )
            )
        figure.update_layout(title="Quality anomalies", yaxis_range=[0, 1])
        return figure

    def summary(self, analytics: QualityAnalytics) -> dict[str, Any]:
        """Compact dictionary for embedding in a report."""
        return {
            "total_conversations": analytics.total_conversations,
            "average_quality": round(analytics.average_quality, 4),
            "bands": dict(analytics.quality_distribution),
            "tiers": {tier: round(score, 4) for tier, score in analytics.tier_performance.items()},
            "anomalies": len(analytics.anomalies),
            "recommendations": list(analytics.recommendations),
        }


def compute_band_counts(scores: np.ndarray, thresholds: dict[str, float] | None = None) -> dict[str, int]:
    """Count scores per quality band, keeping all four bands present."""
    bands = dict.fromkeys(QUALITY_LEVELS, 0)
    for score in scores:
        bands[classify_quality(float(score), thresholds)] += 1
    return bands
