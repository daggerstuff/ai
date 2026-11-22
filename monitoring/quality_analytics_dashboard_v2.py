"""Quality analytics dashboard (v2) over the conversations + conversation_quality schema.

The historic implementation rendered through Streamlit, which is not a declared
dependency of this repository, so the module could not be imported. This version
keeps the analytics — the quality bands, tier and component performance, trend,
anomaly detection and recommendations — and replaces the Streamlit page with a
self-contained HTML renderer.

Unlike the v1 dashboard, which reads the ``quality_metrics`` table, this one reads
the ``conversations`` / ``conversation_quality`` pair: one quality row per
conversation, validated against ``validator_version``.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from string import Template
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "database" / "conversations.db"

CACHE_TTL_SECONDS = 300

TARGET_QUALITY = 0.6

TIER_GAP = 0.05

COMPONENT_FLOOR = 0.7

MIN_ANOMALY_SAMPLES = 3

DATE_RANGE_BOUNDS = 2

BASELINE_RECOMMENDATIONS = 2

Z_SCORE_THRESHOLD = 2.0

IQR_FENCE = 1.5

TREND_WINDOW_DAYS = 30

QUALITY_BINS = [0.0, 0.4, 0.6, 0.8, 1.0]

QUALITY_LABELS = ["Poor", "Fair", "Good", "Excellent"]

QUALITY_COMPONENTS = {
    "overall_quality": "Overall Quality",
    "therapeutic_accuracy": "Therapeutic Accuracy",
    "clinical_compliance": "Clinical Compliance",
    "safety_score": "Safety Score",
    "conversation_coherence": "Conversation Coherence",
    "emotional_authenticity": "Emotional Authenticity",
}

QUALITY_TABLE_COLUMNS = frozenset(
    {
        "quality_id",
        "overall_quality",
        "therapeutic_accuracy",
        "clinical_compliance",
        "safety_score",
        "conversation_coherence",
        "emotional_authenticity",
        "validation_date",
        "validator_version",
    }
)

QUERY_COLUMNS = (
    "conversation_id",
    "dataset_source",
    "tier",
    "title",
    "turn_count",
    "word_count",
    "processing_status",
    "created_at",
    "quality_id",
    "overall_quality",
    "therapeutic_accuracy",
    "clinical_compliance",
    "safety_score",
    "conversation_coherence",
    "emotional_authenticity",
    "validation_date",
    "validator_version",
)

COLOR_SCHEMES = {
    "quality_levels": ["#d32f2f", "#f57c00", "#fbc02d", "#388e3c"],
    "tiers": ["#1976d2", "#7b1fa2", "#d32f2f", "#f57c00", "#388e3c", "#795548"],
    "trends": "#1976d2",
    "anomalies": "#d32f2f",
}

EMPTY_RECOMMENDATION = "No quality data available for analysis"

PAGE = Template(
    """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quality Analytics</title>
<style>
  :root {
    --ground: #0b0b0c;
    --panel: #121214;
    --raised: #18181b;
    --ink: #e8e6df;
    --muted: #8b8b94;
    --rule: #1c1c21;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--ground);
    color: var(--ink);
    font-family: "Instrument Sans Variable", system-ui, sans-serif;
    line-height: 1.55;
  }
  main { max-width: 1120px; margin: 0 auto; padding: 40px 24px 72px; }
  h1 { font-size: clamp(1.75rem, 3vw, 2.5rem); font-weight: 800; letter-spacing: 0.01em; margin: 0 0 8px; }
  h2 {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.6875rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--muted);
    font-weight: 500;
    margin: 40px 0 12px;
  }
  .lede { color: var(--muted); margin: 0 0 32px; max-width: 66ch; }
  .metrics {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1px;
    background: var(--rule);
  }
  .metric { background: var(--panel); padding: 16px 20px; }
  .metric .label {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.625rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--muted);
  }
  .metric .value { font-size: 1.75rem; font-weight: 700; font-variant-numeric: tabular-nums; }
  .metric .note { color: var(--muted); font-size: 0.8125rem; }
  .figure { background: var(--panel); border: 1px solid var(--rule); padding: 8px; }
  ul { padding-left: 20px; }
  li { margin-bottom: 6px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
  th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--rule); }
  th {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.625rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    font-weight: 500;
  }
  td { font-variant-numeric: tabular-nums; }
  footer {
    margin-top: 40px;
    padding-top: 16px;
    border-top: 1px solid var(--rule);
    color: var(--muted);
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.75rem;
  }
</style>
</head>
<body>
<main>
  <h1>Quality Analytics</h1>
  <p class="lede">Conversation quality across tiers and components, with the trend,
  the statistical outliers and the follow-ups they imply.</p>
  <section class="metrics">${metrics}</section>
  <h2>Distribution and tiers</h2>
  <div class="figure">${overview}</div>
  <h2>Component profile</h2>
  <div class="figure">${heatmap}</div>
  <div class="figure">${correlation}</div>
  <h2>Recommended follow-ups</h2>
  ${recommendations}
  <h2>Tier performance</h2>
  ${tiers}
  <footer>Generated ${generated} from ${source} &middot; ${records} conversations &middot; data ${freshness}</footer>
</main>
</body>
</html>
"""
)


@dataclass
class QualityAnalytics:
    """Everything the dashboard reports about one snapshot of quality data."""

    total_conversations: int
    average_quality: float
    quality_distribution: dict[str, int]
    tier_performance: dict[str, dict[str, Any]]
    component_performance: dict[str, dict[str, float]]
    trend_data: list[dict[str, Any]]
    anomalies: list[dict[str, Any]]
    recommendations: list[str]
    data_freshness: str
    analysis_timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class QualityAnalyticsDashboard:
    """Read conversation quality from SQLite and summarise it."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        if not self.db_path.exists():
            raise FileNotFoundError(f"quality database not found: {self.db_path}")
        self.cache_duration = CACHE_TTL_SECONDS
        self.quality_thresholds = {"excellent": 0.8, "good": 0.6, "fair": 0.4, "poor": 0.0}
        self.quality_components = dict(QUALITY_COMPONENTS)
        self.color_schemes = {
            name: list(value) if isinstance(value, list) else value for name, value in COLOR_SCHEMES.items()
        }
        self._cache: dict[str, pd.DataFrame] = {}
        self._cache_timestamps: dict[str, float] = {}

    def _cache_key(
        self,
        tier_filter: list[str] | None,
        date_range: tuple[datetime, datetime] | None,
        min_quality: float | None,
    ) -> str:
        bounds = None
        if date_range is not None and len(date_range) == DATE_RANGE_BOUNDS:
            bounds = (date_range[0].date().isoformat(), date_range[1].date().isoformat())
        return repr((tuple(sorted(tier_filter)) if tier_filter else None, bounds, min_quality))

    def load_quality_data(
        self,
        tier_filter: list[str] | None = None,
        date_range: tuple[datetime, datetime] | None = None,
        min_quality: float | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Return one row per conversation joined to its quality record.

        Args:
            tier_filter: Tiers to keep; ``None`` or a list holding ``"All"`` keeps
                every tier.
            date_range: Inclusive ``(start, end)`` day window on ``created_at``.
            min_quality: Drop conversations whose overall quality is lower.
            force_refresh: Bypass the cached frame for this call.

        Returns:
            A frame ordered newest first, with the query columns plus
            ``quality_category``, ``date``, ``month`` and ``week``.
        """
        key = self._cache_key(tier_filter, date_range, min_quality)
        if not force_refresh:
            cached = self._cache.get(key)
            stamped = self._cache_timestamps.get(key)
            if cached is not None and stamped is not None and time.monotonic() - stamped < self.cache_duration:
                return cached.copy()

        frame = self._read(tier_filter, date_range, min_quality)
        self._cache[key] = frame
        self._cache_timestamps[key] = time.monotonic()
        return frame.copy()

    def _read(
        self,
        tier_filter: list[str] | None,
        date_range: tuple[datetime, datetime] | None,
        min_quality: float | None,
    ) -> pd.DataFrame:
        selected = ", ".join(f"{'q' if column in QUALITY_TABLE_COLUMNS else 'c'}.{column}" for column in QUERY_COLUMNS)
        query = f"""
            SELECT {selected}
            FROM conversations c
            INNER JOIN conversation_quality q ON q.conversation_id = c.conversation_id
            WHERE 1 = 1
        """
        parameters: list[Any] = []
        if tier_filter and "All" not in tier_filter:
            query += f" AND c.tier IN ({', '.join('?' for _ in tier_filter)})"
            parameters.extend(tier_filter)
        if date_range is not None and len(date_range) == DATE_RANGE_BOUNDS:
            query += " AND DATE(c.created_at) >= DATE(?) AND DATE(c.created_at) <= DATE(?)"
            parameters.extend([date_range[0].date().isoformat(), date_range[1].date().isoformat()])
        if min_quality is not None:
            query += " AND q.overall_quality >= ?"
            parameters.append(min_quality)
        query += " ORDER BY c.created_at DESC"

        try:
            with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True) as conn:
                frame = pd.read_sql_query(query, conn, params=parameters)
        except sqlite3.Error, pd.errors.DatabaseError:
            return pd.DataFrame(columns=list(QUERY_COLUMNS))

        if frame.empty:
            return pd.DataFrame(columns=[*QUERY_COLUMNS, "quality_category", "date", "month", "week"])

        frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True, errors="coerce")
        frame["validation_date"] = pd.to_datetime(frame["validation_date"], utc=True, errors="coerce")
        frame["overall_quality"] = pd.to_numeric(frame["overall_quality"], errors="coerce")
        frame["quality_category"] = pd.cut(
            frame["overall_quality"], bins=QUALITY_BINS, labels=QUALITY_LABELS, include_lowest=True
        )
        frame["date"] = frame["created_at"].dt.date
        calendar = frame["created_at"].dt.tz_convert("UTC").dt.tz_localize(None)
        frame["month"] = calendar.dt.to_period("M")
        frame["week"] = calendar.dt.to_period("W")
        return frame

    def calculate_quality_analytics(self, df: pd.DataFrame) -> QualityAnalytics:
        """Summarise one snapshot of quality data.

        Returns the empty report when the frame carries no usable quality column,
        so a schema mismatch degrades to "no data" instead of raising.
        """
        if df is None or df.empty or "overall_quality" not in df.columns:
            return self._empty_analytics()

        frame = df.copy()
        frame["overall_quality"] = pd.to_numeric(frame["overall_quality"], errors="coerce")
        frame = frame.dropna(subset=["overall_quality"])
        if frame.empty:
            return self._empty_analytics()

        average = float(frame["overall_quality"].mean())
        tier_performance = self._tier_performance(frame)
        component_performance = self._component_performance(frame)
        return QualityAnalytics(
            total_conversations=len(frame),
            average_quality=average,
            quality_distribution=self._quality_distribution(frame),
            tier_performance=tier_performance,
            component_performance=component_performance,
            trend_data=self._trend_data(frame),
            anomalies=self._detect_quality_anomalies(frame),
            recommendations=self._generate_recommendations(frame, average, tier_performance, component_performance),
            data_freshness=self._data_freshness(frame),
        )

    def _empty_analytics(self) -> QualityAnalytics:
        return QualityAnalytics(
            total_conversations=0,
            average_quality=0.0,
            quality_distribution={},
            tier_performance={},
            component_performance={},
            trend_data=[],
            anomalies=[],
            recommendations=[EMPTY_RECOMMENDATION],
            data_freshness="No data",
        )

    def _quality_distribution(self, frame: pd.DataFrame) -> dict[str, int]:
        if "quality_category" in frame.columns:
            counts = frame["quality_category"].value_counts()
        else:
            counts = pd.cut(
                frame["overall_quality"], bins=QUALITY_BINS, labels=QUALITY_LABELS, include_lowest=True
            ).value_counts()
        return {str(band): int(count) for band, count in counts.items()}

    def _tier_performance(self, frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
        if "tier" not in frame.columns:
            return {}
        grouped = frame.groupby("tier", observed=True)["overall_quality"].agg(["mean", "count"])
        return {
            str(tier): {"average_quality": float(row["mean"]), "conversation_count": int(row["count"])}
            for tier, row in grouped.iterrows()
        }

    def _component_performance(self, frame: pd.DataFrame) -> dict[str, dict[str, float]]:
        total = len(frame)
        performance: dict[str, dict[str, float]] = {}
        for column, name in QUALITY_COMPONENTS.items():
            if column not in frame.columns:
                continue
            scores = pd.to_numeric(frame[column], errors="coerce").dropna()
            scores = scores[scores > 0]
            performance[name] = {
                "average_score": float(scores.mean()) if len(scores) else 0.0,
                "sample_count": float(len(scores)),
                "coverage_percent": float(len(scores) / total * 100) if total else 0.0,
            }
        return performance

    def _trend_data(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        if "created_at" not in frame.columns:
            return []
        stamps = pd.to_datetime(frame["created_at"], utc=True, errors="coerce")
        recent = frame[stamps >= datetime.now(UTC) - timedelta(days=TREND_WINDOW_DAYS)]
        if recent.empty:
            return []
        daily = recent.groupby(recent["created_at"].dt.date, observed=True)["overall_quality"].agg(["mean", "count"])
        trend = [
            {"date": str(day), "average_quality": float(row["mean"]), "conversation_count": int(row["count"])}
            for day, row in daily.iterrows()
        ]
        return sorted(trend, key=lambda point: point["date"])

    def _data_freshness(self, frame: pd.DataFrame) -> str:
        if "created_at" not in frame.columns:
            return "No data"
        latest = pd.to_datetime(frame["created_at"], utc=True, errors="coerce").max()
        if pd.isna(latest):
            return "No data"
        age = datetime.now(UTC) - pd.Timestamp(latest)
        if age.days == 0:
            return "Current (today)"
        if age.days == 1:
            return "1 day old"
        return f"{age.days} days old"

    def _detect_quality_anomalies(self, df: pd.DataFrame, method: str = "iqr") -> list[dict[str, Any]]:
        """Flag conversations whose overall quality is a statistical outlier."""
        if df is None or df.empty or "overall_quality" not in df.columns or len(df) < MIN_ANOMALY_SAMPLES:
            return []

        frame = df.copy()
        frame["overall_quality"] = pd.to_numeric(frame["overall_quality"], errors="coerce")
        frame = frame.dropna(subset=["overall_quality"])
        if len(frame) < MIN_ANOMALY_SAMPLES:
            return []

        if method == "zscore":
            scores = frame["overall_quality"]
            deviation = float(scores.std(ddof=1))
            if deviation == 0:
                return []
            distance = (scores - float(scores.mean())).abs() / deviation
            outliers = frame[distance > Z_SCORE_THRESHOLD]
            anomaly_type = "zscore_outlier"
        else:
            low = float(frame["overall_quality"].quantile(0.25))
            high = float(frame["overall_quality"].quantile(0.75))
            spread = high - low
            if spread == 0:
                return []
            fence = spread * IQR_FENCE
            outliers = frame[(frame["overall_quality"] < low - fence) | (frame["overall_quality"] > high + fence)]
            anomaly_type = "iqr_outlier"

        return [
            {
                "conversation_id": record["conversation_id"],
                "tier": record.get("tier"),
                "quality_score": float(record["overall_quality"]),
                "anomaly_type": anomaly_type,
            }
            for record in outliers.to_dict("records")
        ]

    def _generate_recommendations(
        self,
        frame: pd.DataFrame,
        average: float,
        tier_performance: dict[str, dict[str, Any]],
        component_performance: dict[str, dict[str, float]],
    ) -> list[str]:
        band = self._band(average)
        recommendations = [f"Overall quality averages {average:.3f} across {len(frame)} conversations ({band})."]
        if average < TARGET_QUALITY:
            recommendations.append(
                f"WARNING: the average sits {TARGET_QUALITY - average:.3f} below the {TARGET_QUALITY:.2f} target."
            )
        else:
            recommendations.append(
                f"GOOD: the average clears the {TARGET_QUALITY:.2f} target by {average - TARGET_QUALITY:.3f}."
            )

        for tier, stats in sorted(tier_performance.items(), key=lambda item: item[1]["average_quality"]):
            if stats["average_quality"] < average - TIER_GAP:
                recommendations.append(
                    f"{tier} averages {stats['average_quality']:.3f} over {stats['conversation_count']} conversations, "
                    f"{average - stats['average_quality']:.3f} under the overall mean."
                )
        for name, stats in sorted(component_performance.items(), key=lambda item: item[1]["average_score"]):
            if stats["sample_count"] and stats["average_score"] < COMPONENT_FLOOR:
                recommendations.append(
                    f"{name} averages {stats['average_score']:.3f} across {int(stats['sample_count'])} samples, "
                    f"below the {COMPONENT_FLOOR:.2f} floor."
                )
        if len(recommendations) == BASELINE_RECOMMENDATIONS:
            recommendations.append("No tier or component falls materially below the snapshot mean.")
        return recommendations

    def _band(self, score: float) -> str:
        for edge, label in zip(reversed(QUALITY_BINS[:-1]), reversed(QUALITY_LABELS), strict=True):
            if score >= edge:
                return label
        return QUALITY_LABELS[0]

    def create_quality_overview_chart(self, analytics: QualityAnalytics) -> go.Figure:
        """Distribution and per-tier averages side by side."""
        figure = make_subplots(rows=1, cols=2, subplot_titles=("Quality distribution", "Average quality by tier"))
        if analytics.quality_distribution:
            bands = [label for label in QUALITY_LABELS if label in analytics.quality_distribution]
            figure.add_trace(
                go.Bar(
                    x=bands,
                    y=[analytics.quality_distribution[label] for label in bands],
                    marker_color=COLOR_SCHEMES["quality_levels"],
                    name="Conversations",
                ),
                row=1,
                col=1,
            )
        if analytics.tier_performance:
            tiers = sorted(
                analytics.tier_performance, key=lambda tier: analytics.tier_performance[tier]["average_quality"]
            )
            figure.add_trace(
                go.Bar(
                    x=tiers,
                    y=[analytics.tier_performance[tier]["average_quality"] for tier in tiers],
                    marker_color=COLOR_SCHEMES["tiers"],
                    name="Average quality",
                ),
                row=1,
                col=2,
            )
        if not figure.data:
            figure.add_annotation(text=EMPTY_RECOMMENDATION, showarrow=False, font={"size": 16})
        return self._layout(figure, height=380)

    def create_detailed_analysis_charts(self, df: pd.DataFrame) -> tuple[go.Figure, go.Figure]:
        """Component means by tier, and the correlation between components."""
        frame = df.copy() if df is not None else pd.DataFrame()
        components = [column for column in QUALITY_COMPONENTS if column in frame.columns]
        if frame.empty or "tier" not in frame.columns or not components:
            return self._placeholder("Component profile by tier"), self._placeholder("Component correlation")

        for column in components:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        means = frame.groupby("tier", observed=True)[components].mean()

        heatmap = go.Figure(
            go.Heatmap(
                z=means.to_numpy(),
                x=[QUALITY_COMPONENTS[column] for column in components],
                y=[str(tier) for tier in means.index],
                colorscale="Blues",
                colorbar={"title": "Mean"},
            )
        )
        correlation = go.Figure(
            go.Heatmap(
                z=frame[components].corr().to_numpy(),
                x=[QUALITY_COMPONENTS[column] for column in components],
                y=[QUALITY_COMPONENTS[column] for column in components],
                colorscale="RdBu",
                zmin=-1.0,
                zmax=1.0,
                colorbar={"title": "r"},
            )
        )
        return self._layout(heatmap, title="Component profile by tier", height=380), self._layout(
            correlation, title="Component correlation", height=440
        )

    def _placeholder(self, title: str) -> go.Figure:
        figure = go.Figure()
        figure.add_annotation(text=EMPTY_RECOMMENDATION, showarrow=False, font={"size": 16})
        return self._layout(figure, title=title, height=300)

    def _layout(self, figure: go.Figure, title: str = "", height: int = 380) -> go.Figure:
        axis = {"gridcolor": "#1c1c21", "zerolinecolor": "#1c1c21"}
        figure.update_layout(
            title=title or None,
            height=height,
            margin={"l": 48, "r": 24, "t": 56, "b": 48},
            paper_bgcolor="#121214",
            plot_bgcolor="#121214",
            font={"color": "#e8e6df", "family": "Instrument Sans Variable, system-ui, sans-serif"},
            xaxis=axis,
            yaxis=dict(axis, range=[0.0, 1.0]) if title == "Average quality by tier" else axis,
            showlegend=False,
        )
        return figure

    def render_html(self, force_refresh: bool = False) -> str:
        """Render the dashboard as a self-contained HTML page."""
        frame = self.load_quality_data(force_refresh=force_refresh)
        analytics = self.calculate_quality_analytics(frame)
        heatmap, correlation = self.create_detailed_analysis_charts(frame)
        return PAGE.substitute(
            metrics=self._metrics_html(analytics),
            overview=self.create_quality_overview_chart(analytics).to_html(full_html=False, include_plotlyjs="cdn"),
            heatmap=heatmap.to_html(full_html=False, include_plotlyjs=False),
            correlation=correlation.to_html(full_html=False, include_plotlyjs=False),
            recommendations=self._recommendations_html(analytics),
            tiers=self._tiers_html(analytics),
            generated=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            source=str(self.db_path),
            records=analytics.total_conversations,
            freshness=analytics.data_freshness,
        )

    def write_dashboard(self, path: str | Path, force_refresh: bool = False) -> Path:
        """Render the dashboard and write it to ``path``."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.render_html(force_refresh=force_refresh), encoding="utf-8")
        return target

    def _metrics_html(self, analytics: QualityAnalytics) -> str:
        cards = [
            ("Conversations", f"{analytics.total_conversations:,}", "Joined quality records in the window"),
            ("Average quality", f"{analytics.average_quality:.3f}", self._band(analytics.average_quality)),
            ("Outliers", f"{len(analytics.anomalies)}", "Flagged by the IQR fence"),
            ("Data freshness", analytics.data_freshness, f"Trend over {len(analytics.trend_data)} days"),
        ]
        return "".join(
            f'<div class="metric"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="note">{note}</div></div>'
            for label, value, note in cards
        )

    def _recommendations_html(self, analytics: QualityAnalytics) -> str:
        items = "".join(f"<li>{recommendation}</li>" for recommendation in analytics.recommendations)
        return f"<ul>{items}</ul>"

    def _tiers_html(self, analytics: QualityAnalytics) -> str:
        if not analytics.tier_performance:
            return "<table><tbody><tr><td>No tier data in this window.</td></tr></tbody></table>"
        rows = "".join(
            f"<tr><td>{tier}</td><td>{stats['average_quality']:.3f}</td><td>{stats['conversation_count']}</td></tr>"
            for tier, stats in sorted(analytics.tier_performance.items(), key=lambda item: -item[1]["average_quality"])
        )
        return (
            "<table><thead><tr><th>Tier</th><th>Average quality</th><th>Conversations</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )


def main(argv: list[str] | None = None) -> int:
    """Render the dashboard to HTML from the command line."""
    parser = argparse.ArgumentParser(description="Render the quality analytics dashboard to HTML.")
    parser.add_argument("--db", default=None, help=f"SQLite database (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--out", default="quality-analytics-dashboard.html", help="Output HTML path")
    arguments = parser.parse_args(argv)

    dashboard = QualityAnalyticsDashboard(db_path=arguments.db)
    target = dashboard.write_dashboard(arguments.out, force_refresh=True)
    sys.stdout.write(f"{target}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
