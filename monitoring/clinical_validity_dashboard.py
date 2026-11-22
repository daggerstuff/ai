"""Clinical validity monitoring dashboard over the clinical_validity_metrics table.

The historic implementation rendered through Streamlit, which is not a declared
dependency of this repository, so the module could not be imported. This version
keeps the metric definitions and the Plotly figures and replaces the Streamlit
page with a self-contained HTML renderer, so the same dashboard can be served by
any static host or opened from disk.

The score ladder and the validity rate come from the backlog requirements: a
measurement counts as valid at the ``good`` threshold (0.6), and the status band
is derived from the most recent score.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from string import Template
from typing import Any

import pandas as pd
import plotly.graph_objects as go

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "database" / "conversations.db"

TABLE = "clinical_validity_metrics"
COLUMNS = (
    "timestamp",
    "clinical_validity_score",
    "sample_size",
    "pipeline_stage",
    "data_source",
    "notes",
)

QUALITY_THRESHOLDS = {"excellent": 0.8, "good": 0.6, "fair": 0.4, "poor": 0.0}

TREND_WINDOWS = {"trend_1h": timedelta(hours=1), "trend_24h": timedelta(hours=24)}

CACHE_TTL_SECONDS = 60

MIN_TREND_POINTS = 2

MIN_DISTRIBUTION_POINTS = 2

RECENT_ROWS = 20

COLOR_SCHEMES = {
    "clinical_validity": ["#d32f2f", "#f57c00", "#fbc02d", "#388e3c"],
    "trend": "#1976d2",
    "target": "#2e7d32",
    "bottleneck": "#c62828",
}

EMPTY_METRICS: dict[str, Any] = {
    "current_score": 0.0,
    "sample_size": 0,
    "trend_1h": 0.0,
    "trend_24h": 0.0,
    "valid_samples_24h": 0,
    "total_samples_24h": 0,
    "clinical_validity_rate": 0.0,
    "status": "no_data",
}

PAGE = Template(
    """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clinical Validity Dashboard</title>
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
  h1 {
    font-size: clamp(1.75rem, 3vw, 2.5rem);
    font-weight: 800;
    letter-spacing: 0.01em;
    margin: 0 0 8px;
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
  .figure { background: var(--panel); border: 1px solid var(--rule); margin-top: 24px; padding: 8px; }
  table { width: 100%; border-collapse: collapse; margin-top: 24px; font-size: 0.875rem; }
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
  <h1>Clinical Validity Monitoring</h1>
  <p class="lede">Clinical validity scores for the training pipeline, measured against a
  0.6 validity target. Every figure below is drawn from the measurements in the database.</p>
  <section class="metrics">${metrics}</section>
  <div class="figure">${gauge}</div>
  <div class="figure">${trend}</div>
  <div class="figure">${distribution}</div>
  ${table}
  <footer>Generated ${generated} from ${source} &middot; ${records} records over ${hours} hours</footer>
</main>
</body>
</html>
"""
)


class ClinicalValidityDashboard:
    """Track clinical validity scores and their trend across the pipeline."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        self.cache_duration = CACHE_TTL_SECONDS
        self.quality_thresholds = dict(QUALITY_THRESHOLDS)
        self.color_schemes = dict(COLOR_SCHEMES)
        self._cached_data: pd.DataFrame | None = None
        self._last_cache_time: datetime | None = None

    def load_clinical_validity_data(self, force_refresh: bool = False, hours_back: int = 24) -> pd.DataFrame:
        """Read the measurements from the last ``hours_back`` hours.

        Returns an empty frame with the expected columns when the database or the
        table is absent, which is the dashboard's documented "no data" state.
        """
        now = datetime.now(UTC)
        if (
            not force_refresh
            and self._cached_data is not None
            and self._last_cache_time is not None
            and (now - self._last_cache_time).total_seconds() < self.cache_duration
        ):
            return self._cached_data.copy()

        frame = self._query(hours_back)
        self._cached_data = frame
        self._last_cache_time = now
        return frame.copy()

    def _query(self, hours_back: int) -> pd.DataFrame:
        if not self.db_path.exists():
            return pd.DataFrame(columns=list(COLUMNS))
        cutoff = (datetime.now(UTC) - timedelta(hours=hours_back)).strftime("%Y-%m-%d %H:%M:%S")
        try:
            with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True) as conn:
                frame = pd.read_sql_query(
                    f"SELECT {', '.join(COLUMNS)} FROM {TABLE} WHERE timestamp >= ? ORDER BY timestamp DESC",
                    conn,
                    params=[cutoff],
                )
        except sqlite3.Error, pd.errors.DatabaseError:
            return pd.DataFrame(columns=list(COLUMNS))
        return self._normalise(frame)

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce timestamps to UTC-aware and scores to numbers, newest first."""
        if df is None or df.empty or "clinical_validity_score" not in df.columns:
            return pd.DataFrame(columns=list(COLUMNS))
        frame = df.copy()
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            frame = frame.dropna(subset=["timestamp"])
        frame["clinical_validity_score"] = pd.to_numeric(frame["clinical_validity_score"], errors="coerce")
        frame = frame.dropna(subset=["clinical_validity_score"])
        if "timestamp" in frame.columns:
            frame = frame.sort_values("timestamp", ascending=False)
        return frame.reset_index(drop=True)

    def calculate_current_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Summarise the most recent measurement, its trend and the validity rate."""
        frame = self._normalise(df)
        if frame.empty:
            return dict(EMPTY_METRICS)

        latest = frame.iloc[0]
        current_score = float(latest["clinical_validity_score"])
        sample_size = 0 if pd.isna(latest.get("sample_size")) else int(latest["sample_size"])

        window = frame[frame["timestamp"] >= datetime.now(UTC) - TREND_WINDOWS["trend_24h"]]
        total_samples = len(window)
        valid_samples = int((window["clinical_validity_score"] >= self.quality_thresholds["good"]).sum())

        return {
            "current_score": current_score,
            "sample_size": sample_size,
            "trend_1h": self._trend(frame, TREND_WINDOWS["trend_1h"]),
            "trend_24h": self._trend(frame, TREND_WINDOWS["trend_24h"]),
            "valid_samples_24h": valid_samples,
            "total_samples_24h": total_samples,
            "clinical_validity_rate": (valid_samples / total_samples * 100) if total_samples else 0.0,
            "status": self.classify(current_score),
        }

    def _trend(self, frame: pd.DataFrame, window: timedelta) -> float:
        """Difference between the newest and oldest score inside ``window``."""
        if frame.empty or "timestamp" not in frame.columns:
            return 0.0
        recent = frame[frame["timestamp"] >= datetime.now(UTC) - window]
        if len(recent) < MIN_TREND_POINTS:
            return 0.0
        return float(recent.iloc[0]["clinical_validity_score"] - recent.iloc[-1]["clinical_validity_score"])

    def classify(self, score: float) -> str:
        """Map a score onto the excellent/good/fair/poor ladder."""
        if score >= self.quality_thresholds["excellent"]:
            return "excellent"
        if score >= self.quality_thresholds["good"]:
            return "good"
        if score >= self.quality_thresholds["fair"]:
            return "fair"
        return "poor"

    def create_clinical_validity_gauge(self, current_score: float) -> go.Figure:
        """Gauge for the current score, with the target marked."""
        figure = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=current_score,
                title={"text": "Clinical Validity Score"},
                delta={"reference": self.quality_thresholds["good"], "relative": True, "valueformat": ".1%"},
                gauge={
                    "axis": {"range": [0.0, 1.0], "tickwidth": 1, "tickcolor": "#8b8b94"},
                    "bar": {"color": COLOR_SCHEMES["trend"]},
                    "bgcolor": "#121214",
                    "borderwidth": 1,
                    "bordercolor": "#1c1c21",
                    "steps": [
                        {
                            "range": [0.0, self.quality_thresholds["fair"]],
                            "color": COLOR_SCHEMES["clinical_validity"][0],
                        },
                        {
                            "range": [self.quality_thresholds["fair"], self.quality_thresholds["good"]],
                            "color": COLOR_SCHEMES["clinical_validity"][1],
                        },
                        {
                            "range": [self.quality_thresholds["good"], self.quality_thresholds["excellent"]],
                            "color": COLOR_SCHEMES["clinical_validity"][2],
                        },
                        {
                            "range": [self.quality_thresholds["excellent"], 1.0],
                            "color": COLOR_SCHEMES["clinical_validity"][3],
                        },
                    ],
                    "threshold": {
                        "line": {"color": COLOR_SCHEMES["target"], "width": 3},
                        "thickness": 0.75,
                        "value": self.quality_thresholds["good"],
                    },
                },
            )
        )
        return self._layout(figure, height=300)

    def create_trend_chart(self, df: pd.DataFrame) -> go.Figure:
        """Score over time with the four threshold bands drawn in."""
        frame = self._normalise(df)
        figure = go.Figure()
        if frame.empty:
            figure.add_annotation(text="No clinical validity data available", showarrow=False, font={"size": 16})
            return self._layout(figure, title="Clinical Validity Trend", height=300)

        figure.add_trace(
            go.Scatter(
                x=frame["timestamp"],
                y=frame["clinical_validity_score"],
                mode="lines+markers",
                name="Clinical Validity Score",
                line={"color": COLOR_SCHEMES["trend"], "width": 2},
                marker={"size": 4},
            )
        )
        for band, colour in zip(
            ("excellent", "good", "fair", "poor"), reversed(COLOR_SCHEMES["clinical_validity"]), strict=True
        ):
            figure.add_hline(
                y=self.quality_thresholds[band],
                line_dash="dash",
                line_color=colour,
                annotation_text=f"{band.title()} ({self.quality_thresholds[band]:.1f})",
                annotation_position="top left",
            )
        return self._layout(figure, title="Clinical Validity Trend", height=320, y_range=[0.0, 1.0])

    def create_distribution_chart(self, df: pd.DataFrame) -> go.Figure:
        """Histogram of the scores with the target marked."""
        frame = self._normalise(df)
        figure = go.Figure()
        if len(frame) < MIN_DISTRIBUTION_POINTS:
            figure.add_annotation(text="Insufficient data for distribution", showarrow=False, font={"size": 16})
            return self._layout(figure, title="Clinical Validity Score Distribution", height=260)

        figure.add_trace(
            go.Histogram(
                x=frame["clinical_validity_score"],
                nbinsx=20,
                name="Score Distribution",
                marker_color=COLOR_SCHEMES["trend"],
                opacity=0.75,
            )
        )
        figure.add_vline(
            x=self.quality_thresholds["good"],
            line_dash="dash",
            line_color=COLOR_SCHEMES["target"],
            annotation_text="Validity target",
            annotation_position="top",
        )
        return self._layout(figure, title="Clinical Validity Score Distribution", height=260)

    def _layout(
        self, figure: go.Figure, title: str = "", height: int = 300, y_range: list[float] | None = None
    ) -> go.Figure:
        axis = {"gridcolor": "#1c1c21", "zerolinecolor": "#1c1c21"}
        layout: dict[str, Any] = {
            "title": title or None,
            "height": height,
            "margin": {"l": 48, "r": 24, "t": 56, "b": 40},
            "paper_bgcolor": "#121214",
            "plot_bgcolor": "#121214",
            "font": {"color": "#e8e6df", "family": "Instrument Sans Variable, system-ui, sans-serif"},
            "xaxis": axis,
            "yaxis": dict(axis, range=y_range) if y_range else axis,
            "hovermode": "x unified",
            "showlegend": False,
        }
        figure.update_layout(**layout)
        return figure

    def render_html(self, hours_back: int = 24, force_refresh: bool = False) -> str:
        """Render the dashboard as a self-contained HTML page."""
        frame = self.load_clinical_validity_data(force_refresh=force_refresh, hours_back=hours_back)
        metrics = self.calculate_current_metrics(frame)
        return PAGE.substitute(
            metrics=self._metrics_html(metrics),
            gauge=self.create_clinical_validity_gauge(metrics["current_score"]).to_html(
                full_html=False, include_plotlyjs="cdn"
            ),
            trend=self.create_trend_chart(frame).to_html(full_html=False, include_plotlyjs=False),
            distribution=self.create_distribution_chart(frame).to_html(full_html=False, include_plotlyjs=False),
            table=self._table_html(frame),
            generated=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            source=str(self.db_path),
            records=len(frame),
            hours=hours_back,
        )

    def write_dashboard(self, path: str | Path, hours_back: int = 24, force_refresh: bool = False) -> Path:
        """Render the dashboard and write it to ``path``."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.render_html(hours_back=hours_back, force_refresh=force_refresh), encoding="utf-8")
        return target

    def _metrics_html(self, metrics: dict[str, Any]) -> str:
        cards = [
            ("Current score", f"{metrics['current_score']:.3f}", f"{metrics['trend_1h']:+.3f} over 1 hour"),
            ("Sample size", f"{metrics['sample_size']:,}", "Samples in the most recent measurement"),
            (
                "Validity rate",
                f"{metrics['clinical_validity_rate']:.1f}%",
                f"Share at or above {self.quality_thresholds['good']:.1f}",
            ),
            (
                "Status",
                metrics["status"].title(),
                f"{metrics['valid_samples_24h']} of {metrics['total_samples_24h']} valid",
            ),
        ]
        return "".join(
            f'<div class="metric"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="note">{note}</div></div>'
            for label, value, note in cards
        )

    def _table_html(self, frame: pd.DataFrame) -> str:
        if frame.empty:
            return "<table><tbody><tr><td>No clinical validity measurements in this window.</td></tr></tbody></table>"
        recent = frame.head(RECENT_ROWS)
        header = "".join(f"<th>{column.replace('_', ' ')}</th>" for column in COLUMNS)
        rows = []
        for record in recent.to_dict("records"):
            cells = []
            for column in COLUMNS:
                value = record.get(column)
                if column == "timestamp" and not pd.isna(value):
                    value = pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")
                elif isinstance(value, float):
                    value = f"{value:.3f}"
                cells.append(f"<td>{'' if pd.isna(value) else value}</td>")
            rows.append(f"<tr>{''.join(cells)}</tr>")
        return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def main(argv: list[str] | None = None) -> int:
    """Render the dashboard to HTML from the command line."""
    parser = argparse.ArgumentParser(description="Render the clinical validity dashboard to HTML.")
    parser.add_argument("--db", default=None, help=f"SQLite database (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--hours", type=int, default=24, help="Window in hours (default: 24)")
    parser.add_argument("--out", default="clinical-validity-dashboard.html", help="Output HTML path")
    arguments = parser.parse_args(argv)

    dashboard = ClinicalValidityDashboard(db_path=arguments.db)
    target = dashboard.write_dashboard(arguments.out, hours_back=arguments.hours, force_refresh=True)
    sys.stdout.write(f"{target}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
