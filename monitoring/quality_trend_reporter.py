"""Reporting layer for the quality trend analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .quality_trend_analyzer import (
    Anomaly,
    Prediction,
    QualityTrendAnalyzer,
    SeasonalPattern,
    TrendAnalysis,
)

DEFAULT_OUTPUT_DIR = Path("reports") / "quality_trends"


@dataclass
class TrendReport:
    """One trend analysis run, ready to serialise."""

    generated_at: str
    days_back: int
    record_count: int
    executive_summary: str
    trend: TrendAnalysis | None = None
    predictions: list[Prediction] = field(default_factory=list)
    anomalies: list[Anomaly] = field(default_factory=list)
    seasonal_patterns: list[SeasonalPattern] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        def _as_dict(item: Any) -> Any:
            if hasattr(item, "__dataclass_fields__"):
                return {key: _as_dict(value) for key, value in item.__dict__.items()}
            if isinstance(item, list):
                return [_as_dict(value) for value in item]
            if isinstance(item, dict):
                return {key: _as_dict(value) for key, value in item.items()}
            return item

        return {
            "generated_at": self.generated_at,
            "days_back": self.days_back,
            "record_count": self.record_count,
            "executive_summary": self.executive_summary,
            "trend": _as_dict(self.trend),
            "predictions": _as_dict(self.predictions),
            "anomalies": _as_dict(self.anomalies),
            "seasonal_patterns": _as_dict(self.seasonal_patterns),
            "action_items": list(self.action_items),
        }


class QualityTrendReporter:
    """Build, save and visualise a quality trend report."""

    def __init__(self, output_dir: str | Path | None = None, analyzer: QualityTrendAnalyzer | None = None) -> None:
        self.output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
        self.analyzer = analyzer or QualityTrendAnalyzer()

    def generate_comprehensive_report(self, days_back: int | None = 180) -> TrendReport:
        """Run the full trend analysis over one window."""
        frame = self.analyzer.load_historical_data(days_back=days_back)
        trend = self.analyzer.analyze_overall_trend(frame)
        predictions = self.analyzer.generate_predictions(frame)
        anomalies = self.analyzer.detect_anomalies(frame)
        seasonal = self.analyzer.detect_seasonal_patterns(frame)

        return TrendReport(
            generated_at=datetime.now(UTC).isoformat(),
            days_back=days_back if days_back is not None else 0,
            record_count=len(frame),
            executive_summary=self._summarise(frame, trend, predictions, anomalies, seasonal),
            trend=trend,
            predictions=predictions,
            anomalies=anomalies,
            seasonal_patterns=seasonal,
            action_items=self._action_items(trend, predictions, anomalies, seasonal),
        )

    def _summarise(
        self,
        frame,
        trend: TrendAnalysis,
        predictions: list[Prediction],
        anomalies: list[Anomaly],
        seasonal: list[SeasonalPattern],
    ) -> str:
        if frame.empty:
            return "No conversations with validated quality metrics fell inside the requested window."

        lines = [f"Analysed {len(frame)} conversations."]
        if trend.points:
            lines.append(
                f"Quality is {trend.direction} at {trend.slope:+.4f} per period "
                f"(R²={trend.r_squared:.2f}, p={trend.p_value:.4f}), moving from "
                f"{trend.start_mean:.3f} to {trend.end_mean:.3f}."
            )
        if predictions:
            latest = predictions[-1]
            lines.append(
                f"The next period is projected at {latest.predicted:.3f} "
                f"(95% interval {latest.lower:.3f}-{latest.upper:.3f})."
            )
        lines.append(
            f"{len(anomalies)} anomalous period(s) detected." if anomalies else "No anomalous periods detected."
        )
        for pattern in seasonal:
            lines.append(
                f"{pattern.period_type.capitalize()} pattern peaks on {pattern.peak} "
                f"and troughs on {pattern.trough} (strength {pattern.strength:.3f})."
            )
        return " ".join(lines)

    def _action_items(
        self,
        trend: TrendAnalysis,
        predictions: list[Prediction],
        anomalies: list[Anomaly],
        seasonal: list[SeasonalPattern],
    ) -> list[str]:
        items: list[str] = []
        if trend.direction == "declining" and trend.significant:
            items.append(
                f"Quality is declining significantly ({trend.slope:+.4f} per period); "
                "review the recent dataset and prompt changes."
            )
        for anomaly in sorted(anomalies, key=lambda entry: abs(entry.z_score), reverse=True)[:3]:
            items.append(
                f"Investigate {anomaly.period}: quality was {anomaly.value:.3f} against an expected "
                f"{anomaly.expected:.3f} ({anomaly.direction}, z={anomaly.z_score:+.2f})."
            )
        if predictions and predictions[-1].predicted < 0.7:
            items.append(
                f"The forecast falls to {predictions[-1].predicted:.3f}; "
                "schedule an intervention before it drops further."
            )
        for pattern in seasonal:
            if pattern.strength > 0.05:
                items.append(
                    f"Account for the {pattern.period_type} effect when scheduling evaluations "
                    f"({pattern.trough} runs lowest)."
                )
        return items

    def save_report(self, report: TrendReport, format: str = "json") -> str:
        """Write the report to ``output_dir`` and return the path written."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")

        if format == "json":
            path = self.output_dir / f"quality_trends_{stamp}.json"
            path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
            return str(path)

        if format in {"md", "markdown"}:
            path = self.output_dir / f"quality_trends_{stamp}.md"
            lines = [
                "# Quality trend report",
                "",
                f"Generated {report.generated_at} over {report.record_count} conversations.",
                "",
                "## Executive summary",
                "",
                report.executive_summary,
                "",
                "## Action items",
                "",
            ]
            lines += [f"- {item}" for item in report.action_items] or ["- None."]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return str(path)

        raise ValueError(f"unsupported report format: {format!r}")

    def create_trend_visualizations(self, report: TrendReport) -> dict[str, dict[str, Any]]:
        """Build plotly figures for the trend report and return them as dicts."""
        import plotly.graph_objects as go

        figures: dict[str, dict[str, Any]] = {}
        trend = report.trend

        if trend is not None and trend.points:
            figure = go.Figure()
            figure.add_trace(
                go.Scatter(
                    x=[point.period for point in trend.points],
                    y=[point.mean for point in trend.points],
                    mode="lines+markers",
                    name="observed",
                )
            )
            if report.predictions:
                figure.add_trace(
                    go.Scatter(
                        x=[prediction.period for prediction in report.predictions],
                        y=[prediction.predicted for prediction in report.predictions],
                        mode="lines+markers",
                        name="forecast",
                        line={"dash": "dash"},
                    )
                )
            figure.update_layout(title="Quality trend and forecast", yaxis_range=[0, 1])
            figures["trend"] = figure.to_dict()

        if report.anomalies:
            figure = go.Figure(
                go.Scatter(
                    x=[anomaly.period for anomaly in report.anomalies],
                    y=[anomaly.z_score for anomaly in report.anomalies],
                    mode="markers",
                    marker={"size": 12},
                )
            )
            figure.update_layout(title="Anomalous periods")
            figures["anomalies"] = figure.to_dict()

        for pattern in report.seasonal_patterns:
            figure = go.Figure(go.Bar(x=list(pattern.pattern), y=list(pattern.pattern.values())))
            figure.update_layout(title=f"Seasonality by {pattern.period_type}", yaxis_range=[0, 1])
            figures[f"seasonality_{pattern.period_type}"] = figure.to_dict()

        if figures:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "trend_visualizations.json").write_text(
                json.dumps(figures, default=str), encoding="utf-8"
            )
        return figures
