"""Reporting layer for the quality comparison results.

Turns the statistical output of :class:`~monitoring.quality_comparator.QualityComparator`
into an executive summary, machine-readable exports and figures. Figures are
built with plotly because matplotlib is not a declared dependency of this tree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from .quality_comparator import (
    BenchmarkAnalysis,
    ComponentComparison,
    DatasetComparison,
    QualityComparator,
    TierComparison,
)

DEFAULT_OUTPUT_DIR = Path("reports") / "quality_comparison"
LARGE_EFFECT = 0.5


@dataclass
class ComparisonReport:
    """Everything one comparison run produced."""

    generated_at: str
    days_back: int
    record_count: int
    executive_summary: str
    tier_comparisons: list[TierComparison] = field(default_factory=list)
    dataset_comparisons: list[DatasetComparison] = field(default_factory=list)
    component_comparisons: list[ComponentComparison] = field(default_factory=list)
    benchmark_analyses: list[BenchmarkAnalysis] = field(default_factory=list)
    rankings: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    action_items: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report, expanding every comparison dataclass."""

        def _as_dict(item: Any) -> Any:
            if hasattr(item, "__dataclass_fields__"):
                return {key: _as_dict(value) for key, value in item.__dict__.items()}
            if isinstance(item, tuple):
                return list(item)
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
            "tier_comparisons": _as_dict(self.tier_comparisons),
            "dataset_comparisons": _as_dict(self.dataset_comparisons),
            "component_comparisons": _as_dict(self.component_comparisons),
            "benchmark_analyses": _as_dict(self.benchmark_analyses),
            "rankings": _as_dict(self.rankings),
            "action_items": list(self.action_items),
        }


def _format_comparison(comparison: TierComparison) -> str:
    direction = "higher" if comparison.difference > 0 else "lower"
    significance = "significant" if comparison.significant else "not significant"
    return (
        f"{comparison.tier_a} scores {abs(comparison.difference):.3f} {direction} than "
        f"{comparison.tier_b} ({significance}, d={comparison.effect_size:.2f}, "
        f"p={comparison.p_value:.4f})"
    )


class QualityComparisonReporter:
    """Build, save and visualise a quality comparison report."""

    def __init__(self, output_dir: str | Path | None = None, comparator: QualityComparator | None = None) -> None:
        self.output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
        self.comparator = comparator or QualityComparator()

    def generate_comprehensive_report(self, days_back: int | None = 30) -> ComparisonReport:
        """Run every comparison the comparator offers over one data window."""
        frame = self.comparator.load_comparison_data(days_back=days_back)
        tiers = self.comparator.compare_tiers(frame)
        datasets = self.comparator.compare_datasets(frame)
        components = self.comparator.compare_components(frame)
        benchmarks = self.comparator.perform_benchmark_analysis(frame)
        rankings = self.comparator.calculate_performance_rankings(frame)

        return ComparisonReport(
            generated_at=datetime.now(UTC).isoformat(),
            days_back=days_back if days_back is not None else 0,
            record_count=len(frame),
            executive_summary=self._summarise(frame, tiers, components, benchmarks),
            tier_comparisons=tiers,
            dataset_comparisons=datasets,
            component_comparisons=components,
            benchmark_analyses=benchmarks,
            rankings=rankings,
            action_items=self._action_items(components, benchmarks, tiers),
        )

    def _action_items(
        self,
        components: list[ComponentComparison],
        benchmarks: list[BenchmarkAnalysis],
        tiers: list[TierComparison],
    ) -> list[str]:
        """Turn the findings into the concrete follow-ups a reader can act on."""
        items: list[str] = []
        for analysis in sorted(benchmarks, key=lambda entry: entry.performance_gap)[:3]:
            if analysis.status == "below_target":
                items.append(
                    f"Lift {analysis.group_type} '{analysis.group}' by "
                    f"{abs(analysis.performance_gap):.3f} to reach the "
                    f"{analysis.benchmark_value:.2f} target."
                )
        if components:
            weakest = components[-1]
            items.append(
                f"Investigate {weakest.component}, the weakest component at {weakest.mean:.3f} "
                f"({weakest.contribution:.1%} of the component total)."
            )
        for comparison in tiers:
            if comparison.significant and comparison.effect_size <= -0.5:
                items.append(
                    f"{comparison.tier_b} trails {comparison.tier_a} by {abs(comparison.difference):.3f} "
                    f"with a large effect (d={comparison.effect_size:.2f}); review its data pipeline."
                )
        return items

    def _summarise(
        self,
        frame: pd.DataFrame,
        tiers: list[TierComparison],
        components: list[ComponentComparison],
        benchmarks: list[BenchmarkAnalysis],
    ) -> str:
        if frame.empty:
            return "No conversations with validated quality metrics fell inside the requested window."

        lines = [
            f"Compared {len(frame)} conversations over the last "
            f"{int((frame['created_at'].max() - frame['created_at'].min()).days) + 1} days.",
            f"Overall quality averages {frame['overall_quality'].mean():.3f} "
            f"(median {frame['overall_quality'].median():.3f}, "
            f"range {frame['overall_quality'].min():.3f}-{frame['overall_quality'].max():.3f}).",
        ]
        if components:
            strongest = components[0]
            weakest = components[-1]
            lines.append(
                f"Strongest component is {strongest.component} at {strongest.mean:.3f}; "
                f"weakest is {weakest.component} at {weakest.mean:.3f}."
            )
        significant = [comparison for comparison in tiers if comparison.significant]
        if significant:
            lines.append(
                f"{len(significant)} of {len(tiers)} tier comparisons are statistically significant "
                f"(Welch's t-test, p < 0.05): "
                + "; ".join(_format_comparison(comparison) for comparison in significant[:3])
                + "."
            )
        elif tiers:
            lines.append(f"None of the {len(tiers)} tier comparisons reach significance at p < 0.05.")
        below = [analysis for analysis in benchmarks if analysis.status == "below_target"]
        if benchmarks:
            lines.append(
                f"{len(below)} of {len(benchmarks)} tier/dataset groups sit below the "
                f"{benchmarks[0].benchmark_value:.2f} clinical target."
            )
        return " ".join(lines)

    def save_report(self, report: ComparisonReport, format: str = "json") -> str:
        """Write the report to ``output_dir`` and return the path written."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        payload = report.to_dict()

        if format == "json":
            path = self.output_dir / f"quality_comparison_{stamp}.json"
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            return str(path)

        if format in {"md", "markdown"}:
            path = self.output_dir / f"quality_comparison_{stamp}.md"
            path.write_text(self._to_markdown(report), encoding="utf-8")
            return str(path)

        if format == "csv":
            path = self.output_dir / f"quality_comparison_{stamp}.csv"
            rows = [
                {
                    "tier_a": comparison.tier_a,
                    "tier_b": comparison.tier_b,
                    "metric": comparison.metric,
                    "difference": comparison.difference,
                    "effect_size": comparison.effect_size,
                    "p_value": comparison.p_value,
                    "significant": comparison.significant,
                }
                for comparison in report.tier_comparisons
            ]
            pd.DataFrame(rows).to_csv(path, index=False)
            return str(path)

        raise ValueError(f"unsupported report format: {format!r}")

    def _to_markdown(self, report: ComparisonReport) -> str:
        lines = [
            "# Quality comparison report",
            "",
            f"Generated {report.generated_at} over {report.record_count} conversations.",
            "",
            "## Executive summary",
            "",
            report.executive_summary,
            "",
            "## Tier rankings",
            "",
            "| rank | tier | mean | std | count |",
            "| --- | --- | --- | --- | --- |",
        ]
        for entry in report.rankings.get("tiers", []):
            lines.append(
                f"| {entry['rank']} | {entry['name']} | {entry['mean']:.3f} | {entry['std']:.3f} | {entry['count']} |"
            )
        lines += [
            "",
            "## Benchmark gaps",
            "",
            "| group | observed | target | gap | status |",
            "| --- | --- | --- | --- | --- |",
        ]
        for analysis in report.benchmark_analyses:
            lines.append(
                f"| {analysis.group_type}:{analysis.group} | {analysis.observed_mean:.3f} | "
                f"{analysis.benchmark_value:.3f} | {analysis.performance_gap:+.3f} | {analysis.status} |"
            )
        return "\n".join(lines) + "\n"

    def create_comparison_visualizations(self, report: ComparisonReport) -> dict[str, dict[str, Any]]:
        """Build plotly figures for the report and return them as dicts."""

        figures: dict[str, dict[str, Any]] = {}

        tiers = report.rankings.get("tiers", [])
        if tiers:
            figure = go.Figure(
                go.Bar(
                    x=[entry["name"] for entry in tiers],
                    y=[entry["mean"] for entry in tiers],
                    error_y={"type": "data", "array": [entry["std"] for entry in tiers]},
                )
            )
            figure.update_layout(title="Mean overall quality by tier", yaxis_range=[0, 1])
            figures["tier_quality"] = figure.to_dict()

        datasets = report.rankings.get("datasets", [])
        if datasets:
            figure = go.Figure(
                go.Bar(
                    x=[entry["name"] for entry in datasets],
                    y=[entry["mean"] for entry in datasets],
                )
            )
            figure.update_layout(title="Mean overall quality by dataset", yaxis_range=[0, 1])
            figures["dataset_quality"] = figure.to_dict()

        components = report.component_comparisons
        if components:
            figure = go.Figure(
                go.Scatterpolar(
                    r=[comparison.mean for comparison in components],
                    theta=[comparison.component for comparison in components],
                    fill="toself",
                )
            )
            figure.update_layout(title="Component profile")
            figures["component_profile"] = figure.to_dict()

        benchmarks = report.benchmark_analyses
        if benchmarks:
            figure = go.Figure(
                go.Bar(
                    x=[f"{analysis.group_type}:{analysis.group}" for analysis in benchmarks],
                    y=[analysis.performance_gap for analysis in benchmarks],
                )
            )
            figure.update_layout(title="Gap to clinical target")
            figures["benchmark_gaps"] = figure.to_dict()

        if figures:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "comparison_visualizations.json").write_text(
                json.dumps(figures, default=str), encoding="utf-8"
            )
        return figures
