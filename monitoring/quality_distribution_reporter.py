"""Reporting layer for the quality distribution analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .quality_distribution_analyzer import DistributionAnalysis, QualityDistributionAnalyzer
from .quality_distribution_comparator import DistributionComparison, QualityDistributionComparator

DEFAULT_OUTPUT_DIR = Path("reports") / "quality_distribution"


@dataclass
class DistributionReport:
    """One distribution analysis run, ready to serialise."""

    generated_at: str
    days_back: int
    executive_summary: str
    overall_distribution: DistributionAnalysis | None = None
    component_distributions: list[DistributionAnalysis] = field(default_factory=list)
    tier_comparison: DistributionComparison | None = None
    component_comparison: DistributionComparison | None = None
    correlations: dict[str, Any] = field(default_factory=dict)
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
            "executive_summary": self.executive_summary,
            "overall_distribution": _as_dict(self.overall_distribution),
            "component_distributions": _as_dict(self.component_distributions),
            "tier_comparison": _as_dict(self.tier_comparison),
            "component_comparison": _as_dict(self.component_comparison),
            "correlations": _as_dict(self.correlations),
            "action_items": list(self.action_items),
        }


class QualityDistributionReporter:
    """Build, save and visualise a quality distribution report."""

    def __init__(
        self,
        output_dir: str | Path | None = None,
        analyzer: QualityDistributionAnalyzer | None = None,
        comparator: QualityDistributionComparator | None = None,
    ) -> None:
        self.output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
        self.analyzer = analyzer or QualityDistributionAnalyzer()
        self.comparator = comparator or QualityDistributionComparator(self.analyzer)

    def generate_comprehensive_report(self, days_back: int | None = 30) -> DistributionReport:
        """Analyse the overall metric, every component, and the group splits."""
        frame = self.analyzer.load_quality_data(days_back=days_back)
        overall = self.analyzer.analyze_quality_distribution(frame["overall_quality"], "overall_quality")
        components = [
            self.analyzer.analyze_quality_distribution(frame[component], component)
            for component in frame.columns
            if component
            in {
                "therapeutic_accuracy",
                "conversation_coherence",
                "emotional_authenticity",
                "clinical_compliance",
                "personality_consistency",
                "language_quality",
                "safety_score",
            }
        ]
        tiers = self.comparator.compare_across_tiers(frame) if "tier" in frame.columns else None
        component_comparison = self.comparator.compare_across_components(frame)
        correlations = self.comparator.calculate_correlation_analysis(frame)

        return DistributionReport(
            generated_at=datetime.now(UTC).isoformat(),
            days_back=days_back if days_back is not None else 0,
            executive_summary=self._summarise(overall, tiers, correlations),
            overall_distribution=overall,
            component_distributions=components,
            tier_comparison=tiers,
            component_comparison=component_comparison,
            correlations=correlations,
            action_items=self._action_items(overall, tiers),
        )

    def _summarise(
        self,
        overall: DistributionAnalysis,
        tiers: DistributionComparison | None,
        correlations: dict[str, Any],
    ) -> str:
        if overall.sample_size == 0:
            return "No conversations with validated quality metrics fell inside the requested window."
        stats = overall.statistics
        normal = [test.test_name for test in overall.normality_tests if test.is_normal]
        lines = [
            f"Analysed {overall.sample_size} conversations.",
            f"Overall quality averages {stats.mean:.3f} (median {stats.median:.3f}, "
            f"sd {stats.std_dev:.3f}, IQR {stats.iqr:.3f}), spanning {stats.min_value:.3f}-{stats.max_value:.3f}.",
            f"Skew is {stats.skewness:+.2f} and excess kurtosis {stats.kurtosis:+.2f}; "
            + (
                f"{', '.join(normal)} cannot reject normality."
                if normal
                else "every normality test rejects the normal shape."
            ),
        ]
        if tiers is not None and tiers.statistical_tests:
            significant = [test for test in tiers.statistical_tests if test["significant"]]
            lines.append(f"{len(significant)} of {len(tiers.statistical_tests)} tier pairs differ significantly.")
        strongest = correlations.get("strongest_correlations", [])
        if strongest:
            top = strongest[0]
            lines.append(
                f"Strongest relationship is {top['metric_a']} vs {top['metric_b']} at r={top['correlation']:.3f}."
            )
        return " ".join(lines)

    def _action_items(
        self,
        overall: DistributionAnalysis,
        tiers: DistributionComparison | None,
    ) -> list[str]:
        items: list[str] = []
        if overall.outliers:
            items.append(
                f"Review the {len(overall.outliers)} conversation(s) outside the IQR fence; "
                "they pull the mean away from the median."
            )
        if overall.statistics.skewness <= -0.5:
            items.append("The distribution is left-skewed: a ceiling effect is compressing the top of the scale.")
        elif overall.statistics.skewness >= 0.5:
            items.append("The distribution is right-skewed: most conversations sit at the low end of the scale.")
        if tiers is not None:
            for test in tiers.statistical_tests:
                if test["significant"] and abs(test["difference"]) >= 0.05:
                    items.append(
                        f"{test['group_a']} and {test['group_b']} differ by {test['difference']:+.3f} "
                        f"(p={test['p_value']:.4f}); confirm the gap is intended."
                    )
        return items

    def save_report(self, report: DistributionReport, format: str = "json") -> str:
        """Write the report to ``output_dir`` and return the path written."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")

        if format == "json":
            path = self.output_dir / f"quality_distribution_{stamp}.json"
            path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
            return str(path)

        if format in {"md", "markdown"}:
            path = self.output_dir / f"quality_distribution_{stamp}.md"
            lines = [
                "# Quality distribution report",
                "",
                f"Generated {report.generated_at}.",
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

    def create_distribution_visualizations(self, report: DistributionReport) -> dict[str, dict[str, Any]]:
        """Build plotly figures for the distribution report and return them as dicts."""
        import plotly.graph_objects as go

        figures: dict[str, dict[str, Any]] = {}
        overall = report.overall_distribution

        if overall is not None and overall.histogram:
            figure = go.Figure(go.Bar(x=list(overall.histogram), y=list(overall.histogram.values())))
            figure.update_layout(title=f"Distribution of {overall.metric}", xaxis_title="score bucket")
            figures["histogram"] = figure.to_dict()

        if report.component_comparison is not None and report.component_comparison.groups:
            figure = go.Figure(
                go.Box(
                    y=[group.mean for group in report.component_comparison.groups],
                    x=[group.name for group in report.component_comparison.groups],
                    name="component means",
                )
            )
            figure.update_layout(title="Component means", yaxis_range=[0, 1])
            figures["components"] = figure.to_dict()

        if report.tier_comparison is not None and report.tier_comparison.groups:
            figure = go.Figure(
                go.Bar(
                    x=[group.name for group in report.tier_comparison.groups],
                    y=[group.mean for group in report.tier_comparison.groups],
                    error_y={
                        "type": "data",
                        "array": [group.std_dev for group in report.tier_comparison.groups],
                    },
                )
            )
            figure.update_layout(title="Mean quality by tier", yaxis_range=[0, 1])
            figures["tiers"] = figure.to_dict()

        matrix = report.correlations.get("correlation_matrix", {})
        if matrix:
            columns = list(matrix)
            figure = go.Figure(
                go.Heatmap(
                    z=[[matrix[row][column] for column in columns] for row in columns],
                    x=columns,
                    y=columns,
                    zmin=-1,
                    zmax=1,
                )
            )
            figure.update_layout(title="Metric correlation matrix")
            figures["correlations"] = figure.to_dict()

        if figures:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "distribution_visualizations.json").write_text(
                json.dumps(figures, default=str), encoding="utf-8"
            )
        return figures
