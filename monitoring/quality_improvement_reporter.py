"""Reporting layer for tracked quality improvement interventions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

from .quality_improvement_tracker import (
    Intervention,
    InterventionImpact,
    QualityImprovementTracker,
)

DEFAULT_OUTPUT_DIR = Path("reports") / "quality_improvement"


@dataclass
class ImprovementReport:
    """One improvement tracking run, ready to serialise."""

    generated_at: str
    report_period_days: int
    executive_summary: str
    interventions: list[Intervention] = field(default_factory=list)
    impacts: list[InterventionImpact] = field(default_factory=list)
    status_counts: dict[str, int] = field(default_factory=dict)
    action_items: list[str] = field(default_factory=list)

    @property
    def active_interventions(self) -> list[Intervention]:
        """Interventions currently running."""
        return [intervention for intervention in self.interventions if intervention.status == "in_progress"]

    @property
    def completed_interventions(self) -> list[Intervention]:
        """Interventions that have finished."""
        return [intervention for intervention in self.interventions if intervention.status == "completed"]

    @property
    def improvement_analyses(self) -> list[InterventionImpact]:
        """Impacts with a measurable before/after window."""
        return [impact for impact in self.impacts if impact.improvement_metrics.get("available")]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "report_period_days": self.report_period_days,
            "executive_summary": self.executive_summary,
            "interventions": [intervention.__dict__ for intervention in self.interventions],
            "impacts": [impact.__dict__ for impact in self.impacts],
            "status_counts": dict(self.status_counts),
            "action_items": list(self.action_items),
        }


class QualityImprovementReporter:
    """Build, save and visualise an improvement report."""

    def __init__(
        self,
        output_dir: str | Path | None = None,
        tracker: QualityImprovementTracker | None = None,
    ) -> None:
        self.output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
        self.tracker = tracker or QualityImprovementTracker()

    def generate_comprehensive_report(self, report_period_days: int = 30) -> ImprovementReport:
        """Collect every intervention and measure the completed ones."""
        interventions = self.tracker.list_interventions()
        impacts: list[InterventionImpact] = []
        for intervention in interventions:
            if intervention.status in {"completed", "in_progress"}:
                impacts.append(self.tracker.analyze_intervention_impact(intervention.intervention_id))

        status_counts: dict[str, int] = {}
        for intervention in interventions:
            status_counts[intervention.status] = status_counts.get(intervention.status, 0) + 1

        return ImprovementReport(
            generated_at=datetime.now(UTC).isoformat(),
            report_period_days=report_period_days,
            executive_summary=self._summarise(interventions, impacts, status_counts),
            interventions=interventions,
            impacts=impacts,
            status_counts=status_counts,
            action_items=self._action_items(interventions, impacts),
        )

    def _summarise(
        self,
        interventions: list[Intervention],
        impacts: list[InterventionImpact],
        status_counts: dict[str, int],
    ) -> str:
        if not interventions:
            return "No quality improvement interventions have been recorded yet."
        breakdown = ", ".join(f"{status} {count}" for status, count in sorted(status_counts.items()))
        measured = [impact for impact in impacts if impact.improvement_metrics.get("available")]
        if not measured:
            return f"{len(interventions)} intervention(s) tracked ({breakdown}); none has a measurable window yet."
        exceeded = sum(1 for impact in measured if impact.verdict == "exceeded")
        regressed = sum(1 for impact in measured if impact.verdict == "regressed")
        average = sum(impact.absolute_change for impact in measured) / len(measured)
        return (
            f"{len(interventions)} intervention(s) tracked ({breakdown}). "
            f"{len(measured)} measured: {exceeded} exceeded their target, {regressed} regressed. "
            f"Mean change on target components is {average:+.4f}."
        )

    def _action_items(self, interventions: list[Intervention], impacts: list[InterventionImpact]) -> list[str]:
        items: list[str] = []
        for intervention in interventions:
            if intervention.status == "created":
                items.append(f"Start '{intervention.name}' ({intervention.intervention_id}); it is still queued.")
            elif intervention.status == "in_progress" and not intervention.measurements:
                items.append(
                    f"Record a progress measurement for '{intervention.name}' ({intervention.intervention_id})."
                )
        for impact in impacts:
            if impact.verdict == "regressed":
                items.append(
                    f"{impact.target_component} regressed by {impact.absolute_change:+.4f} after "
                    f"{impact.intervention_id}; investigate before rolling out further."
                )
            elif impact.verdict == "exceeded":
                items.append(
                    f"{impact.target_component} gained {impact.absolute_change:+.4f} from "
                    f"{impact.intervention_id}; consider making the change permanent."
                )
        return items

    def save_report(self, report: ImprovementReport, format: str = "json") -> str:
        """Write the report to ``output_dir`` and return the path written."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")

        if format == "json":
            path = self.output_dir / f"quality_improvement_{stamp}.json"
            path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
            return str(path)

        if format in {"md", "markdown"}:
            path = self.output_dir / f"quality_improvement_{stamp}.md"
            lines = [
                "# Quality improvement report",
                "",
                f"Generated {report.generated_at} over a {report.report_period_days}-day window.",
                "",
                "## Executive summary",
                "",
                report.executive_summary,
                "",
                "## Interventions",
                "",
            ]
            if report.interventions:
                lines += [
                    f"- {intervention.intervention_id}: {intervention.name} "
                    f"({intervention.status}, target {intervention.target_component})"
                    for intervention in report.interventions
                ]
            else:
                lines.append("- None recorded.")
            lines += ["", "## Action items", ""]
            lines += [f"- {item}" for item in report.action_items] or ["- None."]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return str(path)

        raise ValueError(f"unsupported report format: {format!r}")

    def create_improvement_visualizations(self, report: ImprovementReport) -> dict[str, dict[str, Any]]:
        """Build plotly figures for the improvement report and return them as dicts."""

        figures: dict[str, dict[str, Any]] = {}

        measured = [impact for impact in report.impacts if impact.improvement_metrics.get("available")]
        if measured:
            figure = go.Figure()
            figure.add_trace(
                go.Bar(
                    name="baseline",
                    x=[impact.target_component for impact in measured],
                    y=[impact.baseline_mean for impact in measured],
                )
            )
            figure.add_trace(
                go.Bar(
                    name="current",
                    x=[impact.target_component for impact in measured],
                    y=[impact.current_mean for impact in measured],
                )
            )
            figure.update_layout(title="Baseline vs current by component", barmode="group", yaxis_range=[0, 1])
            figures["baseline_vs_current"] = figure.to_dict()

            figure = go.Figure(
                go.Bar(
                    x=[impact.intervention_id for impact in measured],
                    y=[impact.target_progress for impact in measured],
                )
            )
            figure.update_layout(title="Progress toward expected improvement", yaxis_title="fraction of target")
            figures["target_progress"] = figure.to_dict()

        if report.status_counts:
            figure = go.Figure(
                go.Pie(
                    labels=list(report.status_counts),
                    values=list(report.status_counts.values()),
                    hole=0.5,
                )
            )
            figure.update_layout(title="Intervention status")
            figures["status_mix"] = figure.to_dict()

        if figures:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "improvement_visualizations.json").write_text(
                json.dumps(figures, default=str), encoding="utf-8"
            )
        return figures
