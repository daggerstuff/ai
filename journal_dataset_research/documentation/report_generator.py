"""
Report Generator

Generates structured markdown reports for dataset evaluations, weekly progress,
and final research summaries.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ai.journal_dataset_research.models.dataset_models import (
    DatasetEvaluation,
    DatasetSource,
    ResearchSession,
    WeeklyReport,
)


class ReportGenerator:
    """
    Generates structured markdown reports for research activities.

    Creates evaluation reports, weekly progress reports, and final
    research summary reports.
    """

    def __init__(self, output_directory: str = "reports"):
        """
        Initialize the report generator.

        Args:
            output_directory: Directory to store generated reports
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def generate_evaluation_report(
        self,
        source: DatasetSource,
        evaluation: DatasetEvaluation,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Generate an evaluation report for a dataset.

        Args:
            source: Dataset source information
            evaluation: Dataset evaluation results
            output_path: Optional output path (auto-generated if not provided)

        Returns:
            Path to the generated report
        """
        if output_path is None:
            output_path = (
                self.output_directory
                / f"evaluation_{source.source_id}_{datetime.now().strftime('%Y%m%d')}.md"
            )

        lines = [
            "# Dataset Evaluation Report",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Dataset Information",
            "",
            f"- **Source ID**: {source.source_id}",
            f"- **Title**: {source.title}",
            f"- **Authors**: {', '.join(source.authors)}",
            f"- **Publication Date**: {source.publication_date.strftime('%Y-%m-%d')}",
            f"- **URL**: {source.url}",
        ]

        if source.doi:
            lines.append(f"- **DOI**: {source.doi}")

        lines.extend(
            [
                "",
                "## Evaluation Results",
                "",
                f"**Evaluation Date**: {evaluation.evaluation_date.strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Evaluator**: {evaluation.evaluator or 'Automated'}",
                "",
                "### Scores",
                "",
                f"- **Therapeutic Relevance**: {evaluation.therapeutic_relevance}/10",
                f"- **Data Structure Quality**: {evaluation.data_structure_quality}/10",
                f"- **Training Integration**: {evaluation.training_integration}/10",
                f"- **Ethical Accessibility**: {evaluation.ethical_accessibility}/10",
                f"- **Overall Score**: {evaluation.overall_score:.2f}/10",
                f"- **Priority Tier**: {evaluation.priority_tier.upper()}",
                "",
            ]
        )

        # Add evaluation notes
        if evaluation.therapeutic_relevance_notes:
            lines.extend(
                [
                    "### Therapeutic Relevance Notes",
                    "",
                    evaluation.therapeutic_relevance_notes,
                    "",
                ]
            )

        if evaluation.data_structure_notes:
            lines.extend(
                [
                    "### Data Structure Quality Notes",
                    "",
                    evaluation.data_structure_notes,
                    "",
                ]
            )

        if evaluation.integration_notes:
            lines.extend(
                [
                    "### Training Integration Notes",
                    "",
                    evaluation.integration_notes,
                    "",
                ]
            )

        if evaluation.ethical_notes:
            lines.extend(
                [
                    "### Ethical Accessibility Notes",
                    "",
                    evaluation.ethical_notes,
                    "",
                ]
            )

        # Competitive advantages
        if evaluation.competitive_advantages:
            lines.extend(
                [
                    "### Competitive Advantages",
                    "",
                ]
            )
            for advantage in evaluation.competitive_advantages:
                lines.append(f"- {advantage}")
            lines.append("")

        # Recommendation
        lines.extend(
            [
                "## Recommendation",
                "",
            ]
        )

        if evaluation.priority_tier == "high":
            lines.append(
                "**Priority**: HIGH - This dataset should be prioritized for acquisition and integration."
            )
        elif evaluation.priority_tier == "medium":
            lines.append(
                "**Priority**: MEDIUM - This dataset has value but may require additional consideration."
            )
        else:
            lines.append(
                "**Priority**: LOW - This dataset may have limited value for the training pipeline."
            )

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    def generate_weekly_report(
        self, weekly_report: WeeklyReport, output_path: Optional[Path] = None
    ) -> Path:
        """
        Generate a weekly progress report.

        Args:
            weekly_report: WeeklyReport data
            output_path: Optional output path (auto-generated if not provided)

        Returns:
            Path to the generated report
        """
        if output_path is None:
            output_path = (
                self.output_directory
                / f"weekly_report_week_{weekly_report.week_number}_{weekly_report.end_date.strftime('%Y%m%d')}.md"
            )

        lines = [
            "# Weekly Research Progress Report",
            "",
            f"**Week Number**: {weekly_report.week_number}",
            f"**Report Period**: {weekly_report.start_date.strftime('%Y-%m-%d')} to {weekly_report.end_date.strftime('%Y-%m-%d')}",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Progress Metrics",
            "",
            f"- **Sources Identified**: {weekly_report.sources_identified}",
            f"- **Datasets Evaluated**: {weekly_report.datasets_evaluated}",
            f"- **Access Established**: {weekly_report.access_established}",
            f"- **Datasets Acquired**: {weekly_report.datasets_acquired}",
            f"- **Integration Plans Created**: {weekly_report.integration_plans_created}",
            "",
        ]

        # Key findings
        if weekly_report.key_findings:
            lines.extend(
                [
                    "## Key Findings",
                    "",
                ]
            )
            for i, finding in enumerate(weekly_report.key_findings, 1):
                lines.append(f"{i}. {finding}")
            lines.append("")

        # Challenges
        if weekly_report.challenges:
            lines.extend(
                [
                    "## Challenges",
                    "",
                ]
            )
            for i, challenge in enumerate(weekly_report.challenges, 1):
                lines.append(f"{i}. {challenge}")
            lines.append("")

        # Next week priorities
        if weekly_report.next_week_priorities:
            lines.extend(
                [
                    "## Next Week Priorities",
                    "",
                ]
            )
            for i, priority in enumerate(weekly_report.next_week_priorities, 1):
                lines.append(f"{i}. {priority}")
            lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    def generate_final_summary_report(
        self,
        session: ResearchSession,
        sources: List[DatasetSource],
        evaluations: List[DatasetEvaluation],
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Generate a final research summary report.

        Args:
            session: Research session information
            sources: List of discovered sources
            evaluations: List of dataset evaluations
            output_path: Optional output path (auto-generated if not provided)

        Returns:
            Path to the generated report
        """
        if output_path is None:
            output_path = (
                self.output_directory
                / f"final_summary_{session.session_id}_{datetime.now().strftime('%Y%m%d')}.md"
            )

        lines = [
            "# Research Summary Report",
            "",
            f"**Session ID**: {session.session_id}",
            f"**Session Start**: {session.start_date.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
            f"This report summarizes the research activities for session {session.session_id}. "
            f"A total of {len(sources)} sources were identified and {len(evaluations)} datasets were evaluated.",
            "",
            "## Research Activities",
            "",
        ]

        # Sources by type
        source_types: Dict[str, int] = {}
        for source in sources:
            source_types[source.source_type] = (
                source_types.get(source.source_type, 0) + 1
            )

        lines.append("### Sources by Type")
        for source_type, count in source_types.items():
            lines.append(f"- {source_type}: {count}")
        lines.append("")

        # Evaluation summary
        if evaluations:
            lines.append("### Evaluation Summary")
            lines.append("")

            # Priority breakdown
            priority_counts: Dict[str, int] = {}
            for eval in evaluations:
                priority_counts[eval.priority_tier] = (
                    priority_counts.get(eval.priority_tier, 0) + 1
                )

            lines.append("**Evaluations by Priority**:")
            for priority, count in priority_counts.items():
                lines.append(f"- {priority.upper()}: {count}")
            lines.append("")

            # Score statistics
            scores = [eval.overall_score for eval in evaluations]
            avg_score = sum(scores) / len(scores) if scores else 0
            lines.append("**Score Statistics**:")
            lines.append(f"- Average Score: {avg_score:.2f}/10")
            lines.append(f"- Minimum Score: {min(scores):.2f}/10")
            lines.append(f"- Maximum Score: {max(scores):.2f}/10")
            lines.append("")

            # Top datasets
            sorted_evaluations = sorted(
                evaluations, key=lambda x: x.overall_score, reverse=True
            )
            lines.append("### Top Datasets by Score")
            lines.append("")
            for i, eval in enumerate(sorted_evaluations[:10], 1):
                source = next((s for s in sources if s.source_id == eval.source_id), None)
                title = source.title if source else eval.source_id
                lines.append(
                    f"{i}. {title} - Score: {eval.overall_score:.2f}/10 ({eval.priority_tier})"
                )
            lines.append("")

        # Session progress
        lines.extend(
            [
                "## Session Progress",
                "",
            ]
        )

        for key, value in session.progress_metrics.items():
            lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")

        lines.append("")

        # Weekly targets
        if session.weekly_targets:
            lines.extend(
                [
                    "## Weekly Targets",
                    "",
                ]
            )
            for key, target in session.weekly_targets.items():
                achieved = session.progress_metrics.get(key, 0)
                status = "✅" if achieved >= target else "🔄"
                lines.append(
                    f"{status} **{key.replace('_', ' ').title()}**: {achieved}/{target}"
                )
            lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    def generate_batch_evaluation_report(
        self,
        sources: List[DatasetSource],
        evaluations: List[DatasetEvaluation],
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Generate a batch evaluation report for multiple datasets.

        Args:
            sources: List of dataset sources
            evaluations: List of dataset evaluations
            output_path: Optional output path (auto-generated if not provided)

        Returns:
            Path to the generated report
        """
        if output_path is None:
            output_path = (
                self.output_directory
                / f"batch_evaluation_{datetime.now().strftime('%Y%m%d')}.md"
            )

        lines = [
            "# Batch Dataset Evaluation Report",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Datasets Evaluated**: {len(evaluations)}",
            "",
            "## Summary Statistics",
            "",
        ]

        # Statistics
        scores = [eval.overall_score for eval in evaluations]
        if scores:
            lines.append(f"- **Average Score**: {sum(scores) / len(scores):.2f}/10")
            lines.append(f"- **Minimum Score**: {min(scores):.2f}/10")
            lines.append(f"- **Maximum Score**: {max(scores):.2f}/10")
            lines.append("")

        # Priority breakdown
        priority_counts: Dict[str, int] = {}
        for eval in evaluations:
            priority_counts[eval.priority_tier] = (
                priority_counts.get(eval.priority_tier, 0) + 1
            )

        lines.append("### Evaluations by Priority")
        for priority, count in priority_counts.items():
            lines.append(f"- {priority.upper()}: {count}")
        lines.append("")

        # Individual evaluations
        lines.extend(
            [
                "## Individual Evaluations",
                "",
            ]
        )

        # Sort by score
        sorted_evaluations = sorted(
            evaluations, key=lambda x: x.overall_score, reverse=True
        )

        for evaluation in sorted_evaluations:
            source = next(
                (s for s in sources if s.source_id == evaluation.source_id), None
            )
            if source:
                lines.extend(
                    [
                        f"### {source.title}",
                        "",
                        f"- **Source ID**: {evaluation.source_id}",
                        f"- **Overall Score**: {evaluation.overall_score:.2f}/10",
                        f"- **Priority Tier**: {evaluation.priority_tier.upper()}",
                        f"- **Therapeutic Relevance**: {evaluation.therapeutic_relevance}/10",
                        f"- **Data Structure Quality**: {evaluation.data_structure_quality}/10",
                        f"- **Training Integration**: {evaluation.training_integration}/10",
                        f"- **Ethical Accessibility**: {evaluation.ethical_accessibility}/10",
                        "",
                    ]
                )
            else:
                lines.extend(
                    [
                        f"### {evaluation.source_id}",
                        "",
                        f"- **Overall Score**: {evaluation.overall_score:.2f}/10",
                        f"- **Priority Tier**: {evaluation.priority_tier.upper()}",
                        "",
                    ]
                )

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

