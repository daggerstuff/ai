#!/usr/bin/env python3
"""
Quality Validation Result Analysis System
Analyzes quality validation results and provides detailed insights
"""

import json
import sqlite3
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    from .linear_backlog_action_builder import (
        build_linear_backlog_payload,
        write_linear_backlog_artifact,
    )
    from .linear_backlog_dispatcher import LinearBacklogDispatcher
    from .performance_gap_backlog_converter import (
        BacklogConversionResult,
        PerformanceGapBacklogConverter,
    )
except Exception:
    from linear_backlog_action_builder import (
        build_linear_backlog_payload,
        write_linear_backlog_artifact,
    )
    from linear_backlog_dispatcher import LinearBacklogDispatcher
    from performance_gap_backlog_converter import (
        BacklogConversionResult,
        PerformanceGapBacklogConverter,
    )

warnings.simplefilter("default")


@dataclass
class ValidationResult:
    """Quality validation result"""

    conversation_id: str
    metric: str
    score: float
    passed: bool
    threshold: float
    validation_type: str
    timestamp: datetime
    details: dict[str, Any]


@dataclass
class ValidationAnalysis:
    """Validation analysis results"""

    metric: str
    total_validations: int
    passed_validations: int
    failed_validations: int
    pass_rate: float
    average_score: float
    score_distribution: dict[str, int]
    failure_patterns: list[str]
    recommendations: list[str]


class QualityValidationAnalyzer:
    """Enterprise-grade quality validation result analysis system"""

    def __init__(self, db_path: str = "database/conversations.db"):
        self.db_path = Path(db_path)
        self.output_dir = Path("monitoring/quality_validation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backlog_converter = PerformanceGapBacklogConverter()
        self.linear_artifact_dir = self.output_dir / "linear_backlog_artifacts"
        self.linear_artifact_dir.mkdir(parents=True, exist_ok=True)
        self.linear_dispatcher = LinearBacklogDispatcher(
            queue_path=str(self.linear_artifact_dir / "linear_backlog_queue.jsonl"),
        )

        # Quality validation metrics
        self.validation_metrics = [
            "therapeutic_accuracy",
            "conversation_coherence",
            "emotional_authenticity",
            "clinical_compliance",
            "personality_consistency",
            "language_quality",
            "safety_score",
        ]

        # Validation thresholds
        self.validation_thresholds = {
            "therapeutic_accuracy": 0.7,
            "conversation_coherence": 0.6,
            "emotional_authenticity": 0.65,
            "clinical_compliance": 0.8,
            "personality_consistency": 0.6,
            "language_quality": 0.7,
            "safety_score": 0.9,
        }

        # Score ranges for distribution analysis
        self.score_ranges = {
            "excellent": (0.9, 1.0),
            "good": (0.7, 0.9),
            "fair": (0.5, 0.7),
            "poor": (0.0, 0.5),
        }

    def analyze_validation_results(self, days_back: int = 30) -> dict[str, ValidationAnalysis]:
        """Analyze quality validation results"""

        try:
            # Get validation results (synthetic for demo)
            validation_results = self._get_validation_results(days_back)

            if not validation_results:
                return {}

            # Analyze results for each metric
            analyses = {}

            for metric in self.validation_metrics:
                metric_results = [r for r in validation_results if r.metric == metric]
                if metric_results:
                    analysis = self._analyze_metric_validation(metric, metric_results)
                    if analysis:
                        analyses[metric] = analysis

            return analyses

        except Exception:
            return {}

    def _get_validation_results(self, days_back: int) -> list[ValidationResult]:
        """Get validation results (synthetic for demo)"""
        try:
            # Get conversation data for realistic IDs
            conn = sqlite3.connect(self.db_path)

            query = """
            SELECT conversation_id
            FROM conversations
            LIMIT 1000
            """

            cursor = conn.execute(query)
            conversation_ids = [row[0] for row in cursor.fetchall()]
            conn.close()

            if not conversation_ids:
                return []

            # Generate synthetic validation results
            validation_results = []

            for _ in range(min(5000, len(conversation_ids) * len(self.validation_metrics))):
                conversation_id = np.random.choice(conversation_ids)
                metric = np.random.choice(self.validation_metrics)

                # Generate realistic score based on metric
                base_score = np.random.beta(3, 1.5)  # Skewed towards higher scores
                threshold = self.validation_thresholds[metric]

                # Add some noise and ensure realistic distribution
                score = max(0.0, min(1.0, base_score + np.random.normal(0, 0.1)))
                passed = score >= threshold

                # Create validation result
                result = ValidationResult(
                    conversation_id=conversation_id,
                    metric=metric,
                    score=score,
                    passed=passed,
                    threshold=threshold,
                    validation_type="automated",
                    timestamp=datetime.now(UTC) - timedelta(days=np.random.randint(0, days_back)),
                    details={
                        "validation_method": "nlp_analysis",
                        "confidence": min(0.99, score + np.random.uniform(0, 0.2)),
                        "processing_time_ms": np.random.randint(50, 500),
                    },
                )

                validation_results.append(result)

            return validation_results

        except Exception:
            return []

    def _analyze_metric_validation(self, metric: str, results: list[ValidationResult]) -> ValidationAnalysis | None:
        """Analyze validation results for a specific metric"""
        try:
            if not results:
                return None

            # Basic statistics
            total_validations = len(results)
            passed_validations = len([r for r in results if r.passed])
            failed_validations = total_validations - passed_validations
            pass_rate = (passed_validations / total_validations) * 100

            # Score statistics
            scores = [r.score for r in results]
            average_score = np.mean(scores)

            # Score distribution
            score_distribution = {}
            for range_name, (min_score, max_score) in self.score_ranges.items():
                count = len([s for s in scores if min_score <= s < max_score])
                score_distribution[range_name] = count

            # Failure pattern analysis
            failure_patterns = self._analyze_failure_patterns(metric, [r for r in results if not r.passed])

            # Generate recommendations
            recommendations = self._generate_validation_recommendations(
                metric, pass_rate, average_score, failure_patterns
            )

            return ValidationAnalysis(
                metric=metric,
                total_validations=total_validations,
                passed_validations=passed_validations,
                failed_validations=failed_validations,
                pass_rate=pass_rate,
                average_score=average_score,
                score_distribution=score_distribution,
                failure_patterns=failure_patterns,
                recommendations=recommendations,
            )

        except Exception:
            return None

    def _analyze_failure_patterns(self, metric: str, failed_results: list[ValidationResult]) -> list[str]:
        """Analyze patterns in validation failures"""
        patterns = []

        if not failed_results:
            return patterns

        # Score-based patterns
        failed_scores = [r.score for r in failed_results]
        avg_failed_score = np.mean(failed_scores)
        threshold = self.validation_thresholds[metric]

        if avg_failed_score < threshold - 0.2:
            patterns.append(f"Consistently low scores (avg: {avg_failed_score:.3f}, threshold: {threshold:.3f})")
        elif avg_failed_score >= threshold - 0.05:
            patterns.append(f"Borderline failures (avg: {avg_failed_score:.3f}, threshold: {threshold:.3f})")

        # Temporal patterns
        failure_times = [r.timestamp for r in failed_results]
        if len(failure_times) > 10:
            # Check for clustering in time
            time_diffs = [
                (failure_times[i + 1] - failure_times[i]).total_seconds() for i in range(len(failure_times) - 1)
            ]
            avg_time_diff = np.mean(time_diffs)

            if avg_time_diff < 3600:  # Less than 1 hour average
                patterns.append("Failures clustered in time (potential systematic issue)")

        # Confidence patterns
        failed_confidences = [r.details.get("confidence", 0.5) for r in failed_results]
        avg_confidence = np.mean(failed_confidences)

        if avg_confidence < 0.6:
            patterns.append(f"Low confidence in failed validations (avg: {avg_confidence:.3f})")

        # Metric-specific patterns
        if metric == "therapeutic_accuracy":
            patterns.append("Review clinical training data quality and therapeutic technique validation")
        elif metric == "safety_score":
            patterns.append("CRITICAL: Safety validation failures require immediate attention")
        elif metric == "clinical_compliance":
            patterns.append("Review adherence to clinical guidelines and professional standards")

        return patterns

    def _generate_validation_recommendations(
        self,
        metric: str,
        pass_rate: float,
        average_score: float,
        failure_patterns: list[str],
    ) -> list[str]:
        """Generate recommendations based on validation analysis"""
        recommendations = []

        # Pass rate based recommendations
        if pass_rate < 70:
            recommendations.append(f"URGENT: Low pass rate ({pass_rate:.1f}%) requires immediate intervention")
            recommendations.append("Review and strengthen quality validation criteria")
        elif pass_rate < 85:
            recommendations.append(f"Moderate pass rate ({pass_rate:.1f}%) - consider process improvements")
        else:
            recommendations.append(f"Good pass rate ({pass_rate:.1f}%) - maintain current standards")

        # Score based recommendations
        if average_score < 0.6:
            recommendations.append(f"Low average score ({average_score:.3f}) - comprehensive quality review needed")
        elif average_score < 0.75:
            recommendations.append(f"Moderate average score ({average_score:.3f}) - targeted improvements recommended")

        # Metric-specific recommendations
        if metric == "therapeutic_accuracy":
            recommendations.extend(
                [
                    "Enhance clinical training datasets",
                    "Implement therapeutic technique validation",
                    "Review DSM-5 compliance patterns",
                ]
            )
        elif metric == "conversation_coherence":
            recommendations.extend(
                [
                    "Improve conversation flow validation",
                    "Enhance context awareness training",
                    "Review dialogue consistency patterns",
                ]
            )
        elif metric == "emotional_authenticity":
            recommendations.extend(
                [
                    "Strengthen emotional intelligence training",
                    "Improve sentiment analysis accuracy",
                    "Review emotional response patterns",
                ]
            )
        elif metric == "clinical_compliance":
            recommendations.extend(
                [
                    "Review clinical guideline adherence",
                    "Strengthen professional boundary validation",
                    "Implement ethics compliance checking",
                ]
            )
        elif metric == "safety_score":
            recommendations.extend(
                [
                    "PRIORITY: Strengthen crisis detection systems",
                    "Review safety protocol compliance",
                    "Implement enhanced harm prevention measures",
                ]
            )

        # Pattern-based recommendations
        if any("clustered in time" in pattern for pattern in failure_patterns):
            recommendations.append("Investigate systematic issues causing temporal failure clustering")

        if any("Low confidence" in pattern for pattern in failure_patterns):
            recommendations.append("Improve validation confidence through enhanced NLP models")

        return recommendations

    def create_validation_visualizations(self, analyses: dict[str, ValidationAnalysis]) -> dict[str, str]:
        """Create validation analysis visualizations"""

        viz_files = {}

        try:
            if not analyses:
                return viz_files

            # Set style
            plt.style.use("default")
            sns.set_palette("husl")

            # Create validation dashboard
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle("Quality Validation Analysis Dashboard", fontsize=16, fontweight="bold")

            # Pass rates by metric
            ax = axes[0, 0]
            metrics = list(analyses.keys())
            pass_rates = [analyses[m].pass_rate for m in metrics]

            bars = ax.bar(range(len(metrics)), pass_rates, alpha=0.7)
            ax.set_title("Pass Rates by Metric")
            ax.set_xlabel("Metrics")
            ax.set_ylabel("Pass Rate (%)")
            ax.set_xticks(range(len(metrics)))
            ax.set_xticklabels([m.replace("_", " ").title() for m in metrics], rotation=45, ha="right")
            ax.axhline(y=85, color="green", linestyle="--", alpha=0.7, label="Target (85%)")
            ax.axhline(y=70, color="orange", linestyle="--", alpha=0.7, label="Warning (70%)")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add value labels
            for bar, rate in zip(bars, pass_rates, strict=False):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 1,
                    f"{rate:.1f}%",
                    ha="center",
                    va="bottom",
                )

            # Average scores by metric
            ax = axes[0, 1]
            avg_scores = [analyses[m].average_score for m in metrics]
            thresholds = [self.validation_thresholds[m] for m in metrics]

            x = np.arange(len(metrics))
            width = 0.35

            ax.bar(x - width / 2, avg_scores, width, label="Average Score", alpha=0.7)
            ax.bar(x + width / 2, thresholds, width, label="Threshold", alpha=0.7)

            ax.set_title("Average Scores vs Thresholds")
            ax.set_xlabel("Metrics")
            ax.set_ylabel("Score")
            ax.set_xticks(x)
            ax.set_xticklabels([m.replace("_", " ").title() for m in metrics], rotation=45, ha="right")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Score distribution (stacked bar)
            ax = axes[1, 0]
            score_ranges = ["excellent", "good", "fair", "poor"]
            colors = ["green", "lightgreen", "orange", "red"]

            bottom = np.zeros(len(metrics))
            for i, score_range in enumerate(score_ranges):
                values = [analyses[m].score_distribution.get(score_range, 0) for m in metrics]
                ax.bar(
                    metrics,
                    values,
                    bottom=bottom,
                    label=score_range.title(),
                    color=colors[i],
                    alpha=0.7,
                )
                bottom += values

            ax.set_title("Score Distribution by Metric")
            ax.set_xlabel("Metrics")
            ax.set_ylabel("Count")
            ax.set_xticklabels([m.replace("_", " ").title() for m in metrics], rotation=45, ha="right")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Validation volume over time (synthetic)
            ax = axes[1, 1]
            days = list(range(1, 31))  # Last 30 days
            volumes = [np.random.poisson(100) for _ in days]  # Synthetic daily volumes

            ax.plot(days, volumes, marker="o", linewidth=2, markersize=4)
            ax.set_title("Daily Validation Volume (Last 30 Days)")
            ax.set_xlabel("Days Ago")
            ax.set_ylabel("Validations")
            ax.grid(True, alpha=0.3)

            # Add trend line
            z = np.polyfit(days, volumes, 1)
            p = np.poly1d(z)
            ax.plot(days, p(days), "--", alpha=0.8, color="red", label="Trend")
            ax.legend()

            plt.tight_layout()

            # Save dashboard
            dashboard_file = self.output_dir / "validation_analysis_dashboard.png"
            plt.savefig(dashboard_file, dpi=300, bbox_inches="tight")
            plt.close()

            viz_files["dashboard"] = str(dashboard_file)

            return viz_files

        except Exception:
            return {}

    def export_validation_report(self, analyses: dict[str, ValidationAnalysis], visualizations: dict[str, str]) -> str:
        """Export comprehensive validation analysis report"""

        try:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            report_file = self.output_dir / f"quality_validation_report_{timestamp}.json"

            # Create executive summary
            executive_summary = self._create_validation_summary(analyses)

            # Prepare export data
            backlog_conversion = self.convert_analysis_to_backlog_actions(analyses)
            linear_payload = build_linear_backlog_payload(backlog_conversion)
            linear_backlog_artifact = write_linear_backlog_artifact(
                linear_payload, self.linear_artifact_dir / "linear_backlog_payload.json"
            )
            linear_dispatch = self.linear_dispatcher.dispatch_backlog_actions(linear_payload)
            export_data = {
                "report_metadata": {
                    "generated_at": datetime.now(UTC).isoformat(),
                    "analyzer_version": "1.0.0",
                    "metrics_analyzed": len(analyses),
                    "analysis_period_days": 30,
                },
                "executive_summary": executive_summary,
                "detailed_analysis": {
                    metric: {
                        "total_validations": analysis.total_validations,
                        "passed_validations": analysis.passed_validations,
                        "failed_validations": analysis.failed_validations,
                        "pass_rate": analysis.pass_rate,
                        "average_score": analysis.average_score,
                        "score_distribution": analysis.score_distribution,
                        "failure_patterns": analysis.failure_patterns,
                        "recommendations": analysis.recommendations,
                    }
                    for metric, analysis in analyses.items()
                },
                "visualizations": visualizations,
                "quality_insights": self._generate_quality_insights(analyses),
                "backlog_conversion": self._serialize_backlog_conversion(backlog_conversion),
                "linear_backlog_artifact": linear_backlog_artifact,
                "linear_backlog_dispatch": linear_dispatch,
            }

            # Save report
            with open(report_file, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

            return str(report_file)

        except Exception:
            return ""

    def _create_validation_summary(self, analyses: dict[str, ValidationAnalysis]) -> dict[str, Any]:
        """Create executive summary of validation results"""
        try:
            if not analyses:
                return {"status": "no_data"}

            # Overall statistics
            total_validations = sum(a.total_validations for a in analyses.values())
            total_passed = sum(a.passed_validations for a in analyses.values())
            overall_pass_rate = (total_passed / total_validations) * 100 if total_validations > 0 else 0

            # Average scores
            avg_scores = [a.average_score for a in analyses.values()]
            overall_avg_score = np.mean(avg_scores)

            # Metrics performance
            high_performing = [m for m, a in analyses.items() if a.pass_rate >= 85]
            moderate_performing = [m for m, a in analyses.items() if 70 <= a.pass_rate < 85]
            low_performing = [m for m, a in analyses.items() if a.pass_rate < 70]

            # Critical issues
            critical_metrics = [m for m, a in analyses.items() if m == "safety_score" and a.pass_rate < 95]

            return {
                "status": "analysis_complete",
                "total_validations": total_validations,
                "overall_pass_rate": float(overall_pass_rate),
                "overall_average_score": float(overall_avg_score),
                "metrics_analyzed": len(analyses),
                "high_performing_metrics": len(high_performing),
                "moderate_performing_metrics": len(moderate_performing),
                "low_performing_metrics": len(low_performing),
                "critical_safety_issues": len(critical_metrics),
                "overall_health": "excellent"
                if overall_pass_rate >= 90
                else "good"
                if overall_pass_rate >= 80
                else "fair"
                if overall_pass_rate >= 70
                else "poor",
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _generate_quality_insights(self, analyses: dict[str, ValidationAnalysis]) -> list[str]:
        """Generate actionable quality insights"""
        insights = []

        try:
            # Performance insights
            best_metric = max(analyses.items(), key=lambda x: x[1].pass_rate)
            worst_metric = min(analyses.items(), key=lambda x: x[1].pass_rate)

            insights.append(
                f"Best performing metric: {best_metric[0].replace('_', ' ').title()} ({best_metric[1].pass_rate:.1f}% pass rate)"
            )
            insights.append(
                f"Lowest performing metric: {worst_metric[0].replace('_', ' ').title()} ({worst_metric[1].pass_rate:.1f}% pass rate)"
            )

            # Score insights
            high_score_metrics = [m for m, a in analyses.items() if a.average_score >= 0.8]
            if high_score_metrics:
                insights.append(
                    f"High-scoring metrics ({len(high_score_metrics)}): {', '.join([m.replace('_', ' ').title() for m in high_score_metrics])}"
                )

            # Failure pattern insights
            all_patterns = []
            for analysis in analyses.values():
                all_patterns.extend(analysis.failure_patterns)

            if any("systematic issue" in pattern.lower() for pattern in all_patterns):
                insights.append("Systematic issues detected - investigate infrastructure and processing pipeline")

            # Safety insights
            if "safety_score" in analyses:
                safety_analysis = analyses["safety_score"]
                if safety_analysis.pass_rate < 95:
                    insights.append(
                        f"CRITICAL: Safety validation pass rate ({safety_analysis.pass_rate:.1f}%) below required 95% threshold"
                    )

            # Volume insights
            total_validations = sum(a.total_validations for a in analyses.values())
            insights.append(f"Total validations processed: {total_validations:,} across {len(analyses)} metrics")

        except Exception:
            insights.append("Error generating insights - see logs for details")

        return insights

    def convert_analysis_to_backlog_actions(self, analyses: dict[str, ValidationAnalysis]) -> BacklogConversionResult:
        """Convert validation analysis metrics into backlog conversion recommendations."""
        if not analyses:
            return BacklogConversionResult(
                generated_at=datetime.now(UTC).isoformat(),
                metric_count=0,
                generated_changes=0,
                changes=[],
            )

        conversion_payload: dict[str, dict[str, Any]] = {}

        for metric, analysis in analyses.items():
            conversion_payload[metric] = {
                "pass_rate": analysis.pass_rate,
                "failed_validations": analysis.failed_validations,
                "failure_patterns": analysis.failure_patterns,
            }

        return self.backlog_converter.convert_from_validation_analysis(conversion_payload)

    def _serialize_backlog_conversion(self, result: BacklogConversionResult) -> dict[str, Any]:
        return {
            "generated_at": result.generated_at,
            "metric_count": result.metric_count,
            "generated_changes": result.generated_changes,
            "changes": [change.to_dict() for change in result.changes],
        }


def main():
    """Main execution function"""

    # Initialize analyzer
    analyzer = QualityValidationAnalyzer()

    # Analyze validation results
    analyses = analyzer.analyze_validation_results(days_back=30)

    if not analyses:
        return

    # Create visualizations
    visualizations = analyzer.create_validation_visualizations(analyses)

    # Export report
    analyzer.export_validation_report(analyses, visualizations)

    # Display summary

    # Show key findings
    for _metric, analysis in analyses.items():
        pass

    # Show top recommendations
    all_recommendations = []
    for analysis in analyses.values():
        all_recommendations.extend(analysis.recommendations[:2])  # Top 2 per metric

    if all_recommendations:
        for _rec in list(set(all_recommendations))[:5]:  # Top 5 unique recommendations
            pass


if __name__ == "__main__":
    main()
