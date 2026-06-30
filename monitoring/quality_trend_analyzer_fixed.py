#!/usr/bin/env python3
"""
Quality Trend Analysis and Reporting System - Fixed Version
Analyzes quality trends based on conversation content and metadata
"""

import sqlite3
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

warnings.simplefilter("default")


@dataclass
class QualityTrend:
    """Quality trend data structure"""

    metric: str
    period: str
    values: list[float]
    timestamps: list[datetime]
    trend_direction: str
    trend_strength: float
    statistical_significance: float


@dataclass
class TrendReport:
    """Comprehensive trend report"""

    period: str
    overall_trend: str
    key_insights: list[str]
    recommendations: list[str]
    metrics_summary: dict[str, Any]
    statistical_tests: dict[str, Any]


class QualityTrendAnalyzer:
    """Enterprise-grade quality trend analysis system"""

    def __init__(self, db_path: str = "database/conversations.db"):
        self.db_path = Path(db_path)
        self.output_dir = Path("monitoring/quality_trends")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Quality metrics we can derive from current data
        self.quality_metrics = [
            "conversation_length",  # Based on turn_count
            "content_richness",  # Based on word_count
            "processing_success",  # Based on processing_status
            "tier_distribution",  # Based on tier
            "language_consistency",  # Based on language
            "batch_quality",  # Based on batch processing
        ]

        # Trend analysis parameters
        self.trend_periods = ["daily", "weekly", "monthly"]
        self.min_data_points = 5
        self.significance_threshold = 0.05

    def analyze_quality_trends(self, period: str = "weekly", days_back: int = 90) -> dict[str, QualityTrend]:
        """Analyze quality trends over specified period"""

        try:
            # Get conversation data from database
            conversation_data = self._get_conversation_data(days_back)

            if not conversation_data:
                return {}

            # Group data by period
            grouped_data = self._group_by_period(conversation_data, period)

            # Analyze trends for each metric
            trends = {}
            for metric in self.quality_metrics:
                trend = self._analyze_metric_trend(grouped_data, metric, period)
                if trend:
                    trends[metric] = trend

            return trends

        except Exception:
            return {}

    def _get_conversation_data(self, days_back: int) -> list[dict]:
        """Get conversation data from database"""
        try:
            conn = sqlite3.connect(self.db_path)

            # Calculate date threshold
            date_threshold = datetime.now(UTC) - timedelta(days=days_back)

            query = """
            SELECT
                created_at,
                dataset_source,
                tier,
                turn_count,
                word_count,
                character_count,
                language,
                processing_status,
                batch_id,
                processing_version
            FROM conversations
            WHERE created_at >= ?
            ORDER BY created_at
            """

            cursor = conn.execute(query, (date_threshold.isoformat(),))
            columns = [desc[0] for desc in cursor.description]

            data = []
            for row in cursor.fetchall():
                record = dict(zip(columns, row, strict=False))
                record["created_at"] = datetime.fromisoformat(record["created_at"])
                data.append(record)

            conn.close()
            return data

        except Exception:
            return []

    def _group_by_period(self, data: list[dict], period: str) -> dict[str, list[dict]]:
        """Group data by time period"""
        grouped = {}

        for record in data:
            timestamp = record["created_at"]

            if period == "daily":
                key = timestamp.strftime("%Y-%m-%d")
            elif period == "weekly":
                # Get Monday of the week
                monday = timestamp - timedelta(days=timestamp.weekday())
                key = monday.strftime("%Y-%m-%d")
            elif period == "monthly":
                key = timestamp.strftime("%Y-%m")
            else:
                key = timestamp.strftime("%Y-%m-%d")

            if key not in grouped:
                grouped[key] = []
            grouped[key].append(record)

        return grouped

    def _analyze_metric_trend(
        self, grouped_data: dict[str, list[dict]], metric: str, period: str
    ) -> QualityTrend | None:
        """Analyze trend for specific metric"""
        try:
            # Extract metric values and timestamps
            period_values = []
            timestamps = []

            for period_key, records in sorted(grouped_data.items()):
                # Calculate metric for this period
                metric_value = self._calculate_metric_value(records, metric)
                if metric_value is not None:
                    period_values.append(metric_value)
                    # Convert period key back to datetime
                    if period in {"daily", "weekly"}:
                        timestamps.append(datetime.strptime(period_key, "%Y-%m-%d"))
                    elif period == "monthly":
                        timestamps.append(datetime.strptime(f"{period_key}-01", "%Y-%m-%d"))

            if len(period_values) < self.min_data_points:
                return None

            # Perform trend analysis
            trend_direction, trend_strength, p_value = self._calculate_trend_statistics(period_values)

            return QualityTrend(
                metric=metric,
                period=period,
                values=period_values,
                timestamps=timestamps,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                statistical_significance=p_value,
            )

        except Exception:
            return None

    def _calculate_metric_value(self, records: list[dict], metric: str) -> float | None:
        """Calculate metric value for a period"""
        try:
            if not records:
                return None

            if metric == "conversation_length":
                # Average turn count
                turn_counts = [r["turn_count"] for r in records if r["turn_count"]]
                return np.mean(turn_counts) if turn_counts else None

            if metric == "content_richness":
                # Average word count
                word_counts = [r["word_count"] for r in records if r["word_count"]]
                return np.mean(word_counts) if word_counts else None

            if metric == "processing_success":
                # Percentage of successful processing
                total = len(records)
                successful = len([r for r in records if r["processing_status"] == "processed"])
                return (successful / total) * 100 if total > 0 else None

            if metric == "tier_distribution":
                # Priority tier percentage (higher is better)
                total = len(records)
                priority = len([r for r in records if r["tier"] and "priority" in r["tier"]])
                return (priority / total) * 100 if total > 0 else None

            if metric == "language_consistency":
                # Percentage of English conversations
                total = len(records)
                english = len([r for r in records if r["language"] == "en"])
                return (english / total) * 100 if total > 0 else None

            if metric == "batch_quality":
                # Percentage with batch processing
                total = len(records)
                batched = len([r for r in records if r["batch_id"]])
                return (batched / total) * 100 if total > 0 else None

            return None

        except Exception:
            return None

    def _calculate_trend_statistics(self, values: list[float]) -> tuple[str, float, float]:
        """Calculate trend statistics using linear regression"""
        try:
            x = np.arange(len(values))
            y = np.array(values)

            # Linear regression
            slope, _intercept, r_value, p_value, _std_err = stats.linregress(x, y)

            # Determine trend direction
            if abs(slope) < 0.001:  # Very small slope
                direction = "stable"
            elif slope > 0:
                direction = "improving"
            else:
                direction = "declining"

            # Trend strength (R-squared)
            strength = r_value**2

            return direction, strength, p_value

        except Exception:
            return "unknown", 0.0, 1.0

    def generate_trend_report(self, trends: dict[str, QualityTrend], period: str = "weekly") -> TrendReport:
        """Generate comprehensive trend report"""

        try:
            # Overall trend assessment
            overall_trend = self._assess_overall_trend(trends)

            # Key insights
            insights = self._extract_key_insights(trends)

            # Recommendations
            recommendations = self._generate_recommendations(trends)

            # Metrics summary
            metrics_summary = self._create_metrics_summary(trends)

            # Statistical tests
            statistical_tests = self._perform_statistical_tests(trends)

            return TrendReport(
                period=period,
                overall_trend=overall_trend,
                key_insights=insights,
                recommendations=recommendations,
                metrics_summary=metrics_summary,
                statistical_tests=statistical_tests,
            )


        except Exception:
            return TrendReport(
                period=period,
                overall_trend="unknown",
                key_insights=[],
                recommendations=[],
                metrics_summary={},
                statistical_tests={},
            )

    def _assess_overall_trend(self, trends: dict[str, QualityTrend]) -> str:
        """Assess overall quality trend"""
        if not trends:
            return "insufficient_data"

        improving_count = 0
        declining_count = 0
        stable_count = 0

        for trend in trends.values():
            if trend.statistical_significance < self.significance_threshold:
                if trend.trend_direction == "improving":
                    improving_count += 1
                elif trend.trend_direction == "declining":
                    declining_count += 1
                else:
                    stable_count += 1

        total_significant = improving_count + declining_count + stable_count

        if total_significant == 0:
            return "stable"
        if improving_count > declining_count:
            return "improving"
        if declining_count > improving_count:
            return "declining"
        return "mixed"

    def _extract_key_insights(self, trends: dict[str, QualityTrend]) -> list[str]:
        """Extract key insights from trends"""
        insights = []

        # Find strongest trends
        significant_trends = [
            (name, trend)
            for name, trend in trends.items()
            if trend.statistical_significance < self.significance_threshold
        ]

        # Sort by trend strength
        significant_trends.sort(key=lambda x: x[1].trend_strength, reverse=True)

        for name, trend in significant_trends[:5]:  # Top 5 insights
            direction = trend.trend_direction
            strength = trend.trend_strength
            current_value = trend.values[-1] if trend.values else 0

            if direction == "improving":
                insights.append(
                    f"{name.replace('_', ' ').title()} shows significant improvement (R² = {strength:.3f}, Current: {current_value:.1f})"
                )
            elif direction == "declining":
                insights.append(
                    f"{name.replace('_', ' ').title()} shows concerning decline (R² = {strength:.3f}, Current: {current_value:.1f})"
                )
            else:
                insights.append(
                    f"{name.replace('_', ' ').title()} remains stable with strong consistency (R² = {strength:.3f})"
                )

        # Add volatility insights
        volatile_metrics = [
            name
            for name, trend in trends.items()
            if np.std(trend.values) > np.mean(trend.values) * 0.2  # High coefficient of variation
        ]

        if volatile_metrics:
            insights.append(
                f"High volatility detected in: {', '.join([m.replace('_', ' ') for m in volatile_metrics])}"
            )

        return insights

    def _generate_recommendations(self, trends: dict[str, QualityTrend]) -> list[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Find declining metrics
        declining_metrics = [
            name
            for name, trend in trends.items()
            if trend.trend_direction == "declining" and trend.statistical_significance < self.significance_threshold
        ]

        for metric in declining_metrics:
            if metric == "conversation_length":
                recommendations.append("Review conversation depth - consider enhancing multi-turn dialogue training")
            elif metric == "content_richness":
                recommendations.append("Improve content quality - focus on more detailed and informative responses")
            elif metric == "processing_success":
                recommendations.append("URGENT: Address processing failures - review data pipeline stability")
            elif metric == "tier_distribution":
                recommendations.append("Optimize dataset prioritization - ensure high-quality data sources")
            elif metric == "language_consistency":
                recommendations.append("Review language detection and filtering processes")
            elif metric == "batch_quality":
                recommendations.append("Improve batch processing efficiency and organization")

        # Find improving metrics to reinforce
        improving_metrics = [
            name
            for name, trend in trends.items()
            if trend.trend_direction == "improving" and trend.statistical_significance < self.significance_threshold
        ]

        if improving_metrics:
            improving_names = [m.replace("_", " ") for m in improving_metrics]
            recommendations.append(f"Continue successful practices that improved: {', '.join(improving_names)}")

        # General recommendations
        if len(declining_metrics) > len(improving_metrics):
            recommendations.append("Consider comprehensive data quality review and pipeline optimization")

        return recommendations

    def _create_metrics_summary(self, trends: dict[str, QualityTrend]) -> dict[str, Any]:
        """Create metrics summary"""
        summary = {}

        for name, trend in trends.items():
            summary[name] = {
                "current_value": trend.values[-1] if trend.values else 0,
                "trend_direction": trend.trend_direction,
                "trend_strength": trend.trend_strength,
                "statistical_significance": trend.statistical_significance,
                "is_significant": trend.statistical_significance < self.significance_threshold,
                "volatility": np.std(trend.values) if trend.values else 0,
                "min_value": min(trend.values) if trend.values else 0,
                "max_value": max(trend.values) if trend.values else 0,
                "mean_value": np.mean(trend.values) if trend.values else 0,
                "coefficient_of_variation": (np.std(trend.values) / np.mean(trend.values))
                if trend.values and np.mean(trend.values) > 0
                else 0,
            }

        return summary

    def _perform_statistical_tests(self, trends: dict[str, QualityTrend]) -> dict[str, Any]:
        """Perform statistical tests on trends"""
        tests = {}

        for name, trend in trends.items():
            if len(trend.values) < 3:
                continue

            # Normality test
            try:
                shapiro_stat, shapiro_p = stats.shapiro(trend.values)
                tests[f"{name}_normality"] = {
                    "test": "Shapiro-Wilk",
                    "statistic": shapiro_stat,
                    "p_value": shapiro_p,
                    "is_normal": shapiro_p > 0.05,
                }
            except:
                pass

            # Trend test (Mann-Kendall)
            try:
                tau, p_value = stats.kendalltau(range(len(trend.values)), trend.values)
                tests[f"{name}_trend"] = {
                    "test": "Mann-Kendall",
                    "tau": tau,
                    "p_value": p_value,
                    "has_trend": p_value < 0.05,
                }
            except:
                pass

        return tests


def main():
    """Main execution function"""

    # Initialize analyzer
    analyzer = QualityTrendAnalyzer()

    # Analyze trends for different periods
    periods = ["daily", "weekly"]

    for period in periods:
        # Analyze trends
        trends = analyzer.analyze_quality_trends(period=period, days_back=30)

        if not trends:
            continue

        # Generate report
        report = analyzer.generate_trend_report(trends, period)

        # Show key insights
        if report.key_insights:
            for _insight in report.key_insights[:3]:
                pass

        # Show recommendations
        if report.recommendations:
            for _rec in report.recommendations[:3]:
                pass


if __name__ == "__main__":
    main()
