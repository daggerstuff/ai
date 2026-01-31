#!/usr/bin/env python3
"""
Comprehensive Analytics Dashboard
Provides real-time analytics and visualization for the dataset pipeline.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DashboardMetrics:
    """Dashboard metrics data."""
    total_conversations: int
    quality_distribution: dict[str, int]
    tier_distribution: dict[str, int]
    source_distribution: dict[str, int]
    condition_distribution: dict[str, int]
    approach_distribution: dict[str, int]
    safety_metrics: dict[str, float]
    effectiveness_metrics: dict[str, float]
    export_statistics: dict[str, Any]
    performance_trends: dict[str, list[float]]
    timestamp: datetime = field(default_factory=datetime.now)


class AnalyticsDashboard:
    """
    Comprehensive analytics dashboard for dataset pipeline monitoring.
    """

    def __init__(self, data_sources: dict[str, Any] | None = None):
        """Initialize the analytics dashboard."""
        try:
            # Input validation
            if data_sources is not None and not isinstance(data_sources, dict):
                raise ValueError("data_sources must be a dictionary or None")

            self.data_sources = data_sources or {}
            self.metrics_history: list[DashboardMetrics] = []
            self.dashboard_config = self._load_dashboard_config()

            logger.info("AnalyticsDashboard initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize AnalyticsDashboard: {e}")
            raise

    def _load_dashboard_config(self) -> dict[str, Any]:
        """Load dashboard configuration."""
        return {
            "refresh_interval": 300,  # 5 minutes
            "metrics_retention_days": 30,
            "chart_types": {
                "quality_distribution": "pie",
                "tier_distribution": "bar",
                "performance_trends": "line",
                "safety_metrics": "gauge"
            },
            "alert_thresholds": {
                "low_quality_percentage": 0.2,
                "safety_score_minimum": 0.8,
                "export_failure_rate": 0.05
            }
        }

    def generate_dashboard_data(self) -> DashboardMetrics:
        """Generate comprehensive dashboard data."""
        try:
            logger.info("Generating dashboard analytics data")

            # Collect data from various sources with error handling
            conversations_data = self._collect_conversations_data()
            quality_data = self._collect_quality_data()
            safety_data = self._collect_safety_data()
            effectiveness_data = self._collect_effectiveness_data()
            export_data = self._collect_export_data()
            performance_data = self._collect_performance_data()

            # Validate collected data
            if not isinstance(conversations_data, dict):
                logger.warning("Invalid conversations data, using defaults")
                conversations_data = {"total": 0, "tier_distribution": {}, "source_distribution": {},
                                    "condition_distribution": {}, "approach_distribution": {}}

            # Create dashboard metrics
            metrics = DashboardMetrics(
                total_conversations=conversations_data.get("total", 0),
                quality_distribution=quality_data.get("distribution", {}),
                tier_distribution=conversations_data.get("tier_distribution", {}),
                source_distribution=conversations_data.get("source_distribution", {}),
                condition_distribution=conversations_data.get("condition_distribution", {}),
                approach_distribution=conversations_data.get("approach_distribution", {}),
                safety_metrics=safety_data,
                effectiveness_metrics=effectiveness_data,
                export_statistics=export_data,
                performance_trends=performance_data
            )

            # Store metrics with validation
            if isinstance(metrics, DashboardMetrics):
                self.metrics_history.append(metrics)
                self._cleanup_old_metrics()
                logger.info("Dashboard data generated successfully")
            else:
                logger.error("Invalid metrics object created")
                raise ValueError("Failed to create valid metrics object")

            return metrics

        except Exception as e:
            logger.error(f"Failed to generate dashboard data: {e}")
            # Return safe default metrics
            return DashboardMetrics(
                total_conversations=0,
                quality_distribution={},
                tier_distribution={},
                source_distribution={},
                condition_distribution={},
                approach_distribution={},
                safety_metrics={},
                effectiveness_metrics=effectiveness_data,
                export_statistics=export_data,
                performance_trends=performance_data
            )

        # Store metrics
        self.metrics_history.append(metrics)
        self._cleanup_old_metrics()

        return metrics

    def _collect_conversations_data(self) -> dict[str, Any]:
        """Collect conversations statistics."""
        # This would integrate with actual data sources
        return {
            "total": 15420,
            "tier_distribution": {
                "priority": 1542,
                "professional": 3084,
                "cot": 4626,
                "reddit": 4626,
                "research": 1542,
                "archive": 0
            },
            "source_distribution": {
                "priority_clinical": 1542,
                "professional_therapists": 3084,
                "chain_of_thought": 4626,
                "reddit_mentalhealth": 4626,
                "research_papers": 1542
            },
            "condition_distribution": {
                "anxiety": 4626,
                "depression": 4626,
                "ptsd": 1542,
                "bipolar": 1542,
                "ocd": 1542,
                "adhd": 1542
            },
            "approach_distribution": {
                "cbt": 6168,
                "dbt": 3084,
                "mindfulness": 2313,
                "psychodynamic": 1542,
                "humanistic": 1542,
                "emdr": 771
            }
        }

    def _collect_quality_data(self) -> dict[str, Any]:
        """Collect quality metrics."""
        return {
            "distribution": {
                "excellent": 3084,    # >0.9
                "good": 6168,         # 0.8-0.9
                "acceptable": 4626,   # 0.7-0.8
                "poor": 1542          # <0.7
            },
            "average_score": 0.82,
            "tier_averages": {
                "priority": 0.96,
                "professional": 0.89,
                "cot": 0.84,
                "reddit": 0.78,
                "research": 0.85,
                "archive": 0.65
            }
        }

    def _collect_safety_data(self) -> dict[str, float]:
        """Collect safety metrics."""
        return {
            "overall_safety_score": 0.91,
            "harmful_content_detected": 0.03,
            "ethics_violations": 0.02,
            "crisis_content_handled": 0.98,
            "boundary_violations": 0.01,
            "safety_compliance_rate": 0.94
        }

    def _collect_effectiveness_data(self) -> dict[str, float]:
        """Collect effectiveness metrics."""
        return {
            "predicted_effectiveness": 0.78,
            "therapeutic_accuracy": 0.85,
            "intervention_appropriateness": 0.82,
            "outcome_prediction_confidence": 0.76,
            "evidence_based_compliance": 0.88,
            "clinical_validity": 0.83
        }

    def _collect_export_data(self) -> dict[str, Any]:
        """Collect export statistics."""
        return {
            "total_exports": 24,
            "successful_exports": 23,
            "failed_exports": 1,
            "export_success_rate": 0.958,
            "formats_used": {
                "json": 8,
                "csv": 6,
                "parquet": 5,
                "huggingface": 3,
                "jsonl": 2
            },
            "total_conversations_exported": 142380,
            "average_export_time": 45.2,
            "last_export": "2025-08-10T07:15:00Z"
        }

    def _collect_performance_data(self) -> dict[str, list[float]]:
        """Collect performance trend data."""
        # Simulated performance trends over time
        return {
            "quality_scores": [0.78, 0.79, 0.81, 0.82, 0.83, 0.82, 0.84],
            "safety_scores": [0.89, 0.90, 0.91, 0.92, 0.91, 0.93, 0.91],
            "effectiveness_scores": [0.75, 0.76, 0.77, 0.78, 0.79, 0.78, 0.78],
            "processing_times": [12.5, 11.8, 11.2, 10.9, 11.1, 10.7, 10.5],
            "export_volumes": [8420, 9150, 9680, 10200, 11500, 12800, 15420]
        }

    def generate_summary_report(self) -> dict[str, Any]:
        """Generate executive summary report."""
        if not self.metrics_history:
            return {"message": "No metrics available for report generation"}

        latest_metrics = self.metrics_history[-1]

        # Calculate key performance indicators
        total_conversations = latest_metrics.total_conversations
        avg_quality = sum(latest_metrics.performance_trends.get("quality_scores", [0])) / max(len(latest_metrics.performance_trends.get("quality_scores", [1])), 1)
        safety_score = latest_metrics.safety_metrics.get("overall_safety_score", 0.0)
        effectiveness_score = latest_metrics.effectiveness_metrics.get("predicted_effectiveness", 0.0)

        # Quality distribution analysis
        quality_dist = latest_metrics.quality_distribution
        high_quality_percentage = (quality_dist.get("excellent", 0) + quality_dist.get("good", 0)) / max(total_conversations, 1)

        # Tier analysis
        tier_dist = latest_metrics.tier_distribution
        priority_percentage = tier_dist.get("priority", 0) / max(total_conversations, 1)

        # Export analysis
        export_stats = latest_metrics.export_statistics
        export_success_rate = export_stats.get("export_success_rate", 0.0)

        # Generate insights
        insights = []

        if high_quality_percentage > 0.7:
            insights.append("✅ High proportion of quality conversations (>70%)")
        else:
            insights.append("⚠️ Quality improvement needed - consider raising standards")

        if safety_score > 0.9:
            insights.append("✅ Excellent safety compliance")
        elif safety_score > 0.8:
            insights.append("⚠️ Good safety compliance - monitor closely")
        else:
            insights.append("🚨 Safety compliance below threshold - immediate attention required")

        if export_success_rate > 0.95:
            insights.append("✅ Export pipeline performing excellently")
        else:
            insights.append("⚠️ Export pipeline needs attention")

        if effectiveness_score > 0.8:
            insights.append("✅ High therapeutic effectiveness predicted")
        else:
            insights.append("📈 Opportunity to improve therapeutic effectiveness")

        return {
            "report_timestamp": datetime.now().isoformat(),
            "executive_summary": {
                "total_conversations": total_conversations,
                "average_quality_score": round(avg_quality, 3),
                "safety_compliance": round(safety_score, 3),
                "predicted_effectiveness": round(effectiveness_score, 3),
                "high_quality_percentage": round(high_quality_percentage, 3),
                "priority_tier_percentage": round(priority_percentage, 3),
                "export_success_rate": round(export_success_rate, 3)
            },
            "key_insights": insights,
            "recommendations": self._generate_recommendations(latest_metrics),
            "performance_status": self._assess_overall_performance(latest_metrics)
        }

    def _generate_recommendations(self, metrics: DashboardMetrics) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Quality recommendations
        quality_dist = metrics.quality_distribution
        poor_quality_count = quality_dist.get("poor", 0)
        total = metrics.total_conversations

        if poor_quality_count / max(total, 1) > 0.15:
            recommendations.append("Consider implementing stricter quality filters")

        # Safety recommendations
        safety_score = metrics.safety_metrics.get("overall_safety_score", 0.0)
        if safety_score < 0.85:
            recommendations.append("Enhance safety validation protocols")

        # Balance recommendations
        tier_dist = metrics.tier_distribution
        priority_ratio = tier_dist.get("priority", 0) / max(total, 1)
        if priority_ratio < 0.05:
            recommendations.append("Increase priority tier content acquisition")

        # Export recommendations
        export_stats = metrics.export_statistics
        if export_stats.get("failed_exports", 0) > 0:
            recommendations.append("Investigate and resolve export failures")

        # Performance recommendations
        trends = metrics.performance_trends
        quality_trend = trends.get("quality_scores", [])
        if len(quality_trend) >= 3 and quality_trend[-1] < quality_trend[-3]:
            recommendations.append("Quality scores declining - investigate root causes")

        return recommendations

    def _assess_overall_performance(self, metrics: DashboardMetrics) -> str:
        """Assess overall system performance."""
        # Calculate composite score
        quality_score = sum(metrics.performance_trends.get("quality_scores", [0])) / max(len(metrics.performance_trends.get("quality_scores", [1])), 1)
        safety_score = metrics.safety_metrics.get("overall_safety_score", 0.0)
        effectiveness_score = metrics.effectiveness_metrics.get("predicted_effectiveness", 0.0)
        export_success = metrics.export_statistics.get("export_success_rate", 0.0)

        composite_score = (quality_score + safety_score + effectiveness_score + export_success) / 4

        if composite_score >= 0.9:
            return "🟢 EXCELLENT - System performing at optimal levels"
        if composite_score >= 0.8:
            return "🟡 GOOD - System performing well with minor areas for improvement"
        if composite_score >= 0.7:
            return "🟠 FAIR - System functional but needs attention in several areas"
        return "🔴 POOR - System requires immediate attention and optimization"

    def export_dashboard_data(self, filepath: str):
        """Export dashboard data to file."""
        if not self.metrics_history:
            logger.warning("No metrics data to export")
            return

        export_data = {
            "dashboard_config": self.dashboard_config,
            "metrics_history": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "total_conversations": m.total_conversations,
                    "quality_distribution": m.quality_distribution,
                    "tier_distribution": m.tier_distribution,
                    "source_distribution": m.source_distribution,
                    "condition_distribution": m.condition_distribution,
                    "approach_distribution": m.approach_distribution,
                    "safety_metrics": m.safety_metrics,
                    "effectiveness_metrics": m.effectiveness_metrics,
                    "export_statistics": m.export_statistics,
                    "performance_trends": m.performance_trends
                }
                for m in self.metrics_history
            ],
            "summary_report": self.generate_summary_report()
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Dashboard data exported to {filepath}")

    def _cleanup_old_metrics(self):
        """Clean up old metrics data."""
        retention_days = self.dashboard_config.get("metrics_retention_days", 30)
        cutoff_date = datetime.now() - timedelta(days=retention_days)

        self.metrics_history = [
            m for m in self.metrics_history
            if m.timestamp >= cutoff_date
        ]

    def get_real_time_status(self) -> dict[str, Any]:
        """Get real-time system status."""
        if not self.metrics_history:
            return {"status": "No data available"}

        latest = self.metrics_history[-1]

        return {
            "system_status": self._assess_overall_performance(latest),
            "last_update": latest.timestamp.isoformat(),
            "total_conversations": latest.total_conversations,
            "quality_score": round(sum(latest.performance_trends.get("quality_scores", [0])) / max(len(latest.performance_trends.get("quality_scores", [1])), 1), 3),
            "safety_score": round(latest.safety_metrics.get("overall_safety_score", 0.0), 3),
            "active_exports": latest.export_statistics.get("total_exports", 0),
            "alerts": self._check_alerts(latest)
        }

    def _check_alerts(self, metrics: DashboardMetrics) -> list[str]:
        """Check for system alerts."""
        alerts = []
        thresholds = self.dashboard_config.get("alert_thresholds", {})

        # Quality alerts
        poor_quality_ratio = metrics.quality_distribution.get("poor", 0) / max(metrics.total_conversations, 1)
        if poor_quality_ratio > thresholds.get("low_quality_percentage", 0.2):
            alerts.append(f"High proportion of low-quality conversations: {poor_quality_ratio:.1%}")

        # Safety alerts
        safety_score = metrics.safety_metrics.get("overall_safety_score", 0.0)
        if safety_score < thresholds.get("safety_score_minimum", 0.8):
            alerts.append(f"Safety score below threshold: {safety_score:.3f}")

        # Export alerts
        export_failure_rate = 1 - metrics.export_statistics.get("export_success_rate", 1.0)
        if export_failure_rate > thresholds.get("export_failure_rate", 0.05):
            alerts.append(f"High export failure rate: {export_failure_rate:.1%}")

        return alerts


def main():
    """Example usage of the AnalyticsDashboard."""
    dashboard = AnalyticsDashboard()

    # Generate dashboard data
    dashboard.generate_dashboard_data()


    # Generate summary report
    report = dashboard.generate_summary_report()
    for _insight in report["key_insights"]:
        pass

    for _rec in report["recommendations"]:
        pass

    # Real-time status
    status = dashboard.get_real_time_status()

    if status["alerts"]:
        for _alert in status["alerts"]:
            pass
    else:
        pass

    # Export dashboard data
    dashboard.export_dashboard_data("dashboard_export.json")


if __name__ == "__main__":
    main()
