#!/usr/bin/env python3
"""
Quality Improvement Tracking Demo
Demonstrates improvement tracking with synthetic baseline and current data
"""

import json
import random
import sqlite3
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.simplefilter("default")


@dataclass
class QualityImprovement:
    """Quality improvement tracking data"""

    metric: str
    baseline_value: float
    current_value: float
    improvement_percentage: float
    improvement_direction: str
    confidence_level: float
    recommendation: str


class QualityImprovementDemo:
    """Demo quality improvement tracking system"""

    def __init__(self, db_path: str = "database/conversations.db"):
        self.db_path = Path(db_path)
        self.output_dir = Path("monitoring/quality_improvements")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Quality metrics to track
        self.quality_metrics = [
            "conversation_length",
            "content_richness",
            "processing_efficiency",
            "tier_quality",
            "dataset_diversity",
        ]

    def create_demo_improvements(self) -> dict[str, QualityImprovement]:
        """Create demo improvement data"""

        try:
            # Get base data from database
            base_data = self._get_base_data()

            if not base_data:
                return {}

            # Create synthetic improvements
            improvements = {}

            for metric in self.quality_metrics:
                improvement = self._create_synthetic_improvement(base_data, metric)
                if improvement:
                    improvements[metric] = improvement

            return improvements

        except Exception:
            return {}

    def _get_base_data(self) -> list[dict]:
        """Get base data from database"""
        try:
            conn = sqlite3.connect(self.db_path)

            query = """
            SELECT
                dataset_source,
                tier,
                turn_count,
                word_count,
                processing_status
            FROM conversations
            WHERE turn_count IS NOT NULL
            AND word_count IS NOT NULL
            LIMIT 1000
            """

            cursor = conn.execute(query)
            columns = [desc[0] for desc in cursor.description]

            data = []
            for row in cursor.fetchall():
                record = dict(zip(columns, row, strict=False))
                data.append(record)

            conn.close()
            return data

        except Exception:
            return []

    def _create_synthetic_improvement(self, base_data: list[dict], metric: str) -> QualityImprovement | None:
        """Create synthetic improvement for a metric"""
        try:
            # Calculate baseline value from actual data
            baseline_value = self._calculate_metric_value(base_data, metric)

            if baseline_value is None:
                return None

            # Create synthetic current value with some improvement/decline
            improvement_scenarios = [
                ("improving", random.uniform(5, 25)),  # 5-25% improvement
                ("declining", random.uniform(-20, -3)),  # 3-20% decline
                ("stable", random.uniform(-2, 2)),  # ±2% stable
            ]

            # Weight scenarios (more likely to improve for demo)
            scenario_weights = [0.6, 0.2, 0.2]  # 60% improve, 20% decline, 20% stable
            _direction, change_percent = random.choices(improvement_scenarios, weights=scenario_weights)[0]

            # Calculate current value
            current_value = baseline_value * (1 + change_percent / 100)

            # Ensure realistic bounds
            if metric in ["processing_efficiency", "tier_quality"]:
                current_value = max(0, min(100, current_value))
            elif metric == "conversation_length":
                current_value = max(1, current_value)
            elif metric == "content_richness":
                current_value = max(10, current_value)
            elif metric == "dataset_diversity":
                current_value = max(1, current_value)

            # Recalculate actual improvement percentage
            improvement_percentage = ((current_value - baseline_value) / baseline_value) * 100

            # Determine final direction based on actual change
            if improvement_percentage > 2:
                final_direction = "improving"
            elif improvement_percentage < -2:
                final_direction = "declining"
            else:
                final_direction = "stable"

            # Calculate confidence level
            confidence_level = min(0.95, 0.6 + abs(improvement_percentage) / 100)

            # Generate recommendation
            recommendation = self._generate_improvement_recommendation(metric, improvement_percentage, final_direction)

            return QualityImprovement(
                metric=metric,
                baseline_value=baseline_value,
                current_value=current_value,
                improvement_percentage=improvement_percentage,
                improvement_direction=final_direction,
                confidence_level=confidence_level,
                recommendation=recommendation,
            )

        except Exception:
            return None

    def _calculate_metric_value(self, data: list[dict], metric: str) -> float | None:
        """Calculate metric value for a dataset"""
        try:
            if not data:
                return None

            if metric == "conversation_length":
                values = [r["turn_count"] for r in data if r["turn_count"]]
                return np.mean(values) if values else None

            if metric == "content_richness":
                values = [r["word_count"] for r in data if r["word_count"]]
                return np.mean(values) if values else None

            if metric == "processing_efficiency":
                total = len(data)
                successful = len([r for r in data if r["processing_status"] == "processed"])
                return (successful / total) * 100 if total > 0 else None

            if metric == "tier_quality":
                total = len(data)
                priority = len([r for r in data if r["tier"] and "priority" in str(r["tier"])])
                return (priority / total) * 100 if total > 0 else None

            if metric == "dataset_diversity":
                unique_datasets = len({r["dataset_source"] for r in data if r["dataset_source"]})
                return float(unique_datasets)

            return None

        except Exception:
            return None

    def _generate_improvement_recommendation(self, metric: str, improvement_percentage: float, direction: str) -> str:
        """Generate improvement recommendation"""
        try:
            if direction == "improving":
                if improvement_percentage > 10:
                    return f"Excellent progress in {metric.replace('_', ' ')}! Continue current practices and consider scaling"
                if improvement_percentage > 5:
                    return f"Good improvement in {metric.replace('_', ' ')}. Monitor trends and maintain focus"
                return f"Slight improvement in {metric.replace('_', ' ')}. Continue current approach"

            if direction == "declining":
                if abs(improvement_percentage) > 10:
                    return f"URGENT: Significant decline in {metric.replace('_', ' ')}. Immediate intervention required"
                if abs(improvement_percentage) > 5:
                    return f"Concerning decline in {metric.replace('_', ' ')}. Review and adjust processes"
                return f"Minor decline in {metric.replace('_', ' ')}. Monitor closely and investigate causes"

            # stable
            return f"{metric.replace('_', ' ').title()} remains stable. Consider optimization opportunities"

        except Exception:
            return "Unable to generate recommendation"

    def create_improvement_visualizations(self, improvements: dict[str, QualityImprovement]) -> dict[str, str]:
        """Create improvement tracking visualizations"""

        viz_files = {}

        try:
            # Set style
            plt.style.use("default")
            sns.set_palette("husl")

            # Create improvement dashboard
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(
                "Quality Improvement Tracking Dashboard - Demo",
                fontsize=16,
                fontweight="bold",
            )

            # Improvement percentages
            ax = axes[0, 0]
            metrics = list(improvements.keys())
            percentages = [imp.improvement_percentage for imp in improvements.values()]
            colors = ["green" if p > 2 else "red" if p < -2 else "orange" for p in percentages]

            bars = ax.bar(range(len(metrics)), percentages, color=colors, alpha=0.7)
            ax.set_title("Improvement Percentages by Metric")
            ax.set_xlabel("Metrics")
            ax.set_ylabel("Improvement %")
            ax.set_xticks(range(len(metrics)))
            ax.set_xticklabels([m.replace("_", " ").title() for m in metrics], rotation=45, ha="right")
            ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, percentage in zip(bars, percentages, strict=False):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + (0.5 if height >= 0 else -1),
                    f"{percentage:.1f}%",
                    ha="center",
                    va="bottom" if height >= 0 else "top",
                )

            # Current vs Baseline values
            ax = axes[0, 1]
            baseline_values = [imp.baseline_value for imp in improvements.values()]
            current_values = [imp.current_value for imp in improvements.values()]

            x = np.arange(len(metrics))
            width = 0.35

            ax.bar(x - width / 2, baseline_values, width, label="Baseline", alpha=0.7)
            ax.bar(x + width / 2, current_values, width, label="Current", alpha=0.7)

            ax.set_title("Current vs Baseline Values")
            ax.set_xlabel("Metrics")
            ax.set_ylabel("Value")
            ax.set_xticks(x)
            ax.set_xticklabels([m.replace("_", " ").title() for m in metrics], rotation=45, ha="right")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Confidence levels
            ax = axes[1, 0]
            confidence_levels = [imp.confidence_level for imp in improvements.values()]
            bars = ax.bar(range(len(metrics)), confidence_levels, alpha=0.7, color="skyblue")
            ax.set_title("Confidence Levels")
            ax.set_xlabel("Metrics")
            ax.set_ylabel("Confidence Level")
            ax.set_xticks(range(len(metrics)))
            ax.set_xticklabels([m.replace("_", " ").title() for m in metrics], rotation=45, ha="right")
            ax.axhline(y=0.8, color="red", linestyle="--", alpha=0.7, label="Threshold (0.8)")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add confidence values on bars
            for bar, confidence in zip(bars, confidence_levels, strict=False):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{confidence:.2f}",
                    ha="center",
                    va="bottom",
                )

            # Improvement directions
            ax = axes[1, 1]
            directions = [imp.improvement_direction for imp in improvements.values()]
            direction_counts = pd.Series(directions).value_counts()
            colors = {"improving": "green", "declining": "red", "stable": "orange"}
            pie_colors = [colors.get(d, "gray") for d in direction_counts.index]

            ax.pie(
                direction_counts.values,
                labels=direction_counts.index,
                autopct="%1.1f%%",
                colors=pie_colors,
            )
            ax.set_title("Improvement Directions Distribution")

            plt.tight_layout()

            # Save dashboard
            dashboard_file = self.output_dir / "improvement_tracking_demo.png"
            plt.savefig(dashboard_file, dpi=300, bbox_inches="tight")
            plt.close()

            viz_files["dashboard"] = str(dashboard_file)

            return viz_files

        except Exception:
            return {}

    def export_demo_report(
        self,
        improvements: dict[str, QualityImprovement],
        visualizations: dict[str, str],
    ) -> str:
        """Export demo improvement tracking report"""

        try:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            report_file = self.output_dir / f"quality_improvement_demo_{timestamp}.json"

            # Create executive summary
            improving_count = sum(1 for i in improvements.values() if i.improvement_direction == "improving")
            declining_count = sum(1 for i in improvements.values() if i.improvement_direction == "declining")
            stable_count = sum(1 for i in improvements.values() if i.improvement_direction == "stable")

            avg_improvement = np.mean([i.improvement_percentage for i in improvements.values()])
            avg_confidence = np.mean([i.confidence_level for i in improvements.values()])

            # Prepare export data
            export_data = {
                "report_metadata": {
                    "generated_at": datetime.now(UTC).isoformat(),
                    "report_type": "demo",
                    "tracker_version": "1.0.0",
                    "total_metrics_tracked": len(improvements),
                },
                "executive_summary": {
                    "overall_trend": "improving"
                    if avg_improvement > 2
                    else "declining"
                    if avg_improvement < -2
                    else "stable",
                    "average_improvement_percentage": float(avg_improvement),
                    "average_confidence_level": float(avg_confidence),
                    "metrics_improving": improving_count,
                    "metrics_declining": declining_count,
                    "metrics_stable": stable_count,
                },
                "improvement_tracking": {
                    metric: {
                        "baseline_value": imp.baseline_value,
                        "current_value": imp.current_value,
                        "improvement_percentage": imp.improvement_percentage,
                        "improvement_direction": imp.improvement_direction,
                        "confidence_level": imp.confidence_level,
                        "recommendation": imp.recommendation,
                    }
                    for metric, imp in improvements.items()
                },
                "visualizations": visualizations,
            }

            # Save report
            with open(report_file, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

            return str(report_file)

        except Exception:
            return ""


def main():
    """Main demo execution"""

    # Initialize demo
    demo = QualityImprovementDemo()

    # Create demo improvements
    improvements = demo.create_demo_improvements()

    if not improvements:
        return

    # Create visualizations
    visualizations = demo.create_improvement_visualizations(improvements)

    # Export report
    demo.export_demo_report(improvements, visualizations)

    # Display summary

    # Show key findings
    for _metric, _improvement in improvements.items():
        pass

    # Show top recommendations
    for _metric, _improvement in list(improvements.items())[:3]:
        pass

    # Show summary statistics
    np.mean([i.improvement_percentage for i in improvements.values()])
    sum(1 for i in improvements.values() if i.improvement_direction == "improving")
    sum(1 for i in improvements.values() if i.improvement_direction == "declining")


if __name__ == "__main__":
    main()
