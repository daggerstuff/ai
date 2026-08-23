#!/usr/bin/env python3
"""
Comprehensive Dataset Statistics Dashboard
Provides detailed analytics and insights about dataset composition and characteristics
"""

import json
import sqlite3
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.simplefilter("default")


@dataclass
class DatasetStatistics:
    """Dataset statistics data structure"""

    dataset_name: str
    total_conversations: int
    total_words: int
    total_characters: int
    avg_conversation_length: float
    avg_word_count: float
    language_distribution: dict[str, int]
    tier_distribution: dict[str, int]
    processing_status: dict[str, int]
    quality_metrics: dict[str, float]
    temporal_distribution: dict[str, int]
    unique_characteristics: list[str]


@dataclass
class DatasetInsights:
    """Dataset insights and recommendations"""

    dataset_name: str
    key_insights: list[str]
    strengths: list[str]
    weaknesses: list[str]
    recommendations: list[str]
    comparative_analysis: dict[str, Any]
    optimization_opportunities: list[str]


class DatasetStatisticsDashboard:
    """Enterprise-grade dataset statistics dashboard"""

    def __init__(self, db_path: str = "database/conversations.db"):
        self.db_path = Path(db_path)
        self.output_dir = Path("monitoring/dataset_analytics")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Dashboard configuration
        self.dashboard_config = {
            "refresh_interval": 300,  # 5 minutes
            "max_datasets_display": 20,
            "chart_style": "seaborn-v0_8",
            "color_palette": "husl",
        }

    def generate_comprehensive_statistics(self) -> dict[str, DatasetStatistics]:
        """Generate comprehensive statistics for all datasets"""

        try:
            # Get dataset information
            datasets = self._get_dataset_list()

            if not datasets:
                return {}

            # Generate statistics for each dataset
            dataset_stats = {}

            for dataset_name in datasets:
                stats = self._analyze_dataset(dataset_name)
                if stats:
                    dataset_stats[dataset_name] = stats

            return dataset_stats

        except Exception:
            return {}

    def _get_dataset_list(self) -> list[str]:
        """Get list of unique datasets"""
        try:
            conn = sqlite3.connect(self.db_path)

            query = """
            SELECT DISTINCT dataset_source
            FROM conversations
            WHERE dataset_source IS NOT NULL
            AND dataset_source != ''
            ORDER BY dataset_source
            """

            cursor = conn.execute(query)
            datasets = [row[0] for row in cursor.fetchall()]

            conn.close()
            return datasets

        except Exception:
            return []

    def _analyze_dataset(self, dataset_name: str) -> DatasetStatistics | None:
        """Analyze individual dataset statistics"""
        try:
            conn = sqlite3.connect(self.db_path)

            # Get dataset conversations
            query = """
            SELECT
                conversation_id,
                tier,
                turn_count,
                word_count,
                character_count,
                language,
                processing_status,
                created_at,
                conversations_json
            FROM conversations
            WHERE dataset_source = ?
            """

            cursor = conn.execute(query, (dataset_name,))
            columns = [desc[0] for desc in cursor.description]

            data = []
            for row in cursor.fetchall():
                record = dict(zip(columns, row, strict=False))
                data.append(record)

            conn.close()

            if not data:
                return None

            # Calculate statistics
            df = pd.DataFrame(data)

            # Basic counts
            total_conversations = len(df)
            total_words = df["word_count"].fillna(0).sum()
            total_characters = df["character_count"].fillna(0).sum()

            # Averages
            avg_conversation_length = df["turn_count"].fillna(0).mean()
            avg_word_count = df["word_count"].fillna(0).mean()

            # Distributions
            language_distribution = df["language"].fillna("unknown").value_counts().to_dict()
            tier_distribution = df["tier"].fillna("unknown").value_counts().to_dict()
            processing_status = df["processing_status"].fillna("unknown").value_counts().to_dict()

            # Quality metrics (synthetic for demo)
            quality_metrics = self._calculate_quality_metrics(df)

            # Temporal distribution
            temporal_distribution = self._calculate_temporal_distribution(df)

            # Unique characteristics
            unique_characteristics = self._identify_unique_characteristics(df, dataset_name)

            return DatasetStatistics(
                dataset_name=dataset_name,
                total_conversations=total_conversations,
                total_words=int(total_words),
                total_characters=int(total_characters),
                avg_conversation_length=float(avg_conversation_length),
                avg_word_count=float(avg_word_count),
                language_distribution=language_distribution,
                tier_distribution=tier_distribution,
                processing_status=processing_status,
                quality_metrics=quality_metrics,
                temporal_distribution=temporal_distribution,
                unique_characteristics=unique_characteristics,
            )

        except Exception:
            return None

    def _calculate_quality_metrics(self, df: pd.DataFrame) -> dict[str, float]:
        """Calculate quality metrics for dataset"""
        try:
            # Synthetic quality metrics for demonstration
            return {
                "completeness_score": min(1.0, len(df[df["word_count"] > 0]) / len(df)),
                "consistency_score": min(1.0, len(df[df["processing_status"] == "processed"]) / len(df)),
                "diversity_score": min(1.0, len(df["tier"].unique()) / 5.0),  # Assuming max 5 tiers
                "richness_score": min(1.0, df["word_count"].fillna(0).mean() / 500.0),  # Normalized to 500 words
                "coverage_score": np.random.uniform(0.7, 0.95),  # Synthetic for demo
            }

        except Exception:
            return {}

    def _calculate_temporal_distribution(self, df: pd.DataFrame) -> dict[str, int]:
        """Calculate temporal distribution of conversations"""
        try:
            # Convert created_at to datetime and extract date
            df["created_date"] = pd.to_datetime(df["created_at"]).dt.date

            # Group by date and count
            temporal_dist = df["created_date"].value_counts().head(30).to_dict()

            # Convert dates to strings for JSON serialization
            return {str(date): count for date, count in temporal_dist.items()}

        except Exception:
            return {}

    def _identify_unique_characteristics(self, df: pd.DataFrame, dataset_name: str) -> list[str]:
        """Identify unique characteristics of the dataset"""
        characteristics = []

        try:
            # Analyze conversation length patterns
            avg_length = df["turn_count"].fillna(0).mean()
            if avg_length > 10:
                characteristics.append("Long-form conversations (>10 turns average)")
            elif avg_length < 3:
                characteristics.append("Short-form conversations (<3 turns average)")

            # Analyze word count patterns
            avg_words = df["word_count"].fillna(0).mean()
            if avg_words > 1000:
                characteristics.append("High word density (>1000 words average)")
            elif avg_words < 100:
                characteristics.append("Concise conversations (<100 words average)")

            # Analyze tier distribution
            tier_counts = df["tier"].value_counts()
            if len(tier_counts) > 0:
                dominant_tier = tier_counts.index[0]
                if tier_counts.iloc[0] / len(df) > 0.8:
                    characteristics.append(f"Predominantly {dominant_tier} tier content")

            # Analyze language diversity
            language_counts = df["language"].value_counts()
            if len(language_counts) > 1:
                characteristics.append(f"Multi-language support ({len(language_counts)} languages)")

            # Dataset-specific characteristics
            if "reddit" in dataset_name.lower():
                characteristics.append("Community-generated content")
            elif "clinical" in dataset_name.lower() or "therapy" in dataset_name.lower():
                characteristics.append("Clinical/therapeutic focus")
            elif "psychology" in dataset_name.lower():
                characteristics.append("Psychology domain expertise")

            return characteristics

        except Exception:
            return []

    def generate_dataset_insights(self, dataset_stats: dict[str, DatasetStatistics]) -> dict[str, DatasetInsights]:
        """Generate insights and recommendations for datasets"""

        try:
            insights = {}

            # Calculate overall statistics for comparison
            overall_stats = self._calculate_overall_statistics(dataset_stats)

            for dataset_name, stats in dataset_stats.items():
                insight = self._analyze_dataset_insights(stats, overall_stats)
                if insight:
                    insights[dataset_name] = insight

            return insights

        except Exception:
            return {}

    def _calculate_overall_statistics(self, dataset_stats: dict[str, DatasetStatistics]) -> dict[str, float]:
        """Calculate overall statistics across all datasets"""
        try:
            all_conversations = sum(stats.total_conversations for stats in dataset_stats.values())
            all_words = sum(stats.total_words for stats in dataset_stats.values())

            avg_conversation_length = np.mean([stats.avg_conversation_length for stats in dataset_stats.values()])
            avg_word_count = np.mean([stats.avg_word_count for stats in dataset_stats.values()])

            return {
                "total_conversations": all_conversations,
                "total_words": all_words,
                "avg_conversation_length": avg_conversation_length,
                "avg_word_count": avg_word_count,
                "dataset_count": len(dataset_stats),
            }

        except Exception:
            return {}

    def _analyze_dataset_insights(
        self, stats: DatasetStatistics, overall_stats: dict[str, float]
    ) -> DatasetInsights | None:
        """Analyze insights for individual dataset"""
        try:
            key_insights = []
            strengths = []
            weaknesses = []
            recommendations = []
            optimization_opportunities = []

            # Conversation volume analysis
            if (
                stats.total_conversations
                > overall_stats.get("total_conversations", 0) / len(overall_stats.get("dataset_count", 1)) * 2
            ):
                strengths.append("High conversation volume")
                key_insights.append(f"Contains {stats.total_conversations:,} conversations - above average volume")
            elif stats.total_conversations < 1000:
                weaknesses.append("Low conversation volume")
                recommendations.append("Consider expanding dataset with additional conversations")

            # Quality analysis
            avg_quality = np.mean(list(stats.quality_metrics.values()))
            if avg_quality > 0.8:
                strengths.append("High overall quality metrics")
            elif avg_quality < 0.6:
                weaknesses.append("Quality metrics below threshold")
                recommendations.append("Implement quality improvement measures")

            # Conversation length analysis
            if stats.avg_conversation_length > overall_stats.get("avg_conversation_length", 0) * 1.5:
                strengths.append("Rich, detailed conversations")
                key_insights.append(f"Average {stats.avg_conversation_length:.1f} turns per conversation")
            elif stats.avg_conversation_length < 2:
                weaknesses.append("Very short conversations")
                optimization_opportunities.append("Enhance conversation depth and engagement")

            # Processing status analysis
            processed_rate = stats.processing_status.get("processed", 0) / stats.total_conversations
            if processed_rate > 0.95:
                strengths.append("Excellent processing completion rate")
            elif processed_rate < 0.8:
                weaknesses.append("Processing completion issues")
                recommendations.append("Review and fix processing pipeline issues")

            # Language diversity analysis
            if len(stats.language_distribution) > 1:
                strengths.append("Multi-language support")
                key_insights.append(f"Supports {len(stats.language_distribution)} languages")

            # Tier distribution analysis
            tier_diversity = len(stats.tier_distribution)
            if tier_diversity > 3:
                strengths.append("Good tier diversity")
            elif tier_diversity == 1:
                optimization_opportunities.append("Consider diversifying content tiers")

            # Comparative analysis
            comparative_analysis = {
                "conversation_volume_percentile": self._calculate_percentile(
                    stats.total_conversations,
                    [s.total_conversations for s in overall_stats.get("all_stats", [])],
                ),
                "quality_score_relative": avg_quality,
                "processing_efficiency": processed_rate,
            }

            return DatasetInsights(
                dataset_name=stats.dataset_name,
                key_insights=key_insights,
                strengths=strengths,
                weaknesses=weaknesses,
                recommendations=recommendations,
                comparative_analysis=comparative_analysis,
                optimization_opportunities=optimization_opportunities,
            )

        except Exception:
            return None

    def _calculate_percentile(self, value: float, all_values: list[float]) -> float:
        """Calculate percentile rank of value in list"""
        try:
            if not all_values:
                return 50.0

            sorted_values = sorted(all_values)
            rank = sum(1 for v in sorted_values if v <= value)
            return (rank / len(sorted_values)) * 100

        except Exception:
            return 50.0

    def create_dashboard_visualizations(self, dataset_stats: dict[str, DatasetStatistics]) -> dict[str, str]:
        """Create comprehensive dashboard visualizations"""

        viz_files = {}

        try:
            # Set style
            plt.style.use("default")
            sns.set_palette("husl")

            # Create main dashboard
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle("Dataset Statistics Dashboard", fontsize=16, fontweight="bold")

            # Dataset volume comparison
            ax = axes[0, 0]
            dataset_names = list(dataset_stats.keys())[:10]  # Top 10
            conversation_counts = [dataset_stats[name].total_conversations for name in dataset_names]

            bars = ax.bar(range(len(dataset_names)), conversation_counts, alpha=0.7)
            ax.set_title("Conversation Volume by Dataset")
            ax.set_xlabel("Datasets")
            ax.set_ylabel("Conversation Count")
            ax.set_xticks(range(len(dataset_names)))
            ax.set_xticklabels(
                [name[:15] + "..." if len(name) > 15 else name for name in dataset_names],
                rotation=45,
                ha="right",
            )
            ax.grid(True, alpha=0.3)

            # Add value labels
            for bar, count in zip(bars, conversation_counts, strict=False):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + max(conversation_counts) * 0.01,
                    f"{count:,}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            # Average conversation length comparison
            ax = axes[0, 1]
            avg_lengths = [dataset_stats[name].avg_conversation_length for name in dataset_names]

            ax.bar(range(len(dataset_names)), avg_lengths, alpha=0.7, color="orange")
            ax.set_title("Average Conversation Length")
            ax.set_xlabel("Datasets")
            ax.set_ylabel("Average Turns")
            ax.set_xticks(range(len(dataset_names)))
            ax.set_xticklabels(
                [name[:15] + "..." if len(name) > 15 else name for name in dataset_names],
                rotation=45,
                ha="right",
            )
            ax.grid(True, alpha=0.3)

            # Quality metrics heatmap
            ax = axes[0, 2]
            quality_data = []
            quality_labels = []

            for name in dataset_names[:8]:  # Top 8 for readability
                stats = dataset_stats[name]
                quality_values = list(stats.quality_metrics.values())
                quality_data.append(quality_values)
                quality_labels.append(name[:12] + "..." if len(name) > 12 else name)

            if quality_data:
                quality_matrix = np.array(quality_data)
                im = ax.imshow(quality_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
                ax.set_title("Quality Metrics Heatmap")
                ax.set_yticks(range(len(quality_labels)))
                ax.set_yticklabels(quality_labels)
                ax.set_xticks(range(len(next(iter(dataset_stats.values())).quality_metrics)))
                ax.set_xticklabels(
                    [k.replace("_", " ").title() for k in next(iter(dataset_stats.values())).quality_metrics],
                    rotation=45,
                    ha="right",
                )

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label("Quality Score")

            # Processing status distribution
            ax = axes[1, 0]
            all_statuses = {}
            for stats in dataset_stats.values():
                for status, count in stats.processing_status.items():
                    all_statuses[status] = all_statuses.get(status, 0) + count

            if all_statuses:
                ax.pie(all_statuses.values(), labels=all_statuses.keys(), autopct="%1.1f%%")
                ax.set_title("Overall Processing Status Distribution")

            # Language distribution
            ax = axes[1, 1]
            all_languages = {}
            for stats in dataset_stats.values():
                for lang, count in stats.language_distribution.items():
                    all_languages[lang] = all_languages.get(lang, 0) + count

            if all_languages:
                # Show top 10 languages
                top_languages = dict(sorted(all_languages.items(), key=lambda x: x[1], reverse=True)[:10])
                ax.bar(
                    range(len(top_languages)),
                    list(top_languages.values()),
                    alpha=0.7,
                    color="green",
                )
                ax.set_title("Top 10 Languages Distribution")
                ax.set_xlabel("Languages")
                ax.set_ylabel("Conversation Count")
                ax.set_xticks(range(len(top_languages)))
                ax.set_xticklabels(list(top_languages.keys()), rotation=45, ha="right")
                ax.grid(True, alpha=0.3)

            # Word count distribution
            ax = axes[1, 2]
            word_counts = [stats.total_words for stats in dataset_stats.values()]

            ax.hist(word_counts, bins=20, alpha=0.7, edgecolor="black")
            ax.set_title("Word Count Distribution Across Datasets")
            ax.set_xlabel("Total Words")
            ax.set_ylabel("Number of Datasets")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save dashboard
            dashboard_file = self.output_dir / "dataset_statistics_dashboard.png"
            plt.savefig(dashboard_file, dpi=300, bbox_inches="tight")
            plt.close()

            viz_files["main_dashboard"] = str(dashboard_file)

            return viz_files

        except Exception:
            return {}

    def export_dashboard_report(
        self,
        dataset_stats: dict[str, DatasetStatistics],
        insights: dict[str, DatasetInsights],
        visualizations: dict[str, str],
    ) -> str:
        """Export comprehensive dashboard report"""

        try:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            report_file = self.output_dir / f"dataset_statistics_report_{timestamp}.json"

            # Create executive summary
            executive_summary = self._create_executive_summary(dataset_stats, insights)

            # Prepare export data
            export_data = {
                "report_metadata": {
                    "generated_at": datetime.now(UTC).isoformat(),
                    "dashboard_version": "1.0.0",
                    "total_datasets": len(dataset_stats),
                    "analysis_scope": "comprehensive_statistics",
                },
                "executive_summary": executive_summary,
                "dataset_statistics": {
                    name: {
                        "total_conversations": stats.total_conversations,
                        "total_words": stats.total_words,
                        "total_characters": stats.total_characters,
                        "avg_conversation_length": stats.avg_conversation_length,
                        "avg_word_count": stats.avg_word_count,
                        "language_distribution": stats.language_distribution,
                        "tier_distribution": stats.tier_distribution,
                        "processing_status": stats.processing_status,
                        "quality_metrics": stats.quality_metrics,
                        "temporal_distribution": stats.temporal_distribution,
                        "unique_characteristics": stats.unique_characteristics,
                    }
                    for name, stats in dataset_stats.items()
                },
                "dataset_insights": {
                    name: {
                        "key_insights": insight.key_insights,
                        "strengths": insight.strengths,
                        "weaknesses": insight.weaknesses,
                        "recommendations": insight.recommendations,
                        "comparative_analysis": insight.comparative_analysis,
                        "optimization_opportunities": insight.optimization_opportunities,
                    }
                    for name, insight in insights.items()
                },
                "visualizations": visualizations,
            }

            # Save report
            with open(report_file, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

            return str(report_file)

        except Exception:
            return ""

    def _create_executive_summary(
        self,
        dataset_stats: dict[str, DatasetStatistics],
        insights: dict[str, DatasetInsights],
    ) -> dict[str, Any]:
        """Create executive summary of dataset statistics"""
        try:
            # Overall statistics
            total_conversations = sum(stats.total_conversations for stats in dataset_stats.values())
            total_words = sum(stats.total_words for stats in dataset_stats.values())
            total_datasets = len(dataset_stats)

            # Quality analysis
            all_quality_scores = []
            for stats in dataset_stats.values():
                all_quality_scores.extend(stats.quality_metrics.values())

            avg_quality = np.mean(all_quality_scores) if all_quality_scores else 0

            # Processing analysis
            total_processed = 0
            total_conversations_all = 0
            for stats in dataset_stats.values():
                total_processed += stats.processing_status.get("processed", 0)
                total_conversations_all += stats.total_conversations

            processing_rate = (total_processed / total_conversations_all) * 100 if total_conversations_all > 0 else 0

            # Top performing datasets
            top_datasets = sorted(
                dataset_stats.items(),
                key=lambda x: x[1].total_conversations,
                reverse=True,
            )[:5]

            # Common strengths and weaknesses
            all_strengths = []
            all_weaknesses = []
            for insight in insights.values():
                all_strengths.extend(insight.strengths)
                all_weaknesses.extend(insight.weaknesses)

            common_strengths = [item for item, count in Counter(all_strengths).most_common(3)]
            common_weaknesses = [item for item, count in Counter(all_weaknesses).most_common(3)]

            return {
                "total_datasets": total_datasets,
                "total_conversations": total_conversations,
                "total_words": total_words,
                "average_quality_score": float(avg_quality),
                "processing_completion_rate": float(processing_rate),
                "top_datasets_by_volume": [name for name, _ in top_datasets],
                "common_strengths": common_strengths,
                "common_weaknesses": common_weaknesses,
                "overall_health": "excellent"
                if avg_quality > 0.8 and processing_rate > 95
                else "good"
                if avg_quality > 0.7 and processing_rate > 90
                else "fair"
                if avg_quality > 0.6 and processing_rate > 80
                else "poor",
            }

        except Exception:
            return {}


def main():
    """Main execution function"""

    # Initialize dashboard
    dashboard = DatasetStatisticsDashboard()

    # Generate comprehensive statistics
    dataset_stats = dashboard.generate_comprehensive_statistics()

    if not dataset_stats:
        return

    # Generate insights
    insights = dashboard.generate_dataset_insights(dataset_stats)

    # Create visualizations
    visualizations = dashboard.create_dashboard_visualizations(dataset_stats)

    # Export report
    dashboard.export_dashboard_report(dataset_stats, insights, visualizations)

    # Display summary
    sum(stats.total_conversations for stats in dataset_stats.values())
    sum(stats.total_words for stats in dataset_stats.values())

    # Show top datasets
    top_datasets = sorted(dataset_stats.items(), key=lambda x: x[1].total_conversations, reverse=True)[:5]

    for _i, (_name, _stats) in enumerate(top_datasets, 1):
        pass

    # Show key insights
    if insights:
        sample_insights = next(iter(insights.values()))
        for _insight in sample_insights.key_insights[:3]:
            pass


if __name__ == "__main__":
    main()
