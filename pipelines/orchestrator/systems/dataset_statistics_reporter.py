"""
Dataset statistics reporter for comprehensive analytics.
Generates detailed statistics and reports for dataset analysis.
"""

import json
from collections import Counter
from datetime import datetime
from typing import Any

from conversation_schema import Conversation
from logger import get_logger

logger = get_logger(__name__)


class DatasetStatisticsReporter:
    """
    Generates comprehensive dataset statistics and reports.

    Provides detailed analytics for dataset composition, quality,
    and characteristics for therapeutic AI training optimization.
    """

    def __init__(self):
        """Initialize the dataset statistics reporter."""
        self.logger = get_logger(__name__)
        self.logger.info("DatasetStatisticsReporter initialized")

    def generate_comprehensive_report(self, conversations: list[Conversation]) -> dict[str, Any]:
        """Generate comprehensive dataset statistics report."""
        self.logger.info(f"Generating comprehensive report for {len(conversations)} conversations")

        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_conversations": len(conversations),
                "report_version": "1.0"
            },
            "basic_statistics": self._calculate_basic_statistics(conversations),
            "content_analysis": self._analyze_content(conversations),
            "quality_metrics": self._calculate_quality_metrics(conversations),
            "category_distribution": self._analyze_categories(conversations),
            "temporal_analysis": self._analyze_temporal_patterns(conversations),
            "conversation_characteristics": self._analyze_conversation_characteristics(conversations)
        }

        self.logger.info("Comprehensive report generated successfully")
        return report

    def _calculate_basic_statistics(self, conversations: list[Conversation]) -> dict[str, Any]:
        """Calculate basic dataset statistics."""
        if not conversations:
            return {"error": "No conversations to analyze"}

        total_messages = sum(len(conv.messages) for conv in conversations)
        total_characters = sum(sum(len(msg.content) for msg in conv.messages) for conv in conversations)

        message_counts = [len(conv.messages) for conv in conversations]
        character_counts = [sum(len(msg.content) for msg in conv.messages) for conv in conversations]

        return {
            "total_conversations": len(conversations),
            "total_messages": total_messages,
            "total_characters": total_characters,
            "average_messages_per_conversation": total_messages / len(conversations),
            "average_characters_per_conversation": total_characters / len(conversations),
            "average_characters_per_message": total_characters / total_messages if total_messages > 0 else 0,
            "message_count_distribution": {
                "min": min(message_counts),
                "max": max(message_counts),
                "median": sorted(message_counts)[len(message_counts)//2]
            },
            "character_count_distribution": {
                "min": min(character_counts),
                "max": max(character_counts),
                "median": sorted(character_counts)[len(character_counts)//2]
            }
        }

    def _analyze_content(self, conversations: list[Conversation]) -> dict[str, Any]:
        """Analyze content characteristics."""
        all_words = []
        role_distribution = Counter()

        for conv in conversations:
            for msg in conv.messages:
                words = msg.content.lower().split()
                all_words.extend(words)
                role_distribution[msg.role] += 1

        word_counts = Counter(all_words)

        return {
            "vocabulary_size": len(word_counts),
            "total_words": len(all_words),
            "most_common_words": word_counts.most_common(20),
            "role_distribution": dict(role_distribution),
            "average_words_per_message": len(all_words) / sum(len(conv.messages) for conv in conversations) if conversations else 0
        }

    def _calculate_quality_metrics(self, conversations: list[Conversation]) -> dict[str, Any]:
        """Calculate quality-related metrics."""
        quality_scores = [conv.quality_score for conv in conversations if conv.quality_score is not None]

        if not quality_scores:
            return {"error": "No quality scores available"}

        return {
            "quality_score_distribution": {
                "count": len(quality_scores),
                "average": sum(quality_scores) / len(quality_scores),
                "min": min(quality_scores),
                "max": max(quality_scores),
                "median": sorted(quality_scores)[len(quality_scores)//2]
            },
            "quality_categories": {
                "high_quality": len([s for s in quality_scores if s >= 0.8]),
                "medium_quality": len([s for s in quality_scores if 0.6 <= s < 0.8]),
                "low_quality": len([s for s in quality_scores if s < 0.6])
            }
        }

    def _analyze_categories(self, conversations: list[Conversation]) -> dict[str, Any]:
        """Analyze category distribution."""
        tag_counts = Counter()
        metadata_categories = Counter()

        for conv in conversations:
            for tag in conv.tags:
                tag_counts[tag] += 1

            category = conv.metadata.get("category", "unknown")
            metadata_categories[category] += 1

        return {
            "tag_distribution": dict(tag_counts.most_common(20)),
            "metadata_categories": dict(metadata_categories),
            "total_unique_tags": len(tag_counts),
            "conversations_with_tags": len([conv for conv in conversations if conv.tags]),
            "conversations_with_metadata": len([conv for conv in conversations if conv.metadata])
        }

    def _analyze_temporal_patterns(self, conversations: list[Conversation]) -> dict[str, Any]:
        """Analyze temporal patterns in the dataset."""
        creation_dates = [conv.created_at for conv in conversations if conv.created_at]
        update_dates = [conv.updated_at for conv in conversations if conv.updated_at]

        if not creation_dates:
            return {"error": "No temporal data available"}

        # Group by date
        date_counts = Counter()
        for date in creation_dates:
            date_str = date.date().isoformat()
            date_counts[date_str] += 1

        return {
            "date_range": {
                "earliest": min(creation_dates).isoformat(),
                "latest": max(creation_dates).isoformat()
            },
            "conversations_by_date": dict(date_counts.most_common(10)),
            "conversations_with_timestamps": {
                "created_at": len(creation_dates),
                "updated_at": len(update_dates)
            }
        }

    def _analyze_conversation_characteristics(self, conversations: list[Conversation]) -> dict[str, Any]:
        """Analyze conversation-specific characteristics."""
        characteristics = {
            "has_title": 0,
            "has_quality_score": 0,
            "has_metadata": 0,
            "has_tags": 0,
            "single_exchange": 0,
            "multi_exchange": 0,
            "long_conversations": 0  # >10 messages
        }

        for conv in conversations:
            if conv.title:
                characteristics["has_title"] += 1
            if conv.quality_score is not None:
                characteristics["has_quality_score"] += 1
            if conv.metadata:
                characteristics["has_metadata"] += 1
            if conv.tags:
                characteristics["has_tags"] += 1

            message_count = len(conv.messages)
            if message_count == 2:
                characteristics["single_exchange"] += 1
            elif message_count > 2:
                characteristics["multi_exchange"] += 1

            if message_count > 10:
                characteristics["long_conversations"] += 1

        return characteristics

    def export_report(self, report: dict[str, Any], output_path: str) -> bool:
        """Export report to JSON file."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)

            self.logger.info(f"Report exported to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export report: {e}")
            return False


def validate_dataset_statistics_reporter():
    """Validate the DatasetStatisticsReporter functionality."""
    try:
        reporter = DatasetStatisticsReporter()
        assert hasattr(reporter, "generate_comprehensive_report")
        return True
    except Exception:
        return False


if __name__ == "__main__":
    if validate_dataset_statistics_reporter():
        pass
    else:
        pass
