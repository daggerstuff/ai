#!/usr/bin/env python3
"""
Advanced Analytics Orchestrator

This module integrates real-time streaming processing with advanced pattern recognition
and complexity scoring to provide comprehensive analytics for therapeutic conversations.
"""

import asyncio
import json
import logging
import statistics

# Import streaming components
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from complexity_scorer import ComplexityMetrics, ComplexityScorer

# Import our analytics components
from pattern_recognition_engine import ConversationAnalysis, PatternRecognitionEngine

sys.path.append(str(Path(__file__).parent.parent / "streaming"))
from real_time_processor import FileWatcherDataSource, StreamingEvent, StreamingProcessor


@dataclass
class AnalyticsResult:
    """Combined analytics result for a conversation."""
    conversation_id: str
    timestamp: datetime

    # Pattern recognition results
    patterns_detected: int
    therapeutic_quality: float
    flow_coherence: float
    emotional_depth: float
    clinical_accuracy: float

    # Complexity scoring results
    overall_complexity: float
    linguistic_complexity: float
    therapeutic_depth_complexity: float
    emotional_complexity: float
    clinical_sophistication: float

    # Combined insights
    quality_score: float
    sophistication_level: str  # 'basic', 'intermediate', 'advanced', 'expert'
    recommendations: list[str]

    # Raw analysis objects
    pattern_analysis: ConversationAnalysis = None
    complexity_metrics: ComplexityMetrics = None


@dataclass
class AnalyticsMetrics:
    """System-wide analytics metrics."""
    total_conversations_analyzed: int = 0
    average_quality_score: float = 0.0
    average_complexity_score: float = 0.0
    sophistication_distribution: dict[str, int] = None
    processing_rate: float = 0.0
    error_rate: float = 0.0

    def __post_init__(self):
        if self.sophistication_distribution is None:
            self.sophistication_distribution = {
                "basic": 0, "intermediate": 0, "advanced": 0, "expert": 0
            }


class AdvancedAnalyticsEngine:
    """Advanced analytics engine combining all analysis capabilities."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

        # Initialize analysis engines
        self.pattern_engine = PatternRecognitionEngine()
        self.complexity_scorer = ComplexityScorer()

        # Analytics storage
        self.results_history = deque(maxlen=10000)
        self.metrics = AnalyticsMetrics()

        # Performance tracking
        self.processing_times = deque(maxlen=1000)
        self.error_count = 0

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=self.config.get("worker_threads", 4))

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def analyze_conversation(self, conversation: dict[str, Any]) -> AnalyticsResult:
        """Perform comprehensive analysis of a conversation."""
        start_time = time.time()
        conversation_id = conversation.get("id", f"conv_{int(time.time())}")

        try:
            # Run pattern recognition
            pattern_analysis = self.pattern_engine.analyze_conversation(conversation)

            # Run complexity scoring
            complexity_metrics = self.complexity_scorer.score_conversation_complexity(conversation)

            # Combine results
            result = self._combine_analysis_results(
                conversation_id, pattern_analysis, complexity_metrics
            )

            # Update metrics
            self._update_metrics(result)

            # Store result
            self.results_history.append(result)

            # Track processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)

            self.logger.info(f"Analyzed conversation {conversation_id} in {processing_time:.3f}s")

            return result

        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Error analyzing conversation {conversation_id}: {e}")

            # Return empty result
            return AnalyticsResult(
                conversation_id=conversation_id,
                timestamp=datetime.now(tz=datetime.timezone.utc),
                patterns_detected=0,
                therapeutic_quality=0.0,
                flow_coherence=0.0,
                emotional_depth=0.0,
                clinical_accuracy=0.0,
                overall_complexity=0.0,
                linguistic_complexity=0.0,
                therapeutic_depth_complexity=0.0,
                emotional_complexity=0.0,
                clinical_sophistication=0.0,
                quality_score=0.0,
                sophistication_level="basic",
                recommendations=["Analysis failed - check conversation format"]
            )

    def _combine_analysis_results(self, conversation_id: str,
                                pattern_analysis: ConversationAnalysis,
                                complexity_metrics: ComplexityMetrics) -> AnalyticsResult:
        """Combine pattern recognition and complexity analysis results."""

        # Calculate combined quality score
        quality_score = self._calculate_combined_quality(pattern_analysis, complexity_metrics)

        # Determine sophistication level
        sophistication_level = self._determine_sophistication_level(complexity_metrics)

        # Combine recommendations
        recommendations = self._combine_recommendations(pattern_analysis, complexity_metrics)

        return AnalyticsResult(
            conversation_id=conversation_id,
            timestamp=datetime.now(tz=datetime.timezone.utc),

            # Pattern recognition metrics
            patterns_detected=len(pattern_analysis.patterns),
            therapeutic_quality=pattern_analysis.therapeutic_quality,
            flow_coherence=pattern_analysis.flow_coherence,
            emotional_depth=pattern_analysis.emotional_depth,
            clinical_accuracy=pattern_analysis.clinical_accuracy,

            # Complexity metrics
            overall_complexity=complexity_metrics.overall_complexity,
            linguistic_complexity=complexity_metrics.linguistic_complexity,
            therapeutic_depth_complexity=complexity_metrics.therapeutic_depth,
            emotional_complexity=complexity_metrics.emotional_complexity,
            clinical_sophistication=complexity_metrics.clinical_sophistication,

            # Combined metrics
            quality_score=quality_score,
            sophistication_level=sophistication_level,
            recommendations=recommendations,

            # Raw analysis objects
            pattern_analysis=pattern_analysis,
            complexity_metrics=complexity_metrics
        )

    def _calculate_combined_quality(self, pattern_analysis: ConversationAnalysis,
                                  complexity_metrics: ComplexityMetrics) -> float:
        """Calculate combined quality score from both analyses."""

        # Weight pattern analysis components
        pattern_quality = (
            pattern_analysis.therapeutic_quality * 0.4 +
            pattern_analysis.flow_coherence * 0.2 +
            pattern_analysis.emotional_depth * 0.2 +
            pattern_analysis.clinical_accuracy * 0.2
        )

        # Weight complexity components (higher complexity can indicate higher quality)
        complexity_quality = (
            complexity_metrics.therapeutic_depth * 0.4 +
            complexity_metrics.clinical_sophistication * 0.3 +
            complexity_metrics.emotional_complexity * 0.2 +
            min(complexity_metrics.linguistic_complexity, 0.8) * 0.1  # Cap linguistic complexity
        )

        # Combine with weights favoring pattern analysis
        combined_quality = pattern_quality * 0.7 + complexity_quality * 0.3

        return min(combined_quality, 1.0)

    def _determine_sophistication_level(self, complexity_metrics: ComplexityMetrics) -> str:
        """Determine sophistication level based on complexity metrics."""

        # Calculate weighted sophistication score
        sophistication_score = (
            complexity_metrics.therapeutic_depth * 0.4 +
            complexity_metrics.clinical_sophistication * 0.3 +
            complexity_metrics.emotional_complexity * 0.2 +
            complexity_metrics.overall_complexity * 0.1
        )

        # Map to sophistication levels
        if sophistication_score >= 0.8:
            return "expert"
        if sophistication_score >= 0.6:
            return "advanced"
        if sophistication_score >= 0.4:
            return "intermediate"
        return "basic"

    def _combine_recommendations(self, pattern_analysis: ConversationAnalysis,
                               complexity_metrics: ComplexityMetrics) -> list[str]:
        """Combine recommendations from both analyses."""
        recommendations = []

        # Add pattern analysis recommendations
        recommendations.extend(pattern_analysis.recommendations)

        # Add complexity-based recommendations
        if complexity_metrics.therapeutic_depth < 0.5:
            recommendations.append("Consider incorporating more advanced therapeutic concepts")

        if complexity_metrics.emotional_complexity < 0.4:
            recommendations.append("Explore emotional depth and range more thoroughly")

        if complexity_metrics.clinical_sophistication < 0.3:
            recommendations.append("Consider using more clinical terminology where appropriate")

        if complexity_metrics.linguistic_complexity > 0.8:
            recommendations.append("Consider simplifying language for better accessibility")

        # Remove duplicates and limit to top 5
        unique_recommendations = list(dict.fromkeys(recommendations))
        return unique_recommendations[:5]

    def _update_metrics(self, result: AnalyticsResult):
        """Update system-wide analytics metrics."""
        self.metrics.total_conversations_analyzed += 1

        # Update running averages
        total = self.metrics.total_conversations_analyzed
        self.metrics.average_quality_score = (
            (self.metrics.average_quality_score * (total - 1) + result.quality_score) / total
        )
        self.metrics.average_complexity_score = (
            (self.metrics.average_complexity_score * (total - 1) + result.overall_complexity) / total
        )

        # Update sophistication distribution
        self.metrics.sophistication_distribution[result.sophistication_level] += 1

        # Update processing rate
        if self.processing_times:
            avg_processing_time = statistics.mean(self.processing_times)
            self.metrics.processing_rate = 1.0 / avg_processing_time if avg_processing_time > 0 else 0

        # Update error rate
        total_processed = self.metrics.total_conversations_analyzed + self.error_count
        self.metrics.error_rate = self.error_count / max(total_processed, 1)

    def get_analytics_summary(self) -> dict[str, Any]:
        """Get comprehensive analytics summary."""
        recent_results = list(self.results_history)[-100:]  # Last 100 results

        return {
            "system_metrics": asdict(self.metrics),
            "recent_performance": {
                "conversations_analyzed": len(recent_results),
                "average_quality": statistics.mean([r.quality_score for r in recent_results]) if recent_results else 0,
                "average_complexity": statistics.mean([r.overall_complexity for r in recent_results]) if recent_results else 0,
                "sophistication_trends": self._calculate_sophistication_trends(recent_results)
            },
            "top_patterns": self._get_top_patterns(recent_results),
            "quality_distribution": self._get_quality_distribution(recent_results),
            "processing_performance": {
                "average_processing_time": statistics.mean(self.processing_times) if self.processing_times else 0,
                "processing_rate": self.metrics.processing_rate,
                "error_rate": self.metrics.error_rate
            }
        }


    def _calculate_sophistication_trends(self, results: list[AnalyticsResult]) -> dict[str, float]:
        """Calculate sophistication trends over recent results."""
        if not results:
            return {}

        # Group by time periods (e.g., last hour, last day)
        now = datetime.now(tz=datetime.timezone.utc)
        last_hour = [r for r in results if (now - r.timestamp).total_seconds() < 3600]
        last_day = [r for r in results if (now - r.timestamp).total_seconds() < 86400]

        trends = {}

        if last_hour:
            trends["last_hour_avg_quality"] = statistics.mean([r.quality_score for r in last_hour])
            trends["last_hour_avg_complexity"] = statistics.mean([r.overall_complexity for r in last_hour])

        if last_day:
            trends["last_day_avg_quality"] = statistics.mean([r.quality_score for r in last_day])
            trends["last_day_avg_complexity"] = statistics.mean([r.overall_complexity for r in last_day])

        return trends

    def _get_top_patterns(self, results: list[AnalyticsResult]) -> list[dict[str, Any]]:
        """Get most common patterns from recent results."""
        pattern_counts = defaultdict(int)

        for result in results:
            if result.pattern_analysis:
                for pattern in result.pattern_analysis.patterns:
                    pattern_key = f"{pattern.pattern_type}:{pattern.description}"
                    pattern_counts[pattern_key] += 1

        # Sort by frequency and return top 10
        top_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return [
            {"pattern": pattern, "count": count}
            for pattern, count in top_patterns
        ]

    def _get_quality_distribution(self, results: list[AnalyticsResult]) -> dict[str, int]:
        """Get quality score distribution."""
        distribution = {"high": 0, "medium": 0, "low": 0}

        for result in results:
            if result.quality_score >= 0.7:
                distribution["high"] += 1
            elif result.quality_score >= 0.4:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1

        return distribution


class StreamingAnalyticsOrchestrator:
    """Orchestrates streaming analytics with real-time processing."""

    def __init__(self, config_path: str = "analytics_config.json"):
        self.config = self._load_config(config_path)

        # Initialize components
        self.analytics_engine = AdvancedAnalyticsEngine(self.config.get("analytics", {}))
        self.streaming_processor = StreamingProcessor(self.config.get("streaming", {}))

        # Results storage
        self.real_time_results = deque(maxlen=1000)

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Add analytics event handler to streaming processor
        self.streaming_processor.add_event_handler("conversation", self._handle_conversation_event)

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """Load configuration from file."""
        default_config = {
            "streaming": {
                "queue_size": 5000,
                "worker_threads": 2,
                "batch_size": 50
            },
            "analytics": {
                "worker_threads": 4,
                "enable_real_time": True,
                "store_results": True
            },
            "data_sources": [
                {
                    "type": "file_watcher",
                    "config": {
                        "directory": "data/streaming",
                        "patterns": ["*.jsonl", "*.json"]
                    }
                }
            ]
        }

        try:
            if Path(config_path).exists():
                with open(config_path) as f:
                    config = json.load(f)
                    default_config.update(config)
        except Exception as e:
            self.logger.warning(f"Could not load config from {config_path}: {e}")

        return default_config

    def _handle_conversation_event(self, event: StreamingEvent):
        """Handle conversation events from streaming processor."""
        try:
            # Extract conversation data
            conversation_data = event.data

            # Perform analytics
            result = self.analytics_engine.analyze_conversation(conversation_data)

            # Store result
            self.real_time_results.append(result)

            # Log significant findings
            if result.quality_score > 0.8:
                self.logger.info(f"High-quality conversation detected: {result.conversation_id} (quality: {result.quality_score:.3f})")

            if result.sophistication_level == "expert":
                self.logger.info(f"Expert-level conversation detected: {result.conversation_id}")

            # Check for alerts
            self._check_analytics_alerts(result)

        except Exception as e:
            self.logger.error(f"Error handling conversation event: {e}")

    def _check_analytics_alerts(self, result: AnalyticsResult):
        """Check for analytics-based alerts."""
        alerts = []

        # Quality alerts
        if result.quality_score < 0.3:
            alerts.append(f"Very low quality conversation: {result.conversation_id}")

        # Sophistication alerts
        if result.sophistication_level == "expert" and result.quality_score > 0.8:
            alerts.append(f"Exceptional conversation detected: {result.conversation_id}")

        # Pattern alerts
        if result.patterns_detected == 0:
            alerts.append(f"No therapeutic patterns detected: {result.conversation_id}")

        # Log alerts
        for alert in alerts:
            self.logger.warning(f"ANALYTICS ALERT: {alert}")

    async def start_streaming_analytics(self):
        """Start streaming analytics processing."""
        self.logger.info("Starting streaming analytics orchestrator...")

        # Add data sources to streaming processor
        for source_config in self.config.get("data_sources", []):
            if source_config["type"] == "file_watcher":
                source = FileWatcherDataSource(
                    source_id=f"file_watcher_{int(time.time())}",
                    config=source_config["config"]
                )
                self.streaming_processor.add_data_source(source)

        # Start streaming processor
        await self.streaming_processor.start()

    def get_real_time_analytics(self) -> dict[str, Any]:
        """Get real-time analytics summary."""
        recent_results = list(self.real_time_results)

        if not recent_results:
            return {"message": "No recent analytics data available"}

        # Calculate real-time metrics
        avg_quality = statistics.mean([r.quality_score for r in recent_results])
        avg_complexity = statistics.mean([r.overall_complexity for r in recent_results])

        sophistication_counts = defaultdict(int)
        for result in recent_results:
            sophistication_counts[result.sophistication_level] += 1

        return {
            "total_analyzed": len(recent_results),
            "average_quality": avg_quality,
            "average_complexity": avg_complexity,
            "sophistication_distribution": dict(sophistication_counts),
            "recent_high_quality": len([r for r in recent_results if r.quality_score > 0.7]),
            "recent_expert_level": len([r for r in recent_results if r.sophistication_level == "expert"]),
            "processing_rate": self.analytics_engine.metrics.processing_rate,
            "last_updated": datetime.now(tz=datetime.timezone.utc).isoformat()
        }


# Example usage and testing
async def main():
    """Example usage of the analytics orchestrator."""

    # Create orchestrator
    orchestrator = StreamingAnalyticsOrchestrator()

    # Test with example conversation
    example_conversation = {
        "id": "analytics_test_001",
        "messages": [
            {
                "role": "client",
                "content": "I've been struggling with severe anxiety and depression. The intrusive thoughts are overwhelming, and I feel like I'm losing control of my life."
            },
            {
                "role": "therapist",
                "content": "I hear the pain in your words, and I want you to know that what you're experiencing is valid and treatable. When you mention intrusive thoughts, can you help me understand what these thoughts are telling you? Sometimes our minds can create narratives that feel very real but may not reflect reality."
            },
            {
                "role": "client",
                "content": "They tell me I'm worthless, that everyone would be better off without me. I know logically that's not true, but emotionally it feels so real."
            },
            {
                "role": "therapist",
                "content": "That's a really important distinction you're making between logical and emotional truth. This suggests you have some metacognitive awareness, which is actually a strength we can build on. Let's explore some cognitive restructuring techniques to help you challenge these distorted thoughts when they arise."
            }
        ]
    }

    # Analyze conversation
    result = orchestrator.analytics_engine.analyze_conversation(example_conversation)

    # Print results

    for _rec in result.recommendations:
        pass

    # Get analytics summary
    orchestrator.analytics_engine.get_analytics_summary()


if __name__ == "__main__":
    asyncio.run(main())
