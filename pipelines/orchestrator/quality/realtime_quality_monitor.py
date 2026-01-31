"""
Real-time Quality Metrics for Dataset Acquisition

Live monitoring and assessment of dataset quality during acquisition process
with immediate feedback and quality-based filtering.
"""

import statistics
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from logger import get_logger

logger = get_logger(__name__)


class QualityLevel(Enum):
    """Quality assessment levels."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    REJECTED = "rejected"


@dataclass
class QualityMetrics:
    """Real-time quality metrics."""

    overall_score: float = 0.0
    content_quality: float = 0.0
    format_compliance: float = 0.0
    completeness: float = 0.0
    coherence: float = 0.0
    relevance: float = 0.0
    safety_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class QualityAlert:
    """Quality alert information."""

    alert_type: str
    severity: str  # low, medium, high, critical
    message: str
    metrics: QualityMetrics
    timestamp: datetime = field(default_factory=datetime.now)


class RealtimeQualityMonitor:
    """Real-time quality monitoring during dataset acquisition."""

    def __init__(self, window_size: int = 100, update_interval: float = 1.0):
        self.logger = get_logger(__name__)
        self.window_size = window_size
        self.update_interval = update_interval

        # Quality tracking
        self.quality_history: deque[QualityMetrics] = deque(maxlen=window_size)
        self.quality_thresholds = {
            QualityLevel.EXCELLENT: 0.9,
            QualityLevel.GOOD: 0.8,
            QualityLevel.ACCEPTABLE: 0.6,
            QualityLevel.POOR: 0.4,
            QualityLevel.REJECTED: 0.0,
        }

        # Alert system
        self.alerts: list[QualityAlert] = []
        self.alert_callbacks: list[Callable[[QualityAlert], None]] = []

        # Statistics
        self.total_items_processed = 0
        self.items_by_quality: dict[QualityLevel, int] = dict.fromkeys(QualityLevel, 0)
        self.rejection_rate = 0.0

        # Threading
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.metrics_lock = threading.Lock()

        # Quality assessment rules
        self.quality_rules = {
            "min_content_length": 10,
            "max_content_length": 10000,
            "required_fields": ["content", "messages"],
            "safety_keywords": ["harmful", "dangerous", "illegal"],
            "quality_keywords": ["helpful", "informative", "clear", "relevant"],
        }

        logger.info("RealtimeQualityMonitor initialized")

    def start_monitoring(self):
        """Start real-time quality monitoring."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Quality monitoring already running")
            return

        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        logger.info("Real-time quality monitoring started")

    def stop_monitoring_system(self):
        """Stop quality monitoring."""
        self.stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        logger.info("Real-time quality monitoring stopped")

    def assess_item_quality(self, item: dict[str, Any]) -> QualityMetrics:
        """Assess quality of a single dataset item."""
        metrics = QualityMetrics()

        try:
            # Content quality assessment
            metrics.content_quality = self._assess_content_quality(item)

            # Format compliance assessment
            metrics.format_compliance = self._assess_format_compliance(item)

            # Completeness assessment
            metrics.completeness = self._assess_completeness(item)

            # Coherence assessment
            metrics.coherence = self._assess_coherence(item)

            # Relevance assessment
            metrics.relevance = self._assess_relevance(item)

            # Safety assessment
            metrics.safety_score = self._assess_safety(item)

            # Calculate overall score
            metrics.overall_score = self._calculate_overall_score(metrics)

            # Update tracking
            with self.metrics_lock:
                self.quality_history.append(metrics)
                self.total_items_processed += 1

                # Update quality level counts
                quality_level = self._get_quality_level(metrics.overall_score)
                self.items_by_quality[quality_level] += 1

                # Update rejection rate
                rejected_count = self.items_by_quality[QualityLevel.REJECTED]
                self.rejection_rate = rejected_count / self.total_items_processed

            # Check for alerts
            self._check_quality_alerts(metrics)

            return metrics

        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return QualityMetrics(overall_score=0.0)

    def _assess_content_quality(self, item: dict[str, Any]) -> float:
        """Assess content quality."""
        score = 0.5  # Base score

        try:
            # Extract text content
            content = self._extract_text_content(item)

            if not content:
                return 0.0

            # Length check
            content_length = len(content.split())
            if (
                self.quality_rules["min_content_length"]
                <= content_length
                <= self.quality_rules["max_content_length"]
            ):
                score += 0.2
            else:
                score -= 0.2

            # Quality keywords
            quality_keywords = self.quality_rules["quality_keywords"]
            quality_matches = sum(
                1 for keyword in quality_keywords if keyword.lower() in content.lower()
            )
            score += min(0.2, quality_matches * 0.05)

            # Grammar and structure (simple heuristics)
            sentences = content.count(".") + content.count("!") + content.count("?")
            if sentences > 0:
                avg_sentence_length = content_length / sentences
                if 5 <= avg_sentence_length <= 30:  # Reasonable sentence length
                    score += 0.1

            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.error(f"Content quality assessment failed: {e}")
            return 0.0

    def _assess_format_compliance(self, item: dict[str, Any]) -> float:
        """Assess format compliance."""
        score = 0.0

        try:
            # Check required fields
            required_fields = self.quality_rules["required_fields"]
            present_fields = sum(1 for field in required_fields if field in item)
            score += (present_fields / len(required_fields)) * 0.5

            # Check data types
            if "messages" in item and isinstance(item["messages"], list):
                score += 0.2

                # Check message structure
                valid_messages = 0
                for msg in item["messages"]:
                    if isinstance(msg, dict) and "content" in msg and "role" in msg:
                        valid_messages += 1

                if len(item["messages"]) > 0:
                    score += (valid_messages / len(item["messages"])) * 0.3

            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.error(f"Format compliance assessment failed: {e}")
            return 0.0

    def _assess_completeness(self, item: dict[str, Any]) -> float:
        """Assess data completeness."""
        score = 0.0

        try:
            # Check for essential fields
            essential_fields = ["content", "messages", "id"]
            present_essential = sum(
                1 for field in essential_fields if item.get(field)
            )
            score += (present_essential / len(essential_fields)) * 0.4

            # Check for metadata
            if "metadata" in item and isinstance(item["metadata"], dict):
                score += 0.2

                # Check metadata completeness
                metadata_fields = ["source", "quality_score", "timestamp"]
                present_metadata = sum(
                    1 for field in metadata_fields if field in item["metadata"]
                )
                score += (present_metadata / len(metadata_fields)) * 0.2

            # Check content completeness
            content = self._extract_text_content(item)
            if content and len(content.strip()) > 0:
                score += 0.2

            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.error(f"Completeness assessment failed: {e}")
            return 0.0

    def _assess_coherence(self, item: dict[str, Any]) -> float:
        """Assess content coherence."""
        score = 0.5  # Base score

        try:
            content = self._extract_text_content(item)
            if not content:
                return 0.0

            # Check for conversation flow (if messages present)
            if "messages" in item and isinstance(item["messages"], list):
                messages = item["messages"]
                if len(messages) >= 2:
                    # Check role alternation
                    roles = [msg.get("role", "") for msg in messages]
                    alternating = all(
                        roles[i] != roles[i + 1]
                        for i in range(len(roles) - 1)
                        if roles[i] and roles[i + 1]
                    )
                    if alternating:
                        score += 0.2

                    # Check content relevance between messages
                    contents = [msg.get("content", "") for msg in messages]
                    if len(contents) >= 2:
                        # Simple word overlap check
                        words1 = set(contents[0].lower().split())
                        words2 = set(contents[1].lower().split())
                        overlap = len(words1 & words2) / max(len(words1 | words2), 1)
                        score += overlap * 0.2

            # Check for logical connectors
            connectors = [
                "because",
                "therefore",
                "however",
                "although",
                "since",
                "thus",
            ]
            connector_count = sum(
                1 for connector in connectors if connector in content.lower()
            )
            score += min(0.1, connector_count * 0.02)

            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.error(f"Coherence assessment failed: {e}")
            return 0.5

    def _assess_relevance(self, item: dict[str, Any]) -> float:
        """Assess content relevance."""
        score = 0.5  # Base score

        try:
            content = self._extract_text_content(item)
            if not content:
                return 0.0

            # Check for therapeutic/empathetic content
            therapeutic_keywords = [
                "feel",
                "emotion",
                "support",
                "understand",
                "help",
                "therapy",
                "counseling",
                "mental health",
                "wellbeing",
            ]

            therapeutic_matches = sum(
                1
                for keyword in therapeutic_keywords
                if keyword.lower() in content.lower()
            )
            score += min(0.3, therapeutic_matches * 0.05)

            # Check for conversational patterns
            conversational_patterns = [
                "how are you",
                "i understand",
                "tell me more",
                "that sounds",
            ]
            pattern_matches = sum(
                1
                for pattern in conversational_patterns
                if pattern.lower() in content.lower()
            )
            score += min(0.2, pattern_matches * 0.1)

            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.error(f"Relevance assessment failed: {e}")
            return 0.5

    def _assess_safety(self, item: dict[str, Any]) -> float:
        """Assess content safety."""
        score = 1.0  # Start with perfect safety score

        try:
            content = self._extract_text_content(item)
            if not content:
                return 1.0

            # Check for safety keywords (negative indicators)
            safety_keywords = self.quality_rules["safety_keywords"]
            safety_violations = sum(
                1 for keyword in safety_keywords if keyword.lower() in content.lower()
            )
            score -= safety_violations * 0.2

            # Check for inappropriate content patterns
            inappropriate_patterns = [
                "kill yourself",
                "end it all",
                "not worth living",
                "harmful advice",
                "dangerous behavior",
            ]

            for pattern in inappropriate_patterns:
                if pattern.lower() in content.lower():
                    score -= 0.3

            # Check for personal information (privacy concern)
            pii_patterns = [
                r"\b\d{3}-\d{2}-\d{4}\b",  # SSN pattern
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email pattern
            ]

            import re

            for pattern in pii_patterns:
                if re.search(pattern, content):
                    score -= 0.1

            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.error(f"Safety assessment failed: {e}")
            return 0.5

    def _calculate_overall_score(self, metrics: QualityMetrics) -> float:
        """Calculate overall quality score."""
        weights = {
            "content_quality": 0.25,
            "format_compliance": 0.15,
            "completeness": 0.20,
            "coherence": 0.15,
            "relevance": 0.15,
            "safety_score": 0.10,
        }

        overall = (
            metrics.content_quality * weights["content_quality"]
            + metrics.format_compliance * weights["format_compliance"]
            + metrics.completeness * weights["completeness"]
            + metrics.coherence * weights["coherence"]
            + metrics.relevance * weights["relevance"]
            + metrics.safety_score * weights["safety_score"]
        )

        return max(0.0, min(1.0, overall))

    def _extract_text_content(self, item: dict[str, Any]) -> str:
        """Extract text content from item."""
        content_parts = []

        # Direct content field
        if "content" in item:
            content_parts.append(str(item["content"]))

        # Messages content
        if "messages" in item and isinstance(item["messages"], list):
            for message in item["messages"]:
                if isinstance(message, dict) and "content" in message:
                    content_parts.append(str(message["content"]))

        # Text field
        if "text" in item:
            content_parts.append(str(item["text"]))

        return " ".join(content_parts)

    def _get_quality_level(self, score: float) -> QualityLevel:
        """Get quality level from score."""
        for level, threshold in sorted(
            self.quality_thresholds.items(), key=lambda x: x[1], reverse=True
        ):
            if score >= threshold:
                return level
        return QualityLevel.REJECTED

    def _check_quality_alerts(self, metrics: QualityMetrics):
        """Check for quality-based alerts."""
        alerts = []

        # Low overall quality alert
        if metrics.overall_score < 0.3:
            alerts.append(
                QualityAlert(
                    alert_type="low_quality",
                    severity="high",
                    message=f"Very low quality item detected (score: {metrics.overall_score:.2f})",
                    metrics=metrics,
                )
            )

        # Safety alert
        if metrics.safety_score < 0.5:
            alerts.append(
                QualityAlert(
                    alert_type="safety_concern",
                    severity="critical",
                    message=f"Safety concern detected (score: {metrics.safety_score:.2f})",
                    metrics=metrics,
                )
            )

        # Format compliance alert
        if metrics.format_compliance < 0.5:
            alerts.append(
                QualityAlert(
                    alert_type="format_issue",
                    severity="medium",
                    message=f"Format compliance issue (score: {metrics.format_compliance:.2f})",
                    metrics=metrics,
                )
            )

        # Store and notify alerts
        for alert in alerts:
            self.alerts.append(alert)
            self._notify_alert_callbacks(alert)

    def _notify_alert_callbacks(self, alert: QualityAlert):
        """Notify alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_monitoring.is_set():
            try:
                # Calculate current statistics
                self._update_statistics()

                # Check for system-level alerts
                self._check_system_alerts()

                # Sleep until next update
                self.stop_monitoring.wait(timeout=self.update_interval)

            except Exception as e:
                logger.error(f"Error in quality monitoring loop: {e}")
                self.stop_monitoring.wait(timeout=60)

    def _update_statistics(self):
        """Update quality statistics."""
        with self.metrics_lock:
            if len(self.quality_history) > 0:
                recent_scores = [m.overall_score for m in self.quality_history]
                statistics.mean(recent_scores)

                # Log quality trends
                if len(self.quality_history) >= self.window_size:
                    first_half = recent_scores[: self.window_size // 2]
                    second_half = recent_scores[self.window_size // 2 :]

                    if statistics.mean(second_half) < statistics.mean(first_half) - 0.1:
                        logger.warning("Quality trend declining")

    def _check_system_alerts(self):
        """Check for system-level quality alerts."""
        with self.metrics_lock:
            # High rejection rate alert
            if self.rejection_rate > 0.5:
                alert = QualityAlert(
                    alert_type="high_rejection_rate",
                    severity="high",
                    message=f"High rejection rate: {self.rejection_rate:.2%}",
                    metrics=QualityMetrics(),
                )
                self._notify_alert_callbacks(alert)

    def add_alert_callback(self, callback: Callable[[QualityAlert], None]):
        """Add alert callback."""
        self.alert_callbacks.append(callback)
        logger.info("Added quality alert callback")

    def get_current_statistics(self) -> dict[str, Any]:
        """Get current quality statistics."""
        with self.metrics_lock:
            recent_scores = (
                [m.overall_score for m in self.quality_history]
                if self.quality_history
                else [0]
            )

            return {
                "total_processed": self.total_items_processed,
                "rejection_rate": self.rejection_rate,
                "average_quality": statistics.mean(recent_scores),
                "quality_distribution": {
                    level.value: count for level, count in self.items_by_quality.items()
                },
                "recent_quality_trend": (
                    statistics.mean(recent_scores[-10:])
                    if len(recent_scores) >= 10
                    else statistics.mean(recent_scores)
                ),
                "alert_count": len(self.alerts),
                "timestamp": datetime.now().isoformat(),
            }

    def should_accept_item(self, metrics: QualityMetrics) -> bool:
        """Determine if item should be accepted based on quality."""
        return (
            metrics.overall_score >= self.quality_thresholds[QualityLevel.ACCEPTABLE]
            and metrics.safety_score >= 0.7
        )


# Example alert callback
def quality_alert_handler(alert: QualityAlert):
    """Example quality alert handler."""
    severity_emoji = {"low": "ℹ️", "medium": "⚠️", "high": "🚨", "critical": "🔥"}

    emoji = severity_emoji.get(alert.severity, "❓")
    logger.warning(f"{emoji} QUALITY ALERT [{alert.alert_type}]: {alert.message}")


# Example usage
if __name__ == "__main__":
    monitor = RealtimeQualityMonitor()
    monitor.add_alert_callback(quality_alert_handler)
    monitor.start_monitoring()

    # Example item assessment
    test_item = {
        "id": "test_1",
        "messages": [
            {
                "role": "user",
                "content": "I'm feeling really anxious about my job interview tomorrow.",
            },
            {
                "role": "assistant",
                "content": "I understand that job interviews can be nerve-wracking. It's completely normal to feel anxious. Would you like to talk about what specifically is making you feel this way?",
            },
        ],
        "metadata": {"source": "empathetic_dialogues", "quality_score": 0.85},
    }

    metrics = monitor.assess_item_quality(test_item)

    # Get statistics
    stats = monitor.get_current_statistics()

    monitor.stop_monitoring_system()
