"""
Continuous quality monitor for real-time dataset quality tracking.
Monitors dataset quality during processing with real-time alerts.
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from conversation_schema import Conversation
from logger import get_logger

logger = get_logger(__name__)


@dataclass
class QualityAlert:
    """Quality alert notification."""
    alert_id: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime


class ContinuousQualityMonitor:
    """
    Continuous quality monitor for real-time tracking.

    Monitors dataset quality metrics in real-time and generates
    alerts when quality thresholds are breached.
    """

    def __init__(self):
        """Initialize the continuous quality monitor."""
        self.logger = get_logger(__name__)

        self.quality_thresholds = {
            "overall_quality": {"warning": 0.7, "critical": 0.5},
            "conversation_coherence": {"warning": 0.8, "critical": 0.6},
            "therapeutic_accuracy": {"warning": 0.75, "critical": 0.6},
            "safety_compliance": {"warning": 0.9, "critical": 0.8}
        }

        self.monitoring_active = False
        self.monitor_thread = None
        self.quality_history = []
        self.alert_callbacks = []
        self.current_conversations = []

        self.logger.info("ContinuousQualityMonitor initialized")

    def start_monitoring(self, check_interval: int = 60) -> bool:
        """Start continuous quality monitoring."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return False

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(check_interval,),
            daemon=True
        )
        self.monitor_thread.start()

        self.logger.info(f"Started continuous quality monitoring (interval: {check_interval}s)")
        return True

    def stop_monitoring(self) -> bool:
        """Stop continuous quality monitoring."""
        if not self.monitoring_active:
            self.logger.warning("Monitoring not active")
            return False

        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        self.logger.info("Stopped continuous quality monitoring")
        return True

    def update_conversations(self, conversations: list[Conversation]) -> None:
        """Update the conversations being monitored."""
        self.current_conversations = conversations
        self.logger.debug(f"Updated monitoring dataset: {len(conversations)} conversations")

    def add_alert_callback(self, callback: Callable[[QualityAlert], None]) -> None:
        """Add callback function for quality alerts."""
        self.alert_callbacks.append(callback)
        self.logger.debug("Added alert callback")

    def _monitoring_loop(self, check_interval: int) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                if self.current_conversations:
                    quality_metrics = self._calculate_current_quality()
                    self._check_thresholds(quality_metrics)
                    self._record_quality_history(quality_metrics)

                time.sleep(check_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(check_interval)

    def _calculate_current_quality(self) -> dict[str, float]:
        """Calculate current quality metrics."""
        if not self.current_conversations:
            return {}

        # Calculate basic quality metrics
        total_conversations = len(self.current_conversations)

        # Overall quality (based on conversation structure)
        valid_conversations = sum(
            1 for conv in self.current_conversations
            if len(conv.messages) >= 2 and all(len(msg.content.strip()) > 10 for msg in conv.messages)
        )
        overall_quality = valid_conversations / total_conversations if total_conversations > 0 else 0

        # Conversation coherence (proper turn-taking)
        coherent_conversations = sum(
            1 for conv in self.current_conversations
            if self._check_conversation_coherence(conv)
        )
        conversation_coherence = coherent_conversations / total_conversations if total_conversations > 0 else 0

        # Therapeutic accuracy (presence of therapeutic language)
        therapeutic_conversations = sum(
            1 for conv in self.current_conversations
            if self._check_therapeutic_content(conv)
        )
        therapeutic_accuracy = therapeutic_conversations / total_conversations if total_conversations > 0 else 0

        # Safety compliance (no unhandled safety issues)
        safe_conversations = sum(
            1 for conv in self.current_conversations
            if self._check_safety_compliance(conv)
        )
        safety_compliance = safe_conversations / total_conversations if total_conversations > 0 else 1.0

        return {
            "overall_quality": overall_quality,
            "conversation_coherence": conversation_coherence,
            "therapeutic_accuracy": therapeutic_accuracy,
            "safety_compliance": safety_compliance
        }

    def _check_conversation_coherence(self, conversation: Conversation) -> bool:
        """Check if conversation has proper coherence."""
        if len(conversation.messages) < 2:
            return False

        # Check for proper turn-taking
        roles = [msg.role for msg in conversation.messages]
        return all(roles[i] != roles[i+1] for i in range(len(roles)-1))

    def _check_therapeutic_content(self, conversation: Conversation) -> bool:
        """Check if conversation contains therapeutic content."""
        therapeutic_keywords = [
            "feel", "emotion", "support", "understand", "help",
            "therapy", "therapeutic", "coping", "healing"
        ]

        assistant_content = " ".join([
            msg.content.lower() for msg in conversation.messages
            if msg.role == "assistant"
        ])

        return any(keyword in assistant_content for keyword in therapeutic_keywords)

    def _check_safety_compliance(self, conversation: Conversation) -> bool:
        """Check if conversation handles safety issues appropriately."""
        safety_keywords = ["suicide", "self-harm", "danger", "crisis", "emergency"]

        all_content = " ".join(msg.content.lower() for msg in conversation.messages)

        # If no safety keywords, it's compliant
        if not any(keyword in all_content for keyword in safety_keywords):
            return True

        # If safety keywords present, check for appropriate responses
        assistant_content = " ".join([
            msg.content.lower() for msg in conversation.messages
            if msg.role == "assistant"
        ])

        safety_responses = ["safety", "help", "support", "professional", "crisis"]
        return any(response in assistant_content for response in safety_responses)

    def _check_thresholds(self, quality_metrics: dict[str, float]) -> None:
        """Check quality metrics against thresholds and generate alerts."""
        for metric_name, value in quality_metrics.items():
            if metric_name in self.quality_thresholds:
                thresholds = self.quality_thresholds[metric_name]

                # Check critical threshold
                if value < thresholds["critical"]:
                    alert = QualityAlert(
                        alert_id=f"{metric_name}_critical_{int(time.time())}",
                        severity="critical",
                        message=f"Critical quality issue: {metric_name} = {value:.3f} < {thresholds['critical']}",
                        metric_name=metric_name,
                        current_value=value,
                        threshold=thresholds["critical"],
                        timestamp=datetime.now()
                    )
                    self._trigger_alert(alert)

                # Check warning threshold
                elif value < thresholds["warning"]:
                    alert = QualityAlert(
                        alert_id=f"{metric_name}_warning_{int(time.time())}",
                        severity="warning",
                        message=f"Quality warning: {metric_name} = {value:.3f} < {thresholds['warning']}",
                        metric_name=metric_name,
                        current_value=value,
                        threshold=thresholds["warning"],
                        timestamp=datetime.now()
                    )
                    self._trigger_alert(alert)

    def _trigger_alert(self, alert: QualityAlert) -> None:
        """Trigger quality alert."""
        self.logger.warning(f"Quality Alert [{alert.severity.upper()}]: {alert.message}")

        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")

    def _record_quality_history(self, quality_metrics: dict[str, float]) -> None:
        """Record quality metrics in history."""
        history_entry = {
            "timestamp": datetime.now(),
            "metrics": quality_metrics.copy()
        }

        self.quality_history.append(history_entry)

        # Keep only last 100 entries
        if len(self.quality_history) > 100:
            self.quality_history = self.quality_history[-100:]

    def get_quality_history(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get quality history for specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        return [
            entry for entry in self.quality_history
            if entry["timestamp"] >= cutoff_time
        ]

    def get_current_status(self) -> dict[str, Any]:
        """Get current monitoring status."""
        current_metrics = self._calculate_current_quality() if self.current_conversations else {}

        return {
            "monitoring_active": self.monitoring_active,
            "conversations_monitored": len(self.current_conversations),
            "current_metrics": current_metrics,
            "alert_callbacks_registered": len(self.alert_callbacks),
            "quality_history_entries": len(self.quality_history),
            "thresholds": self.quality_thresholds
        }


def validate_continuous_quality_monitor():
    """Validate the ContinuousQualityMonitor functionality."""
    try:
        monitor = ContinuousQualityMonitor()
        assert hasattr(monitor, "start_monitoring")
        assert hasattr(monitor, "stop_monitoring")
        return True
    except Exception:
        return False


if __name__ == "__main__":
    if validate_continuous_quality_monitor():
        pass
    else:
        pass
