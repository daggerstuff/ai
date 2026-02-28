"""
Simple Feedback Loops System

Simplified feedback system for dataset quality improvement based on model performance.
"""

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from logger import get_logger

logger = get_logger(__name__)


class FeedbackType(Enum):
    """Types of feedback."""

    QUALITY_SCORE = "quality_score"
    PERFORMANCE_METRIC = "performance_metric"
    USER_RATING = "user_rating"
    ERROR_REPORT = "error_report"


@dataclass
class FeedbackData:
    """Feedback data structure."""

    feedback_id: str = ""
    feedback_type: FeedbackType = FeedbackType.QUALITY_SCORE
    value: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"


@dataclass
class ImprovementAction:
    """Action to improve dataset quality."""

    action_id: str = ""
    description: str = ""
    target_metric: str = ""
    expected_improvement: float = 0.0
    implemented: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


class SimpleFeedbackLoops:
    """Simple feedback loops system."""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.feedback_history: list[FeedbackData] = []
        self.improvement_actions: list[ImprovementAction] = []
        self.feedback_lock = threading.Lock()

        # Thresholds for triggering improvements
        self.quality_threshold = 0.7
        self.performance_threshold = 0.8

        logger.info("SimpleFeedbackLoops initialized")

    def submit_feedback(self, feedback: FeedbackData) -> None:
        """Submit feedback data."""
        with self.feedback_lock:
            self.feedback_history.append(feedback)

            # Check if improvement action is needed
            self._evaluate_improvement_need(feedback)

        logger.info(
            f"Feedback submitted: {feedback.feedback_type.value} = {feedback.value}"
        )

    def _evaluate_improvement_need(self, feedback: FeedbackData) -> None:
        """Evaluate if improvement action is needed."""
        if feedback.feedback_type == FeedbackType.QUALITY_SCORE:
            if feedback.value < self.quality_threshold:
                action = ImprovementAction(
                    action_id=f"quality_improvement_{int(time.time())}",
                    description=f"Improve quality score from {feedback.value} to above {self.quality_threshold}",
                    target_metric="quality_score",
                    expected_improvement=self.quality_threshold - feedback.value,
                )
                self.improvement_actions.append(action)
                logger.info(f"Quality improvement action created: {action.action_id}")

        elif feedback.feedback_type == FeedbackType.PERFORMANCE_METRIC:
            if feedback.value < self.performance_threshold:
                action = ImprovementAction(
                    action_id=f"performance_improvement_{int(time.time())}",
                    description=f"Improve performance from {feedback.value} to above {self.performance_threshold}",
                    target_metric="performance",
                    expected_improvement=self.performance_threshold - feedback.value,
                )
                self.improvement_actions.append(action)
                logger.info(
                    f"Performance improvement action created: {action.action_id}"
                )

    def get_recent_feedback(self, hours: int = 24) -> list[FeedbackData]:
        """Get recent feedback within specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        with self.feedback_lock:
            return [
                feedback
                for feedback in self.feedback_history
                if feedback.timestamp > cutoff_time
            ]

    def get_improvement_actions(
        self, implemented_only: bool = False
    ) -> list[ImprovementAction]:
        """Get improvement actions."""
        if implemented_only:
            return [action for action in self.improvement_actions if action.implemented]
        return self.improvement_actions.copy()

    def implement_action(self, action_id: str) -> bool:
        """Mark an improvement action as implemented."""
        for action in self.improvement_actions:
            if action.action_id == action_id:
                action.implemented = True
                logger.info(f"Action implemented: {action_id}")
                return True

        return False

    def get_feedback_summary(self) -> dict[str, Any]:
        """Get feedback summary statistics."""
        with self.feedback_lock:
            if not self.feedback_history:
                return {}

            # Calculate averages by type
            feedback_by_type = {}
            for feedback in self.feedback_history:
                feedback_type = feedback.feedback_type.value
                if feedback_type not in feedback_by_type:
                    feedback_by_type[feedback_type] = []
                feedback_by_type[feedback_type].append(feedback.value)

            summary = {}
            for feedback_type, values in feedback_by_type.items():
                summary[feedback_type] = {
                    "count": len(values),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "latest": values[-1] if values else 0,
                }

            summary["total_feedback"] = len(self.feedback_history)
            summary["pending_actions"] = len(
                [a for a in self.improvement_actions if not a.implemented]
            )
            summary["completed_actions"] = len(
                [a for a in self.improvement_actions if a.implemented]
            )

            return summary

    def generate_feedback_report(
        self, output_path: str = "feedback_report.json"
    ) -> str:
        """Generate comprehensive feedback report."""
        report = {
            "report_type": "Feedback Loops Analysis",
            "generated_at": datetime.now().isoformat(),
            "summary": self.get_feedback_summary(),
            "recent_feedback": [
                {
                    "id": f.feedback_id,
                    "type": f.feedback_type.value,
                    "value": f.value,
                    "source": f.source,
                    "timestamp": f.timestamp.isoformat(),
                }
                for f in self.get_recent_feedback(24)
            ],
            "improvement_actions": [
                {
                    "id": a.action_id,
                    "description": a.description,
                    "target_metric": a.target_metric,
                    "expected_improvement": a.expected_improvement,
                    "implemented": a.implemented,
                    "timestamp": a.timestamp.isoformat(),
                }
                for a in self.improvement_actions
            ],
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Feedback report generated: {output_path}")
        return output_path


# Example usage
if __name__ == "__main__":
    feedback_system = SimpleFeedbackLoops()

    # Submit some test feedback
    feedback1 = FeedbackData(
        feedback_id="test_1",
        feedback_type=FeedbackType.QUALITY_SCORE,
        value=0.6,  # Below threshold
        source="test_system",
    )

    feedback2 = FeedbackData(
        feedback_id="test_2",
        feedback_type=FeedbackType.PERFORMANCE_METRIC,
        value=0.9,  # Above threshold
        source="test_system",
    )

    feedback_system.submit_feedback(feedback1)
    feedback_system.submit_feedback(feedback2)

    # Get summary
    summary = feedback_system.get_feedback_summary()

    # Generate report
    report_path = feedback_system.generate_feedback_report()

    # Clean up
    import os

    if os.path.exists("feedback_report.json"):
        os.remove("feedback_report.json")
