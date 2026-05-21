# -*- coding: utf-8 -*-
"""Reflection-to-Action Pipeline — Sprint 4, Task 5.

Converts reflections into actionable recommendations, therapist notifications,
user-facing summaries, and feedback collection.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from ..schema import MemoryBlock
from .session_consolidation import SessionSummary
from .pattern_detection import PatternReport

log = logging.getLogger(__name__)


class ActionPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class ActionRecommendation:
    recommendation_id: str
    title: str
    description: str
    priority: ActionPriority
    measurable: bool
    related_topics: List[str]
    source: str


@dataclass(frozen=True)
class TherapistNotification:
    notification_id: str
    severity: str
    message: str
    session_id: str
    tenant_id: str
    timestamp_ms: int
    requires_response: bool


@dataclass(frozen=True)
class UserReflectionSummary:
    session_id: str
    summary_text: str
    key_insights: List[str]
    suggested_actions: List[str]
    therapist_approved: bool


@dataclass(frozen=True)
class UserFeedback:
    summary_id: str
    usefulness_rating: float  # 1-5
    helpful_aspects: List[str]
    unhelpful_aspects: List[str]
    timestamp_ms: int


@dataclass
class ActionResult:
    recommendations: List[ActionRecommendation]
    notifications: List[TherapistNotification]
    user_summary: UserReflectionSummary
    elapsed_ms: float


class ActionPipeline:
    """Convert reflection outputs into actionable items."""

    def __init__(
        self,
        therapist_notification_threshold: float = 0.8,
        min_action_confidence: float = 0.6,
    ) -> None:
        self._notification_threshold = therapist_notification_threshold
        self._min_confidence = min_action_confidence
        self._feedback_store: Dict[str, UserFeedback] = {}
        self._counter = 0

    def execute(
        self,
        session_summary: SessionSummary,
        pattern_report: Optional[PatternReport] = None,
    ) -> ActionResult:
        """Generate all action items from a session's reflection."""
        t0 = time.perf_counter()

        recommendations = self._generate_recommendations(
            session_summary, pattern_report
        )
        notifications = self._generate_notifications(session_summary)
        user_summary = self._generate_user_summary(
            session_summary, recommendations
        )

        elapsed = (time.perf_counter() - t0) * 1000

        result = ActionResult(
            recommendations=recommendations,
            notifications=notifications,
            user_summary=user_summary,
            elapsed_ms=round(elapsed, 2),
        )
        log.info(
            "Action pipeline: %d recommendations, %d notifications in %.0f ms",
            len(recommendations),
            len(notifications),
            elapsed,
        )
        return result

    def _generate_recommendations(
        self,
        summary: SessionSummary,
        pattern_report: Optional[PatternReport] = None,
    ) -> List[ActionRecommendation]:
        recommendations: List[ActionRecommendation] = []

        for topic in summary.unresolved_topics:
            self._counter += 1
            priority = (
                ActionPriority.CRITICAL
                if "crisis" in topic
                else ActionPriority.HIGH
                if "high_arousal" in topic
                else ActionPriority.MEDIUM
            )
            recommendations.append(
                ActionRecommendation(
                    recommendation_id=f"rec_{self._counter}",
                    title=f"Follow up on: {topic}",
                    description=f"Session ended with unresolved topic: {topic}. Schedule follow-up discussion.",
                    priority=priority,
                    measurable=True,
                    related_topics=[topic],
                    source="session_consolidation",
                )
            )

        if pattern_report:
            for trend in pattern_report.progress_trends:
                if trend.direction == "declining":
                    self._counter += 1
                    recommendations.append(
                        ActionRecommendation(
                            recommendation_id=f"rec_{self._counter}",
                            title=f"Address declining trend: {trend.metric}",
                            description=f"Detected declining trend in {trend.metric} (slope: {trend.slope:.4f}, confidence: {trend.confidence:.2f}). Consider intervention adjustment.",
                            priority=ActionPriority.HIGH,
                            measurable=True,
                            related_topics=[trend.metric],
                            source="pattern_detection",
                        )
                    )

            for trigger in pattern_report.trigger_patterns:
                if trigger.confidence >= self._notification_threshold:
                    self._counter += 1
                    recommendations.append(
                        ActionRecommendation(
                            recommendation_id=f"rec_{self._counter}",
                            title=f"Monitor trigger: {trigger.trigger}",
                            description=f"High-confidence trigger pattern: '{trigger.trigger}' → '{trigger.response}' ({trigger.confidence:.2f}). Prepare coping strategies.",
                            priority=ActionPriority.HIGH,
                            measurable=True,
                            related_topics=[trigger.trigger],
                            source="pattern_detection",
                        )
                    )

        if summary.emotional_arc.trend == "declining":
            self._counter += 1
            recommendations.append(
                ActionRecommendation(
                    recommendation_id=f"rec_{self._counter}",
                    title="Address declining emotional trajectory",
                    description=f"Emotional valence declined from {summary.emotional_arc.start_valence} to {summary.emotional_arc.end_valence}. Review session approach.",
                    priority=ActionPriority.HIGH,
                    measurable=True,
                    related_topics=["emotional_trajectory"],
                    source="emotional_arc",
                )
            )

        return recommendations

    def _generate_notifications(
        self, summary: SessionSummary
    ) -> List[TherapistNotification]:
        notifications: List[TherapistNotification] = []

        if summary.emotional_arc.trend == "declining" and summary.emotional_arc.end_valence < -0.3:
            self._counter += 1
            notifications.append(
                TherapistNotification(
                    notification_id=f"notif_{self._counter}",
                    severity="high",
                    message=f"Session {summary.session_id} ended with declining emotional trajectory (valence: {summary.emotional_arc.end_valence}). {len(summary.unresolved_topics)} unresolved topics.",
                    session_id=summary.session_id,
                    tenant_id=summary.tenant_id,
                    timestamp_ms=int(time.time() * 1000),
                    requires_response=True,
                )
            )

        crisis_topics = [t for t in summary.unresolved_topics if "crisis" in t]
        if crisis_topics:
            self._counter += 1
            notifications.append(
                TherapistNotification(
                    notification_id=f"notif_{self._counter}",
                    severity="critical",
                    message=f"Crisis content unresolved in session {summary.session_id}: {', '.join(crisis_topics)}",
                    session_id=summary.session_id,
                    tenant_id=summary.tenant_id,
                    timestamp_ms=int(time.time() * 1000),
                    requires_response=True,
                )
            )

        return notifications

    def _generate_user_summary(
        self,
        summary: SessionSummary,
        recommendations: List[ActionRecommendation],
    ) -> UserReflectionSummary:
        key_insights = [f"Theme: {t}" for t in summary.themes[:3]]
        if summary.emotional_arc.trend != "stable":
            key_insights.append(
                f"Emotional trend: {summary.emotional_arc.trend}"
            )

        suggested_actions = [
            r.title for r in recommendations if r.priority != ActionPriority.LOW
        ]

        return UserReflectionSummary(
            session_id=summary.session_id,
            summary_text=summary.summary_text,
            key_insights=key_insights,
            suggested_actions=suggested_actions,
            therapist_approved=False,
        )

    def record_feedback(self, feedback: UserFeedback) -> None:
        self._feedback_store[feedback.summary_id] = feedback

    def get_feedback(self, summary_id: str) -> Optional[UserFeedback]:
        return self._feedback_store.get(summary_id)

    @property
    def feedback_count(self) -> int:
        return len(self._feedback_store)
