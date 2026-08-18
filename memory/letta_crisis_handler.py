"""
Letta Crisis Handler - Integrates Foresight's crisis detection with Letta.

This handler intercepts Letta operations and applies crisis detection,
routing crisis content appropriately and blocking dangerous operations.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CrisisSeverity(Enum):
    """Crisis severity levels."""

    NONE = "none"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CrisisResult:
    """Result of crisis detection."""

    severity: CrisisSeverity
    indicators: list[str]
    requires_action: bool
    suggested_action: str | None = None


class LettaCrisisHandler:
    """Handles crisis detection and response for Letta agents."""

    def __init__(self, crisis_detector: Any, config: dict[str, Any] | None = None):
        """
        Initialize crisis handler with Foresight's crisis detector.

        Args:
            crisis_detector: Foresight's CrisisDetector instance
            config: Optional configuration dict
        """
        self.crisis_detector = crisis_detector
        self.config = config or {}
        self.alert_callback = self.config.get("alert_callback")

    async def check_message(self, message: str) -> CrisisResult:
        """
        Check a message for crisis indicators.

        Args:
            message: User message to analyze

        Returns:
            CrisisResult with severity and recommended actions
        """
        # Use Foresight's crisis detector
        try:
            severity_str = self.crisis_detector.get_severity(message)
        except Exception as e:
            logger.error(f"Crisis detection failed: {e}")
            severity_str = "none"

        # Map to our enum
        severity_map = {
            "none": CrisisSeverity.NONE,
            "medium": CrisisSeverity.MEDIUM,
            "high": CrisisSeverity.HIGH,
            "critical": CrisisSeverity.CRITICAL,
        }
        severity = severity_map.get(severity_str, CrisisSeverity.NONE)

        # Get indicators
        indicators = self._extract_indicators(message)

        # Determine if action required
        requires_action = severity in [CrisisSeverity.HIGH, CrisisSeverity.CRITICAL]

        # Suggest action
        suggested_action = self._suggest_action(severity, indicators)

        return CrisisResult(
            severity=severity, indicators=indicators, requires_action=requires_action, suggested_action=suggested_action
        )

    async def handle_crisis(self, result: CrisisResult, user_id: str, session_id: str | None = None) -> None:
        """
        Handle a crisis situation.

        Args:
            result: CrisisResult from check_message
            user_id: User identifier
            session_id: Optional session identifier
        """
        if result.severity == CrisisSeverity.NONE:
            return

        # Log crisis event
        await self._log_crisis(result, user_id, session_id)

        # Alert if critical
        if result.severity == CrisisSeverity.CRITICAL:
            await self._trigger_alert(result, user_id, session_id)

        # Route to appropriate resources
        await self._route_resources(result, user_id, session_id)

    def should_block_operation(self, result: CrisisResult, operation: str) -> bool:
        """
        Determine if an operation should be blocked due to crisis.

        Args:
            result: CrisisResult from check_message
            operation: Operation being attempted (e.g., "file_write", "code_execution")

        Returns:
            True if operation should be blocked
        """
        return (
            (result.severity == CrisisSeverity.CRITICAL and operation in ["file_write", "file_edit", "code_execution"])
            or ("self-harm" in result.indicators and operation in ["file_write", "shell_command"])
            or ("violence" in result.indicators and operation in ["shell_command", "code_execution"])
        )

    async def _log_crisis(self, result: CrisisResult, user_id: str, session_id: str | None = None) -> None:
        """Log crisis event to Foresight."""
        try:
            logger.info(
                f"Crisis detected for user {user_id} session {session_id}: "
                f"severity={result.severity.value}, "
                f"indicators={result.indicators}"
            )
        except Exception as e:
            logger.error(f"Failed to log crisis: {e}")

    async def _trigger_alert(self, result: CrisisResult, user_id: str, session_id: str | None = None) -> None:
        """Trigger alert for critical crisis."""
        if self.alert_callback:
            try:
                await self.alert_callback(result, user_id, session_id)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        else:
            logger.critical(f"CRISIS ALERT for user {user_id}: {result.severity.value} - {result.suggested_action}")

    async def _route_resources(self, result: CrisisResult, user_id: str, _session_id: str | None = None) -> None:
        """Route to appropriate resources based on crisis type."""
        # Standardized crisis resource mapping (aligned with Foresight-style escalation)
        crisis_resource_map = {
            "suicide": [
                "988 Suicide & Crisis Lifeline: 988",
                "Crisis Text Line: Text HOME to 741741",
                "Suicide & Crisis Lifeline Chat: suicidepreventionlifeline.org/chat",
            ],
            "self-harm": [
                "988 Suicide & Crisis Lifeline: 988",
                "Crisis Text Line: Text HOME to 741741",
            ],
            "violence": [
                "Emergency Services: 911",
                "If in immediate danger, call local emergency services now.",
            ],
            "severe_distress": [
                "988 Suicide & Crisis Lifeline: 988",
                "Crisis Text Line: Text HOME to 741741",
                "Consider immediate support from a licensed clinician",
            ],
        }

        routed_resources = []
        for indicator in result.indicators:
            routed_resources.extend(crisis_resource_map.get(indicator, []))

        if result.severity == CrisisSeverity.CRITICAL:
            routed_resources.append("Emergency Services: 911")
            routed_resources.append("Immediate emergency triage required")

        # Dedupe while preserving order
        unique_resources = list(dict.fromkeys(routed_resources))

        if not unique_resources:
            logger.info(
                "No crisis-specific resources were routed for user %s; severity=%s and no clear indicators.",
                user_id,
                result.severity.value,
            )
            return

        logger.info(
            "Routing crisis resources for user %s (severity=%s): %s",
            user_id,
            result.severity.value,
            unique_resources,
        )

    def _extract_indicators(self, message: str) -> list[str]:
        """Extract crisis indicators from message."""
        indicators = []
        message_lower = message.lower()

        # Check for suicide indicators
        suicide_keywords = [
            "suicide",
            "suicidal",
            "end my life",
            "kill myself",
            "want to die",
            "no reason to live",
            "life not worth living",
        ]
        if any(keyword in message_lower for keyword in suicide_keywords):
            indicators.append("suicide")

        # Check for self-harm indicators
        self_harm_keywords = [
            "self-harm",
            "self harm",
            "cut myself",
            "cutting",
            "hurt myself",
            "self-injury",
            "self injury",
        ]
        if any(keyword in message_lower for keyword in self_harm_keywords):
            indicators.append("self-harm")

        # Check for violence indicators
        violence_keywords = [
            "violence",
            "violent",
            "hurt someone",
            "kill someone",
            "harm someone",
            "attack someone",
            "shoot",
        ]
        if any(keyword in message_lower for keyword in violence_keywords):
            indicators.append("violence")

        # Check for depression/severe distress
        depression_keywords = ["hopeless", "desperate", "can't go on", "give up", "no point", "worthless", "burden"]
        if any(keyword in message_lower for keyword in depression_keywords):
            indicators.append("severe_distress")

        return indicators

    def _suggest_action(self, severity: CrisisSeverity, indicators: list[str]) -> str | None:
        """Suggest action based on crisis severity and type."""
        if severity == CrisisSeverity.CRITICAL:
            if "suicide" in indicators:
                action = (
                    "Provide suicide prevention resources immediately. "
                    "National Suicide Prevention Lifeline: 988 (US), "
                    "International: https://findahelpline.com"
                )
            elif "self-harm" in indicators:
                action = "Provide self-harm support resources. Crisis Text Line: Text HOME to 741741"
            elif "violence" in indicators:
                action = "De-escalate and provide support resources. If immediate danger, contact emergency services."
            else:
                action = "Critical situation detected. Provide supportive resources and encourage professional help."
        elif severity == CrisisSeverity.HIGH:
            if "suicide" in indicators or "self-harm" in indicators:
                action = "Provide supportive resources and encourage reaching out to professionals."
            else:
                action = "Monitor closely and provide supportive resources."
        elif severity == CrisisSeverity.MEDIUM:
            action = "Provide supportive response and monitor for escalation."
        else:
            action = None

        return action
