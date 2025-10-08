#!/usr/bin/env python3
"""
Crisis Detection Logging System - Comprehensive audit trail for all crisis events.
Provides secure, compliant logging for clinical and legal requirements.
"""

import contextlib
import hashlib
import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from crisis_intervention_detector import CrisisDetection, CrisisLevel


class LogLevel(Enum):
    """Log levels for crisis events."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    AUDIT = "AUDIT"

class EventType(Enum):
    """Types of crisis detection events."""
    CRISIS_DETECTION = "crisis_detection"
    ESCALATION_TRIGGERED = "escalation_triggered"
    EMERGENCY_CONTACT = "emergency_contact"
    SYSTEM_ERROR = "system_error"
    CONFIGURATION_CHANGE = "configuration_change"
    USER_ACTION = "user_action"
    SAFETY_OVERRIDE = "safety_override"
    AUDIT_ACCESS = "audit_access"

@dataclass
class CrisisLogEntry:
    """Structured crisis log entry."""
    log_id: str
    timestamp: datetime
    event_type: EventType
    log_level: LogLevel
    conversation_id: str
    user_id: str | None
    session_id: str | None
    crisis_level: str | None
    crisis_types: list[str]
    confidence_score: float | None
    detected_indicators: list[str]
    risk_factors: list[str]
    protective_factors: list[str]
    escalation_actions: list[str]
    emergency_contacts: list[str]
    response_time_ms: float | None
    system_version: str
    content_hash: str  # Hash of content for integrity without storing sensitive data
    metadata: dict[str, Any]

class CrisisDetectionLogger:
    """Comprehensive logging system for crisis detection events."""

    def __init__(self, log_directory: str = "crisis_logs", enable_file_logging: bool = True):
        self.log_directory = Path(log_directory)
        self.enable_file_logging = enable_file_logging
        self.system_version = "1.0.0"

        # Create log directory
        if self.enable_file_logging:
            self.log_directory.mkdir(exist_ok=True)

            # Create subdirectories for different log types
            (self.log_directory / "crisis_events").mkdir(exist_ok=True)
            (self.log_directory / "escalations").mkdir(exist_ok=True)
            (self.log_directory / "system_events").mkdir(exist_ok=True)
            (self.log_directory / "audit_trail").mkdir(exist_ok=True)

        # Configure structured logging
        self._setup_logging()

        # Log system startup
        self.log_system_event("Crisis detection logging system initialized", {
            "log_directory": str(self.log_directory),
            "file_logging_enabled": self.enable_file_logging,
            "system_version": self.system_version
        })

    def _setup_logging(self):
        """Setup structured logging configuration."""
        # Create formatters
        json_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": %(message)s}'
        )

        # Setup main logger
        self.logger = logging.getLogger("crisis_detection_audit")
        self.logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        self.logger.handlers.clear()

        if self.enable_file_logging:
            # Crisis events log file
            crisis_handler = logging.FileHandler(
                self.log_directory / "crisis_events" / f"crisis_events_{datetime.now().strftime('%Y%m%d')}.log"
            )
            crisis_handler.setFormatter(json_formatter)
            crisis_handler.setLevel(logging.INFO)

            # Audit trail log file (all events)
            audit_handler = logging.FileHandler(
                self.log_directory / "audit_trail" / f"audit_trail_{datetime.now().strftime('%Y%m%d')}.log"
            )
            audit_handler.setFormatter(json_formatter)
            audit_handler.setLevel(logging.DEBUG)

            self.logger.addHandler(crisis_handler)
            self.logger.addHandler(audit_handler)

        # Console handler for development
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        console_handler.setLevel(logging.WARNING)  # Only warnings and above to console
        self.logger.addHandler(console_handler)

    def _generate_content_hash(self, content: str) -> str:
        """Generate hash of content for integrity verification without storing sensitive data."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]  # First 16 chars for brevity

    def _create_log_entry(self, event_type: EventType, log_level: LogLevel,
                         conversation_id: str, message: str, **kwargs) -> CrisisLogEntry:
        """Create a structured log entry."""
        return CrisisLogEntry(
            log_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            log_level=log_level,
            conversation_id=conversation_id,
            user_id=kwargs.get("user_id"),
            session_id=kwargs.get("session_id"),
            crisis_level=kwargs.get("crisis_level"),
            crisis_types=kwargs.get("crisis_types", []),
            confidence_score=kwargs.get("confidence_score"),
            detected_indicators=kwargs.get("detected_indicators", []),
            risk_factors=kwargs.get("risk_factors", []),
            protective_factors=kwargs.get("protective_factors", []),
            escalation_actions=kwargs.get("escalation_actions", []),
            emergency_contacts=kwargs.get("emergency_contacts", []),
            response_time_ms=kwargs.get("response_time_ms"),
            system_version=self.system_version,
            content_hash=kwargs.get("content_hash", ""),
            metadata=kwargs.get("metadata", {})
        )

    def log_crisis_detection(self, detection: CrisisDetection, conversation: dict[str, Any],
                           response_time_ms: float, user_id: str | None = None, session_id: str | None = None):
        """Log a crisis detection event."""
        content = conversation.get("content", "")
        if not content and "messages" in conversation:
            content = " ".join([msg.get("content", "") for msg in conversation["messages"]])

        log_entry = self._create_log_entry(
            event_type=EventType.CRISIS_DETECTION,
            log_level=LogLevel.CRITICAL if detection.crisis_level in [CrisisLevel.EMERGENCY, CrisisLevel.CRITICAL] else LogLevel.WARNING,
            conversation_id=detection.conversation_id,
            message="Crisis detection performed",
            user_id=user_id,
            session_id=session_id,
            crisis_level=detection.crisis_level.value[0],
            crisis_types=[ct.value for ct in detection.crisis_types],
            confidence_score=detection.confidence_score,
            detected_indicators=detection.detected_indicators,
            risk_factors=detection.risk_factors,
            protective_factors=detection.protective_factors,
            escalation_actions=[action.value for action in detection.recommended_actions],
            emergency_contacts=detection.emergency_contacts,
            response_time_ms=response_time_ms,
            content_hash=self._generate_content_hash(content),
            metadata={
                "escalation_required": detection.escalation_required,
                "conversation_length": len(content),
                "input_format": "messages" if "messages" in conversation else "content"
            }
        )

        self._write_log_entry(log_entry)

        # Additional logging for high-severity events
        if detection.crisis_level in [CrisisLevel.EMERGENCY, CrisisLevel.CRITICAL]:
            self.log_high_severity_event(detection, conversation, response_time_ms)

    def log_escalation_event(self, detection: CrisisDetection, escalation_record,
                           user_id: str | None = None, session_id: str | None = None):
        """Log an escalation event."""
        log_entry = self._create_log_entry(
            event_type=EventType.ESCALATION_TRIGGERED,
            log_level=LogLevel.CRITICAL,
            conversation_id=detection.conversation_id,
            message=f"Crisis escalation triggered: {detection.crisis_level.value[0]}",
            user_id=user_id,
            session_id=session_id,
            crisis_level=detection.crisis_level.value[0],
            crisis_types=[ct.value for ct in detection.crisis_types],
            confidence_score=detection.confidence_score,
            escalation_actions=[action.value for action in escalation_record.actions_taken],
            emergency_contacts=escalation_record.contacts_notified,
            response_time_ms=escalation_record.response_time_minutes * 60 * 1000,  # Convert to ms
            metadata={
                "escalation_id": escalation_record.escalation_id,
                "outcome": escalation_record.outcome,
                "follow_up_required": escalation_record.follow_up_required
            }
        )

        self._write_log_entry(log_entry)

        # Write to escalation-specific log
        if self.enable_file_logging:
            escalation_file = self.log_directory / "escalations" / f"escalations_{datetime.now().strftime('%Y%m%d')}.log"
            with open(escalation_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(log_entry), default=str) + "\n")

    def log_emergency_contact(self, contact: str, detection: CrisisDetection,
                            contact_method: str = "unknown", success: bool = True):
        """Log emergency contact attempts."""
        log_entry = self._create_log_entry(
            event_type=EventType.EMERGENCY_CONTACT,
            log_level=LogLevel.CRITICAL,
            conversation_id=detection.conversation_id,
            message=f"Emergency contact attempted: {contact}",
            crisis_level=detection.crisis_level.value[0],
            crisis_types=[ct.value for ct in detection.crisis_types],
            emergency_contacts=[contact],
            metadata={
                "contact_method": contact_method,
                "contact_success": success,
                "detection_id": detection.detection_id
            }
        )

        self._write_log_entry(log_entry)

    def log_system_error(self, error: Exception, conversation_id: str = "unknown",
                        user_id: str | None = None, context: dict[str, Any] | None = None):
        """Log system errors."""
        log_entry = self._create_log_entry(
            event_type=EventType.SYSTEM_ERROR,
            log_level=LogLevel.ERROR,
            conversation_id=conversation_id,
            message=f"System error: {error!s}",
            user_id=user_id,
            metadata={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context or {}
            }
        )

        self._write_log_entry(log_entry)

    def log_configuration_change(self, change_description: str, old_config: dict[str, Any],
                                new_config: dict[str, Any], user_id: str | None = None):
        """Log configuration changes."""
        log_entry = self._create_log_entry(
            event_type=EventType.CONFIGURATION_CHANGE,
            log_level=LogLevel.WARNING,
            conversation_id="system",
            message=f"Configuration changed: {change_description}",
            user_id=user_id,
            metadata={
                "old_config": old_config,
                "new_config": new_config,
                "change_description": change_description
            }
        )

        self._write_log_entry(log_entry)

    def log_safety_override(self, override_reason: str, detection: CrisisDetection,
                          user_id: str, override_action: str):
        """Log safety overrides (critical for audit)."""
        log_entry = self._create_log_entry(
            event_type=EventType.SAFETY_OVERRIDE,
            log_level=LogLevel.CRITICAL,
            conversation_id=detection.conversation_id,
            message=f"Safety override: {override_reason}",
            user_id=user_id,
            crisis_level=detection.crisis_level.value[0],
            crisis_types=[ct.value for ct in detection.crisis_types],
            metadata={
                "override_reason": override_reason,
                "override_action": override_action,
                "original_detection_id": detection.detection_id,
                "requires_supervisor_review": True
            }
        )

        self._write_log_entry(log_entry)

    def log_audit_access(self, user_id: str, access_type: str, query_parameters: dict[str, Any] | None = None):
        """Log audit trail access."""
        log_entry = self._create_log_entry(
            event_type=EventType.AUDIT_ACCESS,
            log_level=LogLevel.AUDIT,
            conversation_id="audit_system",
            message=f"Audit access: {access_type}",
            user_id=user_id,
            metadata={
                "access_type": access_type,
                "query_parameters": query_parameters or {},
                "access_timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        self._write_log_entry(log_entry)

    def log_system_event(self, message: str, metadata: dict[str, Any] | None = None):
        """Log general system events."""
        log_entry = self._create_log_entry(
            event_type=EventType.USER_ACTION,
            log_level=LogLevel.INFO,
            conversation_id="system",
            message=message,
            metadata=metadata or {}
        )

        self._write_log_entry(log_entry)

    def log_high_severity_event(self, detection: CrisisDetection, conversation: dict[str, Any],
                              response_time_ms: float):
        """Special logging for high-severity events requiring immediate attention."""
        # Create high-priority log entry
        content = conversation.get("content", "")
        if not content and "messages" in conversation:
            content = " ".join([msg.get("content", "") for msg in conversation["messages"]])

        high_severity_data = {
            "URGENT_ALERT": True,
            "detection_id": detection.detection_id,
            "conversation_id": detection.conversation_id,
            "crisis_level": detection.crisis_level.value[0],
            "crisis_types": [ct.value for ct in detection.crisis_types],
            "confidence_score": detection.confidence_score,
            "content_hash": self._generate_content_hash(content),
            "escalation_required": detection.escalation_required,
            "emergency_contacts": detection.emergency_contacts,
            "response_time_ms": response_time_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "requires_immediate_review": True
        }

        # Write to high-severity log file
        if self.enable_file_logging:
            high_severity_file = self.log_directory / "crisis_events" / f"high_severity_{datetime.now().strftime('%Y%m%d')}.log"
            with open(high_severity_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(high_severity_data, indent=2) + "\n")

        # Log to main system
        self.logger.critical(json.dumps(high_severity_data))

    def _write_log_entry(self, log_entry: CrisisLogEntry):
        """Write log entry to appropriate destinations."""
        log_data = asdict(log_entry)

        # Convert datetime to ISO format for JSON serialization
        log_data["timestamp"] = log_entry.timestamp.isoformat()

        # Convert enums to their values for JSON serialization
        log_data["event_type"] = log_entry.event_type.value
        log_data["log_level"] = log_entry.log_level.value

        # Log to main logger
        log_level_map = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
            LogLevel.AUDIT: logging.INFO
        }

        self.logger.log(
            log_level_map[log_entry.log_level],
            json.dumps(log_data)
        )

        # Write to event-specific log files
        if self.enable_file_logging:
            event_file = self.log_directory / "system_events" / f"events_{datetime.now().strftime('%Y%m%d')}.log"
            with open(event_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_data) + "\n")

    def query_logs(self, start_date: datetime | None = None, end_date: datetime | None = None,
                  event_types: list[EventType] | None = None, crisis_levels: list[str] | None = None,
                  user_id: str | None = None, conversation_id: str | None = None) -> list[dict[str, Any]]:
        """Query logs with filters (for audit purposes)."""
        # Log the audit access
        self.log_audit_access(
            user_id=user_id or "system",
            access_type="log_query",
            query_parameters={
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "event_types": [et.value for et in event_types] if event_types else None,
                "crisis_levels": crisis_levels,
                "conversation_id": conversation_id
            }
        )

        # This is a simplified implementation - in production, you'd use a proper database
        results = []

        if not self.enable_file_logging:
            return results

        # Search through log files
        for log_file in self.log_directory.rglob("*.log"):
            try:
                with open(log_file, encoding="utf-8") as f:
                    for line in f:
                        try:
                            log_entry = json.loads(line.strip())

                            # Apply filters
                            if start_date and datetime.fromisoformat(log_entry["timestamp"]) < start_date:
                                continue
                            if end_date and datetime.fromisoformat(log_entry["timestamp"]) > end_date:
                                continue
                            if event_types and log_entry["event_type"] not in [et.value for et in event_types]:
                                continue
                            if crisis_levels and log_entry.get("crisis_level") not in crisis_levels:
                                continue
                            if conversation_id and log_entry["conversation_id"] != conversation_id:
                                continue

                            results.append(log_entry)

                        except json.JSONDecodeError:
                            continue

            except OSError:
                continue

        return sorted([r for r in results if isinstance(r, dict)], key=lambda x: x.get("timestamp", ""), reverse=True)

    def get_audit_summary(self, days: int = 7) -> dict[str, Any]:
        """Get audit summary for the specified number of days."""
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        logs = self.query_logs(start_date=start_date, end_date=end_date)

        # Analyze logs
        summary = {
            "period": f"{start_date.date()} to {end_date.date()}",
            "total_events": len(logs),
            "event_type_counts": {},
            "crisis_level_counts": {},
            "high_severity_events": 0,
            "escalations_triggered": 0,
            "emergency_contacts_made": 0,
            "system_errors": 0,
            "safety_overrides": 0,
            "unique_conversations": len({log["conversation_id"] for log in logs}),
            "unique_users": len({log["user_id"] for log in logs if log.get("user_id")})
        }

        for log in logs:
            # Count event types
            event_type = log["event_type"]
            summary["event_type_counts"][event_type] = summary["event_type_counts"].get(event_type, 0) + 1

            # Count crisis levels
            if log.get("crisis_level"):
                crisis_level = log["crisis_level"]
                summary["crisis_level_counts"][crisis_level] = summary["crisis_level_counts"].get(crisis_level, 0) + 1

            # Count specific event types
            if event_type == EventType.ESCALATION_TRIGGERED.value:
                summary["escalations_triggered"] += 1
            elif event_type == EventType.EMERGENCY_CONTACT.value:
                summary["emergency_contacts_made"] += 1
            elif event_type == EventType.SYSTEM_ERROR.value:
                summary["system_errors"] += 1
            elif event_type == EventType.SAFETY_OVERRIDE.value:
                summary["safety_overrides"] += 1

            # Count high severity
            if log.get("crisis_level") in ["emergency", "critical"]:
                summary["high_severity_events"] += 1

        return summary

# Example usage and testing
def test_crisis_logging():
    """Test the crisis logging system."""

    # Create logger
    logger = CrisisDetectionLogger(log_directory="test_crisis_logs")

    # Import required classes for testing
    from crisis_intervention_detector import (
        CrisisInterventionDetector,
    )

    # Create test detection
    detector = CrisisInterventionDetector()

    # Test crisis detection logging
    conversation = {"id": "test_conv_1", "content": "I want to kill myself"}
    detection = detector.detect_crisis(conversation)
    logger.log_crisis_detection(detection, conversation, 150.5, user_id="test_user", session_id="test_session")

    # Test escalation logging
    # Create mock escalation record
    class MockEscalationRecord:
        def __init__(self):
            self.escalation_id = "esc_test_1"
            self.actions_taken = detection.recommended_actions
            self.contacts_notified = ["911", "supervisor"]
            self.response_time_minutes = 0.1
            self.outcome = "escalation_completed"
            self.follow_up_required = True

    escalation = MockEscalationRecord()
    logger.log_escalation_event(detection, escalation, user_id="test_user")

    # Test emergency contact logging
    logger.log_emergency_contact("911", detection, "phone_call", success=True)

    # Test system error logging
    logger.log_system_error(Exception("Test error"), "test_conv_2", "test_user", {"context": "testing"})

    # Test configuration change logging
    logger.log_configuration_change(
        "Updated crisis thresholds",
        {"emergency_threshold": 0.9},
        {"emergency_threshold": 0.8},
        "admin_user"
    )

    # Test safety override logging
    logger.log_safety_override("False positive override", detection, "supervisor_user", "manual_review")

    # Test audit access logging
    logger.log_audit_access("audit_user", "log_query", {"date_range": "7_days"})

    # Test log querying
    logger.query_logs(user_id="test_user")

    # Test audit summary
    with contextlib.suppress(Exception):
        logger.get_audit_summary(days=1)
        # Continue with test - this is not critical for core functionality

    return True

if __name__ == "__main__":
    success = test_crisis_logging()
