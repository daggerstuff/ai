"""
Safety and Ethics Validation Audit Trail System

Provides comprehensive audit trails for safety and ethics validation processes,
tracking all validation decisions, interventions, and dataset changes with
reproducible logging and change tracking.
"""

import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""
    VALIDATION_STARTED = "validation_started"
    SAFETY_ISSUE_DETECTED = "safety_issue_detected"
    ETHICS_VIOLATION_DETECTED = "ethics_violation_detected"
    INTERVENTION_REQUIRED = "intervention_required"
    MANUAL_REVIEW_REQUESTED = "manual_review_requested"
    MANUAL_REVIEW_COMPLETED = "manual_review_completed"
    DATASET_CHANGE = "dataset_change"
    CONFIGURATION_CHANGE = "configuration_change"
    ALERT_TRIGGERED = "alert_triggered"
    REMEDIATION_APPLIED = "remediation_applied"
    VALIDATION_COMPLETED = "validation_completed"


class ChangeType(Enum):
    """Types of dataset changes."""
    CONTENT_MODIFIED = "content_modified"
    CONTENT_REMOVED = "content_removed"
    CONTENT_FLAGGED = "content_flagged"
    CONTENT_APPROVED = "content_approved"
    METADATA_UPDATED = "metadata_updated"


@dataclass
class AuditEvent:
    """Individual audit event."""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    conversation_id: str
    user_id: Optional[str]
    details: Dict[str, Any]
    previous_state: Optional[Dict[str, Any]] = None
    new_state: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class DatasetChange:
    """Record of dataset modification."""
    change_id: str
    conversation_id: str
    change_type: ChangeType
    timestamp: datetime
    user_id: Optional[str]
    details: Dict[str, Any]
    previous_content: Optional[str] = None
    new_content: Optional[str] = None
    change_reason: Optional[str] = None
    approval_status: str = "pending"


@dataclass
class RemediationAction:
    """Record of remediation applied."""
    action_id: str
    conversation_id: str
    action_type: str
    timestamp: datetime
    user_id: Optional[str]
    details: Dict[str, Any]
    effectiveness_score: float = 0.0
    follow_up_required: bool = False


class SafetyEthicsAuditTrail:
    """
    Comprehensive audit trail system for safety and ethics validation.
    
    Tracks all validation activities, interventions, manual reviews,
    dataset changes, and remediation actions with full reproducibility.
    """
    
    def __init__(self, audit_log_path: Optional[str] = None):
        """Initialize the audit trail system."""
        self.audit_events: List[AuditEvent] = []
        self.dataset_changes: List[DatasetChange] = []
        self.remediation_actions: List[RemediationAction] = []
        self.audit_log_path = audit_log_path
        self.enabled = True
        
        logger.info("SafetyEthicsAuditTrail initialized")
    
    def log_validation_started(self, conversation_id: str, 
                              user_id: Optional[str] = None,
                              session_id: Optional[str] = None) -> str:
        """Log validation start event."""
        event_id = self._generate_event_id()
        event = AuditEvent(
            event_id=event_id,
            event_type=AuditEventType.VALIDATION_STARTED,
            timestamp=datetime.now(),
            conversation_id=conversation_id,
            user_id=user_id,
            details={"session_id": session_id},
            session_id=session_id
        )
        self._add_audit_event(event)
        return event_id
    
    def log_safety_issue(self, conversation_id: str,
                        safety_issue: Dict[str, Any],
                        user_id: Optional[str] = None) -> str:
        """Log safety issue detection."""
        event_id = self._generate_event_id()
        event = AuditEvent(
            event_id=event_id,
            event_type=AuditEventType.SAFETY_ISSUE_DETECTED,
            timestamp=datetime.now(),
            conversation_id=conversation_id,
            user_id=user_id,
            details=safety_issue
        )
        self._add_audit_event(event)
        return event_id
    
    def log_ethics_violation(self, conversation_id: str,
                           ethics_violation: Dict[str, Any],
                           user_id: Optional[str] = None) -> str:
        """Log ethics violation detection."""
        event_id = self._generate_event_id()
        event = AuditEvent(
            event_id=event_id,
            event_type=AuditEventType.ETHICS_VIOLATION_DETECTED,
            timestamp=datetime.now(),
            conversation_id=conversation_id,
            user_id=user_id,
            details=ethics_violation
        )
        self._add_audit_event(event)
        return event_id
    
    def log_intervention_required(self, conversation_id: str,
                                reason: str,
                                urgency_level: str,
                                user_id: Optional[str] = None) -> str:
        """Log intervention requirement."""
        event_id = self._generate_event_id()
        event = AuditEvent(
            event_id=event_id,
            event_type=AuditEventType.INTERVENTION_REQUIRED,
            timestamp=datetime.now(),
            conversation_id=conversation_id,
            user_id=user_id,
            details={
                "reason": reason,
                "urgency_level": urgency_level
            }
        )
        self._add_audit_event(event)
        return event_id
    
    def log_manual_review_request(self, conversation_id: str,
                                  reviewer_id: str,
                                  reason: str,
                                  user_id: Optional[str] = None) -> str:
        """Log manual review request."""
        event_id = self._generate_event_id()
        event = AuditEvent(
            event_id=event_id,
            event_type=AuditEventType.MANUAL_REVIEW_REQUESTED,
            timestamp=datetime.now(),
            conversation_id=conversation_id,
            user_id=user_id,
            details={
                "reviewer_id": reviewer_id,
                "reason": reason
            }
        )
        self._add_audit_event(event)
        return event_id
    
    def log_manual_review_completion(self, conversation_id: str,
                                    reviewer_id: str,
                                    decision: str,
                                    comments: str,
                                    user_id: Optional[str] = None) -> str:
        """Log manual review completion."""
        event_id = self._generate_event_id()
        event = AuditEvent(
            event_id=event_id,
            event_type=AuditEventType.MANUAL_REVIEW_COMPLETED,
            timestamp=datetime.now(),
            conversation_id=conversation_id,
            user_id=user_id,
            details={
                "reviewer_id": reviewer_id,
                "decision": decision,
                "comments": comments
            }
        )
        self._add_audit_event(event)
        return event_id
    
    def log_dataset_change(self, conversation_id: str,
                          change_type: ChangeType,
                          details: Dict[str, Any],
                          previous_content: Optional[str] = None,
                          new_content: Optional[str] = None,
                          change_reason: Optional[str] = None,
                          user_id: Optional[str] = None) -> str:
        """Log dataset change."""
        change_id = self._generate_event_id()
        change = DatasetChange(
            change_id=change_id,
            conversation_id=conversation_id,
            change_type=change_type,
            timestamp=datetime.now(),
            user_id=user_id,
            details=details,
            previous_content=previous_content,
            new_content=new_content,
            change_reason=change_reason
        )
        self.dataset_changes.append(change)
        
        # Also log as audit event
        event = AuditEvent(
            event_id=change_id,
            event_type=AuditEventType.DATASET_CHANGE,
            timestamp=datetime.now(),
            conversation_id=conversation_id,
            user_id=user_id,
            details={
                "change_type": change_type.value,
                "change_details": details,
                "change_reason": change_reason
            },
            previous_state={"content": previous_content} if previous_content else None,
            new_state={"content": new_content} if new_content else None
        )
        self._add_audit_event(event)
        
        logger.info(f"Dataset change logged: {change_id} for conversation {conversation_id}")
        return change_id
    
    def log_configuration_change(self, config_key: str,
                               old_value: Any,
                               new_value: Any,
                               user_id: Optional[str] = None,
                               reason: Optional[str] = None) -> str:
        """Log configuration change."""
        event_id = self._generate_event_id()
        event = AuditEvent(
            event_id=event_id,
            event_type=AuditEventType.CONFIGURATION_CHANGE,
            timestamp=datetime.now(),
            conversation_id="N/A",
            user_id=user_id,
            details={
                "config_key": config_key,
                "old_value": str(old_value),
                "new_value": str(new_value),
                "reason": reason
            },
            previous_state={"config_value": old_value},
            new_state={"config_value": new_value}
        )
        self._add_audit_event(event)
        return event_id
    
    def log_alert_trigger(self, alert_type: str,
                         severity: str,
                         conversation_id: str,
                         details: Dict[str, Any],
                         user_id: Optional[str] = None) -> str:
        """Log alert trigger."""
        event_id = self._generate_event_id()
        event = AuditEvent(
            event_id=event_id,
            event_type=AuditEventType.ALERT_TRIGGERED,
            timestamp=datetime.now(),
            conversation_id=conversation_id,
            user_id=user_id,
            details={
                "alert_type": alert_type,
                "severity": severity,
                "alert_details": details
            }
        )
        self._add_audit_event(event)
        return event_id
    
    def log_remediation_action(self, conversation_id: str,
                              action_type: str,
                              details: Dict[str, Any],
                              effectiveness_score: float = 0.0,
                              follow_up_required: bool = False,
                              user_id: Optional[str] = None) -> str:
        """Log remediation action."""
        action_id = self._generate_event_id()
        action = RemediationAction(
            action_id=action_id,
            conversation_id=conversation_id,
            action_type=action_type,
            timestamp=datetime.now(),
            user_id=user_id,
            details=details,
            effectiveness_score=effectiveness_score,
            follow_up_required=follow_up_required
        )
        self.remediation_actions.append(action)
        
        # Also log as audit event
        event = AuditEvent(
            event_id=action_id,
            event_type=AuditEventType.REMEDIATION_APPLIED,
            timestamp=datetime.now(),
            conversation_id=conversation_id,
            user_id=user_id,
            details={
                "action_type": action_type,
                "action_details": details,
                "effectiveness_score": effectiveness_score,
                "follow_up_required": follow_up_required
            }
        )
        self._add_audit_event(event)
        
        logger.info(f"Remediation action logged: {action_id} for conversation {conversation_id}")
        return action_id
    
    def log_validation_completion(self, conversation_id: str,
                                 validation_result: Dict[str, Any],
                                 user_id: Optional[str] = None) -> str:
        """Log validation completion."""
        event_id = self._generate_event_id()
        event = AuditEvent(
            event_id=event_id,
            event_type=AuditEventType.VALIDATION_COMPLETED,
            timestamp=datetime.now(),
            conversation_id=conversation_id,
            user_id=user_id,
            details=validation_result
        )
        self._add_audit_event(event)
        return event_id
    
    def _add_audit_event(self, event: AuditEvent) -> None:
        """Add audit event to the trail."""
        if not self.enabled:
            return
        
        self.audit_events.append(event)
        
        # Log to file if configured
        if self.audit_log_path:
            try:
                with open(self.audit_log_path, 'a') as f:
                    json.dump({
                        'event_id': event.event_id,
                        'event_type': event.event_type.value,
                        'timestamp': event.timestamp.isoformat(),
                        'conversation_id': event.conversation_id,
                        'user_id': event.user_id,
                        'details': event.details,
                        'previous_state': event.previous_state,
                        'new_state': event.new_state
                    }, f)
                    f.write('\n')
            except Exception as e:
                logger.error(f"Failed to write audit event to log: {e}")
        
        logger.info(f"Audit event logged: {event.event_type.value} for {event.conversation_id}")
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        timestamp = datetime.now().timestamp()
        return hashlib.sha256(f"{timestamp}".encode()).hexdigest()[:16]
    
    def get_conversation_audit_trail(self, conversation_id: str) -> List[AuditEvent]:
        """Get audit trail for specific conversation."""
        return [
            event for event in self.audit_events
            if event.conversation_id == conversation_id
        ]
    
    def get_user_audit_trail(self, user_id: str) -> List[AuditEvent]:
        """Get audit trail for specific user."""
        return [
            event for event in self.audit_events
            if event.user_id == user_id
        ]
    
    def get_events_by_type(self, event_type: AuditEventType) -> List[AuditEvent]:
        """Get audit events by type."""
        return [
            event for event in self.audit_events
            if event.event_type == event_type
        ]
    
    def get_dataset_changes(self, conversation_id: Optional[str] = None) -> List[DatasetChange]:
        """Get dataset changes, optionally filtered by conversation."""
        if conversation_id:
            return [
                change for change in self.dataset_changes
                if change.conversation_id == conversation_id
            ]
        return self.dataset_changes
    
    def get_remediation_actions(self, conversation_id: Optional[str] = None) -> List[RemediationAction]:
        """Get remediation actions, optionally filtered by conversation."""
        if conversation_id:
            return [
                action for action in self.remediation_actions
                if action.conversation_id == conversation_id
            ]
        return self.remediation_actions
    
    def export_audit_trail(self, output_path: str) -> bool:
        """Export complete audit trail to JSON file."""
        try:
            audit_data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_events": len(self.audit_events),
                "total_dataset_changes": len(self.dataset_changes),
                "total_remediation_actions": len(self.remediation_actions),
                "events": [
                    {
                        "event_id": event.event_id,
                        "event_type": event.event_type.value,
                        "timestamp": event.timestamp.isoformat(),
                        "conversation_id": event.conversation_id,
                        "user_id": event.user_id,
                        "details": event.details,
                        "previous_state": event.previous_state,
                        "new_state": event.new_state
                    }
                    for event in self.audit_events
                ],
                "dataset_changes": [
                    {
                        "change_id": change.change_id,
                        "conversation_id": change.conversation_id,
                        "change_type": change.change_type.value,
                        "timestamp": change.timestamp.isoformat(),
                        "user_id": change.user_id,
                        "details": change.details,
                        "change_reason": change.change_reason,
                        "approval_status": change.approval_status
                    }
                    for change in self.dataset_changes
                ],
                "remediation_actions": [
                    {
                        "action_id": action.action_id,
                        "conversation_id": action.conversation_id,
                        "action_type": action.action_type,
                        "timestamp": action.timestamp.isoformat(),
                        "user_id": action.user_id,
                        "details": action.details,
                        "effectiveness_score": action.effectiveness_score,
                        "follow_up_required": action.follow_up_required
                    }
                    for action in self.remediation_actions
                ]
            }
            
            with open(output_path, 'w') as f:
                json.dump(audit_data, f, indent=2)
            
            logger.info(f"Audit trail exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export audit trail: {e}")
            return False
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """Get audit trail summary statistics."""
        if not self.audit_events:
            return {"message": "No audit events recorded"}
        
        # Count events by type
        event_type_counts = {}
        for event in self.audit_events:
            event_type = event.event_type.value
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
        
        # Count dataset changes by type
        change_type_counts = {}
        for change in self.dataset_changes:
            change_type = change.change_type.value
            change_type_counts[change_type] = change_type_counts.get(change_type, 0) + 1
        
        # Calculate time range
        timestamps = [event.timestamp for event in self.audit_events]
        first_event = min(timestamps) if timestamps else None
        last_event = max(timestamps) if timestamps else None
        
        return {
            "total_events": len(self.audit_events),
            "total_dataset_changes": len(self.dataset_changes),
            "total_remediation_actions": len(self.remediation_actions),
            "event_type_distribution": event_type_counts,
            "change_type_distribution": change_type_counts,
            "users_involved": len(set(event.user_id for event in self.audit_events if event.user_id)),
            "first_event": first_event.isoformat() if first_event else None,
            "last_event": last_event.isoformat() if last_event else None,
            "conversations_affected": len(set(event.conversation_id for event in self.audit_events))
        }
    
    def enable_auditing(self) -> None:
        """Enable audit trail recording."""
        self.enabled = True
        logger.info("Audit trail recording enabled")
    
    def disable_auditing(self) -> None:
        """Disable audit trail recording."""
        self.enabled = False
        logger.info("Audit trail recording disabled")


# Global audit trail instance
_global_audit_trail: Optional[SafetyEthicsAuditTrail] = None


def get_audit_trail() -> SafetyEthicsAuditTrail:
    """Get or create global audit trail instance."""
    global _global_audit_trail
    if _global_audit_trail is None:
        _global_audit_trail = SafetyEthicsAuditTrail()
    return _global_audit_trail


def initialize_audit_trail(audit_log_path: Optional[str] = None) -> SafetyEthicsAuditTrail:
    """Initialize global audit trail with specific configuration."""
    global _global_audit_trail
    _global_audit_trail = SafetyEthicsAuditTrail(audit_log_path)
    return _global_audit_trail


if __name__ == "__main__":
    # Example usage
    audit_trail = get_audit_trail()
    
    # Simulate validation workflow
    conversation_id = "test_conv_001"
    
    # Log validation start
    audit_trail.log_validation_started(conversation_id, user_id="system")
    
    # Log safety issue detection
    safety_issue = {
        "issue_type": "self_harm",
        "risk_level": "high",
        "description": "Detected suicidal ideation with plan",
        "content_snippet": "I have a plan to kill myself tonight"
    }
    audit_trail.log_safety_issue(conversation_id, safety_issue)
    
    # Log intervention requirement
    audit_trail.log_intervention_required(
        conversation_id, 
        reason="Critical safety risk detected",
        urgency_level="immediate"
    )
    
    # Log manual review request
    audit_trail.log_manual_review_request(
        conversation_id,
        reviewer_id="dr_smith",
        reason="Suicidal ideation with detailed plan"
    )
    
    # Log dataset change
    audit_trail.log_dataset_change(
        conversation_id,
        ChangeType.CONTENT_FLAGGED,
        {"reason": "Safety risk flagged for review"},
        previous_content="I have a plan to kill myself tonight",
        change_reason="Suicide risk mitigation"
    )
    
    # Log remediation action
    audit_trail.log_remediation_action(
        conversation_id,
        "safety_intervention",
        {"action": "crisis_intervention_protocol_activated"},
        effectiveness_score=0.95,
        follow_up_required=True
    )
    
    # Log validation completion
    audit_trail.log_validation_completion(
        conversation_id,
        {"final_status": "flagged_for_review", "safety_score": 0.1}
    )
    
    # Get audit summary
    summary = audit_trail.get_audit_summary()
    print("Audit Summary:", json.dumps(summary, indent=2))
    
    # Export audit trail
    audit_trail.export_audit_trail("safety_audit_trail.json")
    print("✅ Safety Ethics Audit Trail - VALIDATION COMPLETE")