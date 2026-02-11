"""
Audit logging utilities for MCP Server.

This module provides audit logging functionality to track all security-relevant
events including tool executions, resource access, and authentication/authorization events.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from ai.sourcing.journal.mcp.config import LoggingConfig

# Get audit logger
audit_logger = logging.getLogger("mcp.audit")


class AuditEventType(Enum):
    """Types of audit events."""

    # Authentication events
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    AUTH_LOGOUT = "auth_logout"

    # Authorization events
    AUTHORIZATION_GRANTED = "authorization_granted"
    AUTHORIZATION_DENIED = "authorization_denied"

    # Tool execution events
    TOOL_EXECUTION_START = "tool_execution_start"
    TOOL_EXECUTION_SUCCESS = "tool_execution_success"
    TOOL_EXECUTION_FAILURE = "tool_execution_failure"

    # Resource access events
    RESOURCE_ACCESS = "resource_access"
    RESOURCE_ACCESS_DENIED = "resource_access_denied"

    # Rate limiting events
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"

    # Security events
    SECURITY_VIOLATION = "security_violation"
    INPUT_SANITIZATION = "input_sanitization"


@dataclass
class AuditEvent:
    """Represents an audit event."""

    event_type: AuditEventType
    timestamp: str
    user_id: Optional[str] = None
    user_role: Optional[str] = None
    resource: Optional[str] = None  # Tool name, resource URI, etc.
    action: Optional[str] = None  # execute, read, etc.
    success: bool = True
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary."""
        result = asdict(self)
        result["event_type"] = self.event_type.value
        return result

    def to_json(self) -> str:
        """Convert audit event to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class AuditLogger:
    """Audit logging handler."""

    def __init__(self, config: LoggingConfig) -> None:
        """
        Initialize audit logger.

        Args:
            config: Logging configuration
        """
        self.config = config
        self.enabled = config.enable_audit_logging
        self.logger = audit_logger

    def _log_event(self, event: AuditEvent) -> None:
        """
        Log audit event.

        Args:
            event: Audit event to log
        """
        if not self.enabled:
            return

        # Log as JSON for structured logging
        self.logger.info(event.to_json())

    def log_auth_success(
        self,
        user_id: str,
        user_role: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> None:
        """
        Log successful authentication.

        Args:
            user_id: User identifier
            user_role: User role
            metadata: Optional additional metadata
            request_id: Optional request ID
        """
        event = AuditEvent(
            event_type=AuditEventType.AUTH_SUCCESS,
            timestamp=datetime.utcnow().isoformat(),
            user_id=user_id,
            user_role=user_role,
            success=True,
            metadata=metadata,
            request_id=request_id,
        )
        self._log_event(event)

    def log_auth_failure(
        self,
        reason: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> None:
        """
        Log failed authentication.

        Args:
            reason: Failure reason
            user_id: Optional user identifier (if known)
            metadata: Optional additional metadata
            request_id: Optional request ID
        """
        event = AuditEvent(
            event_type=AuditEventType.AUTH_FAILURE,
            timestamp=datetime.utcnow().isoformat(),
            user_id=user_id,
            success=False,
            error_message=reason,
            metadata=metadata,
            request_id=request_id,
        )
        self._log_event(event)

    def log_authorization_granted(
        self,
        user_id: str,
        user_role: str,
        resource: str,
        action: str,
        request_id: Optional[str] = None,
    ) -> None:
        """
        Log granted authorization.

        Args:
            user_id: User identifier
            user_role: User role
            resource: Resource identifier
            action: Action performed
            request_id: Optional request ID
        """
        event = AuditEvent(
            event_type=AuditEventType.AUTHORIZATION_GRANTED,
            timestamp=datetime.utcnow().isoformat(),
            user_id=user_id,
            user_role=user_role,
            resource=resource,
            action=action,
            success=True,
            request_id=request_id,
        )
        self._log_event(event)

    def log_authorization_denied(
        self,
        user_id: str,
        user_role: str,
        resource: str,
        action: str,
        reason: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> None:
        """
        Log denied authorization.

        Args:
            user_id: User identifier
            user_role: User role
            resource: Resource identifier
            action: Action attempted
            reason: Optional denial reason
            request_id: Optional request ID
        """
        event = AuditEvent(
            event_type=AuditEventType.AUTHORIZATION_DENIED,
            timestamp=datetime.utcnow().isoformat(),
            user_id=user_id,
            user_role=user_role,
            resource=resource,
            action=action,
            success=False,
            error_message=reason,
            request_id=request_id,
        )
        self._log_event(event)

    def log_tool_execution_start(
        self,
        user_id: str,
        user_role: str,
        tool_name: str,
        tool_params: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Log tool execution start.

        Args:
            user_id: User identifier
            user_role: User role
            tool_name: Tool name
            tool_params: Optional tool parameters (may be sanitized)
            request_id: Optional request ID
            session_id: Optional session ID
        """
        event = AuditEvent(
            event_type=AuditEventType.TOOL_EXECUTION_START,
            timestamp=datetime.utcnow().isoformat(),
            user_id=user_id,
            user_role=user_role,
            resource=tool_name,
            action="execute",
            success=True,
            metadata={"tool_params": tool_params} if tool_params else None,
            request_id=request_id,
            session_id=session_id,
        )
        self._log_event(event)

    def log_tool_execution_success(
        self,
        user_id: str,
        user_role: str,
        tool_name: str,
        execution_time_ms: Optional[float] = None,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Log successful tool execution.

        Args:
            user_id: User identifier
            user_role: User role
            tool_name: Tool name
            execution_time_ms: Optional execution time in milliseconds
            request_id: Optional request ID
            session_id: Optional session ID
        """
        metadata = {}
        if execution_time_ms is not None:
            metadata["execution_time_ms"] = execution_time_ms

        event = AuditEvent(
            event_type=AuditEventType.TOOL_EXECUTION_SUCCESS,
            timestamp=datetime.utcnow().isoformat(),
            user_id=user_id,
            user_role=user_role,
            resource=tool_name,
            action="execute",
            success=True,
            metadata=metadata if metadata else None,
            request_id=request_id,
            session_id=session_id,
        )
        self._log_event(event)

    def log_tool_execution_failure(
        self,
        user_id: str,
        user_role: str,
        tool_name: str,
        error_message: str,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Log failed tool execution.

        Args:
            user_id: User identifier
            user_role: User role
            tool_name: Tool name
            error_message: Error message
            request_id: Optional request ID
            session_id: Optional session ID
        """
        event = AuditEvent(
            event_type=AuditEventType.TOOL_EXECUTION_FAILURE,
            timestamp=datetime.utcnow().isoformat(),
            user_id=user_id,
            user_role=user_role,
            resource=tool_name,
            action="execute",
            success=False,
            error_message=error_message,
            request_id=request_id,
            session_id=session_id,
        )
        self._log_event(event)

    def log_resource_access(
        self,
        user_id: str,
        user_role: str,
        resource_uri: str,
        success: bool = True,
        error_message: Optional[str] = None,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Log resource access.

        Args:
            user_id: User identifier
            user_role: User role
            resource_uri: Resource URI
            success: Whether access was successful
            error_message: Optional error message
            request_id: Optional request ID
            session_id: Optional session ID
        """
        event_type = (
            AuditEventType.RESOURCE_ACCESS
            if success
            else AuditEventType.RESOURCE_ACCESS_DENIED
        )
        event = AuditEvent(
            event_type=event_type,
            timestamp=datetime.utcnow().isoformat(),
            user_id=user_id,
            user_role=user_role,
            resource=resource_uri,
            action="read",
            success=success,
            error_message=error_message,
            request_id=request_id,
            session_id=session_id,
        )
        self._log_event(event)

    def log_rate_limit_exceeded(
        self,
        identifier: str,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> None:
        """
        Log rate limit exceeded event.

        Args:
            identifier: Rate limit identifier
            user_id: Optional user identifier
            request_id: Optional request ID
        """
        event = AuditEvent(
            event_type=AuditEventType.RATE_LIMIT_EXCEEDED,
            timestamp=datetime.utcnow().isoformat(),
            user_id=user_id,
            success=False,
            error_message=f"Rate limit exceeded for identifier: {identifier}",
            metadata={"identifier": identifier},
            request_id=request_id,
        )
        self._log_event(event)

    def log_security_violation(
        self,
        violation_type: str,
        description: str,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> None:
        """
        Log security violation.

        Args:
            violation_type: Type of violation
            description: Violation description
            user_id: Optional user identifier
            resource: Optional resource identifier
            request_id: Optional request ID
        """
        event = AuditEvent(
            event_type=AuditEventType.SECURITY_VIOLATION,
            timestamp=datetime.utcnow().isoformat(),
            user_id=user_id,
            resource=resource,
            success=False,
            error_message=description,
            metadata={"violation_type": violation_type},
            request_id=request_id,
        )
        self._log_event(event)


def create_audit_logger(config: LoggingConfig) -> AuditLogger:
    """
    Create audit logger instance.

    Args:
        config: Logging configuration

    Returns:
        AuditLogger instance
    """
    return AuditLogger(config)

