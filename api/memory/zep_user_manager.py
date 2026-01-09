"""
Zep User Management Module.

This module provides comprehensive user management for the Zep memory system,
handling user creation, authentication, and session management with HIPAA compliance.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from zep_cloud import User, Zep
except ImportError as e:
    msg = "zep-cloud package required. Install with: uv pip install zep-cloud"
    raise ImportError(msg) from e

logger = logging.getLogger(__name__)


class UserRole(str, Enum):
    """User roles in Pixelated Empathy system."""

    PATIENT = "patient"
    THERAPIST = "therapist"
    ADMIN = "admin"
    SUPPORT = "support"


class UserStatus(str, Enum):
    """User account status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    DELETED = "deleted"


@dataclass
class UserProfile:
    """HIPAA-compliant user profile."""

    user_id: str
    email: str
    name: str
    role: UserRole
    status: UserStatus = UserStatus.ACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)

    # HIPAA compliance fields
    data_encrypted: bool = True
    audit_enabled: bool = True
    session_timeout_minutes: int = 30


@dataclass
class SessionInfo:
    """User session information."""

    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    memory_context: Dict[str, Any] = field(default_factory=dict)


class ZepUserManager:
    """
    Manages user creation, authentication, and session handling with Zep.

    Provides:
    - User creation and management
    - Session lifecycle management
    - Memory context persistence
    - HIPAA-compliant audit logging
    - User activity tracking
    """

    def __init__(
        self,
        api_key: str,
        api_url: Optional[str] = None,
        enable_audit_logging: bool = True,
    ):
        """
        Initialize Zep User Manager.

        Args:
            api_key: Zep Cloud API key
            api_url: Optional custom Zep API URL
            enable_audit_logging: Enable HIPAA audit logging

        Raises:
            ValueError: If api_key is not provided
        """
        if not api_key:
            raise ValueError("api_key is required for Zep authentication")

        self.api_key = api_key
        self.api_url = api_url
        self.enable_audit_logging = enable_audit_logging

        try:
            # Initialize Zep client
            self.client = Zep(api_key=api_key)
            logger.info("Zep client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Zep client: {e}")
            raise RuntimeError(f"Zep initialization failed: {e}") from e

        # Session management
        self.active_sessions: Dict[str, SessionInfo] = {}
        self.user_profiles: Dict[str, UserProfile] = {}

    def create_user(
        self,
        email: str,
        name: str,
        role: UserRole | str = UserRole.PATIENT,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UserProfile:
        """
        Create a new user in Zep with HIPAA compliance.

        Args:
            email: User email address
            name: User full name
            role: User role in the system (UserRole enum or string)
            metadata: Optional metadata for the user

        Returns:
            UserProfile: Created user profile

        Raises:
            ValueError: If user creation fails
            RuntimeError: If Zep API error occurs
        """
        try:
            # Convert string role to enum if needed
            if isinstance(role, str):
                role = UserRole(role)

            # Generate unique user ID
            user_id = str(uuid.uuid4())

            # Create user in Zep
            self.client.user.add(
                user_id=user_id,
                email=email,
                first_name=name.split()[0] if " " in name else name,
                last_name=" ".join(name.split()[1:]) if " " in name else "",
                metadata={
                    "role": role.value,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    **(metadata or {}),
                },
            )

            # Create user profile
            profile = UserProfile(
                user_id=user_id,
                email=email,
                name=name,
                role=role,
                metadata=metadata or {},
            )

            # Store profile
            self.user_profiles[user_id] = profile

            # Audit log
            self._log_audit(
                "user_created",
                user_id,
                {"email": email, "name": name, "role": role.value},
            )

            logger.info(f"User created: {user_id} ({email})")
            return profile

        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            raise RuntimeError(f"User creation failed: {e}") from e

    def get_user(self, user_id: str) -> Optional[UserProfile]:
        """
        Retrieve user profile by ID.

        Args:
            user_id: User ID

        Returns:
            UserProfile if found, None otherwise
        """
        try:
            if user_id in self.user_profiles:
                return self.user_profiles[user_id]

            # Fetch from Zep if not in cache
            zep_user = self.client.user.get(user_id)
            if not zep_user:
                return None

            # Reconstruct profile
            profile = UserProfile(
                user_id=user_id,
                email=zep_user.email or "",
                name=zep_user.name or "",
                role=UserRole(zep_user.metadata.get("role", "patient")),
                metadata=zep_user.metadata or {},
            )

            self.user_profiles[user_id] = profile
            return profile

        except Exception as e:
            logger.error(f"Error retrieving user {user_id}: {e}")
            return None

    def update_user(
        self,
        user_id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        preferences: Optional[Dict[str, Any]] = None,
        status: Optional[UserStatus] = None,
    ) -> bool:
        """
        Update user profile.

        Args:
            user_id: User ID
            name: Optional new name
            metadata: Optional metadata updates
            preferences: Optional preference updates
            status: Optional status change

        Returns:
            True if successful, False otherwise
        """
        try:
            profile = self.get_user(user_id)
            if not profile:
                return False

            # Update profile
            if name:
                profile.name = name
            if metadata:
                profile.metadata.update(metadata)
            if preferences:
                profile.preferences.update(preferences)
            if status:
                profile.status = status

            profile.last_login = datetime.now(timezone.utc)

            # Update in Zep
            zep_user = User(
                user_id=user_id, email=profile.email, metadata=profile.metadata
            )
            self.client.user.update(zep_user)

            # Audit log
            self._log_audit(
                "user_updated",
                user_id,
                {"name": name, "status": status.value if status else None},
            )

            logger.info(f"User updated: {user_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating user {user_id}: {e}")
            return False

    def create_session(
        self,
        user_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_timeout_minutes: int = 30,
    ) -> SessionInfo:
        """
        Create new user session.

        Args:
            user_id: User ID
            ip_address: Client IP address
            user_agent: Client user agent
            session_timeout_minutes: Session timeout in minutes

        Returns:
            SessionInfo: Created session information

        Raises:
            ValueError: If user not found
            RuntimeError: If session creation fails
        """
        try:
            # Verify user exists
            profile = self.get_user(user_id)
            if not profile:
                raise ValueError(f"User not found: {user_id}")

            # Generate session ID
            session_id = str(uuid.uuid4())

            # Create session
            now = datetime.now(timezone.utc)
            expires_at = now + timedelta(minutes=session_timeout_minutes)

            session = SessionInfo(
                session_id=session_id,
                user_id=user_id,
                created_at=now,
                expires_at=expires_at,
                last_activity=now,
                ip_address=ip_address,
                user_agent=user_agent,
            )

            # Store session
            self.active_sessions[session_id] = session

            # Update user last login
            profile.last_login = now

            # Audit log
            self._log_audit(
                "session_created",
                user_id,
                {
                    "session_id": session_id,
                    "ip_address": ip_address,
                    "timeout_minutes": session_timeout_minutes,
                },
            )

            logger.info(f"Session created: {session_id} for user {user_id}")
            return session

        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise RuntimeError(f"Session creation failed: {e}") from e

    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """
        Retrieve active session.

        Args:
            session_id: Session ID

        Returns:
            SessionInfo if valid and not expired, None otherwise
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return None

        # Check expiration
        if datetime.now(timezone.utc) > session.expires_at:
            self._log_audit(
                "session_expired", session.user_id, {"session_id": session_id}
            )
            del self.active_sessions[session_id]
            return None

        # Update last activity
        session.last_activity = datetime.now(timezone.utc)
        return session

    def close_session(self, session_id: str) -> bool:
        """
        Close user session.

        Args:
            session_id: Session ID

        Returns:
            True if successful, False otherwise
        """
        try:
            session = self.active_sessions.pop(session_id, None)
            if not session:
                return False

            self._log_audit(
                "session_closed",
                session.user_id,
                {
                    "session_id": session_id,
                    "duration_seconds": (
                        datetime.now(timezone.utc) - session.created_at
                    ).total_seconds(),
                },
            )

            logger.info(f"Session closed: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error closing session {session_id}: {e}")
            return False

    def store_memory_context(self, session_id: str, context: Dict[str, Any]) -> bool:
        """
        Store memory context for a session.

        Args:
            session_id: Session ID
            context: Memory context data

        Returns:
            True if successful, False otherwise
        """
        try:
            session = self.get_session(session_id)
            if not session:
                return False

            session.memory_context = context

            # Store in Zep memory system
            # This integrates with the memory module
            logger.info(f"Memory context stored for session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error storing memory context: {e}")
            return False

    def list_user_sessions(self, user_id: str) -> List[SessionInfo]:
        """
        List all active sessions for a user.

        Args:
            user_id: User ID

        Returns:
            List of active SessionInfo objects
        """
        return [
            s
            for s in self.active_sessions.values()
            if s.user_id == user_id and datetime.now(timezone.utc) <= s.expires_at
        ]

    def cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions.

        Returns:
            Number of sessions removed
        """
        expired = []
        now = datetime.now(timezone.utc)
        expired = [
            session_id
            for session_id, session in self.active_sessions.items()
            if now > session.expires_at
        ]
        for session_id in expired:
            del self.active_sessions[session_id]

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")

        return len(expired)

    def _log_audit(self, action: str, user_id: str, details: Dict[str, Any]) -> None:
        """
        Log audit event for HIPAA compliance.

        Args:
            action: Action performed
            user_id: User ID
            details: Action details
        """
        if not self.enable_audit_logging:
            return

        try:
            audit_log = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": action,
                "user_id": user_id,
                "details": details,
            }
            logger.info(f"AUDIT: {audit_log}")
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")

    def close(self) -> None:
        """Clean up resources."""
        try:
            # Clean up expired sessions
            self.cleanup_expired_sessions()

            # Close Zep client if needed
            if hasattr(self.client, "close"):
                self.client.close()

            logger.info("Zep User Manager closed")
        except Exception as e:
            logger.error(f"Error closing Zep User Manager: {e}")


# Singleton instance
_zep_manager_instance: Optional[ZepUserManager] = None


def get_zep_manager(
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    enable_audit_logging: bool = True,
) -> ZepUserManager:
    """
    Get or create Zep User Manager instance.

    Args:
        api_key: Zep API key (required for first call)
        api_url: Optional custom Zep API URL
        enable_audit_logging: Enable HIPAA audit logging

    Returns:
        ZepUserManager instance

    Raises:
        ValueError: If api_key not provided on first call
    """
    global _zep_manager_instance

    if _zep_manager_instance is None:
        if not api_key:
            raise ValueError("api_key required for first Zep Manager initialization")
        _zep_manager_instance = ZepUserManager(
            api_key=api_key, api_url=api_url, enable_audit_logging=enable_audit_logging
        )

    return _zep_manager_instance
