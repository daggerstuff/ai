"""
Mem0 to Zep Migration Utilities.

Provides tools for migrating from mem0 to Zep Cloud:
- Memory data migration
- User session data mapping
- Conversation history conversion
- Compatibility layer during transition
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from zep_cloud import Message, User, Zep

logger = logging.getLogger(__name__)


@dataclass
class Mem0Migration:
    """Migration context for mem0 to Zep transition."""

    source_system: str = "mem0"
    target_system: str = "zep"
    migrated_users: int = 0
    migrated_memories: int = 0
    migration_timestamp: datetime = None
    errors: List[str] = None

    def __post_init__(self):
        """Initialize migration context."""
        if self.migration_timestamp is None:
            self.migration_timestamp = datetime.now(timezone.utc)
        if self.errors is None:
            self.errors = []


class Mem0ToZepMigrator:
    """
    Handles migration from mem0 to Zep Cloud.

    Provides methods for:
    - User data migration with role mapping
    - Memory history conversion
    - Session state preservation
    - Validation and error handling
    """

    def __init__(self, zep_client: Zep):
        """
        Initialize migrator.

        Args:
            zep_client: Initialized Zep client

        Raises:
            ValueError: If zep_client not provided
        """
        if not zep_client:
            raise ValueError("zep_client is required")

        self.client = zep_client
        self.migration_state = Mem0Migration()

    def _record_error(self, message: str) -> None:
        """Store an error and log it for migration visibility."""
        self.migration_state.errors.append(message)
        logger.error(message)

    def migrate_user(
        self,
        mem0_user_data: Dict[str, Any],
        role: str = "patient",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Migrate user from mem0 to Zep.

        Args:
            mem0_user_data: User data from mem0 (dict with id, name, email, etc.)
            role: User role in Zep (patient, therapist, admin, support)
            metadata: Additional metadata to preserve

        Returns:
            User ID in Zep or None if failed

        Example:
            mem0_user = {
                "id": "mem0-user-123",
                "name": "Jane Doe",
                "email": "jane@example.com"
            }
            zep_user_id = migrator.migrate_user(mem0_user, role="patient")
        """
        try:
            # Create Zep user from mem0 data
            migration_metadata = {
                "migration_source": "mem0",
                "source_user_id": mem0_user_data.get("id"),
                "migration_timestamp": datetime.now(timezone.utc).isoformat(),
                **(metadata or {}),
            }

            zep_user = User(
                name=mem0_user_data.get("name", "Migrated User"),
                email=mem0_user_data.get("email"),
                metadata=migration_metadata,
            )

            # Add user to Zep
            response = self.client.user.add(zep_user)
            user_id = response.id if response else None

            if user_id:
                self.migration_state.migrated_users += 1
                logger.info(f"Migrated user {mem0_user_data.get('id')} -> {user_id}")
                return user_id
            else:
                error_msg = f"Failed to create Zep user for {mem0_user_data.get('id')}"
                self._record_error(error_msg)
                return None

        except Exception as e:
            error_msg = f"Error migrating user {mem0_user_data.get('id')}: {e}"
            self._record_error(error_msg)
            return None

    def migrate_conversation_history(
        self,
        session_id: str,
        mem0_messages: List[Dict[str, Any]],
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Migrate conversation history from mem0 to Zep.

        Args:
            session_id: Zep session ID (or create new one)
            mem0_messages: List of messages from mem0
                (format: [{"role": "user/ai", "content": "..."}, ...])
            user_id: Optional user ID for attribution

        Returns:
            True if successful, False otherwise

        Example:
            mem0_messages = [
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well..."}
            ]
            success = migrator.migrate_conversation_history(
                "zep-session-123",
                mem0_messages
            )
        """
        try:
            return self._extracted_from_migrate_conversation_history_30(
                mem0_messages, session_id
            )
        except Exception as e:
            return self._extracted_from_migrate_memory_snapshot_34(
                "Error migrating conversation for session ", session_id, ": ", e
            )

    # TODO Rename this here and in `migrate_conversation_history`
    def _extracted_from_migrate_conversation_history_30(
        self, mem0_messages, session_id
    ):
        if not mem0_messages:
            logger.info(f"No messages to migrate for session {session_id}")
            return True

        # Convert mem0 messages to Zep format
        zep_messages = []
        for msg in mem0_messages:
            role = self._map_message_role(msg.get("role", "user"))

            # Preserve original metadata
            metadata = {
                "migration_source": "mem0",
                "original_role": msg.get("role"),
            } | msg.get("metadata", {})

            zep_message = Message(
                content=msg.get("content", ""),
                role=role,
                metadata=metadata,
            )
            zep_messages.append(zep_message)

        # Add messages to Zep in batches (Zep may have limits)
        batch_size = 50
        for i in range(0, len(zep_messages), batch_size):
            batch = zep_messages[i : i + batch_size]

            try:
                self.client.thread.add_messages(session_id, messages=batch)
            except Exception as e:
                logger.warning(f"Error adding messages to session {session_id}: {e}")

        self.migration_state.migrated_memories += len(zep_messages)
        logger.info(f"Migrated {len(zep_messages)} messages to session {session_id}")
        return True

    def migrate_memory_snapshot(
        self,
        session_id: str,
        memory_type: str,
        content: Dict[str, Any] or str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Migrate a memory snapshot (e.g., summary, insights, traits).

        Args:
            session_id: Zep session ID
            memory_type: Type of memory (summary, insights, traits, etc.)
            content: Memory content (string or dict)
            metadata: Additional metadata

        Returns:
            True if successful, False otherwise

        Example:
            insights = {
                "key_topics": ["anxiety", "relationships"],
                "emotional_patterns": ["stress response", "avoidance"],
                "coping_strategies": ["breathing exercises", "journaling"]
            }
            migrator.migrate_memory_snapshot(
                "session-123",
                "insights",
                insights
            )
        """
        try:
            # Convert content to JSON if dict
            if isinstance(content, dict):
                content_str = json.dumps(content)
            else:
                content_str = str(content)

            migration_metadata = {
                "migration_source": "mem0",
                "memory_type": memory_type,
            } | (metadata or {})

            message = Message(
                content=content_str,
                role="system",
                metadata=migration_metadata,
            )

            self.client.thread.add_messages(session_id, messages=[message])

            self.migration_state.migrated_memories += 1
            logger.info(f"Migrated {memory_type} snapshot to session {session_id}")
            return True

        except Exception as e:
            return self._extracted_from_migrate_memory_snapshot_34(
                "Error migrating ", memory_type, " snapshot: ", e
            )

    # TODO Rename this helper in `migrate_conversation_history`
    # and `migrate_memory_snapshot`
    def _extracted_from_migrate_memory_snapshot_34(self, arg0, arg1, arg2, e):
        error_msg = f"{arg0}{arg1}{arg2}{e}"
        self._record_error(error_msg)
        return False

    def validate_migration(
        self, zep_user_id: str, zep_session_id: str
    ) -> Dict[str, Any]:
        """
        Validate migration by comparing source and target data.

        Args:
            zep_user_id: Migrated user ID in Zep
            zep_session_id: Migrated session ID in Zep

        Returns:
            Validation report with findings

        Example:
            report = migrator.validate_migration("zep-user-123", "zep-session-456")
            if report["valid"]:
                print("Migration successful")
            else:
                print(f"Issues: {report['issues']}")
        """
        try:
            report = {
                "valid": True,
                "user_id": zep_user_id,
                "session_id": zep_session_id,
                "issues": [],
                "memory_count": 0,
                "first_message": None,
                "last_message": None,
            }

            # Check user exists
            try:
                user = self.client.user.get(zep_user_id)
                if not user:
                    report["issues"].append(f"User {zep_user_id} not found")
                    report["valid"] = False
            except Exception as e:
                report["issues"].append(f"Error retrieving user: {e}")
                report["valid"] = False

            # Check session memory
            try:
                thread = self.client.thread.get(zep_session_id)
                if thread and thread.messages:
                    report["memory_count"] = len(thread.messages)
                    report["first_message"] = (
                        thread.messages[0].created_at.isoformat()
                        if thread.messages
                        else None
                    )
                    report["last_message"] = (
                        thread.messages[-1].created_at.isoformat()
                        if thread.messages
                        else None
                    )
            except Exception as e:
                report["issues"].append(f"Error retrieving memory: {e}")

            return report

        except Exception as e:
            self._record_error(f"Error validating migration: {e}")
            return {
                "valid": False,
                "issues": [str(e)],
            }

    def get_migration_report(self) -> Dict[str, Any]:
        """
        Get comprehensive migration report.

        Returns:
            Migration statistics and status

        Example:
            report = migrator.get_migration_report()
            print(f"Migrated {report['users']} users and {report['memories']} memories")
        """
        return {
            "source_system": self.migration_state.source_system,
            "target_system": self.migration_state.target_system,
            "users_migrated": self.migration_state.migrated_users,
            "memories_migrated": self.migration_state.migrated_memories,
            "timestamp": self.migration_state.migration_timestamp.isoformat(),
            "errors": self.migration_state.errors,
            "success": len(self.migration_state.errors) == 0,
        }

    @staticmethod
    def _map_message_role(mem0_role: str) -> str:
        """
        Map mem0 message role to Zep role.

        Args:
            mem0_role: Role from mem0 (user, ai, assistant, etc.)

        Returns:
            Zep role (user, assistant, system)
        """
        role_map = {
            "user": "user",
            "assistant": "assistant",
            "ai": "assistant",
            "system": "system",
            "therapist": "user",  # Therapist messages as user input
            "ai_therapist": "assistant",  # AI therapist as assistant
        }
        return role_map.get(mem0_role.lower(), "user")

    @staticmethod
    def create_migration_metadata(
        mem0_resource_id: str, resource_type: str
    ) -> Dict[str, Any]:
        """
        Create metadata for migration tracking.

        Args:
            mem0_resource_id: Original ID from mem0
            resource_type: Type of resource (user, memory, session, etc.)

        Returns:
            Metadata dict for Zep resources

        Example:
            metadata = Mem0ToZepMigrator.create_migration_metadata(
                "mem0-user-123",
                "user"
            )
        """
        return {
            "migration_source": "mem0",
            "source_resource_id": mem0_resource_id,
            "resource_type": resource_type,
            "migration_timestamp": datetime.now(timezone.utc).isoformat(),
            "migration_tool": "Pixelated Empathy Migration Utility",
        }


# Singleton instance
_migrator_instance: Optional[Mem0ToZepMigrator] = None


def get_mem0_migrator(zep_client: Optional[Zep] = None) -> Mem0ToZepMigrator:
    """
    Get or create Mem0ToZepMigrator instance.

    Args:
        zep_client: Zep client (required for first call)

    Returns:
        Mem0ToZepMigrator instance

    Raises:
        ValueError: If zep_client not provided on first call
    """
    global _migrator_instance

    if _migrator_instance is None:
        if not zep_client:
            raise ValueError("zep_client required for first call")
        _migrator_instance = Mem0ToZepMigrator(zep_client)

    return _migrator_instance
