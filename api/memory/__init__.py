"""
Memory System Module for Pixelated Empathy.

Provides integrated memory management with Zep Cloud for:
- User session management
- Conversation history persistence
- Emotional state tracking
- Treatment plan storage
- HIPAA-compliant memory encryption
"""

from .mem0_migration import (
    Mem0Migration,
    Mem0ToZepMigrator,
    get_mem0_migrator,
)
from .memory_manager import (
    MemoryContext,
    MemoryManager,
    MemoryMessage,
    MemoryType,
    MessageRole,
    get_memory_manager,
)
from .zep_user_manager import (
    SessionInfo,
    UserProfile,
    UserRole,
    UserStatus,
    ZepUserManager,
    get_zep_manager,
)

__all__ = [
    # Migration Tools
    "Mem0ToZepMigrator",
    "Mem0Migration",
    "get_mem0_migrator",
    # User Management
    "ZepUserManager",
    "UserProfile",
    "UserRole",
    "UserStatus",
    "SessionInfo",
    "get_zep_manager",
    # Memory Management
    "MemoryManager",
    "MemoryMessage",
    "MemoryContext",
    "MemoryType",
    "MessageRole",
    "get_memory_manager",
]
