"""
Memory System Module for Pixelated Empathy.

Provides integrated memory management for:
- User session management
- Conversation history persistence
- Emotional state tracking
- Treatment plan storage
- HIPAA-compliant memory encryption
"""

from .memory_manager import (
    MemoryContext,
    MemoryManager,
    MemoryMessage,
    MemoryType,
    MessageRole,
    get_memory_manager,
)

__all__ = [
    # Memory Management
    "MemoryManager",
    "MemoryMessage",
    "MemoryContext",
    "MemoryType",
    "MessageRole",
    "get_memory_manager",
]
