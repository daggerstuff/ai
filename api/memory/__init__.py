"""
Memory System Module for Pixelated Empathy.

Provides integrated memory management for:
- User session management
- Conversation history persistence
- Emotional state tracking
- Treatment plan storage
- Dream-cycle memory consolidation
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
    "MemoryContext",
    # Memory Management
    "MemoryManager",
    "MemoryMessage",
    "MemoryType",
    "MessageRole",
    "get_memory_manager",
]
