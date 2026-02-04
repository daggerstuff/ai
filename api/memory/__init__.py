"""
Memory System Module for Pixelated Empathy.

Provides integrated memory management for:
- User session management
- Conversation history persistence
- Emotional state tracking
- Treatment plan storage
"""

# Placeholder for future Mem0 integration in this layer
# Currently moving towards direct GeminiMem0Manager usage

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
