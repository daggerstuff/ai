"""
Claude Subconscious v2 - A Proper Architecture.

Key insight: Don't fight async. Embrace it.

Design principles:
1. Async-first, always
2. No global state
3. Explicit is better than implicit
4. Composition over interception
5. Fail loudly, not silently

This module provides:
- SubconsciousContext: An async context manager for memory injection
- SubconsciousClient: A wrapper that adds memory awareness to any LLM client
- get_subconscious_prompt(): A function to enrich prompts with memories

No monkey patching. No global state. No event loop gymnastics.
"""
from .context import SubconsciousContext, get_subconscious_prompt
from .client import SubconsciousClient
from .config import SubconsciousConfig

__all__ = [
    "SubconsciousContext",
    "SubconsciousClient",
    "SubconsciousConfig",
    "get_subconscious_prompt",
]
