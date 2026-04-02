"""
Claude Subconscious v3 - Memory injection via contextvars.

No monkey patching. No global state. Async-safe.

Usage:
    # Set context at session start
    from ai.memory.v3 import set_subconscious, SubconsciousConfig

    config = SubconsciousConfig.from_env()
    token = set_subconscious(config, user_id="alice")

    # Any code can check for memory context
    from ai.memory.v3 import get_subconscious

    state = get_subconscious()
    if state:
        enriched = await state.enrich(user_message)

    # Cleanup at session end
    reset_subconscious(token)
    await state.close()  # Triggers reflection
"""

from .client import SubconsciousClient
from .config import SubconsciousConfig
from .context import (
    SubconsciousState,
    get_subconscious,
    reset_subconscious,
    set_subconscious,
    subconscious_context,
)
from .provider import LocalHindsightProvider, MemoryProvider, SharedMemoryServiceProvider

__all__ = [
    # Context API
    "SubconsciousState",
    "subconscious_context",
    "set_subconscious",
    "get_subconscious",
    "reset_subconscious",
    # Configuration
    "SubconsciousConfig",
    # Client API
    "SubconsciousClient",
    # Provider API
    "MemoryProvider",
    "LocalHindsightProvider",
    "SharedMemoryServiceProvider",
]
