"""
Configuration for Claude Subconscious v2.

Key insight: Configuration should be immutable and explicit.
No global variables, no module-level state.
"""
from dataclasses import dataclass, field
from enum import Enum
import os
from typing import Callable, Optional, Awaitable


class ReflectionTrigger(str, Enum):
    """What triggers reflection."""
    MANUAL = "manual"
    STEP_COUNT = "step_count"
    SESSION_END = "session_end"


@dataclass(frozen=True)  # Immutable!
class SubconsciousConfig:
    """
    Immutable configuration for subconscious memory injection.

    All defaults are read from environment at construction time.
    This config object is safe to share across threads.
    """
    # LLM Backend
    model: str = field(default_factory=lambda: os.environ.get("SUBCONSCIOUS_MODEL", "z-ai/glm4.7"))
    api_key: Optional[str] = field(default_factory=lambda: os.environ.get("NVIDIA_API_KEY"))
    base_url: str = field(default_factory=lambda: os.environ.get("SUBCONSCIOUS_BASE_URL", "https://integrate.api.nvidia.com/v1"))

    # Memory Backend
    memory_provider: str = field(default_factory=lambda: os.environ.get("MEMORY_PROVIDER", "local_hindsight"))
    bank_id: str = field(default_factory=lambda: os.environ.get("HINDSIGHT_BANK_ID", "pixelated"))

    # Reflection Behavior
    trigger: ReflectionTrigger = ReflectionTrigger.STEP_COUNT
    step_threshold: int = 10
    include_crisis_context: bool = True
    auto_consolidate: bool = False
    max_memories_to_retrieve: int = 50

    # Timeout & Retries
    query_timeout_seconds: float = 30.0
    max_retries: int = 3

    # Feature Flags
    enabled: bool = field(default_factory=lambda: os.environ.get("SUBCONSCIOUS_ENABLED", "true").lower() == "true")
    fail_open: bool = True  # If True, continue without memory on failure

    def with_model(self, model: str) -> "SubconsciousConfig":
        """Create a new config with a different model."""
        return SubconsciousConfig(
            model=model,
            api_key=self.api_key,
            base_url=self.base_url,
            memory_provider=self.memory_provider,
            bank_id=self.bank_id,
            trigger=self.trigger,
            step_threshold=self.step_threshold,
            include_crisis_context=self.include_crisis_context,
            auto_consolidate=self.auto_consolidate,
            max_memories_to_retrieve=self.max_memories_to_retrieve,
            query_timeout_seconds=self.query_timeout_seconds,
            max_retries=self.max_retries,
            enabled=self.enabled,
            fail_open=self.fail_open,
        )

    @classmethod
    def from_env(cls) -> "SubconsciousConfig":
        """
        Create config entirely from environment variables.

        This is the recommended way to create config in production.
        """
        return cls()


# Type alias for LLM callback
# The callback takes a prompt and returns the LLM's response
LLMCallback = Callable[[str], Awaitable[str]]
