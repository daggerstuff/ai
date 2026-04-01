"""
Configuration for Claude Subconscious v3.

Immutable, environment-driven, no surprises.
"""
from dataclasses import dataclass, field
import os
from typing import Literal


@dataclass(frozen=True)
class SubconsciousConfig:
    """
    Immutable configuration. Safe to share across threads.

    All defaults from environment. Create with:
        config = SubconsciousConfig.from_env()
    """
    # LLM Backend (for enrichment/reflection)
    model: str = field(default_factory=lambda: os.environ.get("SUBCONSCIOUS_MODEL", "z-ai/glm4.7"))
    api_key: str = field(default_factory=lambda: os.environ.get("NVIDIA_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.environ.get("SUBCONSCIOUS_BASE_URL", "https://integrate.api.nvidia.com/v1"))

    # Memory Backend
    memory_provider: Literal["local_hindsight", "mock"] = "local_hindsight"
    bank_id: str = field(default_factory=lambda: os.environ.get("HINDSIGHT_BANK_ID", "pixelated"))

    # Enrichment behavior
    max_memories: int = 5  # Max memories to inject
    fail_open: bool = True  # Continue without memory on failure
    enabled: bool = field(default_factory=lambda: os.environ.get("SUBCONSCIOUS_ENABLED", "true").lower() == "true")

    # Reflection behavior
    reflect_on_close: bool = True  # Auto-reflect when state closes

    # Timeouts
    query_timeout_ms: int = 5000  # 5 second timeout for memory queries

    @classmethod
    def from_env(cls) -> "SubconsciousConfig":
        """Create from environment variables."""
        return cls()

    def with_user(self, user_id: str) -> "UserConfig":
        """Bind config to a specific user."""
        return UserConfig(base=self, user_id=user_id)


@dataclass(frozen=True)
class UserConfig:
    """Config bound to a user. Used internally."""
    base: SubconsciousConfig
    user_id: str
