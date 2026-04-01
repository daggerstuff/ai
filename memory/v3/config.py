"""
Configuration for Claude Subconscious v3.

Immutable, environment-driven, no surprises.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

# Constants for magic numbers
DEFAULT_MAX_MEMORIES = 5
DEFAULT_QUERY_TIMEOUT_MS = 5000
DEFAULT_MEMORY_PROVIDER = "local_hindsight"
DEFAULT_BANK_ID = "pixelated"
DEFAULT_MODEL = "z-ai/glm4.7"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Environment variable names
ENV_MODEL = "SUBCONSCIOUS_MODEL"
ENV_API_KEY = "NVIDIA_API_KEY"
ENV_BASE_URL = "SUBCONSCIOUS_BASE_URL"
ENV_ENABLED = "SUBCONSCIOUS_ENABLED"
ENV_BANK_ID = "HINDSIGHT_BANK_ID"


@dataclass(frozen=True)
class SubconsciousConfig:
    """
    Immutable configuration. Safe to share across threads.

    All defaults from environment. Create with:
    config = SubconsciousConfig.from_env()
    """

    # LLM Backend (for enrichment/reflection)
    model: str = field(default_factory=lambda: os.environ.get(ENV_MODEL, DEFAULT_MODEL))
    api_key: str = field(default_factory=lambda: os.environ.get(ENV_API_KEY, ""))
    base_url: str = field(
        default_factory=lambda: os.environ.get(ENV_BASE_URL, DEFAULT_BASE_URL)
    )

    # Memory Backend
    memory_provider: Literal["local_hindsight", "mock"] = DEFAULT_MEMORY_PROVIDER
    bank_id: str = field(
        default_factory=lambda: os.environ.get(ENV_BANK_ID, DEFAULT_BANK_ID)
    )

    # Enrichment behavior
    max_memories: int = DEFAULT_MAX_MEMORIES  # Max memories to inject
    fail_open: bool = True  # Continue without memory on failure
    enabled: bool = field(
        default_factory=lambda: os.environ.get(ENV_ENABLED, "true").lower() == "true"
    )

    # Reflection behavior
    reflect_on_close: bool = True  # Auto-reflect when state closes

    # Timeouts
    query_timeout_ms: int = (
        DEFAULT_QUERY_TIMEOUT_MS  # 5 second timeout for memory queries
    )

    # Rate limiting
    max_retries: int = 3
    retry_delay_ms: int = 1000  # 1 second between retries

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate configuration values. Raises ValueError if invalid."""
        if self.max_memories < 1:
            raise ValueError(f"max_memories must be >= 1, got {self.max_memories}")

        if self.query_timeout_ms < 100:
            raise ValueError(
                f"query_timeout_ms must be >= 100, got {self.query_timeout_ms}"
            )

        if self.memory_provider not in ("local_hindsight", "mock"):
            raise ValueError(
                f"memory_provider invalid: {self.memory_provider}"
            )

        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")

        # Warn if API key missing but enabled
        if self.enabled and not self.api_key and self.reflect_on_close:
            logger.warning(
                f"API key missing ({ENV_API_KEY}), skipping reflection"
            )

    @classmethod
    def from_env(cls) -> "SubconsciousConfig":
        """Create from environment variables."""
        return cls()

    def with_user(self, user_id: str) -> "UserConfig":
        """
        Bind config to a specific user.

        Args:
            user_id: Unique identifier for the user

        Returns:
            UserConfig bound to this user

        Raises:
            ValueError: If user_id is empty or invalid
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id cannot be empty")
        return UserConfig(base=self, user_id=user_id.strip())


@dataclass(frozen=True)
class UserConfig:
    """Config bound to a user. Used internally."""

    base: "SubconsciousConfig"
    user_id: str

    def __post_init__(self):
        """Validate user config after initialization."""
        if not self.user_id:
            raise ValueError("user_id cannot be empty")
