"""
Configuration for Claude Subconscious v3.

Immutable, environment-driven, no surprises.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

__all__ = ["SubconsciousConfig", "UserConfig"]

# Constants for magic numbers
DEFAULT_MAX_MEMORIES = 5
DEFAULT_QUERY_TIMEOUT_MS = 5000
DEFAULT_MEMORY_PROVIDER = "local_hindsight"
DEFAULT_BANK_ID = "pixelated"
DEFAULT_MODEL = "z-ai/glm4.7"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MEMORY_SERVICE_BASE_URL = "http://127.0.0.1:5003"
DEFAULT_MEMORY_SERVICE_TIMEOUT_MS = 5000

# Environment variable names
ENV_MODEL = "SUBCONSCIOUS_MODEL"
ENV_API_KEY = "NVIDIA_API_KEY"
ENV_BASE_URL = "SUBCONSCIOUS_BASE_URL"
ENV_ENABLED = "SUBCONSCIOUS_ENABLED"
ENV_BANK_ID = "HINDSIGHT_BANK_ID"
ENV_MEMORY_PROVIDER = "SUBCONSCIOUS_MEMORY_PROVIDER"
ENV_MEMORY_SERVICE_BASE_URL = "SUBCONSCIOUS_MEMORY_BASE_URL"
ENV_MEMORY_SERVICE_ACTOR_ID = "SUBCONSCIOUS_MEMORY_ACTOR_ID"
ENV_MEMORY_SERVICE_ACTOR_SECRET = "SUBCONSCIOUS_MEMORY_ACTOR_SECRET"
ENV_MEMORY_SERVICE_TIMEOUT_MS = "SUBCONSCIOUS_MEMORY_TIMEOUT_MS"


@dataclass(frozen=True)
class SubconsciousConfig:
    """
    Immutable configuration. Safe to share across threads.

    All defaults from environment. Create with:
    config = SubconsciousConfig.from_env()
    """

    # LLM Backend (for enrichment/reflection)
    model: str = DEFAULT_MODEL
    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL

    # Memory Backend
    memory_provider: Literal["local_hindsight", "shared_service", "mock"] = DEFAULT_MEMORY_PROVIDER
    bank_id: str = DEFAULT_BANK_ID
    memory_service_base_url: str = DEFAULT_MEMORY_SERVICE_BASE_URL
    memory_service_actor_id: str = ""
    memory_service_actor_secret: str = ""
    memory_service_timeout_ms: int = DEFAULT_MEMORY_SERVICE_TIMEOUT_MS

    # Enrichment behavior
    max_memories: int = DEFAULT_MAX_MEMORIES  # Max memories to inject
    fail_open: bool = True  # Continue without memory on failure
    enabled: bool = True

    # Reflection behavior
    reflect_on_close: bool = True  # Auto-reflect when state closes

    # Timeouts
    query_timeout_ms: int = (
        DEFAULT_QUERY_TIMEOUT_MS  # 5 second timeout for memory queries
    )

    # Rate limiting
    max_retries: int = 3
    retry_delay_ms: int = 1000  # 1 second between retries

    # Circuit breaker
    circuit_breaker_threshold: int = 5  # Failures before opening circuit
    circuit_breaker_reset_ms: int = 60000  # 1 minute before retry

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

        if self.memory_provider not in ("local_hindsight", "shared_service", "mock"):
            raise ValueError(
                f"memory_provider invalid: {self.memory_provider}"
            )

        if self.memory_service_timeout_ms < 100:
            raise ValueError(
                "memory_service_timeout_ms must be >= 100"
            )

        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")

        if self.memory_provider == "shared_service":
            if not self.memory_service_base_url.strip():
                raise ValueError("memory_service_base_url cannot be empty")
            if not self.memory_service_actor_id.strip():
                raise ValueError(
                    f"{ENV_MEMORY_SERVICE_ACTOR_ID} cannot be empty when using shared_service"
                )
            if not self.memory_service_actor_secret.strip():
                raise ValueError(
                    f"{ENV_MEMORY_SERVICE_ACTOR_SECRET} cannot be empty when using shared_service"
                )

        # Warn if API key missing but enabled
        if self.enabled and not self.api_key and self.reflect_on_close:
            logger.warning(
                f"API key missing ({ENV_API_KEY}), reflection will be skipped"
            )

    @classmethod
    def from_env(cls) -> "SubconsciousConfig":
        """Create from environment variables."""
        return cls(
            model=os.environ.get(ENV_MODEL, DEFAULT_MODEL),
            api_key=os.environ.get(ENV_API_KEY, ""),
            base_url=os.environ.get(ENV_BASE_URL, DEFAULT_BASE_URL),
            memory_provider=os.environ.get(
                ENV_MEMORY_PROVIDER, DEFAULT_MEMORY_PROVIDER
            ),
            bank_id=os.environ.get(ENV_BANK_ID, DEFAULT_BANK_ID),
            memory_service_base_url=os.environ.get(
                ENV_MEMORY_SERVICE_BASE_URL,
                DEFAULT_MEMORY_SERVICE_BASE_URL,
            ),
            memory_service_actor_id=os.environ.get(ENV_MEMORY_SERVICE_ACTOR_ID, ""),
            memory_service_actor_secret=os.environ.get(
                ENV_MEMORY_SERVICE_ACTOR_SECRET, ""
            ),
            memory_service_timeout_ms=int(
                os.environ.get(
                    ENV_MEMORY_SERVICE_TIMEOUT_MS,
                    str(DEFAULT_MEMORY_SERVICE_TIMEOUT_MS),
                )
            ),
            enabled=os.environ.get(ENV_ENABLED, "true").lower() == "true",
        )

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

    def __repr__(self) -> str:
        return f"UserConfig(user_id='{self.user_id}')"
