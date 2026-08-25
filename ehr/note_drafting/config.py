"""Configuration for the AI Note Drafting microservice.

All settings are loaded from environment variables with sensible defaults.
No PHI is stored in configuration objects.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class NoteDraftingSettings(BaseSettings):
    """Settings for the note drafting service.

    Environment variables use the ``NOTE_DRAFTING_`` prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="NOTE_DRAFTING_",
        env_file=".env",
        extra="ignore",
    )

    # NIM endpoint configuration
    nim_url: str = Field(
        default="",
        description="Base URL of the NIM inference endpoint (e.g. https://nim.example.com/v1/chat/completions).",
    )
    nim_api_key: str = Field(
        default="",
        description="API key for NIM endpoint authentication.",
    )
    nim_model: str = Field(
        default="meta/llama-3.3-70b-instruct",
        description="Model identifier to use at the NIM endpoint.",
    )
    nim_timeout_seconds: float = Field(
        default=30.0,
        description="Request timeout in seconds for NIM calls.",
    )
    nim_max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for transient NIM failures.",
    )
    nim_retry_base_delay: float = Field(
        default=1.0,
        description="Base delay in seconds for exponential backoff between retries.",
    )

    # BAA compliance gate
    baa_confirmed: bool = Field(
        default=False,
        description="Must be True to accept drafting requests. Rejects with 403 if not set.",
    )

    # Service configuration
    service_host: str = Field(default="0.0.0.0", description="Host to bind the service.")
    service_port: int = Field(default=8100, description="Port to bind the service.")

    # Transcript constraints
    max_transcript_length: int = Field(
        default=50000,
        description="Maximum allowed transcript character length.",
    )
    min_transcript_length: int = Field(
        default=10,
        description="Minimum required transcript character length.",
    )

    @property
    def is_configured(self) -> bool:
        """Return True if the NIM endpoint is configured."""
        return bool(self.nim_url and self.nim_api_key)


@lru_cache(maxsize=1)
def get_settings() -> NoteDraftingSettings:
    """Return cached settings singleton."""
    return NoteDraftingSettings()
