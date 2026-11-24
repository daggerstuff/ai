"""
Configuration management for the API server.

This module provides configuration loading from environment variables
with sensible defaults.
"""

import os
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """API server settings."""

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    environment: str = "development"  # development, staging, production
    api_version: str = "1.0.0"
    debug: bool = False

    # CORS configuration
    cors_origins: list[str] = [
        "http://localhost:4321",  # Astro dev server
        "http://localhost:3000",  # Alternative dev port
        "http://localhost:5173",  # Vite dev server
    ]

    # Authentication configuration
    auth_enabled: bool = True
    jwt_secret: str = os.getenv("JWT_SECRET", "change-me-in-production")
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60 * 24  # 24 hours

    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Session storage (must match across all components)
    session_storage_path: str = os.getenv("SESSION_STORAGE_PATH", "ai/sourcing/journal/sessions")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors_origins(cls, value: object) -> object:
        """Accept the comma-separated form the project's .env files use.

        pydantic-settings decodes list fields from the environment as JSON, so a
        plain ``CORS_ORIGINS=http://a,http://b`` raises a JSONDecodeError at
        import time and takes the whole module down with it. Splitting on commas
        first means both spellings work.
        """
        if isinstance(value, str) and not value.strip().startswith("["):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        enable_decoding=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
