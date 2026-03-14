"""
Configuration management for the API server.

This module provides configuration loading from environment variables
with sensible defaults.
"""

import os
from functools import lru_cache
from typing import List

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """API server settings."""

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    environment: str = "development"  # development, staging, production
    api_version: str = "1.0.0"
    debug: bool = False

    # CORS configuration - store raw string from env, will be processed
    _cors_origins_raw: str = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:4321,http://localhost:3000,http://localhost:5173",
    )

    @property
    def cors_origins(self) -> List[str]:
        """Get CORS origins as a list, parsing from raw string."""
        # If it looks like a JSON array, try to parse it
        if self._cors_origins_raw.startswith("[") and self._cors_origins_raw.endswith(
            "]"
        ):
            import json

            try:
                return json.loads(self._cors_origins_raw)
            except json.JSONDecodeError:
                pass
        # Fallback: comma-separated string
        return [
            origin.strip()
            for origin in self._cors_origins_raw.split(",")
            if origin.strip()
        ]

    # Authentication configuration
    auth_enabled: bool = True
    jwt_secret: str = os.getenv("JWT_SECRET", "change-me-in-production")
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60 * 24  # 24 hours

    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        """Ensure critical settings are provided in production."""
        if self.environment == "production":
            if self.jwt_secret == "change-me-in-production":
                 raise ValueError("JWT_SECRET must be set in production via environment variable")
        return self

    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Session storage (must match across all components)
    session_storage_path: str = os.getenv(
        "SESSION_STORAGE_PATH", "ai/sourcing/journal/sessions"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        validate_assignment=True,
        extra="ignore",  # Allow extra env vars from parent .env
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
