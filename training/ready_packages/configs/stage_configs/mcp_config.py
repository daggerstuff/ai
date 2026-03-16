"""
Configuration management for MCP Server.

This module provides configuration loading from environment variables
with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai.sourcing.journal.cli.config import load_config


@dataclass
class AuthConfig:
    """Authentication configuration."""

    enabled: bool = True
    api_key_required: bool = True
    jwt_secret: str = os.getenv("MCP_JWT_SECRET", "change-me-in-production")
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60 * 24  # 24 hours
    allowed_api_keys: List[str] = field(default_factory=lambda: [
        key for key in os.getenv("MCP_API_KEYS", "").split(",") if key
    ])


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    enabled: bool = True
    requests_per_minute: int = int(os.getenv("MCP_RATE_LIMIT_PER_MINUTE", "60"))
    requests_per_hour: int = int(os.getenv("MCP_RATE_LIMIT_PER_HOUR", "1000"))
    burst_size: int = int(os.getenv("MCP_RATE_LIMIT_BURST", "10"))


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = os.getenv("MCP_LOG_LEVEL", "INFO")
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = os.getenv("MCP_LOG_FILE")
    enable_audit_logging: bool = True
    audit_log_path: Optional[str] = os.getenv("MCP_AUDIT_LOG_PATH", "logs/mcp_audit.log")


@dataclass
class MCPConfig:
    """MCP Server configuration."""

    # Server configuration
    host: str = os.getenv("MCP_HOST", "0.0.0.0")
    port: int = int(os.getenv("MCP_PORT", "8001"))
    environment: str = os.getenv("MCP_ENVIRONMENT", "development")
    debug: bool = os.getenv("MCP_DEBUG", "false").lower() == "true"

    # Protocol configuration
    protocol_version: str = "2024-11-05"  # MCP protocol version
    server_name: str = "journal-dataset-research-mcp"
    server_version: str = "0.1.0"

    # Authentication
    auth: AuthConfig = field(default_factory=AuthConfig)

    # Rate limiting
    rate_limits: RateLimitConfig = field(default_factory=RateLimitConfig)

    # Logging
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Backend integration
    command_handler_config: Optional[Dict[str, Any]] = None
    session_storage_path: Optional[str] = None

    # Async operations
    async_timeout_seconds: int = int(os.getenv("MCP_ASYNC_TIMEOUT", "3600"))
    progress_update_interval_seconds: int = int(os.getenv("MCP_PROGRESS_UPDATE_INTERVAL", "5"))

    def __post_init__(self) -> None:
        """Post-initialization setup."""
        # Load backend config if not provided
        if self.command_handler_config is None:
            self.command_handler_config = load_config()

        # Set session storage path from backend config if not provided
        if self.session_storage_path is None:
            orchestrator_config = self.command_handler_config.get("orchestrator", {})
            self.session_storage_path = orchestrator_config.get("session_storage_path")

        # Validate configuration
        self._validate()

    def _validate(self) -> None:
        """Validate configuration values."""
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Invalid port number: {self.port}")

        if self.environment not in ("development", "staging", "production"):
            raise ValueError(f"Invalid environment: {self.environment}")

        if self.async_timeout_seconds < 1:
            raise ValueError("async_timeout_seconds must be at least 1")

        if self.progress_update_interval_seconds < 1:
            raise ValueError("progress_update_interval_seconds must be at least 1")

        # Ensure audit log directory exists if audit logging is enabled
        if self.logging.enable_audit_logging and self.logging.audit_log_path:
            log_path = Path(self.logging.audit_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)


def load_mcp_config(config_path: Optional[Path] = None) -> MCPConfig:
    """
    Load MCP configuration from environment variables and optional config file.

    Args:
        config_path: Optional path to configuration file (not yet implemented)

    Returns:
        MCPConfig instance with loaded configuration
    """
    # For now, load from environment variables only
    # Future: support loading from YAML/JSON config file
    return MCPConfig()

