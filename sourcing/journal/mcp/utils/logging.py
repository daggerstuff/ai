"""
Logging configuration for MCP Server.

This module provides logging setup and configuration utilities.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from ai.sourcing.journal.mcp.config import LoggingConfig


def setup_logging(config: LoggingConfig) -> None:
    """
    Set up logging configuration for MCP server.

    Args:
        config: Logging configuration
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level.upper(), logging.INFO))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(config.format)

    # Console handler (always add)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if log file specified)
    if config.file:
        log_path = Path(config.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            config.file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
        )
        file_handler.setLevel(getattr(logging, config.level.upper(), logging.INFO))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Set up MCP-specific logger
    mcp_logger = logging.getLogger("ai.sourcing.journal.mcp")
    mcp_logger.setLevel(getattr(logging, config.level.upper(), logging.INFO))

    # Set up audit logger (if enabled)
    if config.enable_audit_logging and config.audit_log_path:
        audit_log_path = Path(config.audit_log_path)
        audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        audit_handler = logging.handlers.RotatingFileHandler(
            config.audit_log_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=10,  # Keep more audit logs
        )
        audit_handler.setLevel(logging.INFO)
        audit_formatter = logging.Formatter(
            "%(asctime)s - AUDIT - %(levelname)s - %(message)s"
        )
        audit_handler.setFormatter(audit_formatter)

        audit_logger = logging.getLogger("mcp.audit")
        audit_logger.addHandler(audit_handler)
        audit_logger.setLevel(logging.INFO)
        audit_logger.propagate = False  # Don't propagate to root logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get logger instance.

    Args:
        name: Logger name (defaults to MCP logger)

    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"ai.sourcing.journal.mcp.{name}")
    return logging.getLogger("ai.sourcing.journal.mcp")

