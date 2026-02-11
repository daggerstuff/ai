"""
Centralized logging system for the Pixelated Empathy AI dataset pipeline.
Provides a configurable, enterprise-grade logger for consistent monitoring.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from ai.pipelines.orchestrator.storage_config import get_dataset_pipeline_output_root

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = logging.INFO

# Resolve log file path under the dataset pipeline output root (outside the package tree)
LOG_FILE = str(get_dataset_pipeline_output_root() / "logs" / "dataset_pipeline.log")
MAX_BYTES = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5

def get_logger(name: str, level: int = LOG_LEVEL) -> logging.Logger:
    """
    Initializes and returns a configured logger.

    Args:
        name: The name of the logger (typically __name__).
        level: The logging level.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers if logger is already configured
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)

    # Console handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    _configure_handler(stdout_handler, level, logger)
    # File handler with rotation
    try:
        _setup_file_handler(level, logger)
    except (OSError, ValueError) as e:
        logger.warning(f"Could not create file handler for logging: {e}")

    logger.propagate = False

    return logger


def _setup_file_handler(level: int, logger: logging.Logger) -> None:
    """Create and configure a rotating file handler for logging."""
    log_path = Path(LOG_FILE)
    log_dir = log_path.parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Ensure the log file path is a file, not a directory
    if log_path.is_dir():
        raise ValueError(f"Log path is a directory, not a file: {log_path}")

    file_handler = RotatingFileHandler(
        str(log_path),
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    _configure_handler(file_handler, level, logger)


def _configure_handler(handler: logging.Handler, level: int, logger: logging.Logger) -> None:
    """Configure a logging handler with level, formatter, and attach to logger."""
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)

# Example usage:
# from logger import get_logger
# logger = get_logger(__name__)
# logger.info("This is an informational message.")
# logger.warning("This is a warning message.")
