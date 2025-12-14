"""
Centralized logging system for the Pixelated Empathy AI dataset pipeline.
Provides a configurable, enterprise-grade logger for consistent monitoring.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = logging.INFO
LOG_FILE = "logs/dataset_pipeline.log"
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
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(stdout_handler)

    # File handler with rotation
    try:
        log_dir = Path(LOG_FILE).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(file_handler)
    except (OSError, PermissionError) as e:
        logger.warning(f"Could not create file handler for logging: {e}")

    logger.propagate = False

    return logger

# Example usage:
# from logger import get_logger
# logger = get_logger(__name__)
# logger.info("This is an informational message.")
# logger.warning("This is a warning message.")
