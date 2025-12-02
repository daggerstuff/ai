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

# Resolve log file path relative to the dataset_pipeline directory
_DATASET_PIPELINE_DIR = Path(__file__).parent.parent.resolve()
LOG_FILE = str(_DATASET_PIPELINE_DIR / "logs" / "dataset_pipeline.log")
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
        log_path = Path(LOG_FILE)
        log_dir = log_path.parent
        log_dir.mkdir(parents=True, exist_ok=True)

        # Ensure the log file path is a file, not a directory
        if log_path.exists() and log_path.is_dir():
            raise ValueError(f"Log path is a directory, not a file: {log_path}")

        file_handler = RotatingFileHandler(
            str(log_path),
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(file_handler)
    except (OSError, PermissionError, ValueError) as e:
        logger.warning(f"Could not create file handler for logging: {e}")

    logger.propagate = False

    return logger

# Example usage:
# from logger import get_logger
# logger = get_logger(__name__)
# logger.info("This is an informational message.")
# logger.warning("This is a warning message.")
