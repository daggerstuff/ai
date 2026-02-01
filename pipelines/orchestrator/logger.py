"""
Structured logging for the Pixelated Empathy AI dataset pipeline.
Provides a consistent logging interface with support for multiple levels and formats.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PipelineLogger:
    """
    Structured logger for dataset pipeline operations.
    """

    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Clear existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Console handler with formatting
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an info message."""
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a warning message."""
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an error message."""
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a critical message."""
        self.logger.critical(msg, *args, **kwargs)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a debug message."""
        self.logger.debug(msg, *args, **kwargs)


def get_logger(name: str) -> PipelineLogger:
    """
    Returns a PipelineLogger instance for the given name.
    """
    return PipelineLogger(name)


def setup_pipeline_logging(output_dir: Path | None = None) -> None:
    """
    Initializes global logging configuration for the pipeline.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = (
            output_dir / f"pipeline_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
        )
        handlers.append(logging.FileHandler(str(log_file)))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
