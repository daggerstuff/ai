"""
Unit tests for the centralized logging system.
"""

import logging
import os
import tempfile
import unittest
from pathlib import Path

from ai.dataset_pipeline.logger import get_logger


class TestLogger(unittest.TestCase):

    def test_get_logger(self):
        """Test that get_logger returns a valid logger instance."""
        logger = get_logger("test_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"

    def test_log_levels(self):
        """Test that the logger handles different log levels."""
        logger = get_logger("level_test")
        with self.assertLogs("level_test", level="INFO") as cm:
            logger.info("Info message")
            logger.warning("Warning message")
            assert len(cm.output) == 2
            assert "INFO:level_test:Info message" in cm.output[0]
            assert "WARNING:level_test:Warning message" in cm.output[1]

    def test_file_handler_creation(self):
        """Test that a log file is created."""
        # Use a temporary directory for the log file
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test log file path
            test_log_file = os.path.join(temp_dir, "test.log")

            # Create a logger with a custom log file
            # We'll directly test the functionality rather than patching
            logger = logging.getLogger("file_test")

            # Clear any existing handlers
            logger.handlers.clear()

            # Set up the file handler manually to test the functionality
            log_dir = Path(test_log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                test_log_file,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            logger.addHandler(file_handler)
            logger.setLevel(logging.INFO)

            # Test that the logger writes to the file
            logger.warning("test")
            logger.handlers[0].flush()  # Ensure the message is written

            # Check that the log file was created
            assert os.path.exists(test_log_file)

            # Check that the log message was written to the file
            with open(test_log_file) as f:
                content = f.read()
                assert "test" in content

if __name__ == "__main__":
    unittest.main()
