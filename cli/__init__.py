"""
Pixelated AI CLI - Command Line Interface for TechDeck-Python Pipeline Integration

This module provides a comprehensive CLI tool for managing AI pipelines,
configuration, and interactions with the Pixelated Empathy platform.
"""

__version__ = "0.1.0"
__author__ = "Pixelated Empathy Team"
__description__ = "CLI for TechDeck-Python Pipeline Integration"

from .main import cli
from .config import CLIConfig
from .auth import AuthManager
from .pipeline import PipelineManager
from .progress import ProgressTracker
from .utils import setup_logging, validate_environment

__all__ = [
    "cli",
    "CLIConfig", 
    "AuthManager",
    "PipelineManager",
    "ProgressTracker",
    "setup_logging",
    "validate_environment",
]