"""
Dataset Pipeline Package Initialization

This file sets up default configuration and ensures the ai.dataset_pipeline
package is recognized by Python.
"""

# Import the main config class
from .config import Config, get_config

# Make configuration available at package level
__all__ = ["Config", "get_config"]
