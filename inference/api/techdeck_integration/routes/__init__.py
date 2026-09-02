"""
API Routes Module for TechDeck-Python Pipeline Integration.

This module contains all Flask blueprints for the REST API endpoints,
organized by functional domain.
"""

from .datasets import datasets_bp
from .pipeline import pipeline_bp

# Additional route blueprints will be imported here as they are created


__all__ = [
    "datasets_bp",
    "pipeline_bp",
]
