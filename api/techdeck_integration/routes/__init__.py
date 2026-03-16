"""
API Routes Module for TechDeck-Python Pipeline Integration.

This module contains all Flask blueprints for the REST API endpoints,
organized by functional domain.
"""

from .datasets import datasets_bp
from .pipeline import pipeline_bp
# Additional route blueprints will be imported here as they are created
# from .standardization import standardization_bp
# from .validation import validation_bp
# from .analytics import analytics_bp
# from .system import system_bp


__all__ = [
    'datasets_bp',
    'pipeline_bp',
    # 'standardization_bp',
    # 'validation_bp',
    # 'analytics_bp',
    # 'system_bp',
]