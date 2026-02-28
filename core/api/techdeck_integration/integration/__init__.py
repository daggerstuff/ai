"""
Integration module for TechDeck-Python Pipeline Integration.

This module provides integration adapters for external services and systems.
"""

from .redis_client import RedisClient
from .pipeline_orchestrator import PipelineOrchestrator
from .bias_detection import BiasDetectionManager, detect_bias_in_dataset

__all__ = [
    'RedisClient',
    'PipelineOrchestrator', 
    'BiasDetectionManager',
    'detect_bias_in_dataset'
]