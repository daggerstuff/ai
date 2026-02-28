"""
Integration Pipeline for Pixelated Empathy Components
Combines all built components into unified training datasets
"""

from .master_pipeline import IntegrationConfig, MasterIntegrationPipeline

__version__ = "1.0.0"
__all__ = ["IntegrationConfig", "MasterIntegrationPipeline"]
