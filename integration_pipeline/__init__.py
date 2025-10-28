"""
Integration Pipeline for Pixelated Empathy Components
Combines all built components into unified training datasets
"""

from .master_pipeline import MasterIntegrationPipeline
from .component_integrator import ComponentIntegrator
from .dataset_builder import IntegratedDatasetBuilder

__version__ = "1.0.0"
__all__ = ["MasterIntegrationPipeline", "ComponentIntegrator", "IntegratedDatasetBuilder"]