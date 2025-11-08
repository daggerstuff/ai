"""
NVIDIA NeMo Data Designer Integration

This module provides integration with NVIDIA NeMo Data Designer for generating
high-quality, domain-specific synthetic datasets for training, fine-tuning, and
evaluating AI models in the Pixelated Empathy platform.
"""

from ai.data_designer.service import NeMoDataDesignerService
from ai.data_designer.config import DataDesignerConfig
from ai.data_designer.integration import (
    BiasDetectionIntegration,
    DatasetPipelineIntegration,
    TherapeuticDatasetIntegration,
)
from ai.data_designer.edge_case_generator import (
    EdgeCaseGenerator,
    EdgeCaseType,
)
from ai.data_designer.edge_case_api import EdgeCaseAPI

__all__ = [
    'NeMoDataDesignerService',
    'DataDesignerConfig',
    'BiasDetectionIntegration',
    'DatasetPipelineIntegration',
    'TherapeuticDatasetIntegration',
    'EdgeCaseGenerator',
    'EdgeCaseType',
    'EdgeCaseAPI',
]

