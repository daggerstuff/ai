"""
NVIDIA NeMo Data Designer Integration

This module provides integration with NVIDIA NeMo Data Designer for generating
high-quality, domain-specific synthetic datasets for training, fine-tuning, and
evaluating AI models in the Pixelated Empathy platform.
"""

from ai.pipelines.design.service import NeMoDataDesignerService
from ai.pipelines.design.config import DataDesignerConfig
from ai.pipelines.design.integration import (
    BiasDetectionIntegration,
    DatasetPipelineIntegration,
    TherapeuticDatasetIntegration,
)
from ai.pipelines.design.edge_case_generator import (
    EdgeCaseGenerator,
    EdgeCaseType,
)
from ai.pipelines.design.edge_case_api import EdgeCaseAPI

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

