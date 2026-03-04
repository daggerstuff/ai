"""
NVIDIA NeMo Data Designer Integration

This module provides integration with NVIDIA NeMo Data Designer for generating
high-quality, domain-specific synthetic datasets for training, fine-tuning, and
evaluating AI models in the Pixelated Empathy platform.
"""

from ai.core.pipelines.design.config import DataDesignerConfig
from ai.core.pipelines.design.edge_case_api import EdgeCaseAPI
from ai.core.pipelines.design.edge_case_generator import (
    EdgeCaseGenerator,
    EdgeCaseType,
)
from ai.core.pipelines.design.integration import (
    BiasDetectionIntegration,
    DatasetPipelineIntegration,
    TherapeuticDatasetIntegration,
)
from ai.core.pipelines.design.service import NeMoDataDesignerService

__all__ = [
    "NeMoDataDesignerService",
    "DataDesignerConfig",
    "BiasDetectionIntegration",
    "DatasetPipelineIntegration",
    "TherapeuticDatasetIntegration",
    "EdgeCaseGenerator",
    "EdgeCaseType",
    "EdgeCaseAPI",
]
