"""
Journal Dataset Research System

A comprehensive system for researching, evaluating, and acquiring therapeutic
journal datasets from open access sources.
"""

__version__ = "0.1.0"

from ai.pipelines.data_processing.journal.orchestrator import (
    OrchestratorConfig,
    ResearchOrchestrator,
    SessionState,
)

__all__ = ["OrchestratorConfig", "ResearchOrchestrator", "SessionState"]
