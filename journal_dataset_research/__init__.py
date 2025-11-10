"""
Journal Dataset Research System

A comprehensive system for researching, evaluating, and acquiring therapeutic
journal datasets from open access sources.
"""

__version__ = "0.1.0"

from ai.journal_dataset_research.orchestrator import (
    OrchestratorConfig,
    ResearchOrchestrator,
    SessionState,
)

__all__ = ["ResearchOrchestrator", "OrchestratorConfig", "SessionState"]
