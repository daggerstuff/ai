"""
Journal Dataset Research System

A comprehensive system for researching, evaluating, and acquiring therapeutic
journal datasets from open access sources.
"""

__version__ = "0.1.0"

from ai.sourcing.journal.orchestrator import (
    OrchestratorConfig,
    ResearchOrchestrator,
    SessionState,
)

__all__ = ["ResearchOrchestrator", "OrchestratorConfig", "SessionState"]
