"""
Source-specific loader orchestration for the integrated training pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from ai.pipelines.orchestrator.ingestion.dual_persona_loader import (
    DualPersonaLoader,
)
from ai.pipelines.orchestrator.ingestion.edge_case_jsonl_loader import (
    EdgeCaseJSONLLoader,
)
from ai.pipelines.orchestrator.ingestion.pixel_voice_loader import (
    PixelVoiceLoader,
)
from ai.pipelines.orchestrator.ingestion.psychology_knowledge_loader import (
    PsychologyKnowledgeLoader,
)
from ai.pipelines.orchestrator.utils.logger import get_logger

logger = get_logger("dataset_pipeline.source_loader_service")


class SourceLoaderStatsProtocol(Protocol):
    warnings: list[str]
    errors: list[str]


class SourceLoaderService:
    """Own non-standard source loading so the pipeline stays orchestration-only."""

    def __init__(self, *, stats: SourceLoaderStatsProtocol) -> None:
        self.stats = stats

    def load_edge_cases(self, file_path: Path | None = None) -> list[dict[str, Any]]:
        """Load edge case training data."""
        try:
            loader = EdgeCaseJSONLLoader(file_path=file_path)
            if not loader.check_pipeline_output_exists():
                warning = "Edge case data not found. Run edge case pipeline first."
                self._warn(warning)
                return []
            return loader.convert_to_training_format(loader.load_edge_cases())
        except Exception as exc:
            self._error(f"Failed to load edge cases: {exc}")
            return []

    def load_pixel_voice(self, file_path: Path | None = None) -> list[dict[str, Any]]:
        """Load Pixel Voice pipeline data."""
        try:
            loader = PixelVoiceLoader(file_path=file_path)
            if not loader.check_pipeline_output_exists():
                warning = "Pixel Voice data not found. Run Pixel Voice pipeline first."
                self._warn(warning)
                return []
            return loader.convert_to_training_format(loader.load_therapeutic_pairs())
        except Exception as exc:
            self._error(f"Failed to load Pixel Voice data: {exc}")
            return []

    def load_psychology_knowledge(
        self, file_path: Path | None = None
    ) -> list[dict[str, Any]]:
        """Load psychology knowledge base."""
        try:
            loader = PsychologyKnowledgeLoader(file_path=file_path)
            if not loader.check_knowledge_base_exists():
                warning = "Psychology knowledge base not found."
                self._warn(warning)
                return []
            return loader.convert_to_training_format(loader.load_concepts())
        except Exception as exc:
            self._error(f"Failed to load psychology knowledge: {exc}")
            return []

    def load_dual_persona(self, file_path: Path | None = None) -> list[dict[str, Any]]:
        """Load dual persona training data."""
        try:
            loader = DualPersonaLoader(file_path=file_path)
            return loader.convert_to_training_format(loader.load_dialogues())
        except Exception as exc:
            self._error(f"Failed to load dual persona data: {exc}")
            return []

    def _warn(self, message: str) -> None:
        logger.warning(message)
        self.stats.warnings.append(message)

    def _error(self, message: str) -> None:
        logger.error(message)
        self.stats.errors.append(message)


__all__ = ["SourceLoaderService"]
