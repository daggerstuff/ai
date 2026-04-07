"""
Source-specific loader orchestration for the integrated training pipeline.
"""

from __future__ import annotations

import json
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


class SourceConfigProtocol(Protocol):
    max_samples: int | None


class SourceLoaderConfigProtocol(Protocol):
    edge_cases: SourceConfigProtocol
    pixel_voice: SourceConfigProtocol
    psychology_knowledge: SourceConfigProtocol
    dual_persona: SourceConfigProtocol


class SourceLoaderService:
    """Own non-standard source loading so the pipeline stays orchestration-only."""

    def __init__(
        self,
        *,
        stats: SourceLoaderStatsProtocol,
        config: SourceLoaderConfigProtocol,
    ) -> None:
        self.stats = stats
        self.config = config

    def load_edge_cases(self, file_paths: list[Path] | None = None) -> list[dict[str, Any]]:
        """Load edge case training data."""
        candidate_paths = file_paths or [None]
        aggregated: list[dict[str, Any]] = []
        try:
            seen_outputs: set[str] = set()
            for file_path in candidate_paths:
                loader = EdgeCaseJSONLLoader(file_path=file_path)
                if not loader.check_pipeline_output_exists():
                    continue
                source_cap = self.config.edge_cases.max_samples
                for record in loader.convert_to_training_format(
                    loader.load_edge_cases(max_samples=source_cap)
                ):
                    record_key = json.dumps(record, sort_keys=True, ensure_ascii=True)
                    if record_key in seen_outputs:
                        continue
                    seen_outputs.add(record_key)
                    aggregated.append(record)
                    if source_cap is not None and len(aggregated) >= source_cap:
                        return aggregated[:source_cap]
            if aggregated:
                return aggregated
            warning = "Edge case data not found. Run edge case pipeline first."
            self._warn(warning)
            return []
        except Exception as exc:
            self._error(f"Failed to load edge cases: {exc}")
            return []

    def load_pixel_voice(self, file_paths: list[Path] | None = None) -> list[dict[str, Any]]:
        """Load Pixel Voice pipeline data."""
        candidate_paths = file_paths or [None]
        aggregated: list[dict[str, Any]] = []
        try:
            seen_texts: set[str] = set()
            for file_path in candidate_paths:
                loader = PixelVoiceLoader(file_path=file_path)
                if not loader.check_pipeline_output_exists():
                    continue
                for record in loader.convert_to_training_format(loader.load_therapeutic_pairs()):
                    text = str(record.get("text", ""))
                    if text in seen_texts:
                        continue
                    seen_texts.add(text)
                    aggregated.append(record)
            if aggregated:
                return aggregated
            warning = "Pixel Voice data not found. Run Pixel Voice pipeline first."
            self._warn(warning)
            return []
        except Exception as exc:
            self._error(f"Failed to load Pixel Voice data: {exc}")
            return []

    def load_psychology_knowledge(
        self, file_paths: list[Path] | None = None
    ) -> list[dict[str, Any]]:
        """Load psychology knowledge base."""
        file_path = file_paths[0] if file_paths else None
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

    def load_dual_persona(self, file_paths: list[Path] | None = None) -> list[dict[str, Any]]:
        """Load dual persona training data."""
        file_path = file_paths[0] if file_paths else None
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
