"""
Source-loading coordinator for the integrated training pipeline.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Protocol, TypeAlias

from ai.pipelines.orchestrator.orchestration.storage_resolver import (
    StorageCacheError,
    StorageResolver,
)
from ai.pipelines.orchestrator.utils.logger import get_logger

logger = get_logger("dataset_pipeline.data_ingestion_coordinator")


TrainingRecord: TypeAlias = dict[str, object]


class SourceConfigProtocol(Protocol):
    enabled: bool
    source_path: str | None


class StandardTherapeuticConfigProtocol(Protocol):
    enabled: bool


class PipelineConfigProtocol(Protocol):
    edge_cases: SourceConfigProtocol
    pixel_voice: SourceConfigProtocol
    psychology_knowledge: SourceConfigProtocol
    dual_persona: SourceConfigProtocol
    standard_therapeutic: StandardTherapeuticConfigProtocol


CachedSourceLoader = Callable[[Path | None], list[TrainingRecord]]
StandardSourceLoader = Callable[[], list[TrainingRecord]]
IntakeRoutingApplier = Callable[[list[TrainingRecord], str], list[TrainingRecord]]


class DataIngestionCoordinator:
    """Coordinate source loading and intake routing for the training pipeline."""

    def __init__(
        self,
        *,
        config: PipelineConfigProtocol,
        storage_resolver: StorageResolver,
        load_edge_cases: CachedSourceLoader,
        load_pixel_voice: CachedSourceLoader,
        load_psychology_knowledge: CachedSourceLoader,
        load_dual_persona: CachedSourceLoader,
        load_standard_therapeutic: StandardSourceLoader,
        apply_intake_routing: IntakeRoutingApplier,
        samples_by_source: dict[str, int],
    ) -> None:
        self.config = config
        self.storage_resolver = storage_resolver
        self.load_edge_cases = load_edge_cases
        self.load_pixel_voice = load_pixel_voice
        self.load_psychology_knowledge = load_psychology_knowledge
        self.load_dual_persona = load_dual_persona
        self.load_standard_therapeutic = load_standard_therapeutic
        self.apply_intake_routing = apply_intake_routing
        self.samples_by_source = samples_by_source

    def _warm_cached_paths(self) -> dict[str, Path | None]:
        """Warm remote/local cache paths for enabled sources in parallel."""
        source_map: dict[str, str | None] = {}
        if self.config.edge_cases.enabled:
            source_map["edge_cases"] = self.config.edge_cases.source_path
        if self.config.pixel_voice.enabled:
            source_map["pixel_voice"] = self.config.pixel_voice.source_path
        if self.config.psychology_knowledge.enabled:
            source_map["psychology_knowledge"] = self.config.psychology_knowledge.source_path
        if self.config.dual_persona.enabled:
            source_map["dual_persona"] = self.config.dual_persona.source_path

        warmed: dict[str, Path | None] = {}
        if not source_map:
            return warmed

        max_workers = min(4, len(source_map))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(self.storage_resolver.cache_data, source_path): source_name
                for source_name, source_path in source_map.items()
            }
            for future in as_completed(future_map):
                source_name = future_map[future]
                try:
                    warmed[source_name] = future.result()
                except StorageCacheError as exc:
                    logger.error("Failed to warm cached path for %s: %s", source_name, exc)
                    raise

        return warmed

    def load_all_sources(self) -> list[TrainingRecord]:
        """Load every enabled source and apply canonical intake routing."""
        all_training_data: list[TrainingRecord] = []
        warmed_paths = self._warm_cached_paths()

        if self.config.edge_cases.enabled:
            cached_path = warmed_paths.get("edge_cases")
            edge_data = self.load_edge_cases(cached_path)
            edge_data = self.apply_intake_routing(edge_data, "edge_case")
            all_training_data.extend(edge_data)
            self.samples_by_source["edge_cases"] = len(edge_data)
            logger.info("✅ Loaded %s edge case examples", len(edge_data))

        if self.config.pixel_voice.enabled:
            cached_path = warmed_paths.get("pixel_voice")
            voice_data = self.load_pixel_voice(cached_path)
            voice_data = self.apply_intake_routing(voice_data, "voice_persona")
            all_training_data.extend(voice_data)
            self.samples_by_source["pixel_voice"] = len(voice_data)
            logger.info("✅ Loaded %s voice-derived examples", len(voice_data))

        if self.config.psychology_knowledge.enabled:
            cached_path = warmed_paths.get("psychology_knowledge")
            psych_data = self.load_psychology_knowledge(cached_path)
            psych_data = self.apply_intake_routing(psych_data, "psychology_knowledge")
            all_training_data.extend(psych_data)
            self.samples_by_source["psychology_knowledge"] = len(psych_data)
            logger.info("✅ Loaded %s psychology knowledge examples", len(psych_data))

        if self.config.dual_persona.enabled:
            cached_path = warmed_paths.get("dual_persona")
            persona_data = self.load_dual_persona(cached_path)
            persona_data = self.apply_intake_routing(persona_data, "dual_persona")
            all_training_data.extend(persona_data)
            self.samples_by_source["dual_persona"] = len(persona_data)
            logger.info("✅ Loaded %s dual persona examples", len(persona_data))

        if self.config.standard_therapeutic.enabled:
            standard_data = self.load_standard_therapeutic()
            standard_data = self.apply_intake_routing(
                standard_data, "standard_therapeutic"
            )
            all_training_data.extend(standard_data)
            self.samples_by_source["standard_therapeutic"] = len(standard_data)
            logger.info(
                "✅ Loaded %s standard therapeutic examples", len(standard_data)
            )

        return all_training_data


__all__ = ["DataIngestionCoordinator"]
