"""Execution context for dataset assembly workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai.pipelines.orchestrator.ingestion.intake_gates import OrchestratorIntakeGates
from ai.pipelines.orchestrator.ingestion.intake_routing_adapter import (
    apply_intake_routing,
)
from ai.pipelines.orchestrator.orchestration.storage_resolver import StorageResolver
from ai.pipelines.orchestrator.storage_manager import StorageManager


@dataclass
class PipelineExecutionContext:
    """Shared runtime state for orchestrator execution."""

    config: Any
    stats: Any
    intake_gates: OrchestratorIntakeGates = field(default_factory=OrchestratorIntakeGates)
    storage: StorageManager | None = None
    storage_resolver: StorageResolver = field(init=False)

    def __post_init__(self) -> None:
        self.storage_resolver = StorageResolver(self.storage)

    def attach_storage(self, storage: StorageManager | None) -> None:
        self.storage = storage
        self.storage_resolver = StorageResolver(storage)

    def configured_source_paths(self) -> list[str]:
        return [
            self.config.edge_cases.source_path or "",
            self.config.pixel_voice.source_path or "",
            self.config.psychology_knowledge.source_path or "",
            self.config.dual_persona.source_path or "",
            self.config.standard_therapeutic.source_path or "",
        ]

    def cache_data(self, source_path: str | None):
        return self.storage_resolver.cache_data(source_path)

    def apply_intake_routing(
        self, records: list[dict[str, Any]], source_family: str
    ) -> list[dict[str, Any]]:
        return apply_intake_routing(
            records,
            source_family=source_family,
            intake_gates=self.intake_gates,
        )

    def initialize_stage_balance(self, stage_definitions: list[Any]) -> None:
        for stage in stage_definitions:
            if stage.id not in self.stats.stage_balance:
                self.stats.stage_balance[stage.id] = {
                    "target_share": stage.target_share,
                    "actual_samples": 0,
                }


__all__ = ["PipelineExecutionContext"]
