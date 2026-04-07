from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from ai.pipelines.orchestrator.orchestration.data_ingestion_coordinator import (
    DataIngestionCoordinator,
)
from ai.pipelines.orchestrator.orchestration.storage_resolver import StorageCacheError


@dataclass
class _SourceConfigStub:
    enabled: bool = True
    source_path: str | None = None
    source_paths: tuple[str, ...] = ()


@dataclass
class _PipelineConfigStub:
    edge_cases: _SourceConfigStub
    pixel_voice: _SourceConfigStub
    psychology_knowledge: _SourceConfigStub
    dual_persona: _SourceConfigStub
    standard_therapeutic: _SourceConfigStub


class _StorageResolverStub:
    def __init__(self, resolved_paths: dict[str, Path | None]):
        self.resolved_paths = resolved_paths
        self.calls: list[str] = []

    def cache_data(self, source_path: str | None) -> Path | None:
        key = source_path or ""
        self.calls.append(key)
        return self.resolved_paths[key]


def test_data_ingestion_coordinator_warms_cached_sources_and_updates_counts(
    tmp_path: Path,
):
    edge_path = "s3://pixel-data/edge.jsonl"
    voice_path = "s3://pixel-data/voice.jsonl"
    resolver = _StorageResolverStub(
        {
            edge_path: tmp_path / "edge.jsonl",
            voice_path: tmp_path / "voice.jsonl",
        }
    )
    routing_calls: list[tuple[str, int]] = []

    config = _PipelineConfigStub(
        edge_cases=_SourceConfigStub(enabled=True, source_path=edge_path),
        pixel_voice=_SourceConfigStub(enabled=True, source_path=voice_path),
        psychology_knowledge=_SourceConfigStub(enabled=False),
        dual_persona=_SourceConfigStub(enabled=False),
        standard_therapeutic=_SourceConfigStub(enabled=False),
    )

    coordinator = DataIngestionCoordinator(
        config=config,
        storage_resolver=resolver,
        load_edge_cases=lambda paths: [{"text": str(paths[0])}] if paths else [],
        load_pixel_voice=lambda paths: [{"text": str(paths[0])}] if paths else [],
        load_psychology_knowledge=lambda paths: [],
        load_dual_persona=lambda paths: [],
        load_standard_therapeutic=lambda: [],
        apply_intake_routing=(
            lambda records, source_family: routing_calls.append(
                (source_family, len(records))
            )
            or records
        ),
        samples_by_source={},
    )

    loaded = coordinator.load_all_sources()

    assert len(loaded) == 2
    assert resolver.calls == [edge_path, voice_path]
    assert routing_calls == [("edge_case", 1), ("voice_persona", 1)]
    assert coordinator.samples_by_source == {
        "edge_cases": 1,
        "pixel_voice": 1,
    }


def test_data_ingestion_coordinator_dedupes_configured_source_paths():
    config = _SourceConfigStub(
        enabled=True,
        source_path="s3://pixel-data/edge.jsonl",
        source_paths=(
            "s3://pixel-data/edge.jsonl",
            "s3://pixel-data/edge-2.jsonl",
        ),
    )

    assert DataIngestionCoordinator._configured_source_paths(config) == [
        "s3://pixel-data/edge.jsonl",
        "s3://pixel-data/edge-2.jsonl",
    ]


def test_data_ingestion_coordinator_raises_storage_cache_errors(tmp_path: Path):
    edge_path = "s3://pixel-data/edge.jsonl"

    class _FailingResolver:
        def cache_data(self, source_path: str | None) -> Path | None:
            raise StorageCacheError(source_path or "", "failed to cache")

    coordinator = DataIngestionCoordinator(
        config=_PipelineConfigStub(
            edge_cases=_SourceConfigStub(enabled=True, source_path=edge_path),
            pixel_voice=_SourceConfigStub(enabled=False),
            psychology_knowledge=_SourceConfigStub(enabled=False),
            dual_persona=_SourceConfigStub(enabled=False),
            standard_therapeutic=_SourceConfigStub(enabled=False),
        ),
        storage_resolver=_FailingResolver(),
        load_edge_cases=lambda paths: [],
        load_pixel_voice=lambda paths: [],
        load_psychology_knowledge=lambda paths: [],
        load_dual_persona=lambda paths: [],
        load_standard_therapeutic=lambda: [],
        apply_intake_routing=lambda records, source_family: records,
        samples_by_source={},
    )

    with pytest.raises(StorageCacheError, match="failed to cache"):
        coordinator.load_all_sources()
