from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from ai.pipelines.orchestrator.orchestration.data_ingestion_plan import (
    build_source_definitions,
    resolve_source_load_max_workers,
    resolve_source_warm_max_workers,
)
from ai.pipelines.orchestrator.orchestration.data_ingestion_coordinator import (
    DataIngestionCoordinator,
    SourceLoadFailure,
)


@dataclass
class _SourceConfigStub:
    enabled: bool = True
    source_path: str | None = None


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
        self.calls: list[str | None] = []

    def cache_data(self, source_path: str | None) -> Path | None:
        self.calls.append(source_path)
        return self.resolved_paths[source_path or ""]


def test_data_ingestion_coordinator_warms_and_passes_standard_therapeutic_cached_path(
    tmp_path: Path,
):
    edge_path = "s3://pixel-data/curated_sources/consolidated/edge_cases/existing_edge_cases.jsonl"
    standard_path = (
        "s3://pixel-data/curated_sources/consolidated/final_datasets/"
        "ULTIMATE_FINAL_DATASET.jsonl"
    )
    cached_edge = tmp_path / "edge.jsonl"
    cached_standard = tmp_path / "standard.jsonl"
    resolver = _StorageResolverStub(
        {
            edge_path: cached_edge,
            standard_path: cached_standard,
        }
    )
    standard_loader_calls: list[Path | None] = []
    config = _PipelineConfigStub(
        edge_cases=_SourceConfigStub(enabled=True, source_path=edge_path),
        pixel_voice=_SourceConfigStub(enabled=False, source_path=None),
        psychology_knowledge=_SourceConfigStub(enabled=False, source_path=None),
        dual_persona=_SourceConfigStub(enabled=False, source_path=None),
        standard_therapeutic=_SourceConfigStub(
            enabled=True,
            source_path=standard_path,
        ),
    )
    coordinator = DataIngestionCoordinator(
        storage_resolver=resolver,
        source_definitions=build_source_definitions(
            config=config,
            loaders={
                "edge_cases": lambda path: [],
                "pixel_voice": lambda path: [],
                "psychology_knowledge": lambda path: [],
                "dual_persona": lambda path: [],
                "standard_therapeutic": (
                    lambda path: standard_loader_calls.append(path) or []
                ),
            },
        ),
        apply_intake_routing=lambda records, source_family: records,
        samples_by_source={},
    )

    list(coordinator.load_all_sources())

    assert resolver.calls == [edge_path, standard_path]
    assert standard_loader_calls == [cached_standard]


def test_resolve_source_warm_max_workers_uses_env_cap(monkeypatch):
    monkeypatch.setenv("PIXELATED_SOURCE_WARM_MAX_WORKERS", "2")

    assert resolve_source_warm_max_workers(5) == 2


def test_resolve_source_load_max_workers_uses_env_cap(monkeypatch):
    monkeypatch.setenv("PIXELATED_SOURCE_LOAD_MAX_WORKERS", "3")

    assert resolve_source_load_max_workers(5) == 3


def test_data_ingestion_coordinator_iter_loaded_sources_yields_batches(
    tmp_path: Path,
):
    edge_path = "s3://pixel-data/curated_sources/consolidated/edge_cases/existing_edge_cases.jsonl"
    standard_path = (
        "s3://pixel-data/curated_sources/consolidated/final_datasets/"
        "ULTIMATE_FINAL_DATASET.jsonl"
    )
    cached_edge = tmp_path / "edge.jsonl"
    cached_standard = tmp_path / "standard.jsonl"
    resolver = _StorageResolverStub(
        {
            edge_path: cached_edge,
            standard_path: cached_standard,
        }
    )
    config = _PipelineConfigStub(
        edge_cases=_SourceConfigStub(enabled=True, source_path=edge_path),
        pixel_voice=_SourceConfigStub(enabled=False, source_path=None),
        psychology_knowledge=_SourceConfigStub(enabled=False, source_path=None),
        dual_persona=_SourceConfigStub(enabled=False, source_path=None),
        standard_therapeutic=_SourceConfigStub(
            enabled=True,
            source_path=standard_path,
        ),
    )
    coordinator = DataIngestionCoordinator(
        storage_resolver=resolver,
        source_definitions=build_source_definitions(
            config=config,
            loaders={
                "edge_cases": lambda path: [{"text": "edge"}],
                "pixel_voice": lambda path: [],
                "psychology_knowledge": lambda path: [],
                "dual_persona": lambda path: [],
                "standard_therapeutic": lambda path: [{"text": "standard"}],
            },
        ),
        apply_intake_routing=lambda records, source_family: records,
        samples_by_source={},
    )

    batches = list(coordinator.iter_loaded_sources())

    assert {definition.key for definition, _ in batches} == {
        "edge_cases",
        "standard_therapeutic",
    }
    assert sum(len(records) for _, records in batches) == 2


def test_data_ingestion_coordinator_records_zero_samples_for_failed_source(
    tmp_path: Path,
):
    edge_path = "s3://pixel-data/curated_sources/consolidated/edge_cases/existing_edge_cases.jsonl"
    resolver = _StorageResolverStub({edge_path: tmp_path / "edge.jsonl"})
    samples_by_source: dict[str, int] = {}
    config = _PipelineConfigStub(
        edge_cases=_SourceConfigStub(enabled=True, source_path=edge_path),
        pixel_voice=_SourceConfigStub(enabled=False, source_path=None),
        psychology_knowledge=_SourceConfigStub(enabled=False, source_path=None),
        dual_persona=_SourceConfigStub(enabled=False, source_path=None),
        standard_therapeutic=_SourceConfigStub(enabled=False, source_path=None),
    )
    coordinator = DataIngestionCoordinator(
        storage_resolver=resolver,
        source_definitions=build_source_definitions(
            config=config,
            loaders={
                "edge_cases": lambda path: (_ for _ in ()).throw(RuntimeError("boom")),
                "pixel_voice": lambda path: [],
                "psychology_knowledge": lambda path: [],
                "dual_persona": lambda path: [],
                "standard_therapeutic": lambda path: [],
            },
        ),
        apply_intake_routing=lambda records, source_family: records,
        samples_by_source=samples_by_source,
    )

    with pytest.raises(SourceLoadFailure, match="edge_cases: boom"):
        list(coordinator.iter_loaded_sources())

    assert samples_by_source["edge_cases"] == 0
