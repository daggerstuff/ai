"""
Source-definition and worker-cap planning for data ingestion.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Protocol, TypeAlias


TrainingRecord: TypeAlias = dict[str, object]
CachedSourceLoader = Callable[[object | None], list[TrainingRecord]]
StandardSourceLoader = Callable[[object | None], list[TrainingRecord]]


class SourceConfigProtocol(Protocol):
    enabled: bool
    source_path: str | None


@dataclass(frozen=True)
class SourceDefinition:
    key: str
    source_family: str
    log_label: str
    config: SourceConfigProtocol
    loader: CachedSourceLoader | StandardSourceLoader


def build_source_definitions(*, config, loaders: dict[str, CachedSourceLoader | StandardSourceLoader]) -> tuple[SourceDefinition, ...]:
    return (
        SourceDefinition(
            key="edge_cases",
            source_family="edge_case",
            log_label="edge case",
            config=config.edge_cases,
            loader=loaders["edge_cases"],
        ),
        SourceDefinition(
            key="pixel_voice",
            source_family="voice_persona",
            log_label="voice-derived",
            config=config.pixel_voice,
            loader=loaders["pixel_voice"],
        ),
        SourceDefinition(
            key="psychology_knowledge",
            source_family="psychology_knowledge",
            log_label="psychology knowledge",
            config=config.psychology_knowledge,
            loader=loaders["psychology_knowledge"],
        ),
        SourceDefinition(
            key="dual_persona",
            source_family="dual_persona",
            log_label="dual persona",
            config=config.dual_persona,
            loader=loaders["dual_persona"],
        ),
        SourceDefinition(
            key="standard_therapeutic",
            source_family="standard_therapeutic",
            log_label="standard therapeutic",
            config=config.standard_therapeutic,
            loader=loaders["standard_therapeutic"],
        ),
    )


def resolve_source_warm_max_workers(source_count: int) -> int:
    return _resolve_worker_cap(
        source_count=source_count,
        env_var_name="PIXELATED_SOURCE_WARM_MAX_WORKERS",
    )


def resolve_source_load_max_workers(source_count: int) -> int:
    return _resolve_worker_cap(
        source_count=source_count,
        env_var_name="PIXELATED_SOURCE_LOAD_MAX_WORKERS",
    )


def _resolve_worker_cap(*, source_count: int, env_var_name: str) -> int:
    configured_workers = os.getenv(env_var_name, "").strip()
    worker_cap = int(configured_workers) if configured_workers.isdigit() else source_count
    return max(1, min(worker_cap, source_count))


__all__ = [
    "CachedSourceLoader",
    "SourceDefinition",
    "SourceConfigProtocol",
    "StandardSourceLoader",
    "TrainingRecord",
    "build_source_definitions",
    "resolve_source_load_max_workers",
    "resolve_source_warm_max_workers",
]
