"""
Dataset registry helpers (S3-first).

Pixelated uses a single canonical dataset location: S3.
`ai/data/dataset_registry.json` is the authoritative mapping of dataset families to S3 URIs.

Local paths and GDrive mounts are treated as fallbacks for development and legacy workflows only.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetRef:
    """A single dataset reference from the registry."""

    key: str
    s3_path: str
    stage: str | None
    quality_profile: str | None
    type: str | None
    focus: str | None
    fallback_paths: dict[str, str]
    legacy_paths: list[str]


def get_default_registry_path() -> Path:
    # ai/common/dataset_registry.py -> ai/common/ -> ai/ -> project root
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "ai" / "data" / "dataset_registry.json"


def load_registry(path: Path | None = None) -> dict[str, Any]:
    registry_path = path or get_default_registry_path()
    with open(registry_path, encoding="utf-8") as f:
        return json.load(f)


def _iter_registry_dataset_sections(
    registry: dict[str, Any],
) -> Iterable[tuple[str, dict[str, Any]]]:
    datasets = registry.get("datasets", {})
    if isinstance(datasets, dict):
        for section_name, section in datasets.items():
            if isinstance(section, dict):
                yield section_name, section


def _iter_registry_groups(registry: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    """
    Yield all top-level groups that contain dataset-like entries.

    The registry has a primary `datasets` mapping plus other groups like
    `edge_case_sources` and `voice_persona` that also contain entries with `path`.
    """

    yield from _iter_registry_dataset_sections(registry)

    for group_name in ["edge_case_sources", "voice_persona"]:
        group = registry.get(group_name)
        if isinstance(group, dict):
            yield group_name, group


def iter_dataset_refs(registry: dict[str, Any]) -> Iterable[DatasetRef]:
    """
    Iterate all concrete datasets in the registry (not stage/group containers).

    The registry structure is:
    datasets -> <section> -> <datasetName> -> { path, stage, quality_profile, ... }
    """

    for section_name, section in _iter_registry_groups(registry):
        for dataset_name, dataset in section.items():
            if not isinstance(dataset, dict):
                continue

            s3_path = dataset.get("path")
            if not isinstance(s3_path, str) or not s3_path.startswith("s3://"):
                # Skip non-S3 entries; registry is S3-first.
                continue

            fallback_paths = dataset.get("fallback_paths")
            if not isinstance(fallback_paths, dict):
                fallback_paths = {}

            legacy_paths = dataset.get("legacy_paths")
            if not isinstance(legacy_paths, list):
                legacy_paths = []

            yield DatasetRef(
                key=f"{section_name}.{dataset_name}",
                s3_path=s3_path,
                stage=dataset.get("stage") if isinstance(dataset.get("stage"), str) else None,
                quality_profile=dataset.get("quality_profile")
                if isinstance(dataset.get("quality_profile"), str)
                else None,
                type=dataset.get("type") if isinstance(dataset.get("type"), str) else None,
                focus=dataset.get("focus") if isinstance(dataset.get("focus"), str) else None,
                fallback_paths={
                    str(k): str(v) for k, v in fallback_paths.items() if isinstance(v, str)
                },
                legacy_paths=[str(p) for p in legacy_paths if isinstance(p, str)],
            )


def resolve_fallback_path(dataset: DatasetRef, *, prefer: list[str] | None = None) -> str | None:
    """
    Return the first available fallback path by preference order.

    This does not check filesystem existence. Callers decide how/when to validate.
    """

    prefer_keys = prefer or ["local", "gdrive"]
    for key in prefer_keys:
        if val := dataset.fallback_paths.get(key):
            return val
    # Any fallback, deterministic order
    for key in sorted(dataset.fallback_paths.keys()):
        return dataset.fallback_paths[key]
    return None
