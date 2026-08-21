"""Registry-backed source inventory for the fresh training corpus builder.

Factual-only: computes source identity, family, stage, locator, type, and
provenance metadata from the registry. All admissibility verdicts (keep/defer/
reject), rights status, license status, lane eligibility, and benchmark role
are intentionally NOT decided here — those belong to human/dataset-owner review
(see ``docs/dataset-source-review.md``).

The prior version of this file hardcoded keep/reject/lane/license verdicts per
source group, which made the inventory the de-facto ingest gate for the
training corpus. That is a human decision. This module reports what a source IS,
not whether it may be used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ai.utils.common.dataset_registry import load_registry

from .model import CorpusSource

_SIMPLE_FAMILY_BY_GROUP = {
    "cot_reasoning": "reasoning_cot",
    "edge_case_sources": "edge_case_nightmare",
    "professional_therapeutic": "professional_therapeutic",
    "therapeutic": "legacy_compiled_mix",
    "voice_persona": "persona_transcript_derived",
    "wendy_curated_sets": "curated_priority",
}


def _canonical_family(group_name: str, dataset: dict[str, Any]) -> str:
    stage = str(dataset.get("stage") or "")
    dataset_type = str(dataset.get("type") or "")

    if group_name == "supplementary":
        return "knowledge_literature" if dataset_type in {"knowledge_base", "research"} else "unclassified"
    if group_name == "training_v3":
        if stage == "stage4_voice_persona":
            return "persona_archetype"
        if stage == "stage3_edge_stress_test":
            return "edge_case_nightmare"
        if stage == "stage2_therapeutic_expertise":
            return "specialized_domain"
        return "simulation_training"
    return _SIMPLE_FAMILY_BY_GROUP.get(group_name, "unclassified")


def _candidate_locator(dataset: dict[str, Any]) -> Path:
    fallback_paths = dataset.get("fallback_paths")
    if isinstance(fallback_paths, dict):
        for key in ("local", "gdrive", "local_dir", "gdrive_dir"):
            value = fallback_paths.get(key)
            if isinstance(value, str) and value:
                return Path(value).expanduser()
        for value in fallback_paths.values():
            if isinstance(value, str) and value:
                return Path(value).expanduser()

    legacy_paths = dataset.get("legacy_paths")
    if isinstance(legacy_paths, list):
        for value in legacy_paths:
            if isinstance(value, str) and value:
                return Path(value).expanduser()

    path_value = dataset.get("path")
    return Path(str(path_value or "."))


def _provenance_status(dataset: dict[str, Any]) -> str:
    """Factual provenance reachability from the registry entry.

    Reports where the source's locator can be resolved from (registry path,
    fallback paths, both, or unknown). This is an observation, not a verdict
    on whether the source may be used.
    """
    path_value = str(dataset.get("path") or "")
    fallback_paths = dataset.get("fallback_paths")
    has_fallback = isinstance(fallback_paths, dict) and any(
        isinstance(value, str) and value for value in fallback_paths.values()
    )
    if path_value.startswith("s3://") and has_fallback:
        return "registry_and_fallback"
    if path_value.startswith("s3://"):
        return "registry_only"
    if has_fallback:
        return "fallback_only"
    return "unknown"


def inventory_rows(sources: tuple[CorpusSource, ...]) -> list[dict[str, object]]:
    return [
        {
            "source_id": source.source_id,
            "registry_group": source.registry_group,
            "family": source.family,
            "stage": source.stage,
            "source_type": source.source_type,
            "quality_profile": source.quality_profile,
            "focus": source.focus,
            "inventory_decision": source.inventory_decision,
            "rights_status": source.rights_status,
            "license_status": source.license_status,
            "provenance_status": source.provenance_status,
            "benchmark_role": source.benchmark_role,
            "allowed_lanes": list(source.allowed_lanes),
            "default_lane": source.default_lane,
            "locator": str(source.locator),
            "locator_exists": source.locator.exists(),
            "locator_is_file": source.locator.is_file(),
            "notes": list(source.notes),
            "provenance": source.provenance,
        }
        for source in sources
    ]


def build_source_inventory(registry_path: Path) -> tuple[CorpusSource, ...]:
    registry = load_registry(registry_path)
    inventory: list[CorpusSource] = []

    dataset_groups = registry.get("datasets", {})
    if isinstance(dataset_groups, dict):
        for group_name, group in dataset_groups.items():
            if not isinstance(group, dict):
                continue
            for dataset_name, dataset in group.items():
                if isinstance(dataset, dict):
                    inventory.append(_build_source(group_name, dataset_name, dataset))

    for group_name in ("edge_case_sources", "voice_persona", "supplementary"):
        group = registry.get(group_name)
        if not isinstance(group, dict):
            continue
        for dataset_name, dataset in group.items():
            if isinstance(dataset, dict):
                inventory.append(_build_source(group_name, dataset_name, dataset))

    return tuple(inventory)


def discover_approved_sources(registry_path: Path) -> tuple[CorpusSource, ...]:
    """Return sources currently marked ``keep`` with a resolvable lane + locator.

    NOTE: admissibility verdicts are populated by the human review workflow
    (``docs/dataset-source-review.md``), not by this module. Until a human
    signs off on a source it stays ``defer`` and is not returned here.
    """
    approved: list[CorpusSource] = []
    for source in build_source_inventory(registry_path):
        if source.inventory_decision != "keep":
            continue
        if source.default_lane is None:
            continue
        if not source.locator.exists() or not source.locator.is_file():
            continue
        approved.append(source)
    return tuple(approved)


def _build_source(group_name: str, dataset_name: str, dataset: dict[str, Any]) -> CorpusSource:
    return CorpusSource(
        source_id=f"{group_name}.{dataset_name}",
        registry_group=group_name,
        family=_canonical_family(group_name, dataset),
        stage=str(dataset.get("stage") or "stage1_foundation"),
        locator=_candidate_locator(dataset),
        source_type=str(dataset.get("type") or "registry"),
        quality_profile=str(dataset.get("quality_profile")) if dataset.get("quality_profile") else None,
        focus=str(dataset.get("focus")) if dataset.get("focus") else None,
        # Admissibility fields intentionally default to neutral — human review decides.
        inventory_decision="defer",
        rights_status="unknown",
        license_status="unknown",
        provenance_status=_provenance_status(dataset),
        benchmark_role="not_eligible",
        allowed_lanes=(),
        default_lane=None,
        notes=(),
        provenance={
            "registry_path": dataset.get("path"),
            "fallback_paths": dataset.get("fallback_paths", {}),
            "legacy_paths": dataset.get("legacy_paths", []),
        },
    )
