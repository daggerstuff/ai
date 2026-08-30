"""P13 release manifest builder for construction releases (PIX-4584).

Builds a release manifest with evidence for all P13 gates:
distribution, leakage, provenance, PII, hash, and statistics.

Links split manifests to DVC lineage and human-review approval.

The manifest is a JSON document with this top-level structure::

    {
      "release_version": "...",
      "pipeline_version": "...",
      "created_at": "ISO-8601",
      "split_ratio": [70, 15, 15],
      "gates_passed": true,
      "distribution": {...},
      "leakage": {...},
      "provenance": {...},
      "pii": {...},
      "hash": {...},
      "statistics": {...},
      "dvc_lineage": {...},
      "human_review": {...}
    }
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from training.data_versioning import list_available_versions
from training.dataset_splitter_stratified import (
    AXIS_NAMES,
    SPLIT_NAMES,
    _extract_axes,
    _is_evaluation_material,
    _source_family_key,
    _stratum_key,
    get_convo_hash,
)

# P13 evidence categories required by PIX-4584
P13_EVIDENCE_CATEGORIES = (
    "distribution",
    "leakage",
    "provenance",
    "pii",
    "hash",
    "statistics",
)

# Default human-review status for new release manifests
DEFAULT_REVIEW_STATUS = "pending"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _build_distribution_evidence(
    splits: dict[str, list[dict]],
) -> dict:
    """Per-split counts, per-stratum counts, per-axis value distributions."""
    total = sum(len(s) for s in splits.values())
    per_split = {name: len(items) for name, items in splits.items()}

    # Per-stratum counts (using the full 10-axis stratum key)
    stratum_counts: dict[str, Counter] = defaultdict(Counter)
    for split_name, items in splits.items():
        for item in items:
            stratum_counts[_stratum_key(item)][split_name] += 1

    # Per-axis value distributions
    axis_distributions: dict[str, dict[str, Counter]] = {
        name: {split: Counter() for split in SPLIT_NAMES} for name in AXIS_NAMES
    }
    for split_name, items in splits.items():
        for item in items:
            axes = _extract_axes(item)
            for axis_name in AXIS_NAMES:
                axis_distributions[axis_name][split_name][axes[axis_name]] += 1

    # Convert Counters to plain dicts for JSON serialization
    stratum_counts_serializable = {
        stratum: dict(counts) for stratum, counts in stratum_counts.items()
    }
    axis_distributions_serializable = {
        axis: {split: dict(counter) for split, counter in dist.items()}
        for axis, dist in axis_distributions.items()
    }

    return {
        "total_records": total,
        "per_split": per_split,
        "per_stratum": stratum_counts_serializable,
        "per_axis": axis_distributions_serializable,
    }


def _build_leakage_evidence(
    splits: dict[str, list[dict]],
) -> dict:
    """Leakage check results: hash-disjoint, source-family disjoint, eval-in-train."""
    # Hash-disjoint check
    hash_to_splits: dict[str, set[str]] = defaultdict(set)
    for split_name, items in splits.items():
        for item in items:
            hash_to_splits[get_convo_hash(item)].add(split_name)
    hash_leaks = [h for h, names in hash_to_splits.items() if len(names) > 1]

    # Source-family disjoint check
    family_to_splits: dict[str, set[str]] = defaultdict(set)
    for split_name, items in splits.items():
        for item in items:
            family_to_splits[_source_family_key(item)].add(split_name)
    family_leaks = [f for f, names in family_to_splits.items() if len(names) > 1]

    # Evaluation-in-train check
    eval_in_train = [
        get_convo_hash(item)
        for item in splits.get("train", [])
        if _is_evaluation_material(item)
    ]

    return {
        "hash_disjoint": {
            "passed": len(hash_leaks) == 0,
            "leaked_count": len(hash_leaks),
            "leaked_hashes": hash_leaks[:10],  # cap for brevity
        },
        "source_family_disjoint": {
            "passed": len(family_leaks) == 0,
            "leaked_count": len(family_leaks),
            "leaked_families": family_leaks[:10],
        },
        "evaluation_in_train": {
            "passed": len(eval_in_train) == 0,
            "blocked_count": len(eval_in_train),
            "blocked_hashes": eval_in_train[:10],
        },
    }


def _build_provenance_evidence(
    splits: dict[str, list[dict]],
) -> dict:
    """Provenance: unique source URLs, licenses, pipeline version."""
    source_urls: set[str] = set()
    licenses: set[str] = set()
    source_types: set[str] = set()

    for items in splits.values():
        for item in items:
            provenance = item.get("provenance") or {}
            if isinstance(provenance, dict):
                url = provenance.get("source_url")
                if url:
                    source_urls.add(str(url))
                lic = provenance.get("license")
                if lic:
                    licenses.add(str(lic))
                stype = provenance.get("source_type")
                if stype:
                    source_types.add(str(stype))

            # Also check metadata for provenance fields
            meta = item.get("metadata") or item.get("attributes") or {}
            url = meta.get("source_url")
            if url:
                source_urls.add(str(url))
            lic = meta.get("license")
            if lic:
                licenses.add(str(lic))

    return {
        "source_urls": sorted(source_urls),
        "licenses": sorted(licenses),
        "source_types": sorted(source_types),
        "pipeline_version": "modern-dataset-provenance-v1",
    }


def _build_pii_evidence(
    splits: dict[str, list[dict]],
) -> dict:
    """PII redaction status from records."""
    pii_redacted_count = 0
    pii_not_redacted_count = 0
    pii_types_found: set[str] = set()

    for items in splits.values():
        for item in items:
            meta = item.get("metadata") or item.get("attributes") or {}
            pii_status = meta.get("pii_redacted") or item.get("pii_redacted")
            if pii_status is True or pii_status == "true":
                pii_redacted_count += 1
            else:
                pii_not_redacted_count += 1

            types = meta.get("pii_types_found")
            if types and isinstance(types, list):
                pii_types_found.update(str(t) for t in types)

    return {
        "redacted_records": pii_redacted_count,
        "not_redacted_records": pii_not_redacted_count,
        "pii_types_found": sorted(pii_types_found),
        "gate_name": "gate0_pii_redaction",
    }


def _build_hash_evidence(
    splits: dict[str, list[dict]],
    out_dir: Path | None = None,
) -> dict:
    """Content hashes and split file checksums."""
    # Per-split content hash lists
    split_hashes: dict[str, list[str]] = {}
    for split_name, items in splits.items():
        split_hashes[split_name] = [get_convo_hash(item) for item in items]

    # Split file MD5 checksums (if files exist on disk)
    file_checksums: dict[str, str] = {}
    if out_dir:
        for name in SPLIT_NAMES:
            path = out_dir / f"{name}.jsonl"
            if path.exists():
                file_checksums[name] = _md5_file(path)

    return {
        "content_hashes": split_hashes,
        "file_checksums": file_checksums,
        "hash_algorithm": "md5",
    }


def _build_statistics_evidence(
    splits: dict[str, list[dict]],
) -> dict:
    """Aggregate statistics: counts, strata count, axis cardinalities."""
    total = sum(len(s) for s in splits.values())

    # Unique strata
    all_strata: set[str] = set()
    for items in splits.values():
        for item in items:
            all_strata.add(_stratum_key(item))

    # Axis cardinalities (unique values per axis)
    axis_cardinalities: dict[str, set[str]] = {name: set() for name in AXIS_NAMES}
    for items in splits.values():
        for item in items:
            axes = _extract_axes(item)
            for name in AXIS_NAMES:
                axis_cardinalities[name].add(axes[name])

    # Conversation length distribution
    conv_lengths = Counter()
    for items in splits.values():
        for item in items:
            axes = _extract_axes(item)
            conv_lengths[axes["conversation_length"]] += 1

    return {
        "total_records": total,
        "total_strata": len(all_strata),
        "axis_cardinalities": {
            name: len(values) for name, values in axis_cardinalities.items()
        },
        "axis_unique_values": {
            name: sorted(values) for name, values in axis_cardinalities.items()
        },
        "conversation_length_distribution": dict(conv_lengths),
    }


def _build_dvc_lineage() -> dict:
    """DVC lineage: available dataset versions."""
    try:
        versions = list_available_versions()
        return {
            "available_versions": versions,
            "tracked": True,
        }
    except Exception:
        return {
            "available_versions": [],
            "tracked": False,
        }


def _build_human_review() -> dict:
    """Human-review approval status."""
    return {
        "status": DEFAULT_REVIEW_STATUS,
        "approved_by": None,
        "approved_at": None,
        "notes": None,
    }


def _md5_file(path: Path) -> str:
    """Compute MD5 checksum of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_release_manifest(
    splits: dict[str, list[dict]],
    ratio: tuple[int, int, int],
    out_dir: Path | None = None,
    gates_passed: bool = True,
) -> dict:
    """Build a P13 release manifest with all required evidence.

    Produces a JSON-serializable dict containing evidence for all six P13
    categories (distribution, leakage, provenance, PII, hash, statistics)
    plus DVC lineage and human-review approval status.
    """
    return {
        "release_version": None,  # filled by caller after DVC tag
        "pipeline_version": "modern-dataset-provenance-v1",
        "created_at": _utc_now_iso(),
        "split_ratio": list(ratio),
        "gates_passed": gates_passed,
        "p13_evidence_categories": list(P13_EVIDENCE_CATEGORIES),
        "distribution": _build_distribution_evidence(splits),
        "leakage": _build_leakage_evidence(splits),
        "provenance": _build_provenance_evidence(splits),
        "pii": _build_pii_evidence(splits),
        "hash": _build_hash_evidence(splits, out_dir),
        "statistics": _build_statistics_evidence(splits),
        "dvc_lineage": _build_dvc_lineage(),
        "human_review": _build_human_review(),
    }
