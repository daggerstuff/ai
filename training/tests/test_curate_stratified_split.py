"""Characterization tests for the curate_pipeline split path (PIX-4584).

Pins the contract that the live curation pipeline delegates its record ->
split assignment to the canonical stratified splitter
(``training.dataset_splitter_stratified.stratified_split`` +
``integrity_gates``) instead of the legacy per-record hash bucket:

1. The live ``run_pipeline`` split path groups records by source family —
   all records sharing ``(source, user_id)`` land in the same split.
2. The split ratios stay within ±2pp of ``DEFAULT_RATIO`` (70/15/15).
3. ``integrity_gates`` runs on the split result and its verdict is
   recorded in ``PipelineStats.split_integrity``.
4. A deliberately leaky split (same conversation in two splits, or the
   same source family in two splits) fails ``integrity_gates``.

The fixture families are selected by hash search so the deterministic
family-key hashing yields exactly 70/15/15 whole families — the ratio
assertion is exact, not statistical.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from training.curate_pipeline import run_pipeline
from training.dataset_splitter_stratified import DEFAULT_RATIO, integrity_gates

# Fixture constants
_SOURCE = "strat_test_src"
_RECORDS_PER_FAMILY = 5
_FAMILIES_PER_BUCKET = {0: 70, 1: 15, 2: 15}  # train, val, test


def _family_bucket(family_key: str) -> int:
    """Mirror the splitter's family hash bucketing."""
    bucket = int(hashlib.md5(family_key.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < DEFAULT_RATIO[0]:
        return 0
    if bucket < DEFAULT_RATIO[0] + DEFAULT_RATIO[1]:
        return 1
    return 2


def _select_family_ids() -> dict[int, list[str]]:
    """Deterministically pick user_ids whose family keys hash 70/15/15."""
    picked: dict[int, list[str]] = {0: [], 1: [], 2: []}
    index = 0
    while any(
        len(ids) < count
        for ids, count in zip(
            picked.values(), _FAMILIES_PER_BUCKET.values(), strict=True
        )
    ):
        user_id = f"user_{index}"
        bucket = _family_bucket(f"{_SOURCE}::{user_id}")
        if len(picked[bucket]) < _FAMILIES_PER_BUCKET[bucket]:
            picked[bucket].append(user_id)
        index += 1
    return picked


def _build_input_records() -> list[dict[str, Any]]:
    """Build unique 2-message records for every selected family."""
    records: list[dict[str, Any]] = []
    for family_ids in _select_family_ids().values():
        for user_id in family_ids:
            for record_index in range(_RECORDS_PER_FAMILY):
                records.append(
                    {
                        "source": _SOURCE,
                        "user_id": user_id,
                        "task_type": "sft",
                        "messages": [
                            {
                                "role": "user",
                                "content": (
                                    f"[{user_id}] Feelings check {record_index}: "
                                    "I have been having a difficult week and wanted "
                                    "to talk about it with someone."
                                ),
                            },
                            {
                                "role": "assistant",
                                "content": (
                                    f"[{user_id}] Response {record_index}: thank you "
                                    "for sharing that with me — tell me more about "
                                    "what the week has been like."
                                ),
                            },
                        ],
                    }
                )
    return records


def _write_input(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    input_path = tmp_path / "input.jsonl"
    with open(input_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return input_path


def _read_shard(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _family_of(record: dict[str, Any]) -> str:
    """Recover the fixture family id from the first user message content."""
    for message in record.get("messages", []):
        content = str(message.get("content", ""))
        if content.startswith("["):
            return str(content[1:].split("]", 1)[0])
    return "unknown"


def test_curation_split_path_uses_stratification(tmp_path: Path) -> None:
    """The live split path groups by source family and honors DEFAULT_RATIO.

    Red against the legacy ``assign_split`` hash bucket: that path assigns
    each record independently by content hash, so records of the same
    source family scatter across splits.
    """
    records = _build_input_records()
    input_path = _write_input(tmp_path, records)
    output_dir = tmp_path / "out"

    stats = run_pipeline(
        str(input_path),
        str(output_dir),
        dry_run=False,
        stage1_filters=False,
    )

    shards = {
        name: _read_shard(output_dir / "sft_chatml" / f"{name}.jsonl")
        for name in ("train", "val", "test")
    }
    total = sum(len(items) for items in shards.values())
    assert total == len(records), "every staged record must be written exactly once"

    # 1. Source-family grouping: a family never spans two splits.
    family_to_splits: dict[str, set[str]] = defaultdict(set)
    for name, items in shards.items():
        for record in items:
            family_to_splits[_family_of(record)].add(name)
    split_families = {
        family: splits for family, splits in family_to_splits.items()
        if len(splits) > 1
    }
    assert not split_families, (
        f"source families split across shards: {list(split_families)[:5]}"
    )

    # 2. Ratios within ±2pp of DEFAULT_RATIO (exact 70/15/15 by construction).
    tolerance_pp = 2
    for index, name in enumerate(("train", "val", "test")):
        actual_pp = 100 * len(shards[name]) / total
        assert abs(actual_pp - DEFAULT_RATIO[index]) <= tolerance_pp, (
            f"{name} ratio {actual_pp:.2f}pp outside ±{tolerance_pp}pp "
            f"of {DEFAULT_RATIO[index]}pp"
        )

    # 3. integrity_gates ran on the split result and passed.
    gate_result = stats.split_integrity.get("sft_chatml")
    assert gate_result is not None, "split path must record integrity_gates verdict"
    assert gate_result["passed"] is True, gate_result["issues"]

def test_split_path_emits_release_manifest(tmp_path: Path) -> None:
    """The split path writes release_manifest.json next to the master shards.

    Pins the P13 emission contract (PIX-4584): the pipeline emits the release
    manifest through the canonical ``build_release_manifest`` builder, with
    all eight evidence sections, human review left pending, and per-shard
    file checksums matching the shards written by the same run.
    """
    records = _build_input_records()
    input_path = _write_input(tmp_path, records)
    output_dir = tmp_path / "out"

    run_pipeline(str(input_path), str(output_dir), dry_run=False, stage1_filters=False)

    manifest_path = output_dir / "sft_chatml" / "release_manifest.json"
    assert manifest_path.exists(), "split path must emit release_manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest: dict[str, Any] = json.load(f)

    for section in (
        "distribution",
        "leakage",
        "provenance",
        "pii",
        "hash",
        "statistics",
        "dvc_lineage",
        "human_review",
    ):
        assert section in manifest, f"manifest missing evidence section: {section}"
    assert manifest["human_review"]["status"] == "pending"
    assert manifest["split_ratio"] == list(DEFAULT_RATIO)

    # Per-shard checksums must describe the shards this run wrote.
    for name in ("train", "val", "test"):
        shard = output_dir / "sft_chatml" / f"{name}.jsonl"
        digest = hashlib.md5(shard.read_bytes()).hexdigest()
        assert manifest["hash"]["file_checksums"][name] == digest


def test_leaky_split_fails_integrity_gates() -> None:
    """A split that shares a conversation across splits fails the gates."""
    record: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": "Shared conversation content for the leak test."},
            {"role": "assistant", "content": "Shared response content for the leak test."},
        ],
        "metadata": {"source_family": "leaky_family", "source_id": "shared_id"},
    }
    leaky_by_hash: dict[str, list[dict[str, Any]]] = {
        "train": [record],
        "val": [dict(record)],
        "test": [],
    }
    passed, issues = integrity_gates(leaky_by_hash)
    assert passed is False
    assert any("hash-disjoint" in issue for issue in issues)
    assert any("source-family" in issue for issue in issues)


def test_family_leak_across_splits_fails_integrity_gates() -> None:
    """Two different conversations from one family in two splits fail."""
    family_meta = {"source_family": "leaky_family", "source_id": "shared_id"}
    first: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": "First family conversation content."},
            {"role": "assistant", "content": "First family response content."},
        ],
        "metadata": dict(family_meta),
    }
    second: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": "Second family conversation content."},
            {"role": "assistant", "content": "Second family response content."},
        ],
        "metadata": dict(family_meta),
    }
    leaky_by_family: dict[str, list[dict[str, Any]]] = {
        "train": [first],
        "val": [second],
        "test": [],
    }
    passed, issues = integrity_gates(leaky_by_family)
    assert passed is False
    assert any("source-family" in issue for issue in issues)

def _make_record(index: int, language: str) -> dict[str, Any]:
    """A unique record in the stratum identified by ``language``."""
    return {
        "messages": [
            {"role": "user", "content": f"User message number {index} for the gates test."},
            {"role": "assistant", "content": f"Assistant reply number {index} for the gates test."},
        ],
        "metadata": {
            "source_family": f"gates_family_{index}",
            "source_id": f"gates_id_{index}",
            "language": language,
        },
    }

def _balanced_stratum(language: str, start: int, count: int) -> dict[str, list[dict[str, Any]]]:
    """``count`` records split exactly 70/15/15 by position."""
    return {
        "train": [_make_record(start + i, language) for i in range(count * 7 // 10)],
        "val": [
            _make_record(start + count * 7 // 10 + i, language)
            for i in range(count * 15 // 100)
        ],
        "test": [
            _make_record(start + count * 7 // 10 + count * 15 // 100 + i, language)
            for i in range(count - count * 7 // 10 - count * 15 // 100)
        ],
    }

def test_small_stratum_noise_deviation_passes_gates() -> None:
    """A correct split passes even when a small stratum wobbles beyond 2pp.

    Red before the noise-aware balance tolerance: with a fixed ±2pp check,
    a 15-record stratum whose hash assignment landed 12/1/2 fails the gate
    even though the split is correct — pure binomial noise of the family
    hash. The gate must widen its tolerance for strata too small for a
    2pp check to be meaningful.
    """
    splits = _balanced_stratum("en", 0, 700)
    # Small stratum with a noise-level deviation: 12/1/2 (80/6.7/13.3).
    splits["train"].extend(_make_record(1000 + i, "zz") for i in range(12))
    splits["val"].append(_make_record(1012, "zz"))
    splits["test"].extend(_make_record(1013 + i, "zz") for i in range(2))

    passed, issues = integrity_gates(splits)
    assert passed, f"noise-level small-stratum deviation must not fail: {issues}"

def test_small_stratum_gross_imbalance_still_fails_gates() -> None:
    """A small stratum concentrated in one split is still caught.

    The noise-aware tolerance must not become a blanket exemption: a
    60-record stratum placed entirely in train deviates far beyond the
    noise band and must fail the balance gate.
    """
    splits = _balanced_stratum("en", 0, 700)
    splits["train"].extend(_make_record(2000 + i, "zz") for i in range(60))

    passed, issues = integrity_gates(splits)
    assert passed is False
    assert any("domain balance" in issue for issue in issues)
