"""Build a private, source-bound ablation of the full-corpus W&B release.

The original release and active jobs are read-only inputs. The output omits
the 1,522 `crisis_benchmark` examples, whose assistant target is the same task
instruction on every row, so a later matched run can test source-mix dilution.
No provider upload or training job is created.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "ai/data/curated/recovery/gold-20261017/train_master_gold.recovered_candidate.jsonl"
RELEASE_MANIFEST = ROOT / "docs/training/execution-2026-10-18/wandb-full-corpus-release-v2.json"
SOURCE_DIR = ROOT / "ai/data/curated/recovery/gold-20261017/wandb-full-corpus-20261018-v2"
OUTPUT = SOURCE_DIR / "crisis-benchmark-ablation-v2"
SOURCE_LABEL = "crisis_benchmark"
PLACEHOLDER_TARGET = (
    "Assess this input for risk level, propose safety resources, and "
    "generate a compassionate, non-harmful response."
)
EXPECTED_CANDIDATE_SHA = "4f0fd4c5fcfdffeb5434057ff4e122356256f9775540a3b41c1421a75e507663"
EXPECTED_RELEASE_SHA = "138045ba6fb263e0d68bb42c3e8d0a7e3bf7314c3fc79397ecfd805a2c980a76"
EXPECTED_CANDIDATE_ROWS = 190139
EXPECTED_EXCLUDED_ROWS = 1522
EXPECTED_RETAINED_ROWS = 188617
EXPECTED_EXCLUDED_BY_SPLIT = {"train": 1374, "validation": 79, "test": 69}
EXPECTED_SPLIT_COUNTS = {"train": 169793, "validation": 9587, "test": 9237}
SPLITS = ("train", "validation", "test")


@dataclass
class BuildState:
    candidate_rows: int = 0
    retained_rows: int = 0
    excluded_rows: int = 0
    source_counts: Counter[str] = field(default_factory=Counter)
    split_counts: Counter[str] = field(default_factory=Counter)
    excluded_by_split: Counter[str] = field(default_factory=Counter)
    group_sets: dict[str, set[str]] = field(default_factory=lambda: {split: set() for split in SPLITS})
    rowrefs: list[dict[str, Any]] = field(default_factory=list)
    excluded_refs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BuildContext:
    state: BuildState
    source_streams: dict[str, Any]
    output_streams: dict[str, Any]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return sha256(payload)


def write_private(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)
    os.chmod(path, 0o600)


def verify_release_file(path: Path, expected_sha: str, expected_rows: int, label: str) -> None:
    if hash_file(path) != expected_sha:
        raise ValueError(f"source release {label} hash mismatch")
    with path.open(encoding="utf-8") as stream:
        actual_rows = sum(1 for line in stream if line.strip())
    if actual_rows != expected_rows:
        raise ValueError(f"source release {label} row-count mismatch")


def validate_inputs() -> dict[str, Any]:
    if hash_file(CANDIDATE) != EXPECTED_CANDIDATE_SHA:
        raise ValueError("pinned candidate hash mismatch")
    if hash_file(SOURCE_DIR / "train.jsonl") != EXPECTED_RELEASE_SHA:
        raise ValueError("pinned source training payload hash mismatch")
    release = json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(release, dict):
        raise ValueError("source release manifest must be an object")
    if release.get("release_id") != "wandb-full-corpus-20261018-v2":
        raise ValueError("unexpected source release id")
    if release.get("source_candidate", {}).get("sha256") != EXPECTED_CANDIDATE_SHA:
        raise ValueError("source release candidate identity mismatch")
    if release.get("source_candidate", {}).get("row_count") != EXPECTED_CANDIDATE_ROWS:
        raise ValueError("source release candidate row-count mismatch")
    for split in SPLITS:
        partition = release["partitions"][split]
        verify_release_file(
            ROOT / partition["path"],
            partition["sha256"],
            partition["row_count"],
            f"{split} payload",
        )
    internal = release["internal_manifest"]
    verify_release_file(
        ROOT / internal["path"], internal["sha256"], internal["row_count"], "internal manifest"
    )
    rowrefs = release["evidence"]["group_disjointness"]
    verify_release_file(
        ROOT / rowrefs["evidence_path"],
        rowrefs["evidence_sha256"],
        internal["row_count"],
        "rowrefs",
    )
    return release


def record_exclusion(state: BuildState, row_number: int, row_hash: str, reference: dict[str, Any]) -> None:
    split = reference["split"]
    state.excluded_rows += 1
    state.excluded_by_split[split] += 1
    state.excluded_refs.append(
        {
            "candidate_row_number_1based": row_number,
            "candidate_row_sha256": row_hash,
            "content_group_sha256": reference["content_group_sha256"],
            "split": split,
            "exclusion_reason": "source_ablation_identical_task_instruction_target",
        }
    )


def record_retained(
    context: BuildContext,
    row_number: int,
    row_hash: str,
    reference: dict[str, Any],
    payload_line: str,
) -> None:
    state = context.state
    split = reference["split"]
    source = reference["source_label"]
    group = reference["content_group_sha256"]
    context.output_streams[split].write(payload_line)
    state.retained_rows += 1
    state.split_counts[split] += 1
    state.source_counts[source] += 1
    state.group_sets[split].add(group)
    state.rowrefs.append(
        {
            "candidate_row_number_1based": row_number,
            "candidate_row_sha256": row_hash,
            "content_group_sha256": group,
            "split": split,
            "source_label": source,
        }
    )


def process_release_row(
    row_number: int,
    candidate_line: str,
    reference: dict[str, Any],
    context: BuildContext,
) -> None:
    state = context.state
    row = json.loads(candidate_line)
    raw = candidate_line.encode("utf-8")
    row_hash = sha256(raw[:-1] if raw.endswith(b"\n") else raw)
    if reference.get("candidate_row_sha256") != row_hash:
        raise ValueError(f"candidate row hash mismatch at row {row_number}")
    source = reference.get("source_label")
    if row.get("source") != source:
        raise ValueError(f"candidate source label mismatch at row {row_number}")
    split = reference.get("split")
    if split not in SPLITS:
        raise ValueError(f"invalid split in internal manifest at row {row_number}")
    payload_line = context.source_streams[split].readline()
    if not payload_line:
        raise ValueError(f"source payload ended before candidate row {row_number}")
    messages = json.loads(payload_line).get("messages", [])
    target = next(
        (item.get("content", "") for item in reversed(messages) if item.get("role") == "assistant"),
        "",
    )
    if source == SOURCE_LABEL:
        if target != PLACEHOLDER_TARGET:
            raise ValueError(f"unexpected source target at candidate row {row_number}")
        record_exclusion(state, row_number, row_hash, reference)
        return
    record_retained(context, row_number, row_hash, reference, payload_line)


def render_ablation(staging: Path) -> BuildState:
    state = BuildState()
    internal_path = SOURCE_DIR / "manifest_internal.jsonl"
    internal_iter = iter(json.loads(line) for line in internal_path.open(encoding="utf-8"))
    current_internal = next(internal_iter, None)
    state = BuildState()
    output_streams = {split: (staging / f"{split}.jsonl").open("w", encoding="utf-8") for split in SPLITS}
    source_streams = {split: (SOURCE_DIR / f"{split}.jsonl").open(encoding="utf-8") for split in SPLITS}
    context = BuildContext(state, source_streams, output_streams)
    try:
        with CANDIDATE.open(encoding="utf-8") as candidate_stream:
            for row_number, line in enumerate(candidate_stream, start=1):
                state.candidate_rows += 1
                if current_internal is None:
                    continue
                current_number = current_internal["candidate_row_number_1based"]
                if current_number > row_number:
                    continue
                if current_number < row_number:
                    raise ValueError("internal manifest row order is not strictly increasing")
                process_release_row(
                    row_number, line, current_internal, context
                )
                current_internal = next(internal_iter, None)
        if current_internal is not None:
            raise ValueError("internal manifest has unmatched candidate rows")
        _flush_streams(output_streams)
        _verify_streams_consumed(source_streams)
    finally:
        _close_streams(output_streams, source_streams)
    return state


def _flush_streams(streams: dict[str, Any]) -> None:
    for stream in streams.values():
        stream.flush()
        os.fsync(stream.fileno())


def _verify_streams_consumed(streams: dict[str, Any]) -> None:
    for split, stream in streams.items():
        if stream.read(1):
            raise ValueError(f"source {split} payload has unconsumed rows")


def _close_streams(*groups: dict[str, Any]) -> None:
    for streams in groups:
        for stream in streams.values():
            stream.close()


def validate_state(state: BuildState) -> dict[str, int]:
    totals = (state.candidate_rows, state.excluded_rows, state.retained_rows)
    expected = (EXPECTED_CANDIDATE_ROWS, EXPECTED_EXCLUDED_ROWS, EXPECTED_RETAINED_ROWS)
    if totals != expected:
        raise ValueError(f"unexpected totals: actual={totals}, expected={expected}")
    if dict(state.excluded_by_split) != EXPECTED_EXCLUDED_BY_SPLIT:
        raise ValueError(f"unexpected excluded split counts: {dict(state.excluded_by_split)}")
    if dict(state.split_counts) != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"unexpected retained split counts: {dict(state.split_counts)}")
    intersections = {
        "train_validation": len(state.group_sets["train"] & state.group_sets["validation"]),
        "train_test": len(state.group_sets["train"] & state.group_sets["test"]),
        "validation_test": len(state.group_sets["validation"] & state.group_sets["test"]),
    }
    if any(intersections.values()):
        raise ValueError(f"source group overlap after ablation: {intersections}")
    return intersections


def validate_repairs_retained(release: dict[str, Any], state: BuildState) -> dict[str, Any]:
    repair_manifest = release["repairs"]
    repair_path = ROOT / repair_manifest["path"]
    verify_release_file(
        repair_path,
        repair_manifest["sha256"],
        repair_manifest["row_count"],
        "repair overlay",
    )
    repairs = [json.loads(line) for line in repair_path.open(encoding="utf-8")]
    retained = {row["candidate_row_number_1based"]: row["candidate_row_sha256"] for row in state.rowrefs}
    repair_rowrefs = {
        row["candidate_row_number_1based"]: row["candidate_row_sha256"] for row in repairs
    }
    if len(repair_rowrefs) != len(repairs) or not repair_rowrefs:
        raise ValueError("repair overlay contains missing or duplicate candidate row references")
    if any(retained.get(row_number) != row_hash for row_number, row_hash in repair_rowrefs.items()):
        raise ValueError("repaired candidate rows are missing or changed in ablation output")
    return {
        "row_count": len(repairs),
        "repair_overlay_sha256": repair_manifest["sha256"],
        "all_repaired_candidate_rows_retained": True,
    }


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    lines = (json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    return "".join(lines).encode()


def build_manifest(
    staging: Path,
    release: dict[str, Any],
    state: BuildState,
    intersections: dict[str, int],
    repair_validation: dict[str, Any],
) -> dict[str, Any]:
    partitions = {}
    for split in SPLITS:
        path = staging / f"{split}.jsonl"
        os.chmod(path, 0o600)
        partitions[split] = {
            "path": str((OUTPUT / path.name).relative_to(ROOT)),
            "row_count": state.split_counts[split],
            "source_group_count": len(state.group_sets[split]),
            "size_bytes": path.stat().st_size,
            "sha256": hash_file(path),
            "format": "weighted_messages_jsonl_v1",
        }
    rowrefs_path = staging / "rowrefs.jsonl"
    excluded_path = staging / "excluded-rowrefs.jsonl"
    return {
        "schema_version": 1,
        "release_id": "wandb-full-corpus-20261018-v2-ablate-crisis-benchmark",
        "status": "local_research_ablation_not_uploaded",
        "upload_gate": False,
        "provider_candidate": "wandb",
        "scope": "research-only source-family ablation; no clinical or production readiness",
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_release": {
            "release_id": release["release_id"],
            "manifest_path": str(RELEASE_MANIFEST.relative_to(ROOT)),
            "manifest_sha256": hash_file(RELEASE_MANIFEST),
            "candidate_sha256": EXPECTED_CANDIDATE_SHA,
            "candidate_rows": state.candidate_rows,
            "repair_overlay_sha256": release["repairs"]["sha256"],
        },
        "hypothesis": (
            "Removing a source family whose assistant target is one identical generic instruction "
            "may improve target diversity and tool-use behavior."
        ),
        "ablation": {
            "source_label": SOURCE_LABEL,
            "excluded_rows": state.excluded_rows,
            "excluded_by_split": dict(state.excluded_by_split),
            "assistant_target_sha256": sha256(PLACEHOLDER_TARGET.encode()),
            "all_excluded_targets_verified_identical": True,
        },
        "retained_rows": state.retained_rows,
        "retained_source_counts": dict(sorted(state.source_counts.items())),
        "partitions": partitions,
        "rowrefs": {
            "path": str((OUTPUT / rowrefs_path.name).relative_to(ROOT)),
            "rows": state.retained_rows,
            "sha256": hash_file(rowrefs_path),
        },
        "excluded_rowrefs": {
            "path": str((OUTPUT / excluded_path.name).relative_to(ROOT)),
            "rows": state.excluded_rows,
            "sha256": hash_file(excluded_path),
        },
        "validation": {
            "candidate_hash_verified": True,
            "source_train_payload_hash_verified": True,
            "candidate_row_hashes_verified": True,
            "source_label_bindings_verified": True,
            "excluded_targets_identical": True,
            "repairs_retained": repair_validation,
            "group_intersections": intersections,
            "provider_upload_or_job_created": False,
        },
        "limitations": [
            "This removes the entire crisis_benchmark source family; it does not identify "
            "the causal effect of the target alone.",
            "Other source rights/privacy/transfer questions remain governed by the source decision ledger.",
            (
                "The 50 generated continuations in the source release are structurally retained, "
                "but are not clinically validated by this ablation builder."
            ),
            "The ablation is a local experiment candidate and is not approved for provider upload.",
        ],
    }


def build_ablation() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {OUTPUT}")
    release = validate_inputs()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".crisis-ablation-", dir=OUTPUT.parent) as temporary:
        staging = Path(temporary)
        os.chmod(staging, 0o700)
        state = render_ablation(staging)
        intersections = validate_state(state)
        repair_validation = validate_repairs_retained(release, state)
        write_private(staging / "rowrefs.jsonl", jsonl_bytes(state.rowrefs))
        write_private(staging / "excluded-rowrefs.jsonl", jsonl_bytes(state.excluded_refs))
        manifest = build_manifest(staging, release, state, intersections, repair_validation)
        write_private(staging / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
        os.rename(staging, OUTPUT)
        os.chmod(OUTPUT, 0o700)
    return manifest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    manifest = build_ablation()
    logging.info(
        json.dumps(
            {
                "release_id": manifest["release_id"],
                "status": manifest["status"],
                "retained_rows": manifest["retained_rows"],
                "excluded_rows": manifest["ablation"]["excluded_rows"],
                "excluded_by_split": manifest["ablation"]["excluded_by_split"],
                "split_rows": {split: value["row_count"] for split, value in manifest["partitions"].items()},
                "output": str(OUTPUT.relative_to(ROOT)),
                "manifest_sha256": hash_file(OUTPUT / "manifest.json"),
                "provider_job_created": False,
            }
        )
    )


if __name__ == "__main__":
    main()
