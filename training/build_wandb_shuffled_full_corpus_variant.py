#!/usr/bin/env python3
"""Build a deterministic row-shuffled copy of the frozen W&B v4 train split.

The output keeps the exact training rows and split membership unchanged while
shuffling row order and writing aligned hash-only provenance sidecars. It does
not upload data or create a remote model/job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "docs/training/execution-2026-10-19/wandb-full-corpus-current-master-20261019-v4.json"
DEFAULT_OUTPUT = ROOT / "ai/training/output/wandb_full_corpus_v4_shuffled_20261020"
SOURCE_LINEAGE_DIR = (
    ROOT
    / "ai/data/curated/recovery/gold-20261017/"
    "wandb-full-corpus-current-master-20261019-v4"
)
ROWREFS_PATH = SOURCE_LINEAGE_DIR / "rowrefs.jsonl"
INTERNAL_PATH = SOURCE_LINEAGE_DIR / "manifest_internal.jsonl"


class ReleaseError(ValueError):
    """Raised when the source release or row alignment is invalid."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_private(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    path.write_bytes(content)
    os.chmod(path, 0o600)


def load_aligned_rows(train_path: Path, row_count: int) -> list[tuple[bytes, dict[str, Any], dict[str, Any]]]:
    aligned = []
    previous_candidate_row = 0
    with (
        train_path.open("rb") as payload_stream,
        ROWREFS_PATH.open("rb") as rowrefs_stream,
        INTERNAL_PATH.open("rb") as internal_stream,
    ):
        for index, (rowref_line, internal_line) in enumerate(zip(rowrefs_stream, internal_stream, strict=True)):
            rowref = json.loads(rowref_line)
            internal = json.loads(internal_line)
            if rowref.get("candidate_row_number_1based") != internal.get("candidate_row_number_1based"):
                raise ReleaseError(f"candidate row identity mismatch at internal row {index}")
            if rowref.get("candidate_row_sha256") != internal.get("candidate_row_sha256"):
                raise ReleaseError(f"candidate hash mismatch at internal row {index}")
            if rowref.get("content_group_sha256") != internal.get("content_group_sha256"):
                raise ReleaseError(f"content-group mismatch at internal row {index}")
            split = internal.get("split")
            if split != rowref.get("provisional_split"):
                raise ReleaseError(f"split mismatch at internal row {index}")
            if split != "train":
                continue
            candidate_row = internal.get("candidate_row_number_1based")
            if not isinstance(candidate_row, int) or candidate_row <= previous_candidate_row:
                raise ReleaseError("train rows are not in strictly increasing candidate order")
            previous_candidate_row = candidate_row
            try:
                payload_line = next(payload_stream)
            except StopIteration as exc:
                raise ReleaseError("train payload ended before its aligned row references") from exc
            rowref["payload_sha256"] = sha256_bytes(payload_line.rstrip(b"\n"))
            aligned.append((payload_line, rowref, internal))
        if next(payload_stream, None) is not None:
            raise ReleaseError("train payload contains more rows than train row references")
    if len(aligned) != row_count:
        raise ReleaseError("train payload and aligned source row-reference counts disagree")
    return aligned


def serialize_shuffled(
    aligned: list[tuple[bytes, dict[str, Any], dict[str, Any]]], seed: int
) -> tuple[bytearray, bytearray, bytearray]:
    order = list(range(len(aligned)))
    random.Random(seed).shuffle(order)
    shuffled_payload = bytearray()
    shuffled_rowrefs = bytearray()
    shuffled_internal = bytearray()
    for output_index, source_index in enumerate(order, start=1):
        payload_line, rowref, internal = aligned[source_index]
        rowref["shuffle_position_1based"] = output_index
        internal["shuffle_position_1based"] = output_index
        shuffled_payload.extend(payload_line)
        shuffled_rowrefs.extend(json.dumps(rowref, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        shuffled_internal.extend(json.dumps(internal, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    return shuffled_payload, shuffled_rowrefs, shuffled_internal


def write_release(output: Path, payload: bytes, rowrefs: bytes, internal: bytes, metadata: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output, 0o700)
    variant_manifest = {
        **metadata,
        "status": "prepared_locally_not_uploaded",
        "clinical_or_production_readiness": False,
        "provider_job_created": False,
    }
    write_private(output / "trajectories.jsonl", payload)
    write_private(output / "rowrefs.jsonl", rowrefs)
    write_private(output / "manifest_internal.jsonl", internal)
    write_private(output / "metadata.json", (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode())
    write_private(
        output / "release-manifest.json",
        (json.dumps(variant_manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    write_private(
        output / "release-manifest.sha256",
        (sha256_file(output / "release-manifest.json") + "\n").encode(),
    )


def build(manifest_path: Path, output: Path, seed: int) -> dict[str, Any]:
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("status") != "approved_for_research_training" or manifest.get("upload_gate") is not True:
        raise ReleaseError("source manifest is not the frozen research-training release")
    if manifest.get("release_id") != "wandb-full-corpus-current-master-20261019-v4":
        raise ReleaseError("source manifest is not the expected v4 release")
    source_candidate_sha = manifest.get("source_candidate_sha256")
    train = manifest["partitions"]["train"]
    train_path = (ROOT / train["path"]).resolve()
    if sha256_file(train_path) != train["sha256"]:
        raise ReleaseError("source train payload SHA-256 mismatch")

    aligned = load_aligned_rows(train_path, train["row_count"])
    shuffled_payload, shuffled_rowrefs, shuffled_internal = serialize_shuffled(aligned, seed)

    if sha256_bytes(shuffled_payload) == train["sha256"]:
        raise ReleaseError("shuffled payload unexpectedly matches the ordered source")
    if output.exists() and any(output.iterdir()):
        raise ReleaseError("output directory already exists and is not empty")
    payload_sha = sha256_bytes(shuffled_payload)
    rowrefs_sha = sha256_bytes(shuffled_rowrefs)
    internal_sha = sha256_bytes(shuffled_internal)
    manifest_sha = sha256_bytes(manifest_bytes)
    release_id = f"wandb-full-corpus-v4-shuffled-seed-{seed}"
    metadata = {
        "schema_version": 1,
        "release_id": release_id,
        "base_model": "Qwen/Qwen3.6-35B-A3B",
        "assistant_turns_config": "all",
        "loss_mask": "assistant turns only",
        "source": "deterministically shuffled row-order variant of the frozen v4 provider-rendered train split",
        "source_candidate_sha256": source_candidate_sha,
        "parent_release_id": manifest["release_id"],
        "parent_manifest_sha256": manifest_sha,
        "parent_train_payload_sha256": train["sha256"],
        "train_payload_sha256": payload_sha,
        "train_row_count": len(aligned),
        "total_len": len(aligned),
        "rowrefs_sha256": rowrefs_sha,
        "internal_rowrefs_sha256": internal_sha,
        "shuffle": {"algorithm": "Python random.Random(seed).shuffle", "seed": seed},
        "training_config": {"assistant_turns": "all", "batch_size": "auto", "learning_rate": 0.0001},
    }
    metadata["validation_payload_sha256"] = manifest["partitions"]["validation"]["sha256"]
    metadata["validation_row_count"] = manifest["partitions"]["validation"]["row_count"]
    metadata["test_payload_sha256"] = manifest["partitions"]["test"]["sha256"]
    metadata["test_row_count"] = manifest["partitions"]["test"]["row_count"]
    write_release(output, bytes(shuffled_payload), bytes(shuffled_rowrefs), bytes(shuffled_internal), metadata)
    return {
        "release_id": release_id,
        "source_manifest_sha256": manifest_sha,
        "source_train_sha256": train["sha256"],
        "shuffled_train_sha256": payload_sha,
        "rowrefs_sha256": rowrefs_sha,
        "internal_rowrefs_sha256": internal_sha,
        "train_rows": len(aligned),
        "shuffle_seed": seed,
        "output_dir": str(output),
        "provider_job_created": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    try:
        result = build(args.manifest, args.output, args.seed)
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ReleaseError) as exc:
        sys.stderr.write(f"build-wandb-shuffled-release: {type(exc).__name__}: {exc}\n")
        return 2
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
