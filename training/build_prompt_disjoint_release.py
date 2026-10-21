"""Re-split the frozen research corpus by normalized user prompt.

This creates local-only split variants from the existing frozen v4 payloads.
It preserves every released row and changes only split assignment; one variant
retains the previously validated zero-weight system context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASE_RELEASE = ROOT / "ai/data/curated/recovery/gold-20261017/wandb-full-corpus-current-master-20261019-v4"
PRESERVE_RELEASE = (
    ROOT / "ai/data/curated/recovery/gold-20261017/wandb-full-corpus-current-master-20261019-v4-preserve-system"
)
OUT_ROOT = ROOT / "ai/data/curated/recovery/gold-20261017"
SPLITS = ("train", "validation", "test")
TRAIN_BUCKET_LIMIT = 90
VALIDATION_BUCKET_LIMIT = 95


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_user_prompt(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("payload row has no messages array")
    user_turns = [message.get("content", "") for message in messages if message.get("role") == "user"]
    prompt = " ".join("\n".join(user_turns).casefold().split())
    if not prompt:
        raise ValueError("payload row has no non-empty user prompt")
    return prompt


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(row)
    return rows


def verify_pair(base_rows: list[dict[str, Any]], preserve_rows: list[dict[str, Any]], label: str) -> None:
    if len(base_rows) != len(preserve_rows):
        raise ValueError(f"{label} row counts differ between system variants")
    for index, (base, preserve) in enumerate(zip(base_rows, preserve_rows, strict=True), start=1):
        base_messages = base.get("messages")
        preserve_messages = preserve.get("messages")
        if not isinstance(base_messages, list) or not isinstance(preserve_messages, list):
            raise ValueError(f"{label} row {index} has invalid messages")
        projection = [message for message in preserve_messages if message.get("role") != "system"]
        if projection != base_messages:
            raise ValueError(f"{label} row {index} differs outside system context")
        if any(message.get("weight") != 0 for message in preserve_messages if message.get("role") == "system"):
            raise ValueError(f"{label} row {index} gives system context nonzero loss weight")


def assigned_split(prompt_hash: str) -> str:
    bucket = int(prompt_hash[:8], 16) % 100
    if bucket < TRAIN_BUCKET_LIMIT:
        return "train"
    if bucket < VALIDATION_BUCKET_LIMIT:
        return "validation"
    return "test"


def load_variants() -> dict[str, dict[str, list[dict[str, Any]]]]:
    variants: dict[str, dict[str, list[dict[str, Any]]]] = {
        "system-stripped": {},
        "preserve-system": {},
    }
    for split in SPLITS:
        variants["system-stripped"][split] = read_rows(BASE_RELEASE / f"{split}.jsonl")
        preserve_path = PRESERVE_RELEASE / split / "trajectories.jsonl"
        variants["preserve-system"][split] = read_rows(preserve_path)
        verify_pair(variants["system-stripped"][split], variants["preserve-system"][split], split)
    return variants


def plan_assignments(
    base_data: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, str], Counter[str], list[dict[str, Any]]]:
    prompt_groups: dict[str, str] = {}
    old_split_counts: Counter[str] = Counter()
    internal_rows: list[dict[str, Any]] = []
    for old_split in SPLITS:
        for index, row in enumerate(base_data[old_split]):
            prompt_hash = hashlib.sha256(normalized_user_prompt(row).encode()).hexdigest()
            destination = assigned_split(prompt_hash)
            if prompt_groups.setdefault(prompt_hash, destination) != destination:
                raise ValueError("prompt hash was assigned to multiple splits")
            old_split_counts[old_split] += 1
            internal_rows.append(
                {
                    "source_split": old_split,
                    "row_index_1based_within_source_split": index + 1,
                    "prompt_group_sha256": prompt_hash,
                    "destination_split": destination,
                    "row_sha256": canonical_hash(row),
                }
            )
    return prompt_groups, old_split_counts, internal_rows


def write_variant(
    prefix: str,
    variant: str,
    source: dict[str, list[dict[str, Any]]],
    base_data: dict[str, list[dict[str, Any]]],
    old_split_counts: Counter[str],
) -> tuple[dict[str, Path], Counter[str], dict[str, Any]]:
    variant_dir = OUT_ROOT / f"{prefix}-{variant}"
    variant_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    outputs = {split: variant_dir / f"{split}.jsonl" for split in SPLITS}
    for path in outputs.values():
        path.touch(mode=0o600, exist_ok=False)
    row_counts: Counter[str] = Counter()
    for old_split in SPLITS:
        for index, row in enumerate(source[old_split]):
            prompt_hash = hashlib.sha256(normalized_user_prompt(base_data[old_split][index]).encode()).hexdigest()
            destination = assigned_split(prompt_hash)
            with outputs[destination].open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            row_counts[destination] += 1

    files: dict[str, Any] = {}
    for split, path in outputs.items():
        os.chmod(path, 0o600)
        files[split] = {
            "path": str(path.relative_to(ROOT)),
            "rows": row_counts[split],
            "sha256": file_hash(path),
            "bytes": path.stat().st_size,
        }
    details = {
        "directory": str(variant_dir.relative_to(ROOT)),
        "files": files,
        "rows_by_original_split": dict(old_split_counts),
    }
    return outputs, row_counts, details


def verify_outputs(
    outputs: dict[str, dict[str, Path]],
    row_counts: dict[str, Counter[str]],
    details: dict[str, dict[str, Any]],
) -> tuple[dict[str, str], Counter[str]]:
    assignments: dict[str, str] = {}
    projected_rows: dict[str, Counter[str]] = {
        "system-stripped": Counter(),
        "preserve-system": Counter(),
    }
    observed_rows: Counter[str] = Counter()
    for variant, split_paths in outputs.items():
        for split, path in split_paths.items():
            rows = read_rows(path)
            if len(rows) != row_counts[variant][split]:
                raise ValueError(f"{variant}/{split} read-back row count mismatch")
            if file_hash(path) != details[variant]["files"][split]["sha256"]:
                raise ValueError(f"{variant}/{split} read-back hash mismatch")
            observed_rows[variant] += len(rows)
            for row in rows:
                prompt_hash = hashlib.sha256(normalized_user_prompt(row).encode()).hexdigest()
                if assignments.setdefault(prompt_hash, split) != split:
                    raise ValueError("normalized user prompt crosses output splits")
                if variant == "system-stripped":
                    projected = row
                else:
                    projected = {"messages": [m for m in row["messages"] if m.get("role") != "system"]}
                projected_rows[variant][canonical_hash(projected)] += 1
    if observed_rows["system-stripped"] != observed_rows["preserve-system"]:
        raise ValueError("system variants have different output row counts")
    if projected_rows["system-stripped"] != projected_rows["preserve-system"]:
        raise ValueError("output variants differ beyond retained system messages")
    return assignments, observed_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", default="prompt-disjoint-20261019-v1")
    args = parser.parse_args()
    if not args.output_prefix.replace("-", "").replace("_", "").isalnum():
        raise ValueError("output-prefix may contain only letters, digits, '-' and '_'")

    variants = load_variants()
    base_data = variants["system-stripped"]
    prompt_groups, old_split_counts, internal_rows = plan_assignments(base_data)
    outputs: dict[str, dict[str, Path]] = {}
    row_counts: dict[str, Counter[str]] = {}
    out_manifests: dict[str, Any] = {}
    for variant, source in variants.items():
        outputs[variant], row_counts[variant], out_manifests[variant] = write_variant(
            args.output_prefix, variant, source, base_data, old_split_counts
        )
    verified_assignments, observed_rows = verify_outputs(outputs, row_counts, out_manifests)
    if verified_assignments != prompt_groups:
        raise ValueError("output prompt groups disagree with the frozen assignment plan")

    # Write private lineage and release manifest only after every output is complete.
    internal_path = OUT_ROOT / f"{args.output_prefix}-row-lineage.jsonl"
    with internal_path.open("x", encoding="utf-8") as stream:
        for row in internal_rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    os.chmod(internal_path, 0o600)
    manifest = {
        "schema_version": 1,
        "release_id": args.output_prefix,
        "status": "local_candidate_prompt_disjoint_split_audit",
        "upload_gate": False,
        "scope": "research-only; no clinical or production readiness",
        "split_key": "SHA-256 of concatenated user turns after casefold and whitespace normalization",
        "assignment": (
            f"first 8 hex digits modulo 100; train < {TRAIN_BUCKET_LIMIT}, "
            f"validation < {VALIDATION_BUCKET_LIMIT}, else test"
        ),
        "source_releases": {
            "system_stripped_manifest_sha256": file_hash(
                ROOT / "docs/training/execution-2026-10-19/wandb-full-corpus-current-master-20261019-v4.json"
            ),
            "preserve_system_manifest_sha256": file_hash(
                ROOT / "docs/training/execution-2026-10-19/"
                "wandb-full-corpus-current-master-20261019-v4-preserve-system.json"
            ),
        },
        "input_rows": sum(old_split_counts.values()),
        "normalized_prompt_groups": len(prompt_groups),
        "prompt_groups_by_split": dict(Counter(prompt_groups.values())),
        "rows_by_split": {name: dict(count) for name, count in row_counts.items()},
        "preserve_system_pair_validation": "all source rows aligned; non-system projections equal; system weights zero",
        "verification": {
            "output_files_reloaded_and_sha256_checked": True,
            "all_normalized_user_prompt_groups_split_disjoint": True,
            "system_variants_same_rows_and_non_system_projection": True,
            "source_row_count_preserved": observed_rows["system-stripped"],
        },
        "lineage_path": str(internal_path.relative_to(ROOT)),
        "lineage_sha256": file_hash(internal_path),
        "variants": out_manifests,
        "limitations": [
            (
                "Prompt disjointness is exact after casefold and whitespace normalization; "
                "semantic paraphrase overlap is not measured."
            ),
            "Upstream source rights/privacy statuses are unchanged and not resolved by this packaging step.",
            "Not uploaded to any provider; the current v4 training jobs continue unchanged.",
        ],
    }
    manifest_path = ROOT / f"docs/training/execution-2026-10-19/{args.output_prefix}-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.chmod(manifest_path, 0o600)
    sys.stdout.write(
        json.dumps(
            {
                "manifest": str(manifest_path.relative_to(ROOT)),
                "rows": manifest["input_rows"],
                "prompt_groups": manifest["normalized_prompt_groups"],
                "rows_by_split": manifest["rows_by_split"],
                "upload_gate": manifest["upload_gate"],
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
