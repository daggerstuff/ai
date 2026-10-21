"""Build and verify a W&B G04 release from the frozen prompt-disjoint split."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROMPT_MANIFEST = ROOT / "docs/training/execution-2026-10-19/prompt-disjoint-20261019-v3-manifest.json"
SOURCE_MANIFEST = ROOT / "docs/training/execution-2026-10-19/wandb-full-corpus-current-master-20261019-v4.json"
OUTPUT_ID = "wandb-full-corpus-prompt-disjoint-20261020-v1"
OUTPUT_DIR = ROOT / "ai/data/curated/recovery/gold-20261017" / OUTPUT_ID
OUTPUT_MANIFEST = ROOT / "docs/training/execution-2026-10-19" / f"{OUTPUT_ID}.json"
SPLITS = ("train", "validation", "test")


@dataclass
class ReleaseInputs:
    prompt: dict[str, Any]
    source: dict[str, Any]
    lineage: list[dict[str, Any]]
    source_internal: list[dict[str, Any]]
    source_payloads: dict[str, list[dict[str, Any]]]


@dataclass
class ReleaseRows:
    payload_rows: dict[str, list[dict[str, Any]]]
    internal_rows: list[dict[str, Any]]
    rowrefs: list[dict[str, Any]]
    prompt_groups: dict[str, set[str]]
    group_count: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
    os.chmod(path, 0o600)
    return sha256_file(path)


def load_inputs() -> ReleaseInputs:
    prompt = json.loads(PROMPT_MANIFEST.read_text(encoding="utf-8"))
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    source_candidate = source["source_candidate"]
    candidate_path = ROOT / source_candidate["path"]
    if sha256_file(candidate_path) != source_candidate["sha256"]:
        raise ValueError("source candidate hash does not match frozen G04 release")
    if source_candidate["sha256"] != source["source_candidate_sha256"]:
        raise ValueError("source candidate identity is inconsistent")

    lineage_path = ROOT / prompt["lineage_path"]
    if sha256_file(lineage_path) != prompt["lineage_sha256"]:
        raise ValueError("prompt-disjoint lineage hash mismatch")
    lineage = read_jsonl(lineage_path)
    if len(lineage) != source_candidate["row_count"]:
        raise ValueError("prompt-disjoint lineage does not cover the source candidate")

    source_internal_path = ROOT / source["internal_manifest"]["path"]
    if sha256_file(source_internal_path) != source["internal_manifest"]["sha256"]:
        raise ValueError("source G04 membership hash mismatch")
    source_internal = read_jsonl(source_internal_path)
    if len(source_internal) != source_candidate["row_count"]:
        raise ValueError("source G04 membership does not cover the source candidate")

    prompt_manifest = prompt["variants"]["system-stripped"]
    for split in SPLITS:
        identity = prompt_manifest["files"][split]
        path = ROOT / identity["path"]
        if sha256_file(path) != identity["sha256"]:
            raise ValueError(f"prompt-disjoint payload hash mismatch for {split}")
        if len(read_jsonl(path)) != identity["rows"]:
            raise ValueError(f"prompt-disjoint payload row count mismatch for {split}")
    for split in SPLITS:
        identity = source["partitions"][split]
        path = ROOT / identity["path"]
        if sha256_file(path) != identity["sha256"]:
            raise ValueError(f"frozen G04 payload hash mismatch for {split}")
    source_payloads = {name: read_jsonl(ROOT / source["partitions"][name]["path"]) for name in SPLITS}
    return ReleaseInputs(prompt, source, lineage, source_internal, source_payloads)


def source_membership_index(
    source_internal: list[dict[str, Any]],
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[int, dict[str, Any]]]:
    source_rows_by_split: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLITS}
    for row in source_internal:
        split = row.get("split")
        if split not in source_rows_by_split:
            raise ValueError("source G04 membership contains an invalid split")
        source_rows_by_split[split].append(row)
    source_identity_by_location: dict[tuple[str, int], dict[str, Any]] = {}
    for split, rows in source_rows_by_split.items():
        for index, row in enumerate(rows, start=1):
            source_identity_by_location[(split, index)] = row

    candidate_rows = {row["candidate_row_number_1based"]: row for row in source_internal}
    if len(candidate_rows) != len(source_internal):
        raise ValueError("source candidate row identities are not unique")
    return source_identity_by_location, candidate_rows


def lineage_index(
    lineage: list[dict[str, Any]], source_rows_by_location: dict[tuple[str, int], dict[str, Any]], row_count: int
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, str]]:
    group_assignment: dict[str, str] = {}
    location_seen: set[tuple[str, int]] = set()
    lineage_rows_by_location: dict[tuple[str, int], dict[str, Any]] = {}
    for row in lineage:
        src_split = row.get("source_split")
        src_index = row.get("row_index_1based_within_source_split")
        destination = row.get("destination_split")
        prompt_hash = row.get("prompt_group_sha256")
        if destination not in SPLITS:
            raise ValueError("prompt lineage contains an invalid split")
        if not isinstance(src_index, int) or src_index < 1 or not isinstance(prompt_hash, str):
            raise ValueError("prompt lineage has invalid row identity")
        location = (src_split, src_index)
        if location not in source_rows_by_location:
            raise ValueError("prompt lineage points outside the frozen source membership")
        if location in location_seen:
            raise ValueError("prompt lineage repeats a source split row")
        location_seen.add(location)
        lineage_rows_by_location[location] = row
        prior = group_assignment.setdefault(prompt_hash, destination)
        if prior != destination:
            raise ValueError("a normalized prompt group crosses the proposed split")
    if len(location_seen) != row_count:
        raise ValueError("prompt lineage does not map every source split row exactly once")
    return lineage_rows_by_location, group_assignment


def collect_release_rows(
    inputs: ReleaseInputs,
    source_identity_by_location: dict[tuple[str, int], dict[str, Any]],
    lineage_rows_by_location: dict[tuple[str, int], dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    # Match each source row to the exact prompt lineage and candidate identity.
    internal_rows: list[dict[str, Any]] = []
    rowrefs: list[dict[str, Any]] = []
    payload_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLITS}
    for split in SPLITS:
        if len(inputs.source_payloads[split]) != sum(row.get("split") == split for row in inputs.source_internal):
            raise ValueError(f"source payload and row membership counts differ for {split}")
        for index, source_payload_row in enumerate(inputs.source_payloads[split], start=1):
            line = lineage_rows_by_location[(split, index)]
            if canonical_sha256(source_payload_row) != line["row_sha256"]:
                raise ValueError(f"prompt lineage does not match frozen source row {split}:{index}")
            source_identity = source_identity_by_location[(split, index)]
            destination = line["destination_split"]
            candidate_id = source_identity["candidate_row_sha256"]
            candidate_number = source_identity["candidate_row_number_1based"]
            internal_rows.append(
                {
                    "candidate_row_number_1based": candidate_number,
                    "candidate_row_sha256": candidate_id,
                    "prompt_group_sha256": line["prompt_group_sha256"],
                    "split": destination,
                }
            )
            rowrefs.append(
                {
                    "candidate_row_number_1based": candidate_number,
                    "candidate_row_sha256": candidate_id,
                    "prompt_group_sha256": line["prompt_group_sha256"],
                    "provisional_split": destination,
                }
            )
            payload_rows[destination].append(source_payload_row)
    return payload_rows, internal_rows, rowrefs


def verify_release_rows(
    source: dict[str, Any],
    candidate_rows: dict[int, dict[str, Any]],
    internal_rows: list[dict[str, Any]],
    group_assignment: dict[str, str],
) -> dict[str, set[str]]:
    source_candidate = source["source_candidate"]
    if len(candidate_rows) != source_candidate["row_count"]:
        raise ValueError("source candidate row identities are not unique")
    prompt_groups: dict[str, set[str]] = {name: set() for name in SPLITS}
    row_ids: set[int] = set()
    for row in internal_rows:
        number = row["candidate_row_number_1based"]
        if (
            number in row_ids
            or candidate_rows.get(number, {}).get("candidate_row_sha256") != row["candidate_row_sha256"]
        ):
            raise ValueError("prompt split does not preserve exact candidate row identity")
        row_ids.add(number)
        prompt_groups[row["split"]].add(row["prompt_group_sha256"])
    if len(row_ids) != source_candidate["row_count"]:
        raise ValueError("prompt split omits or duplicates candidate row identities")
    split_pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    if any(prompt_groups[left] & prompt_groups[right] for left, right in split_pairs):
        raise ValueError("prompt groups cross release splits")
    if set(group_assignment) != set.union(*prompt_groups.values()):
        raise ValueError("prompt group assignment index differs from release rows")
    return prompt_groups


def write_release(
    inputs: ReleaseInputs,
    rows: ReleaseRows,
) -> dict[str, Any]:
    prompt, source = inputs.prompt, inputs.source
    payload_rows = rows.payload_rows
    internal_rows = rows.internal_rows
    rowrefs = rows.rowrefs
    prompt_groups = rows.prompt_groups
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(OUTPUT_DIR, 0o700)
    partition_data: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        path = OUTPUT_DIR / f"{split}.jsonl"
        digest = write_jsonl(path, payload_rows[split])
        partition_data[split] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": digest,
            "row_count": len(payload_rows[split]),
            "source_group_count": len(prompt_groups[split]),
            "format": source["partitions"][split].get("format", "weighted_messages_jsonl_v1"),
            "split": split,
        }

    internal_path = OUTPUT_DIR / "manifest_internal.jsonl"
    internal_sha = write_jsonl(internal_path, internal_rows)
    rowrefs_path = OUTPUT_DIR / "rowrefs.jsonl"
    rowrefs_sha = write_jsonl(rowrefs_path, rowrefs)
    heldout_path = OUTPUT_DIR / "heldout.jsonl"
    heldout_path.write_text("", encoding="utf-8")
    os.chmod(heldout_path, 0o600)
    heldout_sha = sha256_file(heldout_path)

    manifest = dict(source)
    manifest["release_id"] = OUTPUT_ID
    manifest["created_at_utc"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    manifest["split_policy"] = {
        "key": "prompt_group_sha256",
        "normalization": "casefold and collapse all whitespace across user turns",
        "assignment": prompt["assignment"],
        "groups": rows.group_count,
    }
    manifest["partitions"] = partition_data
    manifest["internal_manifest"] = {
        "path": str(internal_path.relative_to(ROOT)),
        "sha256": internal_sha,
        "row_count": len(internal_rows),
    }
    evidence = dict(source["evidence"])
    group_evidence = dict(evidence["group_disjointness"])
    group_evidence.update(
        {
            "status": "verified",
            "group_key": "prompt_group_sha256",
            "train_group_count": len(prompt_groups["train"]),
            "validation_group_count": len(prompt_groups["validation"]),
            "test_group_count": len(prompt_groups["test"]),
            "pairwise_intersections": {"train_validation": 0, "train_test": 0, "validation_test": 0},
            "evidence_path": str(rowrefs_path.relative_to(ROOT)),
            "evidence_sha256": rowrefs_sha,
            "reserved_eval_group_count": 0,
            "heldout_group_intersections": {"train": 0, "validation": 0, "test": 0},
            "heldout_exclusions_path": str(heldout_path.relative_to(ROOT)),
            "heldout_exclusions_sha256": heldout_sha,
        }
    )
    evidence["group_disjointness"] = group_evidence
    manifest["evidence"] = evidence
    manifest["limitations"] = [
        *source.get("limitations", []),
        "Split assignment was recomputed by normalized user prompt; semantic paraphrase overlap is not measured.",
        "Source rights, privacy, consent, and provider-transfer dispositions are unchanged from the source release.",
    ]
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.chmod(OUTPUT_MANIFEST, 0o600)
    return manifest


def main() -> int:
    inputs = load_inputs()
    source_identity_by_location, candidate_rows = source_membership_index(inputs.source_internal)
    lineage_rows_by_location, group_assignment = lineage_index(
        inputs.lineage,
        source_identity_by_location,
        inputs.source["source_candidate"]["row_count"],
    )
    payload_rows, internal_rows, rowrefs = collect_release_rows(
        inputs, source_identity_by_location, lineage_rows_by_location
    )
    prompt_groups = verify_release_rows(inputs.source, candidate_rows, internal_rows, group_assignment)
    rows = ReleaseRows(payload_rows, internal_rows, rowrefs, prompt_groups, len(group_assignment))
    manifest = write_release(inputs, rows)
    sys.stdout.write(
        json.dumps(
            {
                "manifest": str(OUTPUT_MANIFEST.relative_to(ROOT)),
                "candidate_rows": len(internal_rows),
                "prompt_groups": rows.group_count,
                "rows_by_split": {name: len(rows.payload_rows[name]) for name in SPLITS},
                "groups_by_split": {name: len(prompt_groups[name]) for name in SPLITS},
                "upload_gate": manifest["upload_gate"],
                "candidate_sha256": manifest["source_candidate_sha256"],
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
