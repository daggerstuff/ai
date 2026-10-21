"""Build a full-candidate research SFT release with deterministic privacy scrubbing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "ai/data/curated/recovery/gold-20261017/train_master_gold.recovered_candidate.jsonl"
LEDGER = ROOT / "docs/training/execution-2026-10-17/source-decision-ledger.jsonl"
OUTPUT = ROOT / "ai/data/curated/recovery/gold-20261017/wandb-full-corpus-20261018"
MANIFEST = ROOT / "docs/training/execution-2026-10-18/wandb-full-corpus-release.json"
RELEASE_ID = "wandb-full-corpus-20261018-v1"
CANDIDATE_SHA256 = "4f0fd4c5fcfdffeb5434057ff4e122356256f9775540a3b41c1421a75e507663"
CARD_MIN_DIGITS = 13
CARD_MAX_DIGITS = 19
LUHN_RADIX = 10
TRAIN_BUCKET_LIMIT = 90
VALIDATION_BUCKET_LIMIT = 95
PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)"),
    "card_candidate": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "url": re.compile(r"https?://\S+"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


@dataclass(frozen=True)
class BuildContext:
    references: dict[int, dict[str, str]]
    repairs: dict[int, dict[str, Any]]


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


def luhn_valid(value: str) -> bool:
    digits = [int(char) for char in value if char.isdigit()]
    if not CARD_MIN_DIGITS <= len(digits) <= CARD_MAX_DIGITS:
        return False
    total = 0
    for index, original_digit in enumerate(reversed(digits)):
        digit = original_digit
        if index % 2:
            digit *= 2
            if digit > LUHN_RADIX - 1:
                digit -= LUHN_RADIX - 1
        total += digit
    return total % LUHN_RADIX == 0


def redact(text: str, counts: Counter[str]) -> str:
    for name in ("email", "ssn", "phone", "ipv4", "url"):
        text, count = PATTERNS[name].subn(f"[{name.upper()}_REDACTED]", text)
        counts[name] += count

    def redact_card(match: re.Match[str]) -> str:
        if luhn_valid(match.group()):
            counts["payment_card"] += 1
            return "[PAYMENT_CARD_REDACTED]"
        return match.group()

    return PATTERNS["card_candidate"].sub(redact_card, text)


def source_rowrefs() -> dict[int, dict[str, str]]:
    references: dict[int, dict[str, str]] = {}
    with LEDGER.open(encoding="utf-8") as stream:
        for line in stream:
            entry = json.loads(line)
            for reference in entry["candidate_rows"]:
                references[reference["row_number_1based"]] = {
                    "row_sha256": reference["row_sha256"],
                    "source_label": entry["source_label"],
                }
    return references


def project_messages(
    row: dict[str, Any], redaction_counts: Counter[str]
) -> tuple[list[dict[str, Any]] | None, str | None]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None, "messages_not_list"
    projected: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            continue
        if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
            continue
        projected.append({"role": role, "content": redact(content, redaction_counts)})
    assistant_indices = [i for i, message in enumerate(projected) if message["role"] == "assistant"]
    if not assistant_indices:
        return None, "no_assistant_target"
    projected = projected[: assistant_indices[-1] + 1]
    if not any(message["role"] == "user" for message in projected):
        return None, "no_user_before_assistant"
    return (
        [
            {
                "role": message["role"],
                "content": message["content"],
                "weight": 0 if message["role"] == "user" else 1,
            }
            for message in projected
        ],
        None,
    )


def assign_split(group_hash: str) -> str:
    bucket = int(group_hash[:8], 16) % 100
    return "train" if bucket < TRAIN_BUCKET_LIMIT else "validation" if bucket < VALIDATION_BUCKET_LIMIT else "test"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def empty_stats(staging: Path) -> dict[str, Any]:
    payload_paths = {name: staging / f"{name}.jsonl" for name in ("train", "validation", "test")}
    return {
        "payload_paths": payload_paths,
        "internal_path": staging / "manifest_internal.jsonl",
        "rowrefs_path": staging / "rowrefs.jsonl",
        "heldout_path": staging / "heldout.jsonl",
        "row_counts": Counter(),
        "group_sets": {name: set() for name in payload_paths},
        "source_counts": Counter(),
        "redaction_counts": Counter(),
        "screen_after": Counter(),
        "dropped_no_target": 0,
        "drop_reasons": Counter(),
        "duplicate_groups": set(),
        "seen_groups": set(),
        "repair_count": 0,
        "repair_kinds": Counter(),
    }


def validate_repair(
    row_number: int,
    reference: dict[str, str],
    repair: dict[str, Any] | None,
    stats: dict[str, Any],
    row: dict[str, Any],
) -> None:
    if repair is None:
        return
    if repair.get("candidate_row_sha256") != reference["row_sha256"]:
        raise ValueError(f"repair source hash mismatch at row {row_number}")
    if not isinstance(repair.get("messages"), list):
        raise ValueError(f"repair has no message list at row {row_number}")
    row["messages"] = repair["messages"]
    stats["repair_count"] += 1
    stats["repair_kinds"][repair["repair_kind"]] += 1


def write_row(
    row_number: int,
    raw_line: str,
    streams: dict[str, Any],
    stats: dict[str, Any],
    context: BuildContext,
) -> None:
    reference = context.references[row_number]
    raw_payload = raw_line.rstrip("\n").encode()
    if sha256(raw_payload) != reference["row_sha256"]:
        raise ValueError(f"candidate ledger mismatch at row {row_number}")
    row = json.loads(raw_line)
    if row.get("source") != reference["source_label"]:
        raise ValueError(f"candidate source mapping mismatch at row {row_number}")
    repair = context.repairs.get(row_number)
    validate_repair(row_number, reference, repair, stats, row)
    payload_messages, drop_reason = project_messages(row, stats["redaction_counts"])
    if payload_messages is None:
        stats["dropped_no_target"] += 1
        stats["drop_reasons"][drop_reason or "unknown"] += 1
        return
    for message in payload_messages:
        for name, pattern in PATTERNS.items():
            if name == "card_candidate":
                if any(luhn_valid(match.group()) for match in pattern.finditer(message["content"])):
                    stats["screen_after"]["payment_card"] += 1
            else:
                stats["screen_after"][name] += len(pattern.findall(message["content"]))
    group_hash = canonical_hash(payload_messages)
    split = assign_split(group_hash)
    if group_hash in stats["seen_groups"]:
        stats["duplicate_groups"].add(group_hash)
    stats["seen_groups"].add(group_hash)
    stats["group_sets"][split].add(group_hash)
    stats["source_counts"][reference["source_label"]] += 1
    stats["row_counts"][split] += 1
    streams["payload"][split].write(
        json.dumps({"messages": payload_messages}, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    internal_record = {
        "candidate_row_number_1based": row_number,
        "candidate_row_sha256": reference["row_sha256"],
        "content_group_sha256": group_hash,
        "split": split,
        "source_label": reference["source_label"],
    }
    if repair is not None:
        internal_record["repair_kind"] = repair["repair_kind"]
        internal_record["repair_source_dialogue_id"] = repair.get("source_dialogue_id")
    streams["internal"].write(json.dumps(internal_record, separators=(",", ":")) + "\n")
    streams["rowrefs"].write(
        json.dumps(
            {
                "candidate_row_number_1based": row_number,
                "candidate_row_sha256": reference["row_sha256"],
                "content_group_sha256": group_hash,
                "provisional_split": split,
                **({"repair_kind": repair["repair_kind"]} if repair is not None else {}),
            },
            separators=(",", ":"),
        )
        + "\n"
    )


def render_candidate(
    staging: Path,
    context: BuildContext,
) -> dict[str, Any]:
    stats = empty_stats(staging)
    with ExitStack() as stack:
        candidate = stack.enter_context(CANDIDATE.open(encoding="utf-8"))
        payload_streams = {
            name: stack.enter_context(path.open("w", encoding="utf-8")) for name, path in stats["payload_paths"].items()
        }
        internal_stream = stack.enter_context(stats["internal_path"].open("w", encoding="utf-8"))
        rowrefs_stream = stack.enter_context(stats["rowrefs_path"].open("w", encoding="utf-8"))
        streams = {"payload": payload_streams, "internal": internal_stream, "rowrefs": rowrefs_stream}
        for row_number, raw_line in enumerate(candidate, start=1):
            if row_number not in context.references:
                raise ValueError(f"candidate row {row_number} has no provenance reference")
            write_row(row_number, raw_line, streams, stats, context)
    stats["heldout_path"].write_text("", encoding="utf-8")
    return stats


def create_manifest(
    stats: dict[str, Any], output: Path, release_id: str, repairs_path: Path | None
) -> dict[str, Any]:
    payload_paths = stats["payload_paths"]
    row_counts = stats["row_counts"]
    group_sets = stats["group_sets"]
    source_counts = stats["source_counts"]
    internal_path = stats["internal_path"]
    rowrefs_path = stats["rowrefs_path"]
    heldout_path = stats["heldout_path"]
    redaction_counts = stats["redaction_counts"]
    screen_after = stats["screen_after"]
    dropped_no_target = stats["dropped_no_target"]
    partition_manifest = {
        name: {
            "path": str((output / path.name).relative_to(ROOT)),
            "sha256": hash_file(path),
            "row_count": row_counts[name],
            "source_group_count": len(group_sets[name]),
            "format": "weighted_messages_jsonl_v1",
            "split": name,
        }
        for name, path in payload_paths.items()
    }
    return {
        "schema_version": "g04-release-manifest-v1",
        "status": "approved_for_research_training",
        "upload_gate": True,
        "release_id": release_id,
        "provider": "wandb",
        "scope": {
            "use": "full_candidate_research_fine_tuning_experiment",
            "clinical_or_production_readiness": False,
            "candidate_rows": sum(row_counts.values()) + dropped_no_target,
            "retained_rows": sum(row_counts.values()),
            "dropped_structurally_untrainable_rows": dropped_no_target,
            "dropped_rows_by_reason": dict(stats["drop_reasons"]),
            "repaired_rows": stats["repair_count"],
            "repair_kinds": dict(stats["repair_kinds"]),
            "source_groups": len(stats["seen_groups"]),
        },
        "owner_direction": "User directed use of the full recovered candidate for research training on 2026-10-18.",
        "source_candidate_sha256": CANDIDATE_SHA256,
        "source_candidate": {
            "path": str(CANDIDATE.relative_to(ROOT)),
            "sha256": CANDIDATE_SHA256,
            "row_count": 190139,
        },
        "repairs": (
            {
                "path": str(repairs_path.relative_to(ROOT)),
                "sha256": hash_file(repairs_path),
                "row_count": stats["repair_count"],
                "repair_kinds": dict(stats["repair_kinds"]),
                "human_review": "generated continuations not clinically reviewed; source-triage examples are synthetic",
            }
            if repairs_path is not None
            else None
        ),
        "partitions": partition_manifest,
        "payload_contract": {
            "format": "weighted_messages_jsonl_v1",
            "user_weight": 0,
            "assistant_weight": 1,
            "loss_mask": "assistant-turn-only",
            "projection": "system messages removed; trailing turns after final assistant target removed",
        },
        "evidence": {
            "group_disjointness": {
                "status": "verified",
                "group_key": "content_group_sha256",
                "train_group_count": len(group_sets["train"]),
                "validation_group_count": len(group_sets["validation"]),
                "test_group_count": len(group_sets["test"]),
                "pairwise_intersections": {"train_validation": 0, "train_test": 0, "validation_test": 0},
                "evidence_path": str((output / rowrefs_path.name).relative_to(ROOT)),
                "evidence_sha256": hash_file(rowrefs_path),
                "reserved_eval_group_count": 0,
                "heldout_group_intersections": {"train": 0, "validation": 0, "test": 0},
                "heldout_exclusions_path": str((output / heldout_path.name).relative_to(ROOT)),
                "heldout_exclusions_sha256": hash_file(heldout_path),
            },
            "source_origin": {
                "status": "full_recovered_candidate_rowrefs_verified",
                "source_label_count": len(source_counts),
                "source_counts": dict(sorted(source_counts.items())),
                "decision_basis": (
                    "User-directed full-corpus research experiment; existing source policy dispositions "
                    "remain in internal rowrefs for audit."
                ),
            },
            "privacy": {
                "automated_redactions": dict(redaction_counts),
                "basic_pattern_screen_after_redaction": dict(screen_after),
                "clinical_ner_screen": "not run",
                "limitations": [
                    "Pattern redaction is not comprehensive de-identification.",
                    "No NER or semantic sensitive-information review was run.",
                    "Research experiment only; not clinical or production readiness.",
                ],
            },
        },
        "internal_manifest": {
            "path": str((output / internal_path.name).relative_to(ROOT)),
            "sha256": hash_file(internal_path),
            "row_count": sum(row_counts.values()),
        },
        "limitations": [
            "Research-only SFT experiment; not clinical or production readiness.",
            "The automated identifier screen does not certify de-identification.",
            "Exact-message group split; semantic near-duplicate certification was not run.",
            "Existing source policy dispositions are preserved in the internal row membership manifest.",
            "Fifty generated Psy-Insight continuations are research data and have not received human clinical review.",
        ],
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }


def load_repairs(path: Path) -> dict[int, dict[str, Any]]:
    repairs: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            entry = json.loads(line)
            row_number = entry.get("candidate_row_number_1based")
            if not isinstance(row_number, int) or row_number in repairs:
                raise ValueError(f"invalid or duplicate repair row on line {line_number}")
            repairs[row_number] = entry
    return repairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-id", default=RELEASE_ID)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--repairs", type=Path)
    args = parser.parse_args()
    release_id = args.release_id
    output = args.output if args.output.is_absolute() else ROOT / args.output
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    repairs_path = args.repairs.resolve() if args.repairs else None
    repairs = load_repairs(repairs_path) if repairs_path else {}
    if output.exists() or manifest_path.exists():
        raise ValueError("full-corpus release already exists; use a new versioned release")
    if hash_file(CANDIDATE) != CANDIDATE_SHA256:
        raise ValueError("candidate identity mismatch; refusing to build")
    staging = output.with_name(output.name + ".building")
    if staging.exists():
        raise ValueError(f"staging output already exists: {staging}")
    staging.mkdir(parents=True, mode=0o700)
    context = BuildContext(source_rowrefs(), repairs)
    stats = render_candidate(staging, context)
    if any(stats["screen_after"].values()):
        raise ValueError(f"identifier patterns remain after redaction: {dict(stats['screen_after'])}")
    for path in (
        *stats["payload_paths"].values(),
        stats["internal_path"],
        stats["rowrefs_path"],
        stats["heldout_path"],
    ):
        os.chmod(path, 0o600)
    manifest = create_manifest(stats, output, release_id, repairs_path)
    write_json(manifest_path, manifest)
    os.rename(staging, output)
    summary = {
        "release_id": manifest["release_id"],
        "retained_rows": sum(stats["row_counts"].values()),
        "dropped_structurally_untrainable_rows": stats["dropped_no_target"],
        "dropped_rows_by_reason": dict(stats["drop_reasons"]),
        "split_rows": dict(stats["row_counts"]),
        "unique_content_groups": len(stats["seen_groups"]),
        "duplicate_groups": len(stats["duplicate_groups"]),
        "source_labels": len(stats["source_counts"]),
        "repaired_rows": stats["repair_count"],
        "redactions": dict(stats["redaction_counts"]),
        "post_redaction_identifier_hits": dict(stats["screen_after"]),
        "manifest_sha256": hash_file(manifest_path),
    }
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
