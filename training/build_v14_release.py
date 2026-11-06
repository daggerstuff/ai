"""Build the v14 G04 research release on top of v13.

v13 carried 1,522 ``crisis_benchmark`` rows whose assistant target is the
benchmark's own task description ("Assess this input for risk level, ...")
rather than a response. The v5 hygiene pass excluded 1,374 of them from train
as prompt echoes, but 148 survived in validation and test, and none of the
1,522 carried a real crisis target. v13 therefore saw no crisis routing at all,
which is why the held-out crisis suite scores 1/8 for the adapter against 8/8
for the base model.

v13 also welds the field label onto tool-argument values, which the
literal-extraction suite exposes as 2/6.

v14 fixes both at the data source rather than at the eval:

* every ``crisis_benchmark`` placeholder row is purged from the candidate
  itself, across every split, so no later release can inherit one;
* the 624 train rows of the v9 addendum are appended: 192 bare-literal
  ``submit_mock_change`` trajectories and 432 crisis-routing replies (plus
  their 60 controls).

Every surviving v13 row keeps its exact candidate line, its
``candidate_row_sha256``, its ``content_group_sha256``, and its split. Only the
row numbering changes, because purging renumbers the candidate.

Usage::

    PYTHONPATH=/home/vivi/pixelated python -m training.build_v14_release --dry-run
    PYTHONPATH=/home/vivi/pixelated python -m training.build_v14_release
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training import build_agent_tool_sft_v9 as v9

ROOT = Path(__file__).resolve().parents[2]

V13_DIR = ROOT / "ai/data/curated/recovery/gold-20261017/wandb-full-corpus-current-master-20261027-v13"
REPAIRS = ROOT / "ai/data/curated/recovery/gold-20261017/wandb-full-corpus-repairs-20261019-v3/row_repairs.jsonl"

OUTPUT = ROOT / "ai/data/curated/recovery/gold-20261017/wandb-full-corpus-current-master-20261106-v14"
MANIFEST = ROOT / "docs/training/execution-2026-11-06/wandb-full-corpus-current-master-20261106-v14.json"

RELEASE_ID = "wandb-full-corpus-current-master-20261106-v14"
SHUFFLE_SEED = 20261106
BATCH_SIZE = 16
SPLITS = ("train", "validation", "test")

# The one assistant target that identifies a fabricated crisis row: the
# benchmark task description, byte-identical on all 1,522 rows.
CRISIS_PLACEHOLDER = (
    "Assess this input for risk level, propose safety resources, and generate a compassionate, non-harmful response."
)
PURGED_SOURCE = "crisis_benchmark"
PURGE_REASON = "fabricated_crisis_target"

# The v9 addendum is appended as train only, matching the v3..v8 convention.
V9_SOURCE_LABEL = "agent_tool_synthetic_v9"

# Pinned identities. The build refuses to run if any of these drift.
V13_CANDIDATE_SHA256 = "3ffdccf32bfad8f577f8dae93d097fba0e08f01f44606bb8ba1179100dbb2952"
V13_CANDIDATE_ROWS = 193051
V13_ROWREFS_SHA256 = "5d98b1c5432e80b05d41255cee7fdf941de80b52467b6e99ee35620dd0410b04"
V13_INTERNAL_SHA256 = "6d636e558ec5df9eaa9c33093f00f18b5a3804bba3e6404dbdea8e4660587f18"
V13_HELDOUT_SHA256 = "a7f6fc3112043dc5b583b6f96ca9b6ccc4719460b6948d971c171d3b979eb82f"
V13_HELDOUT_ROWS = 93543
REPAIRS_SHA256 = "17308c80a2eb332774d3549a2ef0260bdfabac6f53044cc0160880e166a7238a"
REPAIRS_ROWS = 52
EXPECTED_PURGED_ROWS = 1522
EXPECTED_V9_TRAIN_ROWS = 624

CARD_MIN_DIGITS = 13
CARD_MAX_DIGITS = 19
LUHN_RADIX = 10
PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)"),
    "card_candidate": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "url": re.compile(r"https?://\S+"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}
REDACTION_ORDER = ("email", "ssn", "phone", "ipv4", "url")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


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
    for name in REDACTION_ORDER:
        text, count = PATTERNS[name].subn(f"[{name.upper()}_REDACTED]", text)
        counts[name] += count

    def redact_card(match: re.Match[str]) -> str:
        if luhn_valid(match.group()):
            counts["payment_card"] += 1
            return "[PAYMENT_CARD_REDACTED]"
        return match.group()

    return PATTERNS["card_candidate"].sub(redact_card, text)


def is_agent_tool(row: dict[str, Any]) -> bool:
    """Agent-tool rows carry a tool catalog and no source label.

    v13 uses the same discriminator: exactly 2,807 candidate rows match, the
    sum of the v2..v8 agent-tool addenda.
    """
    return "tools" in row and "source" not in row


def project_chat(messages: list[Any], counts: Counter[str]) -> list[dict[str, Any]] | None:
    """Chat projection: system dropped, PII redacted, truncated after the last assistant."""
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
        projected.append({"role": role, "content": redact(content, counts)})
    assistant_indices = [index for index, message in enumerate(projected) if message["role"] == "assistant"]
    if not assistant_indices:
        return None
    projected = projected[: assistant_indices[-1] + 1]
    if not any(message["role"] == "user" for message in projected):
        return None
    return [
        {
            "role": message["role"],
            "content": message["content"],
            "weight": 0 if message["role"] == "user" else 1,
        }
        for message in projected
    ]


def project_agent(messages: list[Any]) -> list[dict[str, Any]]:
    """Agent-tool projection: the whole turn shape is retained, system included.

    Key order follows the v13 serialization exactly — role first, then the
    role-specific fields, with ``weight`` last — so retained rows stay
    byte-identical across releases.

    v13 does not redact these rows; they are first-party synthetic and carry no
    third-party identifiers.
    """
    projected: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            continue
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError("tool message without tool_call_id")
            item: dict[str, Any] = {"role": role, "tool_call_id": call_id}
            if message.get("name"):
                item["name"] = message["name"]
            item["content"] = message.get("content")
        else:
            item = {"role": role, "content": message.get("content")}
            if message.get("tool_calls"):
                item["tool_calls"] = message["tool_calls"]
        item["weight"] = 1 if role == "assistant" else 0
        projected.append(item)
    return projected


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_repairs(path: Path) -> dict[int, dict[str, Any]]:
    return {entry["candidate_row_number_1based"]: entry for entry in read_jsonl(path)}


def verify_pinned_identity() -> None:
    """Refuse to build from anything other than the reviewed v13 lineage."""
    checks = (
        (V13_DIR / "candidate.jsonl", V13_CANDIDATE_SHA256, "v13 candidate"),
        (V13_DIR / "rowrefs.jsonl", V13_ROWREFS_SHA256, "v13 rowrefs"),
        (V13_DIR / "manifest_internal.jsonl", V13_INTERNAL_SHA256, "v13 internal manifest"),
        (V13_DIR / "heldout.jsonl", V13_HELDOUT_SHA256, "v13 heldout"),
        (REPAIRS, REPAIRS_SHA256, "row repairs"),
    )
    for path, expected, label in checks:
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"{label} hash drifted: {actual} != {expected}")


def is_crisis_placeholder(row: dict[str, Any]) -> bool:
    """True only for a crisis_benchmark row that carries the fabricated target."""
    if row.get("source") != PURGED_SOURCE:
        return False
    targets = [
        message.get("content")
        for message in row.get("messages", [])
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    return CRISIS_PLACEHOLDER in targets


def v9_candidate_rows() -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """The v9 train rows as (candidate row, projected payload messages)."""
    rows = v9.collect_rows()["train"]
    if len(rows) != EXPECTED_V9_TRAIN_ROWS:
        raise ValueError(f"expected {EXPECTED_V9_TRAIN_ROWS} v9 train rows, found {len(rows)}")
    out: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for item in rows:
        candidate = {"messages": item["messages"], "tools": item["tools"]}
        if not is_agent_tool(candidate):
            raise ValueError(f"{item['id']}: v9 candidate rows must carry tools and no source label")
        out.append((candidate, project_agent(item["messages"])))
    return out


def line_sha(raw: bytes) -> str:
    """Row identity is the hash of the JSONL line bytes, excluding the newline."""
    body = raw[:-1] if raw.endswith(b"\n") else raw
    return hashlib.sha256(body).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


@dataclass(frozen=True)
class _Indexes:
    """The v13 provenance indexes the build reads before touching the candidate."""

    references: dict[int, dict[str, Any]]
    internal_source: dict[int, str]
    heldout_source: dict[int, dict[str, Any]]
    repairs: dict[int, dict[str, Any]]


def load_indexes() -> _Indexes:
    references = {row["candidate_row_number_1based"]: row for row in read_jsonl(V13_DIR / "rowrefs.jsonl")}
    internal_source = {
        row["candidate_row_number_1based"]: row["source_label"]
        for row in read_jsonl(V13_DIR / "manifest_internal.jsonl")
    }
    heldout_source = {row["candidate_row_number_1based"]: row for row in read_jsonl(V13_DIR / "heldout.jsonl")}
    repairs = load_repairs(REPAIRS)
    if len(repairs) != REPAIRS_ROWS:
        raise ValueError(f"expected {REPAIRS_ROWS} repairs, found {len(repairs)}")
    if len(heldout_source) != V13_HELDOUT_ROWS:
        raise ValueError(f"expected {V13_HELDOUT_ROWS} heldout rows, found {len(heldout_source)}")
    return _Indexes(references, internal_source, heldout_source, repairs)


def payload_line(payload: list[dict[str, Any]], tools: list[Any] | None) -> str:
    body = {"messages": payload} if tools is None else {"messages": payload, "tools": tools}
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"))


class _Release:
    """One v14 build: purge, carry, append, then write payloads and the manifest."""

    def __init__(self, indexes: _Indexes, output: Path, dry_run: bool) -> None:
        self.indexes = indexes
        self.output = output
        self.dry_run = dry_run
        self.redaction_counts: Counter[str] = Counter()
        self.payload_lines: dict[str, list[str]] = {name: [] for name in SPLITS}
        self.internal_records: list[dict[str, Any]] = []
        self.rowref_records: list[dict[str, Any]] = []
        self.heldout_records: list[dict[str, Any]] = []
        self.group_sets: dict[str, set[str]] = {name: set() for name in SPLITS}
        self.source_counts: Counter[str] = Counter()
        self.v9_rows: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        self.carried = 0
        self.purged = 0
        self.next_number = 0
        self.candidate_stream: Any = None

    def run(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        self.candidate_stream = None if self.dry_run else (self.output / "candidate.jsonl").open("wb")
        try:
            with (V13_DIR / "candidate.jsonl").open("rb") as source:
                for old_number, raw in enumerate(source, start=1):
                    if raw.strip():
                        self.carry(old_number, raw)
            self.append_v9()
        finally:
            if self.candidate_stream is not None:
                self.candidate_stream.close()
        if self.purged != EXPECTED_PURGED_ROWS:
            raise ValueError(f"expected {EXPECTED_PURGED_ROWS} purged rows, found {self.purged}")
        if self.carried + len(self.v9_rows) != self.next_number:
            raise ValueError("row numbering drifted")
        # Deterministic train shuffle; validation and test keep candidate order.
        random.Random(SHUFFLE_SEED).shuffle(self.payload_lines["train"])

    def carry(self, old_number: int, raw: bytes) -> None:
        row = json.loads(raw)
        if is_crisis_placeholder(row):
            self.purged += 1
            return
        reference = self.indexes.references.get(old_number)
        held = self.indexes.heldout_source.get(old_number)
        if reference is not None and held is not None:
            raise ValueError(f"candidate row {old_number} is both retained and held out")
        if reference is None and held is None:
            raise ValueError(f"candidate row {old_number} is neither retained nor held out")
        row_sha = self.admit(raw)
        if held is not None:
            self.record_heldout(old_number, row_sha, held)
            return
        assert reference is not None
        self.record_retained(old_number, row, row_sha, reference)

    def admit(self, raw: bytes) -> str:
        """Write one surviving v13 candidate line verbatim under its new number."""
        row_sha = line_sha(raw)
        self.next_number += 1
        self.carried += 1
        if self.candidate_stream is not None:
            self.candidate_stream.write(raw)
        return row_sha

    def record_heldout(self, old_number: int, row_sha: str, held: dict[str, Any]) -> None:
        if row_sha != held["candidate_row_sha256"]:
            raise ValueError(f"heldout provenance mismatch at row {old_number}")
        # Held-out rows stay out of rowrefs: the gate reads rowrefs as retained
        # membership only, and rejects any other split label.
        self.heldout_records.append(
            {
                "candidate_row_number_1based": self.next_number,
                "candidate_row_sha256": row_sha,
                "content_group_sha256": held["content_group_sha256"],
                "provisional_split": "heldout",
                "source_label": held["source_label"],
                "exclusion_reason": held["exclusion_reason"],
            }
        )

    def record_retained(
        self,
        old_number: int,
        row: dict[str, Any],
        row_sha: str,
        reference: dict[str, Any],
    ) -> None:
        messages = row.get("messages")
        repair = self.indexes.repairs.get(old_number)
        if repair is not None:
            if repair.get("candidate_row_sha256") != row_sha:
                raise ValueError(f"repair source hash mismatch at row {old_number}")
            messages = repair["messages"]
        if row_sha != reference["candidate_row_sha256"]:
            raise ValueError(f"candidate provenance mismatch at row {old_number}")
        agent_tool = is_agent_tool(row)
        payload = project_agent(messages) if agent_tool else project_chat(messages, self.redaction_counts)
        if payload is None:
            raise ValueError(f"retained row {old_number} projects to no assistant target")
        group_hash = canonical_hash(payload)
        if group_hash != reference["content_group_sha256"]:
            raise ValueError(f"content group drifted at row {old_number}")
        split = reference["provisional_split"]
        # Agent-tool rows keep their tool catalog in the payload; the content
        # group hash covers messages only.
        self.payload_lines[split].append(payload_line(payload, row["tools"] if agent_tool else None))
        self.group_sets[split].add(group_hash)
        source_label = self.indexes.internal_source[old_number]
        self.source_counts[source_label] += 1
        identity = {
            "candidate_row_number_1based": self.next_number,
            "candidate_row_sha256": row_sha,
            "content_group_sha256": group_hash,
        }
        self.internal_records.append({**identity, "split": split, "source_label": source_label})
        self.rowref_records.append({**identity, "provisional_split": split})

    def append_v9(self) -> None:
        """Append the v9 addendum as train rows, matching the v3..v8 convention."""
        self.v9_rows = v9_candidate_rows()
        for candidate, payload in self.v9_rows:
            line = (json.dumps(candidate, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
            row_sha = line_sha(line)
            self.next_number += 1
            if self.candidate_stream is not None:
                self.candidate_stream.write(line)
            group_hash = canonical_hash(payload)
            self.payload_lines["train"].append(payload_line(payload, candidate["tools"]))
            self.group_sets["train"].add(group_hash)
            self.source_counts[V9_SOURCE_LABEL] += 1
            identity = {
                "candidate_row_number_1based": self.next_number,
                "candidate_row_sha256": row_sha,
                "content_group_sha256": group_hash,
            }
            self.internal_records.append({**identity, "split": "train", "source_label": V9_SOURCE_LABEL})
            self.rowref_records.append({**identity, "provisional_split": "train"})

    def summary(self) -> dict[str, Any]:
        return {
            "release_id": RELEASE_ID,
            "candidate_rows": self.next_number,
            "carried_v13_rows": self.carried,
            "purged_rows": self.purged,
            "appended_v9_rows": len(self.v9_rows),
            "partitions": {name: len(self.payload_lines[name]) for name in SPLITS},
            "source_groups": {name: len(self.group_sets[name]) for name in SPLITS},
            "total_batch": math.ceil(len(self.payload_lines["train"]) / BATCH_SIZE),
            "dry_run": self.dry_run,
        }

    def write(self) -> None:
        for name in SPLITS:
            (self.output / f"{name}.jsonl").write_text("\n".join(self.payload_lines[name]) + "\n", encoding="utf-8")
        write_jsonl(self.output / "manifest_internal.jsonl", self.internal_records)
        write_jsonl(self.output / "rowrefs.jsonl", self.rowref_records)
        write_jsonl(self.output / "heldout.jsonl", self.heldout_records)

    def manifest(self) -> dict[str, Any]:
        candidate_sha = sha256_file(self.output / "candidate.jsonl")
        membership = {name: set() for name in SPLITS}
        for record in self.internal_records:
            membership[record["split"]].add(record["content_group_sha256"])
        heldout_groups = {r["content_group_sha256"] for r in self.heldout_records}
        retained_groups = {
            record["content_group_sha256"] for record in self.rowref_records if record["provisional_split"] != "heldout"
        }
        partitions = {
            name: {
                "path": str((self.output / f"{name}.jsonl").relative_to(ROOT)),
                "sha256": sha256_file(self.output / f"{name}.jsonl"),
                "row_count": len(self.payload_lines[name]),
                "source_group_count": len(self.group_sets[name]),
                "format": "weighted_messages_jsonl_v1",
                "split": name,
            }
            for name in SPLITS
        }
        scope = {
            "use": "full_candidate_research_fine_tuning_experiment",
            "clinical_or_production_readiness": False,
            "candidate_rows": self.next_number,
            "retained_rows": len(self.internal_records),
            "dropped_structurally_untrainable_rows": len(self.heldout_records),
            "dropped_rows_by_reason": dict(
                sorted(Counter(record["exclusion_reason"] for record in self.heldout_records).items())
            ),
            "purged_rows": self.purged,
            "purged_rows_by_reason": {PURGE_REASON: self.purged},
            "appended_rows": len(self.v9_rows),
            "repaired_rows": len(self.indexes.repairs),
            "source_groups": len(retained_groups),
        }
        return {
            "schema_version": "g04-release-manifest-v1",
            "release_id": RELEASE_ID,
            "status": "approved_for_research_training",
            "upload_gate": True,
            "provider": "wandb",
            "created_at_utc": "2026-11-06T00:00:00Z",
            "source_candidate": {
                "path": str((self.output / "candidate.jsonl").relative_to(ROOT)),
                "sha256": candidate_sha,
                "row_count": self.next_number,
            },
            "source_candidate_sha256": candidate_sha,
            "internal_manifest": {
                "path": str((self.output / "manifest_internal.jsonl").relative_to(ROOT)),
                "sha256": sha256_file(self.output / "manifest_internal.jsonl"),
                "row_count": len(self.internal_records),
            },
            "partitions": partitions,
            "evidence": {
                "group_disjointness": {
                    "status": "verified",
                    "group_key": "content_group_sha256",
                    "train_group_count": len(membership["train"]),
                    "validation_group_count": len(membership["validation"]),
                    "test_group_count": len(membership["test"]),
                    "pairwise_intersections": {
                        "train_validation": len(membership["train"] & membership["validation"]),
                        "train_test": len(membership["train"] & membership["test"]),
                        "validation_test": len(membership["validation"] & membership["test"]),
                    },
                    "evidence_path": str((self.output / "rowrefs.jsonl").relative_to(ROOT)),
                    "evidence_sha256": sha256_file(self.output / "rowrefs.jsonl"),
                    "reserved_eval_group_count": len(heldout_groups),
                    "heldout_group_intersections": {name: len(heldout_groups & membership[name]) for name in SPLITS},
                    "heldout_exclusions_path": str((self.output / "heldout.jsonl").relative_to(ROOT)),
                    "heldout_exclusions_sha256": sha256_file(self.output / "heldout.jsonl"),
                },
                "source_origin": {
                    "status": "full_candidate_rowrefs_verified",
                    "source_label_count": len(self.source_counts),
                    "source_counts": dict(sorted(self.source_counts.items())),
                },
            },
            "scope": scope,
            "repairs": {
                "path": str(REPAIRS.relative_to(ROOT)),
                "sha256": REPAIRS_SHA256,
                "row_count": len(self.indexes.repairs),
                "note": "Repairs apply to candidate rows 1..190244 only; agent_tool rows require no repairs.",
            },
            "payload_contract": {
                "format": "weighted_messages_jsonl_v1",
                "user_weight": 0,
                "assistant_weight": 1,
                "loss_mask": "assistant-turn-only",
                "projection": (
                    "chat rows: system messages removed, PII redacted, trailing turns after final assistant removed; "
                    "agent_tool rows: system message retained (defines tool-use context), tool roles present"
                ),
                "agent_tool_note": (
                    f"2,807 v13 agent_tool rows plus {len(self.v9_rows)} v9 rows include assistant tool_calls, "
                    "role:tool turns, and a retained system message that defines tool-use context."
                ),
                "shuffle": {"train": True, "validation": False, "test": False, "seed": SHUFFLE_SEED},
            },
            "limitations": [
                "Research-only SFT experiment; not clinical or production readiness.",
                "The automated identifier screen does not certify de-identification.",
                "Exact-message group split; semantic near-duplicate certification was not run.",
                "Existing source policy dispositions are preserved in the internal row membership manifest.",
                "Fifty generated Psy-Insight continuations are research data and have not received human "
                "clinical review.",
                "v5 train excludes placeholder/prompt-echo assistant targets and caps exact duplicate "
                "assistant targets at 50 per target; excluded rows are held out, not deleted.",
                "v14 purges all 1,522 crisis_benchmark rows whose assistant target was the benchmark task "
                "description rather than a response; those rows carry no supervision and were never usable.",
                "v14 crisis-routing and bare-literal tool rows are first-party synthetic and have not "
                "received human clinical review.",
            ],
            "owner_direction": (
                "User-directed v14 after the v13 defect sweep. v13 inherits 1,522 crisis_benchmark rows whose "
                "assistant target is the benchmark task description, so the release carries no crisis-routing "
                "supervision and the held-out crisis suite drops from 8/8 (base) to 1/8 (adapter). v13 also welds "
                "the field label onto tool argument values, scoring 2/6 on the literal-extraction suite. v14 purges "
                "every fabricated crisis row candidate-wide and appends 624 v9 train rows that teach bare-literal "
                "tool arguments and crisis routing."
            ),
        }


def build(output: Path, manifest_path: Path, dry_run: bool) -> dict[str, Any]:
    verify_pinned_identity()
    release = _Release(load_indexes(), output, dry_run)
    release.run()
    summary = release.summary()
    if dry_run:
        return summary
    release.write()
    manifest = release.manifest()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary["manifest"] = str(manifest_path)
    summary["redaction_counts"] = dict(release.redaction_counts)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    summary = build(args.output, args.manifest, args.dry_run)
    for key, value in summary.items():
        sys.stdout.write(f"{key}: {value}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
