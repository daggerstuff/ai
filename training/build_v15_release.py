"""Build the v15 G04 research release on top of the v13 lineage.

v14 fixed the crisis-routing data source, but three candidate-level defects
survived it, and the v14 paired eval measured all three:

* 192 ``agent_tool_synthetic_v7`` rows weld the field label onto the tool
  argument (``{"change": "announcement to \\"Meeting moved\\""}``). The v9
  addendum teaches the bare literal, so v14 trains two contradictory answers
  for the same tool and only recovers one literal-extraction case out of six.
* 662 rows open their final assistant turn with a raw reasoning block. The
  answer is there, behind the block, so the adapter learns to emit scratchpad
  prose as part of the reply.
* 3 rows are nothing but an unterminated reasoning block: no supervision at
  all.

v15 repairs all three *at the candidate*, so every later release inherits the
fix instead of re-applying it at projection time:

* R1 ``welded_tool_argument``: the tool argument carries the bare literal. The
  user prompt and the assistant prose stay byte-identical to v7.
* R2 ``reasoning_preamble_stripped``: the leading reasoning block is removed
  from the final assistant turn and the answer is kept.
* R2 ``reasoning_preamble_dropped_turn``: for the 3 open-only rows the trailing
  user/assistant exchange is dropped, because the supervised turn carries no
  answer to keep.

It also appends the 480-row v10 resource-first crisis family on top of the v9
rows v14 added. Every unrepaired v13 row keeps its exact candidate line, its
``candidate_row_sha256``, its ``content_group_sha256``, and its split. Repaired
rows get a new ``candidate_row_sha256`` and a new ``content_group_sha256``, and
are re-bucketed from the new group hash: chat rows by the same bucket formula
that produced v13, agent-tool rows forced to train by provenance.

Usage::

    PYTHONPATH=/home/vivi/pixelated python -m training.build_v15_release --dry-run
    PYTHONPATH=/home/vivi/pixelated python -m training.build_v15_release
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

from training import build_agent_tool_sft_v7 as v7, build_agent_tool_sft_v10 as v10, build_v14_release as v14

ROOT = v14.ROOT
V13_DIR = v14.V13_DIR
REPAIRS = v14.REPAIRS

OUTPUT = ROOT / "ai/data/curated/recovery/gold-20261017/wandb-full-corpus-current-master-20261107-v15"
MANIFEST = ROOT / "docs/training/execution-2026-11-07/wandb-full-corpus-current-master-20261107-v15.json"

RELEASE_ID = "wandb-full-corpus-current-master-20261107-v15"
SHUFFLE_SEED = 20261107
BATCH_SIZE = 16
SPLITS = v14.SPLITS
CREATED_AT_UTC = "2026-11-07T00:00:00Z"

V10_SOURCE_LABEL = "agent_tool_synthetic_v10"
EXPECTED_V9_TRAIN_ROWS = v14.EXPECTED_V9_TRAIN_ROWS
EXPECTED_V10_TRAIN_ROWS = 480
EXPECTED_PURGED_ROWS = v14.EXPECTED_PURGED_ROWS

# The split recipe v13 used, restated here because v15 has to re-derive it for
# repaired rows: the first four bytes of the content group hash, modulo 100.
SPLIT_BUCKETS = 100
TRAIN_SPLIT_BUCKET = 90
VALIDATION_SPLIT_BUCKET = 95

REPAIR_WELDED = "welded_tool_argument"
REPAIR_STRIPPED = "reasoning_preamble_stripped"
REPAIR_DROPPED = "reasoning_preamble_dropped_turn"

# Measured against the v13 candidate. The build refuses to run if the repaired
# population drifts, so a release can never claim a repair it did not perform.
EXPECTED_REPAIR_COUNTS: dict[str, int] = {
    REPAIR_WELDED: 192,
    REPAIR_STRIPPED: 662,
    REPAIR_DROPPED: 3,
}

# A row is only droppable if a user turn and its assistant reply both go.
MIN_DROPPABLE_MESSAGES = 3

THINK_OPEN = re.compile(r"^\s*<\s*think\b", re.IGNORECASE)
THINK_CLOSE = re.compile(r"<\s*/\s*think\s*>", re.IGNORECASE)
THINK_BLOCK = re.compile(r"<\s*think\b[^>]*>.*?<\s*/\s*think\s*>", re.DOTALL | re.IGNORECASE)

# The welded value v7 emitted, mapped back to the bare literal it wrapped.
WELDED_TO_LITERAL: dict[str, str] = {v7.describe(label, literal): literal for label, literal in v7.CHANGES}

LIMITATIONS: tuple[str, ...] = (
    *v14.LIMITATIONS,
    "v15 repairs 192 agent-tool candidate rows whose submit_mock_change argument carried the field "
    "label welded onto the literal; the argument now carries the bare literal while the user prompt "
    "and the assistant prose are unchanged.",
    "v15 strips a leading reasoning block from 662 candidate rows and drops the trailing exchange "
    "from 3 rows whose final assistant turn was nothing but an unterminated reasoning block; the "
    "supervised answers themselves are unchanged.",
    "Repaired rows are re-bucketed from their new content group hash, so a repaired row can sit in a "
    "different partition than it did in v13.",
    "v15 appends the 480-row resource-first crisis family on top of the v9 rows v14 added; like the "
    "rest of the synthetic crisis and tool families it has not received human clinical review.",
)


def assign_split(group_hash: str) -> str:
    """The v13 bucket recipe: train under 90, validation under 95, else test."""
    bucket = int(group_hash[:8], 16) % SPLIT_BUCKETS
    if bucket < TRAIN_SPLIT_BUCKET:
        return "train"
    if bucket < VALIDATION_SPLIT_BUCKET:
        return "validation"
    return "test"


def json_bytes(row: dict[str, Any]) -> bytes:
    return (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def repair_welded_arguments(messages: list[Any]) -> tuple[list[Any], bool]:
    """Rewrite welded tool arguments to the bare literal, leaving prose alone."""
    repaired: list[Any] = []
    changed = False
    for message in messages:
        if not isinstance(message, dict) or not message.get("tool_calls"):
            repaired.append(message)
            continue
        calls: list[Any] = []
        for call in message["tool_calls"]:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if not isinstance(arguments, (str, bytes, bytearray)):
                calls.append(call)
                continue
            try:
                values = json.loads(arguments)
            except ValueError:
                calls.append(call)
                continue
            if not isinstance(values, dict):
                calls.append(call)
                continue
            unwrapped = {
                key: WELDED_TO_LITERAL.get(value, value) if isinstance(value, str) else value
                for key, value in values.items()
            }
            if unwrapped == values:
                calls.append(call)
                continue
            changed = True
            # v13 serialized arguments with the default separators; keep them.
            calls.append({**call, "function": {**function, "arguments": json.dumps(unwrapped)}})
        repaired.append({**message, "tool_calls": calls})
    return repaired, changed


def drop_trailing_exchange(messages: list[Any]) -> list[Any]:
    """Remove the final user turn and its answer-less assistant reply."""
    if len(messages) < MIN_DROPPABLE_MESSAGES:
        raise ValueError("reasoning-only row has no exchange left to drop")
    trailing_user, trailing_assistant = messages[-2], messages[-1]
    if trailing_assistant.get("role") != "assistant" or trailing_user.get("role") != "user":
        raise ValueError("reasoning-only row does not end with a user/assistant exchange")
    dropped = messages[:-2]
    if not any(isinstance(message, dict) and message.get("role") == "user" for message in dropped):
        raise ValueError("dropping the trailing exchange leaves no user turn")
    return dropped


def repair_reasoning_preamble(messages: list[Any]) -> tuple[list[Any], str | None]:
    """Strip a leading reasoning block from the final assistant turn."""
    positions = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    if not positions:
        return messages, None
    final = positions[-1]
    content = messages[final].get("content")
    if not isinstance(content, str) or not THINK_OPEN.match(content):
        return messages, None
    stripped = THINK_BLOCK.sub("", content).strip() if THINK_CLOSE.search(content) else ""
    if not stripped:
        return drop_trailing_exchange(messages), REPAIR_DROPPED
    repaired = list(messages)
    repaired[final] = {**messages[final], "content": stripped}
    return repaired, REPAIR_STRIPPED


def repair_row(row: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Apply the candidate-level repair a row needs, if any."""
    messages = row.get("messages")
    if not isinstance(messages, list):
        return row, None
    if v14.is_agent_tool(row):
        repaired, changed = repair_welded_arguments(messages)
        return ({**row, "messages": repaired}, REPAIR_WELDED) if changed else (row, None)
    repaired, kind = repair_reasoning_preamble(messages)
    return ({**row, "messages": repaired}, kind) if kind is not None else (row, None)


def v10_candidate_rows() -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """The v10 train rows as (candidate row, projected payload messages)."""
    rows = v10.collect_rows()["train"]
    if len(rows) != EXPECTED_V10_TRAIN_ROWS:
        raise ValueError(f"expected {EXPECTED_V10_TRAIN_ROWS} v10 train rows, found {len(rows)}")
    out: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for item in rows:
        candidate = {"messages": item["messages"], "tools": item["tools"]}
        if not v14.is_agent_tool(candidate):
            raise ValueError(f"{item['id']}: v10 candidate rows must carry tools and no source label")
        out.append((candidate, v14.project_agent(item["messages"])))
    return out


class _Release:
    """One v15 build: purge, repair, carry, append, then write payloads and the manifest."""

    def __init__(self, indexes: v14._Indexes, output: Path, dry_run: bool) -> None:
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
        self.repair_counts: Counter[str] = Counter()
        self.repaired_numbers: set[int] = set()
        self.train_pinned_rows: list[int] = []
        self.v9_rows: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        self.v10_rows: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        self.duplicate_profiles: dict[str, dict[str, Any]] = {}
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
            self.append_addenda()
        finally:
            if self.candidate_stream is not None:
                self.candidate_stream.close()
        self.verify_totals()
        # Deterministic train shuffle; validation and test keep candidate order.
        random.Random(SHUFFLE_SEED).shuffle(self.payload_lines["train"])
        self.audit_duplicate_targets()
        self.verify_group_disjointness()

    def verify_totals(self) -> None:
        if self.purged != EXPECTED_PURGED_ROWS:
            raise ValueError(f"expected {EXPECTED_PURGED_ROWS} purged rows, found {self.purged}")
        if dict(self.repair_counts) != EXPECTED_REPAIR_COUNTS:
            raise ValueError(f"repair population drifted: {dict(self.repair_counts)}")
        appended = len(self.v9_rows) + len(self.v10_rows)
        if self.carried + appended != self.next_number:
            raise ValueError("row numbering drifted")

    def audit_duplicate_targets(self) -> None:
        """Measure duplicate-target concentration and enforce the cap on train."""
        self.duplicate_profiles = {name: v14.duplicate_target_profile(self.payload_lines[name]) for name in SPLITS}
        v14.verify_train_duplicate_policy(self.duplicate_profiles["train"])

    def verify_group_disjointness(self) -> None:
        """Fail the build if any two partitions, or the reserve, share a group."""
        for left, right in combinations(SPLITS, 2):
            overlap = self.group_sets[left] & self.group_sets[right]
            if overlap:
                raise ValueError(f"{left} and {right} share {len(overlap)} content groups")
        heldout_groups = {record["content_group_sha256"] for record in self.heldout_records}
        for name in SPLITS:
            overlap = heldout_groups & self.group_sets[name]
            if overlap:
                raise ValueError(f"held-out rows share {len(overlap)} content groups with {name}")

    def carry(self, old_number: int, raw: bytes) -> None:
        row = json.loads(raw)
        if v14.is_crisis_placeholder(row):
            self.purged += 1
            return
        reference = self.indexes.references.get(old_number)
        held = self.indexes.heldout_source.get(old_number)
        if (reference is None) == (held is None):
            raise ValueError(f"candidate row {old_number} is not exactly one of retained/held out")
        repaired, kind = repair_row(row)
        if kind is not None:
            if held is not None:
                raise ValueError(f"candidate row {old_number} needs a repair but is held out")
            if old_number in self.indexes.repairs:
                raise ValueError(f"candidate row {old_number} is already repaired at projection time")
            self.repair_counts[kind] += 1
            self.repaired_numbers.add(old_number)
        row_sha = self.admit(raw if kind is None else json_bytes(repaired))
        if held is not None:
            self.record_heldout(old_number, row_sha, held)
            return
        assert reference is not None
        self.record_retained(old_number, repaired, row_sha, reference)

    def admit(self, line: bytes) -> str:
        """Write one surviving candidate line under its new number."""
        row_sha = v14.line_sha(line)
        self.next_number += 1
        self.carried += 1
        if self.candidate_stream is not None:
            self.candidate_stream.write(line)
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
        repaired = old_number in self.repaired_numbers
        raw_messages = row.get("messages")
        repair = self.indexes.repairs.get(old_number)
        if repair is not None:
            if repair.get("candidate_row_sha256") != row_sha:
                raise ValueError(f"repair source hash mismatch at row {old_number}")
            raw_messages = repair["messages"]
        if not repaired and row_sha != reference["candidate_row_sha256"]:
            raise ValueError(f"candidate provenance mismatch at row {old_number}")
        if not isinstance(raw_messages, list):
            raise ValueError(f"retained row {old_number} has no message list to project")
        agent_tool = v14.is_agent_tool(row)
        payload = (
            v14.project_agent(raw_messages) if agent_tool else v14.project_chat(raw_messages, self.redaction_counts)
        )
        if payload is None:
            raise ValueError(f"retained row {old_number} projects to no assistant target")
        group_hash = v14.canonical_hash(payload)
        source_label = self.indexes.internal_source[old_number]
        if repaired:
            # A repaired row has a new content group, so it is re-bucketed, except
            # for the agent-tool families v13 admitted to train by provenance.
            split = "train" if agent_tool else assign_split(group_hash)
        else:
            if group_hash != reference["content_group_sha256"]:
                raise ValueError(f"content group drifted at row {old_number}")
            split = reference["provisional_split"]
            bucketed = assign_split(group_hash)
            # v13 pulled a handful of rows forward into train and never pushed one
            # out of its bucket, so anything else means the recipe drifted.
            if split != bucketed:
                if split != "train":
                    raise ValueError(f"row {old_number} sits in {split} ahead of its {bucketed} bucket")
                if not agent_tool:
                    self.train_pinned_rows.append(old_number)
        # Agent-tool rows keep their tool catalog in the payload; the content
        # group hash covers messages only.
        self.payload_lines[split].append(v14.payload_line(payload, row["tools"] if agent_tool else None))
        self.group_sets[split].add(group_hash)
        self.source_counts[source_label] += 1
        identity = {
            "candidate_row_number_1based": self.next_number,
            "candidate_row_sha256": row_sha,
            "content_group_sha256": group_hash,
        }
        self.internal_records.append({**identity, "split": split, "source_label": source_label})
        self.rowref_records.append({**identity, "provisional_split": split})

    def append_addenda(self) -> None:
        """Append the v9 and v10 addenda as train rows, matching the v3..v8 convention."""
        self.v9_rows = v14.v9_candidate_rows()
        self.v10_rows = v10_candidate_rows()
        for source_label, rows in ((v14.V9_SOURCE_LABEL, self.v9_rows), (V10_SOURCE_LABEL, self.v10_rows)):
            for candidate, payload in rows:
                self.append_agent_row(source_label, candidate, payload)

    def append_agent_row(self, source_label: str, candidate: dict[str, Any], payload: list[dict[str, Any]]) -> None:
        line = json_bytes(candidate)
        row_sha = v14.line_sha(line)
        self.next_number += 1
        if self.candidate_stream is not None:
            self.candidate_stream.write(line)
        group_hash = v14.canonical_hash(payload)
        self.payload_lines["train"].append(v14.payload_line(payload, candidate["tools"]))
        self.group_sets["train"].add(group_hash)
        self.source_counts[source_label] += 1
        identity = {
            "candidate_row_number_1based": self.next_number,
            "candidate_row_sha256": row_sha,
            "content_group_sha256": group_hash,
        }
        self.internal_records.append({**identity, "split": "train", "source_label": source_label})
        self.rowref_records.append({**identity, "provisional_split": "train"})

    def summary(self) -> dict[str, Any]:
        train_profile = self.duplicate_profiles.get("train", {})
        return {
            "release_id": RELEASE_ID,
            "candidate_rows": self.next_number,
            "carried_v13_rows": self.carried,
            "purged_rows": self.purged,
            "appended_v9_rows": len(self.v9_rows),
            "appended_v10_rows": len(self.v10_rows),
            "candidate_repaired_rows": sum(self.repair_counts.values()),
            "candidate_repairs_by_kind": dict(sorted(self.repair_counts.items())),
            "train_pinned_chat_rows": len(self.train_pinned_rows),
            "partitions": {name: len(self.payload_lines[name]) for name in SPLITS},
            "source_groups": {name: len(self.group_sets[name]) for name in SPLITS},
            "total_batch": math.ceil(len(self.payload_lines["train"]) / BATCH_SIZE),
            "duplicate_final_targets_over_cap": train_profile.get("final_target", {}).get("targets_over_cap"),
            "duplicate_max_rows_per_final_target": train_profile.get("final_target", {}).get("max_rows_per_target"),
            "dry_run": self.dry_run,
        }

    def write(self) -> None:
        for name in SPLITS:
            (self.output / f"{name}.jsonl").write_text("\n".join(self.payload_lines[name]) + "\n", encoding="utf-8")
        v14.write_jsonl(self.output / "manifest_internal.jsonl", self.internal_records)
        v14.write_jsonl(self.output / "rowrefs.jsonl", self.rowref_records)
        v14.write_jsonl(self.output / "heldout.jsonl", self.heldout_records)

    def manifest(self) -> dict[str, Any]:
        candidate_sha = v14.sha256_file(self.output / "candidate.jsonl")
        membership: dict[str, set[str]] = {name: set() for name in SPLITS}
        for record in self.internal_records:
            membership[record["split"]].add(record["content_group_sha256"])
        heldout_groups = {record["content_group_sha256"] for record in self.heldout_records}
        retained_groups = {record["content_group_sha256"] for record in self.rowref_records}
        partitions = {
            name: {
                "path": str((self.output / f"{name}.jsonl").relative_to(ROOT)),
                "sha256": v14.sha256_file(self.output / f"{name}.jsonl"),
                "row_count": len(self.payload_lines[name]),
                "source_group_count": len(self.group_sets[name]),
                "format": "weighted_messages_jsonl_v1",
                "split": name,
            }
            for name in SPLITS
        }
        appended_by_source = {
            v14.V9_SOURCE_LABEL: len(self.v9_rows),
            V10_SOURCE_LABEL: len(self.v10_rows),
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
            "purged_rows_by_reason": {v14.PURGE_REASON: self.purged},
            "appended_rows": sum(appended_by_source.values()),
            "appended_rows_by_source": appended_by_source,
            "repaired_rows": len(self.indexes.repairs),
            "candidate_repaired_rows": sum(self.repair_counts.values()),
            "candidate_repairs_by_kind": dict(sorted(self.repair_counts.items())),
            "source_groups": len(retained_groups),
        }
        return {
            "schema_version": "g04-release-manifest-v1",
            "release_id": RELEASE_ID,
            "status": "approved_for_research_training",
            "upload_gate": True,
            "provider": "wandb",
            "created_at_utc": CREATED_AT_UTC,
            "source_candidate": {
                "path": str((self.output / "candidate.jsonl").relative_to(ROOT)),
                "sha256": candidate_sha,
                "row_count": self.next_number,
            },
            "source_candidate_sha256": candidate_sha,
            "internal_manifest": {
                "path": str((self.output / "manifest_internal.jsonl").relative_to(ROOT)),
                "sha256": v14.sha256_file(self.output / "manifest_internal.jsonl"),
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
                    "evidence_sha256": v14.sha256_file(self.output / "rowrefs.jsonl"),
                    "reserved_eval_group_count": len(heldout_groups),
                    "heldout_group_intersections": {name: len(heldout_groups & membership[name]) for name in SPLITS},
                    "heldout_exclusions_path": str((self.output / "heldout.jsonl").relative_to(ROOT)),
                    "heldout_exclusions_sha256": v14.sha256_file(self.output / "heldout.jsonl"),
                },
                "source_origin": {
                    "status": "full_candidate_rowrefs_verified",
                    "source_label_count": len(self.source_counts),
                    "source_counts": dict(sorted(self.source_counts.items())),
                },
                "duplicate_target_concentration": self.duplicate_profiles,
                "split_assignment": {
                    "status": "verified",
                    "recipe": "int(content_group_sha256[:8], 16) % 100; train < 90, validation < 95, else test",
                    "note": (
                        "Carried rows reproduce their v13 split; the only exceptions are rows v13 pulled "
                        "forward into train, which are counted here. Repaired rows are re-bucketed."
                    ),
                    "train_pinned_chat_rows": len(self.train_pinned_rows),
                    "train_pinned_row_numbers": self.train_pinned_rows,
                },
                "candidate_repairs": {
                    "status": "applied_at_candidate",
                    "row_count": sum(self.repair_counts.values()),
                    "by_kind": dict(sorted(self.repair_counts.items())),
                    "note": (
                        "Repaired rows carry a new candidate_row_sha256 and content_group_sha256, and are "
                        "re-bucketed from the new group hash; unrepaired rows keep their v13 identity and split."
                    ),
                },
            },
            "scope": scope,
            "repairs": {
                "path": str(REPAIRS.relative_to(ROOT)),
                "sha256": v14.REPAIRS_SHA256,
                "row_count": len(self.indexes.repairs),
                "note": "Repairs apply to candidate rows 1..190244 only; agent_tool rows require no repairs.",
                "candidate_repairs": {
                    "row_count": sum(self.repair_counts.values()),
                    "by_kind": dict(sorted(self.repair_counts.items())),
                },
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
                    f"2,807 v13 agent_tool rows plus {len(self.v9_rows)} v9 rows plus "
                    f"{len(self.v10_rows)} v10 rows include assistant tool_calls, role:tool turns, and a "
                    "retained system message that defines tool-use context."
                ),
                "shuffle": {"train": True, "validation": False, "test": False, "seed": SHUFFLE_SEED},
            },
            "limitations": list(LIMITATIONS),
            "owner_direction": (
                "User-directed v15 after the v14 paired eval. v14 fixed the crisis data source but left three "
                "candidate-level defects: 192 v7 rows weld the field label onto the tool argument, which is why "
                "literal extraction still trails the base model; 662 rows open their final assistant turn with a "
                "raw reasoning block; and 3 rows are nothing but an unterminated reasoning block. v15 repairs all "
                "three at the candidate so later releases inherit the fix, and appends the 480-row v10 "
                "resource-first crisis family so the adapter names the resource instead of only reflecting."
            ),
        }


def build(output: Path, manifest_path: Path, dry_run: bool) -> dict[str, Any]:
    v14.verify_pinned_identity()
    release = _Release(v14.load_indexes(), output, dry_run)
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
