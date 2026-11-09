"""Build the v16 G04 research release on top of the v15 lineage.

The v15 paired eval left exactly one capability failure that both arms shared:
the ``memory_write`` case asks for ``short bullet list`` and both the base model
and the adapter store ``bullet list``. The value is copied verbatim into the tool
argument, so a three-word value with a load-bearing leading modifier loses that
modifier on the way into the call.

v16 keeps the whole v15 pipeline — the candidate-level welded-argument and
reasoning-preamble repairs, the v9 and v10 addenda, the crisis purge, the split
recipe — and appends two hand-written families to train: the 240-row v11
``memory_value_verbatim`` family, which teaches the tool argument to carry a
multi-word value word for word, and the 672-row v12 ``crisis_surface`` family,
which widens the crisis routing surface from six fixed disclosures per category
to twenty paraphrases per category in three resource-first shapes.

v16 also closes the v14 limitation that validation and test were never
duplicate-filtered. Those splits carried up to 1,242 rows of a single
single-token diagnosis target, which made validation loss a measure of the
label mix rather than of the model. The cap is now an admission rule on all
three splits: a retained validation or test chat row whose exact final
assistant target already has 50 lower-hash rows is purged before it reaches the
candidate payload, the internal ledger, or the content-group sets, exactly as
the crisis placeholders are. Because the purge happens at admission, every G04
invariant still holds — the emitted split files, the internal manifest, the
rowrefs and the group counts all describe the same surviving rows.

The pipeline is inherited rather than re-implemented, so the v16 candidate is the
v15 candidate minus the purged held-out rows plus the v11 and v12 rows at the
end, and the train payload stays identical to v15's.

Usage::

    PYTHONPATH=/home/vivi/pixelated python -m training.build_v16_release --dry-run
    PYTHONPATH=/home/vivi/pixelated python -m training.build_v16_release
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from training import (
    build_agent_tool_sft_v11 as v11,
    build_agent_tool_sft_v12 as v12,
    build_v14_release as v14,
    build_v15_release as v15,
)

ROOT = v15.ROOT

OUTPUT = ROOT / "ai/data/curated/recovery/gold-20261017/wandb-full-corpus-current-master-20261108-v16"
MANIFEST = ROOT / "docs/training/execution-2026-11-08/wandb-full-corpus-current-master-20261108-v16.json"

RELEASE_ID = "wandb-full-corpus-current-master-20261108-v16"
SHUFFLE_SEED = 20261108
SPLITS = v15.SPLITS
CREATED_AT_UTC = "2026-11-08T00:00:00Z"

V11_SOURCE_LABEL = "agent_tool_synthetic_v11"
EXPECTED_V11_TRAIN_ROWS = 240

V12_SOURCE_LABEL = "crisis_surface_v12"
EXPECTED_V12_TRAIN_ROWS = 672

# The held-out splits are admitted under the same rule v5 applied to train: at
# most this many rows may share one exact final assistant target. Rows past the
# cap are purged before admission, so the payloads, the internal ledger and the
# group sets never disagree about which rows exist.
EVALUATION_DUPLICATE_TARGET_CAP = v14.DUPLICATE_TARGET_CAP
EVALUATION_SPLITS = ("validation", "test")
EVALUATION_PURGE_REASON = "duplicate_final_target_over_cap"
# Filled from the verified build: the cap removes 4,672 validation and 4,454
# test rows, the ones whose content group hashes sort above the 50th row of
# their final-target bucket.
EXPECTED_EVALUATION_PURGED = {"validation": 4672, "test": 4454}

# v14 recorded that validation and test were never duplicate-filtered. v16
# fixes that, so the sentence is dropped rather than inherited.
_STALE_EVALUATION_LIMITATION = "Validation and test were never duplicate-filtered"

LIMITATIONS_V16: tuple[str, ...] = (
    (
        "The v11 memory-value family covers 12 written-output keys and 12 multi-word values, all in the "
        "singular noun-phrase shape the eval case uses. Values in other grammatical shapes, and keys outside "
        "that pool, are still only covered by whatever the v4 opt-in rows happen to contain."
    ),
    (
        "The v11 rows are appended to train only. Validation and test carry no memory-value rows, so an "
        "adapter that memorises the v11 pool instead of copying the value verbatim would not be caught by the "
        "held-out splits; only the paired eval cases probe that."
    ),
    (
        "The v12 crisis-surface rows are appended to train only and are paraphrases written by hand against "
        "the v9 categories, not against the evaluation cases. Breadth improved on the two categories v15 "
        "failed, but a disclosure phrased outside all twenty registers of its category can still fall back to "
        "the reflective answer the therapy corpus teaches."
    ),
    (
        "Validation and test are purged at admission to at most 50 rows per exact final assistant target, which "
        "brings the held-out splits under the same rule train has used since v5. That rule keeps the rows whose "
        "content group hashes are lowest, so the surviving splits are a deterministic but arbitrary sample of "
        "each over-capped target: validation and test are no longer a superset of the source data, and a target "
        "that was over-represented before is now represented by its 50 lowest-hash rows rather than its most "
        "typical ones."
    ),
    (
        "The purge removes 4,672 validation rows and 4,454 test rows, so the held-out splits are roughly half "
        "the size of v15's. Almost all of that loss is single-token diagnosis labels: 11 exact final targets "
        "exceeded the cap in each split before the purge. Per-target counts before and after are recorded under "
        "evidence.evaluation_duplicate_target_purge in the manifest, and the removed rows are counted under "
        "scope.purged_rows_by_reason."
    ),
)


def _limitations() -> tuple[str, ...]:
    """The inherited limitations minus the v14 one v16 fixes, plus the v16 ones."""
    kept = tuple(text for text in v15.LIMITATIONS if not text.startswith(_STALE_EVALUATION_LIMITATION))
    if len(kept) != len(v15.LIMITATIONS) - 1:
        raise ValueError("the v14 evaluation-split limitation is no longer in the inherited list")
    return (*kept, *LIMITATIONS_V16)


def _projected_target(
    indexes: v14._Indexes, old_number: int, row: dict[str, Any], scratch: Counter[str]
) -> tuple[str, str, bool]:
    """Return the projected final assistant target, content-group hash, and repair flag."""
    repaired_row, kind = v15.repair_row(row)
    repaired = kind is not None
    messages = repaired_row.get("messages") if repaired else row.get("messages")
    repair = indexes.repairs.get(old_number)
    if repair is not None:
        messages = repair["messages"]
    if not isinstance(messages, list):
        raise ValueError(f"retained row {old_number} has no message list to project")
    payload = v14.project_chat(messages, scratch)
    if payload is None:
        raise ValueError(f"retained row {old_number} projects to no assistant target")
    # project_chat truncates after the last assistant turn, so the final element
    # is the row's supervision target.
    return payload[-1]["content"], v14.canonical_hash(payload), repaired


def _evaluation_targets(indexes: v14._Indexes) -> dict[str, dict[str, list[tuple[str, int]]]]:
    """Bucket every retained held-out row by its exact final assistant target.

    Crisis placeholders and agent-tool rows are excluded: the first are purged
    outright and the second are exempt from the cap, matching the train rule.
    The split is the one v15's admission rule will actually use, so a repaired
    row is bucketed by its new content group rather than by its v13 label.
    """
    by_target: dict[str, dict[str, list[tuple[str, int]]]] = {name: {} for name in EVALUATION_SPLITS}
    scratch: Counter[str] = Counter()
    with (v14.V13_DIR / "candidate.jsonl").open("rb") as source:
        for old_number, raw in enumerate(source, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if v14.is_crisis_placeholder(row) or v14.is_agent_tool(row):
                continue
            reference = indexes.references.get(old_number)
            if reference is None:
                continue
            target, group_hash, repaired = _projected_target(indexes, old_number, row, scratch)
            split = v15.assign_split(group_hash) if repaired else reference["provisional_split"]
            if split not in EVALUATION_SPLITS:
                continue
            by_target[split].setdefault(target, []).append((group_hash, old_number))
    return by_target


def _purge_split(
    split: str, buckets: dict[str, list[tuple[str, int]]], cap: int
) -> tuple[dict[int, str], dict[str, Any]]:
    """Decide one split's purges, keeping the lowest content-group hashes."""
    rows_before = sum(len(bucket) for bucket in buckets.values())
    dropped: Counter[str] = Counter()
    purge_plan: dict[int, str] = {}
    for target, bucket in buckets.items():
        if len(bucket) <= cap:
            continue
        bucket.sort()
        for _, old_number in bucket[cap:]:
            purge_plan[old_number] = split
            dropped[target] += 1
    over_cap = {target: len(bucket) for target, bucket in buckets.items() if len(bucket) > cap}
    dropped_rows = sum(dropped.values())
    stats = {
        "rows_before": rows_before,
        "rows_after": rows_before - dropped_rows,
        "max_rows_per_final_target_before": max((len(bucket) for bucket in buckets.values()), default=0),
        "targets_over_cap": len(over_cap),
        "rows_in_over_cap_targets": sum(over_cap.values()),
        "purged_rows": dropped_rows,
        "purged_targets": len(dropped),
        "purged_by_target": dict(sorted(dropped.items(), key=lambda item: (-item[1], item[0]))),
    }
    if rows_before != stats["rows_after"] + dropped_rows:
        raise ValueError(f"{split}: the purge is not a partition of the split")
    return purge_plan, stats


def plan_evaluation_purge(
    indexes: v14._Indexes, cap: int = EVALUATION_DUPLICATE_TARGET_CAP
) -> tuple[dict[int, str], dict[str, dict[str, Any]]]:
    """Decide which retained validation/test rows exceed the per-target cap.

    The cap is the admission rule v5 wrote for train, so it has to be decided
    before a row is admitted rather than after the split file is written: the
    payload, the internal ledger, the rowrefs and the content-group sets must
    all describe the same surviving rows or the G04 gate rejects the release.

    The key is the verbatim final assistant target of the *projected* payload,
    which is what ``duplicate_target_profile`` measures. Agent-tool rows are
    exempt, matching the train rule. Ties break on the candidate row number so
    the plan is order-independent. Returns the purged row numbers keyed to
    their split, plus the before/after per-split statistics.
    """
    by_target = _evaluation_targets(indexes)
    purge_plan: dict[int, str] = {}
    stats: dict[str, dict[str, Any]] = {}
    for split in EVALUATION_SPLITS:
        split_plan, stats[split] = _purge_split(split, by_target[split], cap)
        purge_plan.update(split_plan)
    return purge_plan, stats


def v11_candidate_rows() -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """The v11 train rows as (candidate row, projected payload messages)."""
    rows = v11.collect_rows()["train"]
    if len(rows) != EXPECTED_V11_TRAIN_ROWS:
        raise ValueError(f"expected {EXPECTED_V11_TRAIN_ROWS} v11 train rows, found {len(rows)}")
    out: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for item in rows:
        candidate = {"messages": item["messages"], "tools": item["tools"]}
        if not v14.is_agent_tool(candidate):
            raise ValueError(f"{item['id']}: v11 candidate rows must carry tools and no source label")
        out.append((candidate, v14.project_agent(item["messages"])))
    return out


def v12_candidate_rows() -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """The v12 train rows as (candidate row, projected payload messages).

    These are chat rows, not agent rows: no tool catalog, so ``is_agent_tool``
    stays false and the payload keeps the plain ``messages`` body.
    """
    rows = v12.collect_rows()["train"]
    if len(rows) != EXPECTED_V12_TRAIN_ROWS:
        raise ValueError(f"expected {EXPECTED_V12_TRAIN_ROWS} v12 train rows, found {len(rows)}")
    out: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for item in rows:
        candidate = {"messages": item["messages"], "source": V12_SOURCE_LABEL}
        if v14.is_agent_tool(candidate):
            raise ValueError(f"{item['id']}: v12 candidate rows must not carry a tool catalog")
        # The rows are hand-written, so nothing should redact. Projecting against a
        # throwaway counter keeps the release-wide redaction counts honest: if a
        # v12 row ever trips a pattern, this fails instead of silently hiding it.
        scratch: Counter[str] = Counter()
        payload = v14.project_chat(item["messages"], scratch)
        if payload is None:
            raise ValueError(f"{item['id']}: v12 row projects to no assistant target")
        if scratch and any(scratch.values()):
            raise ValueError(f"{item['id']}: v12 row redacted {dict(scratch)}; synthetic rows must be clean")
        out.append((candidate, payload))
    return out


def stamp_release_identity(
    data: dict[str, Any],
    *,
    v9_rows: int,
    v10_rows: int,
    v11_rows: int,
    v12_rows: int,
) -> dict[str, Any]:
    """Rewrite the addenda-dependent manifest fields for the v16 release."""
    appended_by_source = {
        v14.V9_SOURCE_LABEL: v9_rows,
        v15.V10_SOURCE_LABEL: v10_rows,
        V11_SOURCE_LABEL: v11_rows,
        V12_SOURCE_LABEL: v12_rows,
    }
    data["release_id"] = RELEASE_ID
    data["created_at_utc"] = CREATED_AT_UTC
    data["scope"]["appended_rows"] = sum(appended_by_source.values())
    data["scope"]["appended_rows_by_source"] = appended_by_source
    data["payload_contract"]["shuffle"] = {
        "train": True,
        "validation": False,
        "test": False,
        "seed": SHUFFLE_SEED,
    }
    data["payload_contract"]["agent_tool_note"] = (
        f"2,807 v13 agent_tool rows plus {v9_rows} v9 rows plus {v10_rows} v10 rows plus {v11_rows} v11 rows "
        "include assistant tool_calls, role:tool turns, and a retained system message that defines tool-use "
        "context."
    )
    data["limitations"] = list(_limitations())
    data["owner_direction"] = (
        "User-directed v16 after the v15 paired eval. v15 repaired the welded tool arguments and the reasoning "
        "preambles at the candidate and beat both the base model and v14 on every suite, but two gaps survived: "
        "the memory_write case failed on both arms because a three-word value with a load-bearing leading "
        "modifier collapsed to its head noun in the tool argument, and the crisis suite lost self-harm-active "
        "and eating-disorder-syncope to a warm reflection with no routing. v16 appends the 240-row v11 "
        "memory_value_verbatim family so the argument carries the value word for word, and the 672-row v12 "
        "crisis_surface family so the routing answer is reachable from twenty paraphrases per category instead "
        "of the six fixed disclosures v9 and v10 repeated. v16 also purges validation and test down to 50 rows "
        "per exact final assistant target at admission, so validation loss stops being dominated by a handful of "
        "single-token diagnosis labels and starts tracking the same distribution train is admitted under."
    )
    return data


class _Release(v15._Release):
    """v15's build, with the held-out cap, the v11 and v12 addenda, and the v16 seed."""

    def __init__(self, indexes: v14._Indexes, output: Path, dry_run: bool) -> None:
        super().__init__(indexes, output, dry_run)
        self.v11_rows: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        self.v12_rows: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        self.purge_plan: dict[int, str] = {}
        self.evaluation_purge_stats: dict[str, dict[str, Any]] = {}
        self.evaluation_purged: Counter[str] = Counter()

    def run(self) -> None:
        """v15's run, with the held-out cap decided up front and the v16 seed."""
        self.purge_plan, self.evaluation_purge_stats = self._plan_purge()
        self.output.mkdir(parents=True, exist_ok=True)
        self.candidate_stream = None if self.dry_run else (self.output / "candidate.jsonl").open("wb")
        try:
            with (v15.V13_DIR / "candidate.jsonl").open("rb") as source:
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

    def _plan_purge(self) -> tuple[dict[int, str], dict[str, dict[str, Any]]]:
        """Plan the held-out duplicate-target purge before any row is admitted."""
        return plan_evaluation_purge(self.indexes)

    def carry(self, old_number: int, raw: bytes) -> None:
        """Purge over-capped held-out rows before v15 admits them.

        The row never reaches the candidate payload, the internal ledger, the
        rowrefs or the content-group sets, so every downstream count agrees.
        """
        split = self.purge_plan.get(old_number)
        if split is not None:
            self.evaluation_purged[split] += 1
            return
        super().carry(old_number, raw)

    def verify_totals(self) -> None:
        """v15's totals, plus the v11/v12 addenda and the held-out purge."""
        if self.purged != v15.EXPECTED_PURGED_ROWS:
            raise ValueError(f"expected {v15.EXPECTED_PURGED_ROWS} purged rows, found {self.purged}")
        if dict(self.repair_counts) != v15.EXPECTED_REPAIR_COUNTS:
            raise ValueError(f"repair population drifted: {dict(self.repair_counts)}")
        appended = len(self.v9_rows) + len(self.v10_rows) + len(self.v11_rows) + len(self.v12_rows)
        if self.carried + appended != self.next_number:
            raise ValueError("row numbering drifted")
        purged = {name: self.evaluation_purged[name] for name in EVALUATION_SPLITS}
        if purged != EXPECTED_EVALUATION_PURGED:
            raise ValueError(f"held-out purge drifted: {purged}, expected {EXPECTED_EVALUATION_PURGED}")
        for split in EVALUATION_SPLITS:
            emitted = len(self.payload_lines[split])
            planned = self.evaluation_purge_stats[split]["rows_after"]
            if emitted != planned:
                raise ValueError(f"{split}: emitted {emitted} rows but the purge plan expected {planned}")

    def append_addenda(self) -> None:
        """Append the v9, v10, v11, and v12 addenda as train rows."""
        super().append_addenda()
        self.v11_rows = v11_candidate_rows()
        for candidate, payload in self.v11_rows:
            self.append_agent_row(V11_SOURCE_LABEL, candidate, payload)
        self.v12_rows = v12_candidate_rows()
        for candidate, payload in self.v12_rows:
            self.append_chat_row(V12_SOURCE_LABEL, candidate, payload)

    def append_chat_row(self, source_label: str, candidate: dict[str, Any], payload: list[dict[str, Any]]) -> None:
        """Append one chat-shaped train row: no tool catalog, so no ``tools`` key."""

        line = v15.json_bytes(candidate)
        row_sha = v14.line_sha(line)
        self.next_number += 1
        if self.candidate_stream is not None:
            self.candidate_stream.write(line)
        group_hash = v14.canonical_hash(payload)
        self.payload_lines["train"].append(v14.payload_line(payload, None))
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
        data = super().summary()
        data["release_id"] = RELEASE_ID
        data["appended_v11_rows"] = len(self.v11_rows)
        data["appended_v12_rows"] = len(self.v12_rows)
        data["evaluation_purged_rows"] = {name: self.evaluation_purged[name] for name in EVALUATION_SPLITS}
        return data

    def manifest(self) -> dict[str, Any]:
        data = stamp_release_identity(
            super().manifest(),
            v9_rows=len(self.v9_rows),
            v10_rows=len(self.v10_rows),
            v11_rows=len(self.v11_rows),
            v12_rows=len(self.v12_rows),
        )
        # The held-out purges are admission decisions, so the inherited
        # membership, partition and group-count evidence already describes the
        # surviving rows exactly. Only the purge accounting is restated.
        evaluation_purged = sum(self.evaluation_purged.values())
        data["scope"]["purged_rows"] = self.purged + evaluation_purged
        data["scope"]["purged_rows_by_reason"] = {
            v14.PURGE_REASON: self.purged,
            EVALUATION_PURGE_REASON: evaluation_purged,
        }
        data["evidence"]["duplicate_target_concentration"] = self.duplicate_profiles
        data["evidence"]["evaluation_duplicate_target_purge"] = {
            "cap": EVALUATION_DUPLICATE_TARGET_CAP,
            "rule": "keep the 50 lowest content_group_sha256 rows per exact final assistant target",
            "agent_tool_rows_exempt": True,
            "applies_to": list(EVALUATION_SPLITS),
            "reason": EVALUATION_PURGE_REASON,
            "splits": self.evaluation_purge_stats,
            "after": {
                name: self.duplicate_profiles.get(name, {}).get("final_target", {}) for name in EVALUATION_SPLITS
            },
        }
        return data


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
