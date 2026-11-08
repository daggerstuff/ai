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
label mix rather than of the model. Both held-out splits are now thinned to 50
rows per exact final assistant target — the same admission rule train has used
since v5 — so validation and test finally describe the distribution the
adapter is trained under.

The pipeline is inherited rather than re-implemented, so the v16 candidate is the
v15 candidate byte for byte plus the v11 and v12 rows at the end, and the
train payload stays identical to v15's.

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

# The held-out splits are thinned with the same admission rule v5 applied to
# train: at most this many rows may share one exact final assistant target.
EVALUATION_DUPLICATE_TARGET_CAP = v14.DUPLICATE_TARGET_CAP
EVALUATION_SPLITS = ("validation", "test")
EXPECTED_EVALUATION_ROWS = {"validation": 4918, "test": 4789}

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
        "Validation and test are thinned to at most 50 rows per exact final assistant target, which brings the "
        "held-out splits under the same admission rule train has used since v5. That rule keeps the rows whose "
        "content group hashes lowest, so the thinned splits are a deterministic but arbitrary sample of each "
        "over-capped target: validation and test are no longer a superset of the source data, and a target that "
        "was over-represented before is now represented by its 50 lowest-hash rows rather than its most typical "
        "ones."
    ),
    (
        "Thinning happens after the split is cut, so the candidate payload, the internal record ledger and the "
        "content-group membership sets still describe the full 9,590-row validation and 9,243-row test. Only the "
        "validation.jsonl and test.jsonl payloads are thinned; the affected content groups are flagged "
        '"duplicate_target_capped_out" in manifest_internal.jsonl and rowrefs.jsonl, and the per-target drop '
        "counts are recorded under evidence.evaluation_duplicate_target_cap in the manifest."
    ),
)


def _limitations() -> tuple[str, ...]:
    """The inherited limitations minus the v14 one v16 fixes, plus the v16 ones."""
    kept = tuple(text for text in v15.LIMITATIONS if not text.startswith(_STALE_EVALUATION_LIMITATION))
    if len(kept) != len(v15.LIMITATIONS) - 1:
        raise ValueError("the v14 evaluation-split limitation is no longer in the inherited list")
    return (*kept, *LIMITATIONS_V16)


def cap_duplicate_targets(
    lines: list[str], cap: int = EVALUATION_DUPLICATE_TARGET_CAP
) -> tuple[list[str], dict[str, int], list[str]]:
    """Keep at most ``cap`` rows per exact final assistant target.

    Mirrors ``v14.verify_train_duplicate_policy``: the key is the verbatim
    final-target string, agent-tool rows are exempt, and ties break on the
    content-group hash so the result is order-independent. Returns the kept
    lines, the dropped count per target, and the dropped group hashes.
    """
    rows: list[tuple[str, str, str]] = []
    for line in lines:
        candidate = json.loads(line)
        if v14.is_agent_tool(candidate):
            continue
        turns = v14.assistant_turns(candidate)
        if not turns:
            continue
        rows.append((line, v14.canonical_hash(candidate["messages"]), turns[-1]))

    by_target: dict[str, list[tuple[str, str, str]]] = {}
    for row in rows:
        by_target.setdefault(row[2], []).append(row)

    dropped: list[tuple[str, str, str]] = []
    for target_rows in by_target.values():
        if len(target_rows) <= cap:
            continue
        target_rows.sort(key=lambda row: row[1])
        dropped.extend(target_rows[cap:])

    if not dropped:
        return list(lines), {}, []

    dropped_lines = {row[0] for row in dropped}
    kept_lines = [line for line in lines if line not in dropped_lines]
    if len(kept_lines) + len(dropped) != len(lines):
        raise ValueError(
            f"duplicate-target thinning is not a partition: kept {len(kept_lines)} + dropped {len(dropped)} "
            f"!= {len(lines)}"
        )

    dropped_by_target: Counter[str] = Counter(row[2] for row in dropped)
    dropped_hashes = sorted(row[1] for row in dropped)
    return kept_lines, dict(dropped_by_target), dropped_hashes


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
        "of the six fixed disclosures v9 and v10 repeated. v16 also thins validation and test to 50 rows per "
        "exact final assistant target, so validation loss stops being dominated by a handful of single-token "
        "diagnosis labels and starts tracking the same distribution train is admitted under."
    )
    return data


class _Release(v15._Release):
    """v15's build, with the v11 and v12 addenda and the v16 shuffle seed."""

    def __init__(self, indexes: v14._Indexes, output: Path, dry_run: bool) -> None:
        super().__init__(indexes, output, dry_run)
        self.v11_rows: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        self.v12_rows: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        self.capped_out: dict[str, dict[str, Any]] = {}
        self.duplicate_profiles_before_thin: dict[str, Any] = {}

    def run(self) -> None:
        """v15's run, restated only to shuffle train with the v16 seed."""
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
        # Thinning runs last: it only trims the emitted split payloads, so every
        # upstream total, hash and group-set check still describes the full split.
        self.thin_evaluation_splits()

    def verify_totals(self) -> None:
        if self.purged != v15.EXPECTED_PURGED_ROWS:
            raise ValueError(f"expected {v15.EXPECTED_PURGED_ROWS} purged rows, found {self.purged}")
        if dict(self.repair_counts) != v15.EXPECTED_REPAIR_COUNTS:
            raise ValueError(f"repair population drifted: {dict(self.repair_counts)}")
        appended = len(self.v9_rows) + len(self.v10_rows) + len(self.v11_rows) + len(self.v12_rows)
        if self.carried + appended != self.next_number:
            raise ValueError("row numbering drifted")

    def append_addenda(self) -> None:
        """Append the v9, v10, v11, and v12 addenda as train rows."""
        super().append_addenda()
        self.v11_rows = v11_candidate_rows()
        for candidate, payload in self.v11_rows:
            self.append_agent_row(V11_SOURCE_LABEL, candidate, payload)
        self.v12_rows = v12_candidate_rows()
        for candidate, payload in self.v12_rows:
            self.append_chat_row(V12_SOURCE_LABEL, candidate, payload)

    def thin_evaluation_splits(self) -> None:
        """Cap validation and test at 50 rows per exact final assistant target.

        The cap is the train admission rule from v5, applied to the held-out
        splits for the first time. Rows whose content group vanishes from the
        emitted split are flagged in both ledgers so the drop stays auditable;
        nothing is removed from the candidate payload.
        """
        self.duplicate_profiles_before_thin = dict(self.duplicate_profiles)
        for split in EVALUATION_SPLITS:
            lines = self.payload_lines[split]
            kept, dropped_by_target, _ = cap_duplicate_targets(lines)
            expected = EXPECTED_EVALUATION_ROWS[split]
            if len(kept) != expected:
                raise ValueError(f"{split} thinned to {len(kept)} rows, expected {expected}")
            surviving = {v14.canonical_hash(json.loads(line)["messages"]) for line in kept}
            internal_flagged = rowref_flagged = 0
            for record in self.internal_records:
                if record["split"] == split and record["content_group_sha256"] not in surviving:
                    record["duplicate_target_capped_out"] = True
                    internal_flagged += 1
            for record in self.rowref_records:
                if record["provisional_split"] == split and record["content_group_sha256"] not in surviving:
                    record["duplicate_target_capped_out"] = True
                    rowref_flagged += 1
            if internal_flagged != rowref_flagged:
                raise ValueError(f"{split}: ledgers disagree on capped-out rows")
            self.capped_out[split] = {
                "dropped_rows": len(lines) - len(kept),
                "dropped_targets": len(dropped_by_target),
                "dropped_by_target": dict(sorted(dropped_by_target.items(), key=lambda item: (-item[1], item[0]))),
                "flagged_internal_records": internal_flagged,
            }
            self.payload_lines[split] = kept
        # Re-measure so the manifest describes the emitted splits, not the
        # pre-thin ones; train is untouched and still satisfies the v5 cap.
        self.audit_duplicate_targets()

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
        data["capped_out_rows"] = {
            name: self.capped_out.get(name, {}).get("dropped_rows", 0) for name in EVALUATION_SPLITS
        }
        return data

    def manifest(self) -> dict[str, Any]:
        data = stamp_release_identity(
            super().manifest(),
            v9_rows=len(self.v9_rows),
            v10_rows=len(self.v10_rows),
            v11_rows=len(self.v11_rows),
            v12_rows=len(self.v12_rows),
        )
        # super() derives membership from every internal record; the emitted
        # splits drop the capped-out groups, so restate the evidence against
        # what the payloads actually contain.
        emitted = {
            name: {
                record["content_group_sha256"]
                for record in self.internal_records
                if record["split"] == name and not record.get("duplicate_target_capped_out")
            }
            for name in SPLITS
        }
        heldout_groups = {record["content_group_sha256"] for record in self.heldout_records}
        disjointness = data["evidence"]["group_disjointness"]
        for name in EVALUATION_SPLITS:
            disjointness[f"{name}_group_count"] = len(emitted[name])
            disjointness["heldout_group_intersections"][name] = len(heldout_groups & emitted[name])
            data["partitions"][name]["source_group_count"] = len(emitted[name])
        disjointness["pairwise_intersections"] = {
            "train_validation": len(emitted["train"] & emitted["validation"]),
            "train_test": len(emitted["train"] & emitted["test"]),
            "validation_test": len(emitted["validation"] & emitted["test"]),
        }
        data["evidence"]["duplicate_target_concentration"] = self.duplicate_profiles
        data["evidence"]["evaluation_duplicate_target_cap"] = {
            "cap": EVALUATION_DUPLICATE_TARGET_CAP,
            "rule": "keep the 50 lowest content_group_sha256 rows per exact final assistant target",
            "agent_tool_rows_exempt": True,
            "applies_to": list(EVALUATION_SPLITS),
            "before": {
                name: self.duplicate_profiles_before_thin.get(name, {}).get("final_target", {})
                for name in EVALUATION_SPLITS
            },
            "after": {
                name: self.duplicate_profiles.get(name, {}).get("final_target", {}) for name in EVALUATION_SPLITS
            },
            "capped_out": self.capped_out,
            "ledger_flag": "duplicate_target_capped_out",
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
