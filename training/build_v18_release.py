"""Build the v18 G04 research release by rewriting the corpus's own risk turns.

The v17 paired eval showed that no addendum was going to fix crisis routing. The
base model routes 11/11 crisis cases; every adapter built so far routes fewer,
and v17's 708-row direct-register family made the gap worse rather than better.
The diagnosis was in the corpus itself: 6,518 therapist turns carry a live risk
in their ledger, and only 178 of them name a route. The corpus teaches "risk on
the table -> reflect on it", and 82k rows of that outvote any synthetic family
of a few hundred.

v18 therefore repairs the corpus instead of talking over it. Every arc session's
first risk engagement, each escalation, and a closing turn with live risk is
rewritten in place: the ledger stays byte-identical, the spoken half keeps its
original sentences and questions, and a routing action is prepended in the
session's own terms. A row carrying a rewritten turn is a repaired row, so it
is re-bucketed by its new content group the same way the welded-argument and
reasoning-preamble repairs are, and the held-out duplicate-target purge is
re-planned against the rewritten payloads so the emitted splits and the internal
ledger still describe the same surviving rows.

Everything else is inherited from v17: the candidate-level repairs, the v9, v10,
v11 and v13 addenda, the crisis purge, the held-out cap, the split recipe.

The rewrites come from ``training/rewrite_arc_risk_turns.py`` as a resumable
JSONL keyed by candidate row hash and message index. They are validated against
the candidate row they claim to rewrite — role, ledger, risk level and the
original spoken text must all match — so a rewrite file built for a different
release fails the build instead of silently landing on the wrong turn.

Usage::

    PYTHONPATH=/home/vivi/pixelated python -m training.build_v18_release --dry-run
    PYTHONPATH=/home/vivi/pixelated python -m training.build_v18_release
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from training import (
    add_risk_reply_headers as hdr,
    build_v14_release as v14,
    build_v15_release as v15,
    build_v17_release as v17,
    rewrite_arc_risk_turns as rw,
)

ROOT = v17.ROOT

OUTPUT = ROOT / "ai/data/curated/recovery/gold-20261017/wandb-full-corpus-current-master-20261109-v18"
MANIFEST = ROOT / "docs/training/execution-2026-11-09/wandb-full-corpus-current-master-20261109-v18.json"

RELEASE_ID = "wandb-full-corpus-current-master-20261109-v18"
SHUFFLE_SEED = 20261110
SPLITS = v17.SPLITS
CREATED_AT_UTC = "2026-11-09T00:00:00Z"

# The rewrites were generated against the v17 candidate, which is the v13
# candidate minus the crisis and held-out purges plus the addenda. Retained rows
# carry their source bytes unchanged, so the row hash in the rewrite file is the
# same hash the v13 candidate line has here.
REWRITE_PATH = ROOT / "ai/training/output/arc_risk_rewrites.jsonl"
HEADER_PATH = ROOT / "ai/training/output/arc_risk_headers.jsonl"
REWRITE_SOURCE_CANDIDATE = (
    ROOT / "ai/data/curated/recovery/gold-20261017/wandb-full-corpus-current-master-20261109-v17/candidate.jsonl"
)

EVALUATION_SPLITS = v17.v16.EVALUATION_SPLITS
EVALUATION_DUPLICATE_TARGET_CAP = v17.v16.EVALUATION_DUPLICATE_TARGET_CAP
EVALUATION_PURGE_REASON = v17.v16.EVALUATION_PURGE_REASON
# Re-planned against the rewritten payloads, so these are v18's own numbers, not
# v17's: a rewritten row changes content group, and the cap keeps the 50 lowest
# hashes per final target, so rows move in and out of the split as the payloads
# change.
EXPECTED_EVALUATION_PURGED: dict[str, int] = {"validation": 4_613, "test": 4_401}

LIMITATIONS_V18: tuple[str, ...] = (
    (
        "The rewritten risk turns are produced by a language model, not by a clinician, and only a mechanical "
        "contract is enforced: the ledger is untouched, the original sentences and questions are kept, a "
        "routing term leads, and the reply is capped at 75 words. Register and clinical appropriateness are not "
        "verified, so a rewrite can still read badly or route to the wrong level of care without failing the "
        "build."
    ),
    (
        "The rewrite covers 1,125 selected turns across 541 arc sessions — the first risk engagement, each "
        "escalation, and a closing turn with live risk. A turn the writer rejected or that failed the contract is "
        "routed by the deterministic header instead, which keeps the original answer verbatim under a fixed "
        "routing sentence; a therapist turn that carries risk without being one of those three positions keeps "
        "its original reflective answer, and non-arc rows with acute risk in the ledger are untouched, so the "
        "corpus still contains reflective answers to live risk."
    ),
    (
        "Rewritten rows are re-bucketed by their new content group, so the rewrite moves rows between train, "
        "validation and test rather than holding the v17 split fixed. Validation and test therefore contain "
        "rewritten rows, which is the intent — the held-out splits should measure the behaviour the corpus "
        "teaches — but their loss is no longer comparable to a previous release's."
    ),
    (
        "The routing headers are fixed sets of eight sentences chosen by row hash — hotline sentences for passive "
        "risk, acute-care sentences for active risk — not generated text: each is prepended to an existing reply "
        "that names no resource after an acute disclosure. They are mechanically checked (a routing term leads, "
        "the band matches the ledger, no question, no added softener) but not reviewed, and the same sixteen "
        "sentences recur, so the model may learn them as a template rather than as phrasing. Where the writer did "
        "rewrite a turn, its rewrite is used and the header is dropped."
    ),
    (
        "The headers only cover replies that follow an acute disclosure the disclosure lexicon recognises. A "
        "reply to risk phrased outside that lexicon is left exactly as it was."
    ),
)


def _limitations() -> tuple[str, ...]:
    """v17's limitations plus the ones the in-place rewrite introduces."""
    return (*v17._limitations(), *LIMITATIONS_V18)


def rewritten_content(original: str, new_spoken: str) -> str:
    """Replace the spoken half of a turn, keeping any ledger bytes verbatim."""
    _, _ledger, spoken = rw.split_content(original)
    body = rw.spoken_body(spoken)
    if not body or not original.endswith(body):
        raise ValueError("the spoken text is not the tail of the turn")
    return original[: len(original) - len(body)] + new_spoken


def apply_rewrites(row: dict[str, Any], rewrites: dict[int, str]) -> dict[str, Any]:
    """Return the row with each named assistant turn's spoken text replaced."""
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("row has no message list to rewrite")
    updated = list(messages)
    for msg_index, new_spoken in sorted(rewrites.items()):
        if not 0 <= msg_index < len(updated):
            raise ValueError(f"message index {msg_index} is out of range")
        message = dict(updated[msg_index])
        if message.get("role") != "assistant":
            raise ValueError(f"message {msg_index} is not an assistant turn")
        message["content"] = rewritten_content(message.get("content") or "", new_spoken)
        updated[msg_index] = message
    return {**row, "messages": updated}


def load_rewrite_records(path: Path = REWRITE_PATH) -> list[dict[str, Any]]:
    """The accepted rewrites, deduplicated by row hash and message index."""
    if not path.exists():
        raise FileNotFoundError(f"no rewrite file at {path}")
    records: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            record = json.loads(line)
            if not record.get("new_spoken") or record.get("error"):
                continue
            key = (record["row_sha256"], int(record["msg_index"]))
            previous = records.get(key)
            if previous is not None and previous["new_spoken"] != record["new_spoken"]:
                raise ValueError(f"conflicting rewrites for {key[0]}#{key[1]}")
            records[key] = record
    return list(records.values())


def _validate_turn(record: dict[str, Any], row: dict[str, Any], old_number: int) -> tuple[str, str | None, str]:
    """Check the record describes an assistant turn of the row; return its split content."""
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"rewrite source row {old_number} has no message list")
    msg_index = int(record["msg_index"])
    if not 0 <= msg_index < len(messages):
        raise ValueError(f"rewrite for row {old_number} indexes message {msg_index}, row has {len(messages)}")
    message = messages[msg_index]
    if message.get("role") != "assistant":
        raise ValueError(f"rewrite for row {old_number} targets a {message.get('role')} turn")
    prefix, ledger, spoken = rw.split_content(message.get("content") or "")
    if rw.spoken_body(spoken) != record["original_spoken"]:
        raise ValueError(f"rewrite for row {old_number} was written against different spoken text")
    return prefix, ledger, spoken


def _validate_rewrite(record: dict[str, Any], row: dict[str, Any], old_number: int) -> None:
    """Fail unless the rewrite describes the risk turn it claims to rewrite."""
    _prefix, ledger, _spoken = _validate_turn(record, row, old_number)
    if ledger is None:
        raise ValueError(f"rewrite for row {old_number} targets a turn without a ledger")
    if record.get("ledger_sha256") and record["ledger_sha256"] != _ledger_digest(ledger):
        raise ValueError(f"rewrite for row {old_number} was written against a different ledger")
    level = "active" if rw.risk_level(ledger.get("risk")) == rw.LEVEL_ACTIVE else "passive"
    if record.get("level") not in (None, level):
        raise ValueError(f"rewrite for row {old_number} claims risk level {record['level']} but the ledger is {level}")


def _validate_header(record: dict[str, Any], row: dict[str, Any], old_number: int) -> None:
    """Fail unless the header is a bare routing prefix on the reply it claims."""
    if record.get("kind") != hdr.KIND:
        raise ValueError(f"header for row {old_number} is not marked as a header")
    _prefix, ledger, _spoken = _validate_turn(record, row, old_number)
    if ledger is not None and record.get("ledger_sha256") != hdr.ledger_digest(ledger):
        raise ValueError(f"header for row {old_number} was written against a different ledger")
    original = record["original_spoken"]
    new_spoken = record["new_spoken"]
    if not new_spoken.endswith(original):
        raise ValueError(f"header for row {old_number} does not keep the original reply")
    prefix = new_spoken[: len(new_spoken) - len(original)].strip()
    if not prefix:
        raise ValueError(f"header for row {old_number} adds no prefix")
    hdr.validate_header(prefix, original, record.get("level"))


def _validate(record: dict[str, Any], row: dict[str, Any], old_number: int) -> None:
    if record.get("kind") == hdr.KIND:
        _validate_header(record, row, old_number)
    else:
        _validate_rewrite(record, row, old_number)


def _ledger_digest(ledger: dict[str, Any]) -> str:
    """The digest the rewrite file records: sha256 of the canonical ledger JSON."""
    canonical = json.dumps(ledger, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _drop_shadowed_headers(records: list[dict[str, Any]], headers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A model rewrite outranks the deterministic header for the same turn."""
    rewritten = {(record["row_sha256"], int(record["msg_index"])) for record in records}
    kept: list[dict[str, Any]] = []
    for record in headers:
        key = (record["row_sha256"], int(record["msg_index"]))
        if key in rewritten:
            continue
        kept.append(record)
    return kept


def build_overlay(
    candidate: Path = REWRITE_SOURCE_CANDIDATE,
    records: list[dict[str, Any]] | None = None,
    headers: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, dict[int, str]], dict[str, int]]:
    """Map candidate row hash to ``{message index: replacement spoken text}``.

    Every record has to match a row in the candidate it was generated from; a
    record that matches nothing means the overlay file and the release have
    drifted apart, and it fails the build rather than dropping a known-bad turn.
    """
    if records is None and headers is None:
        records = load_rewrite_records()
        headers = load_rewrite_records(HEADER_PATH) if HEADER_PATH.exists() else []
    model_records = list(records or [])
    headers = _drop_shadowed_headers(model_records, list(headers or []))
    all_records = [*model_records, *headers]
    by_sha: dict[str, list[dict[str, Any]]] = {}
    for record in all_records:
        by_sha.setdefault(record["row_sha256"], []).append(record)
    if not by_sha:
        raise ValueError("no usable rewrites to apply")

    overlay: dict[str, dict[int, str]] = {}
    matched: set[str] = set()
    header_turns = 0
    with candidate.open("rb") as source:
        for old_number, raw in enumerate(source, start=1):
            if not raw.strip():
                continue
            row_sha = v14.line_sha(raw)
            claimed = by_sha.get(row_sha)
            if claimed is None:
                continue
            row = json.loads(raw)
            for record in claimed:
                _validate(record, row, old_number)
                turns = overlay.setdefault(row_sha, {})
                index = int(record["msg_index"])
                if index in turns and turns[index] != record["new_spoken"]:
                    raise ValueError(f"two different replacements for {row_sha}#{index}")
                turns[index] = record["new_spoken"]
                header_turns += record.get("kind") == hdr.KIND
            matched.add(row_sha)

    missing = set(by_sha) - matched
    if missing:
        raise ValueError(f"{len(missing)} rewrites do not match any row in {candidate}")
    turns = sum(len(turns) for turns in overlay.values())
    return overlay, {
        "rows": len(overlay),
        "turns": turns,
        "model_rows": len({record["row_sha256"] for record in model_records}),
        "model_turns": len(model_records),
        "header_rows": len({record["row_sha256"] for record in headers or []}),
        "header_turns": header_turns,
    }


def carried_line(raw: bytes, row: dict[str, Any]) -> tuple[dict[str, Any], str | None, bytes]:
    """The row and bytes v15's admission writes for this row, repair included."""
    repaired, kind = v15.repair_row(row)
    return repaired, kind, raw if kind is None else v15.json_bytes(repaired)


def _projected_target(
    indexes: v14._Indexes,
    old_number: int,
    repaired_row: dict[str, Any],
    repaired: bool,
    rewrites: dict[int, str] | None,
) -> tuple[str, str, bool]:
    """Return the projected final assistant target, content-group hash, and repair flag."""
    if rewrites is not None:
        repaired_row = apply_rewrites(repaired_row, rewrites)
        repaired = True
    messages = repaired_row.get("messages")
    repair = indexes.repairs.get(old_number)
    if repair is not None:
        if rewrites is not None:
            raise ValueError(f"row {old_number} is rewritten and preamble-repaired at the same time")
        messages = repair["messages"]
    if not isinstance(messages, list):
        raise ValueError(f"retained row {old_number} has no message list to project")
    payload = v14.project_chat(messages, Counter())
    if payload is None:
        raise ValueError(f"retained row {old_number} projects to no assistant target")
    # project_chat truncates after the last assistant turn, so the final element
    # is the row's supervision target.
    return payload[-1]["content"], v14.canonical_hash(payload), repaired


def _evaluation_targets(
    indexes: v14._Indexes, overlay: dict[str, dict[int, str]]
) -> tuple[dict[str, dict[str, list[tuple[str, int]]]], dict[int, str]]:
    """Bucket every retained held-out row by its exact final assistant target.

    Crisis placeholders and agent-tool rows are excluded, matching the train
    rule and the v16 planner. The split is the one v15's admission rule will use
    after v18's rewrite, so a rewritten row is bucketed by its new content group.
    """
    by_target: dict[str, dict[str, list[tuple[str, int]]]] = {name: {} for name in EVALUATION_SPLITS}
    sha_by_number: dict[int, str] = {}
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
            repaired_row, kind, line = carried_line(raw, row)
            rewrites = overlay.get(v14.line_sha(line))
            target, group_hash, repaired = _projected_target(
                indexes, old_number, repaired_row, kind is not None, rewrites
            )
            split = v15.assign_split(group_hash) if repaired else reference["provisional_split"]
            if split not in EVALUATION_SPLITS:
                continue
            by_target[split].setdefault(target, []).append((group_hash, old_number))
            sha_by_number[old_number] = v14.line_sha(line)
    return by_target, sha_by_number


def plan_evaluation_purge(
    indexes: v14._Indexes,
    overlay: dict[str, dict[int, str]],
    cap: int = EVALUATION_DUPLICATE_TARGET_CAP,
) -> tuple[dict[int, str], dict[str, dict[str, Any]], set[str]]:
    """Decide which retained validation/test rows exceed the per-target cap.

    Returns the purge plan, the per-split stats, and the row hashes the plan
    drops — an overlaid row can be dropped here, and then its overlay never
    reaches the build.
    """
    by_target, sha_by_number = _evaluation_targets(indexes, overlay)
    purge_plan: dict[int, str] = {}
    stats: dict[str, dict[str, Any]] = {}
    for split in EVALUATION_SPLITS:
        split_plan, stats[split] = v17.v16._purge_split(split, by_target[split], cap)
        purge_plan.update(split_plan)
    purged_shas = {sha_by_number[number] for number in purge_plan if number in sha_by_number}
    return purge_plan, stats, purged_shas


def stamp_release_identity(
    data: dict[str, Any],
    *,
    addenda: dict[str, int],
    rewrites: dict[str, int],
) -> dict[str, Any]:
    """Rewrite the addenda- and rewrite-dependent manifest fields for v18."""
    data["release_id"] = RELEASE_ID
    data["created_at_utc"] = CREATED_AT_UTC
    data["scope"]["appended_rows"] = sum(addenda.values())
    data["scope"]["appended_rows_by_source"] = dict(addenda)
    data["scope"]["rewritten_rows"] = rewrites["rows"]
    data["scope"]["rewritten_turns"] = rewrites["turns"]
    data["scope"]["header_rows"] = rewrites["header_rows"]
    data["scope"]["header_turns"] = rewrites["header_turns"]
    data["scope"]["overlay_rows_dropped_with_purged_rows"] = rewrites["dropped_rows"]
    data["payload_contract"]["shuffle"] = {
        "train": True,
        "validation": False,
        "test": False,
        "seed": SHUFFLE_SEED,
    }
    data["payload_contract"]["agent_tool_note"] = (
        f"2,807 v13 agent_tool rows plus {addenda[v14.V9_SOURCE_LABEL]} v9 rows plus "
        f"{addenda[v15.V10_SOURCE_LABEL]} v10 rows plus {addenda[v17.v16.V11_SOURCE_LABEL]} v11 rows "
        "include assistant tool_calls, role:tool turns, and a retained system message that defines tool-use "
        "context."
    )
    data["limitations"] = list(_limitations())
    data["owner_direction"] = (
        "User-directed v18 after the v17 paired eval. v17 replaced the crisis family with a direct-register "
        "addendum and lost four crisis passes to v16, which confirmed the diagnosis rather than contradicting "
        "it: the base model routes 11/11 crisis cases and every adapter built on this corpus routes fewer, "
        "because 6,518 therapist turns carry a live risk in their ledger and only 178 name a route, against 636 "
        "addendum rows. v18 stops adding families and repairs the corpus instead. Every arc session's first "
        "risk engagement, each escalation, and a closing turn with live risk is rewritten in place: the ledger "
        "bytes are untouched, the original sentences and questions are kept, and a routing action is prepended "
        "in the session's own terms and at the session's own level of care. Rewritten rows are re-bucketed by "
        "their new content group, and the held-out duplicate-target cap is re-planned against the rewritten "
        "payloads so the emitted splits and the internal ledger still agree row for row. The rewrites alone do "
        "not cover the plain therapy rows, so every reply that answers an acute disclosure and names no resource "
        "also gets a deterministic routing sentence prepended, and the original reply kept verbatim after it."
    )
    return data


class _Release(v17._Release):
    """v17's build, with the arc risk turns rewritten in place."""

    def __init__(self, indexes: v14._Indexes, output: Path, dry_run: bool) -> None:
        super().__init__(indexes, output, dry_run)
        self.overlay: dict[str, dict[int, str]] = {}
        self.overlay_counts: dict[str, int] = {}
        self.purged_overlay_rows: dict[str, int] = {}
        self.header_keys: set[tuple[str, int]] = set()
        self.rewritten_rows = 0
        self.rewritten_turns = 0
        self.header_rows_applied = 0
        self.header_turns_applied = 0
        self.rewritten_by_split: Counter[str] = Counter()
        self.seen_rewrite_rows: set[str] = set()
        self.rewritten_hashes: set[str] = set()
        self._admitting_rewrite = False
        self._model_shas: set[str] = set()
        self._header_shas: set[str] = set()

    def run(self) -> None:
        """v17's run, with the purge planned against the rewritten payloads."""
        records = load_rewrite_records()
        headers = load_rewrite_records(HEADER_PATH) if HEADER_PATH.exists() else []
        self.header_keys = {(record["row_sha256"], int(record["msg_index"])) for record in headers}
        self.overlay, _counts = build_overlay(records=records, headers=headers)
        self._model_shas = {record["row_sha256"] for record in records}
        self._header_shas = {record["row_sha256"] for record in headers}
        super().run()

    def _plan_purge(self) -> tuple[dict[int, str], dict[str, dict[str, Any]]]:
        """v16's up-front purge, planned against the rows as the rewrites leave them.

        A rewrite changes the final target and the content group, so the row can
        move between splits; planning on the original bytes would purge the wrong
        rows and report a split size the build cannot emit.
        """
        purge_plan, stats, purged_shas = plan_evaluation_purge(self.indexes, self.overlay)
        # A held-out row the cap drops is never emitted, so its overlay cannot
        # reach the build either; dropping it here keeps the counters honest
        # instead of failing on rows that were never going to ship.
        dropped = {sha for sha in self.overlay if sha in purged_shas}
        for sha in dropped:
            del self.overlay[sha]
        self.purged_overlay_rows = {
            "rows": len(dropped),
            "model_rows": len(dropped & self._model_shas),
            "header_rows": len(dropped & self._header_shas),
        }
        self.overlay_counts = self._recount_overlay()
        return purge_plan, stats

    def _recount_overlay(self) -> dict[str, int]:
        """How many rows and turns the surviving overlay actually replaces."""
        turns = 0
        header_turns = 0
        model_rows = 0
        header_rows = 0
        for row_sha, rewrites in self.overlay.items():
            turns += len(rewrites)
            headers = sum(1 for index in rewrites if (row_sha, index) in self.header_keys)
            header_turns += headers
            header_rows += 1 if headers else 0
            model_rows += 1 if headers < len(rewrites) else 0
        return {
            "rows": len(self.overlay),
            "turns": turns,
            "model_rows": model_rows,
            "model_turns": turns - header_turns,
            "header_rows": header_rows,
            "header_turns": header_turns,
        }

    def carry(self, old_number: int, raw: bytes) -> None:
        """Admit the rewritten bytes for a row whose risk turns were rewritten."""
        row = json.loads(raw)
        if v14.is_crisis_placeholder(row):
            super().carry(old_number, raw)
            return
        repaired_row, kind, line = carried_line(raw, row)
        rewrites = self.overlay.get(v14.line_sha(line))
        if rewrites is None:
            super().carry(old_number, raw)
            return
        self.seen_rewrite_rows.add(v14.line_sha(line))
        if self.purge_plan.get(old_number) is not None:
            raise ValueError(f"row {old_number} is rewritten and purged as a held-out duplicate")
        if self.indexes.heldout_source.get(old_number) is not None:
            raise ValueError(f"row {old_number} is rewritten but belongs to the held-out reserve")
        if old_number in self.indexes.repairs:
            # A projection-time repair replaces the message list wholesale, which
            # would throw the rewrite away without saying so.
            raise ValueError(f"row {old_number} is rewritten and also carries a projection-time repair")
        rewritten = apply_rewrites(repaired_row, rewrites)
        # Marking the row repaired skips the provenance hash check and re-buckets
        # it by its new content group, exactly as the candidate-level repairs do.
        self.repaired_numbers.add(old_number)
        self.rewritten_rows += 1
        self.rewritten_turns += len(rewrites)
        row_sha = v14.line_sha(line)
        headers = sum(1 for index in rewrites if (row_sha, index) in self.header_keys)
        self.header_rows_applied += 1 if headers else 0
        self.header_turns_applied += headers
        self._admitting_rewrite = True
        try:
            super().carry(old_number, v15.json_bytes(rewritten))
        finally:
            self._admitting_rewrite = False
        # The candidate-level repair was applied here rather than inside v15's
        # carry, so count it by hand. The pinned population covers every repaired
        # row, rewritten or not, so the total must not move.
        if kind is not None:
            self.repair_counts[kind] += 1

    def admit(self, line: bytes) -> str:
        """Remember the hash of the bytes a rewritten row was finally admitted as."""
        row_sha = super().admit(line)
        if self._admitting_rewrite:
            self.rewritten_hashes.add(row_sha)
        return row_sha

    def verify_totals(self) -> None:
        """v17's totals, with the re-planned held-out purge and the rewrite coverage."""
        if self.purged != v15.EXPECTED_PURGED_ROWS:
            raise ValueError(f"expected {v15.EXPECTED_PURGED_ROWS} purged rows, found {self.purged}")
        if dict(self.repair_counts) != v15.EXPECTED_REPAIR_COUNTS:
            raise ValueError(f"repair population drifted: {dict(self.repair_counts)}")
        appended = len(self.v9_rows) + len(self.v10_rows) + len(self.v11_rows) + len(self.v13_rows)
        if self.carried + appended != self.next_number:
            raise ValueError("row numbering drifted")
        missing = set(self.overlay) - self.seen_rewrite_rows
        if missing:
            raise ValueError(f"{len(missing)} rewritten rows never reached the build")
        if self.rewritten_rows != len(self.overlay):
            raise ValueError(f"applied {self.rewritten_rows} rewrites to {len(self.overlay)} rewritten rows")
        if self.rewritten_turns != self.overlay_counts["turns"]:
            raise ValueError(
                f"applied {self.rewritten_turns} rewritten turns, overlay has {self.overlay_counts['turns']}"
            )
        if self.header_turns_applied != self.overlay_counts["header_turns"]:
            raise ValueError(
                f"applied {self.header_turns_applied} headers, overlay counts {self.overlay_counts['header_turns']}"
            )
        purged = {name: self.evaluation_purged[name] for name in EVALUATION_SPLITS}
        if purged != EXPECTED_EVALUATION_PURGED:
            raise ValueError(f"held-out purge drifted: {purged}, expected {EXPECTED_EVALUATION_PURGED}")
        for split in EVALUATION_SPLITS:
            emitted = len(self.payload_lines[split])
            planned = self.evaluation_purge_stats[split]["rows_after"]
            if emitted != planned:
                raise ValueError(f"{split}: emitted {emitted} rows but the purge plan expected {planned}")

    def audit_duplicate_targets(self) -> None:
        """Measure the concentration and record which splits took the rewrites."""
        super().audit_duplicate_targets()
        self.rewritten_by_split = Counter(
            record["split"]
            for record in self.internal_records
            if record["candidate_row_sha256"] in self.rewritten_hashes
        )

    def summary(self) -> dict[str, Any]:
        data = super().summary()
        data["release_id"] = RELEASE_ID
        data["rewritten_rows"] = self.rewritten_rows
        data["rewritten_turns"] = self.rewritten_turns
        data["header_rows"] = self.header_rows_applied
        data["header_turns"] = self.header_turns_applied
        data["purged_overlay_rows"] = dict(self.purged_overlay_rows)
        data["rewritten_by_split"] = dict(self.rewritten_by_split)
        return data

    def manifest(self) -> dict[str, Any]:
        return stamp_release_identity(
            super().manifest(),
            addenda={
                v14.V9_SOURCE_LABEL: len(self.v9_rows),
                v15.V10_SOURCE_LABEL: len(self.v10_rows),
                v17.v16.V11_SOURCE_LABEL: len(self.v11_rows),
                v17.V13_SOURCE_LABEL: len(self.v13_rows),
            },
            rewrites={
                "rows": self.overlay_counts["model_rows"],
                "turns": self.overlay_counts["model_turns"],
                "header_rows": self.header_rows_applied,
                "header_turns": self.header_turns_applied,
                "dropped_rows": self.purged_overlay_rows.get("rows", 0),
            },
        )


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
