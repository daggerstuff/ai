"""Build the v21 G04 research release: wider crisis family, corpus register repaired.

v20 closed the ratio hole. 1,760 crisis rows against an 82.6k-row corpus took the
paired crisis suite to 11/11 on both splits and beat v19 on the aggregate, which
confirmed that the register the family teaches is right and that the corpus only
wins when it outnumbers it. The cases v20 still lost were all openers:

    "I'm not going to pretend that's not serious. Let's get you to an emergency
     department tonight..."
    "I'm taking this seriously... Let's slow down. Are you alone right now?"

Both sentences are on the crisis rubric's omit list, and unlike the 50-phrase
coddling sweep v20 rejected, these two are worth removing: they appear verbatim
in an observed v20 failure, so they are not hypothetical.

Auditing where they come from was the surprise. Of 362 "I'm not going to pretend"
occurrences in train, 331 sit in our own generated ``arc_corpus_*`` rows — 367 of
981 arc rows carry it, a 37% density inside a slice we wrote ourselves. The 192
"let's slow down" occurrences are filler sentences in published clinical books.
Neither is legitimate empathy in context, and together they touch 862 of 83,631
train rows (~1.0%), so this is a two-phrase deletion, not a rewrite.

v21 therefore makes two changes:

* register — whole sentences containing either phrase are dropped from train chat
  assistant turns, after the content-group hash gate so provenance still
  describes the carried text. Sentence-level removal avoids the fragments and
  dangling conjunctions a clause-level strip produced.
* surface and volume — the crisis family grows from 1,760 to 2,352 rows (2.8% of
  train) with 576 distinct disclosures, mass still concentrated on the two
  categories that failed in v19.

Everything else is inherited from v20/v17: the candidate-level welded-argument
and reasoning-preamble repairs, the v9, v10 and v11 addenda, the fabricated crisis
purge, the held-out duplicate-target purge, and the split recipe. Validation and
test are byte-identical to v20, so a v21-vs-v20 comparison isolates the addendum
swap and the register repair.

Usage::

    PYTHONPATH=/home/vivi/pixelated python -m training.build_v21_release --dry-run
    PYTHONPATH=/home/vivi/pixelated python -m training.build_v21_release
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from training import (
    build_agent_tool_sft_v21 as v21,
    build_v14_release as v14,
    build_v15_release as v15,
    build_v16_release as v16,
    build_v20_release as v20,
)

ROOT = v20.ROOT

OUTPUT = ROOT / "ai/data/curated/recovery/gold-20261017/wandb-full-corpus-current-master-20261110-v21"
MANIFEST = ROOT / "docs/training/execution-2026-11-10/wandb-full-corpus-current-master-20261110-v21.json"

RELEASE_ID = "wandb-full-corpus-current-master-20261110-v21"
SHUFFLE_SEED = 20261113
SPLITS = v20.SPLITS
CREATED_AT_UTC = "2026-11-10T00:00:00Z"

V21_SOURCE_LABEL = "crisis_direct_v21"
EXPECTED_V21_TRAIN_ROWS = 2352

# Sentences carrying either phrase are dropped from train chat assistant turns.
# "not going to pretend" also catches "I am not going to pretend" and the curly
# apostrophe spelling; "let's slow down" is matched with both apostrophes.
STRIP_PHRASES: tuple[str, ...] = ("not going to pretend", "let's slow down")

# A sentence ends at terminal punctuation followed by whitespace. Arc rows prefix
# a JSON ledger, which is split off before this runs so the ledger is never cut.
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

# v20's limitation list describes the v20 family — its size, its register and its
# train-only placement. v21 replaces that family, so those sentences go.
_STALE_V20_LIMITATIONS = (
    "The v20 crisis-direct rows replace",
    "The v20 replies are hand-written",
    "The v20 family is appended to train only",
)

LIMITATIONS_V21: tuple[str, ...] = (
    (
        "The v21 crisis-direct rows replace the v20 rows rather than adding to them, so the release still "
        "teaches one register per category. The family is 2,352 rows, 2.8% of train, against roughly 80,300 "
        "corpus rows that answer a disclosure by reflecting on it; a disclosure phrased outside the 576 "
        "paraphrases of its category can still fall back to that reflective answer."
    ),
    (
        "The v21 replies are hand-written in a single imperative register and are not reviewed by a "
        "clinician. Every reply is required to name the concrete thing its disclosure named, and no reply "
        "contains a question, but they are still first responses only: there is no safety-plan, follow-up or "
        "de-escalation content, so the family teaches triage and leaves everything downstream of triage to "
        "whatever the corpus already contained."
    ),
    (
        "The register repair deletes whole sentences from train chat rows. It removes two phrases that appear "
        "verbatim in an observed v20 failure, not the wider coddling vocabulary the corpus uses as ordinary "
        "empathy, and it does not rewrite what remains: a turn whose only content was the deleted sentence "
        "would be left untouched rather than emptied, and no turn is regenerated. Validation and test are "
        "untouched, so the repair is only visible through the paired crisis cases."
    ),
)


def _limitations() -> tuple[str, ...]:
    """The inherited limitations minus the v20 family ones, plus the v21 ones."""
    inherited = v20._limitations()
    kept = tuple(text for text in inherited if not text.startswith(_STALE_V20_LIMITATIONS))
    if len(kept) != len(inherited) - len(_STALE_V20_LIMITATIONS):
        raise ValueError("a stale v20 limitation is no longer in the inherited list")
    return (*kept, *LIMITATIONS_V21)


def _normalise(text: str) -> str:
    """Lower-case and fold the curly apostrophe so both spellings match."""
    return text.lower().replace("\u2019", "'")


def split_ledger(content: str) -> tuple[str, str]:
    """Split an arc row's leading JSON ledger from its spoken tail.

    Arc gold rows carry the session ledger as a JSON object prefix followed by the
    spoken reply. Sentence splitting must never reach inside that object, so the
    balanced braces are scanned here and everything after them is the tail.
    """
    if not content.startswith("{"):
        return "", content
    depth = 0
    in_string = False
    escaped = False
    for index, character in enumerate(content):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return content[: index + 1], content[index + 1 :]
    return "", content


def strip_register_sentences(content: str) -> tuple[str, int, list[str]]:
    """Drop whole sentences that carry a banned phrase.

    Returns the rewritten content, how many sentences were removed, and which
    phrases were matched. A turn whose every sentence is banned is returned
    unchanged: emptying a target would teach the model to emit nothing.
    """
    head, tail = split_ledger(content)
    sentences = _SENTENCE.split(tail)
    kept: list[str] = []
    matched: list[str] = []
    for sentence in sentences:
        normalised = _normalise(sentence)
        hits = [phrase for phrase in STRIP_PHRASES if phrase in normalised]
        if hits:
            matched.extend(hits)
            continue
        kept.append(sentence)
    dropped = len(sentences) - len(kept)
    if not dropped or not kept:
        return content, 0, []
    lead = tail[: len(tail) - len(tail.lstrip())]
    rebuilt = lead + " ".join(part.strip() for part in kept)
    return head + rebuilt, dropped, matched


def register_repair_record(
    *,
    rows_changed: int,
    turns_changed: int,
    sentences_dropped: int,
    phrases_dropped: Mapping[str, int],
) -> dict[str, Any]:
    """The manifest record describing the register repair, as its own function.

    Kept separate from ``_Release.manifest`` so the record can be asserted without
    the release-wide manifest, which re-reads the written splits from disk.
    """
    return {
        "rule": "drop whole sentences containing a banned phrase from train chat assistant turns",
        "phrases": list(STRIP_PHRASES),
        "splits": ["train"],
        "applied_after": (
            "content_group_sha256 verification, so the recorded provenance describes the text carried "
            "from the candidate and the strip is a documented repair rather than a silent mutation"
        ),
        "rows_changed": rows_changed,
        "turns_changed": turns_changed,
        "sentences_dropped": sentences_dropped,
        "phrases_dropped": dict(phrases_dropped),
    }


def v21_candidate_rows() -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """The v21 train rows as (candidate row, projected payload messages).

    These are chat rows, not agent rows: v21 carries an empty tool catalog, so
    ``is_agent_tool`` stays false and the payload keeps the plain ``messages``
    body.
    """
    rows = v21.collect_rows()["train"]
    if len(rows) != EXPECTED_V21_TRAIN_ROWS:
        raise ValueError(f"expected {EXPECTED_V21_TRAIN_ROWS} v21 train rows, found {len(rows)}")
    out: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for item in rows:
        candidate = {"messages": item["messages"], "source": V21_SOURCE_LABEL}
        if v14.is_agent_tool(candidate):
            raise ValueError(f"{item['id']}: v21 candidate rows must not carry a tool catalog")
        # The rows are hand-written, so nothing should redact. Projecting against a
        # throwaway counter keeps the release-wide redaction counts honest: if a
        # v21 row ever trips a pattern, this fails instead of silently hiding it.
        scratch: Counter[str] = Counter()
        payload = v14.project_chat(item["messages"], scratch)
        if payload is None:
            raise ValueError(f"{item['id']}: v21 row projects to no assistant target")
        if scratch and any(scratch.values()):
            raise ValueError(f"{item['id']}: v21 row redacted {dict(scratch)}; synthetic rows must be clean")
        out.append((candidate, payload))
    return out


def stamp_release_identity(
    data: dict[str, Any], *, v9_rows: int, v10_rows: int, v11_rows: int, v21_rows: int
) -> dict[str, Any]:
    """Rewrite the identity and addenda fields for the v21 release."""
    appended_by_source = {
        v14.V9_SOURCE_LABEL: v9_rows,
        v15.V10_SOURCE_LABEL: v10_rows,
        v16.V11_SOURCE_LABEL: v11_rows,
        V21_SOURCE_LABEL: v21_rows,
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
        f"2,807 inherited agent_tool rows plus {v9_rows} v9 rows plus {v10_rows} v10 rows plus "
        f"{v11_rows} v11 rows include assistant tool_calls, role:tool turns, and a retained system message "
        "that defines tool-use context."
    )
    data["limitations"] = list(_limitations())
    data["owner_direction"] = (
        "User-directed v21 after the v20 paired eval. v20's ratio fix held — the crisis suite went to 11/11 "
        "on both splits and v20 beat v19 on the aggregate — and the cases it still lost all opened with a "
        "validation sentence, 'I'm not going to pretend that's not serious. Let's get you to...' or 'Let's "
        "slow down. Are you alone right now?'. Both phrases are on the rubric's omit list and both appear "
        "verbatim in an observed failure, so v21 deletes them rather than arguing about the rubric: whole "
        "sentences containing 'not going to pretend' or 'let's slow down' are dropped from train chat "
        "assistant turns. Attribution showed 331 of 362 'not going to pretend' occurrences sit in our own "
        "arc_corpus rows at a 37% density, so the model was learning the phrase from generated therapist "
        "turns, not from the clinical books. The crisis family grows to 2,352 rows (2.8% of train) with 576 "
        "distinct disclosures to keep the ratio moving the same direction."
    )
    return data


class _Release(v20._Release):
    """v20's pipeline with the v21 crisis family and the train register repair."""

    def __init__(self, indexes: v14._Indexes, output: Path, dry_run: bool) -> None:
        super().__init__(indexes, output, dry_run)
        self.v21_rows: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        self.register_stripped_rows = 0
        self.register_stripped_turns = 0
        self.register_stripped_sentences = 0
        self.register_stripped_phrases: Counter[str] = Counter()

    def append_addenda(self) -> None:
        """v15's v9 and v10 addenda, then v11, then v21 — v21 replaces v20."""
        v15._Release.append_addenda(self)
        self.v11_rows = v16.v11_candidate_rows()
        for candidate, payload in self.v11_rows:
            self.append_agent_row(v16.V11_SOURCE_LABEL, candidate, payload)
        self.v21_rows = v21_candidate_rows()
        for candidate, payload in self.v21_rows:
            self.append_chat_row(V21_SOURCE_LABEL, candidate, payload)

    def record_retained(
        self,
        old_number: int,
        row: dict[str, Any],
        row_sha: str,
        reference: dict[str, Any],
    ) -> None:
        """Carry one corpus row, then repair its register if it landed in train.

        The repair runs after the parent has verified the content-group hash, so
        the recorded provenance describes the text as carried from the candidate
        and the strip stays a documented release repair rather than a silent
        mutation of the source. The parent records the row's split in
        ``internal_records``, which is the only place the post-repair split (a
        repaired row is re-bucketed) is available to a subclass.
        """
        super().record_retained(old_number, row, row_sha, reference)
        if self.internal_records[-1]["split"] != "train":
            return
        if v14.is_agent_tool(row):
            return
        self.strip_train_register()

    def strip_train_register(self) -> None:
        """Drop banned sentences from the train row that was just appended."""
        lines = self.payload_lines["train"]
        body = json.loads(lines[-1])
        sentences = 0
        turns = 0
        phrases: Counter[str] = Counter()
        for message in body["messages"]:
            if message["role"] != "assistant":
                continue
            text, dropped, matched = strip_register_sentences(message["content"])
            if not dropped:
                continue
            message["content"] = text
            sentences += dropped
            turns += 1
            phrases.update(matched)
        if not sentences:
            return
        lines[-1] = v14.payload_line(body["messages"], None)
        self.register_stripped_rows += 1
        self.register_stripped_turns += turns
        self.register_stripped_sentences += sentences
        self.register_stripped_phrases.update(phrases)

    def verify_totals(self) -> None:
        """v16's totals, with the v21 family counted in place of v12 and v13."""
        if self.purged != v15.EXPECTED_PURGED_ROWS:
            raise ValueError(f"expected {v15.EXPECTED_PURGED_ROWS} purged rows, found {self.purged}")
        if dict(self.repair_counts) != v15.EXPECTED_REPAIR_COUNTS:
            raise ValueError(f"repair population drifted: {dict(self.repair_counts)}")
        appended = len(self.v9_rows) + len(self.v10_rows) + len(self.v11_rows) + len(self.v21_rows)
        if self.carried + appended != self.next_number:
            raise ValueError("row numbering drifted")
        purged = {name: self.evaluation_purged[name] for name in v16.EVALUATION_SPLITS}
        if purged != v16.EXPECTED_EVALUATION_PURGED:
            raise ValueError(f"held-out purge drifted: {purged}, expected {v16.EXPECTED_EVALUATION_PURGED}")
        for split in v16.EVALUATION_SPLITS:
            emitted = len(self.payload_lines[split])
            planned = self.evaluation_purge_stats[split]["rows_after"]
            if emitted != planned:
                raise ValueError(f"{split}: emitted {emitted} rows but the purge plan expected {planned}")

    def summary(self) -> dict[str, Any]:
        data = super().summary()
        data["release_id"] = RELEASE_ID
        data.pop("appended_v20_rows", None)
        data["appended_v21_rows"] = len(self.v21_rows)
        data["register_stripped_rows"] = self.register_stripped_rows
        data["register_stripped_turns"] = self.register_stripped_turns
        data["register_stripped_sentences"] = self.register_stripped_sentences
        data["register_stripped_phrases"] = dict(self.register_stripped_phrases)
        return data

    def manifest(self) -> dict[str, Any]:
        data = stamp_release_identity(
            super().manifest(),
            v9_rows=len(self.v9_rows),
            v10_rows=len(self.v10_rows),
            v11_rows=len(self.v11_rows),
            v21_rows=len(self.v21_rows),
        )
        data["scope"]["register_repairs"] = register_repair_record(
            rows_changed=self.register_stripped_rows,
            turns_changed=self.register_stripped_turns,
            sentences_dropped=self.register_stripped_sentences,
            phrases_dropped=self.register_stripped_phrases,
        )
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
