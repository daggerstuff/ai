"""Build the v22 G04 research release: volume instead of phrase surgery.

v20 was the best adapter on the board — 52/52 against base, 52/52 against v19,
11/11 on both crisis splits — and it got there by leaving the corpus alone and
making the crisis family big enough to be heard. v21 tested the opposite theory:
it kept the corpus and also deleted two phrases from it. It lost. The crisis
aggregate fell from v20's 84/88 to 73/88, and the two failures were the exact
shapes v21 tried to remove:

* "You've been holding onto a plan to end your life... I'm not going to jump to
  a bunch of questions. I want to stay with you right now. What's it like to say
  that out loud?"
* "That's a lot to carry alone. I'm not going to lecture you about stopping.
  Right now, let's get you somewhere safe and get that wound looked at."

v21 deleted "not going to pretend" 394 times and the model substituted "not
going to jump", which the corpus already teaches 38 times. It deleted "let's
slow down" 192 times and "that's a lot to carry" survived 98 times untouched and
was emitted verbatim. Removing 592 sentences from 448,194 assistant turns moved
the register by 0.13%: a phrase is a surface form, and the habit it belongs to
lives in the mass of the corpus, not in the phrase.

v22 therefore strips nothing and attacks the ratio directly. The family grows on
three axes at once:

* frames — eight deadline frames per disclosure instead of four, so the same
  action is taught as an opening, a follow-through and four ways of saying
  "now" rather than as one fixed sentence,
* actions — eight hand-written actions per category instead of three to five, so
  a category can no longer collapse onto a single sentence the way v13's
  medical_collapse and self_harm_active did,
* disclosures — 722 distinct disclosures instead of 552, weighted toward the
  three categories that actually failed,
* controls — 600 ordinary-problem rows instead of 144, so the family does not
  teach escalation by mass.

That is 5,584 crisis rows plus 600 controls, 6,184 rows against roughly 77,700
corpus rows: 7.0% of train, three times v20's 2.1%. The corpus is carried
exactly as v17, v19 and v20 carried it.

Everything else is inherited: the candidate-level welded-argument and
reasoning-preamble repairs, the v9, v10 and v11 addenda, the fabricated crisis
purge, the held-out duplicate-target purge, and the split recipe.

Usage::

    PYTHONPATH=/home/vivi/pixelated python -m training.build_v22_release --dry-run
    PYTHONPATH=/home/vivi/pixelated python -m training.build_v22_release
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from training import (
    build_agent_tool_sft_v22 as v22,
    build_v14_release as v14,
    build_v15_release as v15,
    build_v16_release as v16,
    build_v20_release as v20,
)

ROOT = v20.ROOT

OUTPUT = ROOT / "ai/data/curated/recovery/gold-20261017/wandb-full-corpus-current-master-20261110-v22"
MANIFEST = ROOT / "docs/training/execution-2026-11-10/wandb-full-corpus-current-master-20261110-v22.json"

RELEASE_ID = "wandb-full-corpus-current-master-20261110-v22"
SHUFFLE_SEED = 20261114
SPLITS = v20.SPLITS
CREATED_AT_UTC = "2026-11-11T00:00:00Z"

V22_SOURCE_LABEL = "crisis_direct_v22"
EXPECTED_V22_TRAIN_ROWS = 6184

# v20's limitation list describes the v20 family — its size, its register and its
# train-only placement. v22 replaces that family, so those sentences go.
_STALE_V20_LIMITATIONS = (
    "The v20 crisis-direct rows replace",
    "The v20 replies are hand-written",
    "The v20 family is appended to train only",
)

LIMITATIONS_V22: tuple[str, ...] = (
    (
        "The v22 crisis-direct rows replace the v20 rows rather than adding to them, so the release still "
        "teaches one register per category. The family is 6,184 rows, 7.0% of train, against roughly 77,700 "
        "corpus rows that answer a disclosure by reflecting on it. Volume is the only thing keeping the "
        "register in the majority: a disclosure phrased outside the 722 paraphrases of its category, or a "
        "case that needs more than one turn, can still fall back to the reflective answer."
    ),
    (
        "The 600 control rows hold the family at 9.3 escalations per ordinary reply, close to the 11 that v20 "
        "trained, but they are synthetic and short. If v22 over-escalates on a disclosure that is distressing "
        "without being acute, that is the ratio to raise next, not the phrase lists."
    ),
    (
        "The v22 replies are hand-written in a single imperative register and are not reviewed by a "
        "clinician. Every reply is required to name the concrete thing its disclosure named, and no reply "
        "contains a question, but they are still first responses only: there is no safety-plan, follow-up or "
        "de-escalation content, so the family teaches triage and leaves everything downstream of triage to "
        "whatever the corpus already contained. Eight actions per category is still a small surface, and the "
        "evaluation can reward a reply for naming an anchor that a real clinician would say differently."
    ),
    (
        "The v22 family is appended to train only, and validation and test carry no crisis-direct rows. An "
        "adapter that memorises the 722 disclosures instead of generalising the register would not be caught "
        "by the held-out splits; only the paired crisis cases probe that."
    ),
    (
        "v22 carries no corpus repair. v21's phrase strip was reverted because it lost to v20: the corpus "
        "still contains the validation register the rubric bans, including 'that's a lot to carry' and 'not "
        "going to <verb>' frames, and the release relies on the family's mass to outvote it rather than on "
        "deleting it. The application layer runs a generation-time register guard, but the release itself "
        "does not guarantee the model will not emit a banned opener."
    ),
)


def _limitations() -> tuple[str, ...]:
    """The inherited limitations minus the v20 family ones, plus the v22 ones."""
    inherited = v20._limitations()
    kept = tuple(text for text in inherited if not text.startswith(_STALE_V20_LIMITATIONS))
    if len(kept) != len(inherited) - len(_STALE_V20_LIMITATIONS):
        raise ValueError("a stale v20 limitation is no longer in the inherited list")
    return (*kept, *LIMITATIONS_V22)


def v22_candidate_rows() -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """The v22 train rows as (candidate row, projected payload messages).

    These are chat rows, not agent rows: v22 carries an empty tool catalog, so
    ``is_agent_tool`` stays false and the payload keeps the plain ``messages``
    body.
    """
    rows = v22.collect_rows()["train"]
    if len(rows) != EXPECTED_V22_TRAIN_ROWS:
        raise ValueError(f"expected {EXPECTED_V22_TRAIN_ROWS} v22 train rows, found {len(rows)}")
    out: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for item in rows:
        candidate = {"messages": item["messages"], "source": V22_SOURCE_LABEL}
        if v14.is_agent_tool(candidate):
            raise ValueError(f"{item['id']}: v22 candidate rows must not carry a tool catalog")
        # The rows are hand-written, so nothing should redact. Projecting against a
        # throwaway counter keeps the release-wide redaction counts honest: if a
        # v22 row ever trips a pattern, this fails instead of silently hiding it.
        scratch: Counter[str] = Counter()
        payload = v14.project_chat(item["messages"], scratch)
        if payload is None:
            raise ValueError(f"{item['id']}: v22 row projects to no assistant target")
        if scratch and any(scratch.values()):
            raise ValueError(f"{item['id']}: v22 row redacted {dict(scratch)}; synthetic rows must be clean")
        out.append((candidate, payload))
    return out


def stamp_release_identity(
    data: dict[str, Any], *, v9_rows: int, v10_rows: int, v11_rows: int, v22_rows: int
) -> dict[str, Any]:
    """Rewrite the identity and addenda fields for the v22 release."""
    appended_by_source = {
        v14.V9_SOURCE_LABEL: v9_rows,
        v15.V10_SOURCE_LABEL: v10_rows,
        v16.V11_SOURCE_LABEL: v11_rows,
        V22_SOURCE_LABEL: v22_rows,
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
        "User-directed v22 after the v21 paired eval. v20 is the release: 52/52 against base, 52/52 against "
        "v19, 11/11 on both crisis splits. v21 tried to fix the two remaining v20 losses by deleting the "
        "phrases that produced them and lost — the crisis aggregate fell from 84/88 to 73/88, and the "
        "failures came back as siblings of the deleted phrases ('not going to jump', 'that's a lot to "
        "carry'). Removing 592 sentences from 448,194 assistant turns moved the register 0.13%, which is the "
        "wrong lever: a phrase is a surface form, and the habit lives in the mass of the corpus. v22 reverts "
        "the strip and attacks the ratio instead — eight deadline frames per disclosure instead of four, "
        "eight hand-written actions per category instead of three to five, and 722 distinct disclosures "
        "instead of 552. That is 6,184 crisis rows, 7.0% of train against v20's 2.1%, with the corpus carried "
        "exactly as v17, v19 and v20 carried it."
    )
    return data


class _Release(v20._Release):
    """v20's pipeline with the larger, multi-action v22 crisis family in place of v20's."""

    def __init__(self, indexes: v14._Indexes, output: Path, dry_run: bool) -> None:
        super().__init__(indexes, output, dry_run)
        self.v22_rows: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []

    def append_addenda(self) -> None:
        """v15's v9 and v10 addenda, then v11, then v22 — v22 replaces v20."""
        v15._Release.append_addenda(self)
        self.v11_rows = v16.v11_candidate_rows()
        for candidate, payload in self.v11_rows:
            self.append_agent_row(v16.V11_SOURCE_LABEL, candidate, payload)
        self.v22_rows = v22_candidate_rows()
        for candidate, payload in self.v22_rows:
            self.append_chat_row(V22_SOURCE_LABEL, candidate, payload)

    def verify_totals(self) -> None:
        """v16's totals, with the v22 family counted in place of v13, v20 and v21."""
        if self.purged != v15.EXPECTED_PURGED_ROWS:
            raise ValueError(f"expected {v15.EXPECTED_PURGED_ROWS} purged rows, found {self.purged}")
        if dict(self.repair_counts) != v15.EXPECTED_REPAIR_COUNTS:
            raise ValueError(f"repair population drifted: {dict(self.repair_counts)}")
        appended = len(self.v9_rows) + len(self.v10_rows) + len(self.v11_rows) + len(self.v22_rows)
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
        data["appended_v22_rows"] = len(self.v22_rows)
        return data

    def manifest(self) -> dict[str, Any]:
        return stamp_release_identity(
            super().manifest(),
            v9_rows=len(self.v9_rows),
            v10_rows=len(self.v10_rows),
            v11_rows=len(self.v11_rows),
            v22_rows=len(self.v22_rows),
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
