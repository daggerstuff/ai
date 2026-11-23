"""Build the v23 G04 research release: train the categories that never escalated.

v22 attacked the register with volume and won the split it was built for: eight
frames per disclosure, eight actions per category, 6,184 rows, 7.0% of train.
What it did not do was check that the rows it generated actually say the thing
the rubric requires. Measured against the strict gate, 816 of v22's 6,184 rows
fail, and they fail in exactly two categories:

* ``passive_ideation`` — 408 rows (51 disclosures x 8 frames), all "no
  escalation". "Everyone would be better off if I wasn't around" was answered
  with a hand-written line that named the anchor and offered nothing else.
* ``means_access`` — 408 rows, same shape. A disclosure about having the method
  and the means in hand was answered without a route out.

Both categories are ``engaged`` in v9's severity map, not ``acute``, so
``validate_crisis_row`` never required an escalation clause from them and
``test_every_acute_reply_escalates`` never looked. The other eight categories
were already 100% escalation-bearing, which is why the gap survived three
rounds of rubric work.

v23 fixes the rows and closes the hole that let them through:

* whole-category replacement — all eight ``passive_ideation`` actions and all
  eight ``means_access`` actions are rewritten to carry a route out (988, a
  crisis line, 911, urgent care, an emergency department) plus the anchor, and
  the means actions add the means-restriction step: hand the pills over, put the
  gun down, get the medication out of the house, get off the bridge,
* detail-free medical_collapse — three of the eight actions invented clinical
  detail the disclosure never gave ("an irregular heartbeat", "a rest problem",
  "how electrolytes crash"). The judge failed replies for it and it was in the
  source rows, not the model. All eight are recast to name only what the
  person said: blacked out, bathroom, purging, five days, faint,
* 24 new disclosures — self_harm_active +8 in the "deeper than usual and I do
  not want to stop" register, passive_ideation +10, medical_collapse +6 —
  bringing the family to 822 disclosures,
* a build-time gate — every generated row is graded with
  ``strict_crisis_gate.structural_violations`` before it can be written, crisis
  rows as crisis and control rows as control. A row that does not escalate, or a
  control row that does, now fails the build instead of the eval.

That is 5,776 crisis rows plus 600 controls, 6,376 rows against roughly 81,900
corpus rows: 7.2% of train, up from v22's 6,184. The corpus is carried exactly
as v17, v19, v20 and v22 carried it — no repair, no strip, no phrase lists.

Usage::

    PYTHONPATH=/home/vivi/pixelated python -m training.build_v23_release --dry-run
    PYTHONPATH=/home/vivi/pixelated python -m training.build_v23_release
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from training import (
    build_agent_tool_sft_v23 as v23,
    build_v14_release as v14,
    build_v15_release as v15,
    build_v16_release as v16,
    build_v22_release as prev,
)

ROOT = prev.ROOT

OUTPUT = ROOT / "ai/data/curated/recovery/gold-20261017/wandb-full-corpus-current-master-20261110-v23"
MANIFEST = ROOT / "docs/training/execution-2026-11-10/wandb-full-corpus-current-master-20261110-v23.json"

RELEASE_ID = "wandb-full-corpus-current-master-20261110-v23"
SHUFFLE_SEED = 20261115
SPLITS = prev.SPLITS
CREATED_AT_UTC = "2026-11-16T00:00:00Z"

V23_SOURCE_LABEL = "crisis_direct_v23"
EXPECTED_V23_TRAIN_ROWS = 6376

# v22's limitation list describes the v22 family — its size, its register, its
# escalation ratio, its placement and the v21 revert. v23 replaces that family
# and keeps the revert history in its own words, so all five go.
_STALE_V22_LIMITATIONS = (
    "The v22 crisis-direct rows replace",
    "The 600 control rows hold the family at 9.3 escalations",
    "The v22 replies are hand-written",
    "The v22 family is appended to train only",
    "v22 carries no corpus repair.",
)

LIMITATIONS_V23: tuple[str, ...] = (
    (
        "The v23 crisis-direct rows replace the v22 rows rather than adding to them, so the release still "
        "teaches one register per category. The family is 6,376 rows, 7.2% of train, against roughly 81,900 "
        "corpus rows that answer a disclosure by reflecting on it. Volume is the only thing keeping the "
        "register in the majority: a disclosure phrased outside the 822 paraphrases of its category, or a "
        "case that needs more than one turn, can still fall back to the reflective answer."
    ),
    (
        "The 600 control rows hold the family at 9.6 escalations per ordinary reply. That ratio is the "
        "release's main untested risk: v22 sat at 9.3 and every one of its crisis failures was a missing "
        "escalation, so the ratio was raised, not lowered. If v23 escalates on a disclosure that is "
        "distressing without being acute, the ratio is the thing to lower next — the control rows are "
        "synthetic and short, so they under-represent the ordinary reply the model should give."
    ),
    (
        "The v23 replies are hand-written in a single imperative register and are not reviewed by a "
        "clinician. Every reply is required to name the concrete thing its disclosure named, and no reply "
        "contains a question, but they are still first responses only: there is no safety-plan, follow-up or "
        "de-escalation content, so the family teaches triage and leaves everything downstream of triage to "
        "whatever the corpus already contained. Eight actions per category is still a small surface, and the "
        "evaluation can reward a reply for naming an anchor that a real clinician would say differently."
    ),
    (
        "The v23 family is appended to train only, and validation and test carry no crisis-direct rows. An "
        "adapter that memorises the 822 disclosures instead of generalising the register would not be caught "
        "by the held-out splits; only the paired crisis cases probe that."
    ),
    (
        "v23 carries no corpus repair. v21's phrase strip was reverted because it lost to v20: the corpus "
        "still contains the validation register the rubric bans, including 'that's a lot to carry' and 'not "
        "going to <verb>' frames, and the release relies on the family's mass to outvote it rather than on "
        "deleting it. The application layer runs a generation-time register guard, but the release itself "
        "does not guarantee the model will not emit a banned opener."
    ),
)


def _limitations() -> tuple[str, ...]:
    """The inherited limitations minus the v22 family ones, plus the v23 ones."""
    inherited = prev._limitations()
    kept = tuple(text for text in inherited if not text.startswith(_STALE_V22_LIMITATIONS))
    if len(kept) != len(inherited) - len(_STALE_V22_LIMITATIONS):
        raise ValueError("a stale v22 limitation is no longer in the inherited list")
    return (*kept, *LIMITATIONS_V23)


def v23_candidate_rows() -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """The v23 train rows as (candidate row, projected payload messages).

    These are chat rows, not agent rows: v23 carries an empty tool catalog, so
    ``is_agent_tool`` stays false and the payload keeps the plain ``messages``
    body.
    """
    rows = v23.collect_rows()["train"]
    if len(rows) != EXPECTED_V23_TRAIN_ROWS:
        raise ValueError(f"expected {EXPECTED_V23_TRAIN_ROWS} v23 train rows, found {len(rows)}")
    out: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for item in rows:
        candidate = {"messages": item["messages"], "source": V23_SOURCE_LABEL}
        if v14.is_agent_tool(candidate):
            raise ValueError(f"{item['id']}: v23 candidate rows must not carry a tool catalog")
        # The rows are hand-written, so nothing should redact. Projecting against a
        # throwaway counter keeps the release-wide redaction counts honest: if a
        # v23 row ever trips a pattern, this fails instead of silently hiding it.
        scratch: Counter[str] = Counter()
        payload = v14.project_chat(item["messages"], scratch)
        if payload is None:
            raise ValueError(f"{item['id']}: v23 row projects to no assistant target")
        if scratch and any(scratch.values()):
            raise ValueError(f"{item['id']}: v23 row redacted {dict(scratch)}; synthetic rows must be clean")
        out.append((candidate, payload))
    return out


def stamp_release_identity(
    data: dict[str, Any], *, v9_rows: int, v10_rows: int, v11_rows: int, v23_rows: int
) -> dict[str, Any]:
    """Rewrite the identity and addenda fields for the v23 release."""
    appended_by_source = {
        v14.V9_SOURCE_LABEL: v9_rows,
        v15.V10_SOURCE_LABEL: v10_rows,
        v16.V11_SOURCE_LABEL: v11_rows,
        V23_SOURCE_LABEL: v23_rows,
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
        "User-directed v23 after auditing v22 against the strict crisis gate. v22 is the volume round: 6,184 "
        "crisis-direct rows, 7.0% of train, eight frames and eight actions per category. It still failed 816 "
        "of those rows — 408 passive_ideation and 408 means_access, every one of them 'no escalation' — "
        "because both categories are engaged rather than acute in the severity map, so the row validator and "
        "the acute-reply test never asked them for a route out. v23 replaces all sixteen actions in those two "
        "categories with escalation-bearing lines (and the means actions add the means-restriction step), "
        "recasts the eight medical_collapse actions detail-free after the judge caught 'an irregular "
        "heartbeat' and 'a rest problem' in the source rows, adds 24 disclosures, and grades every generated "
        "row with the strict gate at build time so a row that does not escalate, or a control row that does, "
        "fails the build instead of the eval. That is 6,376 rows, 7.2% of train, with the corpus carried "
        "exactly as v17, v19, v20 and v22 carried it."
    )
    return data


class _Release(prev._Release):
    """v22's pipeline with the escalation-bearing v23 crisis family in place of v22's."""

    shuffle_seed = SHUFFLE_SEED

    def __init__(self, indexes: v14._Indexes, output: Path, dry_run: bool) -> None:
        super().__init__(indexes, output, dry_run)
        self.v23_rows: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []

    def append_addenda(self) -> None:
        """v15's v9 and v10 addenda, then v11, then v23 — v23 replaces v22."""
        v15._Release.append_addenda(self)
        self.v11_rows = v16.v11_candidate_rows()
        for candidate, payload in self.v11_rows:
            self.append_agent_row(v16.V11_SOURCE_LABEL, candidate, payload)
        self.v23_rows = v23_candidate_rows()
        for candidate, payload in self.v23_rows:
            self.append_chat_row(V23_SOURCE_LABEL, candidate, payload)

    def verify_totals(self) -> None:
        """v16's totals, with the v23 family counted in place of v13, v20, v21 and v22."""
        if self.purged != v15.EXPECTED_PURGED_ROWS:
            raise ValueError(f"expected {v15.EXPECTED_PURGED_ROWS} purged rows, found {self.purged}")
        if dict(self.repair_counts) != v15.EXPECTED_REPAIR_COUNTS:
            raise ValueError(f"repair population drifted: {dict(self.repair_counts)}")
        appended = len(self.v9_rows) + len(self.v10_rows) + len(self.v11_rows) + len(self.v23_rows)
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
        data.pop("appended_v22_rows", None)
        data["appended_v23_rows"] = len(self.v23_rows)
        return data

    def manifest(self) -> dict[str, Any]:
        return stamp_release_identity(
            super().manifest(),
            v9_rows=len(self.v9_rows),
            v10_rows=len(self.v10_rows),
            v11_rows=len(self.v11_rows),
            v23_rows=len(self.v23_rows),
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
