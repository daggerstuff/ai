"""Build the v25 G04 research release: delete the sentence that answered everything.

v24 grounded every ``means_access`` pair in the means the disclosure gave, passed
the strict gate structurally on both splits, and still lost two cases to the same
sentence:

    Call 911 now and say you have the means within reach, then hand it to a person.

It answered the means-access case (a gun in a bedside drawer) without naming the
gun, and it bled onto the suicide-plan case (pills saved for Friday) as well. The
cause is a coverage hole, and it is measured rather than guessed:

* ``ANCHORS["means_access"]`` held six entries — method, means, pills, gun,
  bridge, medication — so a rope, a blade, a pistol, tablets, insulin, a railway
  or a car park counted as "a means the person described without naming".
* 32 of the 67 ``means_access`` disclosures named no recognised anchor, so they
  fell through to the abstract stand-ins.
* That made 336 of the 536 rows (63%) train on "the means" or "the method",
  which made the abstract line the category's default answer. Distribution
  dominance, not per-prompt selection: for the gun disclosure the pool was
  already the two gun actions.

v25 closes the hole three ways. The anchor vocabulary gains 17 entries (pistol,
tablets, painkillers, paracetamol, insulin, antifreeze, rope, ligature, length of
hose, blade, knife, railway, quarry, roof, coast, height, car park). The action
surface grows from 8 to 23, each naming exactly one means. The 5 disclosures that
named nothing at all are rewritten to name one. A disclosure with no recognised
means is no longer answered with an abstract stand-in: every v25 row names a
concrete means the person gave, and "the means within reach" / "the means out of
the house" appear in zero rows. "The method you have worked out" survives, because
it only fires for disclosures that say "method" themselves.

Grounding stays scoped to ``means_access``. The other nine categories keep v24's
exact training distribution, so the comparison stays one variable. Everything
else — frames, anchors, controls, the strict gate at build time, the action-first
contract — is inherited unchanged.

That is 5,904 crisis rows plus 600 controls, 6,504 rows against roughly 81,900
corpus rows: 7.9% of train, unchanged from v24. The corpus is carried exactly as
v17, v19, v20, v22, v23 and v24 carried it — no repair, no strip, no phrase lists.

Usage::

    PYTHONPATH=/home/vivi/pixelated python -m training.build_v25_release --dry-run
    PYTHONPATH=/home/vivi/pixelated python -m training.build_v25_release
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from training import (
    build_agent_tool_sft_v25 as v25,
    build_v14_release as v14,
    build_v15_release as v15,
    build_v16_release as v16,
    build_v24_release as prev,
)

ROOT = prev.ROOT

OUTPUT = ROOT / "ai/data/curated/recovery/gold-20261017/wandb-full-corpus-current-master-20261110-v25"
MANIFEST = ROOT / "docs/training/execution-2026-11-10/wandb-full-corpus-current-master-20261110-v25.json"

RELEASE_ID = "wandb-full-corpus-current-master-20261110-v25"
SHUFFLE_SEED = 20261117
SPLITS = prev.SPLITS
CREATED_AT_UTC = "2026-11-17T00:00:00Z"

V25_SOURCE_LABEL = "crisis_direct_v25"
EXPECTED_V25_TRAIN_ROWS = 6504

# v24's limitation list describes the v24 family — its size, its escalation
# ratio, its hand-written surface, its placement and the v21 revert. v25
# replaces that family, so all five go; the revert history is restated below.
_STALE_V24_LIMITATIONS = (
    "The v24 crisis-direct rows replace",
    "The 600 control rows hold the family at 9.8 escalations",
    "The v24 replies are hand-written",
    "The v24 family is appended to train only",
    "v24 carries no corpus repair.",
)

LIMITATIONS_V25: tuple[str, ...] = (
    (
        "The v25 crisis-direct rows replace the v24 rows rather than adding to them, so the release still "
        "teaches one register per category. The family is 6,504 rows, 7.9% of train, against roughly 81,900 "
        "corpus rows that answer a disclosure by reflecting on it. Grounding covers means_access only: the "
        "other nine categories still rotate their eight actions across every disclosure of the category, so "
        "a reply there can still name a detail the person never gave. That is the measured cause of the "
        "remaining invented-detail risk, and it is untouched by this release on purpose."
    ),
    (
        "The 600 control rows hold v25 at 9.8 escalations per ordinary reply. That ratio is the "
        "release's main untested risk: v22 sat at 9.3 and every one of its crisis failures was a missing "
        "escalation, so the ratio was raised, not lowered. If v25 escalates on a disclosure that is "
        "distressing without being acute, the ratio is the thing to lower next — the control rows are "
        "synthetic and short, so they under-represent the ordinary reply the model should give."
    ),
    (
        "The v25 replies are hand-written in a single imperative register and are not reviewed by a "
        "clinician. Each means_access action names exactly one means, so a disclosure that gives two means "
        "at once (a gun and the pills) is answered about one of them and stays silent about the other rather "
        "than handling both. The family teaches triage only: there is no safety-plan, follow-up or "
        "de-escalation content, and 24 actions for means_access is still a small surface."
    ),
    (
        "The v25 family is appended to train only, and validation and test carry no crisis-direct rows. An "
        "adapter that memorises the 838 disclosures instead of generalising the register would not be caught "
        "by the held-out splits; only the paired crisis cases probe that, and there is one means-access case "
        "against 67 disclosures. 66 of those 67 disclosures name an object the action surface carries. The "
        "anchor vocabulary is still a closed list, so an unlisted means (a barbell, a car exhaust) has no "
        "action of its own: the single remaining abstract disclosure keeps the two fallback replies exercised, "
        "and an unlisted object is answered about the person's own 'means' or 'method' words rather than the "
        "object."
    ),
    (
        "v25 carries no corpus repair. v21's phrase strip was reverted because it lost to v20: the corpus "
        "still contains the validation register the rubric bans, including 'that's a lot to carry' and 'not "
        "going to <verb>' frames, and the release relies on the family's mass to outvote it rather than on "
        "deleting it. The application layer runs a generation-time register guard, but the release itself "
        "does not guarantee the model will not emit a banned opener."
    ),
)


def _limitations() -> tuple[str, ...]:
    """The inherited limitations minus the v24 family ones, plus the v25 ones."""
    inherited = prev._limitations()
    kept = tuple(text for text in inherited if not text.startswith(_STALE_V24_LIMITATIONS))
    if len(kept) != len(inherited) - len(_STALE_V24_LIMITATIONS):
        raise ValueError("a stale v24 limitation is no longer in the inherited list")
    return (*kept, *LIMITATIONS_V25)


def v25_candidate_rows() -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """The v25 train rows as (candidate row, projected payload messages).

    These are chat rows, not agent rows: v25 carries an empty tool catalog, so
    ``is_agent_tool`` stays false and the payload keeps the plain ``messages``
    body.
    """
    rows = v25.collect_rows()["train"]
    if len(rows) != EXPECTED_V25_TRAIN_ROWS:
        raise ValueError(f"expected {EXPECTED_V25_TRAIN_ROWS} v25 train rows, found {len(rows)}")
    out: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for item in rows:
        candidate = {"messages": item["messages"], "source": V25_SOURCE_LABEL}
        if v14.is_agent_tool(candidate):
            raise ValueError(f"{item['id']}: v25 candidate rows must not carry a tool catalog")
        # The rows are hand-written, so nothing should redact. Projecting against a
        # throwaway counter keeps the release-wide redaction counts honest: if a
        # v25 row ever trips a pattern, this fails instead of silently hiding it.
        scratch: Counter[str] = Counter()
        payload = v14.project_chat(item["messages"], scratch)
        if payload is None:
            raise ValueError(f"{item['id']}: v25 row projects to no assistant target")
        if scratch and any(scratch.values()):
            raise ValueError(f"{item['id']}: v25 row redacted {dict(scratch)}; synthetic rows must be clean")
        out.append((candidate, payload))
    return out


def stamp_release_identity(
    data: dict[str, Any], *, v9_rows: int, v10_rows: int, v11_rows: int, v25_rows: int
) -> dict[str, Any]:
    """Rewrite the identity and addenda fields for the v25 release."""
    appended_by_source = {
        v14.V9_SOURCE_LABEL: v9_rows,
        v15.V10_SOURCE_LABEL: v10_rows,
        v16.V11_SOURCE_LABEL: v11_rows,
        V25_SOURCE_LABEL: v25_rows,
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
        "User-directed v25 after the v24 eval: v24 passed the strict gate structurally on both splits and "
        "still scored 10/12 under the harness on both, losing the means-access case and the suicide-plan case "
        "to one sentence, 'Call 911 now and say you have the means within reach'. The cause was measured, not "
        "guessed: ANCHORS['means_access'] held six entries, so 32 of 67 disclosures named no recognised means "
        "and 336 of 536 rows (63%) trained on the abstract stand-ins, making that line the category default. "
        "v25 adds 17 anchors, grows the action surface from 8 to 24 with exactly one means each, rewrites the "
        "13 disclosures that named no object, and removes 'the means within reach' and 'the means out of the "
        "house' from the corpus entirely — 0 abstract rows, down from 336, with 66 of 67 disclosures now "
        "naming a concrete means. That is 6,504 rows, 7.9% of train, "
        "with the other nine categories held at v24's distribution and the corpus carried exactly as v17, "
        "v19, v20, v22, v23 and v24 carried it."
    )
    return data


class _Release(prev._Release):
    """v24's pipeline with the 23-action grounded v25 crisis family in place of v24's."""

    def __init__(self, indexes: v14._Indexes, output: Path, dry_run: bool) -> None:
        super().__init__(indexes, output, dry_run)
        self.v25_rows: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []

    def append_addenda(self) -> None:
        """v15's v9 and v10 addenda, then v11, then v25 — v25 replaces v24."""
        v15._Release.append_addenda(self)
        self.v11_rows = v16.v11_candidate_rows()
        for candidate, payload in self.v11_rows:
            self.append_agent_row(v16.V11_SOURCE_LABEL, candidate, payload)
        self.v25_rows = v25_candidate_rows()
        for candidate, payload in self.v25_rows:
            self.append_chat_row(V25_SOURCE_LABEL, candidate, payload)

    def verify_totals(self) -> None:
        """v16's totals, with the v25 family counted in place of v13 and v20-v24."""
        if self.purged != v15.EXPECTED_PURGED_ROWS:
            raise ValueError(f"expected {v15.EXPECTED_PURGED_ROWS} purged rows, found {self.purged}")
        if dict(self.repair_counts) != v15.EXPECTED_REPAIR_COUNTS:
            raise ValueError(f"repair population drifted: {dict(self.repair_counts)}")
        appended = len(self.v9_rows) + len(self.v10_rows) + len(self.v11_rows) + len(self.v25_rows)
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
        data.pop("appended_v24_rows", None)
        data["appended_v25_rows"] = len(self.v25_rows)
        return data

    def manifest(self) -> dict[str, Any]:
        return stamp_release_identity(
            super().manifest(),
            v9_rows=len(self.v9_rows),
            v10_rows=len(self.v10_rows),
            v11_rows=len(self.v11_rows),
            v25_rows=len(self.v25_rows),
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
