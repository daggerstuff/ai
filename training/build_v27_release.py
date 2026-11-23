"""Build the v27 G04 research release: grounding that covers every crisis category.

v26 grounded three of the ten crisis categories (``means_access``,
``overdose_taken``, ``recent_attempt``) on the object the disclosure named. The
other seven still rotated their eight actions across every disclosure of the
category, so the reply routinely named something the person never gave. Measured
over those seven categories, 4,192 of 4,528 disclosure x action pairs (92.6%)
carried an anchor absent from the disclosure:

* ``self_harm_active`` — "I have been burning myself with a cigarette lighter and
  the skin is broken now" was answered with "Get that cut closed at an emergency
  department tonight", a cut the person never described.
* ``withdrawal`` — opioid withdrawal ("I stopped the painkillers two days ago and
  my legs will not stay still") was answered with "Alcohol and sedative
  withdrawal causes seizures."
* ``suicide_plan`` — "I bought a rope yesterday" was answered with "the pills set
  aside for this".
* ``assault_strangulation`` — "He hit me in the ribs with a chair" was answered
  with "Being choked causes damage."

Those anchors were not chosen from the disclosures at all: they are the paired
crisis cases' ``response_contains_any`` terms, and the eight actions were written
to hit them. The surface was fitted to the eval rather than to what people say.

v27 replaces flat anchor matching with cluster-aware grounding. Each of the seven
categories gets ``ANCHOR_GROUPS`` — named clusters of tokens (self-harm ``cut`` vs
``burn`` vs ``wall`` vs ``scratch`` vs ``infection``; withdrawal ``alcohol`` vs
``benzo`` vs ``opioid`` vs ``sleeping``; psychosis target reachability
``housemate`` vs ``family`` vs ``public`` vs ``pet``; and so on). A disclosure's
clusters are the ones it mentions; an action is only eligible when the clusters it
names are a subset of those, enforced twice: in ``action_pool`` so an inventing
action is never a candidate, and in ``validate_grounded`` so the build fails per
row if one slips through. ``SECONDARY_CLUSTERS`` keeps a disclosure that named a
mechanism from being answered by a date-only or stance-only action. The
invention metric is now concrete-only, because a warning clause ("causes seizures
in the first days") is legitimate where claiming a gun is not.

The seven categories also stop being eval-shaped: each one's action surface is
written per cluster (11 to 13 actions each) and pinned by a test that every action
in a graded case's pool contains one of that case's content terms, without making
every action eval-shaped.

That is 5,904 crisis rows plus 600 controls, 6,504 rows against roughly 81,900
corpus rows: 7.9% of train, unchanged since v24. Frames, controls, the strict gate
at build time and the three v26 categories are inherited unchanged, and the corpus
is carried exactly as v17, v19, v20, v22, v23, v24, v25 and v26 carried it — no
repair, no strip, no phrase lists.

Usage::

    PYTHONPATH=/home/vivi/pixelated python -m training.build_v27_release --dry-run
    PYTHONPATH=/home/vivi/pixelated python -m training.build_v27_release
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from training import (
    build_agent_tool_sft_v27,
    build_v14_release as v14,
    build_v15_release as v15,
    build_v16_release as v16,
    build_v26_release as prev,
)

# Bound as a plain attribute rather than imported under an alias: mypy --strict treats
# `from ... import x as v27` as private, so `release.v27` would not be readable by the
# release tests.
v27 = build_agent_tool_sft_v27

ROOT = prev.ROOT

OUTPUT = ROOT / "ai/data/curated/recovery/gold-20261017/wandb-full-corpus-current-master-20261110-v27"
MANIFEST = ROOT / "docs/training/execution-2026-11-10/wandb-full-corpus-current-master-20261110-v27.json"

RELEASE_ID = "wandb-full-corpus-current-master-20261110-v27"
SHUFFLE_SEED = 20261119
SPLITS = prev.SPLITS
CREATED_AT_UTC = "2026-11-17T00:00:00Z"

V27_SOURCE_LABEL = "crisis_direct_v27"
EXPECTED_V27_TRAIN_ROWS = 6504

# v26's limitation list describes the v26 family — its size, its escalation
# ratio, its hand-written surface, its placement and the v21 revert — and the
# first of them states that grounding covers only three of the ten categories,
# which is exactly what v27 fixes. All five go; the revert history is restated
# below.
_STALE_V26_LIMITATIONS = (
    "The v26 crisis-direct rows replace",
    "The 600 control rows hold v26 at 9.8",
    "The v26 replies are hand-written",
    "The v26 family is appended to train only",
    "v26 carries no corpus repair.",
)

LIMITATIONS_V27: tuple[str, ...] = (
    (
        "The v27 crisis-direct rows replace the v26 rows rather than adding to them, so the release still "
        "teaches one register per category. The family is 6,504 rows, 7.9% of train, against roughly 81,900 "
        "corpus rows that answer a disclosure by reflecting on it. Grounding now covers all ten categories, "
        "but it is grounding by cluster, not by sentence: a reply may only name a cluster the disclosure "
        "named, so a disclosure that gives one detail is answered about that one detail and stays silent about "
        "anything else it said. The invention metric counts concrete clusters only, because warning clauses "
        "are legitimate; replies that name a non-concrete cluster the disclosure did not give are still "
        "reported (51 medical_collapse rows from the seizure warning, 279 psychosis rows from the voice and "
        "harm modality) rather than hidden."
    ),
    (
        "The 600 control rows hold v27 at 9.8 escalations per ordinary reply. That ratio is the release's "
        "main untested risk: v22 sat at 9.3 and every one of its crisis failures was a missing escalation, so "
        "the ratio was raised, not lowered. v26 made escalation frame-invariant and v27 keeps that, which "
        "raises the chance that a disclosure which is distressing without being acute also draws one. If it "
        "does, the ratio is the thing to lower next — the control rows are synthetic and short, so they "
        "under-represent the ordinary reply the model should give."
    ),
    (
        "The v27 replies are hand-written in a single imperative register and are not reviewed by a "
        "clinician. Every grounded action names only the clusters the disclosure named, so a disclosure that "
        "gives two at once (a gun and the pills, a cut and a burn) is answered about one of them. The family "
        "teaches triage only: there is no safety-plan, follow-up or de-escalation content, and action surfaces "
        "of 24 for means_access, 17 for overdose_taken, 18 for recent_attempt, 13 for suicide_plan, 15 for "
        "self_harm_active, 13 for psychosis_risk_other, 13 for assault_strangulation, 11 for medical_collapse, "
        "8 for withdrawal and 8 for passive_ideation are still small."
    ),
    (
        "The v27 family is appended to train only, and validation and test carry no crisis-direct rows. An "
        "adapter that memorises the 838 disclosures instead of generalising the register would not be caught "
        "by the held-out splits; only the paired crisis cases probe that, and there is one case per category "
        "against 48 to 145 disclosures. The cluster vocabularies are closed lists, so an unlisted object (a "
        "lithium blister, a barbell, a car exhaust) has no action of its own and falls to whatever cluster it "
        "does match, or to the category's abstract action. Those abstract actions are capped and the cap is "
        "enforced at build time: 1 abstract means_access disclosure and 37 abstract psychosis disclosures, "
        "which are the ones that name no target at all. Every other category carries no abstract disclosure, "
        "so a new cluster-free disclosure fails the build loudly instead of quietly joining the set."
    ),
    (
        "v27 carries no corpus repair. v21's phrase strip was reverted because it lost to v20: the corpus "
        "still contains the validation register the rubric bans, including 'that's a lot to carry' and 'not "
        "going to <verb>' frames, and the release relies on the family's mass to outvote it rather than on "
        "deleting it. v26's three reflective-lead frames are kept, so escalation competes with that register "
        "from a trained position, but the release does not guarantee the model will not emit a banned opener; "
        "the application layer runs a generation-time register guard for that."
    ),
)


def _limitations() -> tuple[str, ...]:
    """The inherited limitations minus the v26 family ones, plus the v27 ones."""
    inherited = prev._limitations()
    kept = tuple(text for text in inherited if not text.startswith(_STALE_V26_LIMITATIONS))
    if len(kept) != len(inherited) - len(_STALE_V26_LIMITATIONS):
        raise ValueError("a stale v26 limitation is no longer in the inherited list")
    return (*kept, *LIMITATIONS_V27)


def v27_candidate_rows() -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """The v27 train rows as (candidate row, projected payload messages).

    These are chat rows, not agent rows: v27 carries an empty tool catalog, so
    ``is_agent_tool`` stays false and the payload keeps the plain ``messages``
    body.
    """
    rows = v27.collect_rows()["train"]
    if len(rows) != EXPECTED_V27_TRAIN_ROWS:
        raise ValueError(f"expected {EXPECTED_V27_TRAIN_ROWS} v27 train rows, found {len(rows)}")
    out: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for item in rows:
        candidate = {"messages": item["messages"], "source": V27_SOURCE_LABEL}
        if v14.is_agent_tool(candidate):
            raise ValueError(f"{item['id']}: v27 candidate rows must not carry a tool catalog")
        # The rows are hand-written, so nothing should redact. Projecting against a
        # throwaway counter keeps the release-wide redaction counts honest: if a
        # v27 row ever trips a pattern, this fails instead of silently hiding it.
        scratch: Counter[str] = Counter()
        payload = v14.project_chat(item["messages"], scratch)
        if payload is None:
            raise ValueError(f"{item['id']}: v27 row projects to no assistant target")
        if scratch and any(scratch.values()):
            raise ValueError(f"{item['id']}: v27 row redacted {dict(scratch)}; synthetic rows must be clean")
        out.append((candidate, payload))
    return out


def stamp_release_identity(
    data: dict[str, Any], *, v9_rows: int, v10_rows: int, v11_rows: int, v27_rows: int
) -> dict[str, Any]:
    """Rewrite the identity and addenda fields for the v27 release."""
    appended_by_source = {
        v14.V9_SOURCE_LABEL: v9_rows,
        v15.V10_SOURCE_LABEL: v10_rows,
        v16.V11_SOURCE_LABEL: v11_rows,
        V27_SOURCE_LABEL: v27_rows,
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
        "User-directed v27 after the v26 eval: v26 grounded overdose_taken and recent_attempt and made "
        "escalation frame-invariant, and still scored 10/12 under the harness on both splits, losing the "
        "overdose case and the means-access case with byte-identical replies on dev and sealed. Both causes "
        "were measured, not guessed. The overdose reply was a generic recent_attempt line because "
        "recent_attempt's anchors were only time words and overdose_taken's actions never named the substance. "
        "The means-access reply grounded the gun and then never escalated, because all eight frames were "
        "action-first and a self-explaining disclosure had no trained position for the action. v26 grounded "
        "overdose_taken on a substance vocabulary (17 actions) and recent_attempt on a method vocabulary (18 "
        "actions), rewrote the 52 disclosures that named neither, deleted the six abstract overdose actions "
        "no disclosure can reach, and replaced three of the eight frames with reflective-lead frames so "
        "escalation is frame-invariant. v27 then measured the seven categories v26 left ungrounded: 4,192 of "
        "4,528 disclosure x action pairs (92.6%) named an anchor the disclosure never gave, because their "
        "anchors were copied from the paired eval cases' content checks and the eight actions were written to "
        "hit those checks. v27 grounds those seven by cluster, writes a per-cluster action surface for each "
        "(11 to 13 actions), adds SECONDARY_CLUSTERS so a disclosure that named a mechanism is never answered "
        "by a date-only or stance-only action, rewrites the 6 disclosures that named no cluster, and measures "
        "invention on concrete clusters only. Concrete invention across the seven is 0 of 4,528 pairs. That is "
        "6,504 rows, 7.9% of train, with means_access, overdose_taken and recent_attempt held at v26's "
        "surfaces and the corpus carried exactly as v17, v19, v20, v22, v23, v24, v25 and v26 carried it."
    )
    return data


class _Release(prev._Release):
    """v26's pipeline with the cluster-grounded v27 crisis family in place of v26's."""

    shuffle_seed = SHUFFLE_SEED

    def __init__(self, indexes: v14._Indexes, output: Path, dry_run: bool) -> None:
        super().__init__(indexes, output, dry_run)
        self.v27_rows: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []

    def append_addenda(self) -> None:
        """v15's v9 and v10 addenda, then v11, then v27 — v27 replaces v26."""
        v15._Release.append_addenda(self)
        self.v11_rows = v16.v11_candidate_rows()
        for candidate, payload in self.v11_rows:
            self.append_agent_row(v16.V11_SOURCE_LABEL, candidate, payload)
        self.v27_rows = v27_candidate_rows()
        for candidate, payload in self.v27_rows:
            self.append_chat_row(V27_SOURCE_LABEL, candidate, payload)

    def verify_totals(self) -> None:
        """v16's totals, with the v27 family counted in place of v13 and v20-v26."""
        if self.purged != v15.EXPECTED_PURGED_ROWS:
            raise ValueError(f"expected {v15.EXPECTED_PURGED_ROWS} purged rows, found {self.purged}")
        if dict(self.repair_counts) != v15.EXPECTED_REPAIR_COUNTS:
            raise ValueError(f"repair population drifted: {dict(self.repair_counts)}")
        appended = len(self.v9_rows) + len(self.v10_rows) + len(self.v11_rows) + len(self.v27_rows)
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
        data.pop("appended_v26_rows", None)
        data["appended_v27_rows"] = len(self.v27_rows)
        return data

    def manifest(self) -> dict[str, Any]:
        return stamp_release_identity(
            super().manifest(),
            v9_rows=len(self.v9_rows),
            v10_rows=len(self.v10_rows),
            v11_rows=len(self.v11_rows),
            v27_rows=len(self.v27_rows),
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
