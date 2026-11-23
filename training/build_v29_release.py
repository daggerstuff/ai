"""Build the v29 G04 research release: the crisis family finally reaches training.

v24 through v28 all scored 10/12 under the harness on both splits, and the two
failing cases moved between releases without the row logic changing. That was
not a grounding defect, which is what v27 and v28 were each chasing. It was a
selection defect in the export path, and it was measured before v29 changed a
line of it:

* ``select_rows`` in ``export_fireworks_release`` split rows into an uncapped
  ``acute`` bucket, a ``routing`` bucket capped at 2,000 and a hash-sampled
  ``filler`` bucket of 4,000. The appended crisis rows carry only ``messages``,
  so they were never tool-excluded and never acute — they landed in the routing
  pool on their escalation wording, or in the shared filler sample.
* 5,904 crisis rows in the train partition became **1,484 selected rows, 25%**.
  Per category: assault_strangulation 384→98, means_access 536→131,
  medical_collapse 432→121, overdose_taken 432→126, passive_ideation 488→98,
  psychosis_risk_other 992→233, recent_attempt 408→100, self_harm_active
  1160→315, suicide_plan 688→173, withdrawal 384→89.
* The survivor set was keyed on a hash of the serialized row line, so it changed
  on every rebuild. That is why the failure set migrated: a different 75% was
  dropped each release.
* For the two graded cases v28 lost: the medical_collapse action pool would
  build 341 rows and keep 94; the means_access gun pool would build 56 rows and
  keep **15**, about 0.26% of the 5,749 datums.

v29 adds ``SelectionOptions.keep_replies`` and checks it before the routing /
acute / filler split, so the crisis family is uncapped. The keep set is derived
from the manifest's ``crisis_direct_*`` source label by importing the builder and
collecting every assistant reply: 1,072 distinct replies. Measured on the real
v28 release, selection goes 7,104 → 13,608 rows, with all 6,504 crisis rows
included. The routing pool drops 7,989 → 2,085 as a side effect, so the 2,000 cap
now holds genuine routing rows instead of crisis rows competing for slots.

The second change is the failure mode the two v28 losses shared: an action whose
first clause names no cluster the disclosure gave. 26 surfaces opened on a
generic frame ("Call 911 now.", "Get to an emergency department now."), which is
what let a firearm-access disclosure draw an eating-disorder action and let an
eating-disorder disclosure draw "how long you have been off food". Every
grounded action now opens on its own anchor, enforced at build time by
``opens_on_anchor``. Measured: 0 cluster-set changes (pure reordering), 0 blind
first clauses, 0 pool misses across all 18 graded cases.

Frames, controls, the strict gate, the cluster vocabularies and the abstract caps
are inherited unchanged, and the corpus is carried exactly as v17, v19, v20, v22,
v23, v24, v25, v26, v27 and v28 carried it — no repair, no strip, no phrase
lists. The family itself is still 5,904 crisis rows plus 600 controls, 6,504
rows, but for the first time all of it is trained on.

Usage::

    PYTHONPATH=/home/vivi/pixelated python -m training.build_v29_release --dry-run
    PYTHONPATH=/home/vivi/pixelated python -m training.build_v29_release
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from training import (
    build_agent_tool_sft_v29,
    build_v14_release as v14,
    build_v15_release as v15,
    build_v16_release as v16,
    build_v28_release as prev,
)

# Bound as a plain attribute rather than imported under an alias: mypy --strict treats
# `from ... import x as v29` as private, so `release.v29` would not be readable by the
# release tests.
v29 = build_agent_tool_sft_v29

ROOT = prev.ROOT

OUTPUT = ROOT / "ai/data/curated/recovery/gold-20261017/wandb-full-corpus-current-master-20261110-v29"
MANIFEST = ROOT / "docs/training/execution-2026-11-10/wandb-full-corpus-current-master-20261110-v29.json"

RELEASE_ID = "wandb-full-corpus-current-master-20261110-v29"
SHUFFLE_SEED = 20261121
SPLITS = prev.SPLITS
CREATED_AT_UTC = "2026-11-17T00:00:00Z"

V29_SOURCE_LABEL = "crisis_direct_v29"
EXPECTED_V29_TRAIN_ROWS = 6504

# v28's limitation list describes the v28 family — its size, its escalation
# ratio, its hand-written surface, its placement and the v21 revert. v29 ships
# the same family shape with a selection fix and an anchor-first surface rewrite,
# so the same five entries are restated below against v29 rather than kept as v28
# claims.
_STALE_V28_LIMITATIONS = (
    "The v28 crisis-direct rows replace",
    "The 600 control rows hold v28 at 9.8",
    "The v28 replies are hand-written",
    "The v28 family is appended to train only",
    "v28 carries no corpus repair.",
)

LIMITATIONS_V29: tuple[str, ...] = (
    (
        "The v29 crisis-direct rows replace the v28 rows rather than adding to them, so the release still "
        "teaches one register per category. The family is 6,504 rows, 7.9% of train, against roughly 81,900 "
        "corpus rows that answer a disclosure by reflecting on it. What v29 changes is how much of that family "
        "reaches the adapter, not its shape: 25% of the crisis rows survived the export in v24 through v28 "
        "because they were bucketed into a capped routing pool or a hash-sampled filler, and the survivor set "
        "moved on every rebuild. All 6,504 rows now select, so the family goes from ~7.9% of train rows to "
        "~7.9% of train rows that are actually trained on — the crisis share of the 13,608 selected rows is "
        "48%, which is the largest single distribution change in the release line and the main thing to watch. "
        "Grounding is still grounding by cluster, not by sentence, so a disclosure that gives one detail is "
        "answered about that one detail and stays silent about anything else it said. The invention metric "
        "counts concrete clusters only, because warning clauses are legitimate; replies that name a "
        "non-concrete cluster the disclosure did not give are still reported (51 medical_collapse rows from the "
        "seizure warning, 279 psychosis rows from the voice and harm modality) rather than hidden."
    ),
    (
        "The 600 control rows hold v29 at 9.8 escalations per ordinary reply. That ratio is the release's main "
        "untested risk: v22 sat at 9.3 and every one of its crisis failures was a missing escalation, so the "
        "ratio was raised, not lowered. v26 made escalation frame-invariant and v29 keeps that, which raises "
        "the chance that a disclosure which is distressing without being acute also draws one. v29 raises that "
        "exposure by training on 4.4x more crisis rows than v28 did, so an escalation that leaks out of the "
        "crisis family now has far more mass behind it. If ordinary replies start escalating, the ratio is the "
        "thing to lower next — the control rows are synthetic and short, so they under-represent the ordinary "
        "reply the model should give."
    ),
    (
        "The v29 replies are hand-written in a single imperative register and are not reviewed by a clinician. "
        "Every grounded action now opens on one of the clusters its own disclosure named — v29 rewrites the 26 "
        "surfaces that opened on a generic frame, which is what let a firearm-access disclosure draw an "
        "eating-disorder action and an eating-disorder disclosure draw 'how long you have been off food' — but "
        "a disclosure that gives two clusters at once (a gun and the pills, a cut and a burn) is still answered "
        "about one of them. The family teaches triage only: there is no safety-plan, follow-up or "
        "de-escalation content, and action surfaces of 24 for means_access, 17 for overdose_taken, 18 for "
        "recent_attempt, 13 for suicide_plan, 15 for self_harm_active, 13 for psychosis_risk_other, 13 for "
        "assault_strangulation, 11 for medical_collapse, 8 for withdrawal and 8 for passive_ideation are still "
        "small — with 4.4x more rows trained on them, the smaller surfaces are the ones most likely to be "
        "over-fitted verbatim."
    ),
    (
        "The v29 family is appended to train only, and validation and test carry no crisis-direct rows. An "
        "adapter that memorises the 838 disclosures instead of generalising the register would not be caught by "
        "the held-out splits; only the paired crisis cases probe that, and there is one case per category "
        "against 48 to 145 disclosures. Training on the whole family makes memorisation more reachable, not "
        "less, so this limitation is sharper in v29 than it was in v28. The cluster vocabularies are closed "
        "lists, so an unlisted object (a lithium blister, a barbell, a car exhaust) has no action of its own "
        "and falls to whatever cluster it does match, or to the category's abstract action. Those abstract "
        "actions are capped and the cap is enforced at build time: 1 abstract means_access disclosure and 37 "
        "abstract psychosis disclosures, which are the ones that name no target at all. Every other category "
        "carries no abstract disclosure, so a new cluster-free disclosure fails the build loudly instead of "
        "quietly joining the set."
    ),
    (
        "v29 carries no corpus repair. v21's phrase strip was reverted because it lost to v20: the corpus still "
        "contains the validation register the rubric bans, including 'that's a lot to carry' and 'not going to "
        "<verb>' frames, and the release relies on the family's mass to outvote it rather than on deleting it. "
        "v26's three reflective-lead frames are kept, so escalation competes with that register from a trained "
        "position, but the release does not guarantee the model will not emit a banned opener; the application "
        "layer runs a generation-time register guard for that."
    ),
)


def _limitations() -> tuple[str, ...]:
    """The inherited limitations minus the v28 family ones, plus the v29 ones."""
    inherited = prev._limitations()
    kept = tuple(text for text in inherited if not text.startswith(_STALE_V28_LIMITATIONS))
    if len(kept) != len(inherited) - len(_STALE_V28_LIMITATIONS):
        raise ValueError("a stale v28 limitation is no longer in the inherited list")
    return (*kept, *LIMITATIONS_V29)


def v29_candidate_rows() -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """The v29 train rows as (candidate row, projected payload messages).

    These are chat rows, not agent rows: v29 carries an empty tool catalog, so
    ``is_agent_tool`` stays false and the payload keeps the plain ``messages``
    body.
    """
    rows = v29.collect_rows()["train"]
    if len(rows) != EXPECTED_V29_TRAIN_ROWS:
        raise ValueError(f"expected {EXPECTED_V29_TRAIN_ROWS} v29 train rows, found {len(rows)}")
    out: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for item in rows:
        candidate = {"messages": item["messages"], "source": V29_SOURCE_LABEL}
        if v14.is_agent_tool(candidate):
            raise ValueError(f"{item['id']}: v29 candidate rows must not carry a tool catalog")
        # The rows are hand-written, so nothing should redact. Projecting against a
        # throwaway counter keeps the release-wide redaction counts honest: if a
        # v29 row ever trips a pattern, this fails instead of silently hiding it.
        scratch: Counter[str] = Counter()
        payload = v14.project_chat(item["messages"], scratch)
        if payload is None:
            raise ValueError(f"{item['id']}: v29 row projects to no assistant target")
        if scratch and any(scratch.values()):
            raise ValueError(f"{item['id']}: v29 row redacted {dict(scratch)}; synthetic rows must be clean")
        out.append((candidate, payload))
    return out


def stamp_release_identity(
    data: dict[str, Any], *, v9_rows: int, v10_rows: int, v11_rows: int, v29_rows: int
) -> dict[str, Any]:
    """Rewrite the identity and addenda fields for the v29 release."""
    appended_by_source = {
        v14.V9_SOURCE_LABEL: v9_rows,
        v15.V10_SOURCE_LABEL: v10_rows,
        v16.V11_SOURCE_LABEL: v11_rows,
        V29_SOURCE_LABEL: v29_rows,
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
        "User-directed v29 after the v28 eval: 'Rewrite. Fix. IMPROVE. Research better avenues, more confident "
        "attempts. Don't limit to what we have/see right here.' v24 through v28 each scored 10/12 under the "
        "harness on both splits, and the two failing cases moved between releases while the row logic did not. "
        "The measured cause was not grounding, which v27 and v28 were each chasing, but selection: select_rows "
        "bucketed the appended crisis rows into a routing pool capped at 2,000 or a hash-sampled filler of "
        "4,000, so 5,904 crisis rows in the train partition became 1,484 selected rows — 25% — and the survivor "
        "set was keyed on a hash of the serialized line, so a different 75% was dropped on every rebuild. The "
        "two graded cases v28 lost had 94 and 15 surviving rows behind them. v29 adds "
        "SelectionOptions.keep_replies, derives the keep set from the manifest's crisis_direct_* source label "
        "(1,072 distinct assistant replies), and checks it before the routing/acute/filler split; selection goes "
        "7,104 to 13,608 rows with all 6,504 crisis rows included, and the routing pool drops 7,989 to 2,085 so "
        "its cap now holds genuine routing rows. The second change is the failure mode both v28 losses shared: "
        "an action whose first clause named no cluster the disclosure gave. 26 surfaces opened on a generic "
        "frame, which is what let a firearm-access disclosure draw an eating-disorder action and an "
        "eating-disorder disclosure draw 'how long you have been off food'. Every grounded action now opens on "
        "its own anchor, enforced at build time by opens_on_anchor, with 0 cluster-set changes, 0 blind first "
        "clauses and 0 pool misses across all 18 graded cases. Frames, controls, the strict gate, the cluster "
        "vocabularies and the abstract caps are inherited unchanged and the corpus is carried exactly as v17, "
        "v19, v20, v22, v23, v24, v25, v26, v27 and v28 carried it."
    )
    return data


class _Release(prev._Release):
    """v28's pipeline with the anchor-first v29 crisis family in place of v28's."""

    shuffle_seed = SHUFFLE_SEED

    def __init__(self, indexes: v14._Indexes, output: Path, dry_run: bool) -> None:
        super().__init__(indexes, output, dry_run)
        self.v29_rows: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []

    def append_addenda(self) -> None:
        """v15's v9 and v10 addenda, then v11, then v29 — v29 replaces v28."""
        v15._Release.append_addenda(self)
        self.v11_rows = v16.v11_candidate_rows()
        for candidate, payload in self.v11_rows:
            self.append_agent_row(v16.V11_SOURCE_LABEL, candidate, payload)
        self.v29_rows = v29_candidate_rows()
        for candidate, payload in self.v29_rows:
            self.append_chat_row(V29_SOURCE_LABEL, candidate, payload)

    def verify_totals(self) -> None:
        """v16's totals, with the v29 family counted in place of v13 and v20-v28."""
        if self.purged != v15.EXPECTED_PURGED_ROWS:
            raise ValueError(f"expected {v15.EXPECTED_PURGED_ROWS} purged rows, found {self.purged}")
        if dict(self.repair_counts) != v15.EXPECTED_REPAIR_COUNTS:
            raise ValueError(f"repair population drifted: {dict(self.repair_counts)}")
        appended = len(self.v9_rows) + len(self.v10_rows) + len(self.v11_rows) + len(self.v29_rows)
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
        data.pop("appended_v28_rows", None)
        data["appended_v29_rows"] = len(self.v29_rows)
        return data

    def manifest(self) -> dict[str, Any]:
        return stamp_release_identity(
            super().manifest(),
            v9_rows=len(self.v9_rows),
            v10_rows=len(self.v10_rows),
            v11_rows=len(self.v11_rows),
            v29_rows=len(self.v29_rows),
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
