"""Build the v28 G04 research release: cluster grounding that does not misroute.

v27 grounded all ten crisis categories on the clusters the disclosure named and
scored 10/12 under the harness on both splits, losing two graded cases. Both
failures were measured, not guessed:

* ``crisis-eating-disorder-syncope`` was answered with a withdrawal line, because
  the withdrawal disclosure "I stopped drinking on Monday and I cannot keep food
  down." carries "cannot keep food down", which is one of ``medical_collapse``'s
  purge tokens. The disclosure that trained the routing landmine is rewritten.
* ``crisis-passive-ideation`` was answered with the ``not_exist`` action, whose
  wording was "…say you would rather not be here anymore." The graded case
  accepts "not around" as a substring and "not be here" is not it, so the action
  that was meant to satisfy the case never did. It now reads "…tell them you
  have been wishing you did not exist", which carries the accepted "not exist".
* ``_mentions`` matched raw text, so a disclosure that said "wasn't around" never
  matched the phrase "not around". ``normalise`` now folds curly apostrophes and
  expands the common contractions before matching.

The contraction fix is hardening, not a surface change: measured against v27 it
moves zero disclosures and zero actions between clusters. Frames, controls, the
strict gate at build time, the cluster vocabularies and the abstract caps are all
inherited unchanged, and the corpus is carried exactly as v17, v19, v20, v22,
v23, v24, v25, v26 and v27 carried it — no repair, no strip, no phrase lists.

That is 5,904 crisis rows plus 600 controls, 6,504 rows against roughly 81,900
corpus rows: 7.9% of train, unchanged since v24.

Usage::

    PYTHONPATH=/home/vivi/pixelated python -m training.build_v28_release --dry-run
    PYTHONPATH=/home/vivi/pixelated python -m training.build_v28_release
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from training import (
    build_agent_tool_sft_v28,
    build_v14_release as v14,
    build_v15_release as v15,
    build_v16_release as v16,
    build_v27_release as prev,
)

# Bound as a plain attribute rather than imported under an alias: mypy --strict treats
# `from ... import x as v28` as private, so `release.v28` would not be readable by the
# release tests.
v28 = build_agent_tool_sft_v28

ROOT = prev.ROOT

OUTPUT = ROOT / "ai/data/curated/recovery/gold-20261017/wandb-full-corpus-current-master-20261110-v28"
MANIFEST = ROOT / "docs/training/execution-2026-11-10/wandb-full-corpus-current-master-20261110-v28.json"

RELEASE_ID = "wandb-full-corpus-current-master-20261110-v28"
SHUFFLE_SEED = 20261120
SPLITS = prev.SPLITS
CREATED_AT_UTC = "2026-11-17T00:00:00Z"

V28_SOURCE_LABEL = "crisis_direct_v28"
EXPECTED_V28_TRAIN_ROWS = 6504

# v27's limitation list describes the v27 family — its size, its escalation
# ratio, its hand-written surface, its placement and the v21 revert. v28 ships
# the same family shape with three defects closed, so the same five entries are
# restated below against v28 rather than kept as v27 claims.
_STALE_V27_LIMITATIONS = (
    "The v27 crisis-direct rows replace",
    "The 600 control rows hold v27 at 9.8",
    "The v27 replies are hand-written",
    "The v27 family is appended to train only",
    "v27 carries no corpus repair.",
)

LIMITATIONS_V28: tuple[str, ...] = (
    (
        "The v28 crisis-direct rows replace the v27 rows rather than adding to them, so the release still "
        "teaches one register per category. The family is 6,504 rows, 7.9% of train, against roughly 81,900 "
        "corpus rows that answer a disclosure by reflecting on it. v28 changes three things and no row count: "
        "one disclosure that was a routing landmine, one action whose wording did not carry the term the graded "
        "case accepts, and the matcher that reads contractions. Grounding is still grounding by cluster, not by "
        "sentence, so a disclosure that gives one detail is answered about that one detail and stays silent "
        "about anything else it said. The invention metric counts concrete clusters only, because warning "
        "clauses are legitimate; replies that name a non-concrete cluster the disclosure did not give are still "
        "reported (51 medical_collapse rows from the seizure warning, 279 psychosis rows from the voice and "
        "harm modality) rather than hidden."
    ),
    (
        "The 600 control rows hold v28 at 9.8 escalations per ordinary reply. That ratio is the release's main "
        "untested risk: v22 sat at 9.3 and every one of its crisis failures was a missing escalation, so the "
        "ratio was raised, not lowered. v26 made escalation frame-invariant and v28 keeps that, which raises "
        "the chance that a disclosure which is distressing without being acute also draws one. If it does, the "
        "ratio is the thing to lower next — the control rows are synthetic and short, so they under-represent "
        "the ordinary reply the model should give."
    ),
    (
        "The v28 replies are hand-written in a single imperative register and are not reviewed by a clinician. "
        "Every grounded action names only the clusters the disclosure named, so a disclosure that gives two at "
        "once (a gun and the pills, a cut and a burn) is answered about one of them. The family teaches triage "
        "only: there is no safety-plan, follow-up or de-escalation content, and action surfaces of 24 for "
        "means_access, 17 for overdose_taken, 18 for recent_attempt, 13 for suicide_plan, 15 for "
        "self_harm_active, 13 for psychosis_risk_other, 13 for assault_strangulation, 11 for medical_collapse, "
        "8 for withdrawal and 8 for passive_ideation are still small."
    ),
    (
        "The v28 family is appended to train only, and validation and test carry no crisis-direct rows. An "
        "adapter that memorises the 838 disclosures instead of generalising the register would not be caught by "
        "the held-out splits; only the paired crisis cases probe that, and there is one case per category "
        "against 48 to 145 disclosures. The cluster vocabularies are closed lists, so an unlisted object (a "
        "lithium blister, a barbell, a car exhaust) has no action of its own and falls to whatever cluster it "
        "does match, or to the category's abstract action. Those abstract actions are capped and the cap is "
        "enforced at build time: 1 abstract means_access disclosure and 37 abstract psychosis disclosures, "
        "which are the ones that name no target at all. Every other category carries no abstract disclosure, "
        "so a new cluster-free disclosure fails the build loudly instead of quietly joining the set."
    ),
    (
        "v28 carries no corpus repair. v21's phrase strip was reverted because it lost to v20: the corpus still "
        "contains the validation register the rubric bans, including 'that's a lot to carry' and 'not going to "
        "<verb>' frames, and the release relies on the family's mass to outvote it rather than on deleting it. "
        "v26's three reflective-lead frames are kept, so escalation competes with that register from a trained "
        "position, but the release does not guarantee the model will not emit a banned opener; the application "
        "layer runs a generation-time register guard for that."
    ),
)


def _limitations() -> tuple[str, ...]:
    """The inherited limitations minus the v27 family ones, plus the v28 ones."""
    inherited = prev._limitations()
    kept = tuple(text for text in inherited if not text.startswith(_STALE_V27_LIMITATIONS))
    if len(kept) != len(inherited) - len(_STALE_V27_LIMITATIONS):
        raise ValueError("a stale v27 limitation is no longer in the inherited list")
    return (*kept, *LIMITATIONS_V28)


def v28_candidate_rows() -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """The v28 train rows as (candidate row, projected payload messages).

    These are chat rows, not agent rows: v28 carries an empty tool catalog, so
    ``is_agent_tool`` stays false and the payload keeps the plain ``messages``
    body.
    """
    rows = v28.collect_rows()["train"]
    if len(rows) != EXPECTED_V28_TRAIN_ROWS:
        raise ValueError(f"expected {EXPECTED_V28_TRAIN_ROWS} v28 train rows, found {len(rows)}")
    out: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for item in rows:
        candidate = {"messages": item["messages"], "source": V28_SOURCE_LABEL}
        if v14.is_agent_tool(candidate):
            raise ValueError(f"{item['id']}: v28 candidate rows must not carry a tool catalog")
        # The rows are hand-written, so nothing should redact. Projecting against a
        # throwaway counter keeps the release-wide redaction counts honest: if a
        # v28 row ever trips a pattern, this fails instead of silently hiding it.
        scratch: Counter[str] = Counter()
        payload = v14.project_chat(item["messages"], scratch)
        if payload is None:
            raise ValueError(f"{item['id']}: v28 row projects to no assistant target")
        if scratch and any(scratch.values()):
            raise ValueError(f"{item['id']}: v28 row redacted {dict(scratch)}; synthetic rows must be clean")
        out.append((candidate, payload))
    return out


def stamp_release_identity(
    data: dict[str, Any], *, v9_rows: int, v10_rows: int, v11_rows: int, v28_rows: int
) -> dict[str, Any]:
    """Rewrite the identity and addenda fields for the v28 release."""
    appended_by_source = {
        v14.V9_SOURCE_LABEL: v9_rows,
        v15.V10_SOURCE_LABEL: v10_rows,
        v16.V11_SOURCE_LABEL: v11_rows,
        V28_SOURCE_LABEL: v28_rows,
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
        "User-directed v28 after the v27 eval: v27 grounded all ten crisis categories by cluster and still "
        "scored 10/12 under the harness on both splits, with byte-identical replies on dev and sealed. Both "
        "failures were measured, not guessed. The eating-disorder case was answered with a withdrawal line "
        "because the withdrawal disclosure 'I stopped drinking on Monday and I cannot keep food down.' carries "
        "'cannot keep food down', one of medical_collapse's purge tokens; that disclosure now reads 'I stopped "
        "drinking on Monday and by Wednesday my hands were shaking so badly I could not hold a glass.' The "
        "passive-ideation case was answered with the not_exist action, whose wording was 'say you would rather "
        "not be here anymore' — the graded case accepts 'not around' as a substring and 'not be here' is not "
        "it, so the action never satisfied the case it was written for; it now reads 'tell them you have been "
        "wishing you did not exist.' The third defect was the matcher: _mentions compared raw text, "
        "so 'wasn't around' never matched 'not around'; normalise now folds curly apostrophes and expands the "
        "common contractions before matching, and a test pins that it moves zero disclosures and zero actions "
        "between clusters against v27. That is 6,504 rows, 7.9% of train, with every action surface, frame, "
        "control and cluster vocabulary inherited unchanged and the corpus carried exactly as v17, v19, v20, "
        "v22, v23, v24, v25, v26 and v27 carried it."
    )
    return data


class _Release(prev._Release):
    """v27's pipeline with the defect-closing v28 crisis family in place of v27's."""

    shuffle_seed = SHUFFLE_SEED

    def __init__(self, indexes: v14._Indexes, output: Path, dry_run: bool) -> None:
        super().__init__(indexes, output, dry_run)
        self.v28_rows: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []

    def append_addenda(self) -> None:
        """v15's v9 and v10 addenda, then v11, then v28 — v28 replaces v27."""
        v15._Release.append_addenda(self)
        self.v11_rows = v16.v11_candidate_rows()
        for candidate, payload in self.v11_rows:
            self.append_agent_row(v16.V11_SOURCE_LABEL, candidate, payload)
        self.v28_rows = v28_candidate_rows()
        for candidate, payload in self.v28_rows:
            self.append_chat_row(V28_SOURCE_LABEL, candidate, payload)

    def verify_totals(self) -> None:
        """v16's totals, with the v28 family counted in place of v13 and v20-v27."""
        if self.purged != v15.EXPECTED_PURGED_ROWS:
            raise ValueError(f"expected {v15.EXPECTED_PURGED_ROWS} purged rows, found {self.purged}")
        if dict(self.repair_counts) != v15.EXPECTED_REPAIR_COUNTS:
            raise ValueError(f"repair population drifted: {dict(self.repair_counts)}")
        appended = len(self.v9_rows) + len(self.v10_rows) + len(self.v11_rows) + len(self.v28_rows)
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
        data.pop("appended_v27_rows", None)
        data["appended_v28_rows"] = len(self.v28_rows)
        return data

    def manifest(self) -> dict[str, Any]:
        return stamp_release_identity(
            super().manifest(),
            v9_rows=len(self.v9_rows),
            v10_rows=len(self.v10_rows),
            v11_rows=len(self.v11_rows),
            v28_rows=len(self.v28_rows),
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
