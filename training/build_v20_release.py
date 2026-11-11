"""Build the v20 G04 research release with a crisis family the corpus cannot outvote.

v19 fixed the *register* of the synthetic crisis rows and it worked: the paired
crisis suite went from v18's 6-8 of 11 to 9-10 of 11, and v19 became the best
adapter on the board. The two cases it still lost failed for a different reason.
Both replies opened "That's a lot to carry alone. Let's get you to..." — the
warm-therapy register the 82k-row corpus teaches, not the register v19's own
708 crisis rows teach. 708 rows is 0.86% of train; the corpus is 99.1%.

The register itself was never the defect. Auditing the v13 rows directly found
zero coddling phrases and zero questions in all 708, and the corpus coddling
that does exist ("that sounds", "you deserve", "that must be") is legitimate
empathy inside real session text — stripping it would gut the product voice and
still leave the ratio untouched.

v20 therefore keeps the corpus exactly as v17 and v19 carried it and rebuilds
the addendum around three levers:

* volume — 1,760 rows instead of 708 (2.1% of train), mass concentrated on the
  two categories that failed (self_harm_active, psychosis_risk_other),
* anchors — every reply must name the concrete thing the disclosure named (the
  cut, the voices, the pills, the throat, the four days without food), so a
  category-level sentence that answers nothing cannot pass the builder,
* surface — 428 distinct disclosures instead of 224, so a disclosure phrased
  outside the v13 paraphrases still lands inside the family.

Everything else is inherited from v17/v19: the candidate-level welded-argument
and reasoning-preamble repairs, the v9, v10 and v11 addenda, the fabricated
crisis purge, the held-out duplicate-target purge, and the split recipe.

Usage::

    PYTHONPATH=/home/vivi/pixelated python -m training.build_v20_release --dry-run
    PYTHONPATH=/home/vivi/pixelated python -m training.build_v20_release
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from training import (
    build_agent_tool_sft_v20 as v20,
    build_v14_release as v14,
    build_v15_release as v15,
    build_v16_release as v16,
    build_v17_release as v17,
    build_v19_release as v19,
)

ROOT = v19.ROOT

OUTPUT = ROOT / "ai/data/curated/recovery/gold-20261017/wandb-full-corpus-current-master-20261110-v20"
MANIFEST = ROOT / "docs/training/execution-2026-11-10/wandb-full-corpus-current-master-20261110-v20.json"

RELEASE_ID = "wandb-full-corpus-current-master-20261110-v20"
SHUFFLE_SEED = 20261112
SPLITS = v19.SPLITS
CREATED_AT_UTC = "2026-11-10T00:00:00Z"

V20_SOURCE_LABEL = "crisis_direct_v20"
EXPECTED_V20_TRAIN_ROWS = 1760

# v19's limitation list still describes the v13 family — its size, its breadth
# and its train-only placement. v20 replaces that family, so those sentences go.
_STALE_V13_LIMITATIONS = (
    "The v13 crisis-direct rows replace v12's crisis-surface rows",
    "The v13 family is appended to train only",
)

LIMITATIONS_V20: tuple[str, ...] = (
    (
        "The v20 crisis-direct rows replace the v13 rows rather than adding to them, so the release still "
        "teaches one register per category. The family is 1,760 rows, 2.1% of train, against roughly 80,800 "
        "corpus rows that answer a disclosure by reflecting on it; a disclosure phrased outside the 428 "
        "paraphrases of its category can still fall back to that reflective answer."
    ),
    (
        "The v20 replies are hand-written in a single imperative register and are not reviewed by a "
        "clinician. Every reply is required to name the concrete thing its disclosure named, and no reply "
        "contains a question, but they are still first responses only: there is no safety-plan, follow-up or "
        "de-escalation content, so the family teaches triage and leaves everything downstream of triage to "
        "whatever the corpus already contained."
    ),
    (
        "The v20 family is appended to train only, and validation and test carry no crisis-direct rows. An "
        "adapter that memorises the 428 disclosures instead of generalising the register would not be caught "
        "by the held-out splits; only the paired crisis cases probe that."
    ),
)


def _limitations() -> tuple[str, ...]:
    """The inherited limitations minus the v13 family ones, plus the v20 ones."""
    inherited = v19._limitations()
    kept = tuple(text for text in inherited if not text.startswith(_STALE_V13_LIMITATIONS))
    if len(kept) != len(inherited) - len(_STALE_V13_LIMITATIONS):
        raise ValueError("a stale v13 limitation is no longer in the inherited list")
    return (*kept, *LIMITATIONS_V20)


def v20_candidate_rows() -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """The v20 train rows as (candidate row, projected payload messages).

    These are chat rows, not agent rows: v20 carries an empty tool catalog, so
    ``is_agent_tool`` stays false and the payload keeps the plain ``messages``
    body.
    """
    rows = v20.collect_rows()["train"]
    if len(rows) != EXPECTED_V20_TRAIN_ROWS:
        raise ValueError(f"expected {EXPECTED_V20_TRAIN_ROWS} v20 train rows, found {len(rows)}")
    out: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for item in rows:
        candidate = {"messages": item["messages"], "source": V20_SOURCE_LABEL}
        if v14.is_agent_tool(candidate):
            raise ValueError(f"{item['id']}: v20 candidate rows must not carry a tool catalog")
        # The rows are hand-written, so nothing should redact. Projecting against a
        # throwaway counter keeps the release-wide redaction counts honest: if a
        # v20 row ever trips a pattern, this fails instead of silently hiding it.
        scratch: Counter[str] = Counter()
        payload = v14.project_chat(item["messages"], scratch)
        if payload is None:
            raise ValueError(f"{item['id']}: v20 row projects to no assistant target")
        if scratch and any(scratch.values()):
            raise ValueError(f"{item['id']}: v20 row redacted {dict(scratch)}; synthetic rows must be clean")
        out.append((candidate, payload))
    return out


def stamp_release_identity(
    data: dict[str, Any], *, v9_rows: int, v10_rows: int, v11_rows: int, v20_rows: int
) -> dict[str, Any]:
    """Rewrite the identity and addenda fields for the v20 release."""
    appended_by_source = {
        v14.V9_SOURCE_LABEL: v9_rows,
        v15.V10_SOURCE_LABEL: v10_rows,
        v16.V11_SOURCE_LABEL: v11_rows,
        V20_SOURCE_LABEL: v20_rows,
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
        "User-directed v20 after the v19 paired eval. v19's register fix held — the crisis suite went from "
        "6-8 of 11 to 9-10 of 11 — and its two remaining losses both opened 'That's a lot to carry alone. "
        "Let's get you to...'. Auditing the v13 rows showed they were clean: zero coddling phrases, zero "
        "questions. The defect was arithmetic, not register. 708 crisis rows are 0.86% of an 82k-row train "
        "that answers a disclosure by reflecting on it, so generation regressed to the majority register. "
        "v20 keeps the corpus untouched and rebuilds the addendum: 1,760 rows (2.1% of train) concentrated on "
        "the two categories that failed, 428 distinct disclosures instead of 224, and a builder guard that "
        "rejects any reply which does not name the concrete thing its disclosure named."
    )
    return data


class _Release(v17._Release):
    """v17's pipeline with the larger, anchor-enforced v20 crisis family in place of v13."""

    def __init__(self, indexes: v14._Indexes, output: Path, dry_run: bool) -> None:
        super().__init__(indexes, output, dry_run)
        self.v20_rows: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []

    def append_addenda(self) -> None:
        """v15's v9 and v10 addenda, then v11, then v20 — v20 replaces v13."""
        v15._Release.append_addenda(self)
        self.v11_rows = v16.v11_candidate_rows()
        for candidate, payload in self.v11_rows:
            self.append_agent_row(v16.V11_SOURCE_LABEL, candidate, payload)
        self.v20_rows = v20_candidate_rows()
        for candidate, payload in self.v20_rows:
            self.append_chat_row(V20_SOURCE_LABEL, candidate, payload)

    def verify_totals(self) -> None:
        """v16's totals, with the v20 family counted in place of v12 and v13."""
        if self.purged != v15.EXPECTED_PURGED_ROWS:
            raise ValueError(f"expected {v15.EXPECTED_PURGED_ROWS} purged rows, found {self.purged}")
        if dict(self.repair_counts) != v15.EXPECTED_REPAIR_COUNTS:
            raise ValueError(f"repair population drifted: {dict(self.repair_counts)}")
        appended = len(self.v9_rows) + len(self.v10_rows) + len(self.v11_rows) + len(self.v20_rows)
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
        data.pop("appended_v13_rows", None)
        data["appended_v20_rows"] = len(self.v20_rows)
        return data

    def manifest(self) -> dict[str, Any]:
        return stamp_release_identity(
            super().manifest(),
            v9_rows=len(self.v9_rows),
            v10_rows=len(self.v10_rows),
            v11_rows=len(self.v11_rows),
            v20_rows=len(self.v20_rows),
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
