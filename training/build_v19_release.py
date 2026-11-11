"""Build the v19 G04 research release on the v17 lineage with corrected crisis families.

v18 rewrote the corpus's own risk turns in place: 1,917 rows got a routing
sentence prepended to a reply that already opened warmly, and a second
deterministic pass prefixed a routing sentence wherever an acute disclosure was
answered without one. Both taught the same artefact — a crisis reply opens by
handing the person a number — and both produced double openings the adapter
learned as "warm opener first, action dropped". The rubric that scored v18 also
required a resource name on every disclosure, which is what the whole-corpus
rewrite was chasing.

The v19 direction is the opposite one. A person who says they have been cutting,
or that people would be better off without them, is already talking to someone;
redirecting them to a hotline closes the conversation the disclosure opened.
v19 therefore carries no rewrite overlay and no risk headers — it is the v17
pipeline, with the synthetic crisis families corrected:

* every crisis action answers the disclosure it is attached to instead of
  emitting a category-level referral sentence,
* acute categories (a dated plan, an overdose in progress, an injury, psychosis
  with a named target, strangulation, collapse, withdrawal, a recent attempt)
  still name emergency care, because those need a medical response today,
* the two talking categories (passive ideation, means access) never name a
  number: they answer the disclosure and ask for the means to leave the house.

Everything else is inherited from v17: the candidate-level welded-argument and
reasoning-preamble repairs, the v9, v10, v11 and v13 addenda (v13 in place of
v12), the fabricated-crisis purge, the held-out duplicate-target purge, the
split recipe.

Row counts are unchanged from v17 — this release changes the text of the
synthetic crisis rows, not how many of them exist.

Usage::

    PYTHONPATH=/home/vivi/pixelated python -m training.build_v19_release --dry-run
    PYTHONPATH=/home/vivi/pixelated python -m training.build_v19_release
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from training import build_v14_release as v14, build_v17_release as v17

ROOT = v17.ROOT

OUTPUT = ROOT / "ai/data/curated/recovery/gold-20261017/wandb-full-corpus-current-master-20261110-v19"
MANIFEST = ROOT / "docs/training/execution-2026-11-10/wandb-full-corpus-current-master-20261110-v19.json"

RELEASE_ID = "wandb-full-corpus-current-master-20261110-v19"
SHUFFLE_SEED = 20261111
SPLITS = v17.SPLITS
CREATED_AT_UTC = "2026-11-10T00:00:00Z"

# v17's second limitation describes a family that routes everything. v19 splits
# the crisis families in two, so that sentence is replaced rather than inherited.
_STALE_ROUTING_LIMITATION = "The v13 replies are hand-written"

LIMITATIONS_V19: tuple[str, ...] = (
    (
        "The crisis families answer the disclosure instead of referring it. Acute categories name "
        "emergency care; passive ideation and means access name no number at all and ask for the means to "
        "leave the house. Nothing downstream of that first turn is trained: there is no safety-plan, "
        "follow-up or de-escalation content, so the release teaches the first response and leaves the rest "
        "to whatever the corpus already contained."
    ),
    (
        "The v18 risk-turn rewrite and deterministic risk-header overlay is not carried forward. Both "
        "prepended a routing sentence to a reply that already opened warmly, which is the double opening "
        "the adapter learned as a warm opener with the action dropped. The corpus's own risk turns are "
        "therefore unrepaired in v19 and still teach 'risk on the table, reflect on it' outside the "
        "synthetic families, so a disclosure phrased outside those paraphrases can still fall back to the "
        "reflective answer."
    ),
)


def _limitations() -> tuple[str, ...]:
    """The inherited limitations minus the v17 routing one, plus the v19 ones."""
    inherited = v17._limitations()
    kept = tuple(text for text in inherited if not text.startswith(_STALE_ROUTING_LIMITATION))
    if len(kept) != len(inherited) - 1:
        raise ValueError("the v17 routing limitation is no longer in the inherited list")
    return (*kept, *LIMITATIONS_V19)


def stamp_release_identity(
    data: dict[str, Any], *, v9_rows: int, v10_rows: int, v11_rows: int, v13_rows: int
) -> dict[str, Any]:
    """Rewrite the identity and direction fields for the v19 release."""
    data = v17.stamp_release_identity(
        data,
        v9_rows=v9_rows,
        v10_rows=v10_rows,
        v11_rows=v11_rows,
        v13_rows=v13_rows,
    )
    data["release_id"] = RELEASE_ID
    data["created_at_utc"] = CREATED_AT_UTC
    data["payload_contract"]["shuffle"]["seed"] = SHUFFLE_SEED
    data["limitations"] = list(_limitations())
    data["owner_direction"] = (
        "User-directed v19 after the v18 crisis re-run. v18 rewrote the corpus's risk turns and prefixed "
        "routing sentences onto warm replies; the re-run showed the adapter reciting the overlay and the "
        "base model answering in brochure register, and the rubric was itself asking for the hotline "
        "recital. v19 removes the overlay, drops the routing requirement from the rubric, and fixes the "
        "cause in the synthetic families: every crisis action now answers the disclosure it is attached "
        "to. Acute categories keep a terse emergency escalation because a swallowed overdose or a "
        "strangulation needs a medical response today; passive ideation and means access answer the "
        "disclosure and ask for the means to leave the house, because someone who is already talking does "
        "not need to be redirected to a phone number."
    )
    return data


class _Release(v17._Release):
    """v17's build with the corrected crisis families and no rewrite overlay."""

    def summary(self) -> dict[str, Any]:
        data = super().summary()
        data["release_id"] = RELEASE_ID
        return data

    def manifest(self) -> dict[str, Any]:
        return stamp_release_identity(
            super().manifest(),
            v9_rows=len(self.v9_rows),
            v10_rows=len(self.v10_rows),
            v11_rows=len(self.v11_rows),
            v13_rows=len(self.v13_rows),
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
