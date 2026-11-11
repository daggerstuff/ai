"""Build the v17 G04 research release on top of the v16 lineage.

The v16 paired eval closed every gap v15 left except one: on the two crisis
cases it was meant to fix, the adapter routed correctly but the answer opened
with a warm reflection — "That's a lot to carry, and I'm glad you told me" —
before it named any action. The routing lexicon was present, the register was
therapy-shaped, and a person describing a wound that needs closing was asked
how they felt about it first.

v17 keeps the whole v16 pipeline — the candidate-level welded-argument and
reasoning-preamble repairs, the v9, v10 and v11 addenda, the crisis purge, the
held-out duplicate-target purge, the split recipe, the seed — and replaces one
family: the 672-row v12 ``crisis_surface`` addendum is dropped and the 708-row
v13 ``crisis_direct`` addendum is appended in its place. v13 keeps v12's
breadth (twenty paraphrases per category, ten categories, three shapes) but
rewrites every action as an imperative and rejects any reply that softens,
reacts, or asks a question. It also adds a surface v12 never covered: injury
paired with a refusal to stop, which is the exact shape both v16 eval failures
took.

The family is replaced rather than added because two registers teaching the
same category is the 50/50 conflict that made the v7 welded tool arguments and
the v9 bare literals cancel each other out. One register, one target.

Everything downstream of the addenda is inherited, so the v17 candidate is the
v16 candidate minus the v12 rows plus the v13 rows at the end, and the
validation and test payloads stay byte-identical to v16's.

Usage::

    PYTHONPATH=/home/vivi/pixelated python -m training.build_v17_release --dry-run
    PYTHONPATH=/home/vivi/pixelated python -m training.build_v17_release
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from training import (
    build_agent_tool_sft_v13,
    build_v14_release,
    build_v15_release,
    build_v16_release,
)

# Bound to module-level names so callers that import this module can reach them;
# `from training import x as y` is not a re-export under --strict.
v13 = build_agent_tool_sft_v13
v14 = build_v14_release
v15 = build_v15_release
v16 = build_v16_release

ROOT = v16.ROOT

OUTPUT = ROOT / "ai/data/curated/recovery/gold-20261017/wandb-full-corpus-current-master-20261109-v17"
MANIFEST = ROOT / "docs/training/execution-2026-11-09/wandb-full-corpus-current-master-20261109-v17.json"

RELEASE_ID = "wandb-full-corpus-current-master-20261109-v17"
SHUFFLE_SEED = 20261109
SPLITS = v16.SPLITS
CREATED_AT_UTC = "2026-11-09T00:00:00Z"

V13_SOURCE_LABEL = "crisis_direct_v13"
EXPECTED_V13_TRAIN_ROWS = 708

# v16 recorded a limitation about the v12 crisis-surface family. v17 replaces
# that family outright, so the sentence is dropped rather than inherited.
_STALE_V12_LIMITATION = "The v12 crisis-surface rows are appended to train only"

LIMITATIONS_V17: tuple[str, ...] = (
    (
        "The v13 crisis-direct rows replace v12's crisis-surface rows rather than adding to them, so the "
        "register the release teaches for a crisis disclosure is the direct one and only the direct one. A "
        "disclosure phrased outside the twenty paraphrases of its category can still fall back to the "
        "reflective answer the therapy corpus teaches, and the release carries no second family to catch it."
    ),
    (
        "The v13 replies are hand-written in a single imperative register and are not reviewed by a clinician. "
        "They route to emergency care and nothing after it: there is no safety-plan, follow-up or "
        "de-escalation content, so the family teaches triage and leaves everything downstream of triage to "
        "whatever the corpus already contained."
    ),
    (
        "The v13 family is appended to train only, and validation and test carry no crisis-direct rows. An "
        "adapter that memorises the 224 disclosures instead of generalising the routing behaviour would not be "
        "caught by the held-out splits; only the paired crisis cases probe that."
    ),
)


def _limitations() -> tuple[str, ...]:
    """The inherited limitations minus the v12 one v17 replaces, plus the v17 ones."""
    inherited = v16._limitations()
    kept = tuple(text for text in inherited if not text.startswith(_STALE_V12_LIMITATION))
    if len(kept) != len(inherited) - 1:
        raise ValueError("the v12 crisis-surface limitation is no longer in the inherited list")
    return (*kept, *LIMITATIONS_V17)


def v13_candidate_rows() -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """The v13 train rows as (candidate row, projected payload messages).

    These are chat rows, not agent rows: v13 carries an empty tool catalog, so
    ``is_agent_tool`` stays false and the payload keeps the plain ``messages``
    body.
    """
    rows = v13.collect_rows()["train"]
    if len(rows) != EXPECTED_V13_TRAIN_ROWS:
        raise ValueError(f"expected {EXPECTED_V13_TRAIN_ROWS} v13 train rows, found {len(rows)}")
    out: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for item in rows:
        candidate = {"messages": item["messages"], "source": V13_SOURCE_LABEL}
        if v14.is_agent_tool(candidate):
            raise ValueError(f"{item['id']}: v13 candidate rows must not carry a tool catalog")
        # The rows are hand-written, so nothing should redact. Projecting against a
        # throwaway counter keeps the release-wide redaction counts honest: if a
        # v13 row ever trips a pattern, this fails instead of silently hiding it.
        scratch: Counter[str] = Counter()
        payload = v14.project_chat(item["messages"], scratch)
        if payload is None:
            raise ValueError(f"{item['id']}: v13 row projects to no assistant target")
        if scratch and any(scratch.values()):
            raise ValueError(f"{item['id']}: v13 row redacted {dict(scratch)}; synthetic rows must be clean")
        out.append((candidate, payload))
    return out


def stamp_release_identity(
    data: dict[str, Any],
    *,
    v9_rows: int,
    v10_rows: int,
    v11_rows: int,
    v13_rows: int,
) -> dict[str, Any]:
    """Rewrite the addenda-dependent manifest fields for the v17 release."""
    appended_by_source = {
        v14.V9_SOURCE_LABEL: v9_rows,
        v15.V10_SOURCE_LABEL: v10_rows,
        v16.V11_SOURCE_LABEL: v11_rows,
        V13_SOURCE_LABEL: v13_rows,
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
        f"2,807 v13 agent_tool rows plus {v9_rows} v9 rows plus {v10_rows} v10 rows plus {v11_rows} v11 rows "
        "include assistant tool_calls, role:tool turns, and a retained system message that defines tool-use "
        "context."
    )
    data["limitations"] = list(_limitations())
    data["owner_direction"] = (
        "User-directed v17 after the v16 paired eval. v16 closed the v15 memory-value and crisis-breadth gaps "
        "and beat base, v14 and v15 on aggregate, but the two crisis cases it recovered were answered in the "
        "wrong register: the reply opened with a reflection about how hard the disclosure was before naming any "
        "action, which is exactly what a person in an emergency does not need first. v17 replaces the 672-row "
        "v12 crisis_surface family with the 708-row v13 crisis_direct family — same twenty-paraphrase breadth "
        "per category, but every action rewritten as an imperative, every coddling opener rejected, every "
        "question rejected, and a new surface for injury paired with a refusal to stop, which is the shape both "
        "v16 crisis failures took. The family replaces v12 rather than joining it so the release teaches one "
        "register per category instead of two that cancel."
    )
    return data


class _Release(v16._Release):
    """v16's build, with the v13 direct-register family in place of v12."""

    def __init__(self, indexes: v14._Indexes, output: Path, dry_run: bool) -> None:
        super().__init__(indexes, output, dry_run)
        self.v13_rows: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []

    def append_addenda(self) -> None:
        """v15's v9 and v10 addenda, then v11, then v13 — v13 replaces v12."""
        v15._Release.append_addenda(self)
        self.v11_rows = v16.v11_candidate_rows()
        for candidate, payload in self.v11_rows:
            self.append_agent_row(v16.V11_SOURCE_LABEL, candidate, payload)
        self.v13_rows = v13_candidate_rows()
        for candidate, payload in self.v13_rows:
            self.append_chat_row(V13_SOURCE_LABEL, candidate, payload)

    def verify_totals(self) -> None:
        """v16's totals, with the v13 family counted in place of v12."""
        if self.purged != v15.EXPECTED_PURGED_ROWS:
            raise ValueError(f"expected {v15.EXPECTED_PURGED_ROWS} purged rows, found {self.purged}")
        if dict(self.repair_counts) != v15.EXPECTED_REPAIR_COUNTS:
            raise ValueError(f"repair population drifted: {dict(self.repair_counts)}")
        appended = len(self.v9_rows) + len(self.v10_rows) + len(self.v11_rows) + len(self.v13_rows)
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
        data.pop("appended_v12_rows", None)
        data["appended_v13_rows"] = len(self.v13_rows)
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
