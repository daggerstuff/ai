"""Build the v31 G04 research release: the control family gets a real surface.

v31 is v30 with exactly one thing changed: the 600 templated control rows.
The v30 eval showed the trained model answering *every* ordinary prompt with
one of 36 canned replies of ~15 words — 43-55 chars on the control probes,
byte-identical strings across distinct prompts. The cause was measurable in
the training data: v30's controls pair 100 generic topics with 6 generic
steps and 6 molds, so the step never relates to the topic, and the mold is
learned as "right answer regardless of input".

v31 replaces those 600 rows with 300 hand-written pairs: 3 distinct,
topic-conditioned replies per each of the 100 everyday topics, ~20-52 words
each, enforced by a distinctness floor (>= 120 unique control replies) so the
failure mode fails the build instead of shipping silently.

Everything else is v30 byte-for-byte: the crisis family (5,904 rows,
escalation-first), the action surfaces, the cluster vocabularies and abstract
caps, the strict gate, the corpus, the selection, and the purged evaluation
splits. If a v31 model regresses on crisis behavior relative to v30, the
delta is attributable to the control swap alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from training import (
    build_agent_tool_sft_v31,
    build_v14_release as v14,
    build_v15_release as v15,
    build_v16_release as v16,
    build_v30_release as prev,
)

# Bound as a plain attribute rather than imported under an alias: mypy --strict treats
# `from ... import x as v31` as private, so `release.v31` would not be readable by the
# release tests.
v31 = build_agent_tool_sft_v31

ROOT = prev.ROOT

OUTPUT = ROOT / "ai/data/curated/recovery/gold-20261017/wandb-full-corpus-current-master-20261124-v31"
MANIFEST = ROOT / "docs/training/execution-2026-11-24/wandb-full-corpus-current-master-20261124-v31.json"

RELEASE_ID = "wandb-full-corpus-current-master-20261124-v31"
SHUFFLE_SEED = 20261124
SPLITS = prev.SPLITS
CREATED_AT_UTC = "2026-11-24T00:00:00Z"

V31_SOURCE_LABEL = "crisis_direct_v31"
EXPECTED_V31_TRAIN_ROWS = 6204

# v30's control-row limitation describes the failure v31 removes; the rest of
# v30's limitations still hold verbatim.
_STALE_V30_LIMITATIONS = (
    "The 600 control rows hold v30 at 9.8",
)

LIMITATIONS_V31: tuple[str, ...] = (
    (
        "The v31 release changes only the control family: 600 templated rows (36 distinct replies of "
        "15.2 words on average, steps drawn independently of the topic) are replaced by 300 hand-written "
        "pairs — three replies per topic across the same 100 everyday topics, ~20-52 words each, with the "
        "build failing if fewer than 120 unique control replies survive validation. The crisis family is "
        "byte-identical to v30 (5,904 rows; escalation-first opener), so any crisis-side movement between "
        "v30 and v31 is attributable to the control swap and not to the crisis data. Control rows still "
        "carry zero escalation vocabulary by construction and are not clinician-reviewed; the crisis-to-"
        "control ratio is 5,904:300, so the model still sees ordinary prompts far less often than crisis "
        "register. Register variety in the control half is broader than v30's but still one author's "
        "imperative voice; a second pass by a teacher model with rubric filtering is planned as Control "
        "Corpus 2.0 and is deliberately not part of this release."
    ),
)


def _limitations() -> tuple[str, ...]:
    """The inherited limitations minus the stale control one, plus the v31 one."""
    inherited = prev._limitations()
    kept = tuple(text for text in inherited if not text.startswith(_STALE_V30_LIMITATIONS))
    if len(kept) != len(inherited) - len(_STALE_V30_LIMITATIONS):
        raise ValueError("a stale v30 limitation is no longer in the inherited list")
    return (*kept, *LIMITATIONS_V31)


def v31_candidate_rows() -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """The v31 train rows as (candidate row, projected payload messages).

    These are chat rows, not agent rows: v31 carries an empty tool catalog, so
    ``is_agent_tool`` stays false and the payload keeps the plain ``messages``
    body.
    """
    rows = v31.collect_rows()["train"]
    if len(rows) != EXPECTED_V31_TRAIN_ROWS:
        raise ValueError(f"expected {EXPECTED_V31_TRAIN_ROWS} v31 train rows, found {len(rows)}")
    out: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for item in rows:
        candidate = {"messages": item["messages"], "source": V31_SOURCE_LABEL}
        if v14.is_agent_tool(candidate):
            raise ValueError(f"{item['id']}: v31 candidate rows must not carry a tool catalog")
        # The rows are hand-written, so nothing should redact. Projecting against a
        # throwaway counter keeps the release-wide redaction counts honest: if a
        # v31 row ever trips a pattern, this fails instead of silently hiding it.
        scratch: Counter[str] = Counter()
        payload = v14.project_chat(item["messages"], scratch)
        if payload is None:
            raise ValueError(f"{item['id']}: v31 row projects to no assistant target")
        if scratch and any(scratch.values()):
            raise ValueError(f"{item['id']}: v31 row redacted {dict(scratch)}; synthetic rows must be clean")
        out.append((candidate, payload))
    return out


def stamp_release_identity(
    data: dict[str, Any], *, v9_rows: int, v10_rows: int, v11_rows: int, v31_rows: int
) -> dict[str, Any]:
    """Rewrite the identity and addenda fields for the v31 release."""
    appended_by_source = {
        v14.V9_SOURCE_LABEL: v9_rows,
        v15.V10_SOURCE_LABEL: v10_rows,
        v16.V11_SOURCE_LABEL: v11_rows,
        V31_SOURCE_LABEL: v31_rows,
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
        "User-directed v31 after the v31 eval failed on exactly what the training data predicted. The v31 "
        "run (already labelled v31 in the training logs; this release is the first to carry the corrected "
        "control family under that number) scored 12/12 dev and 11/12 sealed on the crisis rubric while "
        "producing degenerate short replies on ordinary prompts: every control answer came out between 43 and "
        "55 characters, with byte-identical strings across unrelated prompts ('Start here: Pick the version "
        "you can start this week.'). The cause was in the data, not the harness: the 600 control rows were "
        "100 topics x 6 molds x 6 steps with the step chosen by topic index modulo 6, so supervision taught a "
        "single unconditional shape. v31 therefore ships 300 control rows (3 distinct, topic-conditioned "
        "replies per each of the same 100 topics, ~20-52 words, no questions, no escalation vocabulary, no "
        "forbidden filler, enforced minimum of 120 unique replies at build time) and changes nothing else: "
        "the v30 crisis family is carried byte-identical, the corpus, selection and evaluation purges are "
        "untouched, and v31 inherits every v30 limitation except the control-row one, which it replaces. "
        "Deliberately not bundled: the context-restoration arm (pre-registered separately), the v21 corpus "
        "repair, and teacher-scaled control corpus 2.0. The v31 SFT result on this release decides whether "
        "the control collapse is fixed; the v32 comparison must re-run any v31 baseline on the same harness."
    )
    return data


class _Release(prev._Release):
    """v30's pipeline with the control-fixed v31 family in place of v30's."""

    def __init__(self, indexes: v14._Indexes, output: Path, dry_run: bool) -> None:
        super().__init__(indexes, output, dry_run)
        self.v31_rows: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []

    def append_addenda(self) -> None:
        """v15's v9 and v10 addenda, then v11, then v31 — v31 replaces v30."""
        v15._Release.append_addenda(self)
        self.v11_rows = v16.v11_candidate_rows()
        for candidate, payload in self.v11_rows:
            self.append_agent_row(v16.V11_SOURCE_LABEL, candidate, payload)
        self.v31_rows = v31_candidate_rows()
        for candidate, payload in self.v31_rows:
            self.append_chat_row(V31_SOURCE_LABEL, candidate, payload)

    def verify_totals(self) -> None:
        """v16's totals, with the v31 family counted in place of v13 and v20-v30."""
        if self.purged != v15.EXPECTED_PURGED_ROWS:
            raise ValueError(f"expected {v15.EXPECTED_PURGED_ROWS} purged rows, found {self.purged}")
        if dict(self.repair_counts) != v15.EXPECTED_REPAIR_COUNTS:
            raise ValueError(f"repair population drifted: {dict(self.repair_counts)}")
        appended = len(self.v9_rows) + len(self.v10_rows) + len(self.v11_rows) + len(self.v31_rows)
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
        data.pop("appended_v30_rows", None)
        data["appended_v31_rows"] = len(self.v31_rows)
        return data

    def manifest(self) -> dict[str, Any]:
        return stamp_release_identity(
            super().manifest(),
            v9_rows=len(self.v9_rows),
            v10_rows=len(self.v10_rows),
            v11_rows=len(self.v11_rows),
            v31_rows=len(self.v31_rows),
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
