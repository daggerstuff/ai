#!/usr/bin/env python3
"""v10 agent-tool data: crisis replies that name the resource first.

The v14 paired evaluation moved crisis routing from 3/11 to 7/11 on the dev
split and from 5/11 to 7/11 on sealed, and every remaining failure fails the
same check: the reply is a warm reflection plus a safety question with no
resource named. The v9 family does name 988, 911 and poison control, but in
most of its eight shapes the action clause sits second, behind a frame
("What you are describing needs help today, not later. {action}"). The adapter
reproduces the frame and drops the clause.

v10 keeps v9's disclosures and action lines exactly and changes one thing:
every reply opens with the action. Eight shapes, all resource-first, so the
family teaches "name the resource" as the invariant rather than as one option
among several.

480 rows: every crisis group and disclosure v9 trains, crossed with 8 shapes,
train only, matching the v3..v9 addendum convention. No controls are added; the
60 v9 controls remain in train and cover the "do not escalate an ordinary
complaint" direction.

Usage::

    PYTHONPATH=/home/vivi/pixelated python -m training.build_agent_tool_sft_v10 --dry-run
    PYTHONPATH=/home/vivi/pixelated python -m training.build_agent_tool_sft_v10
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_agent_tool_sft as base
import build_agent_tool_sft_v9 as v9

ROOT = base.ROOT
RELEASE_ID = "pixelated-agent-tool-sft-2026-11-07-v10"
DEFAULT_OUTPUT = ROOT / "docs/training/agent-tool-sft-2026-11-07-v10"
BASE_MODEL = v9.BASE_MODEL

FAMILY = "crisis_lexicon"
SHAPES_PER_DISCLOSURE = 8

# Every shape starts with the action clause. The frame that follows varies so
# the family does not teach a single fixed script, but the resource is always
# the first thing in the reply -- that ordering is the whole point of the
# family.
RESOURCE_FIRST_SHAPES: tuple[str, ...] = (
    "{action}",
    "{action} Where are you right now?",
    "{action} I will stay with you on this call until it is done.",
    "{action} Is there someone who can be with you tonight?",
    "{action} Do not wait to see whether it passes.",
    "{action} Tell me once you have called, or tell me what is stopping you.",
    "{action} Nothing else today matters more than making that call.",
    "{action} I am not going to dress this up for you.",
)

if len(RESOURCE_FIRST_SHAPES) != SHAPES_PER_DISCLOSURE:
    raise ValueError("the shape table must stay in step with SHAPES_PER_DISCLOSURE")


def v9_replies() -> set[str]:
    """Every reply the v9 family already trains, so v10 adds no duplicate."""
    return {shape.format(action=action) for _, _, _, action in v9.CRISIS_GROUPS for shape in v9.REPLY_SHAPES}


@dataclass(frozen=True)
class CrisisCase:
    """One v9 crisis group and disclosure, paired with the action it requires."""

    category: str
    severity: str
    disclosure: str
    action: str


def make_row(index: int, case: CrisisCase, shape: str) -> dict[str, Any]:
    reply = shape.format(action=case.action)
    example_id = f"{FAMILY}-v10-{index:03d}"
    return {
        "id": example_id,
        "family": FAMILY,
        "group_id": example_id,
        "severity": case.severity,
        "category": case.category,
        "messages": [
            {"role": "user", "content": case.disclosure},
            {"role": "assistant", "content": reply},
        ],
        "tools": [],
    }


def validate_lexicon_row(item: dict[str, Any], action: str) -> None:
    """The v9 crisis contract, plus the ordering this family exists to teach."""
    v9.validate_crisis_row(item)
    reply = item["messages"][1]["content"]
    if not reply.startswith(action):
        raise ValueError(f"{item['id']}: reply does not open with the resource action")
    if reply in v9_replies():
        raise ValueError(f"{item['id']}: reply duplicates a v9 crisis reply")


def collect_rows() -> dict[str, list[dict[str, Any]]]:
    """Build and validate every v10 example in memory."""
    rows: list[dict[str, Any]] = []
    index = 0
    for category, severity, disclosures, action in v9.CRISIS_GROUPS:
        for disclosure in disclosures:
            case = CrisisCase(category, severity, disclosure, action)
            for shape in RESOURCE_FIRST_SHAPES:
                item = make_row(index, case, shape)
                validate_lexicon_row(item, case.action)
                rows.append(item)
                index += 1

    payload_hashes: set[str] = set()
    for item in rows:
        payload_line = json.dumps(
            {"messages": item["messages"], "tools": item["tools"]},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if payload_line in payload_hashes:
            raise ValueError(f"duplicate exported training payload: {item['id']}")
        payload_hashes.add(payload_line)

    texts = v9.evaluation_texts()
    leaked = [item["id"] for item in rows if any(message.get("content") in texts for message in item["messages"])]
    if leaked:
        raise ValueError(f"v10 examples reuse evaluation text: {leaked[:3]}")

    return {"train": rows}


def summarize(all_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = all_rows["train"]
    severities: Counter[str] = Counter(item["severity"] for item in rows)
    categories: Counter[str] = Counter(item["category"] for item in rows)
    return {
        "example_count": len(rows),
        "crisis_severity_counts": dict(sorted(severities.items())),
        "crisis_category_counts": dict(sorted(categories.items())),
        "distinct_disclosures": len({item["messages"][0]["content"] for item in rows}),
        "distinct_replies": len({item["messages"][1]["content"] for item in rows}),
    }


def build(output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")

    all_rows = collect_rows()
    output.mkdir(parents=True, exist_ok=True)
    files = {split: base.write_split(output, split, rows) for split, rows in all_rows.items()}
    summary = summarize(all_rows)

    manifest = {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "purpose": (
            "v15 addendum: v14 crisis replies reflect the disclosure and ask a safety question but "
            "omit the resource, failing the response_contains_any check on 4 of 11 dev cases and 3 of "
            "11 sealed cases; every v10 reply opens with the resource action"
        ),
        "source": "first-party deterministic synthetic trajectories; no external or clinical records",
        "base_model": BASE_MODEL,
        "example_count": summary["example_count"],
        "scenario_families": {FAMILY: summary["example_count"]},
        "crisis_severity_counts": summary["crisis_severity_counts"],
        "crisis_category_counts": summary["crisis_category_counts"],
        "distinct_disclosures": summary["distinct_disclosures"],
        "distinct_replies": summary["distinct_replies"],
        "split_rule": "train only, matching the v3..v9 addendum convention",
        "splits": {name: files[name] for name in files},
        "tools": {},
        "design_notes": {
            "reused_from_v9": (
                "crisis groups, disclosures and action lines are imported from the v9 builder, so the "
                "two families differ in reply ordering only"
            ),
            "resource_first": (
                "every reply begins with the action clause; v9 put it first in 3 of 8 shapes and the "
                "adapter dropped it in the other 5"
            ),
            "crisis_shape": "user then assistant, no system message, matching the arc corpus",
            "crisis_rubric": (
                "acute disclosures name a medical or emergency escalation; non-acute disclosures name a crisis referral"
            ),
            "no_controls": "the 60 v9 control rows stay in train and cover the non-escalation direction",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        summary = summarize(collect_rows())
        for key, value in summary.items():
            sys.stdout.write(f"{key}: {value}\n")
        return 0
    manifest = build(args.output)
    sys.stdout.write(f"wrote {manifest['example_count']} examples to {args.output}\n")
    sys.stdout.write(f"  severities: {manifest['crisis_severity_counts']}\n")
    sys.stdout.write(f"  categories: {manifest['crisis_category_counts']}\n")
    for name, info in manifest["splits"].items():
        sys.stdout.write(f"  {name}: {info['row_count']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
