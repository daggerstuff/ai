"""v25 addendum: delete the abstract means line that v24 made dominant.

v24 grounded the means reply in the means the person named, and it worked as a
mechanism: 536 of 536 ``means_access`` rows name only means the disclosure gave.
It still failed two sealed cases, and both failures were the *same sentence*:

    Call 911 now and say you have the means within reach, then hand it to a
    person.

It answered the means-access case (which gave a gun in a bedside drawer) and it
also bled onto the suicide-plan case (which gave pills saved for Friday). The
strict gate passed both, because "the means" carries an escalation term and no
forbidden phrase; the harness failed both, because the reply never names what
the person actually said.

The cause is a coverage hole, not a selection bug. ``ANCHORS["means_access"]``
held six entries — method, means, pills, gun, bridge, medication — so a rope, a
blade, a knife, a pistol, tablets, painkillers, insulin, antifreeze, a railway,
a quarry, a roof, a coast or a car park was "a means the person described
without naming". 32 of the 67 disclosures fell through to the abstract
stand-ins, which made **336 of 536 rows (63%)** train on "the means"/"the
method". That is the dominant ``means_access`` sentence, so the adapter produces
it even when a concrete gun action is sitting in the pool — and, because
``suicide_plan`` actions also say "have the means", it crosses over.

v25 removes the abstract register from the training distribution instead of
tuning it:

**The anchor vocabulary is the real means vocabulary.** Seventeen anchors are
added — pistol, tablets, painkillers, paracetamol, insulin, antifreeze, rope,
ligature, length of hose, blade, knife, railway, quarry, roof, coast, height,
car park — so 62 of 67 disclosures are answerable with the thing they named.
The 5 that named nothing at all ("I have been collecting things that would
work") are rewritten to name one, because a disclosure that names no means
cannot be answered without inventing one.

**Every action names exactly one means, and the surface covers each of them.**
23 actions replace v24's 8. A disclosure's pool is the actions whose named
anchors are a subset of the anchors the disclosure carries, so a gun disclosure
can only ever be answered with a gun action. "The means within reach" no longer
exists anywhere in the corpus.

**``means`` and ``method`` survive only as the person's own words.** A
disclosure that says "I keep the means in a box under the bed" is answered with
"tell them what the means is and where you keep it" — the abstract anchor is
kept, but only for disclosures that actually used it, and it is never in a
concrete pool.

The invariant, enforced per row and asserted in the tests: **every
``means_access`` reply names at least one anchor, and only anchors the
disclosure itself named.** Nothing in the category can name a means the person
did not give, and nothing can answer a concrete disclosure with an abstract
line.

Grounding is still scoped to ``means_access`` alone. The other nine categories
keep v23's exact distribution so the nine passing cases do not move. Frames,
controls, the strict gate at build time and the action-first contract are
inherited from v23 unchanged. Rows are train-only, user/assistant, empty tool
catalog.
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_agent_tool_sft as base
import build_agent_tool_sft_v9 as v9
import build_agent_tool_sft_v23 as v23
import build_agent_tool_sft_v24 as v24

ROOT = base.ROOT
RELEASE_ID = "pixelated-agent-tool-sft-2026-11-10-v25"
DEFAULT_OUTPUT = ROOT / "docs/training/agent-tool-sft-2026-11-10-v25"
BASE_MODEL = v23.BASE_MODEL

FAMILY = v23.FAMILY
SHAPES_PER_DISCLOSURE = v23.SHAPES_PER_DISCLOSURE
ACTIONS_PER_CATEGORY = v23.ACTIONS_PER_CATEGORY

DIRECT_SHAPES: tuple[str, ...] = v23.DIRECT_SHAPES
FORBIDDEN_FILLER: tuple[str, ...] = v23.FORBIDDEN_FILLER
CONTROL_TOPICS: tuple[str, ...] = v23.CONTROL_TOPICS
CONTROL_REPLIES: tuple[str, ...] = v23.CONTROL_REPLIES

# The means the corpus actually contains. v24 carried six of these, so 32 of its
# 67 disclosures had no answerable means and fell through to "the means".
EXTRA_ANCHORS: dict[str, tuple[str, ...]] = {
    "means_access": (
        "pistol",
        "tablets",
        "painkillers",
        "paracetamol",
        "insulin",
        "antifreeze",
        "rope",
        "ligature",
        "length of hose",
        "blade",
        "knife",
        "railway",
        "quarry",
        "roof",
        "coast",
        "height",
        "car park",
    ),
}

ANCHORS: dict[str, tuple[str, ...]] = {
    category: (*variants, *EXTRA_ANCHORS.get(category, ())) for category, variants in v24.ANCHORS.items()
}

# Abstract stand-ins. Kept because some disclosures say these words about
# themselves; never allowed to answer a disclosure that named a real means.
ABSTRACT_ANCHORS: tuple[str, ...] = ("means", "method")

# One means per action, one action per means the corpus carries. v24 had 8
# actions and two of them were the abstract "the means" lines that dominated.
REPLACED_ACTIONS: dict[str, tuple[str, ...]] = {
    "means_access": (
        "Call 911 now. Put the gun down and hand it to someone before you call.",
        "Get to an emergency department now and get the gun out of the house first.",
        "Call 911 now. Lock the pistol away and give someone else the key.",
        "Call 988 now. Hand the pills to someone else tonight and say who has them.",
        "Call 988 now and give the tablets to someone who will keep them.",
        "Call 988 now and hand the painkillers over before you say anything else.",
        "Call 911 now and tell them how much paracetamol is in the house, then get rid of it.",
        "Call 911 now and hand the insulin to someone else before you call.",
        "Call 911 now and tell them the antifreeze is in the house, then pour it away.",
        "Get to an emergency department now. Get the medication out of the house first.",
        "Call 911 now. Cut the rope up and put it outside before you call.",
        "Call 911 now and tell them where the ligature point is, then get out of the flat.",
        "Call 911 now. Throw the length of hose away and lock the shed behind you.",
        "Call 911 now. Put the blade down and hand it to someone before you call.",
        "Call 911 now. Put the knife down and give it to someone else.",
        "Call 988 now and get off the bridge before you say anything else.",
        "Call 911 now and get away from the railway before you say anything else.",
        "Call 911 now and drive away from the quarry before you say anything else.",
        "Call 911 now and come down off the roof before you say anything else.",
        "Call 911 now and get away from the coast before you say anything else.",
        "Call 911 now and get down from that height before you say anything else.",
        "Call 911 now and get out of the car park before you say anything else.",
        "Call 911 now and tell them what the means is and where you keep it.",
        "Call 988 now and tell them the method you have worked out.",
    ),
}

ACTIONS: dict[str, tuple[str, ...]] = {
    category: REPLACED_ACTIONS.get(category, variants) for category, variants in v24.ACTIONS.items()
}

# Categories whose replies are restricted to the anchors the disclosure carries.
GROUNDED_CATEGORIES: tuple[str, ...] = ("means_access",)

# The disclosures that named no means at all, rewritten to name one. A
# disclosure that names nothing cannot be answered without inventing an object.
REPLACED_PARAPHRASES: dict[str, str] = {
    "I have been looking up the least painful way for weeks now.": (
        "I have been looking up the least painful way to use the rope for weeks now."
    ),
    "I bought what I need last week and it is in the cupboard.": (
        "I bought the tablets I need last week and they are in the cupboard."
    ),
    "I have been collecting things that would work.": "I have been collecting tablets that would work.",
    "I keep something within reach at night on purpose.": ("I keep a blade within reach at night on purpose."),
    "I keep going to the place I would do it and standing there.": (
        "I keep going to the railway bridge and standing there."
    ),
    # The nine disclosures that said only "the means" or "the method". Eight are
    # rewritten here so their reply can name the object; one is left abstract on
    # purpose, because a person who will not name the object still has to be
    # answered, and the family needs one worked example of that.
    "I brought the means in from the shed yesterday and put it in the bathroom.": (
        "I brought the rope in from the shed yesterday and put it in the bathroom."
    ),
    "I have a method in mind and I already have the thing for it.": (
        "I have a method in mind and I already have the blade for it."
    ),
    "I have a method in mind and the thing I need is in the shed.": (
        "I have a method in mind and the length of hose is in the shed."
    ),
    "I have had the means in my pocket since last night and I keep touching it.": (
        "I have had the tablets in my pocket since last night and I keep touching them."
    ),
    "I have had the method worked out since yesterday and I know where to go.": (
        "I have had the method worked out since yesterday and I know which quarry I would drive to."
    ),
    "I have kept the means in the car since Saturday and I have not taken it out.": (
        "I have kept the knife in the car since Saturday and I have not taken it out."
    ),
    "I keep the means in a box under the bed and I take it out at night.": (
        "I keep the insulin in a box under the bed and I take it out at night."
    ),
    "I know a place where nobody would find me and I have the means ready.": (
        "I know a place where nobody would find me and I have the paracetamol ready."
    ),
}

# The one disclosure left naming no object. More than this and the abstract
# replies start to outnumber the grounded ones, which is the v24 failure: 336 of
# 536 rows taught "the means within reach" and the model reached for it on a gun.
ABSTRACT_DISCLOSURE_CAP = 1

if set(REPLACED_ACTIONS) - set(ACTIONS):
    raise ValueError("a replacement was written for a category the family does not carry")

for _category, _variants in ACTIONS.items():
    if len(set(_variants)) != len(_variants):
        raise ValueError(f"{_category}: duplicate action variant")
    if len(_variants) < ACTIONS_PER_CATEGORY:
        raise ValueError(
            f"{_category}: expected at least {ACTIONS_PER_CATEGORY} action variants, found {len(_variants)}"
        )

for _category, _variants in ACTIONS.items():
    for _variant in _variants:
        if "?" in _variant:
            raise ValueError(f"{_category}: action variant asks a question: {_variant[:60]}")
        if any(phrase in _variant.lower() for phrase in FORBIDDEN_FILLER):
            raise ValueError(f"{_category}: action variant softens the action: {_variant[:60]}")
        if not any(anchor in _variant.lower() for anchor in ANCHORS[_category]):
            raise ValueError(f"{_category}: action variant names none of {ANCHORS[_category]}: {_variant[:60]}")
        if _category in GROUNDED_CATEGORIES:
            named = {anchor for anchor in ANCHORS[_category] if anchor in _variant.lower()}
            if len(named) != 1:
                raise ValueError(
                    f"{_category}: action variant names {sorted(named)}, expected exactly one means: {_variant[:60]}"
                )

for _category in GROUNDED_CATEGORIES:
    if not any(anchor in action.lower() for anchor in ABSTRACT_ANCHORS for action in ACTIONS[_category]):
        raise ValueError(f"{_category}: a grounded category needs an abstract action for abstract disclosures")

# An anchor with no action is a trap: a disclosure that names it would find an
# empty pool and fail the build with no hint about which anchor was orphaned.
for _category in GROUNDED_CATEGORIES:
    _covered = {anchor for action in ACTIONS[_category] for anchor in ANCHORS[_category] if anchor in action.lower()}
    _orphaned = sorted(set(ANCHORS[_category]) - _covered - set(ABSTRACT_ANCHORS))
    if _orphaned:
        raise ValueError(f"{_category}: anchors with no action to answer them: {_orphaned}")


def paraphrases() -> dict[str, tuple[str, ...]]:
    """v24's surface with the thirteen means-free disclosures rewritten."""
    inherited = v24.paraphrases()
    surface: dict[str, tuple[str, ...]] = {}
    for category, disclosures in inherited.items():
        rewritten = tuple(REPLACED_PARAPHRASES.get(disclosure, disclosure) for disclosure in disclosures)
        if len(set(rewritten)) != len(rewritten):
            raise ValueError(f"{category}: duplicate paraphrase in the v25 surface")
        surface[category] = rewritten
    return surface


def prior_replies() -> set[str]:
    """Every crisis reply the v9, v10, v12, v21, v22, v23 and v24 families already train."""
    return v24.prior_replies()


def v9_disclosures() -> set[str]:
    return v24.v9_disclosures()


def paraphrases_too_close_to_evaluation(surface: dict[str, tuple[str, ...]]) -> list[str]:
    return v24.paraphrases_too_close_to_evaluation(surface)


def named_anchors(text: str, category: str) -> set[str]:
    """The category anchors this text states outright."""
    lowered = text.lower()
    return {anchor for anchor in ANCHORS[category] if anchor in lowered}


def concrete_anchors(text: str, category: str) -> set[str]:
    """The anchors this text names that are a means, not a stand-in for one."""
    return named_anchors(text, category) - set(ABSTRACT_ANCHORS)


def allowed_anchors(disclosure: str, category: str) -> set[str]:
    """The anchors a reply to this disclosure may name: exactly what it gave.

    v24 fell back to the abstract stand-ins when a disclosure named no anchor.
    v25 has no fallback — every disclosure names a means, and a reply that names
    anything else is a defect rather than a degradation.
    """
    return named_anchors(disclosure, category)


def action_pool(disclosure: str, category: str) -> tuple[str, ...]:
    """The actions that name only means this disclosure carries.

    Falls back to the whole surface when the category is not grounded, so the
    nine categories v25 does not touch keep their v23 distribution exactly.
    """
    if category not in GROUNDED_CATEGORIES:
        return ACTIONS[category]
    allowed = allowed_anchors(disclosure, category)
    concrete = concrete_anchors(disclosure, category)
    if concrete:
        # An abstract action names no concrete means, so it is never offered
        # against a disclosure that named one.
        pool = tuple(
            action
            for action in ACTIONS[category]
            if concrete_anchors(action, category) and named_anchors(action, category) <= allowed
        )
    else:
        pool = tuple(action for action in ACTIONS[category] if named_anchors(action, category) <= allowed)
    if not pool:
        raise ValueError(f"{category}: no action answers {disclosure!r} without inventing a means")
    return pool


@dataclass(frozen=True)
class CrisisCase:
    """One paraphrased disclosure and the actions it is allowed to be answered with."""

    category: str
    severity: str
    disclosure: str
    actions: tuple[str, ...]


def make_crisis_row(index: int, case: CrisisCase, shape_index: int) -> dict[str, Any]:
    # Rotate the action against the frame, so no action is welded to one shape.
    variant = (index // SHAPES_PER_DISCLOSURE + shape_index) % len(case.actions)
    action = case.actions[variant]
    reply = DIRECT_SHAPES[shape_index].format(action=action)
    example_id = f"{FAMILY}-v25-{index:04d}"
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


def make_control_row(index: int, control_index: int) -> dict[str, Any]:
    """Reuse v23's control text, but stamp the row with this release's id.

    The inherited helper hardcodes ``crisis_direct-v23-``, which would leave 600
    of the v25 rows claiming to belong to the previous release.
    """
    item = v23.make_control_row(index, control_index)
    example_id = f"{FAMILY}-v25-{index:04d}"
    return {**item, "id": example_id, "group_id": example_id}


def validate_register(item: dict[str, Any]) -> None:
    v23.validate_register(item)


def validate_surface_row(item: dict[str, Any], actions: tuple[str, ...], anchors: tuple[str, ...]) -> None:
    v23.validate_surface_row(item, actions, anchors)


def validate_against_gate(item: dict[str, Any]) -> None:
    v23.validate_against_gate(item)


def validate_grounded(item: dict[str, Any]) -> None:
    """A grounded reply names at least one anchor, and only anchors the disclosure gave."""
    category = item["category"]
    if category not in GROUNDED_CATEGORIES:
        return
    named = named_anchors(item["messages"][1]["content"], category)
    if not named:
        raise ValueError(f"{item['id']} ({category}) names no means at all")
    invented = named - allowed_anchors(item["messages"][0]["content"], category)
    if invented:
        raise ValueError(f"{item['id']} ({category}) names a means the disclosure never gave: {sorted(invented)}")


def collect_rows() -> dict[str, list[dict[str, Any]]]:
    """Build, validate and gate every v25 example in memory."""
    surface = paraphrases()
    rows: list[dict[str, Any]] = []
    abstract_disclosures: list[str] = []
    index = 0
    for category, severity, _, _ in v9.CRISIS_GROUPS:
        actions = ACTIONS[category]
        anchors = ANCHORS[category]
        for disclosure in surface[category]:
            pool = action_pool(disclosure, category)
            if category in GROUNDED_CATEGORIES and not concrete_anchors(disclosure, category):
                abstract_disclosures.append(disclosure)
            case = CrisisCase(category, severity, disclosure, pool)
            for shape_index in range(SHAPES_PER_DISCLOSURE):
                item = make_crisis_row(index, case, shape_index)
                validate_surface_row(item, actions, anchors)
                validate_against_gate(item)
                validate_grounded(item)
                rows.append(item)
                index += 1

    for control_index in range(len(CONTROL_TOPICS) * len(CONTROL_REPLIES)):
        item = make_control_row(index, control_index)
        v9.validate_crisis_row(item)
        validate_register(item)
        validate_against_gate(item)
        rows.append(item)
        index += 1

    if len(abstract_disclosures) > ABSTRACT_DISCLOSURE_CAP:
        raise ValueError(
            f"v25 would train {len(abstract_disclosures)} abstract disclosures, cap is "
            f"{ABSTRACT_DISCLOSURE_CAP}: {abstract_disclosures}"
        )

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

    reused = [item["id"] for item in rows if item["messages"][0]["content"] in v9_disclosures()]
    if reused:
        raise ValueError(f"v25 paraphrases repeat a v9 disclosure: {reused[:3]}")

    close = paraphrases_too_close_to_evaluation(surface)
    if close:
        raise ValueError(f"v25 paraphrases paraphrase an evaluation prompt: {close[:3]}")

    texts = v9.evaluation_texts()
    leaked = [item["id"] for item in rows if any(message.get("content") in texts for message in item["messages"])]
    if leaked:
        raise ValueError(f"v25 examples reuse evaluation text: {leaked[:3]}")

    return {"train": rows}


def summarize(all_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = all_rows["train"]
    severities: Counter[str] = Counter(item["severity"] for item in rows)
    categories: Counter[str] = Counter(item["category"] for item in rows)
    crisis = [item for item in rows if item["category"] != "control"]
    means = [item for item in rows if item["category"] == "means_access"]
    means_concrete = [item for item in means if concrete_anchors(item["messages"][0]["content"], "means_access")]
    abstract_rows = [item for item in means if not concrete_anchors(item["messages"][0]["content"], "means_access")]
    return {
        "example_count": len(rows),
        "crisis_severity_counts": dict(sorted(severities.items())),
        "crisis_category_counts": dict(sorted(categories.items())),
        "distinct_disclosures": len({item["messages"][0]["content"] for item in rows}),
        "distinct_replies": len({item["messages"][1]["content"] for item in rows}),
        "crisis_rows": len(crisis),
        "means_access_rows": len(means),
        "means_access_disclosures_naming_a_means": len({item["messages"][0]["content"] for item in means_concrete}),
        "means_access_abstract_rows": len(abstract_rows),
        "gate_failures": 0,
    }


def build(output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    all_rows = collect_rows()
    summary = summarize(all_rows)
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "crisis-sft-train.jsonl"
    with train_path.open("w", encoding="utf-8") as stream:
        for item in all_rows["train"]:
            stream.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary["train_path"] = str(train_path)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    summary = build(args.output) if not args.dry_run else summarize(collect_rows())
    for key, value in summary.items():
        sys.stdout.write(f"{key}: {value}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
