"""v23 addendum: make the crisis family survive the strict gate it is graded by.

v22 built the family at a mass the corpus could not outvote — 5,728 rows, 6.8%
of an 84k train, eight frames per disclosure and eight actions per category.
That fixed the ratio. It did not fix the surface, and the hardened gate found
two things the row count cannot hide.

**Two categories carry no escalation at all.** Measured over the 6,184 exported
v22 rows, 816 crisis rows fail the gate's first check — ``means_access`` (408)
and ``passive_ideation`` (408), every disclosure against every frame. Both are
``engaged`` severity in the v9 group table, so ``v9.validate_crisis_row`` never
asked them to escalate and no builder since v9 has either. The result is that
the family teaches a person with a method and the means in the house to be told
nothing about calling anyone. The gate fails every one of those rows, and it is
right to.

**Three medical_collapse actions invent clinical detail.** "Fainting and an
irregular heartbeat after purging needs a monitor tonight", "is how
electrolytes crash", "not a rest problem" — the person never said any of it.
The judge fails a reply that names details the person never mentioned, and the
wording came from the builder, not the model, so no amount of training fixes it.
Every medical_collapse action is recast here without a single clinical noun the
disclosure does not already carry.

The changes are therefore replacements, not additions:

* **All eight ``passive_ideation`` and ``means_access`` actions are replaced**
  with escalation-bearing lines that keep the action-first contract, name an
  anchor from the category, and add the means-restriction step where the
  category calls for one.
* **All eight ``medical_collapse`` actions are replaced** with detail-free
  recasts that keep the anchors (blacked out, bathroom, purge, five days,
  faint) and drop every invented finding.
* **24 new disclosures** — self_harm_active +8 in the "deeper than usual and I
  do not want to stop" register the suite keeps failing, passive_ideation +10,
  medical_collapse +6 — so the replaced actions are not welded to 48 topics.
* **The strict gate runs at build time.** Every generated crisis row is scored
  with ``strict_crisis_gate.structural_violations`` and the build raises if any
  row would fail. A regression in the surface cannot reach the training set
  again.
* **Actions are grounded per disclosure.** The rotation v22 grew from hands
  every disclosure the whole category table, so a reply could pair a faint with
  a blackout the person never reported. An action now only answers a disclosure
  whose anchors it shares, matched as whole words over a vocabulary that
  carries the inflected forms the recast surface actually uses ("fainted",
  "purging", "not waking up"); the stem-only vocabulary matched nothing as
  whole words, which left the two-sign actions answerable by every disclosure.
  A disclosure no specific action can answer without naming something it never
  gave falls to the category's generic escalation instead of the full table:
  the old fallback is what taught a rope disclosure to say "the pills" and a
  blackout disclosure to invent food.

Register rules, frames and control corpus are inherited from v22 unchanged;
the anchor vocabulary gains the inflected forms the recast surfaces use. Rows
are train-only, user/assistant, empty tool catalog.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_agent_tool_sft as base
import build_agent_tool_sft_v9 as v9
import build_agent_tool_sft_v12 as v12
import build_agent_tool_sft_v22 as v22
import strict_crisis_gate as gate

ROOT = base.ROOT
RELEASE_ID = "pixelated-agent-tool-sft-2026-11-10-v23"
DEFAULT_OUTPUT = ROOT / "docs/training/agent-tool-sft-2026-11-10-v23"
BASE_MODEL = v22.BASE_MODEL

FAMILY = v22.FAMILY
SHAPES_PER_DISCLOSURE = v22.SHAPES_PER_DISCLOSURE
ACTIONS_PER_CATEGORY = v22.ACTIONS_PER_CATEGORY

DIRECT_SHAPES: tuple[str, ...] = v22.DIRECT_SHAPES
FORBIDDEN_FILLER: tuple[str, ...] = v22.FORBIDDEN_FILLER
CONTROL_TOPICS: tuple[str, ...] = v22.CONTROL_TOPICS
CONTROL_REPLIES: tuple[str, ...] = v22.CONTROL_REPLIES

# v22's vocabulary is a stem vocabulary ("eat", "purge", "faint"): the
# inflected forms the recast actions and the disclosures actually use were
# reachable only as substrings of those stems, which the whole-word matcher
# correctly refuses. The two recast categories get the forms their surface
# uses; every other category keeps v22's list untouched.
ANCHOR_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "medical_collapse": (
        "fainted",
        "fainting",
        "passed out",
        "collapsed",
        "black out",
        "purged",
        "purging",
        "eating",
        "not eaten",
        "not eating",
        "cannot eat",
        "eating almost nothing",
        "keeping food down",
        "cannot keep food down",
        "kept anything down",
        "keep a meal down",
        "kept a meal down",
        "keep water down",
        "kept water down",
        "keeping water down",
        "nothing to drink",
        "anything to drink",
        "heart",
        "flutter",
        "fluttering",
        "flutters",
        "blood sugar",
        "laxatives",
        "restricting",
        "vomiting",
        "threw up",
    ),
    "passive_ideation": ("decided", "not waking up"),
}
ANCHORS: dict[str, tuple[str, ...]] = {
    category: (*anchors, *ANCHOR_EXTENSIONS.get(category, ()))
    for category, anchors in v22.ANCHORS.items()
}

# Declared inflections: the whole-word matcher reports the spelling a sentence
# used, so the inflected forms fold onto one representative before the subset
# rule reads them - a disclosure that says "fainted" is answered by an action
# that says "passed out", and neither is read as nothing.
INFLECTION_FAMILIES: tuple[tuple[str, ...], ...] = (
    ("faint", "fainted", "fainting", "passed out", "collapsed"),
    ("purge", "purged", "purging"),
    ("eat", "eating", "not eaten", "not eating", "cannot eat", "eating almost nothing"),
    (
        "keep food down",
        "keeping food down",
        "cannot keep food down",
        "kept anything down",
        "keep a meal down",
        "kept a meal down",
    ),
    (
        "keep water down",
        "kept water down",
        "keeping water down",
        "cannot keep water down",
        "nothing to drink",
        "anything to drink",
    ),
    ("flutter", "fluttering", "flutters"),
    ("not wake up", "not waking up"),
)
_ANCHOR_REPRESENTATIVE = {form: family[0] for family in INFLECTION_FAMILIES for form in family}

# Whole-category replacements. Each tuple is the complete action surface for
# that category, not an addendum: the v22 lines for these three categories are
# either silent about escalation or state a finding the person never gave.
REPLACED_ACTIONS: dict[str, tuple[str, ...]] = {
    "medical_collapse": (
        "Call 911 now. You blacked out and that needs a doctor tonight, not a lie down.",
        "Call 911 now and tell them about the purge. Purging like this needs a doctor tonight.",
        "Get to an emergency department now. You passed out, and that needs a doctor tonight.",
        "Get to an emergency department now. Five days of this has gone past what you can manage at home.",
        "Get to an emergency department now. Not keeping food down needs fluids and a doctor tonight.",
        "Go to an emergency department now and tell them how little you have been eating. That needs a doctor tonight.",
        "Call 911 now. A heart that will not hold you up needs a doctor tonight.",
        "Get to an emergency department now. Not keeping water down needs fluids tonight, not another glass.",
    ),
    "passive_ideation": (
        "Call 988 now and say it out loud: you think people would be better off without you.",
        "Call your local crisis line tonight and say you have been thinking about not being around.",
        "Get to urgent care or call 988 and say you do not see a reason to live.",
        "Call 988 now. Thinking you are a burden is what that line answers, tonight.",
        "Call 911 if you have decided you will not wake up tomorrow, and say exactly that.",
        "Get to an emergency department tonight and say you feel you should not exist.",
        "Call 988 now and say you have been hoping you will not wake up, then stay on the line.",
        "Call an ambulance or 988 and say you have been thinking the world is better off without you.",
    ),
    "means_access": (
        "Call 988 now and say the pills are in the house, then hand them to someone else tonight.",
        "Call 911 now and say the gun is where you can reach it. Put it down and step away before you call.",
        "Get to an emergency department now and say the medication is within reach. Leave it with someone before you go.",
        "Call 988 now and tell them the bridge is what you have been thinking about, then stay on the line with them.",
        "Call 911 now and say you can reach the pills, and hand the bottle to someone else.",
        "Get someone to take the gun and the medication out of the house tonight, then call 988.",
        "Call 988 now and tell them you have a method in mind and the thing you need for it.",
        "Get to an emergency department now and say you have a plan and the means to carry it out.",
    ),
}

ACTIONS: dict[str, tuple[str, ...]] = {
    category: REPLACED_ACTIONS.get(category, variants) for category, variants in v22.ACTIONS.items()
}

if set(REPLACED_ACTIONS) - set(ACTIONS):
    raise ValueError("a replacement was written for a category the family does not carry")

for _category, _variants in ACTIONS.items():
    if len(_variants) != ACTIONS_PER_CATEGORY:
        raise ValueError(f"{_category}: expected {ACTIONS_PER_CATEGORY} action variants, found {len(_variants)}")
    if len(set(_variants)) != len(_variants):
        raise ValueError(f"{_category}: duplicate action variant")
    for _variant in _variants:
        if "?" in _variant:
            raise ValueError(f"{_category}: action variant asks a question: {_variant[:60]}")
        if any(phrase in _variant.lower() for phrase in FORBIDDEN_FILLER):
            raise ValueError(f"{_category}: action variant softens the action: {_variant[:60]}")
        if not any(anchor in _variant.lower() for anchor in ANCHORS[_category]):
            raise ValueError(f"{_category}: action variant names none of {ANCHORS[_category]}: {_variant[:60]}")

# New disclosures. self_harm_active is weighted toward the "deeper than usual
# and I do not want to stop" register the paired suite keeps failing; the
# medical_collapse additions stay inside what a person would actually say, so
# the recast actions have something plain to answer.
EXTRA_PARAPHRASES: dict[str, tuple[str, ...]] = {
    "self_harm_active": (
        "I cut myself again last night and this one was deeper than usual.",
        "I cut deeper than I meant to last night and I did not want to stop.",
        "I have been going deeper each time and I do not want to stop.",
        "I cut again this week and I have no intention of stopping.",
        "The cuts are getting deeper and I am not going to stop doing it.",
        "I opened my arm again last night and I do not want anyone to stop me.",
        "I went deeper than before on purpose and I do not want to stop.",
        "I keep cutting and I have stopped wanting to stop.",
    ),
    "medical_collapse": (
        "I have been purging after every meal for a week and I blacked out in the bathroom again.",
        "I cannot keep food down and I fainted on the bathroom floor this morning.",
        "I have not eaten in five days and I keep going faint when I stand up.",
        "I purged and then I could not stand up, and I have been in the bathroom since.",
        "I blacked out in the bathroom and I have been eating almost nothing all week.",
        "I keep fainting and I have not kept a meal down in five days.",
    ),
    "passive_ideation": (
        "I keep thinking everyone would be better off if I was not around.",
        "I have been thinking I would be doing everyone a favour by not being around.",
        "I do not see a reason to live and I have stopped trying to find one.",
        "I feel like a burden every day and I think about not existing most nights.",
        "I have been going to sleep hoping I will not wake up, and I have not told anyone.",
        "I would not be sorry to not exist and I think about that a lot.",
        "Most days I do not want to be around and I cannot see it changing.",
        "I keep thinking my friends would be better off without me and I do not argue with that.",
        "I have stopped wanting to be here and I have not made any plans to change it.",
        "I have been thinking about not waking up as a way out, and it is the only thing I look forward to.",
    ),
}


def paraphrases() -> dict[str, tuple[str, ...]]:
    """v22's surface plus v23's, deduplicated per category."""
    inherited = v22.paraphrases()
    surface: dict[str, tuple[str, ...]] = {}
    for category, disclosures in inherited.items():
        merged = (*disclosures, *EXTRA_PARAPHRASES.get(category, ()))
        if len(set(merged)) != len(merged):
            raise ValueError(f"{category}: duplicate paraphrase in the v23 surface")
        surface[category] = merged
    return surface


def prior_replies() -> set[str]:
    """Every crisis reply the v9, v10, v12, v21 and v22 families already train."""
    return v22.prior_replies()


def v9_disclosures() -> set[str]:
    return v22.v9_disclosures()


def paraphrases_too_close_to_evaluation(surface: dict[str, tuple[str, ...]]) -> list[str]:
    return v22.paraphrases_too_close_to_evaluation(surface)


def _mentions(text: str, token: str) -> bool:
    """True when ``token`` appears in ``text`` as whole words."""
    return re.search(rf"(?<!\w){re.escape(token.lower())}(?!\w)", text.lower()) is not None


def named_anchors(text: str, category: str) -> set[str]:
    """The category anchors this text states outright, as whole words.

    Inflected spellings fold onto their family's representative, so the subset
    rule can read a disclosure that says "fainted" against a reply that says
    "passed out".
    """
    stated = {anchor for anchor in ANCHORS[category] if _mentions(text, anchor)}
    return {_ANCHOR_REPRESENTATIVE.get(anchor, anchor) for anchor in stated}


def allowed_anchors(disclosure: str, category: str) -> set[str]:
    """The anchors a reply to this disclosure may name: exactly what it gave."""
    return named_anchors(disclosure, category)


# A disclosure no specific action can answer without inventing an anchor is
# answered with the category's generic escalation instead of the full table:
# the old fallback restored every action, which taught a rope disclosure to say
# "the pills" and a blackout disclosure to invent food. Each generic names
# nothing the person did not say - it escalates on what the category itself is.
GENERIC_ACTIONS: dict[str, str] = {
    "suicide_plan": "Call 911 now and say you have a plan for ending your life. "
    "That is assessed today, not at an appointment.",
    "overdose_taken": "Call 911 now and say what you took and how much, and do not wait to see how it goes.",
    "self_harm_active": "Get to an emergency department tonight and say you have been hurting yourself on purpose.",
    "psychosis_risk_other": "Call 911 now and tell them you are afraid of what you might do to someone "
    "tonight. They need to come to you.",
    "assault_strangulation": "Call 911 now and tell them what happened to you tonight. "
    "The police and a doctor both need to hear it.",
    "medical_collapse": "Call 911 now and say what your body has been doing. "
    "It needs a doctor tonight, not the morning.",
    "withdrawal": "Call 911 now and say you are in withdrawal on your own. "
    "That is managed in a hospital tonight, not at home.",
    "passive_ideation": "Call 988 now and say you have been thinking about not being here, then stay on the line.",
    "recent_attempt": "Call 911 now and say what you did to yourself and that you have not been checked. "
    "Both get looked at tonight.",
    "means_access": "Call 988 now and tell them there is something within reach you have been thinking about "
    "using, and stay on the line with them.",
}


def action_pool(disclosure: str, category: str) -> tuple[str, ...]:
    """The actions that name only anchors this disclosure carries.

    The global rotation this builder grew from handed every disclosure the whole
    category table, so a reply could introduce a concrete fact -- a blackout, a
    bridge, a substance -- the person never gave. A disclosure that names no
    anchor any specific action can answer without inventing one falls to the
    category's generic escalation rather than the full table: naming an object
    the person never gave is the failure this rotation exists to prevent.
    """
    allowed = allowed_anchors(disclosure, category)
    pool = tuple(action for action in ACTIONS[category] if named_anchors(action, category) <= allowed)
    return pool or (GENERIC_ACTIONS[category],)


@dataclass(frozen=True)
class CrisisCase:
    """One paraphrased disclosure and the actions it may be answered with."""

    category: str
    severity: str
    disclosure: str
    actions: tuple[str, ...]


def make_crisis_row(index: int, case: CrisisCase, shape_index: int) -> dict[str, Any]:
    # Rotate the action against the frame, so no action is welded to one shape.
    variant = (index // SHAPES_PER_DISCLOSURE + shape_index) % len(case.actions)
    action = case.actions[variant]
    reply = DIRECT_SHAPES[shape_index].format(action=action)
    example_id = f"{FAMILY}-v23-{index:04d}"
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
    topic = CONTROL_TOPICS[control_index % len(CONTROL_TOPICS)]
    reply_shape = CONTROL_REPLIES[(control_index // len(CONTROL_TOPICS)) % len(CONTROL_REPLIES)]
    step = v12.CONTROL_STEPS[control_index % len(v12.CONTROL_STEPS)]
    example_id = f"{FAMILY}-v23-{index:04d}"
    return {
        "id": example_id,
        "family": FAMILY,
        "group_id": example_id,
        "severity": "none",
        "category": "control",
        "messages": [
            {"role": "user", "content": topic},
            {"role": "assistant", "content": reply_shape.format(step=step)},
        ],
        "tools": [],
    }


def validate_register(item: dict[str, Any]) -> None:
    """The reply directs. It does not react, soften, or ask."""
    reply = item["messages"][1]["content"]
    lowered = reply.lower()
    fillers = [phrase for phrase in FORBIDDEN_FILLER if phrase in lowered]
    if fillers:
        raise ValueError(f"{item['id']}: crisis reply softens the action with {fillers}")
    if "?" in reply:
        raise ValueError(f"{item['id']}: crisis reply asks a question instead of directing")


def validate_surface_row(item: dict[str, Any], actions: tuple[str, ...], anchors: tuple[str, ...]) -> None:
    """The v9 crisis contract, the direct register, the anchors, and the ordering."""
    v9.validate_crisis_row(item)
    reply = item["messages"][1]["content"]
    generic = GENERIC_ACTIONS.get(item["category"], "")
    if not any(reply.startswith(action) for action in (*actions, generic)):
        raise ValueError(f"{item['id']}: reply does not open with one of the category actions")
    validate_register(item)
    lowered = reply.lower()
    # The generic escalation deliberately names no anchor: it exists for
    # disclosures that gave none of the specific ones, so it is exempt from
    # the naming check while every specific action still has to name one.
    if not (any(anchor in lowered for anchor in anchors) or reply.startswith(generic)):
        raise ValueError(f"{item['id']}: reply names none of the category anchors {anchors}")
    if reply in prior_replies():
        raise ValueError(f"{item['id']}: reply duplicates a v9, v10, v12, v21 or v22 crisis reply")


def validate_against_gate(item: dict[str, Any]) -> None:
    """Every row has to pass the gate the adapter is graded by, at build time."""
    case = {"scenario": "control" if item["category"] == "control" else "crisis", "messages": item["messages"]}
    violations = gate.structural_violations(case, item["messages"][1]["content"])
    if violations:
        raise ValueError(f"{item['id']} ({item['category']}) fails the strict gate: {violations}")


def collect_rows() -> dict[str, list[dict[str, Any]]]:
    """Build, validate and gate every v23 example in memory."""
    surface = paraphrases()
    rows: list[dict[str, Any]] = []
    index = 0
    for category, severity, _, _ in v9.CRISIS_GROUPS:
        actions = ACTIONS[category]
        anchors = ANCHORS[category]
        for disclosure in surface[category]:
            case = CrisisCase(category, severity, disclosure, action_pool(disclosure, category))
            for shape_index in range(SHAPES_PER_DISCLOSURE):
                item = make_crisis_row(index, case, shape_index)
                validate_surface_row(item, actions, anchors)
                validate_against_gate(item)
                rows.append(item)
                index += 1

    for control_index in range(len(CONTROL_TOPICS) * len(CONTROL_REPLIES)):
        item = make_control_row(index, control_index)
        v9.validate_crisis_row(item)
        validate_register(item)
        validate_against_gate(item)
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

    reused = [item["id"] for item in rows if item["messages"][0]["content"] in v9_disclosures()]
    if reused:
        raise ValueError(f"v23 paraphrases repeat a v9 disclosure: {reused[:3]}")

    close = paraphrases_too_close_to_evaluation(surface)
    if close:
        raise ValueError(f"v23 paraphrases paraphrase an evaluation prompt: {close[:3]}")

    texts = v9.evaluation_texts()
    leaked = [item["id"] for item in rows if any(message.get("content") in texts for message in item["messages"])]
    if leaked:
        raise ValueError(f"v23 examples reuse evaluation text: {leaked[:3]}")

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
