"""v17 addendum: the crisis surface in a direct register.

v12 taught the right routing action in the wrong voice. Its frames were
therapy-shaped ("I am not going to move past this part", "Stay with me on this
until it is done") and its actions pleaded ("Please call 911 now"), so v16
answered the one remaining failing case — a disclosure that pairs a deep cut
with "I don't really want to stop" — with a warm reflection and no resource at
all. Coddling is not a softer version of the action; it is a competing answer
that wins, because it is the register the underlying therapy corpus is full of.

This family replaces v12 rather than sitting beside it. Two registers in the
same addendum is the same 50/50 conflict that made the v7 welded arguments and
the v9 bare literals cancel out. Rows are train-only, user/assistant with no
system message, matching the arc corpus.

Register rules, all enforced below:

* the reply opens with the action and never asks a question;
* no validation, reassurance, or self-reference before the action;
* the consequence is stated as a fact, not as a plea;
* ambivalence ("I don't want to stop") gets an answer about the wound, not
  about the feelings.
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
import build_agent_tool_sft_v12 as v12

ROOT = base.ROOT
RELEASE_ID = "pixelated-agent-tool-sft-2026-11-09-v13"
DEFAULT_OUTPUT = ROOT / "docs/training/agent-tool-sft-2026-11-09-v13"
BASE_MODEL = v9.BASE_MODEL

FAMILY = "crisis_direct"
SHAPES_PER_DISCLOSURE = 3

# The action is the first thing the model emits, and the frames only add a
# deadline. Nothing in them reacts to the disclosure.
DIRECT_SHAPES: tuple[str, ...] = (
    "{action}",
    "{action} Do not wait for it to get worse.",
    "{action} This does not keep until morning.",
)

if len(DIRECT_SHAPES) != SHAPES_PER_DISCLOSURE:
    raise ValueError("the shape table must stay in step with SHAPES_PER_DISCLOSURE")

# Phrases that spend the opening tokens on the responder's feelings instead of
# the action. Any one of them makes the reply a therapy reply again.
FORBIDDEN_FILLER: tuple[str, ...] = (
    "please",
    "i am glad",
    "i'm glad",
    "thank you for",
    "that's a lot",
    "that is a lot",
    "i hear you",
    "i understand",
    "you are not alone",
    "it makes sense",
    "that must be",
    "that sounds",
    "let's",
    "let us",
    "we can",
    "would you",
    "can you tell me",
    "take your time",
    "i am sorry",
    "i'm sorry",
    "stay with me",
    "move past",
    "big decisions",
)

# Three direct action variants per category. Every variant satisfies the
# evaluation rubric for its severity, names the concrete consequence of delay,
# and keeps the register clinical: no pleading, no hedging, no question.
ACTIONS: dict[str, tuple[str, ...]] = {
    "suicide_plan": (
        "Call 911 now. A set date and a chosen method is a medical emergency and it gets assessed today.",
        "Call 911 and tell them you have a plan and the means. This does not wait for an appointment.",
        "Get to an emergency department now, or call 911 if you cannot get there safely. A dated plan "
        "with a method is a medical emergency.",
    ),
    "overdose_taken": (
        "Call 911 now and say what you took and how long ago. What you swallowed keeps working for hours.",
        "Call 911 now. An overdose gets worse before it gets better, whatever it feels like yet.",
        "Call poison control and 911 now and stay on the line. This needs an emergency department tonight.",
    ),
    "self_harm_active": (
        "Get that wound assessed at an emergency department tonight. Deep cuts and bleeding that will "
        "not stop need closing, not managing at home.",
        "Call 911 if the bleeding will not stop. Otherwise get to an emergency department or urgent care "
        "tonight and have the wound closed.",
        "Go to an emergency department tonight. Not wanting to stop changes nothing about the wound — it "
        "needs assessing and closing now.",
    ),
    "psychosis_risk_other": (
        "Call 911 now and say you are afraid you will act on what the voices tell you. They need to come "
        "to you tonight.",
        "Call 911 now. A command you are afraid to follow is an emergency and it gets a team at your door.",
        "Call 911 now and tell them someone may be hurt tonight. Stay on the line until they arrive.",
    ),
    "assault_strangulation": (
        "Call 911 now and ask for an ambulance and the police. Being choked causes damage that shows up hours later.",
        "Call 911 now. Strangulation is a medical emergency and a crime, and both get reported tonight.",
        "Get out of the house and call 911. Hands on your throat is an emergency and it needs a medical "
        "assessment tonight.",
    ),
    "medical_collapse": (
        "Call an ambulance now. Days without keeping food down plus a collapse means your heart and "
        "electrolytes get checked tonight.",
        "Call 911 now. Collapse and an irregular heartbeat after purging or restricting is a medical "
        "emergency, not a rest problem.",
        "Get to an emergency department now, by ambulance if you cannot walk steadily. This is not "
        "something to watch overnight at home.",
    ),
    "withdrawal": (
        "Call 911 now. Alcohol and sedative withdrawal causes seizures in the first days and that needs a "
        "medical team.",
        "Get to an emergency department now. Withdrawal seizures happen at this stage and cannot be "
        "managed in a quiet room.",
        "Call 911 or get to an emergency department tonight. This is the window where withdrawal turns into a seizure.",
    ),
    "recent_attempt": (
        "Call 911 now. An attempt in the last day needs the injury assessed and a crisis assessment "
        "before you go home.",
        "Get to an emergency department now. Yesterday's attempt is an open medical risk, not a past event.",
        "Call 911 or get to an emergency department tonight and tell them what you did and when. Both "
        "the injury and the risk get assessed.",
    ),
    "passive_ideation": (
        "Call or text 988 tonight. Wanting to not wake up is what that line is for and it is staffed around the clock.",
        "Call 988 tonight and say what you just said here. A crisis line is the right place for this and "
        "it does not close.",
        "Text 988 now. This is the reason the line exists and it takes the call tonight, not next week.",
    ),
    "means_access": (
        "Call 988 now and get the means out of the house tonight — give it to someone or lock it away.",
        "Call 988 tonight, then hand the means to someone you trust. Having the method within reach is "
        "what turns thinking into acting.",
        "Call 988 now and tell them what you have been keeping close. Then the means leaves the house tonight.",
    ),
}

if set(ACTIONS) != set(v12.SEVERITIES):
    raise ValueError("the action table must cover exactly the v9 crisis categories")
for _category, _variants in ACTIONS.items():
    if len(_variants) != SHAPES_PER_DISCLOSURE:
        raise ValueError(f"{_category}: expected {SHAPES_PER_DISCLOSURE} action variants")
if len({action for variants in ACTIONS.values() for action in variants}) != len(ACTIONS) * SHAPES_PER_DISCLOSURE:
    raise ValueError("two categories share an action variant")

# Disclosures that pair an injury with a refusal to stop or a refusal of help.
# This is the surface v12 missed: its self-harm paraphrases all describe a
# wound, none of them argue against treatment, so the model read the argument
# as the topic and answered it reflectively.
EXTRA_PARAPHRASES: dict[str, tuple[str, ...]] = {
    "self_harm_active": (
        "The wound from last night is deeper than anything I have done and I am not going to stop cutting.",
        "I am bleeding through the dressing and I am not going to stop cutting because someone says so.",
        "I went deeper tonight than I have before and part of me wants to keep going.",
        "The cuts need closing and I do not want anyone touching them.",
        "I have been cutting every night this week and I am not planning to stop.",
        "I cut down to the fat this time and I am not interested in being talked out of it.",
        "There is blood running down my arm and I would rather keep going than have it looked at.",
        "I reopened everything tonight and I do not want help, I want to be left alone with it.",
    ),
    "overdose_taken": (
        "I took them on purpose and I am not going to call anyone about it.",
        "I swallowed the tablets an hour ago and I do not want to be treated for it.",
        "I overdosed and I am not interested in an ambulance.",
        "I took the whole bottle and I would rather go to sleep than go to hospital.",
    ),
}

# The non-escalation direction, in the same register as the crisis replies: a
# plain next step, no validation preamble.
CONTROL_REPLIES: tuple[str, ...] = (
    "{step}.",
    "Start here: {step}.",
    "{step}. Do that before anything else.",
    "One step: {step}.",
    "{step}, today, and nothing wider until it is done.",
    "{step}. If it has not shifted after that, come back with what happened.",
)

SEVERITIES: dict[str, str] = {category: severity for category, severity, _, _ in v9.CRISIS_GROUPS}


def paraphrases() -> dict[str, tuple[str, ...]]:
    """v12's disclosure surface plus the ambivalence disclosures it lacked."""
    surface = {
        category: tuple(v12.PARAPHRASES[category]) + EXTRA_PARAPHRASES.get(category, ()) for category in v12.PARAPHRASES
    }
    for category, disclosures in surface.items():
        if len(set(disclosures)) != len(disclosures):
            raise ValueError(f"{category}: duplicate paraphrase")
    return surface


def prior_replies() -> set[str]:
    """Every crisis reply the v9, v10 and v12 families already train."""
    replies: set[str] = (
        v12.prior_replies()
        | {shape.format(action=action) for _, _, _, action in v9.CRISIS_GROUPS for shape in v12.SURFACE_SHAPES}
        | {
            shape.format(action=action)
            for actions in v12.ACTIONS.values()
            for action in actions
            for shape in v12.SURFACE_SHAPES
        }
    )
    return replies


def v9_disclosures() -> set[str]:
    return {disclosure for _, _, disclosures, _ in v9.CRISIS_GROUPS for disclosure in disclosures}


EVALUATION_DIR = ROOT / "docs/training/execution-2026-10-17"
SHINGLE_WORDS = 6
MAX_SHARED_SHINGLES = 4


def _shingles(text: str) -> set[str]:
    words = " ".join(text.lower().split()).replace(".", " ").replace(",", " ").split()
    width = SHINGLE_WORDS
    return {" ".join(words[index : index + width]) for index in range(len(words) - width + 1)}


def evaluation_prompts() -> list[str]:
    """Every user turn the evaluation suites already use."""
    prompts: list[str] = []
    for path in sorted(EVALUATION_DIR.glob("scenario-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                prompts.append(json.loads(line)["messages"][0]["content"])
    return prompts


def paraphrases_too_close_to_evaluation(surface: dict[str, tuple[str, ...]]) -> list[str]:
    """Disclosures that paraphrase an evaluation prompt instead of covering it.

    The exact-match check in v9 is not enough: expanding a contraction walks
    straight past it while training the case the suites measure.
    """
    evaluation = {shingle for prompt in evaluation_prompts() for shingle in _shingles(prompt)}
    return [
        disclosure
        for disclosures in surface.values()
        for disclosure in disclosures
        if len(_shingles(disclosure) & evaluation) >= MAX_SHARED_SHINGLES
    ]


@dataclass(frozen=True)
class CrisisCase:
    """One paraphrased disclosure and the action variants its category carries."""

    category: str
    severity: str
    disclosure: str
    actions: tuple[str, ...]


def make_crisis_row(index: int, case: CrisisCase, shape_index: int) -> dict[str, Any]:
    # Rotate the action against the frame, so no action is welded to one shape.
    variant = (index // SHAPES_PER_DISCLOSURE + shape_index) % len(case.actions)
    action = case.actions[variant]
    reply = DIRECT_SHAPES[shape_index].format(action=action)
    example_id = f"{FAMILY}-v13-{index:03d}"
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
    topic = v12.CONTROL_TOPICS[control_index % len(v12.CONTROL_TOPICS)]
    reply_shape = CONTROL_REPLIES[(control_index // len(v12.CONTROL_TOPICS)) % len(CONTROL_REPLIES)]
    step = v12.CONTROL_STEPS[control_index % len(v12.CONTROL_STEPS)]
    example_id = f"{FAMILY}-v13-{index:03d}"
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


def validate_surface_row(item: dict[str, Any], actions: tuple[str, ...]) -> None:
    """The v9 crisis contract, the direct register, and the resource-first ordering."""
    v9.validate_crisis_row(item)
    reply = item["messages"][1]["content"]
    if not any(reply.startswith(action) for action in actions):
        raise ValueError(f"{item['id']}: reply does not open with one of the category actions")
    validate_register(item)
    if reply in prior_replies():
        raise ValueError(f"{item['id']}: reply duplicates a v9, v10 or v12 crisis reply")


def collect_rows() -> dict[str, list[dict[str, Any]]]:
    """Build and validate every v13 example in memory."""
    surface = paraphrases()
    rows: list[dict[str, Any]] = []
    index = 0
    for category, severity, _, _ in v9.CRISIS_GROUPS:
        actions = ACTIONS[category]
        for disclosure in surface[category]:
            case = CrisisCase(category, severity, disclosure, actions)
            for shape_index in range(SHAPES_PER_DISCLOSURE):
                item = make_crisis_row(index, case, shape_index)
                validate_surface_row(item, actions)
                rows.append(item)
                index += 1

    for control_index in range(len(v12.CONTROL_TOPICS) * len(CONTROL_REPLIES)):
        item = make_control_row(index, control_index)
        v9.validate_crisis_row(item)
        validate_register(item)
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
        raise ValueError(f"v13 paraphrases repeat a v9 disclosure: {reused[:3]}")

    close = paraphrases_too_close_to_evaluation(surface)
    if close:
        raise ValueError(f"v13 paraphrases paraphrase an evaluation prompt: {close[:3]}")

    texts = v9.evaluation_texts()
    leaked = [item["id"] for item in rows if any(message.get("content") in texts for message in item["messages"])]
    if leaked:
        raise ValueError(f"v13 examples reuse evaluation text: {leaked[:3]}")

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
            "v17 addendum: v16 routes nine of ten crisis categories but answers a deep cut paired with "
            "'I don't really want to stop' with a warm reflection and no resource, because v12 taught the "
            "action inside a therapy voice ('Please call 911', 'Stay with me on this'); v13 replaces v12 "
            "with the same disclosure surface plus the ambivalence surface, in a direct action-first "
            "register with no validation, no self-reference and no questions"
        ),
        "source": "first-party deterministic synthetic trajectories; no external or clinical records",
        "base_model": BASE_MODEL,
        "example_count": summary["example_count"],
        "scenario_families": {FAMILY: summary["example_count"]},
        "crisis_severity_counts": summary["crisis_severity_counts"],
        "crisis_category_counts": summary["crisis_category_counts"],
        "distinct_disclosures": summary["distinct_disclosures"],
        "distinct_replies": summary["distinct_replies"],
        "split_rule": "train only, matching the v3..v12 addendum convention",
        "splits": {name: files[name] for name in files},
        "tools": {},
        "replaces": "pixelated-agent-tool-sft-2026-11-08-v12",
        "design_notes": {
            "register": (
                "the reply opens with the action and states the consequence as fact; no plea, no "
                "validation, no question, no reference to the responder"
            ),
            "replacement_not_addition": (
                "v12 rows are dropped from the release rather than kept beside these, because two "
                "registers in one family cancel each other out"
            ),
            "ambivalence_surface": (
                "twelve disclosures pair an injury with a refusal to stop or a refusal of treatment, the "
                "shape that pulled v16 back into a reflective reply"
            ),
            "surface_breadth": (
                "twenty v12 paraphrases per category plus the ambivalence disclosures, written across the "
                "direct, minimised, euphemistic, physical-symptom and retrospective registers"
            ),
            "action_variants": (
                "three action variants per category, so the reply is not one memorised sentence; every "
                "variant satisfies the evaluation rubric for its severity"
            ),
            "resource_first": (
                "every reply begins with the action clause, matching v10; v9 put it first in 3 of 8 "
                "shapes and the adapter dropped it in the other 5"
            ),
            "crisis_shape": "user then assistant, no system message, matching the arc corpus",
            "no_eval_reuse": (
                "no paraphrase and no reply is an evaluation sentence; the suites stay a generalisation "
                "check on the category, not on the wording"
            ),
            "restraint": (
                "72 ordinary complaints in topics v9 does not use, answered in the same register, so the "
                "family does not teach a hotline script for every hard week"
            ),
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
