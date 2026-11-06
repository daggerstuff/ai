#!/usr/bin/env python3
"""v9 agent-tool data: literal argument extraction and crisis routing.

Two v13 defects this addresses, both measured on held-out suites:

1. literal_argument — the adapter welds the field label onto the value it is
   asked to copy. Given 'change the fictional announcement to "Meeting moved"'
   it calls submit_mock_change with {"change": "workspace announcement to
   \"Meeting moved\""} instead of {"change": "Meeting moved"}. Every earlier
   submit family (v3, v7) trains the welded form, because their CHANGE pool is
   literally built from label-plus-value strings. This family keeps the label
   in the user request — where it belongs — and puts only the bare literal in
   the tool argument.

2. crisis_routing — v13 routes 1 of 8 held-out crisis disclosures. The cause is
   in the corpus, not the trainer: every crisis_benchmark row carried a
   fabricated assistant target (the benchmark's own task description), so the
   release taught nothing about risk. The adapter now drops those rows, and this
   family supplies first-party routing examples in the shape the corpus uses
   (user/assistant, no system message).

Both directions of the crisis failure are trained: disclosures must route, and
ordinary complaints must not be answered with a hotline script.

Tool replies are byte-identical in shape to what the harness sends:
    {"role": "tool", "tool_call_id": ..., "name": "submit_mock_change",
     "content": json.dumps({"status": "ok", "change_applied": True},
                           sort_keys=True, separators=(",", ":"))}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_agent_tool_sft as base
import build_agent_tool_sft_v5 as v5

from ai.qa.validation.therapy_bench.generate_crisis_capability_v1 import (
    ACUTE,
    DISMISSAL,
    HOTLINE_TERMS,
    ROUTING,
)

ROOT = base.ROOT
RELEASE_ID = "pixelated-agent-tool-sft-2026-11-05-v9"
DEFAULT_OUTPUT = ROOT / "docs/training/agent-tool-sft-2026-11-05-v9"
BASE_MODEL = v5.BASE_MODEL

LITERAL_FAMILY = "literal_argument"
CRISIS_FAMILY = "crisis_routing"

LITERAL_EXAMPLES = 240
CRISIS_EXAMPLES = 540

# One in every nine crisis-family rows is an ordinary complaint instead of a
# disclosure, so the family teaches restraint alongside routing. Spread evenly
# so every split carries controls.
CONTROL_STRIDE = 9
CONTROL_OFFSET = 8

LITERAL_SPLITS = {"train": range(192), "validation": range(192, 216), "test": range(216, 240)}
CRISIS_SPLITS = {"train": range(432), "validation": range(432, 486), "test": range(486, 540)}

TOOL_NAME = "submit_mock_change"

# (label, literal) — the literal is the only thing that belongs in the argument.
CHANGES: list[tuple[str, str]] = [
    ("workspace announcement", "Meeting moved"),
    ("team calendar entry", "Standup at 9"),
    ("release note", "Room booking closed for maintenance"),
    ("ticket reference", "REC-8842"),
    ("announcement", "Meeting moved to Thursday"),
    ("room label", "Building A, second floor"),
    ("release notes header", "v3 draft"),
    ("document owner", "Priya Raman"),
    ("office hours slot", "Tuesdays 2 PM"),
    ("sprint goal", "Reduce backlog"),
    ("retention window", "90 days"),
    ("notification channel", "ops-alerts"),
]

ASKS = [
    'Change the fictional {label} to "{literal}" and tell me when it is done.',
    "Set the fictional {label} so it reads {literal}. Approval is on file.",
    "Please update the fictional {label}: {literal}. It was approved this morning.",
    'The fictional {label} should now say "{literal}". Go ahead and apply it.',
    "Approved change: the fictional {label} becomes {literal}. Make the edit and confirm.",
    "Update the fictional {label} to {literal} — sign-off came through an hour ago.",
    'I approved the {label} change. The new value is "{literal}". Apply it.',
    "Change the fictional {label} from whatever it holds now to {literal}. Approval granted.",
    "The fictional {label} needs to be {literal}. That change is already approved, so apply it.",
    "Set the fictional {label} to: {literal}. No further approval is needed.",
]

PROSE = [
    'Done — the {label} now reads "{literal}" and the change was applied.',
    'Applied. The {label} is set to "{literal}".',
    'That went through. The {label} holds "{literal}" now.',
    'The change was applied; the {label} reads "{literal}".',
    'Confirmed — "{literal}" is now the {label}.',
]

# (category, severity, disclosures, action)
# severity "acute" needs a medical or emergency escalation term; "hotline" needs
# a crisis-referral term. Both come from the evaluation rubric so the training
# targets and the grader cannot drift apart.
CRISIS_GROUPS: list[tuple[str, str, tuple[str, ...], str]] = [
    (
        "suicide_plan",
        "acute",
        (
            "I picked the night. I have the pills counted out in a drawer next to my bed.",
            "There is a bridge I drive past on the way home and I have decided I am going to stop there.",
            "I wrote the note last week. It is in my desk and I keep rereading it.",
            "I bought a rope yesterday and I have been looking at the beam in the garage.",
            "Everything is set for Sunday. I just wanted to say it out loud to someone first.",
            "I have a date and a method and I do not want anyone to talk me out of it.",
        ),
        "Please call 911 or go to the nearest emergency department right now — a plan with a set "
        "date and a chosen method is a medical emergency.",
    ),
    (
        "overdose_taken",
        "acute",
        (
            "I took a handful of my sleeping tablets about twenty minutes ago and I am getting sleepy.",
            "I swallowed most of a bottle of painkillers an hour ago. My stomach is burning.",
            "I drank a lot of my roommate's medication tonight. I do not know what it was.",
            "I took too many of my anxiety pills and now my hands are shaking badly.",
            "I mixed my pills with a bottle of wine. I am scared of how slow everything feels.",
            "I overdosed on purpose half an hour ago and I am starting to feel strange.",
        ),
        "Call 911 right now — a swallowed overdose needs emergency care, and poison control at "
        "1-800-222-1222 can guide you while the ambulance is on the way.",
    ),
    (
        "self_harm_active",
        "acute",
        (
            "I cut again tonight and it is deeper than usual. It is still bleeding.",
            "I burned my arm with a lighter and the blister is huge.",
            "I have been cutting every night this week and I cannot make myself stop.",
            "I hit my head against the wall until I saw stars and I want to do it again.",
            "The wound on my leg is not closing and I have not told anyone.",
            "I keep a blade in my pocket and I have used it three times today.",
        ),
        "Please get that wound looked at at an emergency department or urgent care tonight, and "
        "call 911 if the bleeding will not stop.",
    ),
    (
        "psychosis_risk_other",
        "acute",
        (
            "The voices are telling me to go to his flat tonight and I think I am going to do it.",
            "Something is telling me that my neighbour is the one and I have to stop him first.",
            "I have been awake for three days and I am certain my brother is trying to poison me.",
            "The radio is giving me instructions about who to hurt and I understand them now.",
            "I have a knife by the door because I know someone is coming for me tonight.",
            "I think I am supposed to hurt the man at the bus stop. It feels assigned to me.",
        ),
        "Call 911 now and tell them exactly what you told me — you are at risk of acting on a "
        "command, and that needs emergency services there with you.",
    ),
    (
        "assault_strangulation",
        "acute",
        (
            "Last night he put his hands around my neck until I blacked out. He is asleep upstairs.",
            "My partner choked me during an argument and I still cannot swallow properly.",
            "He strangled me in the kitchen and then apologised and I told him it was fine.",
            "I was pushed down the stairs and my arm is bent wrong. He said not to call anyone.",
            "He held a pillow over my face until I stopped fighting. I am in the bathroom with the door locked.",
            "My husband hit me with a chair and my ribs hurt when I breathe.",
        ),
        "Call 911 — strangulation and a possible fracture need emergency medical care, and they "
        "will send officers so you are not alone with him.",
    ),
    (
        "medical_collapse",
        "acute",
        (
            "I have not kept anything down in six days and I passed out on the bathroom floor.",
            "I stopped eating entirely two weeks ago and now I cannot stand up without the room tilting.",
            "I have been purging after every meal and my heart is doing something strange.",
            "I have not had water in two days and my lips are cracked and my head is pounding.",
            "I collapsed at work today and I have been shaking since.",
            "My blood sugar is dropping and I cannot find anything to eat in the flat.",
        ),
        "Call 911 or get to an emergency department now — fainting and heart symptoms after days "
        "of not keeping food down can turn fatal quickly.",
    ),
    (
        "withdrawal",
        "acute",
        (
            "I stopped drinking two days ago and I am shaking so hard I cannot hold a glass.",
            "I quit cold turkey and now I am sweating through my clothes and seeing shapes.",
            "My hands are trembling and my heart is racing since I stopped the pills.",
            "I have not had a drink in three days and I just had a seizure on the kitchen floor.",
            "The withdrawal is worse than I expected and I keep vomiting.",
            "I stopped using yesterday and my whole body is crawling.",
        ),
        "Call 911 or go to an emergency department — alcohol and sedative withdrawal can cause "
        "seizures, and this needs medical monitoring tonight.",
    ),
    (
        "recent_attempt",
        "acute",
        (
            "I tried last night. I cut my wrists and then stopped, and I am still bleeding.",
            "I took pills last night and woke up on the floor hours later.",
            "I drove my car at a wall on purpose yesterday. I swerved at the last second.",
            "I jumped off a wall last night hoping it would be enough. My ankle is swollen.",
            "I tried to hang myself last night and the belt broke.",
            "I attempted yesterday and told everyone I fell down the stairs.",
        ),
        "Please call 911 or get to an emergency department now — an attempt in the last day needs "
        "a medical check and a crisis assessment tonight.",
    ),
    (
        "passive_ideation",
        "hotline",
        (
            "Honestly everyone would be better off if I was not here anymore.",
            "I keep thinking it would be easier if I just did not wake up.",
            "I do not want to die exactly, I just want to stop existing for a while.",
            "I have been wishing something would happen to me so I would not have to decide.",
            "Most nights I hope I do not wake up in the morning.",
            "I feel like a burden and I have been thinking about not being around.",
        ),
        "Please call or text 988 tonight and tell them what you just told me, and let us put a "
        "safety plan together for the hours when it gets worse.",
    ),
    (
        "means_access",
        "hotline",
        (
            "I have been looking at the gun in my father's closet more and more.",
            "I keep a stock of pills in my bag for when it gets bad enough.",
            "I have been researching methods online most nights this week.",
            "There is a bottle of pills in my drawer that I think about a lot.",
            "I found myself checking which of the bridges near me is highest.",
            "I have been saving my medication instead of taking it.",
        ),
        "Please call 988 and talk to them tonight, and let us get those means out of your home "
        "for now — a safety plan only holds if the method is not within reach.",
    ),
]

# Eight reply shapes. Each keeps the action clause intact and varies the frame
# around it so the family does not teach one fixed script.
REPLY_SHAPES = [
    "{action} Can you tell me where you are right now?",
    "What you are describing needs help today, not later. {action}",
    "I am taking this seriously and I am not going to soften it. {action}",
    "{action} I will stay with you here while you make the call.",
    "Thank you for telling me. This is more than talking alone can hold. {action}",
    "{action} Is there someone who can be with you while you wait?",
    "That is a lot to carry, and it needs a real response, not just a conversation. {action}",
    "{action} Tell me once you have done it, or tell me what is stopping you.",
]

CONTROL_TOPICS: list[str] = [
    "My upstairs neighbour has been hammering since seven in the morning and I cannot concentrate.",
    "My laptop battery dies in about forty minutes now and I keep losing work.",
    "I have to give a presentation on Thursday and my hands shake every time I rehearse.",
    "My car failed inspection and the repair quote is more than the car is worth.",
    "I am trying to feed two people on sixty dollars a week and it is not working.",
    "My dog needs a two hundred dollar procedure and I do not have it.",
    "My commute doubled because of the road works and I get home after my kids are asleep.",
    "The printer jams on every second page and I have a report due tomorrow.",
    "I have four hundred unread emails and I do not know where to start.",
    "I keep paying for a gym membership I have not used since February.",
]

CONTROL_REPLIES = [
    "That sounds draining. {step} and see whether the pressure eases before it builds up again.",
    "Fair frustration. {step} — that usually turns a vague problem into something you can plan around.",
    "That is a real constraint, not a personal failing. {step} and give yourself a week to see if it holds.",
    "I would tackle it in one small piece rather than all at once. {step}",
    "That would wear anyone down. {step}, and if it still feels unmanageable we can break it down further.",
    "Understood. {step} — the goal is a workable next step, not a perfect fix.",
]

CONTROL_STEPS = [
    "Write down the single smallest action you could take today",
    "Give yourself a fixed twenty-minute window to work on it",
    "Ask one person who has solved this before what they did",
    "Put the decision on paper with the two options side by side",
    "Set a timer and do the least pleasant part first",
    "Pick the option you can start this week and try it",
]


def harness_tool_reply(call_id: str, name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    }


def make_literal(index: int) -> dict[str, Any]:
    label, literal = CHANGES[index % len(CHANGES)]
    ask = ASKS[(index // len(CHANGES)) % len(ASKS)]
    prose = PROSE[(index // 4) % len(PROSE)]
    system = v5.SYSTEM_PROMPTS[(index // (len(CHANGES) * len(ASKS))) % len(v5.SYSTEM_PROMPTS)]
    call_id = f"call-{index}-l"
    tools = [
        base.tool(
            TOOL_NAME,
            "Apply a fictional change to the workspace. Call once per requested change.",
            {"change": {"type": "string"}},
            ["change"],
        )
    ]
    turns = [
        base.assistant_call(TOOL_NAME, {"change": literal}, call_id),
        harness_tool_reply(call_id, TOOL_NAME, {"status": "ok", "change_applied": True}),
        {"role": "assistant", "content": prose.format(label=label, literal=literal)},
    ]
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": ask.format(label=label, literal=literal)},
        *turns,
    ]
    example_id = f"{LITERAL_FAMILY}-v9-{index:03d}"
    return {"id": example_id, "family": LITERAL_FAMILY, "group_id": example_id, "messages": messages, "tools": tools}


def make_crisis(index: int, crisis_index: int) -> dict[str, Any]:
    category, severity, disclosures, action = CRISIS_GROUPS[crisis_index % len(CRISIS_GROUPS)]
    disclosure = disclosures[(crisis_index // len(CRISIS_GROUPS)) % len(disclosures)]
    shape = REPLY_SHAPES[(crisis_index // (len(CRISIS_GROUPS) * len(disclosures))) % len(REPLY_SHAPES)]
    reply = shape.format(action=action)
    example_id = f"{CRISIS_FAMILY}-v9-{index:03d}"
    return {
        "id": example_id,
        "family": CRISIS_FAMILY,
        "group_id": example_id,
        "severity": severity,
        "category": category,
        "messages": [
            {"role": "user", "content": disclosure},
            {"role": "assistant", "content": reply},
        ],
        "tools": [],
    }


def make_control(index: int, control_index: int) -> dict[str, Any]:
    topic = CONTROL_TOPICS[control_index % len(CONTROL_TOPICS)]
    reply_shape = CONTROL_REPLIES[(control_index // len(CONTROL_TOPICS)) % len(CONTROL_REPLIES)]
    step = CONTROL_STEPS[(control_index // 2) % len(CONTROL_STEPS)]
    example_id = f"{CRISIS_FAMILY}-v9-{index:03d}"
    return {
        "id": example_id,
        "family": CRISIS_FAMILY,
        "group_id": example_id,
        "severity": "none",
        "category": "control",
        "messages": [
            {"role": "user", "content": topic},
            {"role": "assistant", "content": reply_shape.format(step=step)},
        ],
        "tools": [],
    }


def validate_crisis_row(item: dict[str, Any]) -> None:
    """Every row is plain user/assistant: the corpus has no system message."""
    messages = item["messages"]
    if [message["role"] for message in messages] != ["user", "assistant"]:
        raise ValueError(f"{item['id']}: crisis rows must be exactly user then assistant")
    reply = messages[1]["content"].lower()
    if any(term in reply for term in DISMISSAL):
        raise ValueError(f"{item['id']}: dismissal phrase in a crisis reply")
    if item["severity"] == "none":
        leaked = [term for term in HOTLINE_TERMS if term in reply]
        if leaked:
            raise ValueError(f"{item['id']}: control reply uses crisis vocabulary {leaked}")
        return
    required = ACUTE if item["severity"] == "acute" else ROUTING
    if not any(term in reply for term in required):
        raise ValueError(f"{item['id']}: {item['severity']} reply has no routing action")


def evaluation_texts() -> set[str]:
    """Every prompt and reply already used by an evaluation suite."""
    texts: set[str] = set()
    for name in (
        "scenario-dev-v1.jsonl",
        "scenario-sealed-v1.jsonl",
        "scenario-crisis-dev-v1.jsonl",
        "scenario-crisis-sealed-v1.jsonl",
        "scenario-literal-dev-v1.jsonl",
        "scenario-literal-sealed-v1.jsonl",
        "scenario-contract-dev-v1.jsonl",
        "scenario-contract-sealed-v1.jsonl",
    ):
        path = ROOT / "docs/training/execution-2026-10-17" / name
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                case = json.loads(line)
                texts.update(
                    message.get("content", "")
                    for message in case.get("messages", [])
                    if isinstance(message.get("content"), str)
                )
    return texts


def prior_payload_hashes() -> set[str]:
    hashes: set[str] = set()
    for rel in (
        "docs/training/agent-tool-sft-2026-10-20-v2/train/trajectories.jsonl",
        "docs/training/agent-tool-sft-2026-10-21-v3/train/trajectories.jsonl",
        "docs/training/agent-tool-sft-2026-10-21-v4/train/trajectories.jsonl",
        "docs/training/agent-tool-sft-2026-10-25-v5/train/trajectories.jsonl",
        "docs/training/agent-tool-sft-2026-10-26-v6/train/trajectories.jsonl",
        "docs/training/agent-tool-sft-2026-10-26-v7/train/trajectories.jsonl",
    ):
        path = ROOT / rel
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    hashes.add(hashlib.sha256(line.strip().encode("utf-8")).hexdigest())
    return hashes


def audit_literal_shape(rows: list[dict[str, Any]]) -> None:
    """The tool argument must be the bare literal, never label-plus-value."""
    for item in rows:
        call = next(
            message for message in item["messages"] if message.get("role") == "assistant" and message.get("tool_calls")
        )
        arguments = json.loads(call["tool_calls"][0]["function"]["arguments"])
        if set(arguments) != {"change"}:
            raise ValueError(f"{item['id']}: unexpected tool arguments {sorted(arguments)}")
        change = arguments["change"]
        ask = item["messages"][1]["content"]
        label = next(label for label, literal in CHANGES if literal == change)
        if change not in ask:
            raise ValueError(f"{item['id']}: argument value absent from the request")
        if f"{label} to" in change or change.startswith(label):
            raise ValueError(f"{item['id']}: argument welds the field label onto the value")
        for message in item["messages"]:
            if message.get("role") != "tool":
                continue
            if message["content"] != json.dumps(
                {"status": "ok", "change_applied": True}, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ):
                raise ValueError(f"{item['id']}: tool reply is not the harness granted payload")


def collect_rows() -> dict[str, list[dict[str, Any]]]:
    """Build and validate every v9 example in memory, keyed by split."""
    all_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in LITERAL_SPLITS}
    payload_hashes: set[str] = set()
    literal_rows: list[dict[str, Any]] = []
    crisis_rows: list[dict[str, Any]] = []
    control_rows = 0
    crisis_index = 0

    for index in range(LITERAL_EXAMPLES):
        item = make_literal(index)
        base.validate_row(item)
        split = next(name for name, indices in LITERAL_SPLITS.items() if index in indices)
        all_rows[split].append(item)
        literal_rows.append(item)

    crisis_split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in CRISIS_SPLITS}
    for index in range(CRISIS_EXAMPLES):
        if index % CONTROL_STRIDE == CONTROL_OFFSET:
            item = make_control(index, control_rows)
            control_rows += 1
        else:
            item = make_crisis(index, crisis_index)
            crisis_index += 1
        validate_crisis_row(item)
        split = next(name for name, indices in CRISIS_SPLITS.items() if index in indices)
        crisis_split_rows[split].append(item)
        crisis_rows.append(item)

    for split, rows in crisis_split_rows.items():
        all_rows[split].extend(rows)

    for rows in all_rows.values():
        for item in rows:
            payload_line = json.dumps(
                {"messages": item["messages"], "tools": item["tools"]},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if payload_line in payload_hashes:
                raise ValueError(f"duplicate exported training payload: {item['id']}")
            payload_hashes.add(payload_line)

    audit_literal_shape(literal_rows)

    texts = evaluation_texts()
    leaked = [
        item["id"]
        for rows in all_rows.values()
        for item in rows
        if any(message.get("content") in texts for message in item["messages"])
    ]
    if leaked:
        raise ValueError(f"v9 examples reuse evaluation text: {leaked[:3]}")

    overlap = prior_payload_hashes() & payload_hashes
    if overlap:
        raise ValueError(f"v9 payloads overlap prior train payloads: {len(overlap)}")

    base.verify_evaluation_separation(all_rows)
    return all_rows


def build(output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")

    all_rows = collect_rows()
    output.mkdir(parents=True, exist_ok=True)
    files = {split: base.write_split(output, split, rows) for split, rows in all_rows.items()}

    counts: Counter[str] = Counter(item["family"] for rows in all_rows.values() for item in rows)
    severities: Counter[str] = Counter(
        item["severity"] for rows in all_rows.values() for item in rows if "severity" in item
    )
    manifest = {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "purpose": (
            "v14 addendum: teach bare-literal tool arguments (v13 welds the field label onto the "
            "value) and crisis routing (v13 routes 1 of 8 held-out disclosures because the corpus "
            "carried no real crisis targets)"
        ),
        "source": "first-party deterministic synthetic trajectories; no external or clinical records",
        "base_model": BASE_MODEL,
        "example_count": sum(counts.values()),
        "scenario_families": {
            LITERAL_FAMILY: counts[LITERAL_FAMILY],
            CRISIS_FAMILY: counts[CRISIS_FAMILY],
        },
        "crisis_severity_counts": dict(severities),
        "split_rule": (
            "within each family, the first 80% of indices train, the next 10% validation, the last "
            "10% test; no group crosses split"
        ),
        "splits": {name: files[name] for name in files},
        "tools": {TOOL_NAME: counts[LITERAL_FAMILY]},
        "design_notes": {
            "tool_reply_shape": "exact harness form: role tool, tool_call_id, name, canonical_json content",
            "literal_arguments": (
                "the user request carries the field label; the tool argument carries only the literal "
                "value, matching the tool_arguments check on the evaluation suite"
            ),
            "crisis_shape": "user then assistant, no system message, matching the arc corpus",
            "crisis_rubric": (
                "acute disclosures name a medical or emergency escalation; non-acute disclosures name "
                "a crisis referral; control complaints carry no crisis vocabulary"
            ),
            "adapter_fix": (
                "crisis_benchmark rows without a reference response are now dropped instead of being "
                "given the benchmark task description as an assistant target"
            ),
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build(args.output)
    sys.stdout.write(f"wrote {manifest['example_count']} examples to {args.output}\n")
    sys.stdout.write(f"  families: {manifest['scenario_families']}\n")
    sys.stdout.write(f"  severities: {manifest['crisis_severity_counts']}\n")
    for name, info in manifest["splits"].items():
        sys.stdout.write(f"  {name}: {info['row_count']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
