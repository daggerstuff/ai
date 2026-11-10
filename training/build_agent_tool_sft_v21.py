"""v21 addendum: the v20 crisis family, wider, over a corpus that no longer
teaches the preamble.

v20 was the first adapter to sweep both paired-eval baselines it was built
against: 52/52 against base and 52/52 against v19, with crisis at 11/11 on both
dev and sealed. Its only remaining losses were in the v20-vs-v16 pair, and all
three failed on one string:

    "I'm not going to pretend that's not serious. Let's get you to an emergency
     department tonight..."
    "I'm taking this seriously... Let's slow down. Are you alone right now?"

Auditing the corpus for those openers found almost nothing at the start of a
turn — 408 of 83,631 rows — but a great deal of the two phrases mid-turn:
362 occurrences of "I'm not going to pretend" and 192 of "Let's slow down".
Attributing them to source labels showed where they come from. 331 of the 362
sit in ``arc_corpus_*``, and 367 of the 981 arc rows carry one: our own
generated therapist voice, at 37% density, which is why the register keeps
resurfacing even though the crisis addendum itself is clean. The 192 "let's
slow down" sit in the clinical-book rows, where it is a filler sentence.

So v21 does two things v20 could not:

* **It repairs the teaching source.** The v21 release strips those two phrases
  from carried train rows — 568 rows, 597 assistant turns — by dropping the
  sentence that carries them. That is a two-phrase deletion, not the fifty-
  phrase sweep v20 rejected: both phrases are on the crisis rubric's omit list
  and both appear verbatim in an observed v20 failure, while "that sounds",
  "that must be" and "you deserve" are left exactly where they are.
* **It grows the surface again**, concentrated on the three categories that
  keep showing up in the losses: ``self_harm_active``, ``psychosis_risk_other``
  and ``suicide_plan``. Each of those also gains two more action variants, so
  the family is not just the same 3 sentences repeated at higher volume.

Register rules, anchors and forbidden filler are inherited from v20 unchanged:
the reply opens with the action, never asks a question, never validates first,
states the consequence as fact, and must name the concrete thing the disclosure
named. Rows are train-only, user/assistant, empty tool catalog.
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
import build_agent_tool_sft_v20 as v20

ROOT = base.ROOT
RELEASE_ID = "pixelated-agent-tool-sft-2026-11-10-v21"
DEFAULT_OUTPUT = ROOT / "docs/training/agent-tool-sft-2026-11-10-v21"
BASE_MODEL = v20.BASE_MODEL

FAMILY = v20.FAMILY
SHAPES_PER_DISCLOSURE = v20.SHAPES_PER_DISCLOSURE
DIRECT_SHAPES = v20.DIRECT_SHAPES
ANCHORS = v20.ANCHORS

# Both phrases open an observed v20 crisis failure and are on the rubric's omit
# list. The release deletes "not going to pretend" from the corpus, so the
# addendum must not be able to teach it; "taking this seriously" is the same
# validation preamble class as "that matters", which v20 already bans.
EXTRA_FORBIDDEN_FILLER: tuple[str, ...] = ("not going to pretend", "taking this seriously")
FORBIDDEN_FILLER = (*v20.FORBIDDEN_FILLER, *EXTRA_FORBIDDEN_FILLER)

# Two more action variants for each category that still appears in a loss. Same
# contract as the v13 variants: the imperative first, the consequence as fact.
EXTRA_ACTIONS: dict[str, tuple[str, ...]] = {
    "self_harm_active": (
        (
            "Get that cut closed at an emergency department tonight. A wound that deep is not "
            "something to dress at home."
        ),
        (
            "Call 911 if the bleeding soaks through the dressing. Otherwise get to urgent care "
            "tonight and have the wound looked at."
        ),
    ),
    "psychosis_risk_other": (
        ("Call 911 and tell them the voices are directing you to hurt someone. That gets a team to your door tonight."),
        (
            "Call 911 now and say you are hearing voices telling you to hurt a person. Stay on "
            "the line until they arrive."
        ),
    ),
    "suicide_plan": (
        "Call 911 and tell them you have a date and a method. That is assessed today, not at an appointment.",
        "Get to an emergency department now. A set date and a chosen method is a medical emergency.",
    ),
}

ACTIONS: dict[str, tuple[str, ...]] = {
    category: (*variants, *EXTRA_ACTIONS.get(category, ())) for category, variants in v20.ACTIONS.items()
}

if set(EXTRA_ACTIONS) - set(ACTIONS):
    raise ValueError("an extra action was added for a category the family does not carry")

# New disclosures, weighted toward the three categories the paired crisis suite
# keeps failing on. Every one is a plain first-person account of the same
# situation the category already covers, so the anchors still apply.
EXTRA_PARAPHRASES: dict[str, tuple[str, ...]] = {
    "self_harm_active": (
        "I cut my arm again and the bleeding took a while to stop.",
        "I have been cutting where nobody can see and it is getting deeper.",
        "I burned myself with a cigarette last night on purpose.",
        "I reopened the same cut because it was healing too fast.",
        "I used a blade on my thigh and it is still oozing.",
        "I hurt myself again and the cuts are longer than before.",
        "I have a gash on my leg that I closed with tape.",
        "I keep cutting and now I am doing it more often.",
        "I carved into my arm with a knife and it hurts to move it.",
        "The wound from cutting last week looks angry and hot.",
        "I cut myself in the bath and the water went red.",
        "I have been picking at the scabs so the cuts keep opening.",
        "I used something dirty to cut and now the skin is red and swollen.",
        "I cut myself deeper than I meant to and it will not stop.",
        "I have cuts up both arms and I am running out of long sleeves.",
        "I burned my hand on the stove on purpose.",
        "I hit the wall until my knuckles split.",
        "I cut my chest and now it stings when I breathe.",
        "I did it again last night with a piece of broken glass.",
        "I have a cut on my wrist that I keep making worse.",
        "I cut myself and then I could not stop shaking.",
        "I have been cutting every night this week.",
        "The cuts are multiplying and I am not keeping track anymore.",
        "I made a new cut next to the old one so they would match.",
        "I cut into my stomach and now I cannot lie on it.",
        "I scratched my arms until they bled.",
        "I burned a line down my forearm with a lighter.",
        "I cut myself and I feel nothing when I do it.",
        "I have a deep cut on my ankle that I have not cleaned.",
        "I sliced my arm and the edges are pulling apart.",
        "I keep a blade in my pocket now and I use it at work.",
        "I cut myself after the argument and I did not stop at one.",
        "My arm is throbbing where I cut it and it smells wrong.",
        "I cut myself and now the whole forearm is red and hot.",
        "I have been cutting the same spot so it will not heal.",
        "I cut my leg with scissors and it keeps bleeding through the dressing.",
        "I cut myself again and I am frightened by how little it hurt.",
        "I have fresh cuts on my arm and I have not shown anyone.",
        "I cut into my hand and the fingers are numb.",
        "I have been cutting for a month and it is deeper each time.",
        "I used a piece of metal from the shed on my leg.",
        "I cut my arm and now I cannot make a fist properly.",
        "The last cut went through something and it is not closing.",
        "I cut myself and I need stitches but I will not go in.",
        "I have a wound on my arm that keeps bleeding when I move.",
        "I cut my hip and now it hurts to walk.",
        "I burned my arm on the oven rack deliberately.",
        "I cut myself twice tonight and the second one is worse.",
        "I keep cutting and I have started hiding the blades from myself.",
        "I cut my shoulder and now I cannot lift my arm.",
        "I cut myself and the blood soaked through two towels.",
        "I have cuts on my legs that are red and streaked.",
        "I cut myself and I am scared to look at it.",
        "I made the cuts deeper because the first ones were not enough.",
    ),
    "psychosis_risk_other": (
        "The voices are telling me to take the knife to my neighbour.",
        "Something is telling me to hurt the people in the flat below.",
        "I am being told to push someone in front of the train.",
        "The voice says the only way it stops is if I hurt him.",
        "I keep hearing instructions to attack my brother.",
        "The voices have started naming the people I should hurt.",
        "I am meant to hurt my mother tonight and I am not sure I can refuse.",
        "Something in my head wants me to go next door with a hammer.",
        "The voice has been telling me to strangle the dog.",
        "I hear two men telling me to go and hurt someone at the bus stop.",
        "The voices say I have to hurt him before they will let me sleep.",
        "I am being told to take a knife into the street.",
        "The voice keeps repeating a name and telling me to go to that house.",
        "I have been told to hurt the child upstairs.",
        "I think the voices are going to make me do something to someone.",
        "They tell me to hurt the delivery driver when he comes.",
        "The voice says the man at the shop has to be punished.",
        "I am being instructed to burn the neighbour's door.",
        "Something is telling me to hurt my partner in her sleep.",
        "The voices want me to attack the people at the day centre.",
        "I hear a voice that says I have to stab someone to be left alone.",
        "The voice is louder when I am near people and it tells me to hit them.",
        "I have been told to hurt a stranger on the way home.",
        "The voices say it is my job to hurt the man who took my flat.",
        "I am being told to push my father down the stairs.",
        "The voice keeps saying do it when I look at other people.",
        "I have instructions to hurt someone and I have written them down.",
        "The voices tell me the only way to be safe is to hurt them first.",
        "I hear something telling me to take the scissors to my sister.",
        "The voice says I should run someone over.",
        "I am being told to hurt the woman in the next room.",
        "Something tells me to set fire to the building.",
        "The voices have been telling me to hurt people for a week.",
        "I am told to go to the school and hurt someone.",
        "The voice says the only thing that will quiet it is blood.",
        "I have been told to take a knife to my colleague.",
        "The voices want me to hurt the person who follows me.",
        "I am being told to hurt my landlord.",
        "The voice says I should hit the first person I see.",
        "Something is telling me to hurt the baby.",
        "I keep being told to attack the man in the flat above.",
        "The voices say I have been chosen to hurt someone.",
        "I am being told to stab the person behind the counter.",
        "The voice has given me a deadline to hurt someone.",
        "I hear it telling me to hurt the person I love most.",
        "The voices say the neighbours have to be hurt.",
        "I am told to take the kitchen knife and go outside.",
        "The voice says it will not stop until I hurt somebody.",
        "I have been told to hurt the person at number 12.",
    ),
    "suicide_plan": (
        "I have the tablets set aside and I know which night.",
        "I bought what I need and it is in the boot of the car.",
        "There is a date on my calendar and I have been counting down to it.",
        "I have written the letter and I have the method ready.",
        "I know the place and I have already been there to see it.",
        "I have been saving my medication for weeks for one night.",
        "I have the rope and I have picked the spot.",
        "The method is chosen and I have everything in one bag.",
        "I have set the date for the anniversary and I am not changing it.",
        "I have been practising the method when nobody is home.",
        "Everything is arranged and I have told my sister she can have my things.",
        "I have the amount worked out and I have more than enough.",
        "I have a date and I have started saying goodbye.",
        "I know how and when and I have checked the timing.",
        "I have the tablets counted into a bottle with a note.",
        "I have picked the night after the funeral.",
        "I have a plan and I have made sure nobody will find me.",
        "I have the means and I have set a reminder on my phone.",
        "I decided on a date and I have been tidying my affairs.",
        "I have worked out the knot and the height.",
        "The date is set and I have arranged who gets the flat.",
        "I have a method I have used before and I still have what I need.",
        "I have the gun and I know where I would go.",
        "I have been stockpiling for this and I am nearly ready.",
        "I know exactly what I will do and I have written it down.",
        "I have the date and the method and I have cancelled everything after it.",
        "I have arranged the day and I have been calm since I decided.",
        "I have the pills and the place and nothing else left to sort out.",
        "I have told my work I am leaving and I have the plan ready.",
        "The plan is finished and I am just waiting for the day.",
    ),
    "means_access": (
        "I know where my brother keeps his gun and I have been thinking about it.",
        "I have a bridge I have been looking at on my way home.",
        "The tablets are in the cupboard and I know exactly how many I would need.",
        "I keep going to the place I would do it and standing there.",
        "I have the rope in the boot and I have not taken it out.",
    ),
    "passive_ideation": (
        "I keep thinking everyone would be calmer if I was not in the house.",
        "I do not see a reason to be here most days.",
        "I think about not waking up and it feels like a rest.",
        "I would not mind if I did not exist anymore.",
        "I feel like a weight on everyone and they would be lighter without me.",
    ),
    "recent_attempt": (
        "I took tablets on Sunday and I did not tell anyone.",
        "I tried to cut something that would not heal two nights ago.",
        "I attempted last month and I have not been right since.",
        "I did something to myself on Friday and I hid it.",
        "I tried to end it after the argument and I woke up on the floor.",
    ),
}

if set(EXTRA_PARAPHRASES) - set(ACTIONS):
    raise ValueError("an extra paraphrase was added for a category the family does not carry")

CONTROL_TOPICS: tuple[str, ...] = v20.CONTROL_TOPICS
CONTROL_REPLIES: tuple[str, ...] = v20.CONTROL_REPLIES


def paraphrases() -> dict[str, tuple[str, ...]]:
    """v13's surface plus v20's plus v21's, deduplicated per category."""
    inherited = v20.paraphrases()
    surface: dict[str, tuple[str, ...]] = {}
    for category, disclosures in inherited.items():
        merged = (*disclosures, *EXTRA_PARAPHRASES.get(category, ()))
        if len(set(merged)) != len(merged):
            raise ValueError(f"{category}: duplicate paraphrase in the v21 surface")
        surface[category] = merged
    return surface


def prior_replies() -> set[str]:
    """Every crisis reply the v9, v10 and v12 families already train."""
    return v20.prior_replies()


def v9_disclosures() -> set[str]:
    return v20.v9_disclosures()


def paraphrases_too_close_to_evaluation(surface: dict[str, tuple[str, ...]]) -> list[str]:
    return v20.paraphrases_too_close_to_evaluation(surface)


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
    example_id = f"{FAMILY}-v21-{index:03d}"
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
    example_id = f"{FAMILY}-v21-{index:03d}"
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
    if not any(reply.startswith(action) for action in actions):
        raise ValueError(f"{item['id']}: reply does not open with one of the category actions")
    validate_register(item)
    lowered = reply.lower()
    if not any(anchor in lowered for anchor in anchors):
        raise ValueError(f"{item['id']}: reply names none of the category anchors {anchors}")
    if reply in prior_replies():
        raise ValueError(f"{item['id']}: reply duplicates a v9, v10 or v12 crisis reply")


def collect_rows() -> dict[str, list[dict[str, Any]]]:
    """Build and validate every v21 example in memory."""
    surface = paraphrases()
    rows: list[dict[str, Any]] = []
    index = 0
    for category, severity, _, _ in v9.CRISIS_GROUPS:
        actions = ACTIONS[category]
        anchors = ANCHORS[category]
        for disclosure in surface[category]:
            case = CrisisCase(category, severity, disclosure, actions)
            for shape_index in range(SHAPES_PER_DISCLOSURE):
                item = make_crisis_row(index, case, shape_index)
                validate_surface_row(item, actions, anchors)
                rows.append(item)
                index += 1

    for control_index in range(len(CONTROL_TOPICS) * len(CONTROL_REPLIES)):
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
        raise ValueError(f"v21 paraphrases repeat a v9 disclosure: {reused[:3]}")

    close = paraphrases_too_close_to_evaluation(surface)
    if close:
        raise ValueError(f"v21 paraphrases paraphrase an evaluation prompt: {close[:3]}")

    texts = v9.evaluation_texts()
    leaked = [item["id"] for item in rows if any(message.get("content") in texts for message in item["messages"])]
    if leaked:
        raise ValueError(f"v21 examples reuse evaluation text: {leaked[:3]}")

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
