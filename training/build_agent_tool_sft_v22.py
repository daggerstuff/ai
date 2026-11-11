"""v22 addendum: the crisis family at a mass the corpus cannot outvote.

v20 was the best adapter on the board (52/52 against base and 52/52 against
v19, crisis 11/11 on both splits) and it won that way with 1,760 crisis rows,
2.1% of an 83,631-row train. v21 tried to reach the same place from the other
side: it kept the v20 family, added 592 rows to it, and deleted two coddling
phrases from the carried corpus. It lost. The crisis aggregate fell from v20's
84/88 to 73/88, and the two persistent failures are instructive:

    suicide-plan  "You've been holding onto a plan to end your life... I'm not
                   going to jump to a bunch of questions. I want to stay with you
                   right now. What's it like to say that out loud?"
    self-harm     "That's a lot to carry alone. I'm not going to lecture you
                   about stopping. Right now, let's get you somewhere safe..."

Neither sentence is in the addendum. Both are in the corpus. v21 deleted "not
going to pretend" 394 times and the model substituted "not going to jump",
which the corpus already teaches 38 times. Deleting 592 sentences out of 448k
assistant turns moves the register by 0.13%: a phrase-level strip cannot shift a
habit that is thousands of occurrences deep.

v22 therefore does not strip anything. It keeps the v20 corpus exactly as v17
and v19 carried it and attacks the ratio directly:

* **Eight frames instead of four.** Every disclosure is trained against eight
  deadline shapes rather than four, so the same 698 disclosures cover 5,584
  rows. The four new frames are still action-first: they add urgency, never a
  reaction to the disclosure.
* **Eight actions per category instead of three or five.** 44 new hand-written
  action lines, each of which opens with the imperative, names an anchor from
  its category's rubric vocabulary, asks nothing, and states the consequence as
  fact. The family is no longer three sentences repeated at volume.
* **146 new disclosures**, weighted to the categories the paired crisis suite
  keeps failing on (self_harm_active +25, psychosis_risk_other +25,
  suicide_plan +18), so a disclosure phrased outside the v21 paraphrases still
  lands inside the family.

The result is 5,728 rows — 6.8% of an 84k train, up from v20's 2.1%. That is the
lever v21 got wrong: the register is a question of how much of the train teaches
it, not which strings it contains.

Register rules, anchors and forbidden filler are inherited from v20/v21
unchanged. Rows are train-only, user/assistant, empty tool catalog.
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
import build_agent_tool_sft_v21 as v21

ROOT = base.ROOT
RELEASE_ID = "pixelated-agent-tool-sft-2026-11-10-v22"
DEFAULT_OUTPUT = ROOT / "docs/training/agent-tool-sft-2026-11-10-v22"
BASE_MODEL = v21.BASE_MODEL

FAMILY = v21.FAMILY
SHAPES_PER_DISCLOSURE = 8
ACTIONS_PER_CATEGORY = 8

# The four frames v22 adds to v20's. Same contract: the action is emitted
# first and the frame only says how soon. None of them reacts to the
# disclosure, asks a question, or softens the instruction.
EXTRA_SHAPES: tuple[str, ...] = (
    "{action} Do that first, before anything else.",
    "{action} Do it now rather than after one more night.",
    "{action} Tonight, not at the end of the week.",
    "{action} That is the next thing you do.",
)

DIRECT_SHAPES: tuple[str, ...] = (*v21.DIRECT_SHAPES, *EXTRA_SHAPES)

if len(DIRECT_SHAPES) != SHAPES_PER_DISCLOSURE:
    raise ValueError("the shape table must stay in step with SHAPES_PER_DISCLOSURE")

FORBIDDEN_FILLER: tuple[str, ...] = v21.FORBIDDEN_FILLER
ANCHORS: dict[str, tuple[str, ...]] = v21.ANCHORS

# Five more variants for the seven categories that only carried three, three
# more for the three that carried five. Every line opens with the imperative,
# names an anchor from the category's rubric vocabulary, asks nothing, and
# states the consequence as fact.
EXTRA_ACTIONS: dict[str, tuple[str, ...]] = {
    "suicide_plan": (
        "Call 911 and say you have a date and a method worked out. They assess that tonight, not next week.",
        (
            "Get to an emergency department now and take whatever you have been saving with you. "
            "A date and a method is a medical emergency."
        ),
        (
            "Call 911 now. A plan with a method and the pills set aside is a medical emergency, "
            "and it gets seen tonight."
        ),
    ),
    "overdose_taken": (
        "Call 911 now and say what you swallowed. The dose keeps working after the nausea stops.",
        "Call poison control and 911 now. Pills taken in a handful are an overdose that needs a department tonight.",
        "Call 911 now and stay on the line. Sleeping is when that dose stops your breathing.",
        (
            "Get to an emergency department now, or call an ambulance if you are too dizzy to walk. "
            "Swallowed pills keep being absorbed for hours."
        ),
        "Call 911 now and say it was an overdose. Do not wait to see how it goes.",
    ),
    "self_harm_active": (
        (
            "Get that cut closed tonight at an emergency department. A wound that keeps opening needs "
            "stitches, not another dressing."
        ),
        "Call 911 if the bleeding will not stop, and get the wound looked at either way tonight.",
        (
            "Get to urgent care or an emergency department tonight and show them the cut. Deeper than "
            "you meant to go is a wound that needs closing."
        ),
    ),
    "psychosis_risk_other": (
        (
            "Call 911 and say you are hearing voices that tell you to hurt someone. That is the sentence "
            "that gets a team moving."
        ),
        "Call 911 now and tell them the voices have named a person. Do not wait for the voices to get louder.",
        (
            "Get away from the person the voices named and call 911. Being alone with someone you have been "
            "told to hurt is the risk."
        ),
    ),
    "assault_strangulation": (
        "Call 911 now and say you were strangled. Damage to the throat and neck shows up hours later.",
        "Get out of the house and call 911. Hands on the neck is strangulation, and it gets assessed tonight.",
        "Call 911 and ask for an ambulance. Being choked can close the airway after you feel fine.",
        (
            "Leave and call 911 from somewhere he cannot reach you. Strangulation is the strongest predictor "
            "of being killed later, and it gets reported tonight."
        ),
        "Call 911 now and tell them you could not breathe. That is strangulation, and it needs a doctor tonight.",
    ),
    "medical_collapse": (
        (
            "Call an ambulance now. You blacked out because you have not eaten, and that gets your heart "
            "checked tonight."
        ),
        "Get to an emergency department now. Purging and fainting in the bathroom is how electrolytes crash.",
        "Call 911 now and say you fainted. Five days of not eating is a medical emergency, not a diet.",
        (
            "Get to an emergency department by ambulance if you cannot stand. The bathroom purging and the "
            "blackouts are one problem and it gets assessed tonight."
        ),
        "Call an ambulance now. Fainting and an irregular heartbeat after purging needs a monitor tonight.",
    ),
    "withdrawal": (
        ("Call 911 now. Alcohol withdrawal causes seizures in the first days and needs a medical team tonight."),
        ("Get to an emergency department now. Shaking and seeing things is withdrawal that cannot be managed at home."),
        "Call 911 or get to a detox unit tonight. Stopping alcohol on your own is the window where it seizes.",
        (
            "Get to an emergency department now and tell them how much you drink and when you stopped. "
            "Withdrawal gets worse in the first three days."
        ),
        "Call 911 now. Hallucinating during withdrawal means it has gone past what a quiet room can handle.",
    ),
    "recent_attempt": (
        "Call 911 now. An attempt last night needs the injury assessed and a crisis assessment before you sleep.",
        (
            "Get to an emergency department now and tell them you tried yesterday. Both the injury and the "
            "risk get assessed tonight."
        ),
        "Call 911 or get to an emergency department tonight. An attempt overnight is still an open medical risk.",
        ("Get to an emergency department now. What you tried last night can still be doing damage you cannot feel."),
        "Call 911 now and say you made an attempt. That gets the injury and the risk looked at tonight.",
    ),
    "passive_ideation": (
        (
            "You have been thinking people would be better off if you were not around. Say that out loud to "
            "someone in the house tonight and name them."
        ),
        ("You have been hoping you do not wake up. Say that to a person in the room tonight and say their name."),
        "You do not see a reason to live. Say that to a person in the room tonight and name them.",
        "Feeling like a burden is the thing we deal with tonight. Say who you would call and what you would say.",
        "You feel like a burden to everyone in the house. Say that out loud tonight and name who is there.",
    ),
    "means_access": (
        "The pills leave the house tonight. Hand them to someone else and tell me their name.",
        "The gun goes to someone else tonight, not into a cupboard. Tell me who has it before you sleep.",
        "Get the medication out of the house tonight and tell me who is holding it.",
        (
            "You have been keeping a method within reach and that ends tonight. Give it to a person and tell "
            "me their name."
        ),
        "The means go to someone else before you sleep. Tell me who is taking them and where they are going.",
    ),
}

ACTIONS: dict[str, tuple[str, ...]] = {
    category: (*variants, *EXTRA_ACTIONS.get(category, ())) for category, variants in v21.ACTIONS.items()
}

if set(EXTRA_ACTIONS) - set(ACTIONS):
    raise ValueError("an extra action was added for a category the family does not carry")

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

# New disclosures, weighted toward the categories the paired crisis suite keeps
# failing on. Every one is a plain first-person account of the same situation
# the category already covers, so the anchors still apply.
EXTRA_PARAPHRASES: dict[str, tuple[str, ...]] = {
    "self_harm_active": (
        "I opened up my arm with a craft knife after everyone went to bed.",
        "There is a cut on my forearm that I have been hiding under a bandage for a week.",
        "I burnt a row of marks into my leg with a heated spoon.",
        "My thigh is covered in cuts and the newest one has not closed.",
        "I went at my wrist with a broken tile and it bled through my sleeve.",
        "I keep a razor blade taped under my desk and I use it at lunch.",
        "The wound on my hip is red and hot and I have not shown anyone.",
        "I cut into my arm until the skin parted and then I stopped.",
        "I have been burning myself with matches when the day gets too loud.",
        "There are fresh cuts on both forearms and I cannot hide them under a shirt.",
        "I slashed my calf with a kitchen knife and it is still seeping.",
        "I cut myself in the car park at work and drove home with blood on my hand.",
        "The cut on my ankle has gone yellow at the edges and it smells.",
        "I picked the scab off again so the cut would open back up.",
        "I used the edge of a can on my arm and now there is a long wound.",
        "I cut myself twice this week and both times I went further than I planned.",
        "My stomach has a cut on it that rubs against my waistband all day.",
        "I have been cutting since the anniversary and it is nearly every night.",
        "I pressed a hot pan against my forearm on purpose and it has blistered.",
        "I cut into the back of my hand and the bleeding took an hour to slow.",
        "There is a gash on my shin that I have been covering with a sock.",
        "I cut myself and then sat on the bathroom floor until the blood dried.",
        "I have a wound on my arm that is wider than the ones before it.",
        "I keep cutting the same arm and now the scars are raised and tight.",
        "I hit my head against the wall until the skin split and I did it on purpose.",
    ),
    "psychosis_risk_other": (
        "A voice told me this morning that the man in the shop has to be hurt.",
        "I keep hearing a woman's voice telling me to hurt my son.",
        "Something is telling me to take a knife to the neighbour's dog.",
        "The voices have started telling me to hurt the people I work with.",
        "I am being told to put my hands around my wife's neck.",
        "There is a voice that says the only way to sleep is to hurt someone.",
        "I hear instructions to take the hammer to the man at number nine.",
        "The voices say my daughter is not my daughter and that I should hurt her.",
        "I have been told to push the pram into the road.",
        "A voice keeps telling me to hurt the person who sits behind me on the bus.",
        "I am hearing a command to cut the postman when he comes up the path.",
        "The voices want me to hurt someone before the week is out.",
        "Something tells me to hurt the man who watches me from the window.",
        "I have been hearing that I am supposed to hurt my grandfather.",
        "The voice says I should take the scissors and go and see my neighbour.",
        "I keep hearing a man telling me to hurt the child in the flat below.",
        "The voices say I have been given permission to hurt someone tonight.",
        "I am being told to set fire to the car outside.",
        "I hear two voices arguing about who I should hurt first.",
        "The voice tells me the only thing that will make it stop is hurting someone.",
        "I have been told to take a knife to my sister's house.",
        "Something is telling me to hurt the nurse at the clinic tomorrow.",
        "The voices say my flatmate is a danger and that I should deal with him.",
        "I keep hearing that I have to hurt the person in the next seat.",
        "The voice has been telling me to strangle someone since Monday.",
    ),
    "suicide_plan": (
        "I have the tablets in a jar with the day written on the lid.",
        "There is a place I have chosen and I have been there twice to check it.",
        "I have the method decided and the rope is already in the garage.",
        "I know which night and I have started saying goodbye to people.",
        "I have been saving my medication in a drawer for the end of the month.",
        "The date is set for after my sister's birthday and I will not move it.",
        "I have written out the steps and I have everything I need.",
        "I have been rehearsing the method in my head every night.",
        "I picked the spot and the time and I have told nobody.",
        "I have the pills counted into a bottle and a note folded beside it.",
        "I have worked out how many it would take and I have that many.",
        "There is a date in my diary and I have cancelled everything after it.",
        "I have the means in the boot of the car and a place picked out.",
        "I decided the method weeks ago and I have been calm ever since.",
        "I have the date and the method and I am only waiting now.",
        "I have been stockpiling tablets since the spring for one night.",
        "The plan is finished and I have arranged what happens to my things.",
        "I have a method and a date and I have already written the letters.",
    ),
    "overdose_taken": (
        "I took a whole strip of my sleeping tablets about twenty minutes ago.",
        "I swallowed everything in the medicine cabinet and I am starting to feel heavy.",
        "I have taken a bottle of my mother's painkillers and my ears are ringing.",
        "I drank half a bottle of something from the garage and my throat is burning.",
        "I took my whole month of tablets in one go and I am very sleepy.",
        "I swallowed a handful of pills and now I cannot focus.",
        "I have taken an overdose and I have been sick twice.",
        "I emptied the blister packs into my hand and swallowed them.",
        "I took more of my heart medication than prescribed and I feel faint.",
        "I have swallowed something corrosive and my mouth and throat hurt.",
        "I took a double dose of everything in the cupboard about an hour ago.",
        "I have taken enough tablets that my vision is going blurry.",
    ),
    "assault_strangulation": (
        "My partner put both hands around my neck and I could not get air.",
        "He choked me until I saw black and then he let go.",
        "I was strangled last night and my voice has been rough all day.",
        "He held me by the throat against the wall and I could not speak.",
        "My throat is bruised and it hurts to swallow after what he did.",
        "He pressed on my neck until I stopped struggling.",
        "I could not breathe when he had his hands on my throat.",
        "He strangled me in the kitchen and then made me dinner.",
        "There are marks on my neck from his hands and I have been wearing a scarf.",
        "He put me in a choke hold until I nearly passed out.",
        "My neck hurts and my voice is gone since he grabbed me by the throat.",
        "He squeezed my throat and told me to stop talking.",
    ),
    "medical_collapse": (
        "I have not eaten in four days and I passed out on the kitchen floor.",
        "I blacked out getting off the bus this afternoon.",
        "I have been purging after every meal and I fainted in the shower.",
        "I cannot keep water down and I nearly collapsed on the stairs.",
        "I purged four times today and my heart is fluttering.",
        "I have been restricting for a week and I blacked out at my desk.",
        "I fainted in the bathroom and woke up on the tiles.",
        "I have not eaten since Sunday and my hands are going numb.",
        "I threw up everything I ate and then I could not stand up.",
        "I blacked out twice today and my vision keeps going grey.",
        "I have been purging daily and I collapsed in the garden.",
        "I cannot eat and I fainted at the top of the stairs.",
    ),
    "withdrawal": (
        "I stopped drinking four days ago and the shaking has got worse.",
        "I have not had a drink in three days and I am sweating through my shirt.",
        "I am seeing shadows at the edge of the room since I stopped drinking.",
        "My hands are shaking so much I cannot hold a cup.",
        "I quit alcohol last week and I have started hearing a radio that is not on.",
        "I am coming off sleeping tablets and my skin is crawling.",
        "I stopped drinking and now my heart is racing and I cannot sit still.",
        "I have been withdrawing from alcohol and I had a seizure yesterday.",
        "I am shaking and sweating and I keep seeing things move.",
        "I ran out of my medication and the withdrawal is unbearable.",
        "I stopped drinking on Monday and I cannot keep food down.",
        "I am detoxing at home and I have started hallucinating at night.",
    ),
    "recent_attempt": (
        "I took a handful of tablets on Thursday and told nobody.",
        "I tried to hang myself two nights ago and the belt snapped.",
        "I cut my wrists on Saturday and then bandaged them myself.",
        "I attempted last week and I have been dizzy ever since.",
        "I took an overdose on Monday and I did not go to hospital.",
        "I tried to end it in the car and someone knocked on the window.",
        "I did something to myself three days ago and I have not been checked.",
        "I swallowed pills on Sunday night and slept for a day afterwards.",
        "I tried to jump off the car park roof last month and I climbed back down.",
        "I attempted in the bathroom last night and I cleaned up after myself.",
    ),
    "passive_ideation": (
        "I keep thinking my family would be better off without me in the house.",
        "I do not want to be around for another birthday.",
        "I have been hoping I do not wake up in the morning.",
        "Most days I do not see a reason to keep going.",
        "I feel like a burden to my sister and I think about that at night.",
        "I would rather not exist than keep doing this.",
        "I have stopped seeing anything worth living for.",
        "I think everyone would be calmer if I was not being around.",
        "I keep wishing I would not wake up when I go to sleep.",
        "I feel like dead weight to everyone who helps me.",
    ),
    "means_access": (
        "I know where the tablets are kept and I have been counting them.",
        "There is a shotgun in the loft and I have been up there twice.",
        "I have been looking at a bridge near my flat every night.",
        "The medication is in my bedside drawer and I know how many would do it.",
        "I have a method in mind and the thing I need is in the shed.",
        "I keep the means in a box under the bed and I take it out at night.",
        "I know a place where nobody would find me and I have the means ready.",
        "I have been checking how much paracetamol is in the cupboard.",
        "The rope is in the boot of my car and I have not thrown it away.",
        "I have access to my father's pills whenever I want them.",
    ),
}

if set(EXTRA_PARAPHRASES) - set(ACTIONS):
    raise ValueError("an extra paraphrase was added for a category the family does not carry")

CONTROL_TOPICS: tuple[str, ...] = v21.CONTROL_TOPICS
CONTROL_REPLIES: tuple[str, ...] = v21.CONTROL_REPLIES


def paraphrases() -> dict[str, tuple[str, ...]]:
    """v21's surface plus v22's, deduplicated per category."""
    inherited = v21.paraphrases()
    surface: dict[str, tuple[str, ...]] = {}
    for category, disclosures in inherited.items():
        merged = (*disclosures, *EXTRA_PARAPHRASES.get(category, ()))
        if len(set(merged)) != len(merged):
            raise ValueError(f"{category}: duplicate paraphrase in the v22 surface")
        surface[category] = merged
    return surface


def prior_replies() -> set[str]:
    """Every crisis reply the v9, v10, v12 and v21 families already train."""
    return v21.prior_replies()


def v9_disclosures() -> set[str]:
    return v21.v9_disclosures()


def paraphrases_too_close_to_evaluation(surface: dict[str, tuple[str, ...]]) -> list[str]:
    return v21.paraphrases_too_close_to_evaluation(surface)


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
    example_id = f"{FAMILY}-v22-{index:04d}"
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
    example_id = f"{FAMILY}-v22-{index:04d}"
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
        raise ValueError(f"{item['id']}: reply duplicates a v9, v10, v12 or v21 crisis reply")


def collect_rows() -> dict[str, list[dict[str, Any]]]:
    """Build and validate every v22 example in memory."""
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
        raise ValueError(f"v22 paraphrases repeat a v9 disclosure: {reused[:3]}")

    close = paraphrases_too_close_to_evaluation(surface)
    if close:
        raise ValueError(f"v22 paraphrases paraphrase an evaluation prompt: {close[:3]}")

    texts = v9.evaluation_texts()
    leaked = [item["id"] for item in rows if any(message.get("content") in texts for message in item["messages"])]
    if leaked:
        raise ValueError(f"v22 examples reuse evaluation text: {leaked[:3]}")

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
