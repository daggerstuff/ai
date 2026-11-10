#!/usr/bin/env python3
"""v12 agent-tool data: a crisis disclosure surface that is not phrase-locked.

The v15 paired eval passes eight of the eleven crisis cases on both splits and
fails the same two every time: ``self-harm-active`` and
``eating-disorder-syncope``. Both replies are warm reflections with no routing
action anywhere in them, which is the v13 failure mode coming back for the two
categories where the general therapy corpus has the most to say. v9 and v10
train those categories, so the defect is not missing coverage — it is that 480
and 540 rows of six fixed disclosures each do not outweigh a corpus of therapy
sessions that answer a self-harm disclosure with reflection.

The fix is breadth of surface, not more of the same sentences. Every one of the
ten v9 crisis categories gets twenty paraphrases, written across registers: the
direct disclosure, the minimised one, the euphemistic one, the physical-symptom
one, and the retrospective one. Each paraphrase is crossed with the three
resource-first reply shapes below, and each category carries three action
variants so the reply is not one memorised sentence either.

Nothing here copies an evaluation sentence. The paraphrases cover the semantic
category — a plan with a date and means, an injury already inflicted, a collapse
after days of purging — which is what the held-out cases test, so the suites stay
a genuine generalisation check.

The restraint direction is trained too: seventy-two ordinary complaints, in
topics v9 does not use, keep the family from teaching a hotline script for every
hard week.

Usage::

    PYTHONPATH=/home/vivi/pixelated python -m training.build_agent_tool_sft_v12 --dry-run
    PYTHONPATH=/home/vivi/pixelated python -m training.build_agent_tool_sft_v12
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
import build_agent_tool_sft_v10 as v10

ROOT = base.ROOT
RELEASE_ID = "pixelated-agent-tool-sft-2026-11-08-v12"
DEFAULT_OUTPUT = ROOT / "docs/training/agent-tool-sft-2026-11-08-v12"
BASE_MODEL = v9.BASE_MODEL

FAMILY = "crisis_surface"
SHAPES_PER_DISCLOSURE = 3
PARAPHRASES_PER_CATEGORY = 20

# Resource first, always: the reply's first tokens are the routing action, so a
# partially retrieved pattern still names help. The frames stay short and
# distinct from the v9 and v10 shapes so the family does not teach one script.
SURFACE_SHAPES: tuple[str, ...] = (
    "{action}",
    "{action} I am not going to move past this part.",
    "{action} Stay with me on this until it is done.",
)

if len(SURFACE_SHAPES) != SHAPES_PER_DISCLOSURE:
    raise ValueError("the shape table must stay in step with SHAPES_PER_DISCLOSURE")

# Three action variants per category. Every one satisfies the evaluation rubric
# for its severity: acute categories name a medical or emergency escalation,
# non-acute categories name a crisis referral.
ACTIONS: dict[str, tuple[str, ...]] = {
    "suicide_plan": (
        "Call 911 right now, or have someone take you to an emergency department — a dated plan with "
        "a chosen method has to be assessed today, not talked through.",
        "Call 911 now. A set plan and a chosen method is a medical emergency, not something to talk "
        "through over the next few days.",
        "Call 911 or get to an emergency department tonight and tell them you have a plan and "
        "the means — this needs to be assessed today.",
    ),
    "overdose_taken": (
        "Call 911 right now — what you swallowed keeps working in your body for hours, and this needs "
        "an emergency department tonight.",
        "Call 911 now and say what you took and when; poison control can stay on the line with you until help arrives.",
        "Call 911 or get to an emergency department immediately — an overdose can worsen over hours "
        "even if you feel steady right now.",
    ),
    "self_harm_active": (
        "Get that injury assessed at an emergency department tonight, and call 911 on the way if the "
        "bleeding starts again.",
        "Call 911 if the bleeding will not stop; otherwise get to an emergency department or urgent "
        "care tonight so the wound is assessed.",
        "That injury needs a medical assessment tonight — go to an emergency department or urgent "
        "care, and call 911 if it gets worse on the way.",
    ),
    "psychosis_risk_other": (
        "Call 911 right now and stay on the line with them — a command you are afraid you will follow "
        "is an emergency, and they need to come to you.",
        "Call 911 now and say that you are afraid you will act on what the voices are telling "
        "you; they need to come to you.",
        "Call 911 right now and tell them someone may get hurt tonight — an emergency team needs to be "
        "with you before this goes further.",
    ),
    "assault_strangulation": (
        "Call 911 now. Hands around your throat is a medical emergency and a police matter at the same "
        "time, and both need to happen tonight.",
        "Call 911 now; being choked can cause damage that only shows up hours later, and they "
        "can send officers at the same time.",
        "Call 911 and ask for both an ambulance and the police — strangulation needs to be checked "
        "tonight and you should not be alone with him.",
    ),
    "medical_collapse": (
        "Get an ambulance now — collapsing after days of not keeping food down means your heart and "
        "electrolytes need checking at a hospital tonight, not tomorrow.",
        "Call 911 now; passing out and an irregular heartbeat after days of purging or "
        "restricting is a medical emergency.",
        "Call 911 or have someone drive you to an emergency department immediately — collapse and "
        "heart symptoms at this point are not something to watch at home.",
    ),
    "withdrawal": (
        "Call 911 or get to an emergency department — the first days of alcohol or sedative withdrawal "
        "are when seizures happen, and that needs monitoring you cannot do at home.",
        "Call 911 now; withdrawal from alcohol or sedatives can cause seizures, and that needs a "
        "medical team rather than a quiet room.",
        "Call 911 or get to an emergency department tonight — this is the stage of withdrawal where seizures happen.",
    ),
    "recent_attempt": (
        "Call 911 or go straight to an emergency department — an attempt in the last day needs both "
        "the injury looked at and a crisis assessment before you go home.",
        "Call 911 now, or get to an emergency department — an injury from an attempt needs looking at "
        "even if you think it is not serious.",
        "Call 911 tonight; after an attempt, the medical check and the crisis assessment both "
        "happen at the emergency department.",
    ),
    "passive_ideation": (
        "Call 988 tonight — thinking you would be better off dead is exactly what that line is "
        "for, and they will stay on the phone with you.",
        "Call 988 tonight and say what you have just said to me; they will stay with you, and "
        "we can build a safety plan for the bad hours.",
        "Call 988 now and tell them you have been thinking you would be better off dead; a "
        "crisis line is the right place for this tonight.",
    ),
    "means_access": (
        "Call 988 now, and while you are on the phone with them let us arrange for the means to leave "
        "your home tonight.",
        "Call 988 tonight, and give the means to someone you trust or lock them away — having "
        "the method within reach is what turns thinking into acting.",
        "Call 988 now and tell them what you have been keeping close; then let us get it out of the house tonight.",
    ),
}

# Twenty paraphrases per category, none of them a v9 disclosure and none of them
# an evaluation sentence. Registers vary so the category is learned rather than
# the wording.
PARAPHRASES: dict[str, tuple[str, ...]] = {
    "suicide_plan": (
        "I have the whole thing planned out for next week and I have already given most of my things away.",
        "I keep the rope in the car now so I can do it on the way home from work.",
        "I have been practising with the belt to make sure it would hold.",
        "I chose the day. It is the anniversary, and I have already told my sister I will not be at the dinner.",
        "There is a hotel booked for Thursday and a full bottle of tablets in my bag for it.",
        "I have been cutting the pills in half so I can save enough to do it properly.",
        "I know exactly which exit on the motorway I am going to use and how fast I need to hit the barrier.",
        "I wrote down the steps so I would not lose my nerve halfway through.",
        "I have looked up how much of my medication it takes and I have enough in the house.",
        "The plan is set. I only came here because I promised someone I would talk to a professional once.",
        "I picked out what I would wear and left a note in the drawer for my mother.",
        "I have decided on the date and I have been counting down the days rather than dreading them.",
        "I have been timing how long the garage takes to fill up when the car is running.",
        "I have a place picked out where nobody would find me for a few days.",
        "I already put my savings in an envelope with my sister's name on it.",
        "I have been clearing out my flat so nobody has to sort through my things afterwards.",
        "I know what I am going to do and when, and I am not asking you to stop me, I am telling you "
        "so it is not a surprise.",
        "I have been saving my sleeping tablets in a jar for three weeks and I have enough now.",
        "I have told my work I am taking Friday off because I know what I am doing on Thursday night.",
        "I have set up a time on Sunday when the house will be empty.",
    ),
    "overdose_taken": (
        "I took the rest of my prescription about ten minutes ago and I can feel it starting.",
        "I swallowed a packet of painkillers after we argued and now my ears are ringing.",
        "I crushed up my tablets and drank them with vodka an hour ago.",
        "I have taken twice my dose of everything in the cabinet tonight, on purpose.",
        "I ate a whole box of my mother's tablets because I could not think of anything else.",
        "I took about thirty of my pills before I called you and my mouth is dry.",
        "I drank a bottle of cough syrup and took my sleeping tablets on top of it.",
        "I have just swallowed my whole week of medication at once.",
        "I took an overdose tonight. I am not sure how many because I was not counting.",
        "I found my old antidepressants and took the lot about twenty minutes ago.",
        "I took the pills with a glass of wine and now I cannot keep my eyes open.",
        "My chest feels tight and my heart is going fast — I took far too many of my tablets an hour ago.",
        "I swallowed the contents of two blister packs. My stomach is cramping badly.",
        "I took my whole bottle of iron tablets on purpose. I know that was stupid.",
        "I crushed the tablets into water and drank it maybe half an hour ago.",
        "I have just taken every pill in my flat and I am starting to feel sick.",
        "I took my partner's heart medication — not by mistake, I took it on purpose.",
        "I overdosed at about six this evening and I have been throwing up since.",
        "There are empty packets on the table and I cannot remember how many I took.",
        "I took a large amount of paracetamol about four hours ago and I have not told anyone until now.",
    ),
    "self_harm_active": (
        "I cut my thigh tonight and it will not stop bleeding through the towel.",
        "I have been burning myself with a cigarette lighter and the skin is broken now.",
        "I opened up my arm again and I think it needs stitches.",
        "I hit the wall until my knuckles split and I want to keep going.",
        "I cut deeper than I meant to and I feel faint.",
        "I have a fresh cut on my wrist and no one knows it is there.",
        "I have been scratching at my arm until it bleeds every night this week.",
        "I used a broken glass on my shoulder and it is still oozing.",
        "I cut myself in the bath and the water went red before I got out.",
        "I hit myself in the head when it gets bad, and tonight I did it until I saw spots.",
        "I carved something into my arm with a Stanley knife.",
        "I burned my hand on the hob on purpose and the blister has burst.",
        "I cut again after two months clean and I have done it twice tonight.",
        "The cuts on my stomach are getting infected and I keep adding to them.",
        "I have been cutting where no one can see and one of them is quite deep.",
        "I used the blade again just now and I am still holding it.",
        "I pulled out a chunk of my hair and then started cutting at my scalp.",
        "I cut through the top layer and it looks like it needs looking at.",
        "I have been self-harming for an hour and I cannot make myself stop.",
        "I reopened the same wound for the third time tonight.",
    ),
    "psychosis_risk_other": (
        "There is a voice that keeps telling me to hurt my flatmate and it is getting louder.",
        "I am certain my neighbour has been poisoning me and tonight I am going to deal with him.",
        "Something is instructing me to go to the school at pickup time.",
        "The television has been telling me who to hurt and I have written the name down.",
        "I have not slept in four days and I know my brother has to be stopped before he hurts us.",
        "I have been sharpening a knife because I know someone is coming through the door tonight.",
        "The voices say my daughter is not mine and I keep thinking about making her stop.",
        "I have been told to set the building on fire and I have the petrol in the hall.",
        "I am going to confront the man who has been following me, and I am taking something with me.",
        "God is telling me to punish the people at the office and I think I have to obey.",
        "I am sure my partner is an imposter and I have been thinking about what to do to them.",
        "The radio is giving me orders and the last one was about a stranger on my street.",
        "I have a hammer by the door because the men outside are going to come in and I am ready.",
        "I know it sounds mad, but I am supposed to take my father out before he does it to me.",
        "Something in my head keeps saying that killing the dog will stop the messages.",
        "I have been awake for days and I have decided that my neighbour has to be stopped tonight.",
        "The voices tell me my son is a threat and I have been locking him in his room.",
        "I have been told to drive at the crowd at the market and I keep thinking about it.",
        "I am certain there is a chip in my wife's head and I have been thinking about cutting it out.",
        "The instructions have become very clear and they involve hurting someone at my work.",
    ),
    "assault_strangulation": (
        "He squeezed my throat until I saw black spots and then he let go.",
        "My partner pressed on my windpipe last night and I have been hoarse all day.",
        "He put a pillow over my face until I stopped moving, then he said he was sorry.",
        "I have marks on my neck and he told me it was my fault for winding him up.",
        "He choked me against the fridge and I urinated on myself.",
        "I was strangled in the car and I could not get the door open.",
        "He held me under the bath water for a few seconds and laughed.",
        "My boyfriend wrapped a scarf around my neck when I tried to leave.",
        "He hit me in the ribs with a chair and now it hurts to take a breath.",
        "I have a black eye and he has taken my phone, so I am using the neighbour's.",
        "He threw me into the radiator and my arm is swelling up.",
        "He grabbed me by the hair and banged my head against the wall twice.",
        "He punched me in the stomach and I have been bleeding since.",
        "My husband threatened me with a knife at the table in front of the children.",
        "He pushed me down the steps and now I cannot put weight on my ankle.",
        "He strangled me until I lost control of my bladder and I am still shaking.",
        "I have marks on my throat from Saturday and he says if I tell anyone he will finish it.",
        "He held me down by the neck while he shouted and I could not breathe at all.",
        "He bit my face and I have a wound that is opening again.",
        "He twisted my arm behind my back until it cracked and I have not been to a hospital.",
    ),
    "medical_collapse": (
        "I have been throwing up everything I eat for a week and I fainted on the stairs.",
        "I passed out on the platform this morning and I have not eaten properly in days.",
        "My heart keeps fluttering and I cannot stand up without holding onto something.",
        "I have been taking laxatives every day for a month and now my legs are cramping badly.",
        "I blacked out in the shower and woke up on the tiles.",
        "I have not eaten in nine days and my hair is coming out in handfuls.",
        "I have been vomiting blood since this morning and I cannot keep water down.",
        "I collapsed in the supermarket and the staff called an ambulance.",
        "My blood sugar dropped so low earlier that I could not form words.",
        "I have been restricting and overexercising and this afternoon my vision went grey.",
        "I keep fainting and I hit my head on the sink the last time.",
        "I cannot feel my hands and my heart is doing something irregular.",
        "I have not had anything to drink since Tuesday and I am confused about what day it is.",
        "I purged four times today and now my chest hurts and I am dizzy.",
        "I woke up on the bathroom floor with no memory of getting there.",
        "My ribs are showing and I get out of breath just walking to the kitchen.",
        "I have been water loading and I had a seizure this afternoon.",
        "I stood up and everything went white and I came round on the floor.",
        "I am too weak to get up the stairs and I have been on the landing for an hour.",
        "My heart rate is in the thirties according to my watch and I keep nearly passing out.",
    ),
    "withdrawal": (
        "I stopped the tablets four days ago and now the room will not stay still.",
        "I quit drinking at the weekend and I have been vomiting and sweating since.",
        "I came off the benzos myself and my hands are jumping.",
        "I have not had a drink in two days and there are shadows moving at the edge of the room.",
        "I stopped the painkillers suddenly and I am shaking and crying and I cannot get warm.",
        "I had a fit this morning after stopping the alcohol and I have been confused since.",
        "My heart is racing and I keep seeing things out of the corner of my eye since I stopped.",
        "I went cold turkey three days ago and I am hearing a voice that is not there.",
        "I stopped drinking and now I cannot hold a cup of water steady.",
        "I have been sweating through the sheets and my pulse is over a hundred and thirty.",
        "I ran out of my medication five days ago and I am shaking badly.",
        "I stopped the pills and now I feel like insects are under my skin.",
        "I drank a bottle a day for years, I stopped on Monday, and I am terrified of a seizure.",
        "I came off the opioids and my whole body aches and I keep being sick.",
        "I am hallucinating since I stopped drinking and I cannot tell what is real.",
        "I had a seizure last night because I stopped suddenly and I am on my own.",
        "I stopped the sleeping tablets and I have not slept in three days and I am seeing things.",
        "The withdrawal is unbearable and my heart is skipping beats.",
        "I quit on Sunday and now I am vomiting and my hands shake so much I dropped a glass.",
        "I stopped the alcohol two nights ago and this afternoon I could not remember my own address.",
    ),
    "recent_attempt": (
        "I cut my arm last night and I have not had it looked at.",
        "I took pills on Tuesday and I have been dizzy ever since.",
        "I drove into a tree on Wednesday and told them I swerved for a cat.",
        "I jumped from the balcony last night and I can barely walk today.",
        "I tried to hang myself in the shed and the beam gave way.",
        "I slit my wrist last week and I have been hiding it with a bandage.",
        "I took an overdose on Saturday and I have been throwing up since.",
        "I lay down in front of a car yesterday and someone pulled me back.",
        "I tried to suffocate myself with a bag two nights ago and I stopped.",
        "I cut myself badly on Sunday and it is still open.",
        "I tried to drown myself in the bath last night and I got out.",
        "I jumped off the bridge on Thursday. It was not high enough.",
        "I attempted on Monday and I have been pretending I had a fall.",
        "I took a knife to my stomach last night and I did not go deep enough.",
        "I swallowed a bottle of pills on Sunday and did not tell anyone until now.",
        "I tried last night and I am still bleeding a bit.",
        "I went to the woods on Friday with a rope and I came back.",
        "I cut my thigh last week and it is hot and swollen now.",
        "I put a plastic bag over my head on Tuesday and then took it off.",
        "I tried to end it on Saturday and I have been lying about the cuts since.",
    ),
    "passive_ideation": (
        "I do not see the point in any of it and I keep thinking it would be simpler if I was not here.",
        "I have been hoping I get ill enough that it takes the decision out of my hands.",
        "I would not call it suicidal, but I would not mind if I did not wake up.",
        "My kids would be fine without me, probably better.",
        "I have been thinking about how much easier things would be for everyone if I disappeared.",
        "I keep wondering what the point is of getting up in the morning.",
        "I have been wishing I would get hit by a car so it would not be my fault.",
        "I do not want to be here and I do not know how to say that any plainer.",
        "I feel like I am already gone and nobody has noticed.",
        "Sometimes I think about being dead and it feels restful rather than frightening.",
        "I have been hoping something happens to me so I do not have to decide anything.",
        "I am tired of existing and I have been thinking about stopping.",
        "Nobody would miss me if I was not around, not really.",
        "I do not plan anything, but I do not really want to be alive either.",
        "I have been thinking about my funeral and it is the only thing that calms me down.",
        "I keep thinking it would be a relief not to exist.",
        "I have been looking at my life and thinking everyone would be better off without me in it.",
        "I do not have a plan, but I keep thinking about not being here.",
        "I have been thinking a lot about dying — not doing it, just thinking about it.",
        "I do not see a future for myself and I have stopped caring whether I am here for it.",
    ),
    "means_access": (
        "I have been keeping my husband's service pistol where I can get to it.",
        "There is a pack of my old tablets in my desk and I keep counting them.",
        "I have been looking up the least painful way for weeks now.",
        "I bought a length of rope from the hardware shop and it is under my bed.",
        "I keep driving past the quarry and thinking about it.",
        "I have been saving my painkillers in an envelope in case I need them.",
        "I have been looking at the height of the car park and working out whether it would be enough.",
        "The pills are in my coat pocket and I have been carrying them around all week.",
        "I have been reading about ligature points and I know which one in my flat would take my weight.",
        "My brother's shotgun is in the cupboard and I know where the key is.",
        "I have been measuring out how much of my insulin it would take.",
        "I keep a blade in my wallet now, just in case.",
        "I have been stockpiling my medication in a shoebox for a month.",
        "There is a railway line at the bottom of the garden and I have been timing the trains.",
        "I have been looking at my grandfather's old pistol in the loft more than I should.",
        "I have been buying strong painkillers on the internet and keeping them for later.",
        "The antifreeze is in the garage and I have been thinking about it every night.",
        "I know exactly how many of my tablets I would need and I have enough.",
        "I have been looking at the bridge near my work every day on the way in.",
        "I have been keeping a knife under my pillow since the weekend.",
    ),
}

# Seventy-two ordinary complaints in topics v9 does not use, so the family keeps
# the non-escalation direction trained while the disclosure surface grows.
CONTROL_TOPICS: tuple[str, ...] = (
    "My landlord has not returned my deposit and I do not know what I am allowed to do.",
    "I have two essays due on the same day and I have not started either.",
    "My train has been cancelled every morning this week and I keep arriving late.",
    "The office moved to hot-desking and I cannot find anywhere quiet to work.",
    "My sister borrowed my car and brought it back with a scratch and no explanation.",
    "I have been put on a project I did not ask for and I do not know the tools.",
    "The supermarket stopped stocking the one thing my son will eat.",
    "My phone contract went up by eight pounds and nobody told me.",
    "I have been waiting six weeks for a reply about my council tax.",
    "My team keeps scheduling meetings at eight in the morning.",
    "I cannot get the new software to export the report in the right format.",
    "My flatmate leaves dishes in the sink until they smell.",
)

CONTROL_REPLIES: tuple[str, ...] = (
    "That is a fair thing to be annoyed about. {step} and see whether it becomes manageable.",
    "Understood. {step} — that usually turns a vague irritation into something you can act on.",
    "That sounds like a real constraint rather than a personal failing. {step}.",
    "I would take it one piece at a time. {step}",
    "That would wear anyone down. {step}, and we can narrow it further if it is still too big.",
    "Makes sense. {step} — a workable next step, not a perfect fix.",
)

CONTROL_STEPS: tuple[str, ...] = (
    "Write down the one thing you actually need from them",
    "Give it a fixed twenty-minute slot today and stop when the time is up",
    "Ask one person who has dealt with the same thing what worked",
    "Put the two options side by side on paper",
    "Do the smallest part first and set a timer",
    "Pick the version you can start this week",
)

SEVERITIES: dict[str, str] = {category: severity for category, severity, _, _ in v9.CRISIS_GROUPS}

if set(PARAPHRASES) != set(SEVERITIES):
    raise ValueError("the paraphrase table must cover exactly the v9 crisis categories")
for _category, _paraphrases in PARAPHRASES.items():
    if len(_paraphrases) != PARAPHRASES_PER_CATEGORY:
        raise ValueError(f"{_category}: expected {PARAPHRASES_PER_CATEGORY} paraphrases")
    if len(set(_paraphrases)) != len(_paraphrases):
        raise ValueError(f"{_category}: duplicate paraphrase")


def prior_replies() -> set[str]:
    """Every reply the v9 and v10 families already train."""
    replies: set[str] = v10.v9_replies() | {
        shape.format(action=action) for _, _, _, action in v9.CRISIS_GROUPS for shape in v10.RESOURCE_FIRST_SHAPES
    }
    return replies


def v9_disclosures() -> set[str]:
    return {disclosure for _, _, disclosures, _ in v9.CRISIS_GROUPS for disclosure in disclosures}


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
    reply = SURFACE_SHAPES[shape_index].format(action=action)
    example_id = f"{FAMILY}-v12-{index:03d}"
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
    step = CONTROL_STEPS[control_index % len(CONTROL_STEPS)]
    example_id = f"{FAMILY}-v12-{index:03d}"
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


def validate_surface_row(item: dict[str, Any], actions: tuple[str, ...]) -> None:
    """The v9 crisis contract, plus the resource-first ordering v12 exists to teach."""
    v9.validate_crisis_row(item)
    reply = item["messages"][1]["content"]
    if not any(reply.startswith(action) for action in actions):
        raise ValueError(f"{item['id']}: reply does not open with one of the category actions")
    if reply in prior_replies():
        raise ValueError(f"{item['id']}: reply duplicates a v9 or v10 crisis reply")


def collect_rows() -> dict[str, list[dict[str, Any]]]:
    """Build and validate every v12 example in memory."""
    rows: list[dict[str, Any]] = []
    index = 0
    for category, severity, _, _ in v9.CRISIS_GROUPS:
        actions = ACTIONS[category]
        for disclosure in PARAPHRASES[category]:
            case = CrisisCase(category, severity, disclosure, actions)
            for shape_index in range(SHAPES_PER_DISCLOSURE):
                item = make_crisis_row(index, case, shape_index)
                validate_surface_row(item, actions)
                rows.append(item)
                index += 1

    for control_index in range(len(CONTROL_TOPICS) * len(CONTROL_REPLIES)):
        item = make_control_row(index, control_index)
        v9.validate_crisis_row(item)
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
        raise ValueError(f"v12 paraphrases repeat a v9 disclosure: {reused[:3]}")
    reused_controls = [item["id"] for item in rows if item["messages"][0]["content"] in set(v9.CONTROL_TOPICS)]
    if reused_controls:
        raise ValueError(f"v12 controls repeat a v9 control topic: {reused_controls[:3]}")

    texts = v9.evaluation_texts()
    leaked = [item["id"] for item in rows if any(message.get("content") in texts for message in item["messages"])]
    if leaked:
        raise ValueError(f"v12 examples reuse evaluation text: {leaked[:3]}")

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
            "v16 addendum: v15 fails the same two crisis cases on both splits — self-harm-active and "
            "eating-disorder-syncope — with a warm reflection that names no resource, because six fixed "
            "disclosures per category do not outweigh a therapy corpus that answers those disclosures "
            "reflectively; v12 crosses twenty paraphrases per category with three resource-first shapes"
        ),
        "source": "first-party deterministic synthetic trajectories; no external or clinical records",
        "base_model": BASE_MODEL,
        "example_count": summary["example_count"],
        "scenario_families": {FAMILY: summary["example_count"]},
        "crisis_severity_counts": summary["crisis_severity_counts"],
        "crisis_category_counts": summary["crisis_category_counts"],
        "distinct_disclosures": summary["distinct_disclosures"],
        "distinct_replies": summary["distinct_replies"],
        "split_rule": "train only, matching the v3..v11 addendum convention",
        "splits": {name: files[name] for name in files},
        "tools": {},
        "design_notes": {
            "surface_breadth": (
                "twenty paraphrases per v9 crisis category, written across the direct, minimised, "
                "euphemistic, physical-symptom and retrospective registers"
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
                "72 ordinary complaints in topics v9 does not use, so the family does not teach a "
                "hotline script for every hard week"
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
