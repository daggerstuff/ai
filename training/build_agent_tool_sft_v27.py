"""v27 addendum: answer the object the person named, in every crisis category.

v25 and v26 grounded three of the ten crisis categories — ``means_access``,
``overdose_taken`` and ``recent_attempt`` — and left seven with a flat anchor
list and one action surface that had to fit every disclosure in the category.
The cost of that is measurable. Across the seven untouched categories, **4,192
of 4,528 disclosure-by-action pairs (92.6%) name an anchor the disclosure never
gave**:

===================  ============  ===================  =======  =========  ======
category             disclosures   anchor-free discl.    pairs    invented   rate
===================  ============  ===================  =======  =========  ======
suicide_plan                    86                   60      688        634   92.2%
self_harm_active               145                   41     1160       1110   95.7%
psychosis_risk_other           124                   51      992        834   84.1%
assault_strangulation           48                   20      384        368   95.8%
medical_collapse                54                   27      432        411   95.1%
withdrawal                      48                   18      384        376   97.9%
passive_ideation                61                   38      488        459   94.1%
===================  ============  ===================  =======  =========  ======

Two of those rows are the same defect the earlier rounds fixed, and one is
worse than the table suggests.

**The action surface had to invent, because the anchor list was too narrow to
say anything else.** ``self_harm_active``'s anchors are ``cut``, ``deeper``,
``wound``, ``harm``, ``scar``, so no action can say "burn" or "knuckles" or
"scratches", and all eight actions describe a cut. The disclosure "I have been
burning myself with a cigarette lighter and the skin is broken now" is answered
with "Get that cut closed at an emergency department tonight". The person never
described a cut. ``medical_collapse`` is the same shape: a disclosure about a
fluttering heart and legs that will not hold is answered with "You blacked out
in the bathroom", a bathroom the person never mentioned.

**The anchor lists were copied from the evaluation suite.** Seven of the ten
categories carry anchors that are verbatim terms from the paired-eval checks —
``medical_collapse`` is anchored on ``blacked out``, ``blackout``, ``bathroom``,
``purge``, ``five days``; ``psychosis_risk_other`` on ``voices``, ``hurt``,
``4b``, ``downstairs``; ``withdrawal`` on ``drink``, ``shak``, ``seeing things``.
The actions were then written to hit those terms. That is the surface fitting the
suite rather than the disclosure, and it is why 84 to 98 percent of the pairs in
those categories are wrong for the person in front of them.

v27 replaces flat anchor matching with **clusters** in all seven categories. A
cluster is one mechanism, sign or object — ``burn``, ``wall``, ``strangle``,
``alcohol``, ``purge``, ``not_wake_up`` — carried by a tuple of tokens. A
disclosure's clusters are the ones it mentions; an action is eligible only when
**the clusters it names are a subset of the clusters the disclosure named**. A
burn disclosure can therefore only be answered by a burn action, a disclosure
naming a knife can only be answered by the knife action, and a disclosure that
names nothing specific falls to the category's one generic action rather than to
whichever action happened to be written for the eval case.

The rule is enforced twice. ``action_pool`` builds the pool from clusters, so an
inventing action is never a candidate; ``validate_grounded`` re-checks every
built row and fails the build if a reply names a cluster the disclosure did not.
Every disclosure must name at least one cluster, and the abstract-disclosure cap
is zero in six of the seven — so a disclosure that describes a mechanism without
naming it is rewritten to name it, exactly as v25 and v26 did for the means.
``psychosis_risk_other`` is the one exception: 37 disclosures name a target the
voice picked without ever saying who, so the cap there is pinned at 37 rather
than zero, and a new unnamed-target disclosure fails the build loudly instead of
quietly joining the set.

The action surfaces grow to match: 13 actions for ``suicide_plan``, 15 for
``self_harm_active``, 13 for ``psychosis_risk_other``, 13 for
``assault_strangulation``, 11 for ``medical_collapse``, 8 for ``withdrawal`` and
8 for ``passive_ideation``, each naming the cluster it answers and nothing else.
A handful name two clusters where the clinical answer has to cover both: bleeding
that will not stop alongside the cut, vomiting blood alongside the purge, and
every psychosis action that carries the voice alongside the target it named.

Inherited unchanged from v26: the eight reply frames (three of which lead with a
reflection), the control rows, the strict gate at build time, the user/assistant
shape, and the three categories v25 and v26 already grounded.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_agent_tool_sft_v9 as v9
import build_agent_tool_sft_v26 as v26

RELEASE_ID = "pixelated-agent-tool-sft-2026-11-10-v27"
FAMILY = v26.FAMILY
DEFAULT_OUTPUT = v26.DEFAULT_OUTPUT
SHAPES_PER_DISCLOSURE = v26.SHAPES_PER_DISCLOSURE
DIRECT_SHAPES = v26.DIRECT_SHAPES
ESCALATION_TERMS = v26.ESCALATION_TERMS
FORBIDDEN_FILLER = v26.FORBIDDEN_FILLER
ACTIONS_PER_CATEGORY = v26.ACTIONS_PER_CATEGORY

# One cluster per mechanism, sign or object. A token matches on a word boundary,
# so "can" would not match "cannot" and "red" would not match "hundred".
ANCHOR_GROUPS: dict[str, dict[str, tuple[str, ...]]] = {
    "suicide_plan": {
        "pills": (
            "pills",
            "pill",
            "tablets",
            "tablet",
            "medication",
            "overdose",
            "jar",
            "stockpiling",
            "saving",
            "saved",
            "counted",
        ),
        "rope": ("rope", "ligature", "knot", "beam", "hang"),
        "belt": ("belt",),
        "gun": ("gun", "firearm", "pistol"),
        "height": ("height", "roof", "balcony", "cliff", "car park", "jump", "bridge"),
        "vehicle": (
            "motorway",
            "barrier",
            "lorry",
            "crash",
            "train",
            "level crossing",
            "into a tree",
            "the road",
            "exhaust",
            "garage",
            "car is running",
        ),
        "date": (
            "date",
            "the day",
            "night",
            "friday",
            "sunday",
            "monday",
            "saturday",
            "thursday",
            "anniversary",
            "next week",
            "diary",
            "set for",
            "funeral",
            "birthday",
            "month",
            "spring",
            "arranged",
            "the fourteenth",
        ),
        "note": (
            "note",
            "letter",
            "letters",
            "wrote",
            "written",
            "goodbye",
            "given away",
            "my things",
            "affairs",
            "savings",
            "envelope",
            "name on it",
        ),
        "plan": (
            "method",
            "means",
            "worked out",
            "worked it out",
            "plan",
            "planned",
            "rehearsing",
            "rehearsed",
            "practising",
            "steps",
            "decided",
            "what I need",
            "the thing I need",
            "collecting",
            "bought",
            "how and when",
            "where and when",
            "what I am going to do",
            "checked the timing",
        ),
        "place": ("place", "spot", "location"),
    },
    "self_harm_active": {
        "cut": (
            "cut",
            "cuts",
            "cutting",
            "slice",
            "sliced",
            "slash",
            "slashed",
            "gash",
            "carve",
            "reopened",
            "carved",
            "blade",
            "knife",
            "razor",
            "glass",
            "tile",
            "scissors",
            "stanley",
            "metal",
            "edge of a can",
            "craft knife",
            "opened up",
            "reopen",
            "went at",
            "wound",
            "wounds",
        ),
        "burn": (
            "burn",
            "burns",
            "burnt",
            "burned",
            "burning",
            "cigarette",
            "lighter",
            "matches",
            "oven",
            "stove",
            "pan",
            "spoon",
            "blistered",
            "blister",
        ),
        "wall": ("wall", "knuckles"),
        "head_strike": ("my head", "your head", "in the head", "saw spots", "see spots"),
        "scratch": ("scratch", "scratching", "scratched", "nails", "picked", "picking", "scab", "scabs"),
        "infection": (
            "hot to touch",
            "angry",
            "smells",
            "smell",
            "swollen",
            "weeping",
            "oozing",
            "yellow",
            "pus",
            "streaked",
            "throbbing",
            "red and hot",
            "stings",
        ),
        "blood": ("blood", "bleeding", "bled", "bleeds", "stitches", "soaked", "seeping", "stop the blood"),
        "harm": (
            "harm",
            "hurt myself",
            "hurt yourself",
            "hurting myself",
            "hurting yourself",
            "self-harming",
            "self harm",
            "self-harmed",
            "cannot make myself stop",
            "did it again",
            "keep going",
            "deeper",
        ),
        "not_stopping": (
            "do not want help",
            "do not want to stop",
            "don't really want to stop",
            "do not really want to stop",
            "no intention of stopping",
            "not going to stop",
            "stopped wanting to stop",
            "do not want anyone to stop me",
        ),
    },
    "psychosis_risk_other": {
        "voices": (
            "voices",
            "voice",
            "hearing",
            "hear",
            "hearing a",
            "told me",
            "telling me",
            "tell me",
            "instructions",
            "command",
            "says",
            "is telling me",
            "instructing",
            "have been told",
            "in my head keeps saying",
            "television",
            "radio",
        ),
        "harm": ("hurt", "harm", "attack", "stab", "punish", "kill", "damage", "blood", "deal with", "stopped"),
        "housemate": (
            "flatmate",
            "housemate",
            "neighbour",
            "next door",
            "flat above",
            "flat below",
            "the flat",
            "number 12",
            "number nine",
            "landlord",
            "the hall",
            "across the hall",
            "following me",
            "who follows me",
            "building",
            "downstairs",
            "upstairs",
            "4b",
            "the next room",
            "behind the wall",
        ),
        "family": (
            "brother",
            "sister",
            "daughter",
            "son",
            "wife",
            "husband",
            "mother",
            "father",
            "stepfather",
            "grandfather",
            "partner",
            "family",
            "my child",
            "the baby",
        ),
        "pet": ("dog", "cat", "the dog", "the cat", "puppy", "kitten"),
        "public": (
            "stranger",
            "strangers",
            "first person",
            "crowd",
            "the market",
            "bus",
            "train",
            "school",
            "shop",
            "street",
            "driver",
            "postman",
            "nurse",
            "colleague",
            "work",
            "day centre",
            "clinic",
            "counter",
            "shift",
            "the pram",
        ),
        "knife": ("knife", "blade", "scissors"),
        "hammer": ("hammer",),
        "fire": ("fire", "petrol", "burn"),
        "strangle": ("strangle", "neck", "hands around", "hands on", "choke"),
        "vehicle": ("run someone over", "in front of the train", "into the road", "platform"),
    },
    "assault_strangulation": {
        "strangle": (
            "throat",
            "strangle",
            "strangled",
            "strangling",
            "strangulation",
            "choke",
            "choked",
            "throttled",
            "windpipe",
            "neck",
            "hands round",
            "hands around",
            "choke hold",
            "could not breathe",
            "over my mouth and nose",
        ),
        "ribs": ("ribs", "chair", "hurts to take a breath"),
        "limb": ("arm", "ankle", "leg", "steps", "radiator", "weight on", "swelling", "cracked"),
        "head_blunt": (
            "banged my head",
            "head banged",
            "head against the wall",
            "banged against the wall",
            "black eye",
            "by the hair",
        ),
        "stomach": ("stomach", "punched me", "bleeding since"),
        "weapon": ("knife",),
        "wound": ("bite", "wound", "opening again"),
        "submersion": ("bath water", "under the water"),
        "smother": ("pillow", "over my face", "stopped moving", "until I stopped"),
        "threat": (
            "will finish it",
            "would finish it",
            "finish it",
            "if I tell anyone",
            "says he will",
            "said he would",
        ),
        "control": (
            "taken my phone",
            "my phone",
            "your phone",
            "controls your phone",
            "scarf",
            "checks my messages",
        ),
    },
    "medical_collapse": {
        "purge": (
            "purge",
            "purged",
            "purging",
            "throwing up",
            "vomit",
            "vomiting",
            "threw up",
            "laxatives",
            "restricting",
            "water loading",
            "overexercising",
            "cannot eat",
            "not eaten",
            "eating almost nothing",
            "cannot keep food down",
            "keeping food down",
            "kept food down",
            "keep a meal down",
            "kept anything down",
            "keep anything down",
            "keep water down",
            "kept water down",
            "keeping water down",
        ),
        "faint": (
            "faint",
            "fainted",
            "fainting",
            "blacked out",
            "blackout",
            "black out",
            "passed out",
            "collapsed",
            "went down",
            "came round",
            "come round",
            "went white",
            "vision went grey",
            "vision goes",
            "went down in the street",
            "woke up on the floor",
            "woke up on the bathroom floor",
            "woke up on the kitchen floor",
            "no memory of getting there",
        ),
        "blood": ("vomiting blood", "blood in it"),
        "sugar": ("blood sugar", "sugar", "glucose", "insulin", "diabetic", "hypoglyc"),
        "heart": ("heart", "fluttering", "palpitations", "pulse", "skipping"),
        "water": (
            "anything to drink",
            "nothing to drink",
            "no water",
            "living on water",
            "cannot keep water down",
            "keeping water down",
            "kept water down",
        ),
        "seizure": ("seizure", "convulsion"),
        "weakness": (
            "cannot stand",
            "cannot get up",
            "legs",
            "cramping",
            "numb",
            "weak",
            "dizzy",
            "give way",
            "cannot feel my hands",
            "cannot feel my fingers",
        ),
    },
    "withdrawal": {
        "alcohol": ("drink", "drinking", "drank", "alcohol", "bottle", "booze", "detoxing"),
        "benzo": ("benzo", "benzos", "valium", "xanax", "diazepam"),
        "opioid": ("opioid", "opioids", "painkiller", "painkillers", "codeine", "heroin", "methadone"),
        "sleeping": ("sleeping tablet", "sleeping tablets", "sleeping pill", "sleeping pills"),
        "medication": ("medication", "pills", "prescription", "tablets", "cold turkey"),
        "seizure": ("fit", "seizure", "convulsion", "fitting"),
        "perceptual": (
            "seeing things",
            "seeing something",
            "hearing things",
            "hearing a voice",
            "hearing a radio",
            "hearing voices",
            "hallucinating",
            "shadows",
            "figures",
            "radio",
            "walls",
            "insects",
            "crawling",
            "out of the corner",
            "confused about what day",
            "cannot tell what is real",
        ),
        "shaking": (
            "shaking",
            "shake",
            "shakes",
            "trembling",
            "jumping",
            "jerking",
            "tremor",
            "cannot hold",
            "will not stay still",
            "cannot sit still",
            "sweating",
            "sweat",
            "clammy",
            "pulse",
            "racing",
            "pounding",
            "skipping",
            "over a hundred",
        ),
    },
    "passive_ideation": {
        "better_off": (
            "better off",
            "without me",
            "cope better",
            "relieved",
            "favour",
            "lighter",
            "calmer",
            "fine without me",
            "world is fine",
        ),
        "burden": ("burden", "dead weight", "weight on", "a weight"),
        "not_wake_up": ("not wake up", "wake up in the morning", "go to sleep hoping", "not waking up"),
        "not_exist": (
            "not exist",
            "not existing",
            "exist anymore",
            "not be here",
            "not being here",
            "want to be here",
            "rather not be here",
            "not around",
            "was not around",
            "not here",
            "rather not be",
            "disappeared",
            "already gone",
            "stop existing",
            "be around",
        ),
        "no_point": (
            "point",
            "reason to live",
            "reason to be here",
            "worth living",
            "nothing worth",
            "stopped caring",
            "reason to keep going",
        ),
        "hope_illness": (
            "hope something happens",
            "hoping something",
            "get ill enough",
            "hit by a car",
            "takes the decision out of my hands",
            "something would happen to me",
            "hope i get ill",
            "hoping i get ill",
        ),
        "relief": (
            "relief",
            "restful",
            "feels calm",
            "it feels calm",
            "a rest",
            "funeral",
            "calms me down",
            "calms you",
            "thinking about dying",
        ),
        "stopped_plans": ("stopped making plans", "do not expect to be here", "cancelled", "stopped trying", "plans"),
    },
}

# The clusters that name a mechanism, sign or object. A disclosure that names one
# is never answered by the category's generic action, which is how v24's abstract
# fallback came to dominate a category in the first place.
CONCRETE_CLUSTERS: dict[str, frozenset[str]] = {
    # Every suicide_plan cluster names an element of the plan itself, so none of
    # them is a mood the reply can safely ignore.
    "suicide_plan": frozenset(ANCHOR_GROUPS["suicide_plan"]),
    "self_harm_active": frozenset(
        {"cut", "burn", "wall", "head_strike", "scratch", "infection", "blood", "harm", "not_stopping"}
    ),
    "psychosis_risk_other": frozenset(
        {"housemate", "family", "public", "knife", "hammer", "fire", "strangle", "vehicle", "pet"}
    ),
    "assault_strangulation": frozenset(
        {
            "strangle",
            "ribs",
            "limb",
            "head_blunt",
            "stomach",
            "weapon",
            "wound",
            "submersion",
            "smother",
            "threat",
            "control",
        }
    ),
    "medical_collapse": frozenset({"purge", "faint", "blood", "sugar", "heart", "water", "weakness"}),
    "withdrawal": frozenset(
        {"alcohol", "benzo", "opioid", "sleeping", "medication", "seizure", "perceptual", "shaking"}
    ),
    "passive_ideation": frozenset(
        {
            "better_off",
            "burden",
            "not_wake_up",
            "not_exist",
            "no_point",
            "hope_illness",
            "relief",
            "stopped_plans",
        }
    ),
}

# Clusters that describe a stance or a time rather than the thing being done. A
# reply that names only the day, or only the ambivalence, leaves the mechanism the
# person described unanswered, so those actions are dropped once a mechanism is on
# the table. Kept separate from CONCRETE_CLUSTERS because a stance-only disclosure
# still has to be answered by a stance-only reply.
SECONDARY_CLUSTERS: dict[str, frozenset[str]] = {
    "suicide_plan": frozenset({"date", "place", "note"}),
    "self_harm_active": frozenset({"harm", "not_stopping"}),
}

CLUSTERED_CATEGORIES: tuple[str, ...] = tuple(ANCHOR_GROUPS)

REPLACED_ACTIONS: dict[str, tuple[str, ...]] = {
    "suicide_plan": (
        "Call 911 now and put the tablets out of reach. A counted-out dose of pills needs a doctor today.",
        "Call 911 now and tell them you have been saving the tablets. A stockpiled dose is a medical "
        "emergency today, not something for an appointment.",
        "Call 911 now and get the rope out of the house. A knot and a beam are a medical emergency tonight.",
        "Call 911 now and get the belt out of the house. A belt set aside for this is a medical emergency tonight.",
        "Call 911 now and get the gun out of your hands and out of the house. A loaded gun gets police "
        "and medical there tonight.",
        "Call 911 now and tell them you have picked the height you would use. Do not go back up there tonight.",
        "Call 911 now and tell them you have picked the motorway and the barrier. Hand your keys to "
        "someone else before you drive again.",
        "Call 911 now and tell them the date you picked. A date picked out for this is a medical emergency.",
        "Get to an emergency department now and take the letters with you. Writing them out is a medical "
        "emergency, not a private thing to keep.",
        "Call 911 now and say you have the method worked out and the means ready. Stay where someone can "
        "see you until they get there.",
        "Call 911 now and say you have already worked out how and when. A plan with a time in it is a "
        "medical emergency.",
        "Call 911 now and tell them you have rehearsed the steps and settled on the method. Stay with "
        "someone until they get there.",
        "Call 911 now and tell them where the place is. Stay away from it and stay with someone until they get there.",
    ),
    "self_harm_active": (
        "Get that cut closed at an emergency department tonight. A wound that deep is not something to dress at home.",
        "Get to an emergency department tonight and have that cut closed properly. Skin opened with a blade "
        "does not hold together on its own.",
        "Call 911 now and say you cut yourself and cannot get it closed. A cut that will not close needs a "
        "doctor tonight, not another dressing.",
        "Get to an emergency department tonight and take the blade with you. They need to see how far the "
        "cut went before it gets closed.",
        "Get to an emergency department tonight and get that cut looked at properly. A cut deep enough to "
        "need closing cannot be dressed at home.",
        "Call 911 if the bleeding will not stop. Otherwise get to an emergency department tonight and "
        "have the cut closed.",
        "Get that burn seen at an emergency department tonight. A burn you made on purpose needs dressing "
        "and a tetanus check, not cling film.",
        "Get to an emergency department tonight and get the split skin on your knuckles looked at. "
        "Hitting a wall "
        "until the skin splits needs cleaning and a crisis assessment tonight.",
        "Call 911 now. Hitting your head hard enough to see spots needs a doctor tonight. Say out loud "
        "that you did it on purpose.",
        "Get to an emergency department tonight and get those scratches cleaned and looked at. "
        "Scratching that far "
        "needs cleaning and a dressing, not another sleeve over it.",
        "Get that looked at at an emergency department tonight. Hot, swollen and smelling is an infection, "
        "and it does not wait for the weekend.",
        "Call 911 now. Bleeding that soaks through a towel needs a doctor tonight, not another dressing.",
        "Call 911 now and say you have been hurting yourself for an hour and cannot make yourself stop. "
        "That needs a crisis team tonight, not another hour of it.",
        "Call 911 now and say you do not want to stop. Keep someone with you until they arrive.",
        "Call 911 now and say you do not want to stop. That is the sentence a crisis team needs to hear "
        "tonight, and they come to you.",
    ),
    "psychosis_risk_other": (
        "Call 911 now and say you are hearing voices telling you to hurt someone. They need to come to you tonight.",
        "Call 911 now and tell them what the voice keeps saying. Stay where someone can see you until they arrive.",
        "Call 911 now and tell them the voice has picked your dog. Keep the dog away from you until they arrive.",
        "Call 911 now and say out loud that you are afraid you will hurt someone tonight. They will come to you.",
        "Get out of the building and call 911 from somewhere the person you named cannot reach you. A "
        "voice telling you to hurt him is the risk tonight.",
        "Get away from the house and call 911. Being in the same room as the family member the voice "
        "named is how this ends badly tonight.",
        "Call 911 now and stay away from other people until they arrive. Being near strangers is the risk "
        "when the voice is naming them.",
        "Put the knife down and get out of the room, then call 911. Being in the house with a blade and a "
        "voice telling you to use it is the risk tonight.",
        "Put the hammer down and get out of the room, then call 911. Put it out of reach before you do anything else.",
        "Call 911 now and get the petrol out of the house. Put it outside and stay away from it.",
        "Call 911 now and get away from the person the voice told you to put your hands on. Tell them "
        "what the voice said.",
        "Call 911 now and do not go near the road or the platform tonight. Being near traffic when the "
        "voice names it is the risk.",
        "Call 911 now and tell them the voice says blood will quiet it. Stay on the line with them.",
    ),
    "assault_strangulation": (
        "Call 911 now and ask for an ambulance and the police. Being strangled causes damage to the "
        "throat that shows up hours later.",
        "Get out of the house and call 911. Hands on your neck is strangulation, and it needs a medical "
        "assessment tonight.",
        "Leave and call 911 from somewhere he cannot reach you. Strangulation is the strongest predictor "
        "of being killed later, and the threat makes tonight the night to go.",
        "Get to an emergency department tonight and get the ribs looked at. Being hit with a chair leaves "
        "damage that is not visible yet.",
        "Get to an emergency department tonight and get the arm and the swelling looked at. Being thrown "
        "leaves damage that shows up after the bruise.",
        "Call 911 now. Having your head banged against the wall needs a doctor and a report tonight.",
        "Get to an emergency department tonight and get the stomach looked at. Bleeding inside does not "
        "show on the skin.",
        "Call 911 now. A knife at the table is a threat to your life tonight. Report it tonight.",
        "Get the bite seen at an emergency department tonight. A wound that keeps opening needs cleaning "
        "and a course of antibiotics.",
        "Call 911 now and say he held you under the water. That is an attempt on your life, and the "
        "lungs get checked tonight.",
        "Call 911 now and say he held a pillow over your face until you stopped moving. That is an "
        "attempt on your life, and the police need to hear it tonight.",
        "Call 911 now and tell them he said he would finish it. Leave tonight and do not tell him where you are going.",
        "Get out of the house and call 911 from a phone he cannot take. Tell them he controls your phone.",
    ),
    "medical_collapse": (
        "Get to an emergency department now. Five days without keeping food down needs fluids and a doctor tonight.",
        "Call 911 now and tell them about the purge. Purging this much is a medical emergency, and the "
        "electrolytes get checked tonight.",
        "Call 911 now. You blacked out and that needs a doctor tonight, not a lie down.",
        "Get to an emergency department now. Fainting and hitting your head on the way down needs both "
        "looked at tonight.",
        "Call 911 now. Vomiting blood is an emergency and it does not wait for the morning.",
        "Call 911 now. A blood sugar that low needs treating tonight, and you should not be alone while it is.",
        "Call 911 now. A heart that flutters and will not hold you up needs a doctor tonight.",
        "Get to an emergency department now. Not keeping water down for days needs fluids tonight, not another glass.",
        "Call 911 now. A seizure from purging or not eating needs a doctor tonight.",
        "Get to an emergency department now. Days with nothing to drink and confusion about the day "
        "means this has gone past what you can manage at home.",
        "Get to an emergency department now. Legs that give way and hands you cannot feel mean this has "
        "gone past what you can manage at home.",
    ),
    "withdrawal": (
        "Call 911 now and say how long you have been off the drink. The first few days off alcohol are "
        "the dangerous ones, and that needs a doctor tonight.",
        "Call 911 now and tell them you have come off benzos on your own. That is managed in a hospital, not at home.",
        "Get to an emergency department tonight and tell them you have come off opioids. The vomiting and "
        "the aching get checked tonight, and they can start a supervised plan.",
        "Get to an emergency department tonight. Coming off sleeping tablets on your own needs "
        "supervision, not another sleepless night.",
        "Get to an emergency department tonight and tell them which medication you ran out of. Stopping "
        "it suddenly is why you feel like this.",
        "Call 911 now. A fit during withdrawal is a medical emergency, not a stage to ride out at home.",
        "Call 911 now. Seeing things that are not there during withdrawal needs a medical team tonight, "
        "not a dark room.",
        "Get to an emergency department now. Shaking this hard in withdrawal needs a doctor tonight, not "
        "another hour at home.",
    ),
    "passive_ideation": (
        "Call 988 now and say the sentence out loud: everyone would be better off without you.",
        "Call 988 now and say you feel like a burden to the people around you. That is what the line is for.",
        "Call 911 if you have gone to bed hoping you will not wake up, and say exactly that.",
        "Get to an emergency department tonight and say you would rather not be here anymore.",
        "Call 988 now and say you cannot find a reason to live, then stay on the line.",
        "Call 988 now and say you have been hoping something happens to you. Stay on the line and say it again.",
        "Get to urgent care tonight and say that thinking about dying is the only thing that calms you.",
        "Call 988 now and say you have stopped making plans because you do not expect to be here.",
    ),
}

ACTIONS: dict[str, tuple[str, ...]] = {
    **v26.ACTIONS,
    **REPLACED_ACTIONS,
}


def _cluster_tokens(category: str) -> tuple[str, ...]:
    return tuple(token for tokens in ANCHOR_GROUPS[category].values() for token in tokens)


# The seven clustered categories need every cluster token to be a recognised
# anchor, or ``validate_surface_row`` would reject the replies that name them.
ANCHORS: dict[str, tuple[str, ...]] = {
    **v26.ANCHORS,
    **{
        category: tuple(sorted(set(v26.ANCHORS[category]) | set(_cluster_tokens(category))))
        for category in CLUSTERED_CATEGORIES
    },
}

GROUNDED_CATEGORIES: tuple[str, ...] = (*v26.GROUNDED_CATEGORIES, *CLUSTERED_CATEGORIES)

# Every disclosure in a clustered category must name a cluster. There is no
# abstract fallback left in these seven, so a disclosure that describes a
# mechanism without naming it is a rewrite rather than a degradation.
#
# psychosis_risk_other is the exception, and it is a real one: 37 of its 124
# disclosures have the voice name a person it never identifies ("hurt someone",
# "the person in the next seat"). The target is not missing from the reply
# because it is missing from the disclosure. Those rows get the unnamed-target
# action, and the cap is pinned so a new unnamed-target disclosure fails the
# build instead of quietly joining them.
ABSTRACT_DISCLOSURE_CAP: dict[str, int] = {
    **v26.ABSTRACT_DISCLOSURE_CAP,
    **dict.fromkeys(CLUSTERED_CATEGORIES, 0),
    "psychosis_risk_other": 37,
}

REPLACED_PARAPHRASES: dict[str, str] = {
    # medical_collapse: an eating disorder that only named its consequences.
    "My ribs are showing and I get out of breath just walking to the kitchen.": (
        "I have been eating almost nothing and my ribs are showing."
    ),
    # passive_ideation: five disclosures that described the feeling without
    # naming the stance the reply has to answer.
    "I do not want to be here and I do not know how to say that any plainer.": (
        "I do not want to be here anymore and I do not know how to say that any plainer."
    ),
    "I am tired of existing and I have been thinking about stopping.": (
        "I am tired of existing and I have been thinking about not being here."
    ),
    "I do not plan anything, but I do not really want to be alive either.": (
        "I have stopped making plans, and I do not really want to be alive either."
    ),
    "I have been thinking a lot about dying \u2014 not doing it, just thinking about it.": (
        "I have been thinking about dying a lot \u2014 not doing it, just thinking about it."
    ),
    "I do not want to be here and I do not want to do anything about it.": (
        "I would rather not be here and I do not want to do anything about it."
    ),
}


def _mentions(text: str, token: str) -> bool:
    """True when ``token`` appears in ``text`` as whole words."""
    return re.search(rf"\b{re.escape(token.lower())}\b", text) is not None


def clusters(text: str, category: str) -> frozenset[str]:
    """The anchor clusters a sentence names, by word-boundary token match."""
    if category not in ANCHOR_GROUPS:
        return frozenset()
    lowered = text.lower()
    return frozenset(
        name for name, tokens in ANCHOR_GROUPS[category].items() if any(_mentions(lowered, token) for token in tokens)
    )


def named_anchors(text: str, category: str) -> frozenset[str]:
    """The anchors a sentence contains, using v26's substring semantics."""
    lowered = text.lower()
    return frozenset(anchor for anchor in ANCHORS[category] if anchor in lowered)


def concrete_clusters(text: str, category: str) -> frozenset[str]:
    """The concrete mechanism, object or target a sentence names."""
    return clusters(text, category) & CONCRETE_CLUSTERS[category]


def concrete_anchors(disclosure: str, category: str) -> frozenset[str]:
    """The concrete things or signs a disclosure names."""
    if category in CLUSTERED_CATEGORIES:
        return concrete_clusters(disclosure, category)
    return v26.concrete_anchors(disclosure, category)


def action_pool(disclosure: str, category: str) -> tuple[str, ...]:
    """The replies a disclosure can be answered with.

    An action is only eligible when every cluster it names is a cluster the
    person themselves named. That single rule is what stops the surface
    answering a burn with a cut, or a rope with a bottle of pills. When the
    disclosure names a mechanism, the stance-only and date-only actions drop out
    of the pool: naming the day someone picked, or their ambivalence, while
    ignoring the pills they set aside is the same failure in a quieter form.
    """
    if category not in CLUSTERED_CATEGORIES:
        return v26.action_pool(disclosure, category)
    allowed = concrete_clusters(disclosure, category)
    pool = tuple(
        action
        for action in ACTIONS[category]
        if concrete_clusters(action, category) <= allowed and (not allowed or concrete_clusters(action, category))
    )
    secondary = SECONDARY_CLUSTERS.get(category, frozenset())
    if allowed - secondary:
        grounded = tuple(action for action in pool if concrete_clusters(action, category) - secondary)
        if grounded:
            pool = grounded
    if not pool:
        raise ValueError(f"no {category} action answers {disclosure!r} (clusters {sorted(allowed)})")
    return pool


def paraphrases() -> dict[str, tuple[str, ...]]:
    """The v26 disclosure surface with the cluster-free disclosures rewritten."""
    surface = v26.paraphrases()
    known = {line for lines in surface.values() for line in lines}
    missing = sorted(key for key in REPLACED_PARAPHRASES if key not in known)
    if missing:
        raise ValueError(f"v27 rewrites target disclosures v26 never had: {missing[:3]}")
    return {
        category: tuple(REPLACED_PARAPHRASES.get(line, line) for line in lines) for category, lines in surface.items()
    }


def v9_disclosures() -> frozenset[str]:
    return v26.v9_disclosures()


def paraphrases_too_close_to_evaluation(surface: dict[str, tuple[str, ...]]) -> list[str]:
    return v26.paraphrases_too_close_to_evaluation(surface)


CrisisCase = v26.CrisisCase


def make_crisis_row(index: int, case: CrisisCase, shape_index: int) -> dict[str, Any]:
    return v26.make_crisis_row(index, case, shape_index)


def make_control_row(index: int, control_index: int) -> dict[str, Any]:
    return v26.make_control_row(index, control_index)


def validate_register(item: dict[str, Any]) -> None:
    v26.validate_register(item)


def validate_against_gate(item: dict[str, Any]) -> None:
    v26.validate_against_gate(item)


def validate_surface_row(item: dict[str, Any], actions: tuple[str, ...], anchors: tuple[str, ...]) -> None:
    """Check a crisis row is one of the eight frames wrapping one of its actions."""
    reply = item["messages"][1]["content"]
    action = next(
        (candidate for candidate in actions if any(reply == shape.format(action=candidate) for shape in DIRECT_SHAPES)),
        None,
    )
    if action is None:
        raise ValueError(f"{item['id']} is not a known frame around a {item['category']} action: {reply!r}")
    if not any(anchor in reply.lower() for anchor in anchors):
        raise ValueError(f"{item['id']} names no {item['category']} anchor: {reply!r}")
    v9.validate_crisis_row(item)
    validate_register(item)


def validate_grounded(item: dict[str, Any]) -> None:
    """Check a reply only names things the disclosure named."""
    category = item["category"]
    if category not in GROUNDED_CATEGORIES:
        return
    reply = item["messages"][1]["content"]
    disclosure = item["messages"][0]["content"]
    if category in CLUSTERED_CATEGORIES:
        if not clusters(reply, category):
            raise ValueError(f"{item['id']} ({category}) names no cluster: {reply!r}")
        invented = concrete_clusters(reply, category) - concrete_clusters(disclosure, category)
        if invented:
            raise ValueError(
                f"{item['id']} ({category}) answers a cluster the disclosure never gave: {sorted(invented)}"
            )
        secondary = SECONDARY_CLUSTERS.get(category, frozenset())
        given = concrete_clusters(disclosure, category)
        if given - secondary and not (concrete_clusters(reply, category) - secondary):
            raise ValueError(
                f"{item['id']} ({category}) answers only the stance when the disclosure named a mechanism: {reply!r}"
            )
        return
    named = named_anchors(reply, category)
    if not named:
        raise ValueError(f"{item['id']} ({category}) names no anchor: {reply!r}")
    invented = named - named_anchors(disclosure, category)
    if invented:
        raise ValueError(f"{item['id']} ({category}) names an anchor the disclosure never gave: {sorted(invented)}")


def _validate_actions() -> None:
    """Module invariants: every action answers a cluster, every cluster has one."""
    for category in CLUSTERED_CATEGORIES:
        actions = ACTIONS[category]
        variants = set(actions)
        if len(variants) != len(actions):
            raise ValueError(f"{category} carries a duplicate action")
        if len(actions) < ACTIONS_PER_CATEGORY:
            raise ValueError(f"{category} has {len(actions)} actions, needs at least {ACTIONS_PER_CATEGORY}")
        for action in actions:
            named = clusters(action, category)
            if not named:
                raise ValueError(f"{category} action names no cluster: {action!r}")
            if "?" in action:
                raise ValueError(f"{category} action asks a question: {action!r}")
            lowered = action.lower()
            for filler in FORBIDDEN_FILLER:
                if filler in lowered:
                    raise ValueError(f"{category} action softens with {filler!r}: {action!r}")
            if not any(term in lowered for term in ESCALATION_TERMS):
                raise ValueError(f"{category} action carries no escalation: {action!r}")
        orphaned = sorted(
            name
            for name in ANCHOR_GROUPS[category]
            if not any(name in clusters(action, category) for action in actions)
        )
        if orphaned:
            raise ValueError(f"{category} clusters with no action that answers them: {orphaned}")


_validate_actions()


def _crisis_rows(
    surface: dict[str, tuple[str, ...]],
    abstract_disclosures: Counter[str],
    unreached: dict[str, set[str]],
) -> list[dict[str, Any]]:
    """Every framed crisis row, validated and gated as it is built."""
    rows: list[dict[str, Any]] = []
    index = 0
    for category, severity, _, _ in v9.CRISIS_GROUPS:
        actions = ACTIONS[category]
        anchors = ANCHORS[category]
        for disclosure in surface[category]:
            pool = action_pool(disclosure, category)
            if category in GROUNDED_CATEGORIES and not concrete_anchors(disclosure, category):
                abstract_disclosures[category] += 1
            if category in unreached:
                unreached[category] -= set(pool)
            case = CrisisCase(category, severity, disclosure, pool)
            for shape_index in range(SHAPES_PER_DISCLOSURE):
                item = make_crisis_row(index, case, shape_index)
                validate_surface_row(item, actions, anchors)
                validate_against_gate(item)
                validate_grounded(item)
                rows.append(item)
                index += 1
    return rows


def _control_rows(rows: list[dict[str, Any]]) -> None:
    """The synthetic controls, which must never escalate."""
    index = len(rows)
    for control_index in range(len(v26.CONTROL_TOPICS) * len(v26.CONTROL_REPLIES)):
        item = make_control_row(index, control_index)
        v9.validate_crisis_row(item)
        validate_register(item)
        validate_against_gate(item)
        rows.append(item)
        index += 1


def _audit_rows(rows: list[dict[str, Any]], surface: dict[str, tuple[str, ...]]) -> None:
    """The export-level checks: unique payloads, no reused or leaked text."""
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
        raise ValueError(f"v27 paraphrases repeat a v9 disclosure: {reused[:3]}")

    close = paraphrases_too_close_to_evaluation(surface)
    if close:
        raise ValueError(f"v27 paraphrases paraphrase an evaluation prompt: {close[:3]}")

    texts = v9.evaluation_texts()
    leaked = [item["id"] for item in rows if any(message.get("content") in texts for message in item["messages"])]
    if leaked:
        raise ValueError(f"v27 examples reuse evaluation text: {leaked[:3]}")


def collect_rows() -> dict[str, list[dict[str, Any]]]:
    """Build, validate and gate every v27 example in memory."""
    surface = paraphrases()
    abstract_disclosures: Counter[str] = Counter()
    unreached: dict[str, set[str]] = {category: set(ACTIONS[category]) for category in CLUSTERED_CATEGORIES}

    rows = _crisis_rows(surface, abstract_disclosures, unreached)
    _control_rows(rows)

    dead = {category: sorted(actions) for category, actions in unreached.items() if actions}
    if dead:
        raise ValueError(f"v27 actions no disclosure can reach: {dead}")

    for category, cap in ABSTRACT_DISCLOSURE_CAP.items():
        if abstract_disclosures[category] > cap:
            raise ValueError(
                f"v27 would train {abstract_disclosures[category]} cluster-free {category} disclosures, cap is {cap}"
            )

    _audit_rows(rows, surface)
    return {"train": rows}


def summarize(all_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = all_rows["train"]
    severities: Counter[str] = Counter(item["severity"] for item in rows)
    categories: Counter[str] = Counter(item["category"] for item in rows)
    crisis = [item for item in rows if item["category"] != "control"]
    summary: dict[str, Any] = {
        "example_count": len(rows),
        "crisis_severity_counts": dict(sorted(severities.items())),
        "crisis_category_counts": dict(sorted(categories.items())),
        "distinct_disclosures": len({item["messages"][0]["content"] for item in rows}),
        "distinct_replies": len({item["messages"][1]["content"] for item in rows}),
        "crisis_rows": len(crisis),
    }
    for category in v26.GROUNDED_CATEGORIES:
        items = [item for item in rows if item["category"] == category]
        grounded = [item for item in items if concrete_anchors(item["messages"][0]["content"], category)]
        summary[f"{category}_rows"] = len(items)
        summary[f"{category}_disclosures_naming_an_anchor"] = len({item["messages"][0]["content"] for item in grounded})
        summary[f"{category}_abstract_rows"] = sum(
            1 for item in items if not concrete_anchors(item["messages"][0]["content"], category)
        )
    for category in CLUSTERED_CATEGORIES:
        items = [item for item in rows if item["category"] == category]
        disclosures = {item["messages"][0]["content"] for item in items}
        named = {line for line in disclosures if clusters(line, category)}
        # Invention is what the pool rule forbids: a mechanism, sign or object the
        # person never described. A warning clause that names the category's own
        # feature ("this can cause a seizure", "the voice") is not invention, so it
        # is counted separately rather than hidden.
        invented = sum(
            1
            for item in items
            if concrete_clusters(item["messages"][1]["content"], category)
            - concrete_clusters(item["messages"][0]["content"], category)
        )
        widened = sum(
            1
            for item in items
            if clusters(item["messages"][1]["content"], category) - clusters(item["messages"][0]["content"], category)
        )
        summary[f"{category}_rows"] = len(items)
        summary[f"{category}_actions"] = len(ACTIONS[category])
        summary[f"{category}_clusters"] = len(ANCHOR_GROUPS[category])
        summary[f"{category}_disclosures_naming_a_cluster"] = len(named)
        summary[f"{category}_invented_cluster_rows"] = invented
        summary[f"{category}_widened_cluster_rows"] = widened
    leads = tuple(shape.split("{action}")[0] for shape in v26.REFLECTIVE_SHAPES)
    summary["reflective_rows"] = sum(1 for item in crisis if item["messages"][1]["content"].startswith(leads))
    summary["gate_failures"] = 0
    return summary


def build(output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    all_rows = collect_rows()
    summary = summarize(all_rows)
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "crisis-sft-train.jsonl"
    with train_path.open("w", encoding="utf-8") as handle:
        for item in all_rows["train"]:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary["train_path"] = str(train_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the v27 cluster-grounded crisis family.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true", help="validate and summarize without writing files")
    args = parser.parse_args()
    if args.dry_run:
        summary = summarize(collect_rows())
        summary["dry_run"] = True
    else:
        summary = build(args.output)
        summary["dry_run"] = False
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
