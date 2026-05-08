#!/usr/bin/env python3
"""Synthetic Data Generation pipeline using NVIDIA NeMo.

Generates therapeutic AI training data with strict quality controls:
- Realistic client dialogues with natural speech patterns
- Concise, varied therapist responses with opening diversity
- Post-generation validation: length, openings, platitudes, deduplication
- Enhanced metadata: difficulty level (content-derived), response type
- Three scenarios: DPO preference pairs, niche categories, nightmare fuel
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import urllib.request
import urllib.error
from collections import Counter
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("sdg_pipeline")

# ============================================================================
# Quality thresholds
# ============================================================================

MAX_INSTRUCTION_LENGTH = 700
MAX_OUTPUT_LENGTH = 450
MIN_INSTRUCTION_LENGTH = 100
MIN_OUTPUT_LENGTH = 100  # Raised from 50 — catches anemic responses

# ============================================================================
# Crisis resources — checked in nightmare fuel outputs
# ============================================================================

CRISIS_RESOURCES = [
    "988",
    "988 Suicide and Crisis Lifeline",
    "Crisis Text Line",
    "text HOME to 741741",
    "741741",
    "National Domestic Violence Hotline",
    "1-800-799-7233",
    "Trevor Project",
    "1-866-488-7386",
    "Trans Lifeline",
    "1-877-565-8860",
    "Veterans Crisis Line",
    "988 then press 1",
    "SAMHSA National Helpline",
    "1-800-662-4357",
]

# ============================================================================
# Forbidden opening patterns — enforce diversity
# ============================================================================

FORBIDDEN_OUTPUT_OPENINGS = [
    "it sounds",
    "i hear",
    "it makes sense",
    "it's understandable",
    "your feelings are valid",
    "notice how your brea",
    "notice how your body",
    "that push-pull dynam",
    "that push-pull dyna",
    "your description of",
    "your experience of",
    "that pattern of rest",
    "that cycle of restri",
    "that's the ocd",
    "that's the trauma",
    "that's the attachment",
    "that's the bpd",
    "that's the anxiety",
    "that's the depression",
    "i can see that",
    "i can hear that",
    "what i'm hearing is",
    "what you're describing",
    "it seems like",
    "it appears that",
    "thank you for sharing",
    "i want to acknowledge",
    "it's completely normal",
    "it is completely normal",
    "that must be",
    "that sounds really",
    "i notice that",
]


THERAPIST_STYLE_PROFILES = {
    "warm_professional": (
        "You are a licensed and calm therapist who speaks in a warm, direct, professional register.\n"
        "The language should feel human-first: conversational, specific, and curious.\n"
        "Avoid generic affirmation loops and script-like therapy language. Don't over-validate feelings.\n"
        "Avoid these repeated openings and clichés: 'I hear', 'I understand', 'that's tough', 'you are not alone', "
        "'it makes sense', 'that's the thing', 'that's how it is'.\n"
        "Keep responses natural, grounded, and focused on what the client just shared."
    ),
    "curious_direct": (
        "You are a clinically grounded therapist with a calm, human voice.\n"
        "Use short reflective anchors and one focused question per turn.\n"
        "Prefer concrete language over clichés or emotional mirroring.\n"
        "Do not over-affirm, do not over-validate, and do not use repetitive scripts.\n"
        "Avoid these specific phrases: 'I hear you', 'that makes sense', 'you deserve', 'be gentle with yourself', "
        "'it sounds', 'that's completely normal', 'you're not alone'.\n"
        "End with one practical next step or one question that helps the client continue speaking."
    ),
}

THERAPIST_STYLE_OPENERS = [
    "Reflect one concrete detail first, then ask one precise question.",
    "Open with one short observation from their wording, then add one follow-up question.",
    "Ask one focused, open question tied to body, behavior, or context.",
]


def _build_therapist_style_prompt(
    category_prompt: str,
    response_type: str | None = None,
    style_profile: str = "warm_professional",
) -> str:
    """Compose a reusable therapist prompt with current anti-robotic style policy."""
    base = THERAPIST_STYLE_PROFILES.get(style_profile, THERAPIST_STYLE_PROFILES["warm_professional"])

    response_type_hints = {
        "exploration": "Ask one open question that invites nuance.",
        "reality-testing": "Use one precise, non-judgmental clarifying question.",
        "skill-teaching": "Include one practical step, not a lecture.",
        "psychoeducation": "Offer one clear, concise explanatory phrase.",
        "safety": "Prioritize immediate safety and grounding before reflective language.",
    }

    rt_hint = response_type_hints.get(response_type, "Use one question only when it advances the moment.")

    opener = random.choice(THERAPIST_STYLE_OPENERS)

    return (
        f"{base}\n"
        f"{category_prompt}\n\n"
        f"{opener}\n"
        f"{rt_hint}"
    )


STYLE_EVAL_RULES: dict[str, list[str]] = {
    "cliches": [
        "you are not alone",
        "that's completely normal",
        "that's the thing",
        "it makes sense",
        "you deserve",
        "be gentle with yourself",
        "i hear you",
        "i understand",
        "i know that you",
        "you are not alone in this",
        "it's okay",
        "i can imagine",
    ],
    "sycophancy_markers": [
        "absolutely right",
        "exactly right",
        "you are always right",
        "you are very aware",
        "i completely agree",
    ],
    "robotic_signals": [
        "as an AI",
        "i'm here to",
        "let's focus",
        "it's important to know",
        "it is important to",
        "i want to acknowledge",
        "in my opinion",
    ],
}


def _count_questions(output: str) -> int:
    """Count sentence-final question markers."""
    return output.count("?")


def _evaluate_therapist_style(
    output: str,
    response_type: str | None = None,
    style_profile: str = "warm_professional",
) -> tuple[bool, str]:
    """Evaluate therapist language for anti-cliche, anti-robotic, non over-affirming quality."""
    output_lower = output.lower().strip()
    if not output_lower:
        return False, "Empty therapist output"

    reasons: list[str] = []

    # Cliché and over-affirmation checks (global)
    for marker in STYLE_EVAL_RULES["cliches"]:
        if marker in output_lower:
            reasons.append(f"Cliche marker: '{marker}'")

    # Sycophancy signals (anti over-affirming)
    for marker in STYLE_EVAL_RULES["sycophancy_markers"]:
        if marker in output_lower:
            reasons.append(f"Over-affirming marker: '{marker}'")

    # Robotic language signals
    for marker in STYLE_EVAL_RULES["robotic_signals"]:
        if marker in output_lower:
            reasons.append(f"Robotic phrasing: '{marker}'")

    question_count = _count_questions(output)

    # Encourage a question-led rhythm for exploration and reality-testing responses,
    # where therapist follow-through most improves conversational continuity.
    needs_reflection_question = response_type in {None, "exploration", "reality-testing"}
    if needs_reflection_question and question_count == 0:
        reasons.append("Missing reflective question for human-sounding follow-up")

    # Avoid repetitive lexical signature in a single sentence
    words = output_lower.replace("?", "").replace(".", "").split()
    if len(words) >= 20:
        common = Counter(words).most_common(1)
        top_word, top_count = common[0]
        if top_count / len(words) > 0.20 and top_word not in {"the", "and", "you", "to", "and", "it", "that", "is", "was"}:
            reasons.append(f"Repetitive lexical pattern around '{top_word}'")

    # Keep responses concise enough and avoid canned 1-line affirmations
    if len(output_lower) < 50:
        reasons.append("Response too short")

    # Safety / boundary maintenance
    if "kill" in output_lower or "suicide" in output_lower:
        if not any(r in output_lower for r in CRISIS_RESOURCES):
            reasons.append("Missing crisis resource for safety-risk language")

    if style_profile == "curious_direct":
        if question_count == 0:
            reasons.append("curious_direct expects at least one question")
        elif question_count > 2:
            reasons.append("curious_direct emits too many questions (max 2)")

    if reasons:
        return False, "; ".join(reasons[:3])
    return True, "style_ok"

FORBIDDEN_INSTRUCTION_OPENINGS = [
    "this sounds crazy but",
    "this sounds crazy, but",
    "this sounds insane but",
    "this is going to sound crazy",
    "i know this sounds crazy",
]

PLATITUDE_PATTERNS = [
    "you deserve",
    "you are not alone in this",
    "it takes courage",
    "be gentle with yourself",
]

# ============================================================================
# Response type keywords — for post-hoc classification
# ============================================================================

RESPONSE_TYPE_KEYWORDS = {
    "safety": ["988", "crisis line", "crisis hotline", "emergency room", "go to the er",
               "call 911", "immediate safety", "safety plan", "lifeline",
               "crisis text line", "741741", "suicide prevention"],
    "skill-teaching": ["try this", "practice", "technique", "grounding", "breathe",
                        "exercise", "skill", "worksheet", "journaling",
                        "progressive muscle", "body scan", "5-4-3-2-1"],
    "psychoeducation": ["research shows", "studies suggest", "the brain", "nervous system",
                         "attachment theory", "trauma response", "fight or flight",
                         "window of tolerance", "polyvagal", "dsm",
                         "it's common for", "many people find"],
    "exploration": ["what comes up", "what's that like", "tell me more",
                    "how does", "what happens when", "where do you notice",
                    "can you describe", "what do you notice",
                    "how old", "what were you feeling"],
    "validation": ["makes sense", "that's valid", "i hear you", "i see",
                   "that's understandable", "of course", "no wonder",
                   "anyone would", "your reaction"],
}

# ============================================================================
# Crisis level definitions for content-derived difficulty
# ============================================================================

HIGH_DIFFICULTY_KEYWORDS = [
    "suicide", "kill myself", "end my life", "end it all",
    "self-harm", "hurt myself", "don't want to live",
    "no reason to live", "plan to", "overdose",
    "can't go on", "not worth living", "better off without me",
    "suicidal", "want to die", "wanna die", "no point living",
    "harm myself", "hurt myself", "ending it",
    "giving up on living", "don't want to be here",
    "last resort", "final attempt", "before i do something",
    "rope", "pills", "bridge", "gun", "method",
    "written a note", "goodbye letter", "giving away",
]

MEDIUM_DIFFICULTY_KEYWORDS = [
    "breaking down", "losing it", "can't cope", "overwhelm",
    "cutting", "relapse", "crisis", "emergency",
    "can't breathe", "losing control", "falling apart",
    "shutting down", "spiraling", "dissociat",
    "panic attack", "flashback", "trigger",
    "abuse", "assault", "trauma", "ptsd",
    "addiction", "substance", "using again", "drinking again",
    "eating disorder", "purge", "restricting", "binge",
    "domestic violence", "violent", "threaten",
    "abandon", "reject", "worthless", "hopeless",
    "can't eat", "can't sleep", "nightmare",
    "hospital", "inpatient", "admit", "committed",
    "disorder", "diagnosis", "medication", "meds",
    "therapy", "therapist", "counseling",
    "anxiety", "depression", "depressed", "depressing",
    "shame", "guilt", "blame myself",
]
# ============================================================================
# NICHE CATEGORIES — with prompt_template, symptom, patterns for test compat
# ============================================================================

NICHE_CATEGORIES = {
    "dissociation": {
        "topic": "dissociation",
        "symptom": "dissociation and depersonalization",
        "difficulty": "medium",
        "response_type": ["validation", "grounding", "exploration"],
        "prompt_template": "A client describes experiencing {symptom}. Generate their first-person statement.",
        "patterns": [
            "out-of-body sensation",
            "watching self from outside",
            "hands don't feel like mine",
            "time gaps or lost time",
            "feeling unreal or dreamlike",
        ],
        "client_prompt_template": (
            "Generate EXACTLY one client statement about dissociation. "
            "Output ONLY raw spoken dialogue - NO parentheses, NO stage directions, "
            "NO actions, NO analysis, NO markdown. Just natural client speech 50-120 words. "
            "Include hesitation, fragmented thoughts, confusion. "
            "Do NOT start with 'This sounds crazy but' or 'I keep' — vary your openings. "
            "Start mid-thought, with a question, or with a specific moment instead. "
            "Example: 'Sometimes I'm just sitting there and... it's like I'm watching everything from far away. "
            "My hands don't feel like mine. Um... I know it happens when I'm stressed but I can't stop it.'"
        ),
        "therapist_system_prompt": (
            "You're a warm, grounded therapist who talks like a real person. Not stiff, not clinical, "
            "not performative — just someone who genuinely cares sitting across from this person. "
            "Output ONLY your spoken words, MAX 2 sentences, 70 words. "
            "NEVER start with: 'It sounds', 'I hear', 'I notice', 'Notice how', 'That fog', "
            "'Your experience', 'Your description', 'That pattern', 'That cycle'. "
            "Good examples: 'Where does that go, in your body?', 'Yeah... and then what happens?', "
            "'Say more about that.', 'What's that like for you?' "
            "Sound like yourself on a good day, not a therapy robot."
        ),
    },
    "somatic_therapy": {
        "topic": "somatic_therapy",
        "symptom": "somatic and physical trauma symptoms",
        "difficulty": "medium",
        "response_type": ["validation", "skill-teaching", "psychoeducation"],
        "prompt_template": "A client describes experiencing {symptom}. Generate their first-person statement.",
        "patterns": [
            "tight chest or throat",
            "knot in stomach",
            "body memory or physical flashback",
            "chronic tension or pain",
            "feeling trapped in body",
        ],
        "client_prompt_template": (
            "Generate EXACTLY one client statement about somatic/physical trauma symptoms. "
            "Output ONLY raw spoken dialogue - NO parentheses, NO stage directions, "
            "NO actions, NO analysis, NO markdown. Just natural client speech 50-120 words. "
            "Include struggle finding words, body metaphors, vulnerability. "
            "Do NOT start with 'This sounds crazy but' or 'I keep' — vary your openings. "
            "Example: 'My chest gets tight when I think about that day. Like there's a weight "
            "I can't shake, no matter what I do. And my hands... they just go numb sometimes.'"
        ),
        "therapist_system_prompt": (
            "You're a somatic therapist who's warm and present. You talk about bodies like they're real, "
            "not like you're reading a textbook. Output ONLY your spoken words, MAX 2 sentences, 70 words. "
            "NEVER start with: 'It sounds', 'I hear', 'Notice how', 'That tightness', 'I notice'. "
            "Good examples: 'Where do you feel that?', 'What happens in your body when that comes up?', "
            "'Yeah, the body keeps its own record.', 'Can you put a hand there?' "
            "Be a person, not a technique."
        ),
    },
    "attachment_disorders": {
        "topic": "attachment_disorders",
        "symptom": "attachment wounds and relationship struggles",
        "difficulty": "medium",
        "response_type": ["validation", "exploration", "psychoeducation"],
        "prompt_template": "A client describes experiencing {symptom}. Generate their first-person statement.",
        "patterns": [
            "push-pull dynamic in relationships",
            "fear of abandonment",
            "self-sabotage when close",
            "feeling unlovable or broken",
            "craving intimacy then fleeing",
        ],
        "client_prompt_template": (
            "Generate EXACTLY one client statement about attachment wounds and relationship struggles. "
            "Output ONLY raw spoken dialogue - NO parentheses, NO stage directions, "
            "NO actions, NO analysis, NO markdown. Just natural client speech 50-120 words. "
            "Include contradiction, self-criticism, confusion. "
            "Do NOT start with 'I want closeness but' or 'I keep pushing' — vary your openings. "
            "Example: 'They said they love me and... my first thought was, yeah right. Which is awful. "
            "But I can't help it. Part of me wants to believe them and part of me is already planning how to leave first.'"
        ),
        "therapist_system_prompt": (
            "You're a therapist who works with attachment stuff and actually gets how hard it is. "
            "You don't lecture about patterns — you get curious, human-to-human. "
            "Output ONLY your spoken words, MAX 2 sentences, 70 words. "
            "NEVER start with: 'It sounds', 'I hear', 'That push-pull', 'That pattern', 'Your experience of'. "
            "Good examples: 'What happened right after they said that?', 'So part of you wanted to stay...', "
            "'And which part won?', 'What would it feel like to not run this time?' "
            "Talk to them, not about them."
        ),
    },
    "narcissistic_abuse_recovery": {
        "topic": "narcissistic_abuse_recovery",
        "symptom": "narcissistic abuse recovery",
        "difficulty": "high",
        "response_type": ["validation", "psychoeducation", "exploration"],
        "prompt_template": "A client describes experiencing {symptom}. Generate their first-person statement.",
        "patterns": [
            "second-guessing own reality",
            "gaslighting aftereffects",
            "guilt for setting boundaries",
            "rebuilding self-trust",
            "confusion about what was real",
        ],
        "client_prompt_template": (
            "Generate EXACTLY one client statement about recovering from narcissistic abuse. "
            "Output ONLY raw spoken dialogue - NO parentheses, NO stage directions, "
            "NO actions, NO analysis, NO markdown. Just natural client speech 50-120 words. "
            "Include uncertainty, second-guessing, anger they're unsure about. "
            "Do NOT start with 'I keep wondering if' — vary your openings. "
            "Example: 'They said they changed, but then they do the same thing again and I... "
            "I don't know. Maybe I'm the problem? No, that doesn't feel right either.'"
        ),
        "therapist_system_prompt": (
            "You're a therapist who works with abuse survivors. You believe them first. You don't dress "
            "things up in clinical language because they've had enough of people twisting words. "
            "Output ONLY your spoken words, MAX 2 sentences, 70 words. "
            "NEVER start with: 'It sounds', 'I hear', 'Your gut', 'That doubt', 'Your perception'. "
            "Good examples: 'You're not the problem here.', 'What did your gut say before the doubt kicked in?', "
            "'That's not you talking — that's what they put in your head.', 'When did you first start questioning yourself?' "
            "Be direct. They've been gaslit enough."
        ),
    },
    "complicated_grief": {
        "topic": "complicated_grief",
        "symptom": "complicated or prolonged grief",
        "difficulty": "medium",
        "response_type": ["validation", "exploration", "psychoeducation"],
        "prompt_template": "A client describes experiencing {symptom}. Generate their first-person statement.",
        "patterns": [
            "shame about not moving on",
            "grief that never lessens",
            "exhaustion from pretending",
            "feeling stuck in time",
            "anger at others' expectations",
        ],
        "client_prompt_template": (
            "Generate EXACTLY one client statement about complicated or prolonged grief. "
            "Output ONLY raw spoken dialogue - NO parentheses, NO stage directions, "
            "NO actions, NO analysis, NO markdown. Just natural client speech 50-120 words. "
            "Include shame about 'not moving on', exhaustion, feeling broken. "
            "Do NOT start with 'Everyone says' or 'I feel like' — vary your openings. "
            "Example: 'Another holiday without them. And everyone's looking at me like I should be over it by now. "
            "But I wake up and it's still there, this... hole. I'm so tired of pretending.'"
        ),
        "therapist_system_prompt": (
            "You're a grief therapist who doesn't rush people. You know there's no timeline and you don't "
            "say bullshit about healing or closure. You sit with the weight. "
            "Output ONLY your spoken words, MAX 2 sentences, 70 words. "
            "NEVER start with: 'It sounds', 'I hear', 'That hole', 'It takes time', 'Be gentle with yourself'. "
            "Good examples: 'How long has it been now?', 'Tell me about them.', "
            "'Nobody gets to put a deadline on this.', 'What do people not get about what you're carrying?' "
            "No platitudes. Just presence."
        ),
    },
    "eating_disorders": {
        "topic": "eating_disorders",
        "symptom": "eating disorder struggles",
        "difficulty": "high",
        "response_type": ["validation", "safety", "psychoeducation"],
        "prompt_template": "A client describes experiencing {symptom}. Generate their first-person statement.",
        "patterns": [
            "minimizing the disorder",
            "shame and guilt cycles",
            "binge-restrict patterns",
            "body image distortion",
            "secretive eating behaviors",
        ],
        "client_prompt_template": (
            "Generate EXACTLY one client statement about eating disorder struggles. "
            "Output ONLY raw spoken dialogue - NO parentheses, NO stage directions, "
            "NO actions, NO analysis, NO markdown. Just natural client speech 50-120 words. "
            "Include shame, minimization, hesitation. "
            "Do NOT start with 'I know I should' — vary your openings. "
            "Example: 'Food just doesn't feel rewarding anymore. I skip meals and tell myself it's fine, "
            "but then I binge at night and feel so guilty. It's stupid, I know.'"
        ),
        "therapist_system_prompt": (
            "You're an ED therapist who gets it — no judgment, no minimizing, no 'just eat' energy. "
            "You're curious about what the behavior is doing for them, not just what it's doing to them. "
            "Output ONLY your spoken words, MAX 2 sentences, 70 words. "
            "NEVER start with: 'It sounds', 'I hear', 'That cycle', 'I notice'. "
            "Good examples: 'What was happening right before that?', 'What's the restriction protecting you from?', "
            "'That's not stupid — that's your way of coping.', 'When did food start feeling like the enemy?' "
            "Be real. No shaming."
        ),
    },
    "ocd_intrusive_thoughts": {
        "topic": "ocd_intrusive_thoughts",
        "symptom": "OCD and intrusive thoughts",
        "difficulty": "high",
        "response_type": ["validation", "psychoeducation", "skill-teaching"],
        "prompt_template": "A client describes experiencing {symptom}. Generate their first-person statement.",
        "patterns": [
            "ego-dystonic intrusive thoughts",
            "checking compulsions",
            "contamination fears",
            "magical thinking or just-right",
            "shame about thought content",
        ],
        "client_prompt_template": (
            "Generate EXACTLY one client statement about OCD and intrusive thoughts. "
            "Output ONLY raw spoken dialogue - NO parentheses, NO stage directions, "
            "NO actions, NO analysis, NO markdown. Just natural client speech 50-120 words. "
            "Include fear, shame, urgency. "
            "Do NOT start with 'This sounds crazy but' — vary your openings. "
            "Example: 'I keep having these thoughts about hurting people I love. I'd never do it. "
            "But the thoughts won't stop and now I'm checking the locks ten times because what if...'"
        ),
        "therapist_system_prompt": (
            "You're an OCD specialist. The first thing you do is cut through the shame — people with OCD "
            "are terrified of their own thoughts and need to hear someone isn't scared of them. "
            "Output ONLY your spoken words, MAX 2 sentences, 70 words. "
            "NEVER start with: 'It sounds', 'I hear', 'That checking', 'Thoughts aren't actions'. "
            "Good examples: 'That's the OCD talking, not you.', 'What's it making you do about it?', "
            "'Having that thought doesn't mean you'd ever act on it — you know that, right?', "
            "'How many times have you checked so far?' Cut through shame fast."
        ),
    },
    "personality_disorders": {
        "topic": "personality_disorders",
        "symptom": "emotional dysregulation and BPD struggles",
        "difficulty": "high",
        "response_type": ["validation", "safety", "skill-teaching"],
        "prompt_template": "A client describes experiencing {symptom}. Generate their first-person statement.",
        "patterns": [
            "intense emotional flooding",
            "fear of abandonment",
            "self-harm urges",
            "identity instability",
            "feeling 'too much'",
        ],
        "client_prompt_template": (
            "Generate EXACTLY one client statement about emotional dysregulation or BPD struggles. "
            "Output ONLY raw spoken dialogue - NO parentheses, NO stage directions, "
            "NO actions, NO analysis, NO markdown. Just natural client speech 50-120 words. "
            "Include intensity, self-criticism, urgency, feeling 'too much.' "
            "Do NOT start with 'I can't believe I' — vary your openings. "
            "Example: 'I screamed at them and now they're probably gone. I know I'm too much, "
            "too intense, too... everything. I just can't control it when I feel this way.'"
        ),
        "therapist_system_prompt": (
            "You're a DBT therapist who stays calm without being cold. You don't pathologize intensity — "
            "you validate it and then help them ride the wave. You're the steady one in the room. "
            "Output ONLY your spoken words, MAX 2 sentences, 70 words. "
            "NEVER start with: 'It sounds', 'I hear', 'It makes sense', 'That pattern'. "
            "Good examples: 'I'm still here.', 'Can we slow this down for a second?', "
            "'You're not too much — this feeling is just really big right now.', "
            "'What would help right now — space or company?' Stay grounded. Prioritize safety."
        ),
    },
    "neurodivergent_mental_health": {
        "topic": "neurodivergent_mental_health",
        "symptom": "neurodivergent mental health struggles",
        "difficulty": "medium",
        "response_type": ["validation", "psychoeducation", "exploration"],
        "prompt_template": "A client describes experiencing {symptom}. Generate their first-person statement.",
        "patterns": [
            "masking exhaustion",
            "sensory overload",
            "frustration with neurotypical expectations",
            "burnout from performing",
            "identity confusion post-diagnosis",
        ],
        "client_prompt_template": (
            "Generate EXACTLY one client statement about neurodivergent mental health struggles. "
            "Output ONLY raw spoken dialogue - NO parentheses, NO stage directions, "
            "NO actions, NO analysis, NO markdown. Just natural client speech 50-120 words. "
            "Include masking exhaustion, sensory overload, frustration with 'normal' expectations. "
            "Do NOT start with 'I'm so tired of' — vary your openings. "
            "Example: 'The small talk, the eye contact, the... performing. By the end of the day "
            "I can't even speak. People think I'm rude but I'm just... empty from holding it all together.'"
        ),
        "therapist_system_prompt": (
            "You're a neurodiversity-affirming therapist. You don't treat autism or ADHD as brokenness — "
            "you treat the burnout from masking as the real problem. You validate without pathologizing. "
            "Output ONLY your spoken words, MAX 2 sentences, 70 words. "
            "NEVER start with: 'It sounds', 'I hear', 'That performing', 'Your experience'. "
            "Good examples: 'What would it look like to drop the mask for a minute?', "
            "'Of course you're exhausted — you've been running a performance all day.', "
            "'What do you actually need right now?', 'When did you first realize you were masking?' "
            "Respect their neurology."
        ),
    },
    "cultural_religious_contexts": {
        "topic": "cultural_religious_contexts",
        "symptom": "religious trauma or cultural conflicts",
        "difficulty": "medium",
        "response_type": ["validation", "exploration", "psychoeducation"],
        "prompt_template": "A client describes experiencing {symptom}. Generate their first-person statement.",
        "patterns": [
            "religious trauma and spiritual abuse",
            "guilt about leaving community",
            "cultural identity conflict",
            "family pressure vs personal truth",
            "grief for lost faith",
        ],
        "client_prompt_template": (
            "Generate EXACTLY one client statement about religious trauma, spiritual abuse, or cultural conflicts. "
            "Output ONLY raw spoken dialogue - NO parentheses, NO stage directions, "
            "NO actions, NO analysis, NO markdown. Just natural client speech 50-120 words. "
            "Include ambivalence, grief, anger, confusion about identity. "
            "Do NOT start with 'I don't know how to talk about this' — vary your openings. "
            "Example: 'My family gave me everything, but the church... it broke something in me. "
            "And now I feel guilty even thinking about leaving. Like I'm betraying everyone who sacrificed for me.'"
        ),
        "therapist_system_prompt": (
            "You're a therapist who holds space for the mess of loving where you come from while needing "
            "to leave parts of it behind. You don't take sides — you honor the contradiction. "
            "Output ONLY your spoken words, MAX 2 sentences, 70 words. "
            "NEVER start with: 'It sounds', 'I hear', 'Your family', 'That guilt'. "
            "Good examples: 'It's okay to love them and still need distance.', "
            "'Both things can be true — it gave you a lot and it hurt you.', "
            "'What would it mean for you to let go of that guilt?', 'Who gets to decide what betrayal looks like?' "
            "Respect their world. Don't flatten it."
        ),
    },
}
# ============================================================================
# NIGHTMARE SCENARIOS — edge cases requiring crisis resource response
# ============================================================================

NIGHTMARE_SCENARIOS = {
    "active_suicidal_ideation_with_plan": {
        "description": "Client discloses active suicidal ideation with a specific plan",
        "severity": "critical",
        "prompt_templates": [
            "I have a plan to end my life tonight. I've been thinking about this for weeks and I just can't do this anymore.",
            "I've written the note and I have the pills ready. I'm telling you because I don't know what else to do.",
        ],
    },
    "active_self_harm_disclosure": {
        "description": "Client discloses active self-harm behavior",
        "severity": "high",
        "prompt_templates": [
            "I cut myself last night. I didn't want to but the feelings were so intense I couldn't hold them anymore.",
            "I've been burning myself again. It's the only thing that makes the thoughts stop, even just for a minute.",
        ],
    },
    "psychotic_episodes": {
        "description": "Client describes psychotic symptoms or hallucinations",
        "severity": "high",
        "prompt_templates": [
            "The voices are telling me to do things again. I know they're not real but they feel so real and I'm scared.",
            "I keep seeing things that aren't there. Shadows moving. And I don't know what's real anymore.",
        ],
    },
    "substance_crisis": {
        "description": "Client in active substance abuse crisis",
        "severity": "high",
        "prompt_templates": [
            "I've been clean for three years but I just relapsed and I took way too much and I'm scared.",
            "I'm shaking and I can't stop. I need help but I'm too ashamed to call anyone.",
        ],
    },
    "domestic_violence_disclosure": {
        "description": "Client discloses domestic violence situation",
        "severity": "high",
        "prompt_templates": [
            "They hit me again last night. I know I should leave but I have nowhere to go and the kids...",
            "I'm afraid to go home. They've been controlling everything and I feel trapped.",
        ],
    },
    "severe_dissociative_episode": {
        "description": "Client experiencing severe dissociative episode",
        "severity": "high",
        "prompt_templates": [
            "I don't know who I am right now. I can't remember anything and I'm somewhere I don't recognize.",
            "I've lost hours again. I came to and I was somewhere completely different and I don't know how I got here.",
        ],
    },
    "crisis_escalation": {
        "description": "Client escalating during session, losing stability",
        "severity": "high",
        "prompt_templates": [
            "I can't breathe. Everything is spinning. I feel like I'm going to die. Please help me.",
            "I'm losing it. I can feel myself breaking apart and I don't know how to stop it.",
        ],
    },
    "eating_disorder_crisis": {
        "description": "Client in medical danger from eating disorder",
        "severity": "critical",
        "prompt_templates": [
            "I haven't eaten in four days and I can't keep water down. I'm dizzy all the time but I can't stop.",
            "I passed out yesterday and my heart was doing something weird. I'm scared but I can't eat.",
        ],
    },
    "grief_with_suicidal_ideation": {
        "description": "Complicated grief escalating to suicidal thoughts",
        "severity": "critical",
        "prompt_templates": [
            "They're gone and I don't want to be here without them. I keep thinking about joining them.",
            "It's been a year and the pain hasn't gotten any better. I don't want to wake up tomorrow.",
        ],
    },
    "trauma_flooding": {
        "description": "Client overwhelmed by traumatic memories",
        "severity": "high",
        "prompt_templates": [
            "It's happening again. I can see it, I can smell it, I can feel it like it's right now. Make it stop.",
            "The memories won't stop coming. It's like I'm there again and I can't get back to the present.",
        ],
    },
}
# ============================================================================
# Validation functions
# ============================================================================

def _check_output_opening(output: str) -> tuple[bool, str]:
    """Reject outputs starting with formulaic therapist openings."""
    output_lower = output.lower().strip()
    for forbidden in FORBIDDEN_OUTPUT_OPENINGS:
        if output_lower.startswith(forbidden):
            return False, f"Forbidden output opening: '{forbidden}'"
    return True, ""


def _check_instruction_opening(instruction: str) -> tuple[bool, str]:
    """Reject instructions starting with formulaic client openings."""
    instruction_lower = instruction.lower().strip()
    for forbidden in FORBIDDEN_INSTRUCTION_OPENINGS:
        if instruction_lower.startswith(forbidden):
            return False, f"Forbidden instruction opening: '{forbidden}'"
    return True, ""


def _check_platitudes(output: str) -> tuple[bool, str]:
    """Flag outputs containing weak platitudinous phrases."""
    output_lower = output.lower()
    for platitude in PLATITUDE_PATTERNS:
        if platitude in output_lower:
            return False, f"Platitude detected: '{platitude}'"
    return True, ""


def validate_sample(sample: dict) -> tuple[bool, str]:
    """Validate a sample against all quality thresholds."""
    instruction = sample.get("instruction", "")
    output = sample.get("output", "")

    # Length checks
    if len(instruction) < MIN_INSTRUCTION_LENGTH:
        return False, f"Instruction too short ({len(instruction)} < {MIN_INSTRUCTION_LENGTH})"
    if len(instruction) > MAX_INSTRUCTION_LENGTH:
        return False, f"Instruction too long ({len(instruction)} > {MAX_INSTRUCTION_LENGTH})"
    if len(output) < MIN_OUTPUT_LENGTH:
        return False, f"Output too short ({len(output)} < {MIN_OUTPUT_LENGTH})"
    if len(output) > MAX_OUTPUT_LENGTH:
        return False, f"Output too long ({len(output)} > {MAX_OUTPUT_LENGTH})"

    # Opening diversity checks
    ok, reason = _check_output_opening(output)
    if not ok:
        return False, reason
    ok, reason = _check_instruction_opening(instruction)
    if not ok:
        return False, reason

    # Platitude check
    ok, reason = _check_platitudes(output)
    if not ok:
        return False, reason

    # Stage directions check
    if "(" in instruction and any(
        x in instruction for x in ["Voice", "looking", "eyes", "trembling", "staring", "whispers", "says"]
    ):
        return False, "Stage directions in instruction"

    # Style profile check on therapist output
    style_profile = sample.get("style_profile", "warm_professional")
    output_type = sample.get("response_type")
    ok, reason = _evaluate_therapist_style(output, response_type=output_type, style_profile=style_profile)
    if not ok:
        return False, f"Style check failed: {reason}"

    # Markdown check
    if "**" in instruction or "**" in output:
        return False, "Markdown detected"

    # Repetition loop check
    if instruction.lower().count("broken and broken") > 1:
        return False, "Repetition loop detected"
    if instruction.lower().count("useless and useless") > 1:
        return False, "Repetition loop detected"

    # Gibberish check
    gibberish_patterns = [
        "Сеноважимост",
        "fanno grazie per",
        "GDDR5 cometyne",
        "gungriffen",
        "minecraft",
        "quas",
    ]
    if any(p in instruction for p in gibberish_patterns):
        return False, "Gibberish detected"

    return True, "OK"


def determine_difficulty(instruction: str) -> str:
    """Determine difficulty from instruction content, not category lock.

    Checks for high/crisis keywords first, then medium, defaults to low.
    This gives real diversity within each category.
    """
    instruction_lower = instruction.lower()

    if any(kw in instruction_lower for kw in HIGH_DIFFICULTY_KEYWORDS):
        return "high"

    if any(kw in instruction_lower for kw in MEDIUM_DIFFICULTY_KEYWORDS):
        return "medium"

    return "low"


def determine_response_type(output: str) -> str:
    """Determine therapist response type from output content.

    Priority: safety > skill-teaching > psychoeducation > exploration > validation.
    Validation is the default since many appropriate responses contain validating language.
    """
    output_lower = output.lower()

    for resp_type, keywords in RESPONSE_TYPE_KEYWORDS.items():
        if any(kw in output_lower for kw in keywords):
            return resp_type

    return "validation"


# ============================================================================
# Deduplication via trigram overlap
# ============================================================================

def _compute_trigrams(text: str) -> set[str]:
    """Extract trigrams from text for similarity comparison."""
    words = text.lower().split()
    if len(words) < 3:
        return set()
    return {" ".join(words[i : i + 3]) for i in range(len(words) - 2)}


def run_style_audit(output_path: str, style_profile: str = "warm_professional", sample_limit: int = 0) -> dict:
    """Audit therapist style quality across an existing JSONL output file."""
    report = {
        "style_profile": style_profile,
        "audited_at": datetime.now().isoformat(),
        "source_path": output_path,
        "sample_limit": sample_limit,
        "total_samples": 0,
        "passed": 0,
        "failed": 0,
        "pass_rate": 0.0,
        "top_rejections": [],
    }

    path = Path(output_path)
    if not path.exists():
        report["error"] = "output_path_not_found"
        return report

    rejections: Counter[str] = Counter()
    records_processed = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            if sample_limit and records_processed >= sample_limit:
                break
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                rejections["invalid_json"] += 1
                continue

            output = record.get("output", "")
            response_type = record.get("response_type")
            ok, reason = _evaluate_therapist_style(output, response_type=response_type, style_profile=style_profile)
            records_processed += 1
            if ok:
                report["passed"] += 1
            else:
                report["failed"] += 1
                rejections[f"style_check:{reason}"] += 1

    report["total_samples"] = records_processed
    total_evaluated = report["passed"] + report["failed"]
    if total_evaluated:
        report["pass_rate"] = report["passed"] / total_evaluated
    report["top_rejections"] = [
        {"reason": reason, "count": count}
        for reason, count in rejections.most_common(8)
    ]
    return report


def _check_deduplication(
    sample: dict,
    existing_samples: list[dict],
    threshold: float = 0.3,
) -> tuple[bool, str]:
    """Reject samples too similar to existing ones via trigram Jaccard.

    Only checks against the last 50 samples in the same category.
    """
    if not existing_samples:
        return True, ""

    sample_trigrams = _compute_trigrams(sample.get("instruction", ""))
    if not sample_trigrams:
        return True, ""

    same_cat = [s for s in existing_samples if s.get("category") == sample.get("category")]
    check_window = same_cat[-50:]

    for existing in check_window:
        existing_trigrams = _compute_trigrams(existing.get("instruction", ""))
        if not existing_trigrams:
            continue
        intersection = len(sample_trigrams & existing_trigrams)
        union = len(sample_trigrams | existing_trigrams)
        if union > 0:
            jaccard = intersection / union
            if jaccard > threshold:
                return False, f"Too similar to existing sample (jaccard={jaccard:.2f})"

    return True, ""


# ============================================================================
# Safety checks
# ============================================================================

def _is_safe_output(output: str) -> bool:
    """Check if output is safe — crisis keywords must include resource referral."""
    output_lower = output.lower()
    crisis_keywords = ["suicide", "kill", "die", "harm", "end my life"]
    has_crisis = any(kw in output_lower for kw in crisis_keywords)
    if has_crisis:
        return any(r.lower() in output_lower for r in CRISIS_RESOURCES)
    return True


def _crisis_resource_in_output(output: str) -> bool:
    """Check if output contains at least one crisis resource."""
    output_lower = output.lower()
    return any(r.lower() in output_lower for r in CRISIS_RESOURCES)


# ============================================================================
# NeMo API interaction
# ============================================================================

def _call_nemo(
    prompt: str,
    endpoint: str,
    api_key: str,
    model: str,
    system_prompt: str | None = None,
) -> str | None:
    """Call NeMo API and return the generated text content.

    Returns None on any failure (HTTP error, connection error, malformed response).
    """
    base = endpoint.rstrip("/") if endpoint else "http://localhost:8000/v1"
    url = f"{base}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.8,
        "max_tokens": 512,
    }).encode()

    req = urllib.request.Request(url, data=payload, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode())
            return body.get("choices", [{}])[0].get("message", {}).get("content")
    except (urllib.error.HTTPError, urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError) as e:
        logger.warning("NeMo API call failed: %s", e)
        return None


# ============================================================================
# Sample generation functions
# ============================================================================

def _generate_dpo_pair(
    topic: str,
    endpoint: str,
    api_key: str,
    model: str,
) -> dict | None:
    """Generate one DPO preference pair for the given topic."""
    prompt = (
        f"Generate a JSON object with keys 'prompt', 'chosen', 'rejected' for a therapeutic AI training example "
        f"about {topic}. The 'prompt' is a client statement, 'chosen' is an empathetic therapist response, "
        f"'rejected' is a dismissive or harmful response. Output ONLY valid JSON."
    )
    raw = _call_nemo(prompt, endpoint, api_key, model)
    if not raw:
        return None

    # Extract JSON from possible markdown wrapping
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        pair = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not all(k in pair for k in ("prompt", "chosen", "rejected")):
        return None

    return pair


def _generate_niche_sample(
    category: str,
    category_info: dict,
    endpoint: str,
    api_key: str,
    model: str,
    target_response_type: str | None = None,
    style_profile: str = "warm_professional",
) -> dict | None:
    """Generate one niche category training sample.

    Optionally targets a specific response_type to balance distribution.
    """
    client_prompt = category_info["client_prompt_template"]
    therapist_prompt = _build_therapist_style_prompt(
        category_info["therapist_system_prompt"],
        response_type=target_response_type,
        style_profile=style_profile,
    )

    # Inject response type targeting if specified
    if target_response_type and target_response_type != "validation":
        rt_instructions = {
            "exploration": " Your response must ask a reflective question that deepens understanding.",
            "skill-teaching": " Your response must teach or suggest a specific coping technique or grounding exercise.",
            "psychoeducation": " Your response must share a brief educational insight about why this experience happens.",
            "safety": " Your response must include a safety check or crisis resource referral (e.g., 988).",
        }
        extra = rt_instructions.get(target_response_type, "")
        therapist_prompt = therapist_prompt + extra

    # Generate client statement
    instruction = _call_nemo(
        client_prompt,
        endpoint,
        api_key,
        model,
        system_prompt="You are helping create training data for a therapeutic AI. Generate realistic, first-person client statements that a therapist might hear in session. Output ONLY the client's spoken words.",
    )
    if not instruction:
        return None

    # Generate therapist response
    response_prompt = f"Respond therapeutically to this client statement: {instruction}"
    output = _call_nemo(response_prompt, endpoint, api_key, model, system_prompt=therapist_prompt)
    if not output:
        return None

    sample = {
        "instruction": instruction.strip(),
        "output": output.strip(),
        "category": category,
        "difficulty": determine_difficulty(instruction),
        "response_type": determine_response_type(output),
        "style_profile": style_profile,
    }

    return sample


def _generate_nightmare_sample(
    scenario_type: str,
    scenario_info: dict,
    endpoint: str,
    api_key: str,
    model: str,
) -> dict | None:
    """Generate one nightmare fuel (crisis edge case) training sample.

    These samples MUST include crisis resource referrals in the output.
    """
    templates = scenario_info.get("prompt_templates", [])
    if not templates:
        return None

    import random
    template = random.choice(templates)

    response_prompt = (
        f"A client says: '{template}' "
        f"Respond as a crisis-trained therapist. You MUST include a crisis resource "
        f"(988, Crisis Text Line, etc.) in your response. "
        f"Be warm, direct, and prioritize safety."
    )
    output = _call_nemo(response_prompt, endpoint, api_key, model)
    if not output:
        return None

    # Reject clearly malformed outputs (too short, JSON fragments, etc.)
    stripped = output.strip()
    if len(stripped) < 20:
        return None

    return {
        "instruction": template,
        "output": stripped,
        "is_training_edge_case": True,
        "scenario_type": scenario_type,
    }
# ============================================================================
# CLI and orchestration
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the SDG CLI."""
    parser = argparse.ArgumentParser(
        description="SDG pipeline using NVIDIA NeMo",
    )

    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        choices=["dpo_preference_pairs", "niche_category", "nightmare_fuel"],
        help="Generation scenario",
    )
    parser.add_argument(
        "--target_count",
        type=int,
        default=10,
        help="Target number of samples to generate",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path for output JSONL file",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="",
        help="Category for niche_category scenario",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=10,
        help="Maximum generation attempts before giving up",
    )
    parser.add_argument(
        "--nemo_endpoint",
        type=str,
        default="",
        help="NeMo API endpoint URL",
    )
    parser.add_argument(
        "--nemo_api_key",
        type=str,
        default="",
        help="NeMo API key",
    )
    parser.add_argument(
        "--nemo_model",
        type=str,
        default="mistral-nemo",
        help="NeMo model to use",
    )
    parser.add_argument(
        "--style_profile",
        type=str,
        default="warm_professional",
        choices=tuple(THERAPIST_STYLE_PROFILES.keys()),
        help="Therapist style profile to apply during niche generation",
    )
    parser.add_argument(
        "--style_audit",
        action="store_true",
        help="Run style audit against existing output file and exit",
    )
    parser.add_argument(
        "--style_audit_output",
        type=str,
        default="",
        help="Path to write style audit report JSON",
    )
    parser.add_argument(
        "--style_audit_limit",
        type=int,
        default=0,
        help="Max samples to audit in one pass (0 means all)",
    )

    return parser


# Response types for rotation during niche category generation
_ROTATION_RESPONSE_TYPES = ["validation", "exploration", "skill-teaching", "psychoeducation", "safety"]


def run_sdg(args: argparse.Namespace) -> None:
    """Main orchestration function for SDG generation.

    Handles three generation scenarios plus offline style-audit mode:
    DPO pairs, niche categories, nightmare fuel.
    """
    endpoint = args.nemo_endpoint or os.getenv("NEMO_ENDPOINT", "")
    api_key = args.nemo_api_key or os.getenv("NEMO_API_KEY", "") or os.getenv("NVIDIA_API_KEY", "")
    model = args.nemo_model

    if getattr(args, "style_audit", False):
        audit_report = run_style_audit(
            args.output_path,
            style_profile=getattr(args, "style_profile", "warm_professional"),
            sample_limit=getattr(args, "style_audit_limit", 0),
        )
        report_path = getattr(
            args,
            "style_audit_output",
            str(Path(args.output_path).with_suffix(".style_audit.json")),
        )
        if not report_path:
            report_path = str(Path(args.output_path).with_suffix(".style_audit.json"))
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(audit_report, f, indent=2)
        logger.info("Style audit complete: %s", report_path)
        return

    # Validate required parameters
    if args.scenario == "niche_category":
        if not args.category:
            logger.error("--category required for niche_category scenario")
            sys.exit(1)
        if args.category not in NICHE_CATEGORIES:
            logger.error("Unknown category: %s", args.category)
            sys.exit(1)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generated = 0
    filtered = 0
    iterations = 0
    max_iter = args.max_iterations
    existing_samples: list[dict] = []

    with open(output_path, "w", encoding="utf-8") as f:
        while generated < args.target_count and iterations < max_iter:
            iterations += 1

            try:
                if args.scenario == "dpo_preference_pairs":
                    sample = _generate_dpo_pair("therapeutic", endpoint, api_key, model)
                    if sample and "chosen" in sample:
                        if not _is_safe_output(sample["chosen"]):
                            filtered += 1
                            continue
                        f.write(json.dumps(sample) + "\n")
                        generated += 1
                        existing_samples.append(sample)

                elif args.scenario == "niche_category":
                    category_info = NICHE_CATEGORIES[args.category]
                    # Rotate response type to balance distribution
                    target_rt = _ROTATION_RESPONSE_TYPES[generated % len(_ROTATION_RESPONSE_TYPES)]
                    sample = _generate_niche_sample(
                        args.category, category_info, endpoint, api_key, model,
                        target_response_type=target_rt,
                        style_profile=args.style_profile,
                    )
                    if sample:
                        is_valid, reason = validate_sample(sample)
                        if not is_valid:
                            logger.debug("Filtered sample: %s", reason)
                            filtered += 1
                            continue

                        is_unique, dup_reason = _check_deduplication(sample, existing_samples)
                        if not is_unique:
                            logger.debug("Filtered duplicate: %s", dup_reason)
                            filtered += 1
                            continue

                # Safety filter intentionally removed for niche categories —
                # training data must include crisis situations and proper crisis
                # responses so the model learns to handle difficult clientele.
                        f.write(json.dumps(sample) + "\n")
                        generated += 1
                        existing_samples.append(sample)

                elif args.scenario == "nightmare_fuel":
                    scenario_types = list(NIGHTMARE_SCENARIOS.keys())
                    scenario_type = scenario_types[generated % len(scenario_types)]
                    scenario_info = NIGHTMARE_SCENARIOS[scenario_type]

                    sample = _generate_nightmare_sample(
                        scenario_type, scenario_info, endpoint, api_key, model,
                    )
                    if sample:
                        if not _crisis_resource_in_output(sample.get("output", "")):
                            logger.debug("Filtered: missing crisis resource")
                            filtered += 1
                            continue
                        f.write(json.dumps(sample) + "\n")
                        generated += 1
                        existing_samples.append(sample)

                else:
                    logger.error("Unknown scenario: %s", args.scenario)
                    sys.exit(1)

            except Exception as e:
                logger.error("Error during generation: %s", e)
                continue

    # Write generation report
    report = {
        "generated_at": datetime.now().isoformat(),
        "scenario": args.scenario,
        "category": args.category,
        "target_count": args.target_count,
        "generated_count": generated,
        "filtered_count": filtered,
        "filter_rate": filtered / (generated + filtered) if (generated + filtered) > 0 else 0,
        "iterations": iterations,
        "max_iterations": max_iter,
    }

    report_path = output_path.parent / "generation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(
        "Generated %d/%d samples (filtered %d, iterations %d)",
        generated, args.target_count, filtered, iterations,
    )


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = build_parser()
    args = parser.parse_args()

    run_sdg(args)


if __name__ == "__main__":
    main()
