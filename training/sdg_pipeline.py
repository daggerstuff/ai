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
import sys
import urllib.request
import urllib.error
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
]

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
    "safety": ["safe", "crisis", "call", "988", "emergency", "right now", "help is"],
    "skill-teaching": ["try", "skill", "technique", "grounding", "breathe", "exercise", "practice", "notice"],
    "psychoeducation": ["explain", "often", "many people", "common", "research", "understanding"],
    "exploration": ["what", "how", "tell me", "share", "explore", "when", "where", "who"],
    "validation": ["understand", "makes sense", "normal", "okay", "valid", "heard"],
}

# ============================================================================
# Crisis level definitions for content-derived difficulty
# ============================================================================

HIGH_DIFFICULTY_KEYWORDS = [
    "suicide", "kill myself", "end my life", "end it all",
    "self-harm", "hurt myself", "don't want to live",
    "no reason to live", "plan to", "overdose",
    "can't go on", "not worth living", "better off without me",
]

MEDIUM_DIFFICULTY_KEYWORDS = [
    "breaking down", "losing it", "can't cope", "overwhelm",
    "cutting", "relapse", "crisis", "emergency",
    "can't breathe", "losing control", "falling apart",
    "shutting down", "spiraling", "dissociat",
    "panic attack", "flashback", "trigger",
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
            "Respond as a real therapist in session. Output ONLY your spoken words — "
            "no stage directions, no analysis, no markdown. MAX 2 sentences, 70 words absolute max. "
            "CRITICAL: Do NOT start with 'It sounds', 'I hear', or 'It makes sense'. "
            "Instead open with: a direct observation ('Your mind...'), a gentle question ('When...'), "
            "a normalizing statement ('That fog...'), or a reflection of their exact words. "
            "Be warm, human, clinically grounded. Vary every response."
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
            "Respond as a somatic therapist. Output ONLY your spoken words — "
            "no stage directions, no analysis, no markdown. MAX 2 sentences, 70 words absolute max. "
            "CRITICAL: Do NOT start with 'It sounds', 'I hear', or 'It makes sense'. "
            "Instead open with: a body-focused question ('Where...'), an observation ('That tightness...'), "
            "or a grounding invitation ('Notice...'). Be warm and present. Vary every response."
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
            "Respond as an attachment-informed therapist. Output ONLY your spoken words — "
            "no stage directions, no analysis, no markdown. MAX 2 sentences, 70 words absolute max. "
            "CRITICAL: Do NOT start with 'It sounds', 'I hear', or 'It makes sense'. "
            "Instead open with: naming the pattern ('That push-pull...'), a gentle challenge ('What if...'), "
            "or reflecting their exact words. Be warm and non-judgmental. Vary every response."
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
            "Respond as a trauma therapist specializing in narcissistic abuse. Output ONLY your spoken words — "
            "no stage directions, no analysis, no markdown. MAX 2 sentences, 70 words absolute max. "
            "CRITICAL: Do NOT start with 'It sounds', 'I hear', or 'It makes sense'. "
            "Instead open with: affirming their perception ('Your gut...'), naming the pattern ('That doubt...'), "
            "or a trust-building statement. Be warm and affirming. Vary every response."
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
            "Respond as a grief therapist. Output ONLY your spoken words — "
            "no stage directions, no analysis, no markdown. MAX 2 sentences, 70 words absolute max. "
            "CRITICAL: Do NOT start with 'It sounds', 'I hear', or 'It makes sense'. "
            "Instead open with: honoring the loss ('That hole...'), a grief-specific reflection, "
            "or a gentle question about the person. Avoid cliches about 'healing'. Vary every response."
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
            "Respond as an eating disorder therapist. Output ONLY your spoken words — "
            "no stage directions, no analysis, no markdown. MAX 2 sentences, 70 words absolute max. "
            "CRITICAL: Do NOT start with 'It sounds', 'I hear', or 'It makes sense'. "
            "Instead open with: naming the pattern ('That cycle...'), a non-judgmental observation, "
            "or a safety check. Be warm and non-shaming. Never minimize their struggle. Vary every response."
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
            "Respond as an OCD specialist. Output ONLY your spoken words — "
            "no stage directions, no analysis, no markdown. MAX 2 sentences, 70 words absolute max. "
            "CRITICAL: Do NOT start with 'It sounds', 'I hear', or 'It makes sense'. "
            "Instead open with: immediate normalization ('Thoughts aren't actions.'), "
            "naming the OCD pattern ('That checking...'), or reducing shame. Be warm and reassuring. Vary every response."
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
            "Respond as a DBT-trained therapist. Output ONLY your spoken words — "
            "no stage directions, no analysis, no markdown. MAX 2 sentences, 70 words absolute max. "
            "CRITICAL: Do NOT start with 'It sounds', 'I hear', or 'It makes sense'. "
            "Instead open with: a grounding anchor ('Right now...'), validating the intensity directly, "
            "or offering a skill. Be calm and grounded. Prioritize safety. Vary every response."
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
            "Respond as a neurodiversity-affirming therapist. Output ONLY your spoken words — "
            "no stage directions, no analysis, no markdown. MAX 2 sentences, 70 words absolute max. "
            "CRITICAL: Do NOT start with 'It sounds', 'I hear', or 'It makes sense'. "
            "Instead open with: naming the masking ('That performing...'), an ND-affirming observation, "
            "or a question about needs. Avoid pathologizing. Be respectful and present. Vary every response."
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
            "Respond as a culturally competent therapist. Output ONLY your spoken words — "
            "no stage directions, no analysis, no markdown. MAX 2 sentences, 70 words absolute max. "
            "CRITICAL: Do NOT start with 'It sounds', 'I hear', or 'It makes sense'. "
            "Instead open with: honoring their background ('Your family...'), naming the complexity, "
            "or holding space for the contradiction. Be respectful and present. Vary every response."
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
) -> str | None:
    """Call NeMo API and return the generated text content.

    Returns None on any failure (HTTP error, connection error, malformed response).
    """
    url = f"{endpoint}/v1/chat/completions" if endpoint else "http://localhost:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
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
) -> dict | None:
    """Generate one niche category training sample.

    Optionally targets a specific response_type to balance distribution.
    """
    client_prompt = category_info["client_prompt_template"]
    therapist_prompt = category_info["therapist_system_prompt"]

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
    )
    if not instruction:
        return None

    # Generate therapist response
    response_prompt = f"Respond therapeutically to this client statement: {instruction}"
    output = _call_nemo(response_prompt, endpoint, api_key, model)
    if not output:
        return None

    sample = {
        "instruction": instruction.strip(),
        "output": output.strip(),
        "category": category,
        "difficulty": determine_difficulty(instruction),
        "response_type": determine_response_type(output),
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

    return parser


# Response types for rotation during niche category generation
_ROTATION_RESPONSE_TYPES = ["validation", "exploration", "skill-teaching", "psychoeducation", "safety"]


def run_sdg(args: argparse.Namespace) -> None:
    """Main orchestration function for SDG generation.

    Handles all three scenarios: DPO pairs, niche categories, nightmare fuel.
    Includes validation, deduplication, and safety filtering.
    """
    endpoint = args.nemo_endpoint or os.getenv("NEMO_ENDPOINT", "")
    api_key = args.nemo_api_key or os.getenv("NEMO_API_KEY", "") or os.getenv("NVIDIA_API_KEY", "")
    model = args.nemo_model

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

                        if not _is_safe_output(sample["output"]):
                            filtered += 1
                            continue

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
