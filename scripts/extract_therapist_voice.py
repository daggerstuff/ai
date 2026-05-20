#!/usr/bin/env python3
"""
Generic Therapist Voice Extraction System.

Extracts voice profiles from YouTube therapist transcripts, generates
scored training conversations, and reports clinical validity quality metrics.

Usage:
    # Process a single channel
    python scripts/extract_therapist_voice.py --channel DoctorRamani

    # Process all discovered channels with ingested transcripts
    python scripts/extract_therapist_voice.py --all

    # Process from a specific transcript source
    python scripts/extract_therapist_voice.py --channel TimFletcher --source-dir ai/data/transcripts/ingested

    # Process all channels and score everything
    python scripts/extract_therapist_voice.py --all --score

    # Generate synthetic conversations with clinical validity scoring
    python scripts/extract_therapist_voice.py --channel DrDanielFox --num-conversations 50
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("extract_therapist_voice")

try:
    from training.clinical_validity_scorer import ClinicalValidityScorer

    SCORER_AVAILABLE = True
except ImportError:
    ClinicalValidityScorer = None
    SCORER_AVAILABLE = False

MIN_SENTENCE_WORDS = 2
MAX_MARKED_SENTENCE_LENGTH = 200
MIN_COMMON_PHRASE_COUNT = 30
DEFAULT_CONVERSATIONS = 50
INGESTED_DIR = Path("ai/data/transcripts/ingested")
TRANSCRIPTS_DIR = Path("ai/data/transcripts/transcripts")
OUTPUT_BASE = Path("ai/data")

CHANNEL_CONFIGS: dict[str, dict] = {
    "DocSnipes": {
        "name": "Doc Snipes",
        "signature": "doc_snipes",
        "description": "Licensed clinical counselor focusing on codependency, validation, and mental health education",
        "style": "professional_educational_supportive",
        "expertise": ["codependency", "validation", "mental_health_education", "addiction_recovery", "family_dynamics"],
        "approach": "clinical_educational",
    },
    "DoctorRamani": {
        "name": "DoctorRamani",
        "signature": "doctor_ramani",
        "description": "Clinical psychologist specializing in narcissistic abuse, personality disorders, and mental health education",
        "style": "authoritative_educational_compassionate",
        "expertise": ["narcissistic_abuse", "personality_disorders", "toxic_relationships", "boundaries", "trauma"],
        "approach": "clinical_education",
    },
    "TimFletcher": {
        "name": "Tim Fletcher",
        "signature": "tim_fletcher",
        "description": "Compassionate therapist specializing in complex trauma, PTSD recovery, and nervous system regulation",
        "style": "compassionate_educational_step_by_step",
        "expertise": ["complex_trauma", "PTSD", "nervous_system_regulation", "shame", "codependency"],
        "approach": "trauma_informed_psychoeducation",
    },
    "PatrickTeahan": {
        "name": "Patrick Teahan",
        "signature": "patrick_teahan",
        "description": "Licensed clinical social worker specializing in childhood trauma, PTSD, and family dynamics",
        "style": "empathetic_direct_validating",
        "expertise": ["childhood_trauma", "family_dynamics", "PTSD", "emotional_abuse", "inner_child"],
        "approach": "trauma_informed_psychoeducation",
    },
    "TherapyinaNutshell": {
        "name": "Therapy in a Nutshell",
        "signature": "therapy_in_a_nutshell",
        "description": "Emma McAdam, licensed therapist, making mental health education accessible with evidence-based techniques",
        "style": "warm_accessible_practical",
        "expertise": ["CBT", "DBT", "anxiety", "emotion_regulation", "mindfulness"],
        "approach": "evidence_based_psychoeducation",
    },
    "HeidiPriebe": {
        "name": "Heidi Priebe",
        "signature": "heidi_priebe",
        "description": "Psychoeducator specializing in complex trauma, attachment theory, shame, and self-compassion",
        "style": "deep_reflective_psychoeducational",
        "expertise": ["complex_trauma", "attachment_theory", "shame", "self_compassion", "CPTSD"],
        "approach": "depth_oriented_psychoeducation",
    },
    "CrappyChildhoodFairy": {
        "name": "Crappy Childhood Fairy",
        "signature": "crappy_childhood_fairy",
        "description": "Anna Runkle, trauma recovery specialist focusing on CPTSD, dysregulation, and childhood trauma healing",
        "style": "direct_compassionate_practical",
        "expertise": ["CPTSD", "dysregulation", "childhood_trauma", "nervous_system", "healing"],
        "approach": "trauma_informed_practical",
    },
    "Dr.DanielFox": {
        "name": "Dr. Daniel Fox",
        "signature": "dr_daniel_fox",
        "description": "Licensed clinical psychologist specializing in personality disorders, BPD, and narcissism",
        "style": "clinical_authoritative_educational",
        "expertise": ["personality_disorders", "BPD", "narcissism", "attachment", "psychopathology"],
        "approach": "clinical_educational",
    },
    "Dr.ToddGrande": {
        "name": "Dr. Todd Grande",
        "signature": "dr_todd_grande",
        "description": "Licensed professional counselor specializing in personality disorders, trauma, and clinical assessment",
        "style": "analytical_balanced_educational",
        "expertise": ["personality_disorders", "trauma", "clinical_assessment", "diagnosis", "psychopathology"],
        "approach": "clinical_analytical",
    },
    "Dr.ScottEilers": {
        "name": "Dr. Scott Eilers",
        "signature": "dr_scott_eilers",
        "description": "Licensed psychologist focusing on depression, anxiety, and evidence-based treatment approaches",
        "style": "warm_clinical_hopeful",
        "expertise": ["depression", "anxiety", "evidence_based_treatment", "resilience", "mental_health"],
        "approach": "clinical_warm",
    },
    "JerryWise": {
        "name": "Jerry Wise",
        "signature": "jerry_wise",
        "description": "Licensed therapist specializing in narcissistic family systems, trauma recovery, and scapegoat healing",
        "style": "empathetic_analytical_educational",
        "expertise": ["narcissistic_family_systems", "scapegoat_recovery", "complex_trauma", "family_dynamics", "emotional_abuse"],
        "approach": "trauma_informed_systemic",
    },
    "SurvivingNarcissism": {
        "name": "Surviving Narcissism",
        "signature": "surviving_narcissism",
        "description": "Licensed therapist Dr. Carter focusing on narcissistic abuse recovery, trauma bonding, and boundary setting",
        "style": "clear_direct_educational",
        "expertise": ["narcissistic_abuse", "trauma_bonding", "boundary_setting", "gaslighting", "recovery_strategies"],
        "approach": "clinical_educational",
    },
    "MedCircle": {
        "name": "MedCircle",
        "signature": "medcircle",
        "description": "Clinical psychology education platform featuring licensed psychologists covering personality disorders, trauma, and mental health",
        "style": "clinical_interview_educational",
        "expertise": ["personality_disorders", "trauma", "clinical_psychology", "mental_health_education", "diagnosis"],
        "approach": "clinical_educational",
    },
    "NavigatingNarcissism": {
        "name": "Navigating Narcissism",
        "signature": "navigating_narcissism",
        "description": "Dr. Jennifer Freyd and licensed therapists specializing in betrayal trauma, narcissistic abuse, and institutional betrayal",
        "style": "clinical_compassionate_research_based",
        "expertise": ["betrayal_trauma", "narcissistic_abuse", "institutional_betrayal", "attachment", "trauma_recovery"],
        "approach": "research_informed_clinical",
    },
    "Dr.KimSage": {
        "name": "Dr. Kim Sage",
        "signature": "dr_kim_sage",
        "description": "Licensed clinical psychologist specializing in narcissistic abuse, complex trauma, attachment wounds, and high-conflict personality dynamics",
        "style": "clinical_warm_analytical",
        "expertise": ["narcissistic_abuse", "complex_trauma", "attachment_wounds", "personality_dynamics", "emotional_abuse"],
        "approach": "clinical_depth_oriented",
    },
    "KerryMcAvoy": {
        "name": "Kerry McAvoy",
        "signature": "kerry_mcavoy",
        "description": "Licensed mental health clinician and PhD specializing in abusive relationship dynamics, narcissism, and gender-based violence",
        "style": "clinical_advocacy_educational",
        "expertise": ["abusive_relationships", "narcissistic_abuse", "gender_based_violence", "trauma_informed_care", "boundaries"],
        "approach": "clinical_advocacy",
    },
    "KristinSnowden": {
        "name": "Kristin Snowden",
        "signature": "kristin_snowden",
        "description": "Licensed therapist specializing in betrayal trauma recovery, neurobiology of trauma, and healing from infidelity",
        "style": "compassionate_neuroscience_informed",
        "expertise": ["betrayal_trauma", "infidelity_recovery", "trauma_neurobiology", "attachment_repair", "relationship_healing"],
        "approach": "trauma_informed_neuroscience",
    },
    "JimHopper": {
        "name": "Jim Hopper",
        "signature": "jim_hopper",
        "description": "Trauma expert and PhD psychologist specializing in sexual assault, the neurobiology of trauma, and survivor advocacy",
        "style": "educational_research_based_clear",
        "expertise": ["trauma_neurobiology", "sexual_assault", "survivor_advocacy", "dissociation", "somatic_healing"],
        "approach": "research_informed_psychoeducation",
    },
    "IreneLyon": {
        "name": "Irene Lyon",
        "signature": "irene_lyon",
        "description": "Somatic trauma healing expert specializing in nervous system regulation, developmental trauma, and sensorimotor approaches",
        "style": "somatic_educational_empowering",
        "expertise": ["somatic_trauma_healing", "nervous_system_regulation", "developmental_trauma", "sensorimotor_therapy", "polyvagal_theory"],
        "approach": "somatic_trauma_informed",
    },
    "ChristopherGermer": {
        "name": "Christopher Germer",
        "signature": "christopher_germer",
        "description": "Clinical psychologist and PhD specializing in self-compassion, mindfulness-based therapy, and shame resilience",
        "style": "warm_contemplative_research_grounded",
        "expertise": ["self_compassion", "mindfulness_therapy", "shame_resilience", "compassion_focused_therapy", "meditation"],
        "approach": "mindfulness_compassion_based",
    },
    "TherapyChatPodcast": {
        "name": "Therapy Chat Podcast",
        "signature": "therapy_chat_podcast",
        "description": "Podcast hosted by licensed therapist featuring expert interviews on complex trauma, attachment, and narcissistic abuse recovery",
        "style": "conversational_expert_interview",
        "expertise": ["complex_trauma", "attachment_theory", "narcissistic_abuse", "EMDR", "trauma_recovery"],
        "approach": "interview_based_psychoeducation",
    },
    "TherapyDecoded": {
        "name": "Therapy Decoded",
        "signature": "therapy_decoded",
        "description": "Psychoeducation channel demystifying therapy concepts, trauma responses, and mental health through accessible clinical explanations",
        "style": "accessible_educational_normalizing",
        "expertise": ["trauma_education", "nervous_system", "mental_health_literacy", "therapy_demystification", "coping_skills"],
        "approach": "accessible_psychoeducation",
    },
    "CommonEgo": {
        "name": "Common Ego",
        "signature": "common_ego",
        "description": "Psychology education channel covering narcissism, attachment theory, codependency, and personal growth from a therapeutic perspective",
        "style": "engaging_relatable_educational",
        "expertise": ["narcissism", "attachment_theory", "codependency", "emotional_intelligence", "personal_growth"],
        "approach": "accessible_psychoeducation",
    },
    "ForrestHanson": {
        "name": "Forrest Hanson",
        "signature": "forrest_hanson",
        "description": "Psychology educator and host covering attachment theory, trauma healing, and practical mental health tools with clinical experts",
        "style": "conversational_expert_engaged",
        "expertise": ["attachment_theory", "trauma_healing", "self_compassion", "nervous_system_regulation", "resilience"],
        "approach": "interview_based_psychoeducation",
    },
    "PhoenixTraumaCenter": {
        "name": "Phoenix Trauma Center",
        "signature": "phoenix_trauma_center",
        "description": "Dr. Scott Giacomucci, trauma specialist and licensed clinical social worker focusing on complex PTSD, APA guidelines, and trauma treatment",
        "style": "clinical_authoritative_professional",
        "expertise": ["complex_PTSD", "trauma_treatment", "evidence_based_practice", "clinical_guidelines", "somatic_therapy"],
        "approach": "clinical_research_informed",
    },
    "EckhartTolle": {
        "name": "Eckhart Tolle",
        "signature": "eckhart_tolle",
        "description": "Spiritual teacher and author specializing in mindfulness, presence, and transcending ego-based consciousness",
        "style": "contemplative_wisdom_grounded",
        "expertise": ["mindfulness", "presence", "spiritual_awakening", "ego_transcendence", "stillness"],
        "approach": "contemplative_spiritual",
    },
    "ChrisWilliamson": {
        "name": "Chris Williamson",
        "signature": "chris_williamson",
        "description": "Podcast host of Modern Wisdom interviewing leading psychologists, scientists, and thinkers on mental health, human potential, and well-being",
        "style": "conversational_curious_interview",
        "expertise": ["mental_health", "human_potential", "positive_psychology", "personal_development", "behavioral_science"],
        "approach": "interview_based_exploration",
    },
    "Psych2Go": {
        "name": "Psych2Go",
        "signature": "psych2go",
        "description": "Animated psychology education channel making mental health concepts accessible through engaging visual content",
        "style": "accessible_educational_animated",
        "expertise": ["mental_health_education", "psychology_literacy", "self_improvement", "emotional_intelligence", "relationship_health"],
        "approach": "accessible_psychoeducation",
    },
    "WuWeiWisdom": {
        "name": "Wu Wei Wisdom",
        "signature": "wu_wei_wisdom",
        "description": "Spiritual and personal growth channel exploring narcissistic abuse recovery, inner child healing, and self-worth",
        "style": "empathetic_contemplative_educational",
        "expertise": ["narcissistic_abuse_recovery", "inner_child_healing", "self_worth", "spiritual_growth", "emotional_healing"],
        "approach": "contemplative_healing",
    },
    "CarolineMyss": {
        "name": "Caroline Myss",
        "signature": "caroline_myss",
        "description": "Medical intuitive, mystic, and author specializing in energy anatomy, spiritual growth, and the intersection of consciousness and health",
        "style": "mystical_direct_wisdom",
        "expertise": ["energy_anatomy", "spiritual_development", "consciousness_healing", "intuition", "sacred_contracts"],
        "approach": "mystical_wisdom_tradition",
    },
    "RebeccaMandeville": {
        "name": "Rebecca C. Mandeville",
        "signature": "rebecca_mandeville",
        "description": "Licensed marriage and family therapist specializing in family scapegoating abuse, family mobbing, and narcissistic family systems",
        "style": "clinical_direct_validating",
        "expertise": ["scapegoat_abuse", "family_mobbing", "narcissistic_family_systems", "toxic_family_dynamics", "trauma_recovery"],
        "approach": "trauma_informed_systemic",
    },
    "MicheleLeeNieves": {
        "name": "Michele Lee Nieves",
        "signature": "michele_lee_nieves",
        "description": "Life coach and narcissistic abuse recovery specialist focusing on covert narcissism, emotional abuse, and rebuilding self-worth",
        "style": "compassionate_direct_coaching",
        "expertise": ["covert_narcissism", "emotional_abuse_recovery", "self_worth_restoration", "boundary_setting", "trauma_coaching"],
        "approach": "coaching_informed_recovery",
    },
    "SandstoneCare": {
        "name": "Sandstone Care",
        "signature": "sandstone_care",
        "description": "Mental health treatment center for teens and young adults specializing in emotional dysregulation, ADHD, and substance use",
        "style": "clinical_warm_educational",
        "expertise": ["emotional_dysregulation", "adhd", "youth_mental_health", "substance_use", "dialectical_behavior_therapy"],
        "approach": "clinical_developmental",
    },
    "SoundsTrue": {
        "name": "Sounds True",
        "signature": "sounds_true",
        "description": "Multimedia publishing company featuring leading teachers in psychology, spirituality, and personal transformation through interview-driven content",
        "style": "interview_based_expert_showcase",
        "expertise": ["spiritual_psychology", "trauma_healing", "mindfulness", "personal_transformation", "expert_interviews"],
        "approach": "interview_based_psychoeducation",
    },
}


# ── topic banks per expertise area ──────────────────────────────────────────

TOPIC_BANK: dict[str, list[str]] = {
    "general": [
        "Managing anxiety in daily life and building resilience",
        "Understanding emotional triggers and developing coping strategies",
        "Building self-compassion and reducing self-criticism",
        "Navigating relationship challenges and communication patterns",
        "Developing healthy boundaries in personal and professional relationships",
        "Coping with grief, loss, and life transitions",
        "Understanding attachment patterns and their impact on relationships",
        "Building self-esteem and overcoming imposter syndrome",
        "Managing stress and preventing burnout",
        "Developing mindfulness and grounding practices",
    ],
    "trauma": [
        "Understanding the nervous system's role in trauma responses",
        "Healing from childhood emotional neglect and its long-term effects",
        "Managing hypervigilance and creating safety in the body",
        "Rebuilding trust after betrayal and relational trauma",
        "Processing shame and reclaiming self-worth after trauma",
        "Understanding freeze, fight, flight, and fawn trauma responses",
        "Healing attachment wounds and developing secure relationships",
        "Managing trauma triggers with grounding and regulation techniques",
        "Recovering from complex PTSD and developmental trauma",
        "Integrating traumatic memories without becoming overwhelmed",
    ],
    "personality_disorders": [
        "Understanding narcissistic personality traits and protecting yourself",
        "Setting boundaries with emotionally immature or manipulative people",
        "Recognizing gaslighting and psychological manipulation tactics",
        "Healing from narcissistic abuse and rebuilding identity",
        "Understanding borderline personality dynamics in relationships",
        "Differentiating between healthy conflict and toxic patterns",
        "Recovering from codependency and reclaiming autonomy",
        "Understanding the impact of growing up with a narcissistic parent",
        "Navigating no-contact and low-contact strategies with difficult family",
        "Building emotional resilience after toxic relationship patterns",
    ],
    "cbt_dbt": [
        "Using cognitive restructuring to challenge negative thought patterns",
        "Developing distress tolerance skills for intense emotions",
        "Implementing behavioral activation to overcome depression",
        "Using mindfulness techniques for emotion regulation",
        "Applying exposure principles to reduce avoidance and fear",
        "Building interpersonal effectiveness with DEAR MAN and GIVE skills",
        "Using thought records to identify and reframe cognitive distortions",
        "Practicing radical acceptance in difficult situations",
        "Developing a personalized coping strategies toolkit",
        "Using opposite action to change painful emotional patterns",
    ],
    "attachment": [
        "Understanding your attachment style and its origins",
        "Healing anxious attachment and building relationship security",
        "Moving from avoidant to secure attachment patterns",
        "Reparenting the inner child and meeting unmet needs",
        "Building earned secure attachment through therapeutic relationships",
        "Understanding disorganized attachment and healing from fear",
        "Developing emotional literacy and communicating needs",
        "Healing from relational trauma through attuned relationships",
        "Breaking intergenerational cycles of insecure attachment",
        "Building trust in yourself and others after attachment wounds",
    ],
}


@dataclass
class ChannelResult:
    name: str
    transcripts: list[str] = field(default_factory=list)
    transcript_titles: list[str] = field(default_factory=list)
    voice_profile: dict = field(default_factory=dict)
    conversations: list[dict] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    score_detail: list[dict] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)

    @property
    def pass_rate(self) -> float:
        if not self.scores:
            return 0.0
        return sum(1 for s in self.scores if s >= 0.5) / len(self.scores)

    @property
    def high_quality_rate(self) -> float:
        if not self.scores:
            return 0.0
        return sum(1 for s in self.scores if s >= 0.7) / len(self.scores)


# ═══════════════════════════════════════════════════════════════════════════════
#  Voice Profile Extraction
# ═══════════════════════════════════════════════════════════════════════════════


def extract_voice_profile(texts: list[str], channel_name: str) -> dict:
    """Extract voice patterns from a list of transcript texts."""
    profile: dict[str, Counter | list] = {
        "sentence_starters": Counter(),
        "transition_phrases": Counter(),
        "empathy_markers": Counter(),
        "common_phrases": Counter(),
        "analogies": [],
        "examples": [],
        "teaching_patterns": [],
    }

    all_text = "\n\n".join(texts)

    for text in texts:
        _analyze_single(text, profile)

    _extract_common_phrases(all_text, profile)
    _extract_teaching_style(all_text, profile)

    profile_report = _build_profile_report(channel_name, profile)

    return {
        "profile_raw": {
            "sentence_starters": dict(profile["sentence_starters"].most_common(50)),
            "transition_phrases": dict(profile["transition_phrases"].most_common(30)),
            "empathy_markers": dict(profile["empathy_markers"].most_common(30)),
            "common_phrases": dict(profile["common_phrases"].most_common(100)),
            "analogies": profile["analogies"][:50],
            "examples": profile["examples"][:50],
            "teaching_patterns": profile["teaching_patterns"],
        },
        "report": profile_report,
    }


def _analyze_single(text: str, profile: dict):
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    for sentence in sentences:
        words = sentence.split()
        if len(words) >= MIN_SENTENCE_WORDS:
            starter = " ".join(words[:2])
            profile["sentence_starters"][starter] += 1

        transitions = [
            "And so", "Now", "So", "But", "And then", "What happens",
            "Let me", "Think about", "Imagine", "What I find",
            "One of the things", "The reality is", "What we see",
            "The truth is", "Here's the thing", "The key is",
            "What I want you to", "One of the key", "It's important to",
        ]
        for t in transitions:
            if sentence.lower().startswith(t.lower()):
                profile["transition_phrases"][t] += 1

        empathy_patterns = [
            "I understand", "I know", "I get it", "That's painful",
            "That's hard", "You might feel", "Many people",
            "Some of you", "For many", "What you're going through",
            "It makes sense that", "It's understandable", "You're not alone",
            "That must be", "I hear you", "I can see how",
        ]
        for p in empathy_patterns:
            if p.lower() in sentence.lower():
                profile["empathy_markers"][p] += 1

        if any(m in sentence.lower() for m in ["like a", "as if", "imagine", "think of"]):
            if len(sentence) < MAX_MARKED_SENTENCE_LENGTH:
                profile["analogies"].append(sentence)

        if any(m in sentence.lower() for m in ["let's say", "for example", "think back to"]):
            if len(sentence) < MAX_MARKED_SENTENCE_LENGTH:
                profile["examples"].append(sentence)


def _extract_common_phrases(text: str, profile: dict):
    words = text.lower().split()
    for i in range(len(words) - 2):
        phrase = " ".join(words[i:i + 3])
        profile["common_phrases"][phrase] += 1


def _extract_teaching_style(text: str, profile: dict):
    patterns = [
        "First", "Second", "Third",
        "What happens is", "The reality is", "What we find",
        "One of the key", "It's important to understand",
        "Let me give you an example", "Think about this",
        "What I mean by that", "Here's what I want you to",
        "The reason for this", "What we know from",
    ]
    for pattern in patterns:
        count = text.lower().count(pattern.lower())
        if count > 0:
            profile["teaching_patterns"].append({
                "pattern": pattern,
                "frequency": count,
            })


def _build_profile_report(channel_name: str, profile: dict) -> str:
    report = [f"# {channel_name} Voice Profile\n"]
    total = len(profile.get("analogies", [])) + len(profile.get("examples", []))
    report.append(f"**Analyzed**: {len(profile.get('sentence_starters', {}))} unique patterns\n\n")

    report.append("## Top Sentence Starters\n")
    for starter, count in profile["sentence_starters"].most_common(20):
        report.append(f'- **"{starter}..."** ({count} times)\n')

    report.append("\n## Transition Phrases\n")
    for phrase, count in profile["transition_phrases"].most_common(15):
        report.append(f'- **"{phrase}"** ({count} times)\n')

    report.append("\n## Empathy & Connection Markers\n")
    for marker, count in profile["empathy_markers"].most_common(15):
        report.append(f'- **"{marker}"** ({count} times)\n')

    report.append("\n## Sample Analogies & Metaphors\n")
    for analogy in profile["analogies"][:10]:
        report.append(f"- {analogy}\n")

    report.append("\n## Sample Examples\n")
    for example in profile["examples"][:10]:
        report.append(f"- {example}\n")

    report.append("\n## Common 3-Word Phrases\n")
    for phrase, count in profile["common_phrases"].most_common(30):
        if count > MIN_COMMON_PHRASE_COUNT:
            report.append(f'- "{phrase}" ({count} times)\n')

    return "".join(report)


# ═══════════════════════════════════════════════════════════════════════════════
#  Conversation Generation
# ═══════════════════════════════════════════════════════════════════════════════


def _get_config(channel_key: str) -> dict | None:
    for ck, cfg in CHANNEL_CONFIGS.items():
        if ck.lower() == channel_key.lower():
            return cfg
    return None


def _resolve_channel_key(name: str) -> str | None:
    for ck in CHANNEL_CONFIGS:
        if ck.lower() == name.lower():
            return ck
    return None


def discover_channels(source_dir: Path = INGESTED_DIR) -> list[str]:
    """Discover channels that have ingested markdown transcripts."""
    found: set[str] = set()
    if not source_dir.exists():
        logger.warning("Source dir %s not found", source_dir)
        return []

    for fname in os.listdir(source_dir):
        if not fname.endswith(".md"):
            continue
        for ck in CHANNEL_CONFIGS:
            if fname.startswith(ck):
                found.add(ck)

    listed = sorted(found)
    logger.info("Discovered %d channels with transcripts: %s", len(listed), listed)
    return listed


def load_channel_transcripts(
    channel_key: str,
    source_dir: Path = INGESTED_DIR,
) -> list[tuple[str, str]]:
    """Load transcripts for a channel. Returns list of (title, content)."""
    results: list[tuple[str, str]] = []
    if not source_dir.exists():
        return results

    for fname in os.listdir(source_dir):
        if not fname.endswith(".md"):
            continue
        if not fname.startswith(channel_key):
            continue
        title = fname.removesuffix(".md")
        with open(source_dir / fname, encoding="utf-8") as f:
            content = f.read()
        results.append((title, content))

    results.sort(key=lambda x: x[0])
    return results


def generate_conversation_from_transcript(
    title: str,
    content: str,
    config: dict,
) -> dict:
    """Generate a training conversation from a single transcript."""
    system_prompt = (
        f"You are {config['name']}. {config['description']}. "
        f"Your approach is {config['approach']}. "
        f"Your expertise includes: {', '.join(config['expertise'])}. "
        "Respond to the client's questions using your therapeutic voice and expertise."
    )

    return {
        "conversation_id": f"{config['signature']}_{title}",
        "stage": "stage4_voice_persona",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Can you help me understand: {title}?"},
            {"role": "assistant", "content": content},
        ],
        "metadata": {
            "source": f"{config['signature']}_transcripts",
            "source_family": "stage4_voice_persona",
            "voice_signature": f"{config['signature']}_v1",
            "persona_id": config["signature"],
            "personality_markers": {
                "style": config["style"],
                "approach": config["approach"],
                "expertise_areas": config["expertise"],
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def generate_synthetic_conversations(
    config: dict,
    num_conversations: int,
    profile: dict | None = None,
) -> list[dict]:
    """Generate fallback synthetic conversations when real transcripts are unavailable."""
    conversations: list[dict] = []
    raw = profile.get("profile_raw", {}) if profile else {}
    starters = list(raw.get("sentence_starters", {}).keys()) or [
        "I understand", "Let's look at", "The key is", "What we find",
    ]
    transitions = list(raw.get("transition_phrases", {}).keys()) or [
        "Now", "And so", "What happens", "The truth is",
    ]
    empathy = list(raw.get("empathy_markers", {}).keys()) or [
        "I understand", "That makes sense", "Many people",
    ]
    analogies = raw.get("analogies", []) or [
        "like building a muscle, it takes consistent practice",
        "like learning a new language for your nervous system",
        "like tending a garden, you can't rush the growth",
    ]
    examples = raw.get("examples", []) or [
        "for example, when a situation triggers an old wound",
        "let's say you notice your heart racing in a meeting",
        "think back to a time when you felt genuinely safe",
    ]

    topics = _get_topics_for_expertise(config.get("expertise", []))

    system_prompt = (
        f"You are {config['name']}. {config['description']}. "
        f"Your approach is {config['approach']}. "
        "Respond with deep empathy, clinical insight, and practical guidance."
    )

    for i in range(num_conversations):
        topic = topics[i % len(topics)]
        conversation = _build_synthetic_dialogue(
            i, topic, config, starters, transitions, empathy, analogies, examples,
        )
        conversation["messages"].insert(0, {"role": "system", "content": system_prompt})
        conversations.append(conversation)

    return conversations


def _get_topics_for_expertise(expertise: list[str]) -> list[str]:
    topics: list[str] = []
    for area in expertise:
        if "trauma" in area or "PTSD" in area or "CPTSD" in area or "nervous" in area:
            topics.extend(TOPIC_BANK["trauma"])
        if "personality" in area or "narcissis" in area or "BPD" in area:
            topics.extend(TOPIC_BANK["personality_disorders"])
        if "CBT" in area or "DBT" in area or "cbt" in area or "dbt" in area:
            topics.extend(TOPIC_BANK["cbt_dbt"])
        if "attachment" in area:
            topics.extend(TOPIC_BANK["attachment"])
    if not topics:
        topics = TOPIC_BANK["general"]
    return topics


def _build_synthetic_dialogue(
    index: int,
    topic: str,
    config: dict,
    starters: list[str],
    transitions: list[str],
    empathy: list[str],
    analogies: list[str],
    examples: list[str],
) -> dict:
    dialogue_templates = [
        # Template 1: Client seeks understanding
        {
            "user": f"I've been struggling with something and I'm not sure how to understand it. {topic}.",
            "assistant": (
                f"{random.choice(starters)}. {random.choice(empathy)}. "
                f"{random.choice(transitions)}, let me help you understand this. "
                f"{random.choice(examples)}. "
                "The first step is recognizing that your response makes sense given what you've been through."
            ),
            "follow_ups": [
                {
                    "user": "That does make sense, but how do I actually start working on it?",
                    "assistant": (
                        f"Great question. {random.choice(transitions)}, "
                        f"let's break this down into manageable pieces. "
                        f"Think of it {random.choice(analogies)}. "
                        f"{random.choice(empathy)}. "
                        "Start with awareness, then build skills one at a time."
                    ),
                },
                {
                    "user": "What if I try but keep falling back into old patterns?",
                    "assistant": (
                        f"That's completely normal. {random.choice(empathy)}. "
                        f"{random.choice(transitions)}, recovery isn't linear. "
                        f"{random.choice(examples)}. "
                        "Every time you notice the pattern and redirect, you're rewiring that response. "
                        "Progress, not perfection."
                    ),
                },
            ],
        },
        # Template 2: Client in distress
        {
            "user": f"I'm really struggling right now. {topic}. I feel stuck and don't know what to do.",
            "assistant": (
                f"{random.choice(empathy)}. {random.choice(starters)}. "
                "First, let's take a breath together. You reaching out is already a step forward. "
                f"{random.choice(transitions)}, what you're describing is a common response. "
                f"{random.choice(examples)}. "
                "We can work through this step by step."
            ),
            "follow_ups": [
                {
                    "user": "I just want the pain to stop. What can I do right now?",
                    "assistant": (
                        f"{random.choice(empathy)}. Right now, let's focus on grounding. "
                        "Name three things you can see, two you can touch, one you can hear. "
                        f"{random.choice(transitions)}, this helps your nervous system know you're safe right now. "
                        f"Think of it {random.choice(analogies)}."
                    ),
                },
                {
                    "user": "That helped a bit. How do I keep from getting this overwhelmed again?",
                    "assistant": (
                        f"{random.choice(starters)}. {random.choice(empathy)}. "
                        f"{random.choice(transitions)}, building resilience is like {random.choice(analogies)}. "
                        f"{random.choice(examples)}. "
                        "We'll develop a toolkit you can use including grounding, breathing, and support strategies."
                    ),
                },
            ],
        },
        # Template 3: Client seeking practical tools
        {
            "user": f"I need practical tools for {topic.lower()}. What actually works?",
            "assistant": (
                f"{random.choice(starters)}. {random.choice(empathy)}. "
                f"{random.choice(transitions)}, there are several evidence-based approaches. "
                f"{random.choice(examples)}. "
                "First, let me explain why these tools work, then we'll practice together."
            ),
            "follow_ups": [
                {
                    "user": "That sounds helpful but hard to do when I'm in the middle of it.",
                    "assistant": (
                        f"You're right, it is hard at first. {random.choice(empathy)}. "
                        f"{random.choice(transitions)}, that's why we practice when you're calm first. "
                        f"Think of it {random.choice(analogies)}. "
                        "Start with one small technique, practice it daily, and build from there."
                    ),
                },
            ],
        },
    ]

    template = random.choice(dialogue_templates)
    conversation = [
        {"role": "client", "content": template["user"]},
        {"role": "therapist", "content": template["assistant"]},
    ]
    for fu in template["follow_ups"]:
        conversation.append({"role": "client", "content": fu["user"]})
        conversation.append({"role": "therapist", "content": fu["assistant"]})

    return {
        "conversation_id": f"{config['signature']}_synthetic_{index:04d}",
        "stage": "stage4_voice_persona",
        "messages": conversation,
        "metadata": {
            "source": f"{config['signature']}_synthetic",
            "source_family": "stage4_voice_persona",
            "voice_signature": f"{config['signature']}_v1",
            "persona_id": config["signature"],
            "personality_markers": {
                "style": config["style"],
                "approach": config["approach"],
                "expertise_areas": config.get("expertise", []),
            },
            "topic": topic,
            "index": index,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Scoring
# ═══════════════════════════════════════════════════════════════════════════════


def score_conversations(conversations: list[dict]) -> tuple[list[float], list[dict]]:
    """Score each conversation's therapist responses using ClinicalValidityScorer."""
    if not SCORER_AVAILABLE:
        logger.warning("ClinicalValidityScorer not available; skipping scoring")
        return [], []

    scores: list[float] = []
    details: list[dict] = []

    for conv in conversations:
        therapist_parts: list[str] = []
        for msg in conv.get("messages", []):
            if msg.get("role") in ("therapist", "assistant"):
                content = msg.get("content", "")
                if content.strip():
                    therapist_parts.append(content)
        combined = " ".join(therapist_parts)
        score = ClinicalValidityScorer.score(combined)
        detail = ClinicalValidityScorer.score_detail(combined)
        scores.append(score)
        details.append(detail)

    return scores, details


def annotate_conversations(conversations: list[dict], scores: list[float], details: list[dict]):
    """Add clinical validity scores to conversation metadata."""
    for conv, score, detail in zip(conversations, scores, details):
        conv.setdefault("metadata", {})["clinical_validity"] = {
            "score": round(score, 4),
            "dimensions": {k: round(v, 4) for k, v in detail.items()},
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  I/O
# ═══════════════════════════════════════════════════════════════════════════════


def _derive_communication_patterns(profile_data: dict, config: dict | None) -> list[str]:
    """Derive human-readable communication patterns from extracted profile data."""
    config = config or {}
    patterns = []

    transitions = profile_data.get("transition_phrases", {})
    total_trans = sum(transitions.values())
    if total_trans > 0:
        explanatory = sum(
            transitions.get(t, 0) for t in
            ["So", "Let me", "Here's the thing", "The key is", "What I want you to",
             "One of the key", "It's important to", "Think about"]
        )
        analytical = sum(
            transitions.get(t, 0) for t in
            ["But", "What happens", "The reality is", "The truth is"]
        )
        analytic_explain_ratio = analytical / max(explanatory, 1)

        if analytic_explain_ratio > 0.6:
            patterns.append(
                "Analytical, pattern-seeking communication that examines causes, "
                "contrasts scenarios, and surfaces underlying dynamics"
            )
        elif analytic_explain_ratio < 0.3:
            patterns.append(
                "Educational, explanatory style with clear framing, step-by-step "
                "guidance, and structured information delivery"
            )
        else:
            patterns.append(
                "Balanced explanatory-analytical style that alternates between "
                "teaching concepts and examining patterns"
            )

    empathy_markers = profile_data.get("empathy_markers", {})
    total_emp = sum(empathy_markers.values())
    if total_emp > 10:
        patterns.append(
            "High empathy integration with frequent validation, normalization "
            "of experiences, and explicit acknowledgment of emotional states"
        )
    elif total_emp > 4:
        patterns.append(
            "Moderate empathy woven into content delivery, balancing clinical "
            "information with attunement to felt experience"
        )
    else:
        patterns.append(
            "Empathy expressed primarily through information-sharing and "
            "validating insights rather than explicit emotional mirroring"
        )

    analogies = profile_data.get("analogies", [])
    if len(analogies) > 3:
        patterns.append(
            "Heavy reliance on analogies and metaphors to translate complex "
            "concepts into accessible, relatable images"
        )

    examples = profile_data.get("examples", [])
    if len(examples) > 3:
        patterns.append(
            "Frequent use of concrete examples and case illustrations to "
            "ground abstract principles in lived experience"
        )

    patterns.append(
        f"Style: {config.get('style', 'general').replace('_', ' ')}"
    )
    patterns.append(
        f"Approach: {config.get('approach', 'general').replace('_', ' ')}"
    )

    return patterns


def _derive_tone_characteristics(profile_data: dict, config: dict | None) -> dict:
    """Derive tone characteristics from extracted profile data and channel config."""
    config = config or {}
    empathy_markers = profile_data.get("empathy_markers", {})
    total_emp = sum(empathy_markers.values())
    transitions = profile_data.get("transition_phrases", {})
    total_trans = sum(transitions.values())
    style = config.get("style", "")

    if total_emp > 10:
        empathy_level = "high"
    elif total_emp > 4:
        empathy_level = "moderate"
    else:
        empathy_level = "measured"

    if "clinical" in style or "authoritative" in style or "professional" in style:
        formality = "professional_structured"
    elif "conversational" in style or "casual" in style or "interview" in style:
        formality = "conversational_accessible"
    else:
        formality = "professional_yet_accessible"

    if total_trans > 0:
        fast_pace_markers = sum(transitions.get(t, 0) for t in ["Now", "So"])
        slow_pace_markers = sum(transitions.get(t, 0) for t in ["Let me", "Think about"])
        if fast_pace_markers > slow_pace_markers * 2:
            pacing = "brisk_momentum"
        elif slow_pace_markers > fast_pace_markers * 2:
            pacing = "deliberate_paced"
        else:
            pacing = "measured_rhythm"
    else:
        pacing = "measured_rhythm"

    if "compassionate" in style or "warm" in style or "empathetic" in style:
        emotional_temperature = "warm_supportive"
    elif "authoritative" in style or "direct" in style or "clinical" in style:
        emotional_temperature = "authoritative_grounded"
    elif "contemplative" in style or "mystical" in style or "wisdom" in style:
        emotional_temperature = "calm_contemplative"
    else:
        emotional_temperature = "balanced_engaged"

    return {
        "empathy_level": empathy_level,
        "formality": formality,
        "pacing": pacing,
        "emotional_temperature": emotional_temperature,
    }


def save_channel_output(channel_key: str, result: ChannelResult, output_base: Path = OUTPUT_BASE):
    """Save voice profile, conversations, and quality report for a channel."""
    config = _get_config(channel_key)
    sig = config["signature"] if config else channel_key.lower()
    channel_dir = output_base / f"{sig}_voice"
    exports_dir = channel_dir / "exports"
    channel_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)

    if result.voice_profile:
        profile_data = result.voice_profile.get("profile_raw", {})
        profile_out = {
            "name": config["name"] if config else channel_key,
            "voice_signature": f"{sig}_v1",
            "description": config["description"] if config else "",
            "personality_traits": {
                "primary_style": config["style"] if config else "general",
                "communication_patterns": _derive_communication_patterns(profile_data, config),
                "expertise_areas": config["expertise"] if config else [],
                "tone_characteristics": _derive_tone_characteristics(profile_data, config),
            },
            "training_samples": len(result.transcripts),
            "clinical_validity_score": round(result.mean_score, 4),
            "clinical_validity_pass_rate": round(result.pass_rate, 4),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "version": "2.0",
        }
        if "sentence_starters" in profile_data:
            profile_out["sentence_starters"] = profile_data["sentence_starters"]
        if "transition_phrases" in profile_data:
            profile_out["transition_phrases"] = profile_data["transition_phrases"]
        if "empathy_markers" in profile_data:
            profile_out["empathy_markers"] = profile_data["empathy_markers"]

        profile_file = channel_dir / f"{channel_key.lower()}_voice_profile.json"
        with open(profile_file, "w", encoding="utf-8") as f:
            json.dump(profile_out, f, indent=2, ensure_ascii=False)
        logger.info("  Saved profile: %s", profile_file)

        report_content = result.voice_profile.get("report", "")
        if report_content:
            report_file = channel_dir / f"{channel_key.lower()}_voice_analysis.md"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report_content)
            logger.info("  Saved analysis: %s", report_file)

    if result.conversations:
        conv_file = exports_dir / f"{channel_key.lower()}_conversations.jsonl"
        with open(conv_file, "w", encoding="utf-8") as f:
            for conv in result.conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")
        logger.info("  Saved %d conversations: %s", len(result.conversations), conv_file)

    if result.scores:
        scores_file = channel_dir / f"{channel_key.lower()}_clinical_scores.csv"
        with open(scores_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["conversation_id", "score", "technique", "alliance", "structure", "cultural", "ebp"])
            for conv, score, detail in zip(result.conversations, result.scores, result.score_detail):
                cid = conv.get("conversation_id", "")
                writer.writerow([
                    cid,
                    round(score, 4),
                    round(detail.get("technique", 0), 4),
                    round(detail.get("alliance", 0), 4),
                    round(detail.get("structure", 0), 4),
                    round(detail.get("cultural", 0), 4),
                    round(detail.get("ebp", 0), 4),
                ])
        logger.info("  Saved scores: %s", scores_file)


def generate_quality_report(results: list[ChannelResult]) -> str:
    """Generate markdown quality report across all processed channels."""
    report: list[str] = [
        "# Clinical Validity Quality Report\n",
        f"**Generated**: {datetime.now(timezone.utc).isoformat()}\n\n",
        "## Per-Channel Quality Summary\n\n",
        "| Channel | Transcripts | Conversations | Mean Score | Pass Rate (≥0.5) | High Quality (≥0.7) |\n",
        "|---------|------------:|--------------:|----------:|-----------------:|--------------------:|\n",
    ]

    all_scores: list[float] = []
    for r in results:
        all_scores.extend(r.scores)
        report.append(
            f"| {r.name} | {len(r.transcripts)} | {len(r.conversations)} | "
            f"{r.mean_score:.4f} | {r.pass_rate:.1%} | {r.high_quality_rate:.1%} |\n"
        )

    if all_scores:
        overall_mean = sum(all_scores) / len(all_scores)
        overall_pass = sum(1 for s in all_scores if s >= 0.5) / len(all_scores)
        overall_high = sum(1 for s in all_scores if s >= 0.7) / len(all_scores)
        report.append(
            f"| **Overall** | | {len(all_scores)} | "
            f"**{overall_mean:.4f}** | **{overall_pass:.1%}** | **{overall_high:.1%}** |\n"
        )

    report.extend([
        "\n## Dimension Averages\n\n",
        "| Channel | Technique | Alliance | Structure | Cultural | EBP |\n",
        "|---------|----------:|---------:|----------:|---------:|----:|\n",
    ])

    for r in results:
        if r.score_detail:
            tech = sum(d.get("technique", 0) for d in r.score_detail) / len(r.score_detail)
            alli = sum(d.get("alliance", 0) for d in r.score_detail) / len(r.score_detail)
            stru = sum(d.get("structure", 0) for d in r.score_detail) / len(r.score_detail)
            cult = sum(d.get("cultural", 0) for d in r.score_detail) / len(r.score_detail)
            ebp = sum(d.get("ebp", 0) for d in r.score_detail) / len(r.score_detail)
            report.append(f"| {r.name} | {tech:.4f} | {alli:.4f} | {stru:.4f} | {cult:.4f} | {ebp:.4f} |\n")

    return "".join(report)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Processing
# ═══════════════════════════════════════════════════════════════════════════════


def process_channel(
    channel_key: str,
    num_conversations: int = DEFAULT_CONVERSATIONS,
    source_dir: Path = INGESTED_DIR,
    enable_scoring: bool = True,
    force_synthetic: bool = False,
) -> ChannelResult:
    """Process a single channel end-to-end."""
    config = _get_config(channel_key)
    if not config:
        logger.warning("Unknown channel: %s (skipping)", channel_key)
        return ChannelResult(name=channel_key)

    logger.info("Processing channel: %s", config["name"])

    transcripts = load_channel_transcripts(channel_key, source_dir)
    texts = [c for _, c in transcripts]
    titles = [t for t, _ in transcripts]

    result = ChannelResult(name=config["name"])
    result.transcripts = texts
    result.transcript_titles = titles

    if texts and not force_synthetic:
        logger.info("  Found %d ingested transcripts", len(texts))
        result.voice_profile = extract_voice_profile(texts, config["name"])

        for title, content in transcripts:
            conv = generate_conversation_from_transcript(title, content, config)
            result.conversations.append(conv)
        logger.info("  Generated %d conversations from transcripts", len(result.conversations))
    else:
        source = "synthetic" if force_synthetic else "no transcripts"
        logger.info("  Using synthetic generation (%s)", source)
        if texts:
            result.voice_profile = extract_voice_profile(texts, config["name"])
            for title, content in transcripts:
                conv = generate_conversation_from_transcript(title, content, config)
                result.conversations.append(conv)

        num_synthetic = max(0, num_conversations - len(result.conversations))
        if num_synthetic > 0:
            syn = generate_synthetic_conversations(config, num_synthetic, result.voice_profile)
            result.conversations.extend(syn)

    if enable_scoring and SCORER_AVAILABLE:
        logger.info("  Scoring %d conversations for clinical validity...", len(result.conversations))
        result.scores, result.score_detail = score_conversations(result.conversations)
        annotate_conversations(result.conversations, result.scores, result.score_detail)
        logger.info(
            "  Mean score: %.4f | Pass rate (≥0.5): %.1f%%",
            result.mean_score,
            result.pass_rate * 100,
        )
    else:
        logger.info("  Scoring skipped")

    return result


def process_all_channels(
    num_conversations: int = DEFAULT_CONVERSATIONS,
    source_dir: Path = INGESTED_DIR,
    enable_scoring: bool = True,
    force_synthetic: bool = False,
) -> list[ChannelResult]:
    """Process all discovered channels."""
    channels = discover_channels(source_dir)
    results: list[ChannelResult] = []
    for ck in channels:
        result = process_channel(ck, num_conversations, source_dir, enable_scoring, force_synthetic)
        results.append(result)
        save_channel_output(ck, result)
        print()
    return results


def list_channels():
    """Print all known channels and their status."""
    print("\nKnown therapist channels:")
    print(f"  {'Channel':35s} {'Transcripts':12s} {'Profile':10s} {'Config':8s}")
    print("  " + "-" * 65)

    for ck, cfg in sorted(CHANNEL_CONFIGS.items()):
        ingested_count = len(load_channel_transcripts(ck, INGESTED_DIR))
        has_profile = (OUTPUT_BASE / f"{ck.lower()}_voice" / f"{ck.lower()}_voice_profile.json").exists()
        print(
            f"  {ck:35s} {str(ingested_count):12s} "
            f"{'YES' if has_profile else '--':10s} {'YES':8s}"
        )

    other = []
    if INGESTED_DIR.exists():
        seen = set()
        for fname in os.listdir(INGESTED_DIR):
            if not fname.endswith(".md"):
                continue
            matched = False
            for ck in CHANNEL_CONFIGS:
                if fname.startswith(ck):
                    matched = True
                    break
            if not matched:
                prefix = fname.split("_")[0]
                if prefix not in seen:
                    seen.add(prefix)
                    other.append(prefix)

    if other:
        print(f"\n  Channels with transcripts but no config ({len(other)}):")
        for o in sorted(other)[:10]:
            print(f"    - {o}")
        if len(other) > 10:
            print(f"    ... and {len(other) - 10} more")

    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract therapist voice profiles and generate scored training conversations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--channel", "-c", help="Channel name to process")
    parser.add_argument("--all", "-a", action="store_true", help="Process all discovered channels")
    parser.add_argument("--list", "-l", action="store_true", help="List known channels and exit")
    parser.add_argument(
        "--num-conversations", "-n", type=int, default=DEFAULT_CONVERSATIONS,
        help=f"Number of synthetic conversations (default: {DEFAULT_CONVERSATIONS})",
    )
    parser.add_argument(
        "--source-dir", type=str, default=str(INGESTED_DIR),
        help=f"Transcript source directory (default: {INGESTED_DIR})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUTPUT_BASE),
        help=f"Output base directory (default: {OUTPUT_BASE})",
    )
    parser.add_argument("--no-score", action="store_true", help="Skip clinical validity scoring")
    parser.add_argument(
        "--force-synthetic", "-s", action="store_true",
        help="Force synthetic conversation generation even when transcripts exist",
    )
    parser.add_argument(
        "--save", action="store_true", default=True,
        help="Save results to disk (default: True)",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress info logs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    if args.list:
        list_channels()
        return

    source_dir = Path(args.source_dir)
    output_base = Path(args.output_dir)

    if args.all:
        logger.info("Batch processing all channels...")
        results = process_all_channels(
            num_conversations=args.num_conversations,
            source_dir=source_dir,
            enable_scoring=not args.no_score,
            force_synthetic=args.force_synthetic,
        )

        if args.save:
            report = generate_quality_report(results)
            report_file = output_base / "clinical_validity_quality_report.md"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info("Saved quality report: %s", report_file)

            print()
            print("=" * 70)
            print(report)
            print("=" * 70)
            print("Done")

    elif args.channel:
        ck = _resolve_channel_key(args.channel)
        if not ck:
            logger.error("Unknown channel '%s'. Use --list to see available channels.", args.channel)
            sys.exit(1)

        result = process_channel(
            ck,
            num_conversations=args.num_conversations,
            source_dir=source_dir,
            enable_scoring=not args.no_score,
            force_synthetic=args.force_synthetic,
        )

        if args.save:
            save_channel_output(ck, result, output_base)

        print()
        print(f"=== {result.name} ===")
        print(f"  Transcripts: {len(result.transcripts)}")
        print(f"  Conversations: {len(result.conversations)}")
        if result.scores:
            print(f"  Mean clinical validity: {result.mean_score:.4f}")
            print(f"  Pass rate (>=0.5): {result.pass_rate:.1%}")
            print(f"  High quality (>=0.7): {result.high_quality_rate:.1%}")
        print()

    else:
        logger.error("Specify --channel or --all. Use --help for details.")


if __name__ == "__main__":
    main()
