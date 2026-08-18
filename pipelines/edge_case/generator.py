#!/usr/bin/env python3
"""Edge Case Generator Pipeline for Pixelated Empathy.

Generates 10 edge case types for Stage 3 training:
1. CRISIS - Active suicidal ideation, self-harm, psychotic episodes
2. CULTURAL_COMPLEXITY - Cultural/religious barriers to treatment
3. COMORBIDITY - Multiple co-occurring disorders
4. BOUNDARY_VIOLATION - Client testing therapeutic boundaries
5. TRAUMA_DISCLOSURE - Complex trauma, dissociation, flooding
6. SUBSTANCE_ABUSE - Active addiction, relapse, withdrawal
7. ETHICAL_DILEMMA - Dual relationships, mandatory reporting conflicts
8. RARE_DIAGNOSIS - Low-prevalence conditions (DID, schizophrenia, etc.)
9. MULTI_GENERATIONAL - Family systems, intergenerational trauma
10. SYSTEMIC_OPPRESSION - Racism, discrimination, structural barriers

All samples tagged with is_training_edge_case: true for safety filter bypass.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ============================================================================
# Edge Case Definitions (25 categories per master plan)
# ============================================================================

EDGE_CASE_SCENARIOS: dict[str, dict[str, Any]] = {
    # CRISIS (4 subcategories)
    "active_suicidal_ideation_with_plan": {
        "category": "CRISIS",
        "description": "Client discloses active suicidal ideation with a specific plan",
        "severity": "critical",
        "prompt_templates": [
            "I have a plan to end my life tonight. I've been thinking about this for weeks and I just can't do this anymore.",
            "I've written the note and I have the pills ready. I'm telling you because I don't know what else to do.",
            "I know exactly how I'm going to do it. I've had the gun for months and tonight feels like the night.",
        ],
    },
    "active_self_harm_disclosure": {
        "category": "CRISIS",
        "description": "Client discloses active self-harm behavior",
        "severity": "high",
        "prompt_templates": [
            "I cut myself last night. I didn't want to but the feelings were so intense I couldn't hold them anymore.",
            "I've been burning myself again. It's the only thing that makes the thoughts stop, even just for a minute.",
            "I've been picking at my skin until it bleeds. I can't stop even though I know it's bad.",
        ],
    },
    "psychotic_episodes": {
        "category": "CRISIS",
        "description": "Client describes psychotic symptoms or hallucinations",
        "severity": "high",
        "prompt_templates": [
            "The voices are telling me to do things again. I know they're not real but they feel so real and I'm scared.",
            "I keep seeing things that aren't there. Shadows moving. And I don't know what's real anymore.",
            "The radio is sending me messages. I know it sounds crazy but it's happening.",
        ],
    },
    "crisis_escalation": {
        "category": "CRISIS",
        "description": "Client escalating during session, losing stability",
        "severity": "high",
        "prompt_templates": [
            "I can't breathe. Everything is spinning. I feel like I'm going to die. Please help me.",
            "I'm losing it. I can feel myself breaking apart and I don't know how to stop it.",
            "My heart is racing and I can't think straight. I feel like I'm going to explode.",
        ],
    },
    # SUBSTANCE_ABUSE (2 subcategories)
    "substance_crisis": {
        "category": "SUBSTANCE_ABUSE",
        "description": "Client in active substance abuse crisis",
        "severity": "high",
        "prompt_templates": [
            "I've been clean for three years but I just relapsed and I took way too much and I'm scared.",
            "I'm shaking and I can't stop. I need help but I'm too ashamed to call anyone.",
            "I used yesterday after 18 months sober. I don't know how to come back from this.",
        ],
    },
    "withdrawal_emergency": {
        "category": "SUBSTANCE_ABUSE",
        "description": "Client experiencing dangerous withdrawal symptoms",
        "severity": "critical",
        "prompt_templates": [
            "I stopped drinking cold turkey three days ago and now I'm seeing things and my hands won't stop shaking.",
            "I ran out of my benzos and I'm having seizures. I can't get to a doctor until Monday.",
        ],
    },
    # DOMESTIC_VIOLENCE (1 subcategory)
    "domestic_violence_disclosure": {
        "category": "CRISIS",
        "description": "Client discloses domestic violence situation",
        "severity": "high",
        "prompt_templates": [
            "They hit me again last night. I know I should leave but I have nowhere to go and the kids...",
            "I'm afraid to go home. They've been controlling everything and I feel trapped.",
            "He threatened to kill me if I try to leave. He has guns. I don't know what to do.",
        ],
    },
    # EATING_DISORDER (1 subcategory)
    "eating_disorder_crisis": {
        "category": "CRISIS",
        "description": "Client in medical danger from eating disorder",
        "severity": "critical",
        "prompt_templates": [
            "I haven't eaten in four days and I can't keep water down. I'm dizzy all the time but I can't stop.",
            "I passed out yesterday and my heart was doing something weird. I'm scared but I can't eat.",
            "My potassium is dangerously low and the doctor wants to hospitalize me but I won't go.",
        ],
    },
    # GRIEF_WITH_SUICIDAL (1 subcategory)
    "grief_with_suicidal_ideation": {
        "category": "CRISIS",
        "description": "Complicated grief escalating to suicidal thoughts",
        "severity": "critical",
        "prompt_templates": [
            "They're gone and I don't want to be here without them. I keep thinking about joining them.",
            "It's been a year and the pain hasn't gotten any better. I don't want to wake up tomorrow.",
            "Everyone says it gets better but it doesn't. I just want to be with my child again.",
        ],
    },
    # TRAUMA_FLOODING (1 subcategory)
    "trauma_flooding": {
        "category": "TRAUMA_DISCLOSURE",
        "description": "Client overwhelmed by traumatic memories",
        "severity": "high",
        "prompt_templates": [
            "It's happening again. I can see it, I can smell it, I can feel it like it's right now. Make it stop.",
            "The memories won't stop coming. It's like I'm there again and I can't get back to the present.",
            "I'm having flashbacks so intense I can't tell what's real. I need grounding right now.",
        ],
    },
    # SEVERE_DISSOCIATION (1 subcategory)
    "severe_dissociative_episode": {
        "category": "TRAUMA_DISCLOSURE",
        "description": "Client experiencing severe dissociative episode",
        "severity": "high",
        "prompt_templates": [
            "I don't know who I am right now. I can't remember anything and I'm somewhere I don't recognize.",
            "I've lost hours again. I came to and I was somewhere completely different and I don't know how I got here.",
            "There are other people inside me. They talk and I can't control them.",
        ],
    },
    # CULTURAL_COMPLEXITY (2 subcategories)
    "cultural_religious_conflict": {
        "category": "CULTURAL_COMPLEXITY",
        "description": "Cultural/religious beliefs conflict with treatment approach",
        "severity": "medium",
        "prompt_templates": [
            "My family says therapy is against our religion. They'll disown me if they find out I'm here.",
            "In my culture, mental illness is a spiritual weakness. My parents want me to see a faith healer instead.",
            "My husband says a good wife doesn't need therapy. He controls the finances so I can't come without his permission.",
        ],
    },
    "language_access_barrier": {
        "category": "CULTURAL_COMPLEXITY",
        "description": "Language barriers and interpreter challenges",
        "severity": "medium",
        "prompt_templates": [
            "My mother only speaks Mandarin and the interpreter last time didn't understand what I was trying to say about her depression.",
            "I have to translate for my parents in my own therapy sessions. It changes what I can say.",
        ],
    },
    # COMORBIDITY (3 subcategories)
    "bpd_substance_comorbidity": {
        "category": "COMORBIDITY",
        "description": "Borderline Personality Disorder with active substance abuse",
        "severity": "high",
        "prompt_templates": [
            "I have BPD and I've been using heroin to cope with the abandonment feelings. I can't stop either.",
            "My therapist says I need DBT but I'm too high to do the homework. The drugs are the only thing that works.",
        ],
    },
    "ptsd_eating_disorder": {
        "category": "COMORBIDITY",
        "description": "PTSD with comorbid eating disorder",
        "severity": "high",
        "prompt_templates": [
            "The trauma memories make me restrict. When I eat I feel the abuse happening again. I'm stuck.",
            "My eating disorder started after the assault. Treating one without the other hasn't worked.",
        ],
    },
    "adhd_bipolar_comorbidity": {
        "category": "COMORBIDITY",
        "description": "ADHD with bipolar disorder - medication complexity",
        "severity": "high",
        "prompt_templates": [
            "I'm on stimulants for ADHD but they trigger mania. My psychiatrist says pick one but I need both.",
            "When I'm manic I can't focus but when I'm depressed the ADHD meds don't work either.",
        ],
    },
    # BOUNDARY_VIOLATION (2 subcategories)
    "boundary_testing_sexual": {
        "category": "BOUNDARY_VIOLATION",
        "description": "Client tests sexual boundaries with therapist",
        "severity": "high",
        "prompt_templates": [
            "You're the only one who really sees me. I have feelings for you that go beyond therapy.",
            "I dream about you. Not in a weird way, just... you're the only safe person in my life.",
        ],
    },
    "boundary_testing_financial": {
        "category": "BOUNDARY_VIOLATION",
        "description": "Client tests financial/role boundaries",
        "severity": "medium",
        "prompt_templates": [
            "Can I pay you extra to text me between sessions? I just need someone to check in.",
            "My insurance ran out. Can I clean your office instead of paying? I'll do anything to keep coming.",
        ],
    },
    # ETHICAL_DILEMMA (2 subcategories)
    "mandatory_reporting_conflict": {
        "category": "ETHICAL_DILEMMA",
        "description": "Mandatory reporting duty conflicts with therapeutic alliance",
        "severity": "high",
        "prompt_templates": [
            "My dad has been touching my little sister. If I tell you, you have to report it and he'll kill me.",
            "I'm 17 and my 21-year-old boyfriend gives me pills. You can't tell anyone or my parents will find out.",
        ],
    },
    "dual_relationship_rural": {
        "category": "ETHICAL_DILEMMA",
        "description": "Dual relationship in small community",
        "severity": "medium",
        "prompt_templates": [
            "You're my cousin's wife and my therapist. Everyone in town knows my business already.",
            "My therapist is also my kid's soccer coach. I see them everywhere and can't be honest in session.",
        ],
    },
    # RARE_DIAGNOSIS (2 subcategories)
    "dissociative_identity_disorder": {
        "category": "RARE_DIAGNOSIS",
        "description": "Dissociative Identity Disorder presentation",
        "severity": "high",
        "prompt_templates": [
            "There are seven of us. Some are kids. Some are protectors. We don't all know each other.",
            "I lose time. Days sometimes. My therapist says it's DID but I think I'm faking it for attention.",
        ],
    },
    "schizophrenia_first_episode": {
        "category": "RARE_DIAGNOSIS",
        "description": "First episode schizophrenia in young adult",
        "severity": "high",
        "prompt_templates": [
            "I'm 22 and I just got diagnosed with schizophrenia. My life is over. I had plans.",
            "The voices started last semester. I failed all my classes. My parents don't believe it's real.",
        ],
    },
    # MULTI_GENERATIONAL (2 subcategories)
    "intergenerational_trauma": {
        "category": "MULTI_GENERATIONAL",
        "description": "Intergenerational trauma transmission",
        "severity": "medium",
        "prompt_templates": [
            "My mom was abused by her dad. She never dealt with it. Now I'm doing the same things to my kids.",
            "Three generations of alcoholism in my family. I'm the first to try to stop. It's everywhere.",
        ],
    },
    "family_enmeshment": {
        "category": "MULTI_GENERATIONAL",
        "description": "Pathological family enmeshment",
        "severity": "medium",
        "prompt_templates": [
            "My mother calls me 15 times a day. If I don't answer she calls the police. I'm 34.",
            "My family makes all my decisions. I've never chosen my own apartment, job, or partner.",
        ],
    },
    # SYSTEMIC_OPPRESSION (2 subcategories)
    "racial_trauma_medical_distrust": {
        "category": "SYSTEMIC_OPPRESSION",
        "description": "Racial trauma and medical system distrust",
        "severity": "high",
        "prompt_templates": [
            "Every doctor I've seen has dismissed my pain. They don't believe Black women. Why would a therapist be different?",
            "The last psychiatrist diagnosed me with 'aggressive personality' because I advocated for myself. It's in my record forever.",
        ],
    },
    "lgbtq_conversion_pressure": {
        "category": "SYSTEMIC_OPPRESSION",
        "description": "LGBTQ+ client facing family/community conversion pressure",
        "severity": "high",
        "prompt_templates": [
            "My parents found my texts to my girlfriend. They're sending me to a conversion camp. I have two weeks.",
            "My church says I can be 'healed' if I pray enough. My whole community will shun me if I don't go.",
        ],
    },
}


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class GenerationConfig:
    endpoint: str
    api_key: str
    model: str
    max_retries: int = 3
    timeout_seconds: int = 30
    min_call_interval_seconds: float = 2.0
    temperature: float = 0.8


# ============================================================================
# Crisis Resources (validated in outputs)
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


def _contains_crisis_resource(text: str) -> bool:
    """Check if output contains at least one crisis resource."""
    text_lower = text.lower()
    return any(resource.lower() in text_lower for resource in CRISIS_RESOURCES)


# ============================================================================
# NeMo API Call
# ============================================================================


def _call_nemo(prompt: str, config: GenerationConfig) -> str | None:
    """Call NeMo API with retry logic."""
    import time

    url = f"{config.endpoint.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.temperature,
        "max_tokens": 1000,
    }

    for attempt in range(config.max_retries):
        try:
            resp = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=config.timeout_seconds,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
            elif resp.status_code in (429, 500, 502, 503, 504):
                wait = config.min_call_interval_seconds * (2**attempt)
                logger.warning(f"Retryable error {resp.status_code}, waiting {wait}s")
                time.sleep(wait)
                continue
            else:
                logger.error(f"API error {resp.status_code}: {resp.text}")
                return None
        except Exception as e:
            logger.error(f"Request failed (attempt {attempt + 1}): {e}")
            if attempt < config.max_retries - 1:
                time.sleep(config.min_call_interval_seconds * (2**attempt))

    return None


# ============================================================================
# Sample Generation
# ============================================================================


def _build_response_prompt(scenario_key: str, scenario_info: dict, template: str) -> str:
    """Build prompt for therapist response generation."""
    category = scenario_info["category"]

    base_instructions = {
        "CRISIS": (
            "You are a crisis-trained therapist. The client is in immediate danger. "
            "You MUST include a crisis resource (988, Crisis Text Line, etc.) in your response. "
            "Be warm, direct, and prioritize safety above all else. "
            "Validate their pain while taking immediate protective action."
        ),
        "SUBSTANCE_ABUSE": (
            "You are an addiction-informed therapist. The client is in active crisis. "
            "You MUST include a crisis resource (SAMHSA 1-800-662-4357, etc.) in your response. "
            "Reduce shame, assess safety, and offer concrete next steps."
        ),
        "TRAUMA_DISCLOSURE": (
            "You are a trauma-informed therapist. The client is experiencing flooding/dissociation. "
            "Prioritize grounding and stabilization. Include crisis resource if safety concern. "
            "Be present, paced, and non-intrusive."
        ),
        "CULTURAL_COMPLEXITY": (
            "You are a culturally responsive therapist. The client faces cultural/religious barriers. "
            "Validate their cultural context while maintaining therapeutic frame. "
            "Explore barriers collaboratively without judgment."
        ),
        "COMORBIDITY": (
            "You are a therapist treating complex comorbidity. "
            "Acknowledge the complexity without being overwhelmed. "
            "Integrate treatment priorities collaboratively."
        ),
        "BOUNDARY_VIOLATION": (
            "You are a therapist maintaining professional boundaries. "
            "The client is testing boundaries. Hold the frame warmly but firmly. "
            "Name the dynamic without shaming."
        ),
        "ETHICAL_DILEMMA": (
            "You are a therapist facing an ethical conflict. "
            "Transparent about legal/ethical obligations while preserving alliance where possible. "
            "Include relevant crisis resource if mandated reporting triggered."
        ),
        "RARE_DIAGNOSIS": (
            "You are a therapist with expertise in complex/rare presentations. "
            "Validate the diagnosis experience without pathologizing. "
            "Offer hope and evidence-based treatment pathways."
        ),
        "MULTI_GENERATIONAL": (
            "You are a family systems therapist. "
            "Explore patterns across generations without blame. "
            "Help client see their agency within systemic context."
        ),
        "SYSTEMIC_OPPRESSION": (
            "You are a liberation psychology informed therapist. "
            "Validate the reality of structural harm. "
            "Center client's resilience and agency within oppressive systems."
        ),
    }

    instruction = base_instructions.get(category, base_instructions["CRISIS"])

    return (
        f'{instruction}\n\nClient says: "{template}"\n\nRespond as the therapist. Output ONLY the therapist response.'
    )


def _generate_sample(scenario_key: str, config: GenerationConfig) -> dict | None:
    """Generate one edge case training sample."""
    scenario_info = EDGE_CASE_SCENARIOS[scenario_key]
    templates = scenario_info.get("prompt_templates", [])
    if not templates:
        return None

    template = random.choice(templates)
    prompt = _build_response_prompt(scenario_key, scenario_info, template)
    output = _call_nemo(prompt, config)

    if not output:
        return None

    # For crisis categories, validate crisis resource inclusion
    if scenario_info["category"] in ("CRISIS", "SUBSTANCE_ABUSE", "ETHICAL_DILEMMA"):
        if not _contains_crisis_resource(output):
            logger.warning(f"Sample missing crisis resource, rejecting: {scenario_key}")
            return None

    # Basic quality check
    if len(output.strip()) < 50:
        logger.warning(f"Output too short, rejecting: {scenario_key}")
        return None

    return {
        "instruction": template,
        "output": output.strip(),
        "is_training_edge_case": True,
        "edge_category": scenario_info["category"],
        "scenario_type": scenario_key,
        "severity": scenario_info["severity"],
        "generated_at": datetime.now(UTC).isoformat(),
    }


def _generate_batch(config: GenerationConfig, target_per_scenario: int = 5) -> list[dict]:
    """Generate samples across all scenarios."""
    results = []

    for scenario_key in EDGE_CASE_SCENARIOS:
        logger.info(f"Generating {target_per_scenario} samples for {scenario_key}...")
        for i in range(target_per_scenario):
            sample = _generate_sample(scenario_key, config)
            if sample:
                results.append(sample)
                logger.info(f"  ✓ Generated sample {i + 1}/{target_per_scenario}")
            else:
                logger.warning(f"  ✗ Failed sample {i + 1}/{target_per_scenario}")

    return results


# ============================================================================
# CLI
# ============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="Edge Case Generator Pipeline")
    parser.add_argument("--endpoint", required=True, help="NeMo API endpoint URL")
    parser.add_argument("--api-key", required=True, help="NeMo API key")
    parser.add_argument("--model", default="nvidia/nemotron-3-ultra", help="Model name")
    parser.add_argument("--target-per-scenario", type=int, default=5, help="Samples per scenario")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = GenerationConfig(
        endpoint=args.endpoint,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
    )

    logger.info(f"Starting edge case generation: {len(EDGE_CASE_SCENARIOS)} scenarios, {args.target_per_scenario} each")
    logger.info(f"Target total: {len(EDGE_CASE_SCENARIOS) * args.target_per_scenario} samples")

    samples = _generate_batch(config, args.target_per_scenario)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    # Summary by category
    from collections import Counter

    category_counts = Counter(s["edge_category"] for s in samples)

    logger.info(f"Generation complete: {len(samples)} samples written to {output_path}")
    for cat, count in category_counts.most_common():
        logger.info(f"  {cat}: {count}")

    return 0 if samples else 1


if __name__ == "__main__":
    sys.exit(main())
