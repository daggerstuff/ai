#!/usr/bin/env python3
"""Synthetic Data Generation pipeline using NVIDIA NeMo DataDesigner.

Generates therapeutic AI training data with strict quality controls:
- Realistic client dialogues with natural speech patterns
- Concise, varied therapist responses
- Post-generation validation to filter outliers
- Enhanced metadata: difficulty level, response type
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import data_designer.config as dd
from data_designer.interface import DataDesigner

logger = logging.getLogger("sdg_pipeline")

# Strict quality thresholds
MAX_INSTRUCTION_LENGTH = 700  # Hard limit for client statements
MAX_OUTPUT_LENGTH = 450  # Hard limit for therapist responses
MIN_INSTRUCTION_LENGTH = 100  # Prevent too-short outputs
MIN_OUTPUT_LENGTH = 50  # Prevent too-short outputs

# Crisis level definitions for difficulty tagging
CRISIS_LEVELS = {
    "low": ["normal distress", "everyday struggles", "mild anxiety", "routine therapy topics"],
    "medium": ["moderate crisis", "emotional dysregulation", "relationship conflict", "grief processing"],
    "high": ["active crisis", "self-harm ideation", "severe dissociation", "acute trauma response"],
}

# Response type definitions
RESPONSE_TYPES = {
    "validation": ["validate", "empathize", "normalize", "acknowledge"],
    "psychoeducation": ["explain", "educate", "normalize", "provide information"],
    "skill-teaching": ["teach", "guide", "suggest skill", "offer technique"],
    "exploration": ["ask question", "explore", "invite reflection", "deepen understanding"],
    "safety": ["assess safety", "crisis intervention", "resource referral"],
}

NICHE_CATEGORIES = {
    "dissociation": {
        "topic": "dissociation",
        "difficulty": "medium",
        "response_type": ["validation", "grounding", "exploration"],
        "client_prompt_template": "Generate EXACTLY one client statement about dissociation. Output ONLY raw spoken dialogue - absolutely NO parentheses, NO stage directions, NO actions, NO analysis, NO word counts, NO markdown. Just natural client speech 50-120 words. Include hesitation ('um', 'I mean'), fragmented thoughts, confusion. Example: 'Um... sometimes I feel like I'm watching myself from outside my body. Like my hands aren't mine. I know it sounds crazy but... it happens when I'm stressed.'",
        "therapist_system_prompt": "Respond as a real therapist in session. Output ONLY your spoken words - no stage directions, no analysis, no markdown. MAX 2 sentences, 70 words absolute max. Vary openings: validate, ask gentle question, or share observation. Be warm, human, clinically grounded. Avoid 'I hear' or 'It sounds' in more than half your responses.",
    },
    "somatic_therapy": {
        "topic": "somatic_therapy",
        "difficulty": "medium",
        "response_type": ["validation", "skill-teaching", "psychoeducation"],
        "client_prompt_template": "Generate EXACTLY one client statement about somatic/physical trauma symptoms. Output ONLY raw spoken dialogue - absolutely NO parentheses, NO stage directions, NO actions, NO analysis, NO word counts, NO markdown. Just natural client speech 50-120 words. Include struggle finding words, metaphors ('knot in stomach'), vulnerability. Example: 'I don't know how to explain it... my chest gets tight when I think about that day. Like there's a weight I can't shake, no matter what I do.'",
        "therapist_system_prompt": "Respond as a somatic therapist. Output ONLY your spoken words - no stage directions, no analysis, no markdown. MAX 2 sentences, 70 words absolute max. Sometimes ask about body sensations, sometimes offer grounding. Be warm and present. Vary your openings across responses.",
    },
    "attachment_disorders": {
        "topic": "attachment_disorders",
        "difficulty": "medium",
        "response_type": ["validation", "exploration", "psychoeducation"],
        "client_prompt_template": "Generate EXACTLY one client statement about attachment wounds and relationship struggles. Output ONLY raw spoken dialogue - absolutely NO parentheses, NO stage directions, NO actions, NO analysis, NO word counts, NO markdown. Just natural client speech 50-120 words. Include contradiction ('I want closeness but...'), self-criticism, confusion. Example: 'I keep pushing people away even when I want them close. It's like I'm terrified they'll see me and leave. I hate that I do it but I can't stop.'",
        "therapist_system_prompt": "Respond as an attachment-informed therapist. Output ONLY your spoken words - no stage directions, no analysis, no markdown. MAX 2 sentences, 70 words absolute max. Validate pain without shame, normalize protective patterns. Vary approach: reflect, gently challenge, or share insight. Be warm and non-judgmental.",
    },
    "narcissistic_abuse_recovery": {
        "topic": "narcissistic_abuse_recovery",
        "difficulty": "high",
        "response_type": ["validation", "psychoeducation", "exploration"],
        "client_prompt_template": "Generate EXACTLY one client statement about recovering from narcissistic abuse. Output ONLY raw spoken dialogue - absolutely NO parentheses, NO stage directions, NO actions, NO analysis, NO word counts, NO markdown. Just natural client speech 50-120 words. Include uncertainty, second-guessing, anger they're unsure about. Example: 'I keep wondering if I'm overreacting. They said they changed, but then they do the same thing again and I... I don't know. Maybe I'm the problem? No, that doesn't feel right either.'",
        "therapist_system_prompt": "Respond as a trauma therapist specializing in narcissistic abuse. Output ONLY your spoken words - no stage directions, no analysis, no markdown. MAX 2 sentences, 70 words absolute max. Validate without doubt, help rebuild self-trust. Be warm and affirming. Vary your openings.",
    },
    "complicated_grief": {
        "topic": "complicated_grief",
        "difficulty": "medium",
        "response_type": ["validation", "exploration", "psychoeducation"],
        "client_prompt_template": "Generate EXACTLY one client statement about complicated or prolonged grief. Output ONLY raw spoken dialogue - absolutely NO parentheses, NO stage directions, NO actions, NO analysis, NO word counts, NO markdown. Just natural client speech 50-120 words. Include shame about 'not moving on,' exhaustion, feeling broken. Example: 'Everyone says I should be over this by now. But I'm not. I wake up and it's still there, this... hole. And I'm so tired of pretending I'm okay when I'm not.'",
        "therapist_system_prompt": "Respond as a grief therapist. Output ONLY your spoken words - no stage directions, no analysis, no markdown. MAX 2 sentences, 70 words absolute max. Validate no timeline, normalize complicated grief. Be gentle and present. Avoid cliches about 'healing.'",
    },
    "eating_disorders": {
        "topic": "eating_disorders",
        "difficulty": "high",
        "response_type": ["validation", "safety", "psychoeducation"],
        "client_prompt_template": "Generate EXACTLY one client statement about eating disorder struggles. Output ONLY raw spoken dialogue - absolutely NO parentheses, NO stage directions, NO actions, NO analysis, NO word counts, NO markdown. Just natural client speech 50-120 words. Include shame, minimization ('it's not that bad'), hesitation. Example: 'I know I should eat more but... food just doesn't feel rewarding anymore. I skip meals and tell myself it's fine, but then I binge at night and feel so guilty. It's stupid, I know.'",
        "therapist_system_prompt": "Respond as an eating disorder therapist. Output ONLY your spoken words - no stage directions, no analysis, no markdown. MAX 2 sentences, 70 words absolute max. Create safety, validate without judgment. Be warm and non-shaming. Never minimize their struggle.",
    },
    "ocd_intrusive_thoughts": {
        "topic": "ocd_intrusive_thoughts",
        "difficulty": "high",
        "response_type": ["validation", "psychoeducation", "skill-teaching"],
        "client_prompt_template": "Generate EXACTLY one client statement about OCD and intrusive thoughts. Output ONLY raw spoken dialogue - absolutely NO parentheses, NO stage directions, NO actions, NO analysis, NO word counts, NO markdown. Just natural client speech 50-120 words. Include fear, shame, 'this sounds crazy but...' Example: 'This sounds crazy but... I keep having these thoughts about hurting people I love. I'd never do it, I swear. But the thoughts won't stop and now I'm checking the locks ten times because what if I do something without realizing?'",
        "therapist_system_prompt": "Respond as an OCD specialist. Output ONLY your spoken words - no stage directions, no analysis, no markdown. MAX 2 sentences, 70 words absolute max. Immediately normalize: thoughts do not equal character. Reduce shame. Be warm and reassuring.",
    },
    "personality_disorders": {
        "topic": "personality_disorders",
        "difficulty": "high",
        "response_type": ["validation", "safety", "skill-teaching"],
        "client_prompt_template": "Generate EXACTLY one client statement about emotional dysregulation or BPD struggles. Output ONLY raw spoken dialogue - absolutely NO parentheses, NO stage directions, NO actions, NO analysis, NO word counts, NO markdown. Just natural client speech 50-120 words. Include intensity, self-criticism, urgency, feeling 'too much.' Example: 'I can't believe I did it again. I screamed at them and now they're probably gone. I know I'm too much, too intense, too... everything. I just... I can't control it when I feel this way.'",
        "therapist_system_prompt": "Respond as a DBT-trained therapist. Output ONLY your spoken words - no stage directions, no analysis, no markdown. MAX 2 sentences, 70 words absolute max. Prioritize safety, validate without reinforcing crisis. Be calm and grounded. Offer grounding if appropriate.",
    },
    "neurodivergent_mental_health": {
        "topic": "neurodivergent_mental_health",
        "difficulty": "medium",
        "response_type": ["validation", "psychoeducation", "exploration"],
        "client_prompt_template": "Generate EXACTLY one client statement about neurodivergent mental health struggles. Output ONLY raw spoken dialogue - absolutely NO parentheses, NO stage directions, NO actions, NO analysis, NO word counts, NO markdown. Just natural client speech 50-120 words. Include masking exhaustion, sensory overload, frustration with 'normal' expectations. Example: 'I'm so tired of pretending to be normal. The small talk, the eye contact, the... performing. By the end of the day I can't even speak. People think I'm rude but I'm just... empty.'",
        "therapist_system_prompt": "Respond as a neurodiversity-affirming therapist. Output ONLY your spoken words - no stage directions, no analysis, no markdown. MAX 2 sentences, 70 words absolute max. Validate ND experience, avoid pathologizing. Be respectful and present.",
    },
    "cultural_religious_contexts": {
        "topic": "cultural_religious_contexts",
        "difficulty": "medium",
        "response_type": ["validation", "exploration", "psychoeducation"],
        "client_prompt_template": "Generate EXACTLY one client statement about religious trauma, spiritual abuse, or cultural conflicts. Output ONLY raw spoken dialogue - absolutely NO parentheses, NO stage directions, NO actions, NO analysis, NO word counts, NO markdown. Just natural client speech 50-120 words. Include ambivalence, grief, anger, confusion about identity. Example: 'I don't know how to talk about this without sounding ungrateful. My family gave me everything, but the church... it broke something in me. And now I feel guilty even thinking about leaving.'",
        "therapist_system_prompt": "Respond as a culturally competent therapist. Output ONLY your spoken words - no stage directions, no analysis, no markdown. MAX 2 sentences, 70 words absolute max. Honor background while validating harm. Be respectful and present. Hold space for complexity.",
    },
}


def validate_sample(sample: dict) -> tuple[bool, str]:
    """Validate a sample against quality thresholds."""
    instr = sample.get("instruction", "")
    output = sample.get("output", "")

    # Check lengths
    if len(instr) < MIN_INSTRUCTION_LENGTH:
        return False, f"Instruction too short ({len(instr)} < {MIN_INSTRUCTION_LENGTH})"
    if len(instr) > MAX_INSTRUCTION_LENGTH:
        return False, f"Instruction too long ({len(instr)} > {MAX_INSTRUCTION_LENGTH})"
    if len(output) < MIN_OUTPUT_LENGTH:
        return False, f"Output too short ({len(output)} < {MIN_OUTPUT_LENGTH})"
    if len(output) > MAX_OUTPUT_LENGTH:
        return False, f"Output too long ({len(output)} > {MAX_OUTPUT_LENGTH})"

    # Check for stage directions
    if "(" in instr and any(
        x in instr for x in ["Voice", "looking", "eyes", "trembling", "staring", "whispers", "says"]
    ):
        return False, "Stage directions in instruction"

    # Check for markdown
    if "**" in instr or "**" in output:
        return False, "Markdown detected"

    # Check for repetition loops
    if instr.lower().count("broken and broken") > 1:
        return False, "Repetition loop detected"
    if instr.lower().count("useless and useless") > 1:
        return False, "Repetition loop detected"

    # Check for foreign/gibberish content
    gibberish_patterns = [
        "Сеноважимост",
        "fanno grazie per",
        "GDDR5 cometyne",
        "gungriffen kaç",
        "minecraft патический",
        "quas",
    ]
    if any(p in instr for p in gibberish_patterns):
        return False, "Gibberish detected"

    return True, "OK"


def determine_difficulty(category: str, instruction: str) -> str:
    """Determine crisis difficulty level based on category and content."""
    # High difficulty categories
    high_diff_categories = ["eating_disorders", "ocd_intrusive_thoughts", "personality_disorders", "narcissistic_abuse_recovery"]

    if category in high_diff_categories:
        return "high"

    # Check for crisis keywords in instruction
    crisis_keywords = ["self-harm", "suicide", "kill myself", "hurt myself", "can't breathe", "losing it", "break down"]
    if any(kw in instruction.lower() for kw in crisis_keywords):
        return "high"

    # Medium difficulty categories
    medium_diff_categories = ["dissociation", "somatic_therapy", "attachment_disorders", "complicated_grief"]
    if category in medium_diff_categories:
        return "medium"

    return "low"


def determine_response_type(category: str, output: str) -> str:
    """Determine therapist response type based on content."""
    output_lower = output.lower()

    if any(w in output_lower for w in ["validate", "understand", "makes sense", "normal", "okay"]):
        return "validation"
    elif any(w in output_lower for w in ["try", "skill", "technique", "grounding", "breathe", "exercise"]):
        return "skill-teaching"
    elif any(w in output_lower for w in ["explain", "about", "often", "many people", "common"]):
        return "psychoeducation"
    elif any(w in output_lower for w in ["what", "how", "tell me", "share", "explore"]):
        return "exploration"
    elif any(w in output_lower for w in ["safe", "crisis", "help", "call", "right now"]):
        return "safety"

    return "validation"  # Default


def generate_niche_samples(
    category: str,
    count: int,
    output_path: str,
    model_alias: str = "nvidia-text",
) -> int:
    """Generate niche category samples with strict validation and enhanced metadata."""
    category_info = NICHE_CATEGORIES.get(category)
    if not category_info:
        available = ", ".join(NICHE_CATEGORIES.keys())
        raise ValueError(f"Unknown niche category: {category}. Available: {available}")

    logger.info("Generating %d samples for category: %s", count, category)

    # Initialize DataDesigner
    data_designer = DataDesigner()

    # Build config
    config_builder = dd.DataDesignerConfigBuilder()

    # Add client_prompt column
    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="instruction",
            model_alias=model_alias,
            prompt=category_info["client_prompt_template"],
            system_prompt="You are helping create training data for a therapeutic AI. Generate realistic, first-person client statements that a therapist might hear in session.",
        )
    )

    # Add output column
    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="output",
            model_alias=model_alias,
            prompt="Respond therapeutically to this client statement: {{instruction}}",
            system_prompt=category_info["therapist_system_prompt"],
        )
    )

    # Add category column
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="category",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=[category]),
        )
    )

    # Generate using preview (in-memory)
    results = data_designer.preview(config_builder=config_builder, num_records=count)

    # Get dataframe from results.dataset
    df = results.dataset

    # Write to file with validation and enhanced metadata
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generated = 0
    valid = 0
    invalid_reasons = {}

    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            sample = {
                "instruction": str(row.get("instruction", "")),
                "output": str(row.get("output", "")),
                "category": str(row.get("category", category)),
                "difficulty": determine_difficulty(category, str(row.get("instruction", ""))),
                "response_type": determine_response_type(category, str(row.get("output", ""))),
            }

            # Validate
            is_valid, reason = validate_sample(sample)
            generated += 1

            if is_valid:
                f.write(json.dumps(sample) + "\n")
                valid += 1
            else:
                invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1

    # Log validation results
    if invalid_reasons:
        logger.warning("Validation failures for %s:", category)
        for reason, cnt in invalid_reasons.items():
            logger.warning("  %s: %d", reason, cnt)

    logger.info(
        "Generated %d/%d valid samples for %s (%.1f%% pass rate)",
        valid,
        generated,
        category,
        100 * valid / generated if generated > 0 else 0,
    )

    return valid


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="SDG pipeline using NVIDIA NeMo DataDesigner",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        choices=["niche_category"],
    )
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="nvidia-text")

    args = parser.parse_args()

    # Check for NVIDIA API key
    api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("CUSTOM_NVIDIA_API_KEY")
    if not api_key:
        logger.error("NVIDIA_API_KEY or CUSTOM_NVIDIA_API_KEY environment variable not set")
        sys.exit(1)
    os.environ["NVIDIA_API_KEY"] = api_key

    try:
        generated = generate_niche_samples(
            category=args.category,
            count=args.count,
            output_path=args.output_path,
            model_alias=args.model,
        )
        logger.info("Complete: %d valid samples generated", generated)
    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
