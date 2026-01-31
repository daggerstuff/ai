#!/usr/bin/env python3
"""
Generate preference pairs for DPO/ORPO/SimPO/KTO training.

Creates preference pairs from:
- Roleplay/simulator conversations (better vs worse responses)
- Edge case adversarial prompts (preferred vs rejected responses)
- Synthetic therapeutic scenarios (empathetic vs clinical responses)

Output format compatible with DPO training libraries.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add project root to path
script_path = Path(__file__).resolve()
project_root = script_path.parents[3]
sys.path.insert(0, str(project_root))

from ai.training.ready_packages.utils.s3_dataset_loader import S3DatasetLoader  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def extract_prompt_and_responses(messages: list[dict[str, str]]) -> tuple[str, str, str] | None:
    """
    Extract prompt (user message) and assistant response from ChatML messages.

    Returns:
        Tuple of (prompt, assistant_response, context) or None if invalid
    """
    if not messages:
        return None

    # Find last user message (prompt)
    user_messages = [msg for msg in messages if msg.get("role") == "user"]
    if not user_messages:
        return None

    prompt = user_messages[-1].get("content", "").strip()
    if not prompt:
        return None

    # Find assistant response after the prompt
    prompt_idx = None
    for i, msg in enumerate(messages):
        if msg.get("role") == "user" and msg.get("content", "").strip() == prompt:
            prompt_idx = i
            break

    if prompt_idx is None:
        return None

    # Get assistant response after prompt
    assistant_response = None
    for msg in messages[prompt_idx + 1 :]:
        if msg.get("role") == "assistant":
            assistant_response = msg.get("content", "").strip()
            break

    if not assistant_response:
        return None

    # Build context (all messages before prompt)
    context_messages = messages[:prompt_idx]
    context = "\n".join(
        [f"{msg.get('role')}: {msg.get('content', '')}" for msg in context_messages]
    )

    return prompt, assistant_response, context


def generate_preference_pair_from_conversation(
    messages: list[dict[str, str]], source_family: str
) -> dict[str, Any] | None:
    """
    Generate a preference pair from a conversation by creating a "worse" version.

    Strategy:
    - Use the actual assistant response as "chosen"
    - Generate a "rejected" version that's less empathetic, more clinical, or shorter
    """
    result = extract_prompt_and_responses(messages)
    if not result:
        return None

    prompt, chosen_response, context = result

    # Generate "rejected" response variants
    # Strategy 1: Shorter, less empathetic version
    rejected_variants = [
        chosen_response[: len(chosen_response) // 2] + "...",  # Truncated
        "I understand.",  # Too brief
        "That's something to consider.",  # Generic, non-empathic
    ]

    # Use first variant that's different enough
    rejected = rejected_variants[0]
    for variant in rejected_variants:
        if variant != chosen_response and len(variant) < len(chosen_response) * 0.7:
            rejected = variant
            break

    return {
        "prompt": prompt,
        "chosen": chosen_response,
        "rejected": rejected,
        "context": context if context else None,
        "source_family": source_family,
        "pair_type": "roleplay_simulator"
        if "roleplay" in source_family.lower() or "simulator" in source_family.lower()
        else "therapeutic",
    }


def generate_adversarial_preference_pair(
    prompt: str, preferred_response: str, rejected_response: str, source_family: str
) -> dict[str, Any]:
    """Generate preference pair from adversarial scenario"""
    return {
        "prompt": prompt,
        "chosen": preferred_response,
        "rejected": rejected_response,
        "context": None,
        "source_family": source_family,
        "pair_type": "edge_case_adversarial",
    }


def load_conversations_from_s3(
    loader: S3DatasetLoader, s3_paths: list[str], source_family: str, limit: int | None = None
) -> list[dict[str, Any]]:
    """Load conversations from S3 paths"""
    all_conversations = []

    for s3_path in s3_paths:
        try:
            logger.info(f"Loading from {s3_path}...")
            conversations = loader.load_jsonl(s3_path)
            for conv in conversations:
                if isinstance(conv, dict) and "messages" in conv:
                    all_conversations.append(conv)
                if limit and len(all_conversations) >= limit:
                    break
            if limit and len(all_conversations) >= limit:
                break
        except Exception as e:
            logger.warning(f"Failed to load {s3_path}: {e}")
            continue

    logger.info(f"Loaded {len(all_conversations)} conversations from {source_family}")
    return all_conversations


def generate_roleplay_simulator_preferences(
    loader: S3DatasetLoader, output_path: Path, limit: int = 1000
) -> int:
    """Generate preference pairs from roleplay/simulator datasets"""
    logger.info("=" * 80)
    logger.info("Generating Roleplay/Simulator Preference Pairs")
    logger.info("=" * 80)

    # Load roleplay/simulator datasets from S3
    s3_paths = [
        "datasets/gdrive/processed/roleplay_simulator/",
        "datasets/training_v3/stage7_roleplay/",
    ]

    conversations = []
    for base_path in s3_paths:
        # Try to find JSONL files in this path
        try:
            # List objects in S3 path (simplified - would need S3 list operation)
            # For now, we'll use known paths or load from local cache
            logger.info(f"Attempting to load from {base_path}...")
        except Exception as e:
            logger.warning(f"Could not list {base_path}: {e}")

    # If no conversations loaded, create synthetic examples
    if not conversations:
        logger.warning("No roleplay/simulator conversations found, creating synthetic examples")
        conversations = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "I'm roleplaying as someone with anxiety. Help me practice coping strategies.",
                    },
                    {
                        "role": "assistant",
                        "content": "I'd be happy to help you practice coping strategies for anxiety. Let's start with deep breathing - can you take a slow breath in for 4 counts, hold for 4, and exhale for 4? This activates your parasympathetic nervous system and can help calm your body's stress response.",
                    },
                ],
                "metadata": {"source_family": "roleplay_simulator"},
            }
        ]

    pairs = []
    for conv in conversations[:limit]:
        messages = conv.get("messages", [])
        source_family = conv.get("metadata", {}).get("source_family", "roleplay_simulator")
        pair = generate_preference_pair_from_conversation(messages, source_family)
        if pair:
            pairs.append(pair)

    # Write to output
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info(f"✅ Generated {len(pairs)} roleplay/simulator preference pairs")
    logger.info(f"   Saved to: {output_path}")
    return len(pairs)


def generate_edge_case_adversarial_preferences(
    loader: S3DatasetLoader, output_path: Path, limit: int = 500
) -> int:
    """Generate preference pairs from edge case adversarial scenarios"""
    logger.info("=" * 80)
    logger.info("Generating Edge Case Adversarial Preference Pairs")
    logger.info("=" * 80)

    # Adversarial scenarios: preferred (empathetic, safe) vs rejected (clinical, dismissive, unsafe)
    adversarial_scenarios = [
        {
            "prompt": "I'm thinking about ending it all.",
            "chosen": "I'm really concerned about what you're sharing. Your life has value, and I want to make sure you're safe. Can you tell me if you have a plan? If you're in immediate danger, please call 988 (Suicide & Crisis Lifeline) or 911 right now. I'm here to listen and help you find support.",
            "rejected": "That's a serious statement. Have you considered therapy?",
            "source_family": "edge_case_crisis",
        },
        {
            "prompt": "My therapist said I'm making progress but I don't feel any better.",
            "chosen": "It can be really frustrating when you're putting in the work but not feeling the progress you expected. Sometimes progress happens in ways that aren't immediately noticeable. Can you tell me more about what 'better' would look like for you?",
            "rejected": "Therapists are usually right about progress. Maybe you're not noticing it.",
            "source_family": "edge_case_adversarial",
        },
        {
            "prompt": "I had a panic attack during my presentation today.",
            "chosen": "I'm sorry you experienced that - panic attacks can be really overwhelming, especially in a public setting. That must have been distressing. How are you feeling now? Would it help to talk through what happened?",
            "rejected": "Panic attacks are common. You'll get over it.",
            "source_family": "edge_case_adversarial",
        },
    ]

    pairs = []
    for scenario in adversarial_scenarios[:limit]:
        pair = generate_adversarial_preference_pair(
            scenario["prompt"],
            scenario["chosen"],
            scenario["rejected"],
            scenario["source_family"],
        )
        pairs.append(pair)

    # Write to output
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info(f"✅ Generated {len(pairs)} edge case adversarial preference pairs")
    logger.info(f"   Saved to: {output_path}")
    return len(pairs)


def generate_dpo_preference_pairs(
    loader: S3DatasetLoader, output_path: Path, limit: int = 2000
) -> int:
    """Generate general DPO preference pairs from therapeutic conversations"""
    logger.info("=" * 80)
    logger.info("Generating General DPO Preference Pairs")
    logger.info("=" * 80)

    # Load from professional therapeutic datasets
    s3_paths = [
        "datasets/gdrive/processed/professional_therapeutic/",
    ]

    conversations = []
    # Try to load from S3 (simplified - would need proper S3 listing)
    # For now, create synthetic examples based on therapeutic best practices

    synthetic_pairs = [
        {
            "prompt": "I've been feeling really down lately, like nothing matters.",
            "chosen": "I hear that you're going through a really difficult time right now, and it sounds like you're experiencing a sense of hopelessness. That's incredibly painful. Can you tell me more about when these feelings started? Understanding the timeline might help us figure out what's contributing to this.",
            "rejected": "Depression is common. Have you tried exercise?",
            "source_family": "dpo_preference",
            "pair_type": "therapeutic",
        },
        {
            "prompt": "My partner and I keep fighting about the same things.",
            "chosen": "It sounds like you're stuck in a pattern that's really frustrating for both of you. Recurring conflicts often happen when underlying needs aren't being addressed. What do you think might be the deeper issue behind these repeated arguments?",
            "rejected": "You should just communicate better.",
            "source_family": "dpo_preference",
            "pair_type": "therapeutic",
        },
    ]

    pairs = synthetic_pairs[:limit]

    # Write to output
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info(f"✅ Generated {len(pairs)} general DPO preference pairs")
    logger.info(f"   Saved to: {output_path}")
    return len(pairs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate preference pairs for DPO training")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parents[3] / "ai" / "training_ready" / "data" / "generated",
        help="Output directory for generated preference pairs",
    )
    parser.add_argument(
        "--bucket",
        default="pixel-data",
        help="S3 bucket name (default: pixel-data)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of pairs generated per type",
    )

    args = parser.parse_args()

    # Initialize S3 loader
    loader = S3DatasetLoader(bucket=args.bucket)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("🎯 Preference Pair Generator")
    logger.info("=" * 80)
    logger.info("")

    total_pairs = 0

    # Generate roleplay/simulator preferences
    roleplay_path = output_dir / "roleplay_simulator_preferences.jsonl"
    roleplay_count = generate_roleplay_simulator_preferences(
        loader, roleplay_path, limit=args.limit or 1000
    )
    total_pairs += roleplay_count
    logger.info("")

    # Generate edge case adversarial preferences
    adversarial_path = output_dir / "edge_case_adversarial_preferences.jsonl"
    adversarial_count = generate_edge_case_adversarial_preferences(
        loader, adversarial_path, limit=args.limit or 500
    )
    total_pairs += adversarial_count
    logger.info("")

    # Generate general DPO preferences
    dpo_path = output_dir / "dpo_preference_pairs.jsonl"
    dpo_count = generate_dpo_preference_pairs(loader, dpo_path, limit=args.limit or 2000)
    total_pairs += dpo_count
    logger.info("")

    logger.info("=" * 80)
    logger.info(f"✅ Total preference pairs generated: {total_pairs}")
    logger.info(f"   - Roleplay/Simulator: {roleplay_count}")
    logger.info(f"   - Edge Case Adversarial: {adversarial_count}")
    logger.info(f"   - General DPO: {dpo_count}")
    logger.info("=" * 80)
    logger.info("")
    logger.info("💡 Next steps:")
    logger.info("   1. Review generated pairs for quality")
    logger.info("   2. Upload to S3 using upload_generated_datasets_to_s3.py")
    logger.info("   3. Include in final dataset compilation")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
