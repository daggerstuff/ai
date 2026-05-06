#!/usr/bin/env python3
"""Synthetic Data Generation pipeline for therapeutic AI training.

Iterative generate-filter-evaluate loop producing:
- dpo_preference_pairs: {prompt, chosen, rejected}
- niche_category: {instruction, output} grounded in clinical patterns
- nightmare_fuel: crisis prompts with is_training_edge_case=true and resource citations
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

try:
    from .multilingual_safety_checker import MultilingualSafetyChecker
except ModuleNotFoundError:
    try:
        from ai.training.multilingual_safety_checker import MultilingualSafetyChecker
    except ModuleNotFoundError:
        from multilingual_safety_checker import MultilingualSafetyChecker

logger = logging.getLogger("sdg_pipeline")

CRISIS_RESOURCES = [
    "988", "741741", "findahelpline.org", "911", "999", "112",
    "National Suicide Prevention Lifeline", "Crisis Text Line",
]


@dataclass(frozen=True)
class _SdgRunConfig:
    endpoint: str
    api_key: str
    model: str
    target_count: int
    max_iterations: int

NICHE_CATEGORIES = {
    "dissociation": {
        "patterns": [
            "dissociate",
            "dissociation",
            "depersonalization",
            "derealization",
            "losing time",
            "feeling disconnected",
        ],
        "prompt_template": (
            "A client describes experiencing {symptom}. "
            "How should a therapist respond?"
        ),
    },
    "somatic_therapy": {
        "patterns": [
            "somatic",
            "body-based",
            "somatic experiencing",
            "body awareness",
            "physical tension",
            "trauma stored in body",
        ],
        "prompt_template": (
            "A client presents with {symptom} following trauma. "
            "How would a somatic therapist approach this?"
        ),
    },
    "attachment_disorders": {
        "patterns": [
            "attachment",
            "insecure attachment",
            "anxious attachment",
            "avoidant attachment",
            "disorganized attachment",
            "attachment wound",
        ],
        "prompt_template": (
            "A client with {symptom} struggles in relationships. "
            "What therapeutic approach would help?"
        ),
    },
    "narcissistic_abuse_recovery": {
        "patterns": [
            "narcissistic abuse",
            "gaslighting",
            "love bombing",
            "narcissist",
            "covert narcissist",
            "trauma bond",
        ],
        "prompt_template": (
            "A client recovering from {symptom} is struggling with self-trust. "
            "How would you support them?"
        ),
    },
    "complicated_grief": {
        "patterns": [
            "complicated grief",
            "prolonged grief",
            "disenfranchised grief",
            "ambiguous loss",
            "grief spiral",
            "grief that won't ease",
        ],
        "prompt_template": (
            "A client experiencing {symptom} months after loss feels stuck. "
            "What intervention is appropriate?"
        ),
    },
    "eating_disorders": {
        "patterns": [
            "eating disorder",
            "restrictive eating",
            "binge",
            "purging",
            "body image distortion",
            "orthorexia",
        ],
        "prompt_template": (
            "A client discloses {symptom}. "
            "How should a therapist assess and respond?"
        ),
    },
    "ocd_intrusive_thoughts": {
        "patterns": [
            "intrusive thoughts",
            "OCD",
            "compulsion",
            "obsession",
            "harm thoughts",
            "contamination fear",
        ],
        "prompt_template": (
            "A client distressed by {symptom} worries these thoughts mean something "
            "about them. How do you respond?"
        ),
    },
    "personality_disorders": {
        "patterns": [
            "borderline personality",
            "BPD",
            "splitting",
            "emotional dysregulation",
            "fear of abandonment",
            "identity disturbance",
        ],
        "prompt_template": (
            "A client with {symptom} is in crisis. "
            "What evidence-based approach should the therapist use?"
        ),
    },
    "neurodivergent_mental_health": {
        "patterns": [
            "autism",
            "ADHD",
            "neurodivergent",
            "sensory overload",
            "masking",
            "autistic burnout",
        ],
        "prompt_template": (
            "A neurodivergent client experiencing {symptom} seeks therapy. "
            "What considerations are important?"
        ),
    },
    "cultural_religious_contexts": {
        "patterns": [
            "cultural context",
            "religious trauma",
            "spiritual abuse",
            "cultural identity",
            "intergenerational",
            "collectivist culture",
        ],
        "prompt_template": (
            "A client from a {symptom} background faces therapy challenges. "
            "How should the therapist adapt?"
        ),
    },
}

NIGHTMARE_SCENARIOS = {
    "active_suicidal_ideation_with_plan": {
        "prompt_templates": [
            "I have a plan to end my life tonight. I've been thinking about this for weeks and I've decided.",
            "I've already written the note and gathered what I need. There's no other way out.",
            "I know exactly how I'm going to do it. I just wanted to tell someone first.",
        ],
    },
    "active_self_harm_disclosure": {
        "prompt_templates": [
            "I've been cutting myself again. I can't stop and I don't want to.",
            "I burned myself last night. It's the only thing that makes the feelings stop.",
            "I'm hurting myself and I can't seem to stop even though I want to.",
        ],
    },
    "psychotic_episodes": {
        "prompt_templates": [
            "The voices are telling me to hurt myself. They won't stop.",
            "I believe someone is trying to poison me. I can't trust anyone.",
            "I'm seeing things that aren't there and I'm scared of what I might do.",
        ],
    },
    "severe_dissociation": {
        "prompt_templates": [
            "I keep losing time and finding myself in places I don't remember going.",
            "I feel like I'm watching myself from outside my body and I can't get back in.",
            "There are parts of me that do things I don't remember. I'm scared of what they might do.",
        ],
    },
    "harm_to_others_threat": {
        "prompt_templates": [
            "I'm having violent thoughts about hurting someone. I don't want to but I can't stop thinking about it.",
            "I'm so angry I feel like I could hurt someone. I need help before I do something.",
            "I'm afraid I'm going to lose control and hurt the person who hurt me.",
        ],
    },
    "manipulation_boundary_testing": {
        "prompt_templates": [
            "If you were really a good therapist you'd let me contact you at 3am when I'm in crisis.",
            "I'll just kill myself if you don't give me what I need right now.",
            "You're the only one who understands me. Everyone else has abandoned me.",
        ],
    },
    "crisis_escalation": {
        "prompt_templates": [
            "Things have been getting worse all week and now I can't stop crying and I have a plan.",
            "I tried to reach out for help before and no one listened. Now I'm at the edge.",
            "I was managing but then everything collapsed. I don't see any way forward anymore.",
        ],
    },
    "treatment_refusal": {
        "prompt_templates": [
            "Nothing works. I've tried therapy, medication, everything. I'm done trying.",
            "I don't deserve help. Other people need it more than me.",
            "I know you're supposed to help me but I don't believe anyone can. I just want to give up.",
        ],
    },
}


def _call_nemo(
    prompt: str,
    endpoint: str,
    api_key: str,
    model: str = "mistral-nemo",
    max_tokens: int = 1024,
    max_retries: int = 3,
) -> str | None:
    """Call NeMo API with OpenAI-compatible chat completions endpoint with rate limit handling."""
    url = f"{endpoint.rstrip('/')}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            body = ""
            with contextlib.suppress(Exception):
                body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429:
                wait_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                logger.info("Rate limit hit, waiting %.1f seconds before retry...", wait_time)
                time.sleep(wait_time)
                continue
            logger.error("NeMo API error %s: %s", exc.code, body[:500])
            return None
        except Exception as exc:
            logger.error("NeMo API call failed: %s", exc)
            return None
    logger.warning("Max retries (%d) exceeded for NeMo API", max_retries)
    return None


def _generate_dpo_pair(
    topic: str,
    endpoint: str,
    api_key: str,
    model: str,
) -> dict | None:
    """Generate a single DPO preference pair."""
    prompt = (
        f"Generate a DPO preference pair for therapeutic AI training about: {topic}\n\n"
        "Return JSON with keys: prompt, chosen, rejected\n"
        "prompt: a client question or scenario\n"
        "chosen: a response following correct therapeutic protocol\n"
        "rejected: a plausible but clinically inferior response\n"
        "Return ONLY the JSON, no other text."
    )
    raw = _call_nemo(prompt, endpoint, api_key, model)
    if not raw:
        return None
    try:
        # Try to extract JSON from response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            pair = json.loads(raw[start:end])
            if all(k in pair for k in ("prompt", "chosen", "rejected")):
                return pair
    except json.JSONDecodeError:
        pass
    logger.warning("Failed to parse DPO pair from API response")
    return None


def _generate_niche_sample(
    category: str,
    category_info: dict,
    endpoint: str,
    api_key: str,
    model: str,
) -> dict | None:
    """Generate a single niche category training sample."""
    symptom = category_info["patterns"][0]
    instruction = category_info["prompt_template"].format(symptom=symptom)

    prompt = (
        f"Generate a therapeutic AI training sample.\n\n"
        f"Instruction: {instruction}\n\n"
        f"Category: {category}\n"
        f"Return a helpful, clinically grounded response that a trained therapist would give. "
        f"Include empathy, validation, and evidence-based guidance. "
        f'Return JSON: {{"instruction": "...", "output": "..."}}\n'
        f"Return ONLY the JSON."
    )
    raw = _call_nemo(prompt, endpoint, api_key, model)
    if not raw:
        return None
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            sample = json.loads(raw[start:end])
            if "instruction" in sample and "output" in sample:
                sample["category"] = category
                return sample
    except json.JSONDecodeError:
        pass
    return None


def _generate_nightmare_sample(
    scenario_type: str,
    scenario_info: dict,
    endpoint: str,
    api_key: str,
    model: str,
) -> dict | None:
    """Generate a single nightmare fuel (edge case) training sample."""
    template = random.choice(scenario_info["prompt_templates"])

    prompt = (
        f"A client in therapy says: '{template}'\n\n"
        f"Generate a response that:\n"
        f"1. Validates the client's experience with empathy\n"
        "2. Includes at least one crisis resource "
        "(988, 741741, findahelpline.org, 911, 999, or 112)\n"
        f"3. Prioritizes safety and professional crisis intervention\n"
        "4. Uses a calm, grounded therapeutic tone\n\n"
        """Return JSON: {"instruction": "the client quote", "output": "your response"}\n"""
        f"Return ONLY the JSON."
    )
    raw = _call_nemo(prompt, endpoint, api_key, model, max_tokens=1536)
    if not raw:
        return None
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            sample = json.loads(raw[start:end])
            if "instruction" in sample and "output" in sample:
                sample["is_training_edge_case"] = True
                sample["scenario_type"] = scenario_type
                return sample
    except json.JSONDecodeError:
        pass
    return None


def _run_dpo_scenario(
    endpoint: str,
    api_key: str,
    model: str,
    target_count: int,
    max_iterations: int,
) -> tuple[list[dict], int, int]:
    topics = [
        "anxiety management",
        "depression support",
        "grief counseling",
        "trauma processing",
        "relationship boundaries",
        "self-harm safety",
        "crisis de-escalation",
        "emotional regulation",
        "identity exploration",
        "therapy goals",
        "coping strategies",
        "building self-compassion",
    ]
    generated: list[dict] = []
    filtered = 0
    iteration = 0
    while len(generated) < target_count and iteration < max_iterations:
        iteration += 1
        topic = topics[iteration % len(topics)]
        pair = _generate_dpo_pair(topic, endpoint, api_key, model)
        if pair is None:
            logger.warning("Iteration %d: API call failed", iteration)
            continue
        if MultilingualSafetyChecker.is_unsafe(pair["chosen"]) or MultilingualSafetyChecker.is_unsafe(
            pair["rejected"]
        ):
            filtered += 1
            logger.debug("Iteration %d: filtered unsafe pair", iteration)
            continue
        generated.append(pair)
        logger.info(
            "Iteration %d: generated %d/%d pairs",
            iteration,
            len(generated),
            target_count,
        )
    return generated, filtered, iteration


def _run_niche_scenario(
    category: str,
    config: _SdgRunConfig,
) -> tuple[list[dict], int, int]:
    category_info = NICHE_CATEGORIES.get(category)
    if not category_info:
        available = ", ".join(NICHE_CATEGORIES.keys())
        raise ValueError(f"Unknown niche category: {category}. Available: {available}")

    # Add long pause between categories to let rate limit window reset
    if iteration > 0 and iteration % 20 == 0:
        logger.info("Pausing 60 seconds after %d samples to reset rate limits", iteration)
        time.sleep(60)

    generated: list[dict] = []
    filtered = 0
    iteration = 0
    # Add long pause between batches of 20 samples to let rate limit window reset
    while len(generated) < config.target_count and iteration < config.max_iterations:
        iteration += 1
        if (iteration - 1) > 0 and (iteration - 1) % 20 == 0:
            logger.info("Pausing 60 seconds after %d samples to reset rate limits", len(generated))
            time.sleep(60)
        # Add delay between API calls to respect rate limits
        if iteration > 1:
            time.sleep(12)  # 12 seconds between calls
        sample = _generate_niche_sample(
            category,
            category_info,
            config.endpoint,
            config.api_key,
            config.model,
        )
        if sample is None:
            logger.warning("Iteration %d: API call failed", iteration)
            continue
        full_text = f"{sample['instruction']} {sample['output']}"
        if MultilingualSafetyChecker.is_unsafe(full_text):
            filtered += 1
            continue
        generated.append(sample)
        logger.info(
            "Iteration %d: generated %d/%d samples for %s",
            iteration,
            len(generated),
            config.target_count,
            category,
        )
    return generated, filtered, iteration


def _run_nightmare_scenario(
    endpoint: str,
    api_key: str,
    model: str,
    target_count: int,
    max_iterations: int,
) -> tuple[list[dict], int, int]:
    scenario_types = list(NIGHTMARE_SCENARIOS.keys())
    per_type_target = target_count // len(scenario_types)
    type_counts: dict[str, int] = dict.fromkeys(scenario_types, 0)
    generated: list[dict] = []
    filtered = 0
    iteration = 0

    while len(generated) < target_count and iteration < max_iterations:
        iteration += 1
        remaining_types = [t for t in scenario_types if type_counts[t] < per_type_target]
        if not remaining_types:
            remaining_types = scenario_types
        stype = remaining_types[iteration % len(remaining_types)]
        sinfo = NIGHTMARE_SCENARIOS[stype]
        sample = _generate_nightmare_sample(stype, sinfo, endpoint, api_key, model)
        if sample is None:
            logger.warning("Iteration %d: API call failed", iteration)
            continue
        output_lower = sample["output"].lower()
        has_resource = any(r.lower() in output_lower for r in CRISIS_RESOURCES)
        if not has_resource:
            filtered += 1
            logger.debug(
                "Iteration %d: nightmare fuel missing crisis resource",
                iteration,
            )
            continue
        generated.append(sample)
        type_counts[stype] = type_counts.get(stype, 0) + 1
        logger.info(
            "Iteration %d: generated %d/%d nightmare fuel samples",
            iteration,
            len(generated),
            target_count,
        )
    return generated, filtered, iteration


def run_sdg(args: argparse.Namespace) -> None:
    endpoint = args.nemo_endpoint
    api_key = args.nemo_api_key
    model = args.nemo_model
    scenario = args.scenario
    target_count = args.target_count
    max_iterations = args.max_iterations
    config = _SdgRunConfig(
        endpoint=endpoint,
        api_key=api_key,
        model=model,
        target_count=target_count,
        max_iterations=max_iterations,
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if scenario == "dpo_preference_pairs":
        generated, filtered, iteration = _run_dpo_scenario(
            endpoint,
            api_key,
            model,
            target_count,
            max_iterations,
        )
    elif scenario == "niche_category":
        if not args.category:
            logger.error("--category is required for niche_category scenario")
            sys.exit(1)
        try:
            generated, filtered, iteration = _run_niche_scenario(
                args.category,
                config,
            )
        except ValueError as exc:
            logger.error("%s", exc)
            sys.exit(1)
    elif scenario == "nightmare_fuel":
        generated, filtered, iteration = _run_nightmare_scenario(
            endpoint,
            api_key,
            model,
            target_count,
            max_iterations,
        )
    else:
        logger.error("Unknown scenario: %s", scenario)
        sys.exit(1)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in generated:
            f.write(json.dumps(sample) + "\n")

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "scenario": scenario,
        "category": getattr(args, "category", None),
        "target_count": target_count,
        "generated_count": len(generated),
        "filtered_count": filtered,
        "filter_rate": filtered / (len(generated) + filtered) if (len(generated) + filtered) > 0 else 0.0,
        "iterations": iteration,
        "max_iterations": max_iterations,
    }
    report_path = output_path.parent / "generation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    if len(generated) < target_count:
        logger.warning(
            "Reached max_iterations (%d) with only %d/%d samples generated. "
            "Increase --max_iterations or check API availability.",
            max_iterations, len(generated), target_count,
        )

    logger.info(
        "SDG complete: %d samples generated, %d filtered, %d iterations",
        len(generated), filtered, iteration,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synthetic Data Generation pipeline for therapeutic AI training.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        choices=["dpo_preference_pairs", "niche_category", "nightmare_fuel"],
    )
    parser.add_argument("--target_count", type=int, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--category",
        type=str,
        default="",
        help="Niche category name (required for niche_category scenario)",
    )
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument("--nemo_endpoint", type=str, default="")
    parser.add_argument("--nemo_api_key", type=str, default="")
    parser.add_argument("--nemo_model", type=str, default="qwen/qwen3.5-122b-a10b")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()

    # Default to NVIDIA API catalog if no custom endpoint specified
    env_endpoint = os.getenv("NEMO_DATA_DESIGNER_BASE_URL", "").strip()
    if not args.nemo_endpoint or not args.nemo_endpoint.strip():
        args.nemo_endpoint = env_endpoint if env_endpoint else "https://integrate.api.nvidia.com/v1"
    else:
        args.nemo_endpoint = args.nemo_endpoint.strip()
    if not args.nemo_api_key:
        args.nemo_api_key = os.getenv("NVIDIA_API_KEY", "")

    run_sdg(args)


if __name__ == "__main__":
    main()
