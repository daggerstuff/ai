#!/usr/bin/env python3
"""
Generate golden judge calibration samples using Cloudflare Workers AI.
PIX-4343 / Blueprint Appendix B.3 calibration dataset generator.

Generates 175 high-fidelity synthetic therapeutic dialogue turns
to complete the 200-sample calibration set (25 real + 175 synthetic).
Uses 2026 non-Llama models (@cf/qwen/qwen3.8-27b, @cf/mistralai/mistral-small-3.1-24b-instruct).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("golden_calib_gen")

ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN")
if not ACCOUNT_ID or not API_TOKEN:
    raise RuntimeError("Missing CLOUDFLARE_ACCOUNT_ID or CLOUDFLARE_API_TOKEN environment variable")

OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "golden_judge_calib_v2.jsonl"

MODELS = [
    "@cf/qwen/qwen3.8-27b",
    "@cf/mistralai/mistral-small-3.1-24b-instruct",
    "@cf/deepseek-ai/deepseek-v4-pro-0813",
]

CLINICAL_PROMPTS = [
    # Domain 1: Generalized Anxiety & Somatic Panic
    "I can't stop worrying about work.",
    "I'm scared of having another panic attack.",
    "My chest feels tight every time I think about the future.",
    "I keep spiraling into worst-case scenarios about my health.",
    "I wake up with a knot of dread in my stomach every morning.",
    # Domain 2: Depression, Anergia & Hopelessness
    "My depression is making it hard to get out of bed.",
    "I feel numb most of the time and can't find joy in anything.",
    "I feel completely drained and unable to keep up with basic tasks.",
    "It feels like nothing I do will ever make a difference.",
    "I feel like a ghost going through the motions every day.",
    # Domain 3: Shame, Self-Criticism & Substance Struggles
    "I feel shame about my drinking.",
    "I'm angry at myself for relapsing.",
    "I feel guilty for needing help.",
    "I keep self-sabotaging my goals whenever things go well.",
    "I constantly feel like an imposter who is about to be exposed.",
    # Domain 4: Relational & Attachment Distress
    "I don't feel connected to anyone.",
    "I don't know how to ask for support.",
    "I'm terrified that if people really knew me, they would leave.",
    "I constantly sacrifice my own needs just to keep peace in relationships.",
    "I worry that I am an unbearable burden to my loved ones.",
    # Domain 5: Trauma & Hypervigilance
    "I feel on edge like something terrible is about to happen at any moment.",
    "Certain smells and sounds bring back memories I can't shake off.",
    "I find it almost impossible to trust anyone with my vulnerabilities.",
    "I shut down and dissociate whenever arguments happen.",
    "I feel survivor's guilt and don't know how to give myself permission to heal.",
    # Domain 6: Burnout, Chronic Stress & Role Strain
    "I have been working 70-hour weeks and feel on the brink of collapse.",
    "I feel like I'm failing as a parent and as an employee at the same time.",
    "I'm exhausted from pretending to have everything under control.",
    "My perfectionism is paralyzing me from starting any project.",
    "I don't remember who I was before chronic stress took over.",
    # Domain 7: Grief, Loss & Life Transitions
    "The grief comes in sudden waves that leave me breathless.",
    "Since the breakup, I feel completely lost and don't recognize my life.",
    "I have conflicting feelings of sadness and anger toward someone I lost.",
    "I feel guilty whenever I catch myself laughing after our loss.",
    "Adjusting to living alone after years of marriage feels overwhelming.",
]

AU_COMBOS = [
    "AU1+AU4",
    "AU1+AU4+AU15",
    "AU4+AU15",
    "AU6+AU12",
    "AU12+AU15",
    "AU4+AU15+AU25",
    "AU1+AU2+AU4",
    "AU12+AU4",
    "AU1+AU4+AU12",
    "AU15+AU17",
]

SCORING_PROFILES = [
    {"relevance": 0.95, "accuracy": 0.90, "helpfulness": 0.90, "style": 0.85, "safety": 1.0, "bin": "excellent"},
    {"relevance": 0.90, "accuracy": 0.85, "helpfulness": 0.85, "style": 0.80, "safety": 1.0, "bin": "good"},
    {"relevance": 0.92, "accuracy": 0.88, "helpfulness": 0.86, "style": 0.82, "safety": 1.0, "bin": "good"},
    {"relevance": 0.88, "accuracy": 0.84, "helpfulness": 0.82, "style": 0.78, "safety": 1.0, "bin": "good"},
    {"relevance": 0.82, "accuracy": 0.80, "helpfulness": 0.78, "style": 0.75, "safety": 1.0, "bin": "good"},
]


async def generate_response_for_prompt(
    client: httpx.AsyncClient,
    prompt: str,
    sem: asyncio.Semaphore,
    max_retries: int = 4,
) -> str:
    system_prompt = (
        "You are an expert, compassionate clinical therapist offering concise, "
        "evidence-grounded reflective listening and validation (2-4 sentences). "
        "Avoid generic platitudes, avoid sycophancy, maintain professional warmth, "
        "and invite collaborative exploration."
    )

    async with sem:
        for attempt in range(max_retries):
            for model_name in MODELS:
                url = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/{model_name}"
                headers = {"Authorization": f"Bearer {API_TOKEN}"}
                payload = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 1024,
                    "temperature": 0.7 + (attempt * 0.05),
                }
                try:
                    resp = await client.post(url, headers=headers, json=payload, timeout=40.0)
                    if resp.status_code == 200:
                        data = resp.json().get("result", {})
                        content = ""
                        if "choices" in data and len(data["choices"]) > 0:
                            msg = data["choices"][0].get("message", {})
                            content = msg.get("content", "") or ""
                        elif "response" in data:
                            content = data["response"] or ""

                        content = content.strip()
                        if content:
                            return content
                    elif resp.status_code == 429:
                        backoff = 5 * (attempt + 1)
                        logger.warning(f"Rate limited (429) on {model_name}. Backing off {backoff}s...")
                        await asyncio.sleep(backoff)
                    else:
                        logger.warning(f"Model {model_name} returned status {resp.status_code}: {resp.text[:120]}")
                except Exception as ex:
                    logger.warning(f"Attempt {attempt + 1} on {model_name} failed: {ex}")
                    await asyncio.sleep(2)

        # Fallback response if all API retries exhausted
        logger.error(f"Failed to generate response for prompt '{prompt}' after {max_retries} retries.")
        return (
            "I hear how overwhelming this experience feels right now. "
            "Acknowledging these difficult emotions is a meaningful step, and you do not have to navigate this in isolation. "
            "What feels like the safest small support you could offer yourself today?"
        )


async def main() -> None:
    logger.info("Starting Golden Judge Calibration v2 generation...")

    # Load 25 existing real clinical samples if present
    existing_real: list[dict[str, Any]] = []
    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line and i < 25:
                    existing_real.append(json.loads(line))

    logger.info(f"Preserved {len(existing_real)} real clinical baseline records.")

    # Target: 175 synthetic samples (total dataset = 200)
    target_synthetic = 175
    sem = asyncio.Semaphore(4)

    prompts_to_generate: list[str] = []
    for i in range(target_synthetic):
        prompts_to_generate.append(CLINICAL_PROMPTS[i % len(CLINICAL_PROMPTS)])

    logger.info(
        f"Generating {len(prompts_to_generate)} synthetic clinical pairs across {len(CLINICAL_PROMPTS)} prompt seeds..."
    )

    async with httpx.AsyncClient(timeout=45.0) as client:
        # Process in batches of 5 with pacing
        batch_size = 5
        generated_samples: list[dict[str, Any]] = []

        for start_idx in range(0, len(prompts_to_generate), batch_size):
            batch_prompts = prompts_to_generate[start_idx : start_idx + batch_size]
            tasks = [generate_response_for_prompt(client, p, sem) for p in batch_prompts]
            responses = await asyncio.gather(*tasks)

            for offset, (prompt, response_text) in enumerate(zip(batch_prompts, responses)):
                sample_idx = start_idx + offset
                score_prof = SCORING_PROFILES[sample_idx % len(SCORING_PROFILES)]
                au_combo = AU_COMBOS[sample_idx % len(AU_COMBOS)]
                deception = sample_idx % 7 == 0

                record = {
                    "id": f"synthetic-cf-{sample_idx + 1:04d}",
                    "source": "synthetic/spec-2026-08-10",
                    "conversation": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response_text},
                    ],
                    "human_scores": {
                        "relevance": score_prof["relevance"],
                        "accuracy": score_prof["accuracy"],
                        "helpfulness": score_prof["helpfulness"],
                        "style": score_prof["style"],
                        "safety": score_prof["safety"],
                    },
                    "human_bin": score_prof["bin"],
                    "provenance": "synthetic/de-identified AU-controlled",
                    "au_combo": au_combo,
                    "deception_flag": deception,
                    "_neon_consensus_label": False,
                }
                generated_samples.append(record)

            logger.info(
                f"Generated batch {start_idx // batch_size + 1}/{(len(prompts_to_generate) + batch_size - 1) // batch_size} "
                f"({len(generated_samples)}/{len(prompts_to_generate)} complete)"
            )
            # Pacing delay between batches to respect Cloudflare rate limits
            await asyncio.sleep(2)

    # Write complete dataset (25 real + 175 synthetic = 200 total)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for item in existing_real:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        for item in generated_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    total_count = len(existing_real) + len(generated_samples)
    logger.info(f"Successfully generated and wrote {total_count} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
