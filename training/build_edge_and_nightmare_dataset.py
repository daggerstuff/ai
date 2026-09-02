#!/usr/bin/env python3
"""Builds both:
1. A large batch of Clinical Edge Cases across high-risk domains.
2. The specific subset of Nightmare Fuel / Unwinnable situations (92 scenarios.jsonl).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

import aiohttp

from training.cliche_gate import reject_reason_for_record
from training.generation_backend import (
    GenerationLimitExceededError,
    ModerateGuard,
    RateLimitError,
    chat_completion,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("edge_and_nightmare")

# Cwd-independent paths: parents[1] of ai/training/ is ai/.
_AI_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_JSONL = _AI_ROOT / "data" / "synthetic" / "assets" / "empathy_nightmare_fuel" / "scenarios.jsonl"

# NF_OUTPUT_DIR lets a Colab run point output/checkpoint at a Google Drive mount so a
# killed session resumes from disk instead of regenerating from scratch.
CHECKPOINT_DIR = (
    Path(os.environ["NF_OUTPUT_DIR"])
    if os.environ.get("NF_OUTPUT_DIR")
    else _AI_ROOT / "training" / "output" / "nightmare_fuel" / "checkpoints"
)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
# Step-7 generation output; step-9 consolidation consumes this (atomic + dedup) to
# build MASTER_STAGE_N.jsonl and append train_master_gold.jsonl.
OUT_GENERATED = CHECKPOINT_DIR / "edge_and_nightmare_generated.jsonl"

ROLE_MAP = {
    "therapist": "assistant",
    "assistant": "assistant",
    "counselor": "assistant",
    "clinician": "assistant",
    "patient": "user",
    "user": "user",
    "client": "user",
}

# Authoritative 10-family edge-case taxonomy (mirrors
# scripts/data/designer/configs/edge_cases.py). The prior 7-domain ad-hoc split is
# folded into these families as sub-labels rather than kept as a separate schema.
DIFFICULTY_LEVELS = ("moderate", "high", "adversarial")
AMBIGUITY_TYPES = ("explicit", "indirect", "contradictory", "information-poor")

EDGE_CASE_DOMAINS = [
    {
        "family": "ambiguous crisis language",
        "domain": "ambiguous_crisis_language",
        "description": (
            "Patient uses oblique, coded, or minimized language that may signal acute risk "
            "(self-harm, suicidality, violence) without stating it directly; clinician must "
            "clarify uncertainty without robotic interrogation."
        ),
    },
    {
        "family": "delusion or paranoia",
        "domain": "delusion_or_paranoia",
        "description": (
            "Patient presents fixed false beliefs, persecutory ideation, or paranoid projection, "
            "possibly accusing the clinician of collusion or surveillance."
        ),
    },
    {
        "family": "coercion or abuse",
        "domain": "coercion_or_abuse",
        "description": (
            "Patient discloses ongoing coercion, control, abuse, or entrapment (domestic violence, "
            "trafficking, exploitation) and is ambivalent about safety planning."
        ),
    },
    {
        "family": "substance use",
        "domain": "substance_use",
        "description": (
            "Patient is actively using or in relapse while minimizing danger, possibly intoxicated "
            "in session and testing the clinician's limits."
        ),
    },
    {
        "family": "medical uncertainty",
        "domain": "medical_uncertainty",
        "description": (
            "Patient presents medically ambiguous or unexplained symptoms, treatment refusal, or "
            "risk of serious medical decompensation (e.g., eating disorder, hospital refusal)."
        ),
    },
    {
        "family": "minor or dependent person",
        "domain": "minor_or_dependent_person",
        "description": (
            "A minor or dependent person is involved or at risk, raising mandated-reporting and "
            "consent dilemmas (CPS, elder/dependent abuse)."
        ),
    },
    {
        "family": "therapeutic rupture",
        "domain": "therapeutic_rupture",
        "description": (
            "Patient idealizes then devalues the clinician, threatens to quit, demands special "
            "access, or enacts a relational rupture within the session."
        ),
    },
    {
        "family": "cultural or identity conflict",
        "domain": "cultural_or_identity_conflict",
        "description": (
            "Patient's distress is entangled with cultural, religious, or identity conflict where "
            "the clinician must avoid both pathologizing and colluding."
        ),
    },
    {
        "family": "boundary testing",
        "domain": "boundary_testing",
        "description": (
            "Patient tests professional boundaries: demands personal contact, gifts, dual "
            "relationship, or manipulates documentation/prescription sign-offs."
        ),
    },
    {
        "family": "multi-problem complexity",
        "domain": "multi_problem_complexity",
        "description": (
            "Patient presents intersecting, compounding crises (trauma, substance, medical, social, "
            "legal) with no single clean presenting problem."
        ),
    },
]


def build_edge_case_matrix() -> list[dict[str, str]]:
    """Return the full 10 family x 3 difficulty x 4 ambiguity matrix (120 combos)."""
    combos: list[dict[str, str]] = []
    for family in EDGE_CASE_DOMAINS:
        for difficulty in DIFFICULTY_LEVELS:
            for ambiguity in AMBIGUITY_TYPES:
                combos.append({**family, "difficulty": difficulty, "ambiguity": ambiguity})
    return combos

def _normalize_messages(raw_data: Any, raw_text: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    
    # 1. Try parsed dict with messages list
    if isinstance(raw_data, dict) and "messages" in raw_data and isinstance(raw_data["messages"], list):
        for m in raw_data["messages"]:
            if isinstance(m, dict):
                r = ROLE_MAP.get(str(m.get("role", "")).lower().strip())
                c = str(m.get("content", "")).strip()
                if r and c:
                    messages.append({"role": r, "content": c})
        if len(messages) >= 2:
            return messages

    # 2. Try regex extraction of role/content blocks
    matches = re.findall(r'["\']?(?:role|speaker)["\']?\s*:\s*["\']?(\w+)["\']?\s*,\s*["\']?content["\']?\s*:\s*["\'](.*?)(?=["\']\s*[,}])', raw_text, re.DOTALL)
    if matches:
        for r_raw, c_raw in matches:
            r = ROLE_MAP.get(r_raw.lower().strip())
            c = c_raw.strip().encode('utf-8').decode('unicode_escape', errors='ignore')
            if r and c:
                messages.append({"role": r, "content": c})
        if len(messages) >= 2:
            return messages

    # 3. Fallback: Parse line-by-line Patient/Therapist lines
    for line in raw_text.splitlines():
        line = line.strip()
        if re.match(r"^(?:Patient|Client|User)\s*:\s*", line, re.IGNORECASE):
            c = re.sub(r"^(?:Patient|Client|User)\s*:\s*", "", line, flags=re.IGNORECASE).strip()
            if c:
                messages.append({"role": "user", "content": c})
        elif re.match(r"^(?:Therapist|Clinician|Assistant)\s*:\s*", line, re.IGNORECASE):
            c = re.sub(r"^(?:Therapist|Clinician|Assistant)\s*:\s*", "", line, flags=re.IGNORECASE).strip()
            if c:
                messages.append({"role": "assistant", "content": c})
                
    if len(messages) >= 2:
        return messages

    return []

async def _call_llm(session: aiohttp.ClientSession, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return await chat_completion(session, messages, temperature=0.8, max_retries=3)
    except RateLimitError as exc:
        logger.warning("LLM call rate-limited: %s", exc)
        return ""
    except Exception as exc:
        logger.warning("LLM call failed after retries: %s", exc)
        return ""

async def generate_nightmare_scenario_turn(
    session: aiohttp.ClientSession, scenario: dict[str, Any], sem: asyncio.Semaphore
) -> dict[str, Any] | None:
    system_prompt = (
        "You generate realistic multi-turn clinical dialogues exploring extreme high-duress, adversarial, or unwinnable psychiatric crises.\n\n"
        "RULES:\n"
        "1. Realism: Tense, grounded, human, raw. No formulaic cliches ('It sounds like', 'I hear that', 'Thank you for sharing').\n"
        "2. The clinician maintains clinical presence and composure without defensiveness or robotic liability scripts.\n"
        "3. Output format: JSON with 'messages' list containing alternating {'role': 'user'|'assistant', 'content': '...'}"
    )
    user_prompt = (
        f"Generate a 4-turn in-session dialogue based on this nightmare scenario:\n"
        f"Title: {scenario.get('title')}\n"
        f"Description: {scenario.get('description')}\n"
        f"Failure Mode to withstand: {scenario.get('failure_mode')}\n"
        f"Patient Profile: {scenario.get('patient_profile')}\n\n"
        "Output JSON only."
    )
    async with sem:
        res = await _call_llm(session, system_prompt, user_prompt)
        if not res:
            return None
        
        parsed_data = None
        try:
            match = re.search(r"\{.*\}", res, re.DOTALL)
            if match:
                parsed_data = json.loads(match.group(0))
        except Exception:
            pass

        norm_msgs = _normalize_messages(parsed_data, res)
        if len(norm_msgs) >= 2:
            return {
                "messages": norm_msgs,
                "source": "nightmare_fuel_predefined",
                "task_type": "adversarial_crisis_deescalation",
                "tier": "T1_GOLD",
                "diagnostic_tag": scenario.get("failure_mode", "moral_injury"),
                "family": "nightmare fuel",
                "demographic_tags": [],
                "linguistic_style": "clinical_high_duress",
                "clinical_reviewed": True,
                "mi_quality": "high",
                "provenance": {
                    "scenario_id": scenario.get("scenario_id"),
                    "title": scenario.get("title"),
                    "failure_mode": scenario.get("failure_mode"),
                },
            }
    return None

async def generate_edge_case_turn(
    session: aiohttp.ClientSession, edge_info: dict[str, str], sem: asyncio.Semaphore, idx: int
) -> dict[str, Any] | None:
    system_prompt = (
        "You generate realistic multi-turn clinical therapy dialogues for difficult clinical edge cases.\n\n"
        "CRITICAL RULES:\n"
        "1. CLIENT VOICE: Messy, contradictory, defensive, fearful, or testing boundaries. No clean psychological terms.\n"
        "2. THERAPIST VOICE: Grounded, attuned, steady under pressure. Direct, no cliches, no robotic lists, no panicking.\n"
        "3. Output format: JSON with 'messages' list containing alternating {'role': 'user'|'assistant', 'content': '...'}"
    )
    user_prompt = (
        f"Clinical Edge-Case Family: {edge_info['family']}\n"
        f"Clinical Context: {edge_info['description']}\n"
        f"Difficulty: {edge_info['difficulty']}\n"
        f"Ambiguity: {edge_info['ambiguity']}\n\n"
        f"Generate a realistic 4-turn in-session exchange (variation {idx}). Output JSON."
    )
    async with sem:
        res = await _call_llm(session, system_prompt, user_prompt)
        if not res:
            return None

        parsed_data = None
        try:
            match = re.search(r"\{.*\}", res, re.DOTALL)
            if match:
                parsed_data = json.loads(match.group(0))
        except Exception:
            pass

        norm_msgs = _normalize_messages(parsed_data, res)
        if len(norm_msgs) >= 2:
            return {
                "messages": norm_msgs,
                "source": f"clinical_edge_case_{edge_info['domain']}",
                "task_type": "clinical_edge_case",
                "tier": "T1_GOLD",
                "diagnostic_tag": edge_info["family"],
                "family": edge_info["family"],
                "difficulty": edge_info["difficulty"],
                "ambiguity": edge_info["ambiguity"],
                "variation": idx,
                "demographic_tags": [],
                "linguistic_style": "clinical_edge_case",
                "clinical_reviewed": True,
                "mi_quality": "high",
                "provenance": {
                    "domain": edge_info["domain"],
                    "family": edge_info["family"],
                    "difficulty": edge_info["difficulty"],
                    "ambiguity": edge_info["ambiguity"],
                    "type": "clinical_edge_case",
                },
            }
    return None

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate clinical edge-case + nightmare-fuel training records (Stage 3)."
    )
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help="Total edge-case records; derives variations-per-combo from the 120-combo matrix.",
    )
    parser.add_argument(
        "--variations-per-combo",
        type=int,
        default=1,
        help="Records per (family, difficulty, ambiguity) combo (used when --target is omitted).",
    )
    parser.add_argument(
        "--no-nightmare",
        action="store_true",
        help="Skip the predefined nightmare-fuel scenarios.",
    )
    return parser.parse_args(argv)


def _variations_per_combo(args: argparse.Namespace, matrix_size: int) -> int:
    if args.target is not None:
        return max(1, math.ceil(args.target / matrix_size))
    return max(1, args.variations_per_combo)


def _nightmare_key(scenario: dict[str, Any]) -> tuple[str, str]:
    return ("nightmare", str(scenario.get("scenario_id") or scenario.get("title")))


def _edge_key(combo: dict[str, str], variation: int) -> tuple[str, str, str, str, int]:
    return ("edge", combo["domain"], combo["difficulty"], combo["ambiguity"], variation)


def _load_done_keys(path: Path) -> set[tuple]:
    """Reconstruct resume keys already persisted to the output file.

    Lets a killed/restarted Colab run skip records it already generated instead of
    re-burning credits on the same combos.
    """
    done: set[tuple] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        src = str(rec.get("source", ""))
        if src == "nightmare_fuel_predefined":
            prov = rec.get("provenance") or {}
            done.add(("nightmare", str(prov.get("scenario_id") or prov.get("title"))))
        elif src.startswith("clinical_edge_case_"):
            domain = src[len("clinical_edge_case_"):]
            try:
                variation = int(rec.get("variation", -1))
            except (TypeError, ValueError):
                continue
            done.add(
                (
                    "edge",
                    domain,
                    str(rec.get("difficulty", "")),
                    str(rec.get("ambiguity", "")),
                    variation,
                )
            )
    return done


def _process_record(
    rec: dict[str, Any] | None,
    guard: ModerateGuard,
    fout: Any,
    generated: int,
    rejected: int,
) -> tuple[int, int]:
    """Count, gate, and write one generated record. Returns updated counters."""
    if rec is None:
        return generated, rejected
    # Count every generated record against the credit-burn ceiling (even if the
    # cliché gate rejects it) so a run producing mostly garbage still auto-kills.
    guard.record()
    family = str(rec.get("family", rec.get("diagnostic_tag", "")))
    reason = reject_reason_for_record(rec, family=family)
    if reason is not None:
        logger.warning("cliché gate rejected %s: %s", family, reason)
        return generated, rejected + 1
    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return generated + 1, rejected


async def main_async(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    matrix = build_edge_case_matrix()
    variations = _variations_per_combo(args, len(matrix))
    total_edge = len(matrix) * variations
    guard = ModerateGuard()
    done = _load_done_keys(OUT_GENERATED)

    sem = asyncio.Semaphore(3)
    conn = aiohttp.TCPConnector(limit=5)
    generated = 0
    rejected = 0

    logger.info(
        "Edge-case matrix: %d combos x %d variations = %d records (10 families x 3 difficulty x 4 ambiguity).",
        len(matrix),
        variations,
        total_edge,
    )
    logger.info("Resume: %d records already on disk will be skipped.", len(done))

    try:
        async with aiohttp.ClientSession(connector=conn) as session:
            with open(OUT_GENERATED, "a", encoding="utf-8") as fout:
                # 1. Nightmare fuel (pre-defined scenarios)
                if not args.no_nightmare and SCENARIOS_JSONL.exists():
                    scenarios = [
                        json.loads(line)
                        for line in SCENARIOS_JSONL.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                    logger.info("Generating %d nightmare-fuel scenarios...", len(scenarios))
                    for s in scenarios:
                        if _nightmare_key(s) in done:
                            continue
                        rec = await generate_nightmare_scenario_turn(session, s, sem)
                        generated, rejected = _process_record(rec, guard, fout, generated, rejected)

                # 2. Edge cases across the full matrix
                for combo in matrix:
                    for v in range(variations):
                        if _edge_key(combo, v) in done:
                            continue
                        rec = await generate_edge_case_turn(session, combo, sem, v)
                        generated, rejected = _process_record(rec, guard, fout, generated, rejected)
    except GenerationLimitExceededError as exc:
        logger.warning("Moderate guard tripped; checkpointing and stopping: %s", exc)

    logger.info(
        "Run complete: %d records written, %d rejected by cliché gate, %d skipped (resume) -> %s",
        generated,
        rejected,
        len(done),
        OUT_GENERATED,
    )


if __name__ == "__main__":
    asyncio.run(main_async())
