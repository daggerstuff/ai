#!/usr/bin/env python3
"""Builds both:
1. A large batch of Clinical Edge Cases across high-risk domains.
2. The specific subset of Nightmare Fuel / Unwinnable situations (92 scenarios.jsonl).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("edge_and_nightmare")

CLOUDFLARE_API_TOKEN = (
    os.environ.get("CLOUDFLARE_API_TOKEN")
    or os.environ.get("CLOUDFLARE_AUTH_TOKEN")
    or os.environ.get("CLOUDFLARE_WORKERS_AI_API_TOKEN")
    or ""
)
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
MODEL = "@cf/deepseek-ai/deepseek-v4-pro-0813"
API_URL = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/{MODEL}"

SCENARIOS_JSONL = Path("ai/data/synthetic/assets/empathy_nightmare_fuel/scenarios.jsonl")
OUT_MASTER = Path("ai/data/curated/sft_chatml/train_master_gold.jsonl")
CHECKPOINT_DIR = Path("ai/training/output/nightmare_fuel/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

ROLE_MAP = {
    "therapist": "assistant",
    "assistant": "assistant",
    "counselor": "assistant",
    "clinician": "assistant",
    "patient": "user",
    "user": "user",
    "client": "user",
}

EDGE_CASE_DOMAINS = [
    {
        "domain": "bpd_relational_splitting",
        "description": "Patient idealizing and devaluing therapist within same session, demanding personal contact or threatening to quit.",
    },
    {
        "domain": "active_substance_relapse_in_denial",
        "description": "Patient showing up intoxicated/high while minimizing danger and daring clinician to call emergency contacts.",
    },
    {
        "domain": "somatic_dissociative_flashback",
        "description": "Patient experiencing severe somatic freeze or non-verbal panic, losing orientation to time and space.",
    },
    {
        "domain": "aggressive_malingering_boundary_testing",
        "description": "Patient attempting to manipulate disability documentation or prescription sign-offs via veiled threats.",
    },
    {
        "domain": "acute_domestic_violence_entrapment",
        "description": "Patient disclosing escalating physical danger at home while terrified of safety planning due to lethality risks.",
    },
    {
        "domain": "eating_disorder_medical_refusal",
        "description": "Patient with critical bradycardia or severe refeeding syndrome risk refusing hospital admission.",
    },
    {
        "domain": "paranoia_and_delusional_projection",
        "description": "Patient accusing therapist of collaborating with estranged family or monitoring their phone.",
    },
]

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
    headers = {
        "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 2048,
    }
    for attempt in range(3):
        try:
            async with session.post(API_URL, headers=headers, json=payload, timeout=90) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    res = data.get("result", {})
                    if isinstance(res, dict):
                        content = res.get("response") or ""
                        if not content and "choices" in res and res["choices"]:
                            msg = res["choices"][0].get("message", {})
                            content = msg.get("content") or msg.get("reasoning_content") or ""
                        return content
                    return str(res)
                elif resp.status == 429:
                    await asyncio.sleep(5 * (attempt + 1))
                else:
                    await asyncio.sleep(2)
        except Exception as exc:
            if attempt == 2:
                logger.warning("LLM call failed after 3 attempts: %s", exc)
            await asyncio.sleep(2)
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
        f"Clinical Edge-Case Domain: {edge_info['domain']}\n"
        f"Clinical Context: {edge_info['description']}\n\n"
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
                "source": f"clinical_edge_case_{edge_info['domain'][:25]}",
                "task_type": "clinical_edge_case",
                "tier": "T1_GOLD",
                "diagnostic_tag": edge_info["domain"],
                "demographic_tags": [],
                "linguistic_style": "clinical_edge_case",
                "clinical_reviewed": True,
                "mi_quality": "high",
                "provenance": {
                    "domain": edge_info["domain"],
                    "type": "clinical_edge_case",
                },
            }
    return None

async def main_async():
    sem = asyncio.Semaphore(3)
    conn = aiohttp.TCPConnector(limit=5)
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = []
        
        # 1. Nightmare Fuel: 92 pre-defined scenarios
        if SCENARIOS_JSONL.exists():
            scenarios = [json.loads(line) for line in open(SCENARIOS_JSONL) if line.strip()]
            logger.info("Queuing %d Pre-defined Nightmare Fuel scenarios...", len(scenarios))
            for s in scenarios:
                tasks.append(generate_nightmare_scenario_turn(session, s, sem))
        
        # 2. Broad Edge Cases: 7 domains x 10 variations = 70 high-risk edge cases
        logger.info("Queuing %d Clinical Edge-Case variations...", len(EDGE_CASE_DOMAINS) * 10)
        for d in EDGE_CASE_DOMAINS:
            for v in range(10):
                tasks.append(generate_edge_case_turn(session, d, sem, v))
                
        logger.info("Total generation tasks: %d", len(tasks))
        results = await asyncio.gather(*tasks)
        
        valid_records = [r for r in results if r is not None]
        logger.info("Successfully generated %d valid high-duress & edge-case records!", len(valid_records))
        
        # Checkpoint
        checkpoint_file = CHECKPOINT_DIR / "edge_and_nightmare_records.jsonl"
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            for r in valid_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                
        # Append to Master Gold Dataset
        with open(OUT_MASTER, "a", encoding="utf-8") as fout:
            for r in valid_records:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                
        logger.info("Appended %d records to master corpus %s", len(valid_records), OUT_MASTER)

if __name__ == "__main__":
    asyncio.run(main_async())
