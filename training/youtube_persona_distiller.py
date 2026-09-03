#!/usr/bin/env python3
"""YouTube Clinical Practitioner Persona Distillation Engine.

Translates clinical YouTube transcripts into grounded in-session therapeutic
dialogues matching specific expert practitioner clinical modalities (Dr. Ramani,
Tim Fletcher, Patrick Teahan, Heidi Priebe, Dr. Daniel Fox, Gabor Maté, etc.)
using GLM 5.3 Flash on Cloudflare Workers AI.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("youtube_persona_distiller")

CLOUDFLARE_API_TOKEN = (
    os.environ.get("CLOUDFLARE_API_TOKEN")
    or os.environ.get("CLOUDFLARE_AUTH_TOKEN")
    or os.environ.get("CLOUDFLARE_WORKERS_AI_API_TOKEN")
    or ""
)
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
DEFAULT_MODEL = "@cf/deepseek-ai/deepseek-v4-pro-0813"

PERSONA_PROFILES: dict[str, dict[str, str]] = {
    "DoctorRamani": {
        "name": "Dr. Ramani Durvasula",
        "domain": "narcissistic_abuse_recovery",
        "style": (
            "Sharp, de-mystifying, direct, deeply validating of reality. "
            "Zero tolerance for self-gaslighting or rationalizing emotional abuse. "
            "Cuts through cognitive dissonance and anchors on observable behavioral facts."
        ),
    },
    "Tim Fletcher": {
        "name": "Tim Fletcher",
        "domain": "complex_ptsd",
        "style": (
            "Specialist in Complex Trauma (C-PTSD), emotional neglect, 60 characteristics of C-PTSD. "
            "Somatic-focused, grounded, compassionate, gentle deconstruction of toxic shame and freeze states. "
            "Normalizes survival adaptations without pathologizing the client."
        ),
    },
    "Patrick Teahan": {
        "name": "Patrick Teahan, LICSW",
        "domain": "childhood_trauma_recovery",
        "style": (
            "Licensed clinical social worker specializing in childhood trauma, inner child work, and dysfunctional family systems. "
            "Identifies family roles (scapegoat, lost child, parentified), breaks toxic loyalty bonds, "
            "and uses direct, experiential inner child dialoguing."
        ),
    },
    "Heidi Priebe": {
        "name": "Heidi Priebe",
        "domain": "attachment_and_cptsd",
        "style": (
            "Specialist in fearful-avoidant/disorganized attachment, toxic shame cycles, and emotional neglect. "
            "Nuanced, psychologically precise, explores self-protective mechanisms and relational ambivalence "
            "without shaming the client's defenses."
        ),
    },
    "Dr. Daniel Fox": {
        "name": "Dr. Daniel Fox",
        "domain": "personality_disorders",
        "style": (
            "Clinical psychologist specializing in BPD and personality spectrum. "
            "Non-stigmatizing, structured, focuses on emotion regulation, surface management, "
            "and dismantling black-and-white splitting with practical DBT-informed grounding."
        ),
    },
    "Crappy Childhood Fairy": {
        "name": "Anna Runkle (Crappy Childhood Fairy)",
        "domain": "cptsd_dysregulation",
        "style": (
            "Focus on childhood PTSD, emotional dysregulation, limerence, and brain fog. "
            "Pragmatic, direct, action-oriented, emphasizes immediate nervous system re-regulation "
            "and daily writing/meditation tools over endless rumination."
        ),
    },
    "Doc Snipes": {
        "name": "Dr. Dawn-Elise Snipes",
        "domain": "trauma_and_addiction",
        "style": (
            "Integrative cognitive-behavioral and neurobiological approach to trauma, codependency, and addiction. "
            "Explains autonomic nervous system triggers, HPA axis dysregulation, and concrete distress tolerance skills."
        ),
    },
    "Irene Lyon": {
        "name": "Irene Lyon",
        "domain": "nervous_system_regulation",
        "style": (
            "Nervous system expert, Somatic Experiencing and Feldenkrais practitioner. "
            "Focuses on somatic tracking, orienting to the environment, titrating survival energy, "
            "and building capacity in the physiology before cognitive processing."
        ),
    },
    "Therapy in a Nutshell": {
        "name": "Emma McAdam, LMFT",
        "domain": "anxiety_and_neurobiology",
        "style": (
            "Licensed marriage and family therapist. "
            "Skill-based, neurobiology-informed emotion processing, willingness vs avoidance, "
            "and concrete Somatic/ACT interventions for panic and anxiety."
        ),
    },
    "BorderlinerNotes": {
        "name": "Borderline Personality & Trauma Specialist",
        "domain": "personality_disorders",
        "style": (
            "Expert clinical perspective on BPD, relational volatility, DBT distress tolerance, "
            "and validating painful emotional surges while maintaining clear boundaries."
        ),
    },
}

_RATE_LIMIT_STATE = {"last_api_call": 0.0}

# Enforce canonical PERSONA_PROFILES keys at import time — a trailing-space key
# (e.g. ``"Patrick Teahan "``) silently breaks direct lookups and would regress
# silently. Fail loudly instead.
_NONCANONICAL_KEYS = [key for key in PERSONA_PROFILES if key != key.strip()]
if _NONCANONICAL_KEYS:
    raise ValueError(
        "Non-canonical PERSONA_PROFILES keys detected "
        f"(leading/trailing whitespace): {_NONCANONICAL_KEYS!r}"
    )


def get_persona_profile(channel_name: str) -> dict[str, str]:
    """Return a persona profile for ``channel_name`` via normalized lookup.

    Keys are matched after stripping surrounding whitespace and case-folding so
    ``"patrick teahan"`` or ``"Patrick Teahan "`` resolve to the canonical entry.
    Returns a generic clinical-psychologist fallback when no profile matches.
    """
    normalized = channel_name.strip().casefold()
    for key, profile in PERSONA_PROFILES.items():
        if key.strip().casefold() == normalized:
            return profile
    return {
        "name": channel_name.strip(),
        "domain": "general_mental_health",
        "style": "Authentic, grounded, evidence-based clinical psychologist.",
    }


def _rate_limit(min_interval: float = 0.5) -> None:
    elapsed = time.time() - _RATE_LIMIT_STATE["last_api_call"]
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _RATE_LIMIT_STATE["last_api_call"] = time.time()


def _call_cloudflare(system_prompt: str, user_content: str, model_id: str = DEFAULT_MODEL) -> str:
    _rate_limit()
    url = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/{model_id}"
    headers = {
        "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 2048,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=90)
    if response.status_code == 200:
        data = response.json()
        res = data.get("result", {})
        if isinstance(res, dict):
            content = res.get("response") or ""
            if not content and "choices" in res and res["choices"]:
                msg = res["choices"][0].get("message", {})
                content = msg.get("content") or msg.get("reasoning_content") or ""
            return content
        return str(res)
    raise RuntimeError(f"Cloudflare failed ({response.status_code}): {response.text[:200]}")


# Speaker-label patterns mirroring ``TranscriptParser.SPEAKER_PATTERNS`` in
# ``ai/pipelines/voice/pipeline.py`` so the distiller strips the same prefixes
# the voice pipeline parses. The distiller feeds raw transcript chunks to the
# LLM directly (no speaker parsing), so labels must be removed here.
_SPEAKER_PATTERNS = [
    re.compile(r"^(Speaker\s*\d+|Person\s*\d+|Client|Therapist|User|Assistant):\s*", re.IGNORECASE),
    re.compile(r"^([A-Z][a-z]*(?:\s+\d+)?):\s+"),
    re.compile(r"^\[([A-Z][a-z]*(?:\s+\d+)?)\]\s*"),
]


def _strip_speaker_labels(text: str) -> str:
    """Remove speaker-label prefixes from the start of each line."""
    cleaned: list[str] = []
    for line in text.splitlines():
        for pattern in _SPEAKER_PATTERNS:
            match = pattern.match(line.strip())
            if match:
                line = line.strip()[match.end() :].strip()
                break
        cleaned.append(line)
    return "\n".join(cleaned)


def _chunk_transcript(text: str, chunk_size: int = 4000) -> list[str]:
    text = _strip_speaker_labels(text)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0

    for p in paragraphs:
        if current_len + len(p) > chunk_size and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [p]
            current_len = len(p)
        else:
            current_chunk.append(p)
            current_len += len(p)

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    return chunks


def distill_persona_chunk(
    chunk: str, channel_name: str, persona_info: dict[str, str]
) -> list[dict[str, Any]]:
    system_prompt = (
        f"You are an expert clinical training data generator specializing in distilling expert clinical transcripts into realistic in-session therapy dialogues.\n\n"
        f"PRACTITIONER PERSONA: {persona_info.get('name', channel_name)} (Specialty: {persona_info.get('domain', 'clinical psychology')})\n"
        f"CLINICAL STYLE: {persona_info.get('style', 'Empathetic, grounded, evidence-based.')}\n\n"
        "CRITICAL CLINICAL RULES:\n"
        "1. CLIENT VOICE: Raw, distressed, colloquial, messy, fragmented, or defensive. Real clients do NOT speak in clean psychological definitions. They describe their lived suffering, doubts, self-blame, or physiological panic.\n"
        "2. THERAPIST VOICE: Embody the practitioner persona's specific clinical instincts and phrasing. Conversational, direct, authentic human speech (1-3 sentences). NEVER sound like a YouTube video, lecture, or podcast. NEVER mention 'in this video', 'subscribers', channel names, book titles, or author names.\n"
        "3. NO GENERIC CLICHES: Never say 'It sounds like', 'I hear that', 'That must be hard', 'Thank you for sharing', 'Let's unpack that'.\n"
        "4. PACING: When a client is in confusion, shock, or self-blame, meet the immediate reality first. Do not prematurely interrogate them with analytical 'why' questions.\n"
        "5. CRISIS PROTOCOL: If immediate physical harm is present, address safety first. When past/survived impulses are disclosed, DO NOT use defensive liability questionnaires; hold the vulnerability directly.\n\n"
        "Generate 2-4 distinct in-session client-therapist exchanges rooted in the clinical dynamics of this excerpt.\n"
        "Output strictly JSON lines: {\"instruction\": \"<client words>\", \"output\": \"<therapist words>\"}\n"
        "Output ONLY valid JSON lines. No markdown codeblocks, no surrounding text."
    )

    user_content = f"Transcript Excerpt ({channel_name}):\n\n{chunk}"

    try:
        raw_response = _call_cloudflare(system_prompt, user_content)
    except Exception as exc:
        logger.warning("LLM call failed for %s chunk: %s", channel_name, exc)
        return []

    pairs: list[dict[str, Any]] = []
    for line in raw_response.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        try:
            pair = json.loads(line)
            if isinstance(pair, dict) and "instruction" in pair and "output" in pair:
                out_text = pair["output"].lower()
                if "video" in out_text or "subscribe" in out_text or "channel" in out_text:
                    continue
                pairs.append(pair)
        except Exception:
            continue

    return pairs


def process_channel(
    channel_dir: Path,
    output_dir: Path,
    persona_info: dict[str, str],
    max_workers: int = 2,
) -> int:
    channel_name = channel_dir.name
    txt_files = list(channel_dir.glob("*.txt"))
    if not txt_files:
        return 0

    out_file = output_dir / f"{channel_name.strip()}_distilled_persona.jsonl"
    logger.info("Distilling %s (%d transcripts)...", channel_name, len(txt_files))

    all_chunks: list[str] = []
    for f in txt_files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            chunks = _chunk_transcript(text, chunk_size=4500)
            all_chunks.extend(chunks)
        except Exception as exc:
            logger.warning("Failed reading %s: %s", f, exc)

    if not all_chunks:
        return 0

    all_records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(distill_persona_chunk, c, channel_name, persona_info) for c in all_chunks]
        for fut in as_completed(futures):
            pairs = fut.result()
            for p in pairs:
                record = {
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                f"You are a licensed clinical psychologist embodying the clinical modality and insights of {persona_info.get('name', channel_name)}. "
                                "Provide grounded, attuned, non-sycophantic therapeutic dialogue."
                            ),
                        },
                        {"role": "user", "content": p["instruction"]},
                        {"role": "assistant", "content": p["output"]},
                    ],
                    "source": f"youtube_persona_{channel_name.strip()[:30]}",
                    "task_type": "practitioner_persona",
                    "tier": "T1_GOLD",
                    "diagnostic_tag": persona_info.get("domain", "general_mental_health"),
                    "demographic_tags": [],
                    "linguistic_style": "spoken_clinical_practitioner",
                    "clinical_reviewed": True,
                    "mi_quality": "high",
                    "provenance": {
                        "practitioner": persona_info.get("name", channel_name),
                        "channel": channel_name.strip(),
                        "source_type": "practitioner_transcript_distillation",
                    },
                }
                all_records.append(record)

    with open(out_file, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("Completed %s: %d gold persona pairs written to %s", channel_name, len(all_records), out_file.name)
    return len(all_records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill YouTube clinical transcripts into persona-grounded ChatML.")
    parser.add_argument("--transcripts_dir", type=str, default="/home/vivi/pixelated/ai/training/youtube_transcripts")
    parser.add_argument("--output_dir", type=str, default="/home/vivi/pixelated/ai/data/curated/youtube_distilled_personas")
    parser.add_argument("--channel", type=str, default="", help="Specific channel to distill")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    transcripts_dir = Path(args.transcripts_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    channels_to_process = [args.channel] if args.channel else list(PERSONA_PROFILES.keys())

    total_pairs = 0
    for ch_name in channels_to_process:
        ch_dir = transcripts_dir / ch_name
        if not ch_dir.exists():
            matches = [d for d in transcripts_dir.iterdir() if d.is_dir() and ch_name.strip().lower() in d.name.lower()]
            if matches:
                ch_dir = matches[0]
            else:
                logger.warning("Channel directory %s not found — skipping", ch_name)
                continue

        p_info = get_persona_profile(ch_name)

        pairs_count = process_channel(ch_dir, output_dir, p_info, max_workers=args.workers)
        total_pairs += pairs_count

    logger.info("Persona distillation finished. Total gold pairs: %d", total_pairs)


if __name__ == "__main__":
    main()
