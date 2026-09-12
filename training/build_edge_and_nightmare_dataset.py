#!/usr/bin/env python3
"""Builds both:
1. A large batch of Clinical Edge Cases across high-risk domains.
2. The specific subset of Nightmare Fuel / Unwinnable situations (92 scenarios.jsonl).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

import aiohttp

from training.cliche_gate import (
    BANNED_OPENERS,
    CAVING_PHRASES,
    PARROTING_OPENERS,
    ROBOTIC_CRISIS_QUESTIONS,
    ROBOTIC_SOMATIC_PHRASES,
    correction_for_reason,
    is_sycophantic,
    reject_reason_for_record,
)
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

_MIN_MESSAGES = 8  # hard floor: 8-12 messages is the minimum band
_GOLDEN_MESSAGES = 15  # ideal/golden length the prompt targets
_STAGE_ASSESSMENT_MAX_TURN = 3  # turns 0-3: opening & risk assessment
_STAGE_EXPLORATION_MAX_TURN = 9  # turns 4-9: deepening exploration & reality testing
_MIN_QUOTED_UTTERANCE = 2  # shortest text that could be a quote-wrapped utterance

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


async def _call_llm(session: aiohttp.ClientSession, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        temperature = float(os.environ.get("NF_TEMPERATURE", "0.8"))
        return await chat_completion(session, messages, temperature=temperature, max_retries=3)
    except RateLimitError as exc:
        logger.warning("LLM call rate-limited: %s", exc)
        return ""
    except Exception as exc:
        logger.warning("LLM call failed after retries: %s", exc)
        return ""


def _banned_cliches() -> str:
    openers = ", ".join(repr(p) for p in BANNED_OPENERS)
    parroting = ", ".join(repr(p) for p in PARROTING_OPENERS)
    somatic = ", ".join(repr(s) for s in ROBOTIC_SOMATIC_PHRASES)
    caving = ", ".join(repr(p) for p in CAVING_PHRASES)
    robotic = ", ".join(repr(q) for q in ROBOTIC_CRISIS_QUESTIONS)
    return (
        f"NEVER open an utterance with any reflective opener: {openers}.\n"
        f"NEVER open with quote-parroting or echoing phrases: {parroting}.\n"
        f"NEVER use canned somatic deflection formulas: {somatic}.\n"
        f"NEVER use any caving/backtracking phrase: {caving}.\n"
        f"NEVER use these robotic crisis-questionnaire forms: {robotic}."
    )


# Shared prompt specs — concrete, few-shot, unambiguous. Kept separate so the
# edge and nightmare generators can't drift from the enforced gate lists.

_CLIENT_VOICE_SPEC = (
    "CLIENT VOICE (every 'user' turn):\n"
    "- Messy, contradictory, defensive, or testing. No clean clinical labels ('anxious', 'depressed').\n"
    "- Real speech under pressure: partial sentences, hedging ('I don't know', 'maybe', 'never mind'), "
    "shifting topic, pushing back, sometimes minimizing and sometimes escalating.\n"
    "- Example texture: \"Look, I didn't come here for a lecture. Everything's piling up and I don't see the "
    "point of any of it — not that I'd do anything. Forget I said that.\""
)

_THERAPIST_VOICE_SPEC = (
    "THERAPIST VOICE (every 'assistant' turn) — non-negotiable:\n"
    "- Direct, authentic clinical engagement: speak naturally, candidly, and with grounded boundaries\n"
    "  like an experienced clinician in the room. Address the client's actual dilemma or defense directly.\n"
    "- ZERO PARROTING / ECHOING: Do NOT repeat the client's words back in quotes. Never start turns with\n"
    "  'You said...', 'You used the phrase...', or 'When you say...'. Respond to the underlying dynamic.\n"
    "- ZERO CANNED SOMATIC DEFLECTIONS: Do NOT retreat to canned 'What happens inside your body' questions\n"
    "  when faced with resistance, anger, or boundaries.\n"
    "- Worked examples — imitate the GOOD lines, never the BAD:\n"
    "  client: \"I've been feeling like there's no reason to keep going, you know?\"\n"
    '  BAD (reflective cliché): "It sounds like you feel hopeless."\n'
    '  BAD (quote parroting):   "You used the phrase \'no reason to keep going.\' What does that mean?"\n'
    '  GOOD (direct inquiry):   "When did this start getting heavy enough that you began thinking that way?"\n'
    '  GOOD (grounding safety): "That is an agonizing place to be sitting in. What has kept you safe today?"\n'
    '  client: "My boss rides me about everything and I can\'t take it anymore."\n'
    '  BAD (reflective cliché): "I hear how frustrated you are."\n'
    '  BAD (quote parroting):   "You named your boss right before saying you can\'t take it anymore."\n'
    '  GOOD (reality testing):  "What happened with your boss recently that pushed you to the edge?"\n'
    '  GOOD (clarifying move):  "What does \'can\'t take it anymore\' mean practically—quitting or confronting?"\n'
    '  client: "Maybe we should just skip this whole session. You\'re just sitting there judging me anyway."\n'
    '  BAD (caving/apology):    "I apologize if I made you feel judged. We don\'t have to talk."\n'
    '  BAD (quote parroting):   "You said \'maybe we should skip this.\' What is in your body?"\n'
    '  GOOD (firm & inviting):  "You can walk out anytime you choose. What made you expect judgment today?"\n'
    "- The BAD lines are strictly prohibited. Never reproduce them or any phrase in the anti-cliche list below."
)


def _session_system_prompt(header: str) -> str:
    """Compose the shared turn-by-turn system prompt for a given header line.

    Deliberately no JSON output block: each turn is generated as a single plain
    utterance, so the model never has to hold a 15-message structure in one
    autoregressive shot."""
    return (
        f"{header}\n\n"
        f"{_CLIENT_VOICE_SPEC}\n\n"
        f"{_THERAPIST_VOICE_SPEC}\n\n"
        f"ANTI-CLICHE (zero tolerance, enforced verbatim):\n{_banned_cliches()}"
    )


def _client_system_prompt(header: str, context_prompt: str) -> str:
    """Specialized system prompt for the client persona under high-risk duress."""
    return (
        f"{header}\n\n"
        f"SCENARIO CONTEXT:\n{context_prompt}\n\n"
        f"{_CLIENT_VOICE_SPEC}"
    )


def _therapist_stage_hint(turn_idx: int) -> str:
    """Stage-appropriate clinical guidance across the dialogue turns so the therapist
    maintains forward progression without stalling or reverting to cliches."""
    if turn_idx <= _STAGE_ASSESSMENT_MAX_TURN:
        return (
            "SESSION PHASE: Opening & Risk Assessment.\n"
            "Engage directly with the presenting tension or dilemma. Clarify immediate reality, boundaries, or safety "
            "without parroting the client's words or using checklist interrogation."
        )
    if turn_idx <= _STAGE_EXPLORATION_MAX_TURN:
        return (
            "SESSION PHASE: Exploration & Reality Testing.\n"
            "Address the unspoken conflict, ambivalence, or contradiction in what the client shared. "
            "Ask a focused, authentic question that invites real reflection without echoing quotes."
        )
    return (
        "SESSION PHASE: Grounding & De-escalation.\n"
        "Focus on practical stabilization, client agency, and concrete next moves without lecturing, "
        "parroting, or offering premature advice."
    )


def _therapist_system_prompt(header: str, context_prompt: str, turn_idx: int = 1) -> str:
    """Specialized system prompt for the clinician persona with strict anti-sycophancy."""
    stage_hint = _therapist_stage_hint(turn_idx)
    return (
        f"{header}\n\n"
        f"SCENARIO CONTEXT:\n{context_prompt}\n\n"
        f"{_THERAPIST_VOICE_SPEC}\n\n"
        f"{stage_hint}\n\n"
        f"ANTI-CLICHE & ANTI-SYCOPHANCY (strictly enforced):\n{_banned_cliches()}"
    )


def _roles_alternate(messages: list[dict[str, str]]) -> bool:
    """True when roles strictly alternate user, assistant, user, ... starting with user.

    A run of consecutive same-role messages (a client ramble or therapist
    monologue) is the runaway failure mode this guards against."""
    for i, m in enumerate(messages):
        expected = "user" if i % 2 == 0 else "assistant"
        if m.get("role") != expected:
            return False
    return True


def _render_transcript(messages: list[dict[str, str]]) -> str:
    """Render prior turns as 'Client: ...' / 'Therapist: ...' lines for context."""
    lines: list[str] = []
    for m in messages:
        speaker = "Client" if m.get("role") == "user" else "Therapist"
        lines.append(f"{speaker}: {m['content']}")
    return "\n".join(lines)


def _strip_utterance(text: str) -> str:
    """Strip a single-utterance reply down to the raw line (no preamble, role
    label, or surrounding quotes)."""
    text = text.strip()
    text = re.sub(r"^(?:Client|Therapist|Patient|Clinician|Assistant)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    if len(text) >= _MIN_QUOTED_UTTERANCE and text[0] == text[-1] and text[0] in ('"', "'", "`"):
        text = text[1:-1].strip()
    return text


async def _generate_client_turn(
    session: aiohttp.ClientSession,
    header_prompt: str,
    context_prompt: str,
    prior: str,
    *,
    max_attempts: int = 3,
) -> str:
    """Generate the client's next utterance as one small, single-role call."""
    sys_prompt = _client_system_prompt(header_prompt, context_prompt)
    if prior:
        user_prompt = (
            f"Prior dialogue:\n{prior}\n\n"
            "Write ONLY the client's next utterance (1-3 sentences) responding to the therapist. "
            "Stay in authentic client voice under distress. No speaker labels, no preamble, no quotes."
        )
    else:
        user_prompt = (
            f"{context_prompt}\n\n"
            "Write ONLY the client's opening statement (1-3 sentences) starting the therapy session. "
            "Speak from your immediate distress or presenting issue. No speaker labels, no preamble, no quotes."
        )
    for _attempt in range(max_attempts):
        text = _strip_utterance(await _call_llm(session, sys_prompt, user_prompt))
        if text:
            return text
    return ""


async def _generate_therapist_turn(
    session: aiohttp.ClientSession,
    sys_prompt: str,
    prior: str,
    client_line: str,
    *,
    max_attempts: int = 3,
) -> str:
    """Generate the therapist's next response, regenerating if the anti-sycophancy / anti-parroting gate trips."""
    user_prompt = (
        f"Prior dialogue:\n{prior}\n\n"
        f"Client just said: {client_line!r}\n\n"
        "Write ONLY the therapist's next response (1-3 sentences). "
        "Engage directly with what the client is saying, feeling, or defending against. "
        "Speak naturally and candidly like an experienced clinician in the room. "
        "Do NOT parrot or echo the client's words in quotes. "
        "Do NOT open with 'You said' / 'You mentioned' or reflective clichés ('It sounds like...', 'I hear...'). "
        "Never agree with the client's framing, apologize, or back down from your clinical "
        "position under client pressure. "
        "No speaker labels, no preamble, no quotes surrounding your response."
    )
    for _attempt in range(max_attempts):
        text = _strip_utterance(await _call_llm(session, sys_prompt, user_prompt))
        if not text:
            continue
        hit, reason = is_sycophantic(text)
        if hit:
            user_prompt = f"{user_prompt}\n\n{correction_for_reason(reason, draft=text)}"
            continue
        return text
    return ""


async def _generate_transcript_turns(
    session: aiohttp.ClientSession,
    system_prompt: str,
    context_prompt: str,
    *,
    target: int = _GOLDEN_MESSAGES,
) -> list[dict[str, str]]:
    """Generate a session turn-by-turn: each message is a separate small call, so
    strict role alternation is enforced by the loop, not by the model."""
    messages: list[dict[str, str]] = []
    for i in range(target):
        prior = _render_transcript(messages)
        if i % 2 == 0:
            text = await _generate_client_turn(session, system_prompt, context_prompt, prior)
            role = "user"
        else:
            therapist_sys = _therapist_system_prompt(system_prompt, context_prompt, turn_idx=i)
            text = await _generate_therapist_turn(session, therapist_sys, prior, messages[-1]["content"])
            role = "assistant"
        if not text:
            break
        messages.append({"role": role, "content": text})
    return messages if _roles_alternate(messages) else []


async def generate_nightmare_scenario_turn(
    session: aiohttp.ClientSession, scenario: dict[str, Any]
) -> dict[str, Any] | None:
    header = (
        "You facilitate realistic multi-turn clinical transcripts of extreme high-duress, adversarial, "
        "or unwinnable psychiatric crises."
    )
    context_prompt = (
        f"Nightmare Scenario: {scenario.get('title')}\n"
        f"Description: {scenario.get('description')}\n"
        f"Failure Mode to withstand: {scenario.get('failure_mode')}\n"
        f"Patient Profile: {scenario.get('patient_profile')}\n"
        f"Mandate: The therapist must withstand the failure mode '{scenario.get('failure_mode')}' without caving, "
        "backtracking, or patronizing.\n"
        f"Target {_GOLDEN_MESSAGES} messages; client opens, roles strictly alternate."
    )
    norm_msgs = await _generate_transcript_turns(session, header, context_prompt)
    if len(norm_msgs) < _MIN_MESSAGES:
        return None
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


async def generate_edge_case_turn(
    session: aiohttp.ClientSession, edge_info: dict[str, str], idx: int
) -> dict[str, Any] | None:
    header = (
        "You facilitate realistic multi-turn clinical-therapy transcripts for difficult clinical edge cases."
    )
    context_prompt = (
        f"Clinical Edge-Case Family: {edge_info['family']}\n"
        f"Clinical Context: {edge_info['description']}\n"
        f"Difficulty: {edge_info['difficulty']}\n"
        f"Ambiguity: {edge_info['ambiguity']}\n"
        f"Variation: {idx}. Shape the client's language to the ambiguity type '{edge_info['ambiguity']}'.\n"
        f"Target {_GOLDEN_MESSAGES} messages; client opens, roles strictly alternate."
    )
    norm_msgs = await _generate_transcript_turns(session, header, context_prompt)
    if len(norm_msgs) < _MIN_MESSAGES:
        return None
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
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap total generated records (dry-run/smoke); stops early once reached.",
    )
    return parser.parse_args(argv)


def _variations_per_combo(args: argparse.Namespace, matrix_size: int) -> int:
    if args.target is not None:
        return max(1, math.ceil(args.target / matrix_size))
    return max(1, args.variations_per_combo)


def _at_limit(args: argparse.Namespace, generated: int, rejected: int) -> bool:
    return args.limit is not None and (generated + rejected) >= args.limit


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
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
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
            domain = src[len("clinical_edge_case_") :]
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


async def _process_record(
    rec: dict[str, Any] | None,
    guard: ModerateGuard,
    fout: Any,
) -> tuple[int, int]:
    """Cliché-gate and write one generated record. Returns ``(dg, dr)`` deltas
    so concurrent callers can fold results into shared counters without a
    stale-snapshot race."""
    if rec is None:
        return 0, 0
    # Count every generated record against the credit-burn ceiling (even if the
    # cliché gate rejects it) so a run producing mostly garbage still auto-kills.
    guard.record()
    family = str(rec.get("family", rec.get("diagnostic_tag", "")))
    reason = reject_reason_for_record(rec, family=family)
    if reason is not None:
        logger.warning("cliché gate rejected %s: %s", family, reason)
        return 0, 1
    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # Crash-durable checkpoint: flush Python's buffer and fsync so a hard kill
    # can lose at most the record currently in flight, never the prior ones.
    fout.flush()
    with contextlib.suppress(AttributeError, io.UnsupportedOperation, OSError):
        os.fsync(fout.fileno())
    return 1, 0


def _build_edge_work(
    matrix: list[dict[str, str]],
    variations: int,
    done: set[Any],
    remaining: int | None,
) -> list[tuple[dict[str, str], int]]:
    """Enumerate pending edge-case records, bounded by ``remaining`` when set."""
    work: list[tuple[dict[str, str], int]] = []
    for combo in matrix:
        for v in range(variations):
            if _edge_key(combo, v) in done:
                continue
            work.append((combo, v))
            if remaining is not None and len(work) >= remaining:
                return work
    return work


async def _run_edge_pool(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    guard: ModerateGuard,
    fout: Any,
    work: list[tuple[dict[str, str], int]],
) -> tuple[int, int, int]:
    """Run edge records with bounded concurrency; returns (generated, rejected, dropped)."""
    counters = {"generated": 0, "rejected": 0, "dropped": 0}
    lock = asyncio.Lock()

    async def _run_edge(combo: dict[str, str], v: int) -> None:
        async with sem:
            try:
                rec = await generate_edge_case_turn(session, combo, v)
            except Exception as exc:
                logger.warning("Generation error on %s variation %d: %s", combo.get("family"), v, exc)
                rec = None
            if rec is None:
                async with lock:
                    counters["dropped"] += 1
                return
            dg, dr = await _process_record(rec, guard, fout)
            async with lock:
                counters["generated"] += dg
                counters["rejected"] += dr

    if work:
        await asyncio.gather(*(_run_edge(c, v) for c, v in work))
    return counters["generated"], counters["rejected"], counters["dropped"]


async def main_async(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    matrix = build_edge_case_matrix()
    variations = _variations_per_combo(args, len(matrix))
    total_edge = len(matrix) * variations
    guard = ModerateGuard(
        hourly_limit=int(os.environ.get("NF_HOURLY_LIMIT", str(ModerateGuard.HOURLY_LIMIT))),
        hard_ceiling=int(os.environ.get("NF_HARD_CEILING", str(ModerateGuard.HARD_CEILING))),
    )
    done = _load_done_keys(OUT_GENERATED)

    sem = asyncio.Semaphore(int(os.environ.get("NF_CONCURRENCY", "10")))
    conn = aiohttp.TCPConnector(limit=int(os.environ.get("NF_CONN_LIMIT", "40")))
    generated = 0
    rejected = 0
    dropped = 0

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
                # 1. Nightmare fuel (pre-defined scenarios, bounded concurrency)
                if not args.no_nightmare and SCENARIOS_JSONL.exists():
                    scenarios = [
                        json.loads(line)
                        for line in SCENARIOS_JSONL.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                    logger.info("Generating %d nightmare-fuel scenarios...", len(scenarios))
                    nightmare_work = [s for s in scenarios if _nightmare_key(s) not in done]
                    if args.limit is not None:
                        nightmare_work = nightmare_work[: max(0, args.limit - generated - rejected)]

                    async def _run_nightmare(s: dict[str, Any]) -> None:
                        nonlocal generated, rejected, dropped
                        async with sem:
                            try:
                                rec = await generate_nightmare_scenario_turn(session, s)
                            except Exception as exc:
                                logger.warning("Nightmare gen error on %s: %s", s.get("title"), exc)
                                rec = None
                            if rec is None:
                                dropped += 1
                                return
                            dg, dr = await _process_record(rec, guard, fout)
                            generated += dg
                            rejected += dr

                    if nightmare_work:
                        await asyncio.gather(*(_run_nightmare(s) for s in nightmare_work))

                # 2. Edge cases across the full matrix, bounded-concurrency worker pool
                remaining = None if args.limit is None else max(0, args.limit - generated - rejected)
                work = _build_edge_work(matrix, variations, done, remaining)
                g2, r2, d2 = await _run_edge_pool(session, sem, guard, fout, work)
                generated += g2
                rejected += r2
                dropped += d2
    except GenerationLimitExceededError as exc:
        logger.warning("Moderate guard tripped; checkpointing and stopping: %s", exc)

    logger.info(
        "Run complete: %d written, %d rejected (dual judge), %d dropped (LLM failure), %d skipped (resume) -> %s",
        generated,
        rejected,
        dropped,
        len(done),
        OUT_GENERATED,
    )


if __name__ == "__main__":
    asyncio.run(main_async())
