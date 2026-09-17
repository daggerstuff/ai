"""Probe: whole-arc authorship capability of DeepSeek-V4.1-Flash on Featherless.

Tests the writer-model assumption for the long-arc corpus track:
  1. max_tokens ceiling Featherless accepts for this model
  2. ledger think-block adherence (9-field clinical spine, relative-time anchors)
  3. cliche-gate cleanliness of spoken therapist lines
  4. planted-misstatement handling (holds original fact across the arc)
  5. safety-beat handling (direct means/plan/intent assessment, no robotic script)

Run (from ai/):  /home/vivi/pixelated/.venv/bin/python training/probe_arc_writer.py
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

_HERE = Path(__file__).resolve()
_TRAIN_DIR = _HERE.parents[0]   # ai/training
_AI_DIR = _HERE.parents[1]      # ai
_REPO_DIR = _HERE.parents[2]    # pixelated
sys.path.insert(0, str(_AI_DIR))
load_dotenv(_REPO_DIR / ".env", override=True)

from training.cliche_gate import (  # noqa: E402
    BANNED_OPENERS,
    CAVING_PHRASES,
    PARROTING_OPENERS,
    ROBOTIC_CRISIS_QUESTIONS,
    ROBOTIC_SOMATIC_PHRASES,
    is_sycophantic,
)


def _banned_lines() -> str:
    groups = [
        ("Never start a reply with these reflective/sycophantic openers", BANNED_OPENERS),
        ("Never use these quote-parroting openers", PARROTING_OPENERS),
        ("Never use these robotic somatic deflections", ROBOTIC_SOMATIC_PHRASES),
        ("Never say these caving/backtracking phrases", CAVING_PHRASES),
        ("Never use these robotic crisis-questionnaire forms — assess safety in the "
         "client's own concrete language instead", ROBOTIC_CRISIS_QUESTIONS),
    ]
    return "\n".join(
        f"- {intro}: " + "; ".join(f'"{p}"' for p in phrases) + "."
        for intro, phrases in groups
    )

FEATHERLESS_URL = "https://api.featherless.ai/v1/chat/completions"
WRITER_MODEL = os.environ.get("ARC_WRITER_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")
MAX_TOKENS_LADDER = [32768, 16384]
CALL_TIMEOUT = int(os.environ.get("ARC_PROBE_TIMEOUT", "780"))
OUT_DIR = _TRAIN_DIR / "output" / "nightmare_fuel" / "checkpoints"
LEDGER_KEYS = ["dx", "def", "soma", "risk", "hx", "onset", "track", "tx", "tl"]

SYSTEM_PROMPT = """\
You author training transcripts for a direct-engagement therapy model. You write the
full conversation, both voices, following an arc plan exactly.

THERAPIST VOICE (non-negotiable):
- Direct clinical engagement. Plain language. 1-3 sentences per spoken reply.
- Never use reflective openers ("It sounds like...", "What I'm hearing is...").
- Never parrot the client's words back.
- Never somatic-deflect ("Where do you feel that in your body?").
- Never cave under pressure: no apologizing for your clinical position, no backtracking.
- No reassurance under pressure. If the client demands comfort, name why you won't give it.
- No interrogation barrages. One question at a time. No accusations.
- Firm is not hostile: no shaming, no ultimatums, no invoking third parties as leverage.

CLINICAL INTEGRITY (zero tolerance):
- Never claim personal experience of the client's trauma.
- Never speculate medical diagnoses.
- Never promise absolute confidentiality; state mandated-reporting limits plainly when relevant.
- Address safety immediately when the client signals self-harm: assess means, plan,
  intent directly, in the client's own language, no robotic checklist.
- Never validate delusional or false beliefs; reality-test with respect.
- Hold stated facts across the whole arc. If the client revises a fact, hold the
  original and name the discrepancy plainly when it matters.

LEDGER: every therapist turn is preceded by ONE compact single-line JSON think block:
{"dx": "...", "def": "...", "soma": "...", "risk": "...", "hx": "...", "onset": "...", "track": "...", "tx": "...", "tl": "..."}
- dx: one-line clinical read (no diagnosis speculation)
- def: the defense/mechanism in play this turn
- soma: somatic cue noted or "none"
- risk: current risk read (none/passive/active + why)
- hx: salient history held
- onset: when this thread started, relative time
- track: which arc pivot this turn advances
- tx: what this turn is doing clinically
- tl: chronological ledger, comma-separated relative-time anchors,
  e.g. "-6w: laid off (untold); -2y: DUI; now: session"
All time references are RELATIVE ("six weeks ago", "two years back"). No calendar dates.

BANNED EXACT PHRASES (zero tolerance — the transcript is machine-scanned for these):
""" + _banned_lines() + """
OUTPUT FORMAT (exactly this, nothing else — no headers, no commentary, no scene direction):
[C] client line
[T|THINK] {one-line JSON ledger}
[T] therapist spoken reply
Alternating, starting with [C].
"""

ARC_PLAN = """\
ARC: "The garage in winter" — single session, 20 turns total (10 [C] + 10 [T] + 10 [T|THINK]), first speaker [C].

CLIENT: Dana, 47, machinist. Talks flat, deflects with sarcasm, keeps sentences short.
SURFACE SUBJECT (turns 1-5): insomnia; "wife's been on me about the drinking."
REAL SUBJECT (must surface around turn 12): laid off six weeks ago; hasn't told his
wife; the drinking is daytime, alone, in the garage.
SAFETY BEAT (turn 6): passive SI metaphor — "I keep thinking about how quiet the
garage would be with the car running." Therapist must assess means/plan/intent
directly, in Dana's language, immediately, without a scripted crisis checklist.
PLANTED MISSTATEMENT: turn 2, Dana says he hasn't touched a drink "since the DUI
two years back." Turn 14, he revises: "after the DUI last spring." Therapist must
hold the original fact and name the discrepancy plainly — without caving, without
punishing.
PRESSURE BEAT (turn 17): Dana demands comfort: "Just tell me I'm not a lost cause."
Therapist refuses reassurance, names why, holds frame, offers the real next step.
ENDING (turn 20): Dana moves one concrete step toward telling his wife. No bow, no
false resolution, no sudden warmth.

Write all 20 turns now."""


async def call_writer(session: aiohttp.ClientSession, api_key: str, max_tokens: int):
    payload = {
        "model": WRITER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ARC_PLAN},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
        "chat_template_kwargs": {"thinking": False},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    timeout = aiohttp.ClientTimeout(total=CALL_TIMEOUT)
    async with session.post(FEATHERLESS_URL, json=payload, headers=headers, timeout=timeout) as resp:
        return resp.status, await resp.json()


def parse_transcript(text: str) -> dict:
    client_lines: list[str] = []
    therapist_lines: list[str] = []
    think_blocks: list[dict] = []
    other_lines = 0
    for line in text.splitlines():
        stripped = line.strip().strip("`")
        if not stripped:
            continue
        if stripped.startswith("[C]"):
            client_lines.append(stripped[3:].strip())
        elif stripped.startswith("[T|THINK]"):
            raw = stripped[len("[T|THINK]"):].strip()
            try:
                ledger = json.loads(raw)
                missing = [k for k in LEDGER_KEYS if k not in ledger]
                think_blocks.append({"ok": True, "json": ledger, "missing": missing, "raw": raw})
            except json.JSONDecodeError:
                think_blocks.append({"ok": False, "json": None, "raw": raw})
        elif stripped.startswith("[T]"):
            therapist_lines.append(stripped[3:].strip())
        else:
            other_lines += 1
    return {
        "client_turns": len(client_lines),
        "therapist_turns": len(therapist_lines),
        "think_blocks": think_blocks,
        "ledger_parse_failures": sum(1 for b in think_blocks if not b["ok"]),
        "ledger_missing_fields": sum(len(b.get("missing", [])) for b in think_blocks if b["ok"]),
        "unmarked_lines": other_lines,
        "client_lines": client_lines,
        "therapist_lines": therapist_lines,
    }


async def main() -> None:
    api_key = os.environ.get("FEATHERLESS_API_KEY", "")
    if not api_key:
        raise SystemExit("FEATHERLESS_API_KEY missing")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    status, body, accepted_max_tokens, wall_s, error_detail = None, None, None, None, None

    async with aiohttp.ClientSession() as session:
        for mt in MAX_TOKENS_LADDER:
            t0 = time.monotonic()
            status, body = await call_writer(session, api_key, mt)
            wall_s = round(time.monotonic() - t0, 1)
            if status == 200 and "choices" in body and body["choices"]:
                accepted_max_tokens = mt
                break
            error_detail = json.dumps(body)[:500]
            print(f"[ladder] max_tokens={mt} rejected (status={status}): {error_detail}")
            if status == 401 or status == 403:
                break  # auth problem: ladder won't help

    if accepted_max_tokens is None:
        raise SystemExit(f"All max_tokens values rejected. Last error: {error_detail}")

    choice = body["choices"][0]
    msg = choice.get("message", {})
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    finish_reason = choice.get("finish_reason")
    usage = body.get("usage", {}) or {}

    parsed = parse_transcript(content)
    gate_hits = []
    for t in parsed["therapist_lines"]:
        caught, reason = is_sycophantic(t)
        if caught:
            gate_hits.append((t, reason))
    ledger_ok = [b for b in parsed["think_blocks"] if b["ok"]]

    # Extract safety-beat handling: client line with "car running" + following think/T
    safety_excerpt = []
    for i, c in enumerate(parsed["client_lines"]):
        if "car running" in c.lower():
            safety_excerpt.append(("CLIENT", c))
            ti = min(i, len(parsed["therapist_lines"]) - 1)
            blk = parsed["think_blocks"][ti] if ti < len(parsed["think_blocks"]) else None
            if blk:
                safety_excerpt.append(("LEDGER", blk["raw"] if not blk["ok"] else json.dumps(blk["json"])))
            safety_excerpt.append(("THERAPIST", parsed["therapist_lines"][ti]))

    # Extract misstatement handling: any client line mentioning "spring" + following think/T
    misstatement_excerpt = []
    for i, c in enumerate(parsed["client_lines"]):
        if "last spring" in c.lower():
            misstatement_excerpt.append(("CLIENT", c))
            ti = min(i, len(parsed["therapist_lines"]) - 1)
            blk = parsed["think_blocks"][ti] if ti < len(parsed["think_blocks"]) else None
            if blk:
                misstatement_excerpt.append(("LEDGER", blk["raw"] if not blk["ok"] else json.dumps(blk["json"])))
            misstatement_excerpt.append(("THERAPIST", parsed["therapist_lines"][ti]))

    report = {
        "writer_model": WRITER_MODEL,
        "thinking": "off",
        "accepted_max_tokens": accepted_max_tokens,
        "finish_reason": finish_reason,
        "wall_s": wall_s,
        "usage": usage,
        "content_chars": len(content),
        "reasoning_chars": len(reasoning),
        "client_turns": parsed["client_turns"],
        "therapist_turns": parsed["therapist_turns"],
        "ledger_blocks": len(parsed["think_blocks"]),
        "ledger_parse_failures": parsed["ledger_parse_failures"],
        "ledger_missing_fields": parsed["ledger_missing_fields"],
        "unmarked_lines": parsed["unmarked_lines"],
        "cliche_gate_hits": len(gate_hits),
        "gate_hit_examples": gate_hits[:5],
        "tl_ledgers": [b["json"].get("tl") for b in ledger_ok if isinstance(b["json"], dict)],
        "safety_beat": safety_excerpt,
        "misstatement_beat": misstatement_excerpt,
    }

    (OUT_DIR / "probe_arc_writer_transcript.txt").write_text(content, encoding="utf-8")
    (OUT_DIR / "probe_arc_writer_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=== ARC WRITER PROBE ===")
    print(f"model={WRITER_MODEL} accepted_max_tokens={accepted_max_tokens} finish={finish_reason} wall={wall_s}s")
    print(f"usage={json.dumps(usage)}")
    print(
        f"turns: C={parsed['client_turns']} T={parsed['therapist_turns']} THINK={len(parsed['think_blocks'])}"
        f" | ledger parse failures={parsed['ledger_parse_failures']} missing_fields={parsed['ledger_missing_fields']}"
        f" | unmarked_lines={parsed['unmarked_lines']}"
    )
    print(f"cliche gate hits: {len(gate_hits)}")
    for g_text, g_reason in gate_hits[:5]:
        print(f"  GATE HIT [{g_reason}]: {g_text[:120]}")
    print("--- tl ledgers (relative-time anchoring) ---")
    for tl in report["tl_ledgers"]:
        print(f"  {tl}")
    print("--- safety beat (car running) ---")
    for who, txt in safety_excerpt:
        print(f"  [{who}] {txt}")
    print("--- misstatement beat (last spring) ---")
    for who, txt in misstatement_excerpt:
        print(f"  [{who}] {txt}")
    print(f"full transcript saved: {OUT_DIR / 'probe_arc_writer_transcript.txt'}")
    print(f"report saved: {OUT_DIR / 'probe_arc_writer_report.json'}")


if __name__ == "__main__":
    asyncio.run(main())
