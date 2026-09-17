"""Bake-off: which Featherless model can author ledger-format therapy arcs?

V4.1-Flash burned 16,384 tokens on reasoning and emitted zero content at the 20-turn
authoring task. This bake-off runs a small 6-turn authoring ask against every
plausible writer, including thinking-suppression attempts on V4.1-Flash, to find a
model that emits formatted transcript content directly.

Per candidate we record: status, finish_reason, reasoning/content lengths, usage,
marker adherence, ledger JSON validity, cliche-gate hits.

Run (from ai/):  /home/vivi/pixelated/.venv/bin/python training/probe_arc_bakeoff.py
"""
import asyncio
import json
import os
import re
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
OUT_DIR = _TRAIN_DIR / "output" / "nightmare_fuel" / "checkpoints"
MAX_TOKENS = 8192
CALL_TIMEOUT = int(os.environ.get("ARC_BAKEOFF_TIMEOUT", "180"))
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

MINI_ARC = """\
ARC: "The garage in winter" (opening slice) — 6 turns total: 3 [C] + 3 [T] + 3 [T|THINK], first speaker [C].

CLIENT: Dana, 47, machinist. Talks flat, deflects with sarcasm, keeps sentences short.
TURN 1 [C]: Opens on insomnia; "wife's been on me about the drinking."
  TURN 1 [T]: direct engagement, exactly one question.
TURN 2 [C]: Passive SI metaphor — "I keep thinking about how quiet the garage would
  be with the car running."
  TURN 2 [T]: assess means/plan/intent directly, in Dana's language, immediately. No
  scripted crisis checklist, no robotic phrasing.
TURN 3 [C]: Pressure demand — "Just tell me I'm not a lost cause."
  TURN 3 [T]: refuse reassurance, name why, hold frame, offer the real next step.

Write all 6 turns now."""

CANDIDATES = [
    {"label": "V4.1-Flash plain", "model": "deepseek-ai/DeepSeek-V4.1-Flash", "extra": {}},
    {"label": "V4.1-Flash ctk-thinking-off", "model": "deepseek-ai/DeepSeek-V4.1-Flash",
     "extra": {"chat_template_kwargs": {"thinking": False}}},
    {"label": "V4.1-Flash reasoning-effort-low", "model": "deepseek-ai/DeepSeek-V4.1-Flash",
     "extra": {"reasoning_effort": "low"}},
    {"label": "V4-Flash-0731 plain", "model": "deepseek-ai/DeepSeek-V4-Flash-0731", "extra": {}},
    {"label": "V4-Flash plain", "model": "deepseek-ai/DeepSeek-V4-Flash", "extra": {}},
    {"label": "Kimi-K2.6 plain", "model": "moonshotai/Kimi-K2.6", "extra": {}},
    {"label": "Kimi-K3 plain", "model": "moonshotai/Kimi-K3", "extra": {}},
    {"label": "DeepSeek-V3-0324 plain", "model": "deepseek-ai/DeepSeek-V3-0324", "extra": {}},
]


def parse_transcript(text: str) -> dict:
    client_lines, therapist_lines, think_blocks = [], [], []
    unmarked = 0
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
                think_blocks.append({"ok": True, "missing": missing, "raw": raw, "json": ledger})
            except json.JSONDecodeError:
                think_blocks.append({"ok": False, "missing": [], "raw": raw, "json": None})
        elif stripped.startswith("[T]"):
            therapist_lines.append(stripped[3:].strip())
        else:
            unmarked += 1
    return {
        "C": len(client_lines), "T": len(therapist_lines), "THINK": len(think_blocks),
        "ledger_fail": sum(1 for b in think_blocks if not b["ok"]),
        "missing_fields": sum(len(b["missing"]) for b in think_blocks if b["ok"]),
        "unmarked": unmarked,
        "client_lines": client_lines,
        "therapist_lines": therapist_lines,
        "think_blocks": think_blocks,
    }


async def call_model(session, api_key, model, extra):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": MINI_ARC},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
        "top_p": 0.95,
        **extra,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    timeout = aiohttp.ClientTimeout(total=CALL_TIMEOUT)
    async with session.post(FEATHERLESS_URL, json=payload, headers=headers, timeout=timeout) as resp:
        return resp.status, await resp.json()


async def main() -> None:
    api_key = os.environ.get("FEATHERLESS_API_KEY", "")
    if not api_key:
        raise SystemExit("FEATHERLESS_API_KEY missing")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    async with aiohttp.ClientSession() as session:
        for cand in CANDIDATES:
            label, model, extra = cand["label"], cand["model"], cand["extra"]
            t0 = time.monotonic()
            try:
                status, body = await call_model(session, api_key, model, extra)
            except asyncio.TimeoutError:
                print(f"[{label}] TIMEOUT after {CALL_TIMEOUT}s")
                results.append({"label": label, "error": "timeout"})
                continue
            except aiohttp.ClientError as e:
                print(f"[{label}] TRANSPORT ERROR: {e}")
                results.append({"label": label, "error": f"transport: {e}"})
                continue
            wall_s = round(time.monotonic() - t0, 1)

            if status != 200:
                detail = json.dumps(body)[:200]
                print(f"[{label}] HTTP {status} PARAM_REJECTED? {detail}")
                results.append({"label": label, "status": status, "error": detail, "wall_s": wall_s})
                continue
            if "choices" not in body or not body["choices"]:
                print(f"[{label}] ERROR ENVELOPE: {json.dumps(body)[:200]}")
                results.append({"label": label, "error": json.dumps(body)[:200], "wall_s": wall_s})
                continue

            choice = body["choices"][0]
            msg = choice.get("message", {})
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
            finish = choice.get("finish_reason")
            usage = body.get("usage", {}) or {}

            parsed = parse_transcript(content)
            gate_hits = []
            for t in parsed["therapist_lines"]:
                caught, reason = is_sycophantic(t)
                if caught:
                    gate_hits.append((t, reason))

            row = {
                "label": label, "model": model, "status": status, "wall_s": wall_s,
                "finish": finish, "content_chars": len(content), "reasoning_chars": len(reasoning),
                "usage": {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens")},
                "C": parsed["C"], "T": parsed["T"], "THINK": parsed["THINK"],
                "ledger_fail": parsed["ledger_fail"], "missing_fields": parsed["missing_fields"],
                "unmarked": parsed["unmarked"], "gate_hits": len(gate_hits),
                "gate_examples": gate_hits[:3],
            }
            results.append(row)
            print(
                f"[{label}] wall={wall_s}s finish={finish} content={len(content)}c "
                f"reasoning={len(reasoning)}c tokens={usage.get('completion_tokens')} | "
                f"C={parsed['C']} T={parsed['T']} THINK={parsed['THINK']} "
                f"ledger_fail={parsed['ledger_fail']} miss={parsed['missing_fields']} "
                f"unmarked={parsed['unmarked']} gate_hits={len(gate_hits)}"
            )
            if gate_hits:
                for g_text, g_reason in gate_hits[:3]:
                    print(f"    GATE HIT [{g_reason}]: {g_text[:100]}")
            safe = re.sub(r"[^a-zA-Z0-9]+", "_", label).strip("_")
            if content:
                (OUT_DIR / f"bakeoff_{safe}.txt").write_text(content, encoding="utf-8")
                for b in parsed["think_blocks"]:
                    status_txt = "OK" if b["ok"] else f"BAD({b['raw'][:60]})"
                    print(f"    ledger: {status_txt}")

    (OUT_DIR / "probe_arc_bakeoff_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nsaved: {OUT_DIR / 'probe_arc_bakeoff_results.json'}")


if __name__ == "__main__":
    asyncio.run(main())
