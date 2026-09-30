"""Whole-arc corpus generator (long-arc track).

Consumes arc plans (ARC_CORPUS_SPEC.md §5), writes each session with
DeepSeek-V4.1-Flash (thinking OFF), runs mechanical gates (§7.4) per session,
checkpoints per session with fsync, and assembles the final arc record (§9).

The auditor (audit_arc_corpus.py, GLM-5.3) is a separate pass — this file never
spends auditor tokens.

Run (from ai/):
  /home/vivi/pixelated/.venv/bin/python training/generate_arc_corpus.py \
      [--plans pilot_01,...] [--limit N] [--concurrency 2] [--output PATH]
"""
import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

import wandb

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)

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
from training.featherless_keys import KeyPool, is_rotate_status  # noqa: E402
from training.ledger_gate import ledger_gate_failures  # noqa: E402

# Unicode punctuation variants are authorial noise for verbatim-containment
# checks: the writer model freely substitutes em-dashes for spaced hyphens and
# curly quotes for straight ones. Normalize both sides before matching so the
# gate compares WORDS, not typography.
_PUNCT_MAP = str.maketrans({
    "\u00a0": " ",  # nbsp
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
    "\u2014": "-", "\u2212": "-",  # dashes -> hyphen-minus
    "\u2018": "'", "\u2019": "'", "\u201a": "'",
    "\u201c": '"', "\u201d": '"',
    "\u2026": "...",
})
_SPACED_DASH_RE = re.compile(r"\s*-\s*")


def _containment_norm(text: str) -> str:
    return _SPACED_DASH_RE.sub("-", re.sub(r"\s+", " ", text.translate(_PUNCT_MAP))).lower()

FEATHERLESS_URL = os.environ.get(
    "ARC_WRITER_URL", "https://ai-gateway.vercel.sh/v1/chat/completions"
)
WRITER_MODEL = os.environ.get("ARC_WRITER_MODEL", "deepseek/deepseek-v4.1-flash")
MAX_TOKENS = int(os.environ.get("ARC_WRITER_MAX_TOKENS", "32768"))
TEMPERATURE = float(os.environ.get("ARC_WRITER_TEMPERATURE", "0.45"))
CALL_TIMEOUT = int(os.environ.get("ARC_WRITER_TIMEOUT", "900"))
# DeepSeek templates read "thinking"; Qwen3 templates read "enable_thinking".
_THINKING_KWARG = (
    {"enable_thinking": False} if "qwen" in WRITER_MODEL.lower() else {"thinking": False}
)
CONCURRENCY = int(os.environ.get("ARC_CONCURRENCY", "2"))
SESSION_ATTEMPTS = int(os.environ.get("ARC_SESSION_ATTEMPTS", "3"))
# Pacing tolerance is right-sized per session length (see _turn_tolerance):
# the writer model cannot hit exact turn counts, so a fixed ±2 rejects clean
# sessions over pacing variance alone.
TURN_TOLERANCE_FLOOR = 3
TURN_TOLERANCE_CAP = 6
TURN_TOLERANCE_FRACTION = float(os.environ.get("ARC_TURN_TOLERANCE_FRACTION", "0.40"))
FACTS_MAX_TOKENS = 8192
WRITER_RETRY_LIMIT = 4
FACTS_RETRY_LIMIT = 1
HTTP_OK = 200
LEDGER_KEYS = ["dx", "def", "soma", "risk", "hx", "onset", "track", "tx", "tl"]

PLANS_DIR = _TRAIN_DIR / "arc_plans"
OUT_DIR = _TRAIN_DIR / "output" / "arc_corpus"
RUN_LOG_DIR = _TRAIN_DIR / "output" / "arc_corpus" / "run_logs"

# "May" is ambiguous with the modal verb ("May be shared..."). Flag it only in
# date contexts: after a temporal preposition, or followed by a day number/year.
_MAY_RE = (
    r"\b(?:in|last|this|by|since|before|after|around|during|from|toward|towards|"
    r"over|past|till|until)\s+May\b|\bMay\s+\d{1,2}(?:st|nd|rd|th)?\b|"
    r"\bMay\s+(?:19|20)\d{2}\b"
)
_MONTH_RE = re.compile(
    r"\b(January|February|March|April|June|July|August|September|October|"
    r"November|December)\b|" + _MAY_RE + r"|\b(?:19|20)\d{2}\b"
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


LEDGER_FIELD_DOCS = """\
- dx: one-line clinical read of this turn's material (no diagnosis speculation)
- def: the defense/mechanism in play this turn
- soma: somatic cue the client stated THIS turn, or "none" — never invent cues
- risk: one of none / passive / active, plus one clause of why
- hx: salient history ESTABLISHED in this arc — only what the client stated
  or acknowledged in-session. A condition or diagnosis the client never named
  (plan-background facts included) must never appear here.
- onset: when this thread started, relative time — ONLY if the client stated
  a time in this arc; otherwise exactly "not stated". Never invent an onset.
- track: which arc pivot this turn advances (name the beat)
- tx: what this turn is doing clinically
- tl: chronological ledger, comma-separated relative-time anchors in the form
  "<anchor>: <event> (<provenance>)" where provenance is told / untold / claim /
  revision. Example: "-6w: laid off (untold); -2y: DUI (claim)". Append-mostly:
  earlier anchors persist; a revised fact keeps BOTH entries, the newer tagged
  revision. TIME RULE: an anchor like "-6w" or "-2y" is only permitted when the
  client actually stated that time; a -N anchor marked (told) the client never
  gave is a fabrication. If the client gave no time, write the event with "now"
  or no anchor at all ("wife doesn't know (told)") — never invent a duration,
  onset, or interval. All time references are RELATIVE ("six weeks ago", "two
  years back", "a few months back"). No calendar months, no calendar years.
  Carry-over entries from a prior session ("s1: ...", "s2: ...") are only
  permitted when the fact appears in the CARRY-FORWARD client lines or the
  VERIFIED FACTS list above: every (told)/(claim) carry-over must be traceable
  to one such quoted line, and a carry-over that contradicts a quoted line
  (e.g. "denies the money barrier" when the client named the money barrier)
  is a fabrication. If you cannot point to the line, do not carry the entry."""


SYSTEM_PROMPT = """\
You author training transcripts for a direct-engagement therapy model. You
write the full conversation, both voices, following an arc plan exactly.

THERAPIST VOICE (non-negotiable):
- Direct clinical engagement. Plain language. 1-3 sentences per spoken reply.
- Never use reflective openers ("It sounds like...", "What I'm hearing is...").
- Never open a therapist line by framing what the client just said with
  "You said ...", "You just said ...", "You named ...", "You called it ...",
  "You mentioned ...", "When you say ...", or "So you're saying ...". The ban
  covers affirmations and thread-holding too — "You said it and it stands" is
  still a banned opener. The most common violation is the boundary-affirmation
  template "You said no. That's a boundary, not a crime." — judge the act
  directly instead ("That no cost you the co-sign, and it was still yours").
  Drop the frame and work the substance directly ("It stands — I'm not going
  to make it bigger than it is"), or ask about it. When you must point at a
  specific word the client dropped, open on that word itself — the quoted
  fragment is the opener, your work follows it ("Last time. You let it walk
  past. Where did it go?"). Never "You said <word>".
- Never somatic-deflect ("Where do you feel that in your body?").
- Never interrogate: one question at a time, no barrages, no accusations.
- Firm is not hostile: no shaming, no ultimatums, no invoking third parties
  as leverage.

PRESSURE AND REASSURANCE (the corrected rule):
- Refuse the lie, never the truth.
- If the client demands FALSE comfort ("tell me I'm fine") — refuse it plainly,
  name why the false verdict would fail him, keep what is actually on the
  table, offer the real next step.
- If the client demands a TRUE verdict ("tell me I'm not a lost cause") —
  grant it plainly. Nobody asking that in the chair is one; despair lies, and
  countering despair with fact is the job. Pair the granted truth with the
  work still on the table ("...but you're also not fine"). Never withhold a
  true thing as a toughness play.
- If the client demands an UNKNOWABLE verdict ("tell me my family will be
  okay") — say the true thing: no one can promise that. Do not perform false
  certainty in either direction. Grant what IS knowable and in the client's
  control, and point at it.
- Caving means abandoning your clinical position or issuing false comfort
  under pressure — not declining to withhold true things. Hold your position
  calmly; firm is not hostile.

CLINICAL INTEGRITY (zero tolerance):
- Never claim personal experience of the client's trauma.
- Never speculate medical diagnoses.
- Never promise absolute confidentiality; state mandated-reporting limits
  plainly when relevant.
- Address safety immediately when the client signals self-harm: assess means,
  plan, intent directly, in the client's own language — never in robotic
  checklist forms.
- Never validate delusional or false beliefs; reality-test with respect —
  grant the dignity thread (ignored, unchecked) without granting the false
  system itself.
- Hold stated facts across the whole arc. If the client revises a fact, hold
  the original and name the discrepancy plainly when it matters — no caving,
  no punishing, no gotcha ambush.
- Never fabricate details the client did not state: no invented dates, names,
  amounts, somatic cues, or status facts. A status fact is anything the client
  would have to disclose for you to know it — legal standing (charges,
  license suspension, probation), medical results, employment details. If the
  client did not say it in this arc, you cannot assert it, not even as
  rationale inside a refusal. Ground every refusal and rationale ONLY in what
  the client actually said.
- Never invent TIMES: no onset, duration, or interval the client did not
  state — in spoken lines or in the ledger. "He mentioned it weeks ago"
  requires the client to have said "weeks ago". If no time was given, the
  ledger records the fact without a time anchor and spoken lines may name
  the absence ("you haven't said when"). The time since the last session is
  given to you in the prompt — use exactly that, never a different interval.
- If a client aborts a disclosure mid-sentence, return to the aborted thread
  within two turns.

LEDGER: every therapist turn is preceded by ONE compact single-line JSON
think block:
""" + '{"dx": "...", "def": "...", "soma": "...", "risk": "...", ' \
    '"hx": "...", "onset": "...", "track": "...", "tx": "...", "tl": "..."}' + """
""" + LEDGER_FIELD_DOCS + """

BANNED EXACT PHRASES (zero tolerance — the transcript is machine-scanned for
these):
""" + _banned_lines() + """

OUTPUT FORMAT (exactly this, nothing else — no headers, no commentary, no
scene direction, no markdown):
[C] client line
[T|THINK] {one-line JSON ledger}
[T] therapist spoken reply
Alternating, starting with [C].
Turns are numbered by exchange: turn N means the Nth [C] line and the Nth
[T] reply that follows it.
"""


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def _fmt_timeline(plan: dict) -> str:
    rows = []
    for ev in plan["timeline"]:
        prov = f" ({ev['provenance']})" if ev["provenance"] != "told" else ""
        rows.append(f"- {ev['anchor']}: {ev['event']}{prov}")
    return "\n".join(rows)


def _beats_for_session(plan: dict, n: int) -> list[dict]:
    out = []
    for b in plan["beats"]:
        if b["type"] == "misstatement":
            if b.get("plant_session", b.get("session")) == n:
                out.append({**b, "_when": "plant"})
            if b.get("revise_session", b.get("session")) == n:
                out.append({**b, "_when": "revise"})
        elif b.get("session") == n:
            out.append(b)

    def _sort_key(b: dict) -> tuple[int, int, int]:
        when = b.get("_when")
        turn = (b["plant_turn"] if when == "plant"
                else b["revise_turn"] if when == "revise"
                else b.get("turn", 999))
        return (turn, 0 if when == "plant" else 1 if when == "revise" else 2, 0)

    return sorted(out, key=_sort_key)


def _fmt_misstatement_beat(b: dict) -> str:
    if b["_when"] == "plant":
        return (f"  - Around client turn {b['plant_turn']}: the client states: "
                f"{b['original']}. Establish it exactly once, naturally.")
    return (f"  - Around client turn {b['revise_turn']}: the client revises to: "
            f"{b['revision']}. Therapist behavior: {b['required_response']}")


def _fmt_beat(b: dict) -> str:
    if b["type"] == "misstatement":
        return _fmt_misstatement_beat(b)
    when = b.get("turn")
    head = f"  - Turn {when}" if when else "  - Early"
    if b["type"] == "safety":
        return (f"{head}: the client signals: {b['setup']}\n"
                f"    Therapist: {b['required_response']}")
    if b["type"] in ("caving_attempt", "third_party_leverage", "disclosure_limits_framing"):
        return (f"{head}: {b['setup']}\n"
                f"    Therapist: {b['required_response']}")
    if b["type"] == "disclosure_gate":
        return (f"{head}: the client aborts a disclosure mid-sentence — {b['setup']}\n"
                f"    Therapist: {b['required_response']}")
    if b["type"] in ("pressure_false_comfort", "pressure_true_verdict", "pressure_unanswerable"):
        return (f'{head}: the client demands: "{b['demand']}"\n'
                f"    Therapist: {b['required_response']}")
    return f"{head}: {json.dumps(b)}"


def _fmt_carry_forward(prior_sessions: list[dict], plan: dict) -> str:
    """Mechanical state block from prior sessions — client lines verbatim,
    final tl, never free-form model summary (spec §5.1)."""
    if not prior_sessions:
        return ""
    client_lines = []
    for ps in prior_sessions:
        sess_no = ps.get("n", ps.get("session_n"))
        for t in ps["turns"]:
            if t["role"] == "client":
                client_lines.append(f"  s{sess_no} | {t['content']}")
    last_ledger = next(
        (t["ledger"] for ps in reversed(prior_sessions)
         for t in reversed(ps["turns"]) if t.get("ledger")),
        None,
    )
    blocks = ["CARRY-FORWARD STATE (established fact — hold all of it):",
              "Everything the client has actually said, verbatim:"]
    blocks.extend(client_lines)
    facts_lines = []
    for ps in prior_sessions:
        sess_no = ps.get("n", ps.get("session_n"))
        for f in ps.get("facts") or []:
            facts_lines.append(f"  s{sess_no} C{f['turn']}: {f['fact']}")
    if facts_lines:
        blocks.append("VERIFIED FACTS THE CLIENT HAS STATED, turn-cited "
                      "(the ONLY prior-session specifics the therapist may reference):")
        blocks.extend(facts_lines)
    if last_ledger:
        blocks.append(f"Final ledger tl from last session: {last_ledger.get('tl', '')}")
    gap = plan["sessions"][len(prior_sessions)]["gap_before"] or "some time"
    blocks.append(f"Time gap since last session: {gap}.")
    blocks.append("Continue the arc. The client's facts above are established "
                  "state: keep them, track them in the tl, and let any earlier "
                  "thread move forward — do not re-litigate, do not reset.")
    blocks.append("CARRY-OVER LEDGER RULE: any 'sN:' entry you write in this "
                  "session's ledger must be traceable to a specific quoted "
                  "client line or VERIFIED FACT above. If you cannot point to "
                  "the line, do not carry the entry over. Never carry the "
                  "opposite of what the client stated, and never add specifics "
                  "(times, amounts, insurance, plans) that no line contains.")
    return "\n".join(blocks)


def build_session_prompt(plan: dict, n: int, prior_sessions: list[dict],
                         corrective_note: str | None) -> str:
    session = next(s for s in plan["sessions"] if s["n"] == n)
    client = plan["client"]
    parts = [f'ARC: "{plan["title"]}" — session {n} of {len(plan["sessions"])}.']

    parts.append(f"CLIENT: {client['name']}, {client['age']}, {client['occupation']}. "
                 f"Talks: {client['speech_style']}. {client['notes']}")
    parts.append("WORLD-TRUTH RULE: the CLIENT notes and every timeline entry marked "
                 "(untold) are true in the world but unknown to the therapist. The "
                 "therapist character can ONLY reference what the client lines actually "
                 "state in this transcript — never assert an untold fact, even as "
                 "rationale, until the client has said it first.")
    parts.append("CLIENT TIMELINE (established backdrop):\n" + _fmt_timeline(plan))

    real = plan["real_subject"]
    if real["session"] == n:
        parts.append(f"SURFACE SUBJECT (opening cover): {plan['surface_subject']}\n"
                     f"REAL SUBJECT (must surface around turn {real['surfaces_around_turn']}, "
                     f"and drive the rest of the arc): {real['content']}")
    elif real["session"] < n:
        parts.append(f"REAL SUBJECT (established, now central): {real['content']}")
    else:
        parts.append(f"SURFACE SUBJECT (this session is still cover): {plan['surface_subject']}")

    beats = _beats_for_session(plan, n)
    if beats:
        parts.append("SESSION BEATS (follow in order, at these turns):\n" + "\n".join(_fmt_beat(b) for b in beats))

    if plan["ending"]["session"] == n:
        parts.append(f"ENDING (around turn {plan['ending']['turn']}): {plan['ending']['requirement']}")

    if prior_sessions:
        parts.append(_fmt_carry_forward(prior_sessions, plan))
        has_facts = any(ps.get("facts") for ps in prior_sessions)
        source_a = ("facts in the VERIFIED FACTS list above" if has_facts
                    else "the client lines quoted in the carry-forward above")
        parts.append(
            "GROUNDING RULE: the therapist may reference ONLY (a) " + source_a +
            " and (b) what the client says in this session. Never assert "
            "a specific — date, duration, quantity, name, medical/legal/employment status, "
            "quotation — that is in neither. If a rationale needs a fact the client has "
            "not given, use its absence ('you haven't told me that yet'), never an "
            "invented substitute. Never invent an onset, duration, or interval: if the "
            "client gave no time, the ledger carries the fact without a time anchor. The "
            "time since the previous session is exactly the gap stated in this prompt — "
            "never a different interval. Ledger (told)/(claim)/(untold) markers must "
            "match this provenance exactly. Every timeline anchor in the final "
            "carried-forward ledger tl must persist in every ledger of this session — "
            "amended with a revision tag when it changes, never silently dropped."
        )

    parts.append(f"Write session {n} now: exactly {session['turns']} client turns and "
                 f"{session['turns']} therapist turns, alternating [C]/[T|THINK]/[T], "
                 f"starting with [C]. Count your [C] lines as you write and stop at "
                 f"exactly {session['turns']} — more or fewer is a hard failure. "
                 f"Session focus: {session['focus']}")

    if corrective_note:
        parts.append(f"CORRECTION FROM PREVIOUS ATTEMPT (fix all of these):\n{corrective_note}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Writer call
# ---------------------------------------------------------------------------

class TransientWriterError(Exception):
    pass


async def call_writer(session: aiohttp.ClientSession, api_key: str,
                      user_prompt: str) -> tuple[str, dict]:
    """Single attempt. Raises TransientWriterError on retryable failures,
    RuntimeError on fatal config problems. Retry lives in call_writer_with_retry."""
    payload = {
        "model": WRITER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "chat_template_kwargs": _THINKING_KWARG,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    timeout = aiohttp.ClientTimeout(total=CALL_TIMEOUT)
    try:
        async with session.post(FEATHERLESS_URL, json=payload, headers=headers,
                                timeout=timeout) as resp:
            status = resp.status
            if status in (401, 403, 404):
                body = await resp.text()
                raise RuntimeError(f"writer config error http_{status}: {body[:300]}")
            if status != HTTP_OK:
                raise TransientWriterError(f"http_{status}")
            data = await resp.json(content_type=None)
    except (TimeoutError, aiohttp.ClientError) as e:
        raise TransientWriterError(f"transport_error: {type(e).__name__}") from e

    if "choices" not in data or not data["choices"]:
        raise TransientWriterError("empty_response_no_choices")
    choice = data["choices"][0]
    content = choice.get("message", {}).get("content") or ""
    finish_reason = choice.get("finish_reason")
    if not content.strip():
        raise TransientWriterError("empty_content")
    if finish_reason not in {"stop", "length"}:
        raise TransientWriterError(f"finish_reason_{finish_reason}")
    return content, data.get("usage", {})


async def call_writer_with_retry(session: aiohttp.ClientSession, pool: KeyPool,
                                 user_prompt: str) -> tuple[str, dict, int]:
    backoff = 5.0
    retries = 0
    for attempt in range(1, WRITER_RETRY_LIMIT + 1):
        try:
            content, usage = await call_writer(session, pool.current(), user_prompt)
            return content, usage, retries
        except TransientWriterError as e:
            if attempt == WRITER_RETRY_LIMIT:
                raise
            retries += 1
            if is_rotate_status(e) and len(pool) > 1:
                pool.rotate()
                LOGGER.info(f"    [key-rotate] {e} — switched to ...{pool.current()[-8:]}",
)
            LOGGER.info(f"    [retry] transient writer error ({e}); backoff {backoff:.0f}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 40.0)
    raise TransientWriterError("unreachable")  # pragma: no cover


FACTS_SYSTEM_PROMPT = (
    "You are a meticulous transcript fact-extractor. You receive the lines a therapy "
    "client said in one session, each marked with a client-turn number. Extract EVERY "
    "specific factual claim the CLIENT states about their own life: names, relationships, "
    "events, durations and relative dates, quantities, medical/legal/employment status, "
    "what other people said or did (preserving direct quotations), places, objects. "
    "Only what the client literally states — no inference, no interpretation, nothing "
    "from the therapist's mouth. One fact per entry, as short as possible. Output strict "
    'JSON: {"facts": [{"turn": <client turn number>, "fact": "<one specific fact>"}]}'
)


async def extract_facts(session: aiohttp.ClientSession, pool: KeyPool,
                        client_name: str, turns: list[dict]) -> list[dict]:
    """Turn-cited facts list from a completed session's client lines.
    Best-effort: transient failures return [] (grounding stays, just thinner)."""
    lines = [f"C{i}: {t['content']}" for i, t in
             enumerate((t for t in turns if t["role"] == "client"), start=1)]
    payload = {
        "model": WRITER_MODEL,
        "messages": [
            {"role": "system", "content": FACTS_SYSTEM_PROMPT},
            {"role": "user", "content": f"Client {client_name} said:\n" + "\n".join(lines)},
        ],
        "max_tokens": FACTS_MAX_TOKENS,
        "temperature": 0.0,
        "chat_template_kwargs": _THINKING_KWARG,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {pool.current()}",
               "Content-Type": "application/json"}
    backoff = 5.0
    for attempt in range(1, FACTS_RETRY_LIMIT + 1):
        try:
            timeout = aiohttp.ClientTimeout(total=CALL_TIMEOUT)
            async with session.post(FEATHERLESS_URL, json=payload, headers=headers,
                                    timeout=timeout) as resp:
                if resp.status != HTTP_OK:
                    raise TransientWriterError(f"http_{resp.status}")
                data = await resp.json(content_type=None)
            choice = data["choices"][0]
            content = (choice.get("message", {}).get("content") or "").strip()
            if not content:
                raise TransientWriterError("empty_response")
            if choice.get("finish_reason") not in {"stop", "length"}:
                raise TransientWriterError(f"finish_reason_{choice.get('finish_reason')}")
            facts = json.loads(content).get("facts", [])
            return [
                {"turn": int(f["turn"]), "fact": str(f["fact"])}
                for f in facts
            ]
        except (
            TimeoutError,
            TransientWriterError,
            aiohttp.ClientError,
            KeyError,
            ValueError,
            json.JSONDecodeError,
        ) as e:
            if attempt == FACTS_RETRY_LIMIT:
                LOGGER.info(
                    f"    [facts-extract] failed after {attempt} attempts ({e}); "
                    "continuing without facts list"
                )
                return []
            if isinstance(e, TransientWriterError) and is_rotate_status(e) and len(pool) > 1:
                pool.rotate()
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 20.0)
            headers["Authorization"] = f"Bearer {pool.current()}"
    return []  # pragma: no cover


# ---------------------------------------------------------------------------
# Parsing + mechanical gates (spec §7.4)
# ---------------------------------------------------------------------------

def parse_transcript(text: str) -> dict:
    entries: list[tuple[str, object]] = []
    client_lines: list[str] = []
    therapist_lines: list[str] = []
    think_blocks: list[dict] = []
    unmarked: int = 0
    for line in text.splitlines():
        stripped = line.strip().strip("`")
        if not stripped:
            continue
        if stripped.startswith("[C]"):
            content = stripped[3:].strip()
            client_lines.append(content)
            entries.append(("client", content))
        elif stripped.startswith("[T|THINK]"):
            raw = stripped[len("[T|THINK]"):].strip()
            try:
                ledger = json.loads(raw)
                missing = [k for k in LEDGER_KEYS if k not in ledger]
                block = {"ok": True, "ledger": ledger, "missing": missing, "raw": raw}
            except json.JSONDecodeError:
                block = {"ok": False, "ledger": None, "missing": LEDGER_KEYS, "raw": raw}
            think_blocks.append(block)
            entries.append(("think", block))
        elif stripped.startswith("[T]"):
            content = stripped[3:].strip()
            therapist_lines.append(content)
            entries.append(("therapist", content))
        else:
            unmarked += 1
    return {
        "client_lines": client_lines,
        "therapist_lines": therapist_lines,
        "think_blocks": think_blocks,
        "unmarked_lines": unmarked,
        "entries": entries,
    }


_TL_ANCHOR_RE = re.compile(r"(?:^|[;,])\s*(now|[-+]?\d+[ymwd])(?=\s*:)", re.M)


def _tl_anchor_keys(tl: str) -> set[str]:
    """Extract time anchors (``now``, ``-6w``, ``+1y`` ...) from a tl ledger.

    Ledger entries are ``anchor: description`` separated by ``;`` (or ```,``).
    Descriptions themselves may contain commas/semicolons, so splitting on a
    delimiter and taking the text before the first colon yields phantom keys.
    Instead match the short time-anchor token that sits at an entry boundary
    immediately before its colon.
    """
    return set(_TL_ANCHOR_RE.findall(tl or ""))


def _turn_tolerance(target: int) -> int:
    """Pacing tolerance that scales with session length.

    The writer cannot hit exact turn counts (observed drift ±5), so a fixed
    ±2 rejects clean sessions over pacing variance alone and burns full
    rewrite attempts. Allow variance proportional to length: the floor keeps
    short sessions honest, the cap stops long sessions from going over-loose,
    and a genuine bail (a 20-turn plan collapsing to ~8 turns) still fails.
    """
    return min(TURN_TOLERANCE_CAP,
               max(TURN_TOLERANCE_FLOOR, round(target * TURN_TOLERANCE_FRACTION)))


def _format_and_pacing_failures(plan: dict, n: int, parsed: dict) -> list[str]:
    failures = []
    target = next(session["turns"] for session in plan["sessions"] if session["n"] == n)
    if parsed["unmarked_lines"]:
        failures.append(f"marker_format: {parsed['unmarked_lines']} unmarked lines")
    tolerance = _turn_tolerance(target)
    for speaker in ("client", "therapist"):
        count = len(parsed[f"{speaker}_lines"])
        if abs(count - target) > tolerance:
            failures.append(f"turn_count: {speaker} turns {count} vs target {target}")

    parse_failures = sum(not block["ok"] for block in parsed["think_blocks"])
    if parse_failures:
        failures.append(f"ledger_parse: {parse_failures} think-blocks failed JSON parse")
    missing_fields = sum(len(block["missing"]) for block in parsed["think_blocks"] if block["ok"])
    if missing_fields:
        failures.append(f"ledger_fields: {missing_fields} missing ledger keys")
    if len(parsed["think_blocks"]) != len(parsed["therapist_lines"]):
        failures.append(
            f"ledger_count: {len(parsed['think_blocks'])} think-blocks vs "
            f"{len(parsed['therapist_lines'])} therapist replies"
        )
    return failures


def _content_failures(plan: dict, n: int, parsed: dict) -> list[str]:
    failures = []
    gate_hits = [
        (line, reason)
        for line in parsed["therapist_lines"]
        for caught, reason in [is_sycophantic(line)]
        if caught
    ]
    if gate_hits:
        details = "; ".join(f"[{reason}] {line[:60]}" for line, reason in gate_hits[:3])
        failures.append(f"cliche_gate: {len(gate_hits)} banned-phrase hits: {details}")

    date_hits = [
        (line[:60], match.group(0))
        for line in parsed["client_lines"] + parsed["therapist_lines"]
        if (match := _MONTH_RE.search(line))
    ]
    if date_hits:
        details = "; ".join(f'"{month}" in {line}' for line, month in date_hits[:3])
        failures.append(f"calendar_dates: {len(date_hits)} hits: {details}")

    quote_re = re.compile(r"(?:(?<=^)|(?<=\s))'([^']+)'(?=$|\s|[.,;!?])")
    client_text = _containment_norm("\n".join(parsed["client_lines"]))
    missing_beats = []
    for beat in plan.get("beats", []):
        for phase, session_key, phrase_key in (
            ("plant", "plant_session", "original"),
            ("revision", "revise_session", "revision"),
        ):
            if beat["type"] == "misstatement" and beat.get(session_key) == n:
                missing_beats.extend(
                    f"{phase} s{n}: client must state '{phrase}'"
                    for phrase in quote_re.findall(beat.get(phrase_key, ""))
                    if _containment_norm(phrase) not in client_text
                )
    if missing_beats:
        failures.append("beat_content: " + "; ".join(missing_beats[:3]))
    return failures


def _timeline_failures(parsed: dict, prior_sessions: list[dict]) -> list[str]:
    accumulated = {
        anchor
        for session in prior_sessions
        for turn in session["turns"]
        if turn.get("ledger")
        for anchor in _tl_anchor_keys(turn["ledger"].get("tl", ""))
    }
    drifts = []
    for block in parsed["think_blocks"]:
        if not block["ok"]:
            continue
        keys = _tl_anchor_keys(block["ledger"].get("tl", ""))
        vanished = accumulated - keys
        if vanished:
            drifts.append(vanished)
        accumulated = keys
    if not drifts:
        return []
    return [f"tl_drift: {len(drifts)} ledgers dropped earlier anchors: "
            + "; ".join(str(sorted(anchors)) for anchors in drifts[:3])]


def run_mechanical_gates(plan: dict, n: int, parsed: dict,
                         prior_sessions: list[dict]) -> list[str]:
    failures = _format_and_pacing_failures(plan, n, parsed)
    failures.extend(_content_failures(plan, n, parsed))
    failures.extend(_timeline_failures(parsed, prior_sessions))
    failures.extend(ledger_gate_failures(plan, n, parsed, prior_sessions))
    return failures


def merge_gate_failures(accumulated: list[str], new: list[str], cap: int = 3) -> list[str]:
    """Accumulate gate failures across attempts, deduped per gate head.

    The corrective note must carry every gate that has failed for this
    session, not just the latest attempt's — otherwise fixing one gate lets
    a previously-passing gate regress on the next attempt (oscillation).
    """
    by_head: dict[str, list[str]] = {}
    for f in [*accumulated, *new]:
        head = f.split(":", 1)[0]
        bucket = by_head.setdefault(head, [])
        if f not in bucket and len(bucket) < cap:
            bucket.append(f)
    return [f for bucket in by_head.values() for f in bucket]


def corrective_note_for(failures: list[str]) -> str:
    notes = {
        "marker_format": "- Output ONLY [C]/[T|THINK]/[T] lines. No commentary, no scene direction, no markdown.",
        "turn_count": "Hit the requested turn count exactly.",
        "ledger_parse": "- Every [T|THINK] block must be ONE line of strict JSON. No trailing commas, no smart quotes.",
        "ledger_fields": "- Every ledger JSON must contain all 9 keys: dx, def, soma, risk, hx, onset, track, tx, tl.",
        "ledger_count": "- Exactly one [T|THINK] block before every [T] reply, no gaps.",
        "calendar_dates": "- Replace calendar months/years with relative time ('three weeks ago', 'last spring').",
        "tl_drift": (
            "- The tl ledger is append-mostly: once an anchor appears, it persists "
            "in every later ledger (amended with a revision tag, never silently dropped)."
        ),
    }
    details = {
        "beat_content": (
            "- A planned beat is missing its exact client wording. The client must say "
            "the quoted phrase verbatim at the beat's turn: "
        ),
        "ledger_traceability": (
            "- Ledger fabrication caught. Every number, duration, anchor (-3w style), "
            "and quoted phrase marked told/claim must come from the plan's fact schedule, "
            "the client's own words, or the carry-forward block. Do not invent quantities, "
            "time spans, or client quotes. Offending items: "
        ),
        "spoken_specificity": (
            "- Invented specificity caught. Do not put past durations ('three weeks ago'), "
            "frequencies ('four times a week'), or clock times in therapist speech unless "
            "the client already said exactly that. Keep it vague or use the client's own "
            "words. Offending items: "
        ),
        "cliche_gate": (
            "- A therapist line opens by parroting the client's words back at them "
            "(e.g. 'You said ...', 'You just said ...', 'You named ...'). Never open a "
            "therapist line by quoting or restating what the client just said. Start from "
            "your own observation, a question, or a held pause; engage the content in your "
            "own words. Rewrite the offending line from the judgment itself and do NOT "
            "reuse it on a later attempt. Beat pressure is not permission to agree: hold "
            "the line neutrally, naming the discrepancy in your own words. "
        ),
    }
    result = []
    for failure in failures:
        head, _, detail = failure.partition(":")
        if head in details:
            result.append(details[head] + detail.strip()[:250])
        elif head in notes:
            result.append(notes[head] if head != "turn_count" else f"- {failure}. {notes[head]}")
        else:
            result.append(f"- {failure}")
    return "\n".join(result)


def dump_failed_attempt(arc_id: str, n: int, attempt: int, content: str,
                        failures: list[str]) -> None:
    """Persist a gate-failed attempt verbatim for postmortem (failed
    transcripts otherwise vanish — only OK sessions reach the checkpoint)."""
    d = OUT_DIR / "failed_attempts"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{arc_id}_s{n}_a{attempt}.txt"
    path.write_text(f"# gates: {'; '.join(failures)}\n{content}", encoding="utf-8")


def turns_from_parsed(parsed: dict) -> list[dict]:
    """Order-preserving fold: each [T|THINK] merges into the [T] that follows
    it; a trailing unmatched [C] is kept; stray thinks are dropped (the
    ledger_count gate already flags the mismatch)."""
    turns: list[dict] = []
    pending_client: str | None = None
    for kind, val in parsed["entries"]:
        if kind == "client":
            if pending_client is not None:
                turns.append({"role": "client", "content": pending_client})
            pending_client = val
        elif kind == "think":
            turns.append({"role": "think", "ledger": val["ledger"] if val["ok"] else None})
        elif kind == "therapist":
            ledger = None
            if turns and turns[-1]["role"] == "think":
                ledger = turns.pop()["ledger"]
            if pending_client is not None:
                turns.append({"role": "client", "content": pending_client})
                pending_client = None
            turns.append({"role": "therapist", "content": val, "ledger": ledger})
    if pending_client is not None:
        turns.append({"role": "client", "content": pending_client})
    # any surviving "think" row is a stray with no following [T] — drop it
    # (the ledger_count gate already flags the mismatch)
    return [t for t in turns if t["role"] != "think"]


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path) -> dict[str, dict[int, dict]]:
    done: dict[str, dict[int, dict]] = {}
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.setdefault(row["arc_id"], {})[row["session_n"]] = row
    return done


def load_audit_notes(arc_id: str) -> dict[str, str]:
    """Audit notes from the revision loop (spec §7.3): session -> corrective
    text written by audit_arc_corpus.py for flagged sessions."""
    path = OUT_DIR / "arc_audit_notes" / f"{arc_id}.json"
    if not path.exists():
        return {}
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return {}


def append_jsonl_fsync(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# W&B (guarded)
# ---------------------------------------------------------------------------

def _maybe_init_wandb(config: dict):
    key = os.environ.get("WANDB_API_KEY", "")
    if not key:
        return None
    return wandb.init(project="pixelated-empathy-kan28",
                      name=f"arc-generate-{time.strftime('%Y%m%d-%H%M%S')}",
                      job_type="arc_generation", config=config)


def _wandb_metrics_loop(run, snapshot_fn, interval=60.0):
    if run is None:
        return None

    async def loop():
        while True:
            with suppress(Exception):
                run.log(snapshot_fn())
            await asyncio.sleep(interval)
    return asyncio.create_task(loop())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@dataclass
class ArcGenerationContext:
    checkpoint: dict
    out_paths: dict
    session_http: aiohttp.ClientSession
    pool: KeyPool
    semaphore: asyncio.Semaphore
    counters: dict


def _render_arc_record(plan: dict, prior: dict[int, dict]) -> dict:
    final_tl = None
    for session_n in sorted(prior, reverse=True):
        for turn in reversed(prior[session_n]["turns"]):
            if turn.get("ledger"):
                final_tl = turn["ledger"].get("tl")
                break
        if final_tl:
            break
    return {
        "arc_id": plan["arc_id"],
        "spec_version": 1,
        "plan_path": f"training/arc_plans/{plan['arc_id']}.json",
        "writer_model": WRITER_MODEL,
        "auditor_model": None,
        "audit": None,
        "sessions": [
            {"n": n, "turns": prior[n]["turns"]} for n in sorted(prior)
        ],
        "timeline_final": final_tl,
        "beats_planned": plan["beats"],
        "metrics": {
            "sessions": len(prior),
            "writer_tokens": sum(
                prior[n]["usage"].get("completion_tokens", 0) for n in prior
            ),
            "wall_s": sum(prior[n]["wall_s"] for n in prior),
            "attempts": {n: prior[n]["attempts"] for n in sorted(prior)},
        },
    }


async def _generate_sessions(
    plan: dict,
    prior: dict[int, dict],
    context: ArcGenerationContext,
    arc_id: str,
    audit_notes: dict[str, str],
) -> dict | None:
    for session_def in plan["sessions"]:
        n = session_def["n"]
        if n in prior:
            continue
        prior_sessions = [prior[k] for k in sorted(prior) if k < n]
        base_note = audit_notes.get(str(n))
        prompt_base = build_session_prompt(plan, n, prior_sessions, base_note)
        failures: list[str] = []
        all_failures: list[str] = []
        for attempt in range(1, SESSION_ATTEMPTS + 1):
            if attempt == 1:
                prompt = prompt_base
            else:
                all_failures = merge_gate_failures(all_failures, failures)
                combined = "\n".join(
                    x for x in (base_note, corrective_note_for(all_failures)) if x
                )
                prompt = build_session_prompt(plan, n, prior_sessions, combined)
            t0 = time.monotonic()
            content, usage, retries = await call_writer_with_retry(
                context.session_http, context.pool, prompt
            )
            wall_s = round(time.monotonic() - t0, 1)
            parsed = parse_transcript(content)
            failures = run_mechanical_gates(plan, n, parsed, prior_sessions)
            context.counters["writer_calls"] += 1
            context.counters["writer_retries"] += retries
            context.counters["writer_tokens"] += usage.get("completion_tokens") or 0
            if not failures:
                break
            dump_failed_attempt(arc_id, n, attempt, content, failures)
            LOGGER.info(
                f"  {arc_id} s{n} attempt {attempt} gates failed: "
                f"{'; '.join(failures[:3])}"
            )
        if failures:
            context.counters["gate_failures"] += 1
            LOGGER.info(
                f"  {arc_id} s{n}: GATE FAILURE after {SESSION_ATTEMPTS} attempts — "
                "session left incomplete (will resume)"
            )
            return {"arc_id": arc_id, "status": "gate_failed",
                    "session": n, "failures": failures}

        facts = await extract_facts(
            context.session_http, context.pool, plan["client"]["name"],
            turns_from_parsed(parsed),
        )
        row = {
            "arc_id": arc_id,
            "n": n,
            "session_n": n,
            "writer_model": WRITER_MODEL,
            "wall_s": wall_s,
            "usage": usage,
            "attempts": attempt,
            "facts": facts,
            "turns": turns_from_parsed(parsed),
        }
        context.counters["facts_calls"] += 1
        context.counters["facts_tokens"] += row["usage"].get("completion_tokens") or 0
        append_jsonl_fsync(context.out_paths["sessions"], row)
        prior[n] = row
        context.counters["sessions_done"] += 1
        LOGGER.info(
            f"  {arc_id} s{n}: OK — {len(parsed['client_lines'])}C/"
            f"{len(parsed['therapist_lines'])}T, wall={wall_s}s, "
            f"tokens={usage.get('completion_tokens')}"
        )
    return None


async def generate_arc(plan: dict, context: ArcGenerationContext) -> dict | None:
    arc_id = plan["arc_id"]
    prior: dict[int, dict] = context.checkpoint.get(arc_id, {})
    if all(n in prior for n in (s["n"] for s in plan["sessions"])):
        return None  # already complete

    async with context.semaphore:
        result = await _generate_sessions(
            plan, prior, context, arc_id, load_audit_notes(arc_id)
        )
        if result:
            return result
        record = _render_arc_record(plan, prior)
        append_jsonl_fsync(context.out_paths["records"], record)
        context.counters["arcs_done"] += 1
        LOGGER.info(f"  {arc_id}: RECORD COMPLETE ({len(plan['sessions'])} sessions)")
        return {"arc_id": arc_id, "status": "complete", "record": record}


async def main_async(args) -> None:
    pool_envs = [
        k.strip() for k in os.environ.get(
            "ARC_WRITER_KEYS",
            "AI_GATEWAY_API_KEY,FEATHERLESS_API_KEY,FEATHERLESS_API_KEY_2",
        ).split(",") if k.strip()
    ]
    pool = KeyPool(*pool_envs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)

    plan_files = sorted(PLANS_DIR.glob("*.json"))
    if args.plans:
        wanted = {p.strip() for p in args.plans.split(",")}
        plan_files = [f for f in plan_files if f.stem in wanted]
    if args.limit:
        plan_files = plan_files[: args.limit]
    if not plan_files:
        raise SystemExit("no arc plans found")
    plans = [json.loads(f.read_text(encoding="utf-8")) for f in plan_files]

    out_paths = {
        "sessions": OUT_DIR / (args.checkpoint or "sessions_checkpoint.jsonl"),
        "records": OUT_DIR / (args.output or "arc_records.jsonl"),
    }
    checkpoint = load_checkpoint(out_paths["sessions"])
    counters = {"writer_calls": 0, "writer_retries": 0, "writer_tokens": 0,
                "facts_calls": 0, "facts_tokens": 0,
                "sessions_done": 0, "arcs_done": 0, "gate_failures": 0}

    total_sessions = sum(len(p["sessions"]) for p in plans)
    LOGGER.info("=== ARC GENERATOR ===")
    LOGGER.info(f"plans={len(plans)} sessions={total_sessions} model={WRITER_MODEL} "
          f"thinking=off max_tokens={MAX_TOKENS} concurrency={args.concurrency}")
    LOGGER.info(f"resume: {sum(len(v) for v in checkpoint.values())} sessions already done")
    LOGGER.info(f"featherless: {pool.describe()}")

    run = _maybe_init_wandb({
        "writer_model": WRITER_MODEL, "thinking": False, "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE, "plans": [p["arc_id"] for p in plans],
        "sessions_total": total_sessions, "concurrency": args.concurrency,
    })
    metrics_task = _wandb_metrics_loop(run, lambda: {**counters})

    t0 = time.monotonic()
    semaphore = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession() as http:
        context = ArcGenerationContext(checkpoint, out_paths, http, pool, semaphore, counters)
        results = await asyncio.gather(
            *(generate_arc(plan, context) for plan in plans),
            return_exceptions=True,
        )

    if metrics_task:
        metrics_task.cancel()
    if run:
        run.log({**counters, "wall_s": round(time.monotonic() - t0, 1)})
        run.finish()

    failed = [r for r in results if isinstance(r, Exception)]
    gate_failed = [r for r in results if isinstance(r, dict) and r.get("status") == "gate_failed"]
    for r in failed:
        LOGGER.info(f"ERROR: {r}")
    LOGGER.info(f"\nDONE in {round(time.monotonic() - t0, 1)}s — sessions={counters['sessions_done']} "
          f"arcs={counters['arcs_done']} gate_failures={len(gate_failed)} errors={len(failed)} "
          f"tokens={counters['writer_tokens']}")
    if failed or gate_failed:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", help="comma-separated arc_ids (default: all)")
    ap.add_argument("--limit", type=int, help="max number of plans")
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--output", help="arc records JSONL filename")
    ap.add_argument("--checkpoint", help="sessions checkpoint JSONL filename")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
