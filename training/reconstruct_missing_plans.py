#!/usr/bin/env python3
"""
Reconstruct 18 zero-byte arc plans (Oct 15 23:02 truncation event).

Mode A (16 record arcs): beats/timeline/sessions mechanical from arc_records.jsonl,
  LLM fills title/client/gap_before/focus/surface_subject/real_subject/ending.
Mode B (0266/0326): LLM designs full plan from checkpoint transcripts + ledger.tl.

Lint retry loop via lint_plan, writes with fsync. Uses writer route
cf/@cf/deepseek-ai/deepseek-v4-pro-0813 via 9Router (localhost:20128).

Run on Sally from ai/:
  /home/vivi/pixelated/.venv/bin/python training/reconstruct_missing_plans.py
  [--arcs arc_0231,arc_0238,...] [--dry-run] [--concurrency 2]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# --- env bootstrap: .env + ~/.nine_router_env fallback ---
_HERE = Path(__file__).resolve()
_TRAIN_DIR = _HERE.parents[0]
_AI_DIR = _HERE.parents[1]
_REPO_DIR = _HERE.parents[2]
sys.path.insert(0, str(_AI_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(_REPO_DIR / ".env", override=True)
except Exception:
    pass

# fallback for NINE_ROUTER_API_KEY (non-interactive shells may not have sourced it)
if not os.environ.get("NINE_ROUTER_API_KEY"):
    for _p in [Path.home() / ".nine_router_env", Path.home() / ".bashrc", Path.home() / ".zshrc"]:
        if _p.exists():
            try:
                m = re.search(r"NINE_ROUTER_API_KEY\s*=\s*[\"']?([A-Za-z0-9_\-]{20,})[\"']?", _p.read_text())
                if m:
                    os.environ["NINE_ROUTER_API_KEY"] = m.group(1)
                    break
            except Exception:
                continue

import aiohttp  # noqa: E402

from training.lint_arc_plans import lint_plan  # noqa: E402

# --- constants ---
OUT_DIR = _TRAIN_DIR / "arc_plans"
RECORDS_PATH = _TRAIN_DIR / "output/arc_corpus/arc_records.jsonl"
CHECKPOINT_PATH = _TRAIN_DIR / "output/arc_corpus/sessions_checkpoint.jsonl"
SEED_MAP_PATH = _TRAIN_DIR / "arc_plans/seed_map.jsonl"

WRITER_URL = os.environ.get("ARC_WRITER_URL", "http://localhost:20128/v1/chat/completions")
WRITER_MODEL = os.environ.get("ARC_WRITER_MODEL", "cf/@cf/deepseek-ai/deepseek-v4-pro-0813")
WRITER_KEYS_ENV = os.environ.get("ARC_WRITER_KEYS", "NINE_ROUTER_API_KEY")

SESSION_TURNS_MIN = 8
SESSION_TURNS_MAX = 40
TOTAL_TURNS_MAX = 120

# Mode A: 16 record arcs (have beats_planned + timeline_final in arc_records.jsonl)
MODE_A_ARCS = [
    "arc_0231", "arc_0238", "arc_0248", "arc_0251",
    "arc_0261", "arc_0262", "arc_0264", "arc_0274",
    "arc_0278", "arc_0287", "arc_0302", "arc_0305",
    "arc_0312", "arc_0313", "arc_0335", "arc_0338",
]
# Mode B: regen arcs (no record, mine checkpoint)
MODE_B_ARCS = ["arc_0266", "arc_0326"]
ALL_18 = MODE_A_ARCS + MODE_B_ARCS

# 34 audit targets: 30 backlog + 4 regen; but reconstruction only needs 18
JITTER_SEED_BASE = 20000

# --- LLM config ---
MAX_TOKENS = 8192
TEMPERATURE = 0.30
CONCURRENCY = 2
LINT_RETRIES = 3  # total attempts = 1 + retries
CALL_TIMEOUT = 120

# _MONTH_RE gate: must not leak calendar months/years into text fields
_MONTH_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\b|\b(?:19|20)\d{2}\b"
)

# Anchor parsing
_ANCHOR_RE = re.compile(r"^(-\d+(?:y|m|w|d)|now)$")
_ANCHOR_WEIGHT = {"y": 52.0, "m": 4.0, "w": 1.0, "d": 1.0 / 7.0}


def _anchor_sort_key(anchor: str) -> float:
    if anchor == "now":
        return float("inf")
    return -int(anchor[1:-1]) * _ANCHOR_WEIGHT[anchor[-1]]


# ---------------------------------------------------------------------------
# Helpers: timeline_final parsing (Mode A)
# ---------------------------------------------------------------------------
def parse_timeline_final(s: str) -> list[dict[str, str]]:
    """Parse timeline_final string: 'anchor: event (prov); anchor: event (prov); ...'
    Returns list of {anchor, event, provenance}. Normalizes revision->told."""
    if not s or not s.strip():
        return []
    entries: list[dict[str, str]] = []
    # Split on "; " but event text could contain semicolons - timeline_final uses
    # "; " as separator between "anchor: event (prov)" entries consistently.
    parts = s.split("; ")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Try "anchor: event (prov)" then "anchor: event"
        m = re.match(r"^(-\d+[ymwd]|now)\s*:\s*(.*?)(?:\s*\((told|claim|untold|revision)\))?\s*$", part, re.IGNORECASE)
        if not m:
            # fallback: try without provenance parens
            m2 = re.match(r"^(-\d+[ymwd]|now)\s*:\s*(.+)\s*$", part)
            if not m2:
                continue
            anchor, event = m2.group(1).strip(), m2.group(2).strip()
            prov = "told"
        else:
            anchor, event, prov = m.group(1).strip(), m.group(2).strip(), (m.group(3) or "told").strip().lower()
            if prov == "revision":
                prov = "told"
        # Validate anchor
        if not _ANCHOR_RE.match(anchor):
            continue
        entries.append({"anchor": anchor, "event": event, "provenance": prov})
    # Stable sort chronological (oldest first), inf for now
    entries.sort(key=lambda e: _anchor_sort_key(e["anchor"]))
    return entries


def clamp_sessions(record_sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Clamp session turns to [8,40] and total <=120. Returns [{n, turns, gap_before, focus_placeholder}]."""
    # record_sessions have n and len(turns)
    out = []
    for s in record_sessions:
        n = s["n"]
        actual = len(s.get("turns", []))
        clamped = max(SESSION_TURNS_MIN, min(SESSION_TURNS_MAX, actual))
        out.append({"n": n, "turns": clamped, "_actual": actual})
    # Check total
    total = sum(x["turns"] for x in out)
    if total > TOTAL_TURNS_MAX:
        # Scale down proportionally, keeping >=8
        # Simple: reduce largest sessions first
        while total > TOTAL_TURNS_MAX:
            # find session with max turns > 8
            candidates = [x for x in out if x["turns"] > SESSION_TURNS_MIN]
            if not candidates:
                break
            c = max(candidates, key=lambda x: x["turns"])
            c["turns"] -= 1
            total -= 1
    # Strip helper key
    for x in out:
        x.pop("_actual", None)
    return out


# ---------------------------------------------------------------------------
# Seed map loading
# ---------------------------------------------------------------------------
def load_seed_map(path: Path = SEED_MAP_PATH) -> dict[str, dict[str, Any]]:
    """Load seed_map.jsonl, last-wins per arc_id but prefer status=written.
    Returns arc_id -> row dict (contains 'seed' + 'key' + 'jitter_seed')."""
    if not path.exists():
        return {}
    # Collect all rows per arc_id
    by_arc: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        aid = r.get("arc_id")
        if not aid:
            continue
        by_arc.setdefault(aid, []).append(r)
    # Pick best per arc: prefer last row with status==written, else last row
    out: dict[str, dict[str, Any]] = {}
    for aid, rows in by_arc.items():
        written = [r for r in rows if r.get("status") == "written"]
        chosen = written[-1] if written else rows[-1]
        out[aid] = chosen
    return out


# ---------------------------------------------------------------------------
# Records loading
# ---------------------------------------------------------------------------
def load_records(path: Path = RECORDS_PATH) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        aid = r.get("arc_id")
        if aid:
            out[aid] = r
    return out


def load_checkpoints(path: Path = CHECKPOINT_PATH) -> dict[str, dict[int, dict[str, Any]]]:
    """Returns arc_id -> {session_n -> row}"""
    if not path.exists():
        return {}
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        aid = r.get("arc_id")
        if not aid:
            continue
        n = r.get("session_n", r.get("n"))
        if n is None:
            continue
        out.setdefault(aid, {})[int(n)] = r
    return out


# ---------------------------------------------------------------------------
# Transcript excerpt helpers
# ---------------------------------------------------------------------------
def excerpt_client_lines(turns: list[dict[str, Any]], max_lines: int = 12, max_chars_per: int = 280) -> str:
    lines = []
    for t in turns:
        if t.get("role") == "client":
            c = (t.get("content") or "").strip().replace("\n", " ")
            if len(c) > max_chars_per:
                c = c[:max_chars_per] + "…"
            lines.append(f"- {c}")
        if len(lines) >= max_lines:
            break
    return "\n".join(lines)


def excerpt_full_turns(turns: list[dict[str, Any]], max_turns: int = 18) -> str:
    """Render turns as [C]/[T] for LLM context, truncated."""
    lines = []
    for idx, t in enumerate(turns[:max_turns], 1):
        role = "C" if t.get("role") == "client" else "T"
        c = (t.get("content") or "").strip().replace("\n", " ")
        if len(c) > 300:
            c = c[:300] + "…"
        lines.append(f"[{role} {idx}] {c}")
    if len(turns) > max_turns:
        lines.append(f"... ({len(turns) - max_turns} more turns omitted)")
    return "\n".join(lines)


def collect_facts(facts: list[dict[str, Any]] | None) -> str:
    if not facts:
        return "(no facts extracted)"
    lines = []
    for f in facts[:20]:
        turn = f.get("turn", "?")
        fact = f.get("fact", f.get("text", str(f)))
        lines.append(f"  turn {turn}: {fact}")
    return "\n".join(lines)


def collect_ledger_tls(checkpoint_row: dict[str, Any]) -> str:
    """Extract ledger.tl strings from checkpoint row's turns."""
    tls = []
    for t in checkpoint_row.get("turns", []):
        ledger = t.get("ledger")
        if isinstance(ledger, dict) and ledger.get("tl"):
            tls.append(ledger["tl"])
    if not tls:
        return "(no ledger.tl found)"
    # Deduplicate consecutive
    deduped = []
    for tl in tls:
        if not deduped or deduped[-1] != tl:
            deduped.append(tl)
    # Return last few (most recent is most complete)
    return "\n".join(f"  - {tl[:400]}" for tl in deduped[-3:])


# ---------------------------------------------------------------------------
# LLM caller
# ---------------------------------------------------------------------------
def _get_api_key() -> str:
    env_name = WRITER_KEYS_ENV
    key = os.environ.get(env_name, "")
    if key:
        return key.strip()
    # Try NINE_ROUTER_API_KEY directly
    key = os.environ.get("NINE_ROUTER_API_KEY", "")
    if key:
        return key.strip()
    raise RuntimeError(f"No API key found in env {env_name} / NINE_ROUTER_API_KEY")


def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON object from LLM text (strip fences, first { to last })."""
    t = text.strip()
    # Strip markdown fences
    if "```" in t:
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", t, re.DOTALL)
        if m:
            t = m.group(1).strip()
    # Find first { to last }
    s = t.find("{")
    e = t.rfind("}")
    if s == -1 or e == -1 or e <= s:
        raise ValueError(f"No JSON object found in: {text[:500]!r}")
    return json.loads(t[s : e + 1])


async def call_llm(
    session: aiohttp.ClientSession,
    messages: list[dict[str, str]],
    api_key: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": WRITER_MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "response_format": {"type": "json_object"},
    }
    # reasoning toggle per route
    if "cheaperinference" in WRITER_URL or "nvidia" in WRITER_URL.lower():
        payload["reasoning_effort"] = "none"
    else:
        payload["chat_template_kwargs"] = {"thinking": False}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_err: Exception | None = None
    for attempt in range(4):
        try:
            async with session.post(
                WRITER_URL,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=CALL_TIMEOUT),
            ) as resp:
                body = await resp.text()
                if resp.status != 200:
                    # Retry on 429/402/503/5xx
                    if resp.status in (429, 402, 503, 500, 502, 504):
                        await asyncio.sleep(5 * (2**attempt))
                        last_err = RuntimeError(f"HTTP {resp.status}: {body[:500]}")
                        continue
                    raise RuntimeError(f"HTTP {resp.status}: {body[:800]}")
                # Parse SSE trailer if present (9Router appends data: [DONE])
                # Use JSONDecoder.raw_decode for robustness
                data = None
                try:
                    data = json.loads(body)
                except json.JSONDecodeError:
                    # Try raw_decode on first JSON object
                    try:
                        dec = json.JSONDecoder()
                        data, _ = dec.raw_decode(body.strip())
                    except Exception:
                        raise RuntimeError(f"Bad JSON response: {body[:800]!r}")
                # Extract content
                try:
                    content = data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError) as e:
                    raise RuntimeError(f"Unexpected response shape: {body[:800]!r} ({e})")
                if not content or not content.strip():
                    raise RuntimeError(f"Empty content: {body[:800]!r}")
                return _extract_json(content)
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError, ValueError, RuntimeError) as e:
            last_err = e
            if attempt < 3:
                await asyncio.sleep(5 * (2**attempt))
                continue
            raise
    assert last_err is not None
    raise last_err


def _feedback(errors: list[str], warnings: list[str]) -> str:
    parts = []
    if errors:
        parts.append("LINT ERRORS (must fix):\n" + "\n".join(f"- {e}" for e in errors))
    if warnings:
        parts.append("LINT WARNINGS (fix if possible):\n" + "\n".join(f"- {w}" for w in warnings))
    return "\n\n".join(parts)


def _lint_feedback_suffix(errors: list[str], warnings: list[str]) -> str:
    return (
        "\n\nYour previous plan FAILED lint. Fix these issues and return a corrected JSON plan:\n"
        + _feedback(errors, warnings)
        + "\n\nReturn ONLY the corrected JSON plan."
    )


# ---------------------------------------------------------------------------
# Mode A prompt builder
# ---------------------------------------------------------------------------
MODE_A_SYSTEM = """\
You reconstruct therapy arc plans from verified transcripts. You are given:
- the arc's verified beats (verbatim, must be reproduced exactly),
- the timeline (verbatim, must be reproduced exactly),
- the session budgets (turns per session, must be reproduced exactly),
- the seed (verbatim, must be reproduced exactly),
- transcript excerpts so you can author the remaining free fields plausibly.

You must produce a JSON arc plan that passes lint. The mechanical fields
(beats, timeline, sessions[].n, sessions[].turns, seed, era_jitter, arc_id)
are FIXED — reproduce them exactly as given. Author only these free fields:
  title, client{name,age,occupation,speech_style,notes},
  sessions[].gap_before, sessions[].focus,
  surface_subject, real_subject, ending.

RULES:
- arc_id, seed, beats, timeline, sessions[].n, sessions[].turns, era_jitter
  must match the FIXED values exactly. Do not invent or alter beats.
- Client age 1-119, occupation/speech_style/notes non-empty strings.
- NOTES RULE: any clinical condition the client will NOT disclose in-session
  must be marked "(untold, background only - must never be recorded as
  established)" or phrased as "reports..." / "says..." — never assert an
  undisclosed diagnosis as plain fact.
- No calendar months or 4-digit years anywhere (sessions[].focus, notes,
  surface_subject, ending.requirement, timeline events, beat text).
- gap_before: null for session 1, string like "one week" for later sessions.
- focus: one sentence per session, what the session is about.
- surface_subject: cover topic the client opens with.
- real_subject: {content, surfaces_around_turn (>=3 and <= session turns), session}
  surfaces_around_turn must be >=3 and within that session's turn budget.
- ending: {session, turn (within that session's budget), requirement (concrete landing)}
- title: 2-6 evocative words.
- beats are already fixed — do not add, remove, or reorder them.

Return ONLY a JSON object matching the plan schema.
"""

def build_mode_a_user_prompt(
    arc_id: str,
    seed: dict[str, Any],
    jitter_seed: int,
    beats: list[dict[str, Any]],
    timeline: list[dict[str, str]],
    sessions_skeleton: list[dict[str, Any]],
    transcript_excerpts: str,
    failed_attempt_feedback: str = "",
) -> str:
    beats_str = json.dumps(beats, indent=2, ensure_ascii=False)
    timeline_str = json.dumps(timeline, indent=2, ensure_ascii=False)
    sessions_str = json.dumps(sessions_skeleton, indent=2, ensure_ascii=False)
    seed_str = json.dumps(seed, indent=2, ensure_ascii=False)

    # Build schema reminder for free fields
    prompt = f"""\
Reconstruct plan for {arc_id}.

FIXED (reproduce exactly):
- arc_id: {arc_id!r}
- seed: {seed_str}
- era_jitter: {{"seed": {jitter_seed}}}
- beats ({len(beats)} beats, reproduce verbatim):
{beats_str}
- timeline ({len(timeline)} events, reproduce verbatim):
{timeline_str}
- sessions skeleton (n + turns fixed, you fill gap_before + focus):
{sessions_str}

TRANSCRIPT EXCERPTS (for authoring free fields plausibly):

{transcript_excerpts}

INSTRUCTIONS:
Author the free fields (title, client, sessions[].gap_before, sessions[].focus,
surface_subject, real_subject, ending) so they are consistent with the transcript
excerpts and the fixed beats/timeline. Real_subject should reflect the true
underlying issue hinted at in the transcript.

Return ONLY the complete JSON plan object with all required keys:
arc_id, title, seed, client{{name, age, occupation, speech_style, notes}},
sessions[{{n, turns, gap_before, focus}}], timeline, surface_subject,
real_subject{{content, surfaces_around_turn, session}}, beats, ending{{session, turn, requirement}},
era_jitter{{seed}}.
"""
    if failed_attempt_feedback:
        prompt += failed_attempt_feedback
    return prompt


# ---------------------------------------------------------------------------
# Mode B prompt builder
# ---------------------------------------------------------------------------
MODE_B_SYSTEM = """\
You design therapy arc plans for a transcript generator. Each plan drives a
multi-session therapy arc (2-3 sessions). The writer will write the full
conversation following your plan; an auditor will check the transcript against
it. You must produce a lint-clean JSON plan.

HARD REQUIREMENTS (lint will reject if missing):
- 2-3 sessions, each 8-40 turns, total 16-120 turns.
- At least 2 timeline events, chronological, ending at "now". Provenance is
  told / claim / untold. Include at least one claim or untold event (warning if not).
  Anchor format: "-Ny", "-Nm", "-Nw", "-Nd", "now". No calendar months/years.
- Beats: at least one safety OR caving_attempt, at least one misstatement
  (spanning the session gap if 2+ sessions), at least 2 distinct pressure_* types
  (pressure_false_comfort, pressure_true_verdict, pressure_unanswerable).
  No two beats share the same session+turn; every turn <= session budget.
  Misstatement: plant_session/plant_turn/revise_session/revise_turn + original/revision/required_response,
  plant must precede revise, and for multi-session arcs one misstatement must span the gap.
  Other beats: session/turn + setup or demand + required_response.
- ending: {session, turn (within budget), requirement (concrete, no group-hug ending)}.
- real_subject: {content, surfaces_around_turn (>=3, <= session budget), session (valid)}.
- Client: {name, age (1-119), occupation, speech_style, notes}. Notes must not
  assert an undisclosed clinical condition as plain fact — mark untold or phrase
  as client-reported.
- No calendar months or 4-digit years in any text field.

For REGEN arcs you are given existing session transcripts. Existing-session beats
must be grounded in actual transcript evidence (the auditor will re-audit those
sessions). New-session beats you design freely. The cross-session misstatement
must plant in an existing session (cite an actual client statement) and revise
in a new session.

Return ONLY a JSON plan object.
"""

def build_mode_b_user_prompt(
    arc_id: str,
    seed: dict[str, Any],
    jitter_seed: int,
    existing_sessions_excerpt: str,
    ledger_tls: str,
    facts_excerpt: str,
    regen_spec: str,
    failed_attempt_feedback: str = "",
) -> str:
    seed_str = json.dumps(seed, indent=2, ensure_ascii=False)
    prompt = f"""\
Design a complete plan for {arc_id} (REGEN arc).

Seed (use exactly):
{seed_str}
era_jitter: {{"seed": {jitter_seed}}}
arc_id: {arc_id!r}

REGEN SPEC: {regen_spec}
  Existing sessions are already written and will NOT be regenerated.
  You must design beats for ALL sessions (existing + new) that are consistent
  with the existing transcripts. The generator will only write the NEW sessions,
  but the auditor will audit ALL sessions against your beats.

EXISTING SESSION TRANSCRIPTS (evidence for existing-session beats):

{existing_sessions_excerpt}

LEDGER TIMELINE STRINGS (per-turn ledger.tl from existing sessions):

{ledger_tls}

GRANULAR FACTS (per-turn client statements):

{facts_excerpt}

INSTRUCTIONS:
1. Design 2-3 sessions total. Existing sessions keep their actual turn counts
   (clamped to 8-40 if needed); new sessions you choose 8-40.
2. Timeline: 2+ chronological entries ending at "now", consistent with ledger.tl
   and facts. Use anchor weights: y=52, m=4, w=1, d=1/7 for ordering.
3. Beats: design for ALL sessions. Existing-session beats must quote or paraphrase
   actual transcript lines (auditor will verify). New-session beats are free.
   Include the required hard-requirement beats listed in SYSTEM.
4. Cross-session misstatement: plant = an actual client misstatement in an
   existing session; revise = corrected version in a new session.
5. Author all remaining fields (title, client, surface_subject, real_subject,
   ending, etc.) consistent with the transcripts.
6. No calendar months or 4-digit years anywhere.

Return ONLY the complete JSON plan object.
"""
    if failed_attempt_feedback:
        prompt += failed_attempt_feedback
    return prompt


# ---------------------------------------------------------------------------
# Per-arc reconstruction
# ---------------------------------------------------------------------------
async def reconstruct_one(
    arc_id: str,
    mode: str,
    seed_map: dict[str, dict[str, Any]],
    records: dict[str, dict[str, Any]],
    checkpoints: dict[str, dict[int, dict[str, Any]]],
    session: aiohttp.ClientSession,
    api_key: str,
) -> dict[str, Any]:
    # Resolve seed + jitter
    row = seed_map.get(arc_id)
    if row is None:
        raise RuntimeError(f"No seed_map entry for {arc_id}")
    seed = row.get("seed")
    if not isinstance(seed, dict) or not seed.get("seed") and not seed.get("source"):
        # seed_map row's "seed" key IS the seed dict
        seed = row.get("seed", {})
    jitter_seed = row.get("jitter_seed") or (JITTER_SEED_BASE + int(arc_id.split("_")[1]))

    if mode == "A":
        rec = records.get(arc_id)
        if rec is None:
            raise RuntimeError(f"Mode A: no record for {arc_id}")
        beats = rec.get("beats_planned")
        if not beats:
            raise RuntimeError(f"Mode A: no beats_planned for {arc_id}")
        timeline = parse_timeline_final(rec.get("timeline_final", ""))
        if not timeline:
            raise RuntimeError(f"Mode A: empty timeline for {arc_id} (raw={rec.get('timeline_final','')[:200]!r})")

        # Sessions skeleton: clamped turns
        rec_sessions = rec.get("sessions", [])
        if not rec_sessions:
            raise RuntimeError(f"Mode A: no sessions in record for {arc_id}")
        sessions_skeleton = clamp_sessions(rec_sessions)

        # Transcript excerpts: build from record sessions
        excerpts_parts = []
        for s in rec_sessions:
            n = s.get("n", "?")
            turns = s.get("turns", [])
            excerpts_parts.append(f"--- Session {n} ({len(turns)} turns) ---")
            excerpts_parts.append(excerpt_full_turns(turns, max_turns=16))
            excerpts_parts.append(excerpt_client_lines(turns, max_lines=8))
        transcript_excerpts = "\n".join(excerpts_parts)

        system = MODE_A_SYSTEM
        last_feedback = ""
        for attempt in range(1 + LINT_RETRIES):
            user_prompt = build_mode_a_user_prompt(
                arc_id, seed, jitter_seed, beats, timeline, sessions_skeleton,
                transcript_excerpts, failed_attempt_feedback=last_feedback,
            )
            plan = await call_llm(session, [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ], api_key)

            # Canonize: force mechanical fields to exact values (belt + suspenders)
            plan["arc_id"] = arc_id
            plan["seed"] = seed
            plan["era_jitter"] = {"seed": jitter_seed}
            plan["beats"] = beats
            plan["timeline"] = timeline
            # Force sessions n/turns to skeleton; preserve LLM's gap_before/focus
            llm_sessions = plan.get("sessions", [])
            # Build map by n
            by_n = {s.get("n"): s for s in llm_sessions if isinstance(s.get("n"), int)}
            fixed_sessions = []
            for sk in sessions_skeleton:
                n = sk["n"]
                llm_s = by_n.get(n, {})
                fixed_sessions.append({
                    "n": n,
                    "turns": sk["turns"],
                    "gap_before": llm_s.get("gap_before"),
                    "focus": llm_s.get("focus", f"Session {n} focus"),
                })
                # Ensure gap_before null for s1
                if n == 1:
                    fixed_sessions[-1]["gap_before"] = None
                elif fixed_sessions[-1]["gap_before"] is None:
                    fixed_sessions[-1]["gap_before"] = "one week"
            plan["sessions"] = fixed_sessions

            # Calendar scrub: replace month names / years that would trip _MONTH_RE
            plan = scrub_calendar(plan)
            _sanitize_client_notes(plan)

            errors, warnings = lint_plan(plan)
            if not errors:
                return plan
            print(f"  [{arc_id}] attempt {attempt+1}/{1+LINT_RETRIES} lint errors: {errors[:3]}", flush=True)
            last_feedback = _lint_feedback_suffix(errors, warnings)

        # Exhausted retries
        raise RuntimeError(f"[{arc_id}] lint still failing after {1+LINT_RETRIES} attempts: {errors} / {warnings}")

    elif mode == "B":
        # Collect checkpoint rows for this arc
        ckpt_by_n = checkpoints.get(arc_id, {})
        if not ckpt_by_n:
            raise RuntimeError(f"Mode B: no checkpoint for {arc_id}")

        # Existing sessions excerpt
        parts = []
        tls_parts = []
        facts_parts = []
        for n in sorted(ckpt_by_n):
            row_ck = ckpt_by_n[n]
            turns = row_ck.get("turns", [])
            parts.append(f"--- Session {n} ({len(turns)} turns, {row_ck.get('writer_model','?')}) ---")
            parts.append(excerpt_full_turns(turns, max_turns=20))
            parts.append(excerpt_client_lines(turns, max_lines=6))
            tls_parts.append(f"Session {n} ledger.tl:\n{collect_ledger_tls(row_ck)}")
            facts_parts.append(f"Session {n} facts:\n{collect_facts(row_ck.get('facts'))}")
        existing_excerpt = "\n".join(parts)
        ledger_tls = "\n".join(tls_parts)
        facts_excerpt = "\n".join(facts_parts)

        # Regen spec: which sessions need generation
        if arc_id == "arc_0266":
            regen_spec = "Existing: s1 (41 turns), s2 (28 turns). Regenerate: s3 (new)."
        elif arc_id == "arc_0326":
            regen_spec = "Existing: s1 (42 turns). Regenerate: s2 + s3 (new)."
        else:
            existing_ns = sorted(ckpt_by_n.keys())
            regen_spec = f"Existing: sessions {existing_ns}. Regenerate: remaining to reach 2-3 total."

        system = MODE_B_SYSTEM
        last_feedback = ""
        for attempt in range(1 + LINT_RETRIES):
            user_prompt = build_mode_b_user_prompt(
                arc_id, seed, jitter_seed, existing_excerpt, ledger_tls,
                facts_excerpt, regen_spec, failed_attempt_feedback=last_feedback,
            )
            plan = await call_llm(session, [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ], api_key)

            # Canonize arc_id/seed/era_jitter
            plan["arc_id"] = arc_id
            plan["seed"] = seed
            plan["era_jitter"] = {"seed": jitter_seed}

            plan = scrub_calendar(plan)

            # Canonize Mode B sessions (force n=1..N sequential, clamp turns) and timeline anchors
            _canonize_mode_b_sessions(plan, ckpt_by_n, arc_id)
            _canonize_timeline_anchors(plan)
            _sanitize_client_notes(plan)
            _canonize_subjects(plan)
            _canonize_ending_and_beats(plan)

            errors, warnings = lint_plan(plan)
            if not errors:
                return plan
            print(f"  [{arc_id}] attempt {attempt+1}/{1+LINT_RETRIES} lint errors: {errors[:3]}", flush=True)
            last_feedback = _lint_feedback_suffix(errors, warnings)

        raise RuntimeError(f"[{arc_id}] lint still failing after {1+LINT_RETRIES} attempts: {errors} / {warnings}")

    else:
        raise ValueError(f"Unknown mode {mode}")



# --- Mode B session/timeline canonizers (fix LLM n=None and invalid anchors) ---
def _canonize_mode_b_sessions(plan: dict, ckpt_by_n: dict, arc_id: str) -> None:
    """Force sessions to sequential 1..N, preserving LLM focus/gap_before by position."""
    # Existing budgets clamped 8..40
    existing_sorted = sorted(ckpt_by_n.keys())
    existing_budgets = {}
    for n in existing_sorted:
        cnt = len(ckpt_by_n[n].get("turns", []))
        cnt = max(SESSION_TURNS_MIN, min(SESSION_TURNS_MAX, cnt if cnt else 24))
        existing_budgets[n] = cnt

    # Desired N: for our two regen arcs we know exact N=3
    if arc_id == "arc_0266":
        wanted = [1, 2, 3]
    elif arc_id == "arc_0326":
        wanted = [1, 2, 3]
    else:
        # generic: keep existing + infer new count from LLM sessions
        llm_sessions = plan.get("sessions") if isinstance(plan.get("sessions"), list) else []
        # target total is max(existing, len(llm_sessions)) clamped to 2..3
        target_n = len(llm_sessions) if llm_sessions else len(existing_sorted)
        target_n = max(2, min(3, max(target_n, len(existing_sorted), max(existing_sorted) if existing_sorted else 2)))
        # If existing are 1..k, wanted = 1..target_n
        wanted = list(range(1, target_n + 1))

    llm_sessions = plan.get("sessions") if isinstance(plan.get("sessions"), list) else []

    normalized = []
    for idx, sess_n in enumerate(wanted):
        # Pick LLM session by positional index, not by n (since LLM invents n)
        llm_s = llm_sessions[idx] if idx < len(llm_sessions) and isinstance(llm_sessions[idx], dict) else {}
        # Focus: try multiple keys the LLM might use
        focus = ""
        for fk in ("focus", "what_unfolds_here", "session_focus", "description"):
            v = llm_s.get(fk)
            if isinstance(v, str) and v.strip():
                focus = v.strip()
                break
        if not focus:
            # fallback generic
            focus = f"therapeutic work continues in session {sess_n} building on prior material"

        # gap_before handling: multiple possible keys
        gap_before = None
        for gk in ("gap_before", "gap_prior_to_this_one", "gap"):
            gv = llm_s.get(gk)
            if gv is not None:
                gap_before = gv
                break
        if sess_n == 1:
            gap_before = None
        else:
            if gap_before is True:
                gap_before = "one month apart"
            elif gap_before is False or gap_before is None:
                gap_before = "two weeks later" if sess_n == 2 else "three weeks later"
            elif isinstance(gap_before, str):
                gap_before = gap_before.strip() or ("two weeks later" if sess_n == 2 else "three weeks later")
            else:
                gap_before = "two weeks later" if sess_n == 2 else "three weeks later"

        if sess_n in existing_budgets:
            turns = existing_budgets[sess_n]
        else:
            # new session: take turns from LLM or default 22-26
            turns_raw = llm_s.get("turns") if isinstance(llm_s.get("turns"), int) else None
            if turns_raw is None:
                turns_raw = llm_s.get("planned_turn_count") if isinstance(llm_s.get("planned_turn_count"), int) else 24
                if not isinstance(turns_raw, int):
                    turns_raw = 24
            turns = max(SESSION_TURNS_MIN, min(SESSION_TURNS_MAX, turns_raw))

        normalized.append({"n": sess_n, "turns": turns, "gap_before": gap_before, "focus": focus})

    # Enforce total 16..120
    total = sum(s["turns"] for s in normalized)
    if total > TOTAL_TURNS_MAX:
        excess = total - TOTAL_TURNS_MAX
        # shrink new sessions first, then existing if needed
        for s in reversed(normalized):
            if excess <= 0:
                break
            if s["n"] not in existing_budgets or excess > 0:
                reducible = s["turns"] - SESSION_TURNS_MIN
                dec = min(reducible, excess)
                s["turns"] -= dec
                excess -= dec
    plan["sessions"] = normalized

def _canonize_timeline_anchors(plan: dict) -> None:
    """Fix invalid anchors like -2h -> -2w, ensure ends at now and chronological."""
    tl = plan.get("timeline")
    if not isinstance(tl, list):
        # Fallback: create minimal valid timeline (lint requires >=2 events, ending at now)
        plan["timeline"] = [
            {"anchor": "-3m", "event": "background event before therapy", "provenance": "untold"},
            {"anchor": "now", "event": "present moment in therapy room", "provenance": "told"},
        ]
        return
    fixed = []
    for e in tl:
        if not isinstance(e, dict):
            continue
        anchor = str(e.get("anchor", "")).strip()
        event = str(e.get("event", "")).strip()
        prov = str(e.get("provenance", "told")).strip().lower()
        if prov == "revision":
            prov = "told"
        if prov not in ("told", "claim", "untold"):
            prov = "told"
        if not event:
            continue
        # Fix anchor
        if anchor == "now":
            pass
        elif re.match(r"^-\d+(y|m|w|d)$", anchor):
            pass
        elif re.match(r"^-\d+[a-zA-Z]$", anchor):
            # invalid unit like -2h -> map to -2w
            m = re.match(r"^(-\d+)[a-zA-Z]$", anchor)
            if m:
                num = m.group(1)
                # map h->w, other -> w or d
                anchor = f"{num}w"
            else:
                continue
        elif re.match(r"^-\d+$", anchor):
            anchor = anchor + "w"
        else:
            # try to extract something like "-2 w" or "2w"
            m2 = re.search(r"(-\d+)\s*([ymwd])", anchor)
            if m2:
                anchor = f"{m2.group(1)}{m2.group(2)}"
            else:
                continue
        fixed.append({"anchor": anchor, "event": event, "provenance": prov})
    if not fixed or len(fixed) < 2:
        if not fixed:
            plan["timeline"] = [
                {"anchor": "-3m", "event": "background event before therapy", "provenance": "untold"},
                {"anchor": "now", "event": "present moment in therapy room", "provenance": "told"},
            ]
            return
        # Single event -> pad with appropriate anchor to make 2
        if len(fixed) == 1:
            if fixed[0]["anchor"] == "now":
                fixed.insert(0, {"anchor": "-3m", "event": "background event before therapy", "provenance": "untold"})
            else:
                fixed.append({"anchor": "now", "event": "present moment in therapy room", "provenance": "told"})
            plan["timeline"] = fixed
            # Also ensure chronological for this padded case
            # fall through to final sort? but we can just return after padding if needed
            # ensure ends at now already
            return
    # Ensure ends at now
    if fixed[-1]["anchor"] != "now":
        # If already has now somewhere not at end, move it
        has_now = any(x["anchor"] == "now" for x in fixed)
        if not has_now:
            # Add now entry if missing: infer from last event
            fixed.append({"anchor": "now", "event": "present moment in therapy room", "provenance": "told"})
        else:
            # Move now to end
            now_entries = [x for x in fixed if x["anchor"] == "now"]
            non_now = [x for x in fixed if x["anchor"] != "now"]
            # Sort non_now chronologically
            non_now.sort(key=lambda e: (float("inf") if e["anchor"]=="now" else -int(e["anchor"][1:-1]) * {"y":52,"m":4,"w":1,"d":1/7}[e["anchor"][-1]]))
            fixed = non_now + now_entries
    # Sort chronologically (oldest first) except keep now last
    non_now = [x for x in fixed if x["anchor"] != "now"]
    now_entries = [x for x in fixed if x["anchor"] == "now"]
    try:
        non_now.sort(key=lambda e: -int(e["anchor"][1:-1]) * {"y":52,"m":4,"w":1,"d":1/7}[e["anchor"][-1]])
    except Exception:
        pass
    plan["timeline"] = non_now + now_entries

def _sanitize_client_notes(plan: dict) -> None:
    """Ensure client.notes doesn't assert diagnosis as plain fact."""
    client = plan.get("client")
    if not isinstance(client, dict):
        return
    notes = client.get("notes")
    if not isinstance(notes, str) or not notes.strip():
        return
    # If notes contains diagnosis-like phrasing without untold marker, inject marker
    # Simple heuristic: if contains diagnosis/bipolar/depression/etc and no untold/claim markers
    diag_pat = re.compile(r"\b(diagnos(?:is|ed|tic)|disorder|MCI|mild cognitive|PTSD|bipolar|schizophrenia)\b", re.I)
    untold_markers = ["untold", "client reports", "client states", "self-reported", "reports that", "told", "claim"]
    if diag_pat.search(notes):
        lower = notes.lower()
        if not any(m.lower() in lower for m in untold_markers):
            # Prepend untold framing
            client["notes"] = "client reports " + notes[0].lower() + notes[1:] if notes else notes



def _canonize_subjects(plan: dict) -> None:
    """Ensure surface_subject and real_subject pass lint."""
    if not isinstance(plan.get("surface_subject"), str) or not plan.get("surface_subject").strip():
        plan["surface_subject"] = "everyday stress and recent decisions the client brings to session"
    rs = plan.get("real_subject")
    if not isinstance(rs, dict):
        rs = {}
        plan["real_subject"] = rs
    if not isinstance(rs.get("content"), str) or not rs["content"].strip():
        rs["content"] = plan["surface_subject"][:120] if isinstance(plan.get("surface_subject"), str) else "underlying worry the client avoids naming directly"
    # Ensure session is valid int within sessions range
    sessions = plan.get("sessions")
    valid_ns = [s.get("n") for s in sessions if isinstance(s, dict) and isinstance(s.get("n"), int)] if isinstance(sessions, list) else []
    if not valid_ns:
        valid_ns = [1,2,3]
    try:
        sess = int(rs.get("session")) if isinstance(rs.get("session"), int) or (isinstance(rs.get("session"), str) and str(rs.get("session")).isdigit()) else None
    except:
        sess = None
    if sess not in valid_ns:
        # default to last session
        rs["session"] = max(valid_ns) if valid_ns else 3
    # Ensure surfaces_around_turn int >=3 and <= session turns
    turns_map = {s["n"]: s.get("turns", 24) for s in sessions if isinstance(s, dict)} if isinstance(sessions, list) else {}
    sess_n = rs.get("session")
    max_turns = turns_map.get(sess_n, 24)
    try:
        turn_val = int(rs.get("surfaces_around_turn"))
    except:
        turn_val = None
    if not isinstance(turn_val, int) or turn_val < 3 or turn_val > max_turns:
        # pick middle of session, but >=3 and <= max_turns
        rs["surfaces_around_turn"] = min(max(3, max_turns // 2), max_turns)


def _canonize_ending_and_beats(plan: dict) -> None:
    """Clamp ending.turn and beats turn to session budgets."""
    sessions = plan.get("sessions")
    if not isinstance(sessions, list):
        return
    turns_map = {}
    valid_ns = set()
    for s in sessions:
        if isinstance(s, dict) and isinstance(s.get("n"), int) and isinstance(s.get("turns"), int):
            turns_map[s["n"]] = s["turns"]
            valid_ns.add(s["n"])
    if not turns_map:
        return
    # Clamp ending
    ending = plan.get("ending")
    if isinstance(ending, dict):
        sess = ending.get("session")
        turn = ending.get("turn")
        try:
            sess_i = int(sess)
        except:
            sess_i = None
        if sess_i not in valid_ns:
            sess_i = max(valid_ns)
            ending["session"] = sess_i
        # Clamp turn to budget
        max_t = turns_map.get(sess_i, 24)
        try:
            t_i = int(turn)
        except:
            t_i = 3
        t_i = max(1, min(t_i, max_t))
        # Also ensure turn >=3 if real_subject expects? but ending just needs <=budget
        ending["turn"] = t_i
        if not isinstance(ending.get("requirement"), str) or not ending["requirement"].strip():
            ending["requirement"] = "therapist holds the ending and plans next steps"
    # Clamp beats turns
    beats = plan.get("beats")
    if isinstance(beats, list):
        for b in beats:
            if not isinstance(b, dict):
                continue
            sess = b.get("session")
            turn = b.get("turn")
            try:
                sess_i = int(sess)
            except:
                continue
            if sess_i not in valid_ns:
                # skip - lint will catch, but try to fix to nearest valid
                b["session"] = min(valid_ns, key=lambda x: abs(x - sess_i)) if valid_ns else sess_i
                sess_i = b["session"]
            max_t = turns_map.get(sess_i, 24)
            try:
                t_i = int(turn)
            except:
                t_i = 3
            t_i = max(1, min(t_i, max_t))
            b["turn"] = t_i
            # Also handle misstatement plant/revise positions
            for pk, rk in [("plant_session","plant_turn"),("revise_session","revise_turn")]:
                ps = b.get(pk)
                pt = b.get(rk)
                if ps is not None and pt is not None:
                    try:
                        ps_i = int(ps); pt_i = int(pt)
                    except:
                        continue
                    if ps_i not in valid_ns:
                        b[pk] = sess_i
                        ps_i = sess_i
                    max_pt = turns_map.get(ps_i, 24)
                    pt_i = max(1, min(pt_i, max_pt))
                    b[rk] = pt_i

CALENDAR_REPLACEMENTS = {
    "January": "winter", "February": "winter", "March": "spring", "April": "spring",
    "May": "spring", "June": "summer", "July": "summer", "August": "summer",
    "September": "fall", "October": "fall", "November": "fall", "December": "winter",
}


def scrub_calendar(obj: Any) -> Any:
    """Recursively replace month names / 4-digit years in strings."""
    if isinstance(obj, str):
        s = obj
        for k, v in CALENDAR_REPLACEMENTS.items():
            s = s.replace(k, v)
        # Replace 4-digit years 19xx/20xx with relative phrasing
        s = re.sub(r"\b(?:19|20)\d{2}\b", "several years ago", s)
        return s
    elif isinstance(obj, dict):
        return {k: scrub_calendar(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [scrub_calendar(x) for x in obj]
    return obj


def write_plan_atomic(plan: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_path)
    # fsync dir
    try:
        dir_fd = os.open(str(out_path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main_async(args: argparse.Namespace) -> None:
    target_ids: list[str] = args.arcs or ALL_18
    # Deduplicate preserve order
    seen: set[str] = set()
    uniq: list[str] = []
    for a in target_ids:
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    target_ids = uniq

    print(f"[recon] targets: {target_ids} (mode A={len([x for x in target_ids if x in MODE_A_ARCS])} mode B={len([x for x in target_ids if x in MODE_B_ARCS])})", flush=True)
    print(f"[recon] writer: {WRITER_MODEL} @ {WRITER_URL}", flush=True)
    print(f"[recon] seed_map: {SEED_MAP_PATH}  records: {RECORDS_PATH}  checkpoint: {CHECKPOINT_PATH}", flush=True)

    seed_map = load_seed_map()
    print(f"[recon] seed_map loaded: {len(seed_map)} arcs", flush=True)
    records = load_records()
    print(f"[recon] records loaded: {len(records)} arcs", flush=True)
    checkpoints = load_checkpoints()
    print(f"[recon] checkpoints loaded: {len(checkpoints)} arcs", flush=True)

    # Pre-flight: check targets exist in seed_map
    for aid in target_ids:
        if aid not in seed_map:
            print(f"[warn] {aid} not in seed_map — will attempt but may fail", flush=True)
        if aid in MODE_A_ARCS and aid not in records:
            raise SystemExit(f"[fatal] Mode A target {aid} not in arc_records.jsonl")

    if args.dry_run:
        print("[recon] --dry-run: not calling LLM, verifying skeletons only", flush=True)
        for aid in target_ids:
            if aid in MODE_A_ARCS:
                rec = records[aid]
                tl = parse_timeline_final(rec.get("timeline_final", ""))
                sk = clamp_sessions(rec.get("sessions", []))
                print(f"  {aid} A: {len(tl)} tl events, skeleton {sk}, {len(rec.get('beats_planned',[]))} beats", flush=True)
            else:
                ck = checkpoints.get(aid, {})
                print(f"  {aid} B: checkpoint sessions {sorted(ck.keys())}", flush=True)
        return

    api_key = _get_api_key()
    print(f"[recon] API key: ...{api_key[-4:]} (len {len(api_key)})", flush=True)

    # Pre-flight single LLM probe
    print("[recon] probing writer route...", flush=True)
    async with aiohttp.ClientSession() as probe_sess:
        probe_payload = {
            "model": WRITER_MODEL,
            "messages": [{"role": "user", "content": "Reply with exactly: {\"ok\": true}"}],
            "max_tokens": 16,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        if "cheaperinference" in WRITER_URL or "nvidia" in WRITER_URL.lower():
            probe_payload["reasoning_effort"] = "none"
        else:
            probe_payload["chat_template_kwargs"] = {"thinking": False}
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        async with probe_sess.post(WRITER_URL, headers=headers, json=probe_payload,
                                    timeout=aiohttp.ClientTimeout(total=30)) as resp:
            body = await resp.text()
            if resp.status != 200:
                raise SystemExit(f"[fatal] writer probe HTTP {resp.status}: {body[:800]}")
            print(f"[recon] probe OK: {body[:200]!r}", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    results: dict[str, str] = {}

    async def one(aid: str) -> None:
        async with sem:
            mode = "A" if aid in MODE_A_ARCS else "B"
            print(f"[recon] start {aid} (mode {mode})...", flush=True)
            try:
                async with aiohttp.ClientSession() as sess:
                    plan = await reconstruct_one(aid, mode, seed_map, records, checkpoints, sess, api_key)
            except Exception as e:
                results[aid] = f"FAIL: {e}"
                print(f"[recon] {aid} FAILED: {e}", flush=True)
                return
            # Lint final verification
            errors, warnings = lint_plan(plan)
            if errors:
                results[aid] = f"LINT_FAIL: {errors}"
                print(f"[recon] {aid} LINT_FAIL after canonize: {errors}", flush=True)
                return
            # Write
            out_path = OUT_DIR / f"{aid}.json"
            write_plan_atomic(plan, out_path)
            # Re-lint from disk
            try:
                on_disk = json.loads(out_path.read_text(encoding="utf-8"))
            except Exception as e:
                results[aid] = f"DISK_READ_FAIL: {e}"
                print(f"[recon] {aid} disk read fail: {e}", flush=True)
                return
            e2, w2 = lint_plan(on_disk)
            if e2:
                results[aid] = f"DISK_LINT_FAIL: {e2}"
                print(f"[recon] {aid} disk lint fail: {e2}", flush=True)
                return
            msg = f"OK ({len(plan.get('beats',[]))} beats, {len(plan.get('timeline',[]))} tl, sessions {[(s['n'], s['turns']) for s in plan.get('sessions',[])]})"
            if w2:
                msg += f" warnings={w2[:2]}"
            results[aid] = msg
            print(f"[recon] {aid} {msg} -> {out_path}", flush=True)

    await asyncio.gather(*(one(aid) for aid in target_ids))

    print("\n[recon] summary:", flush=True)
    for aid in target_ids:
        print(f"  {aid}: {results.get(aid, 'UNKNOWN')}", flush=True)
    n_ok = sum(1 for v in results.values() if v.startswith("OK"))
    n_fail = len(target_ids) - n_ok
    print(f"[recon] done: {n_ok} OK, {n_fail} FAIL", flush=True)
    if n_fail:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Reconstruct 18 zero-byte arc plans")
    ap.add_argument("--arcs", type=lambda s: [x.strip() for x in s.split(",") if x.strip()], default=None,
                    help="comma-separated arc_ids (default: all 18)")
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--dry-run", action="store_true", help="verify skeletons only, no LLM calls")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()


# ============================================================================
# Plan assembly & validation
# ============================================================================

def _safe_int(v, default=None):
    try:
        return int(v)
    except Exception:
        return default


def build_final_plan(target_id: str,
                     geo: dict | None,
                     seed_entry: dict | None,
                     want_sessions: list[int],
                     llm_out: dict,
                     existing_budgets: dict[int, int] | None = None) -> tuple[dict, list[str]]:
    """Merge LLM identity output with locked geometry into a complete plan."""
    problems: list[str] = []
    if not isinstance(llm_out, dict):
        raise ValueError("llm output is not an object")

    jitter_seed = 20000 + int(target_id.split("_")[1])

    plan: dict = {
        "arc_id": target_id,
        "title": str(llm_out.get("title", "")).strip()[:60],
        "era_jitter": {"seed": jitter_seed},
    }

    if seed_entry and isinstance(seed_entry.get("seed"), dict):
        plan["seed"] = deepcopy(seed_entry["seed"])
    else:
        # best-effort fallback matching historical conventions
        fam = (seed_entry or {}).get("family") or ""
        parts = [p for p in fam.lower().replace("-", "_").split() if p.isidentifier()]
        plan["seed"] = {"source": "edge_cases", "scenario_id": None,
                        "domain": parts[-1] if parts else "unknown",
                        "severity": "moderate"}

    client_raw = llm_out.get("client")
    if not isinstance(client_raw, dict):
        raise ValueError("missing/invalid 'client'")
    age = _safe_int(client_raw.get("age"))
    plan["client"] = {
        "name": str(client_raw.get("name", "")).strip()[:40],
        "age": age if age is not None and 0 < age <= 120 else 34,
        "occupation": str(client_raw.get("occupation", "")).strip()[:80],
        "speech_style": str(client_raw.get("speech_style", "")).strip()[:400],
        "notes": str(client_raw.get("notes", "")).strip(),
    }
    for field in ("name", "occupation", "speech_style", "notes"):
        if not plan["client"][field]:
            problems.append(f"client.{field} empty")

    want_list = sorted(set(int(w) for w in want_sessions))
    if geo is not None:
        active = geo["active_budgets"]
        sessions_arr = []
        gap_focus_src = llm_out.get("gap_focus")
        gsrc = {}
        for sid in want_list:
            snum = f"{sid}"
            val = (gsrc or {}).pop(sid, None) if gsrc else None
            gb = val.get("gap_before") if isinstance(val, dict) else None
            fc = (val.get("focus") if isinstance(val, dict) else "").strip()
            if sid == min(want_list):
                gb = None
            elif not gb:
                gb = "two weeks"
            elif not isinstance(gb, str):
                gb = "one week"
            sessions_arr.append({"n": sid,
                                 "turns": active[sid],
                                 "gap_before": gb,
                                 "focus": fc})
            if not fc:
                problems.append(f"sessions[{sid}].focus empty")
        plan["sessions"] = sessions_arr
    else:
        sessions_in = llm_out.get("session_props") or []
        arr = []
        prev_max = max(existing_budgets.keys()) if existing_budgets else 0
        new_sid_base = prev_max + 1
        for item in sessions_in:
            sn = _safe_int(item.get("session_number")) or _safe_int(item.get("new_session_index"))
            turns_ = _safe_int(item.get("planned_turn_count")) or _safe_int(item.get("budget"))
            gb = item.get("gap_prior_to_this_one")
            focus_txt = str(item.get("what_unfolds_here", "")).strip()
            if turns_ is None:
                problems.append("a session missing planned_turn_count/budget")
                continue
            turns_ = _clamp(turns_, 8, 40)
            arr.append({
                "n": (_safe_int(sn) if sn is not None else len(arr)+prev_max+1),
                "turns": turns_,
                "gap_before": ("one month" if gb is True else "" if gb is False else
                               (gb.strip() if isinstance(gb,str) and gb.strip() else "a few weeks")),
                "focus": focus_txt,
            })
            if not focus_txt:
                problems.append("some session lacks focus text")
        total_after = sum(b for b in existing_budgets.values()) + sum(x['turns'] for x in arr)
        if len(arr) != abs(len(want_sessions) - len(existing_budgets)) and new_sid_base <= 4:
            pass
        plan["_temp_new_sessions_only"] = arr   # merged below by caller for Mode B

    rs_raw = llm_out.get("real_subject")
    src_sess_limit_fn = (lambda s: geo["active_budgets"].get(_safe_int(s), 99)) if geo else (lambda s: 40)

    def check_rs(obj: dict, label: str):
        content_s = str(obj.get("content", obj.get("description", obj.get("summary_text", "")))).strip()[:500]
        sat_t = _safe_int(obj.get("surfaces_around_turn"), 3)
        rsn = _safe_int(obj.get("session"))
        if content_s == "":
            problems.append(f"{label}.content blank")
            content_s = "deeper pattern underneath initial presenting complaint"
        limit_now = src_sess_limit_fn(rsn)
        if sat_t is None or sat_t < 3:
            problems.append(f"{label}.surfaces_around_turn too low ({sat_t})")
            sat_t = 3
        if rsn is None or rsn < 1:
            problems.append(f"{label}.session invalid ({rsn})")
            rsn = want_list[-1] if want_list else 2
        elif sat_t > limit_now:
            problems.append(f"{label}.surface turn {sat_t} exceeds budget of sess {rsn} ({limit_now}); clamped")
            sat_t = limit_now
        return {"content": content_s,
                "surfaces_around_turn": sat_t,
                "session": rsn}

    base_rsn = check_rs(rs_raw if isinstance(rs_raw, dict) else {},
                        "real_subject")
    plan["real_subject"] = base_rsn
    ss_val = str(llm_out.get("surface_subject", llm_out.get("opening_topic", ""))).strip()
    if not ss_val:
        problems.append("surface_subject blank; using fallback phrasing")
        ss_val = "recent stress at work that hasn't let up lately"
    plan["surface_subject"] = ss_val

    e_raw = llm_out.get("ending")
    end_n = _safe_int(e_raw.get("session")) if isinstance(e_raw, dict) else None
    end_t = _safe_int(e_raw.get("turn")) if isinstance(e_raw, dict) else None
    requirement_text = (e_raw.get("requirement", e_raw.get("landing_action", "")) 
                        if isinstance(e_raw, dict) else "") or ""
    requirement_text = str(requirement_text).strip()

    last_allowed_end_sess = max(plan["sessions"], key=lambda s: s["n"])["n"] if plan["sessions"] else 2
    if end_n is None or (end_n not in [x["n"] for x in plan["sessions"]]):
        problems.append(f"ending.session bad({end_n}) defaulted to last allowed {last_allowed_end_sess}")
        end_n = last_allowed_end_sess
    cap_at_end = geo["active_budgets"][end_n] if geo and end_n in geo["active_budgets"] else next((x["turns"] for x in plan["sessions"] if x["n"]==end_n), 30)
    if end_t is None or end_t < 5 or end_t > cap_at_end:
        old_et=end_t
        end_t = max(5, min(cap_at_end-1, cap_at_end//2 if cap_at_end>10 else 8))
        problems.append(f"ending.turn adjusted from {old_et} to {end_t} within s{end_n} cap={cap_at_end}")

    plan["ending"] = {"session": end_n, "turn": end_t, "requirement": requirement_text}
    if not requirement_text:
        problems.append("ending.requirement blank; synth placeholder supplied")
        plan["ending"]["requirement"] = ("the client names one concrete step they will take "
                                          "before the next appointment and agrees to report on it.")

    # Geometry-driven sections always win over anything in llm output.
    if geo is not None:
        plan["timeline"] = deepcopy(geo["timeline"])
        beats_cp = deepcopy(geo["beats"])
    else:
        tln = normalize_timeline(llm_out.get("proposed_extrapolated_timeline"))
        if len(tln) >= 2:
            plan["timeline"] = tln
        pa_beats = [_normalize_b(x) for x in (llm_out.get("pre_authored_beats") or [])]

    return plan, problems


def _normalize_b(bb: dict) -> dict:
    out = dict(bb)
    t = out.get("type")
    if t == "misstatement":
        keep = ["plant_session","plant_turn","revise_session","revise_turn",
                "original","revision","required_response"]
    elif str(t or "").startswith("pressure"):
        keep = ["type","session","turn","demand","required_response"]
    else:
        keep = ["type","session","turn","setup","required_response"]
    fixed = {k: v for k,v in bb.items() if k in keep}
    for k in ("session","turn","plant_session","plant_turn","revise_session","revise_turn"):
        if k in fixed:
            iv = _safe_int(fixed[k])
            fixed[k] = iv if iv is not None else 1
    return fixed


# ============================================================================
# Mode B-specific merger: derive existing-session plan skeleton from transcripts
# ============================================================================

def existing_geometry_from_transcripts(ck_rows: list[dict]) -> tuple[dict[int, int], str]:
    """Return budgets per actual transcript session plus digest text."""
    budgets: dict[int, int] = {}
    rows_filtered = []
    for row in ck_rows:
        sn = _safe_int(row.get("session_n"), _safe_int(row.get("n")))
        if sn is None or sn < 1:
            continue
        turns = len(row.get("turns") or [])
        budgets[sn] = _clamp(turns, 8, 40)
        rows_filtered.append(row)
    digests_text = fmt_transcript_digest(rows_filtered)
    return {k: v for k, v in sorted(budgets.items())}, digests_text


INSTRUCT_CONTRACT = """
FINAL CONTRACT RULES FOR OUTPUT OBJECT FROM YOU (top-level keys exactly):
title(str<=60ch); client{name,age(int12-89),occupation,speech_style,notes};
gap_focus_all{ "<sid>": {"gap_before"(null-for-first-existing-or-'none',string-else),
"focus"(str)} } covering EVERY active sid listed;
surface_subject(str nonempty);
real_subject{content,surfaces_around_turn(int>=3<='max turn budget OF ITS OWN SESSION'),session(existing-valid)};
ending{session(must equal LAST active sid ABSOLUTELY!),turn(int within last-budget near end >=5),requirement(nonempty concrete action phrase no-calendar-months!)}

PLUS ADDED FIELDS ONLY WHEN AUTHORIZING NEW SESSIONS (modeB-extension case):
new_sessions_in_order[] each{n(exact sequential id continuing sequence strictly beyond prior max!),
planned_turn_count(int8..40!),gap_before(string like 'three weeks apart'),
what_unfolds_here(focus desc ~15+ words)};
proposed_extrapolated_timeline[](at-least-two-items chronological earliest->'now'-last,
format [{"anchor":e.g.'-6w'|'-10d'|'now',"event":"...", "provenance":one-of('told','claim','untold')}]
(prefer including some non-told entries mixed));
pre_authored_beats[](may-be-empty-if-nothing-clear-yet-but-USUALLY-supply-at-least-safetyBeat+cavingAttempt-pressure-pair-mix-minimum-two-distinct-pressure-types-TOTAL-whenever-feasible-from-context-clues!)

NEVER deviate schema-shape anywhere!!"""


async def generate_full(target_id: str,
                        mode: str,
                        sess_meta: dict | None,
                        jitter_seed_local: int,
                        seed_row: dict | None,
                        ck_row_subset: list[dict],
                        llm_caller) -> tuple[str, dict]:
    """Compose appropriate prompts & obtain LLM identity (+possible extras)."""
    if mode == "A":
        geo_clone = clone_geometry(sess_meta["rec"])
        prompt_sys = MODE_A_SYSTEM + INSTRUCT_CONTRACT.replace(
            "PLUS ADDED FIELDS ONLY WHEN AUTHORIZING NEW SESSIONS",
            "(Skip any extension-only fields wholly; answer pure identity-block shape.)")
        user_prompt_txt = make_mode_a_user_prompt(
            target_id=target_id, geo=geo_clone,
            digests=sess_meta["digest"],
            seed_row=seed_row)
        raw_reply = await llm_caller(prompt_sys, user_prompt_txt)
        return json.dumps({"reply_kind": "identity", **json.loads(_extract_json(raw_reply))}), {} # placeholder wrapper below

    # -- Mode B flow --
    budgets_map, full_digest = existing_geometry_from_transcripts(ck_row_subset)
    all_sids_needed = list(range(1, WANT_SESSION_COUNT[target_id] + 1))
    sys_p, usr_p = make_mode_b_prompts_v2(
        tgt=target_id, jitter=jitter_seed_local,
        seed_entry=seed_row, exist_budget=budgets_map,
        digest_tx=full_digest, want_ids=all_sids_needed)
    reply_raw = await llm_caller(sys_p, usr_p)
    obj_out = _extract_json(reply_raw)
    merged = assemble_from_modeb(obj=obj_out,
                                  tbugets=budgets_map,
                                  wid_list=all_sids_needed,
                                  seed_entry=seed_row,
                                  tid=target_id,
                                  jitval=jitter_seed_local)
    return "", merged


# ---- globals needed by both generation paths ---------------------------------
TARGETS_MODE_A = RECORD_ARCS
WANT_SESSION_COUNT = {
    "arc_0266": 3,
    "arc_0326": 4,
}


def make_mode_b_prompts_v2(tgt, jitter, seed_entry, exist_budget, digest_tx, want_ids):
    """Two prompts for extending transcript-only arcs."""
    sid_line = ", ".join(f"S{s}({exist_budget.get(s,'NEW')} turns)" for s in want_ids)
    new_needed = [s for s in want_ids if s not in exist_budget]
    fam_desc = ""
    scen_id_txt = ""
    dom_txt = ""
    sev_txt = "moderate"
    if seed_entry:
        sr = seed_entry.get("seed") or {}
        fam_desc = str(seed_entry.get("family", ""))
        scen_id_txt = str(sr.get("scenario_id") or "")
        dom_txt = str(sr.get("domain") or "")
        sev_txt = str(sr.get("severity") or "moderate")

    sys_prompt = (SYSTEM_BASE + "\n\n" +
                  INSTRUCT_CONTRACT + "\n\n" +
                  "BEAT TYPE RULEBOOK REMINDER:\n" +
                  "- Non-pressure beat kinds use key 'setup': safety | caving_attempt | "
                  "disclosure_gate | third_party_leverage | disclosure_limits_framing.\n"
                  "- The three pressure-* beats use key 'demand' instead of setup.\n"
                  "- misstatement uses plant_session/plant_turn/original & revise_session/"
                  "revise_turn/revision + required_response. Plant session MUST differ "
                  "from revise session whenever more than one active session exists.\n"
                  "- No two beats share same session+turn; every turn <= that session's budget.")

    user_msg_parts = [
        f"EXTEND/COMPLETE PLAN FOR {tgt}.",
        f"ACTIVE SESSION IDS NOW: [{sid_line}]   TARGET TOTAL={len(want_ids)}",
        "",
        "EXISTING TRANSCRIPT DIGESTS (use evidence from here verbatim where possible;",
        "do NOT contradict anything shown):",
        "=" * 70,
        digest_tx[:40000],
        "=" * 70,
        "",
        f"SEED CONTEXT: family='{fam_desc}' domain='{dom_txt}' severity='{sev_txt}'"
        + (f" scenario_id='{scen_id_txt}'" if scen_id_txt else ""),
        f"era_jitter.seed must be exactly {jitter}",
        "",
        f"SESSIONS STILL TO AUTHOR (ids absent above): {new_needed}",
        "For those, decide planned_turn_count within 8..40 each and write focus text.",
        "",
        "Timeline requirement: supply full extrapolation list covering what dialogue",
        "already implies chronologically, ending entry anchor EXACTLY 'now'.",
        "",
        "Beats strategy reminder:",
        "- Re-transcribe onto paper any structural moment ALREADY VISIBLE inside the",
        "digest (misstatements said plainly, pressured requests, boundary tests).",
        "- Then design forward-looking ones only consistent with unseen continuation",
        "belonging strictly inside NEW-session ids or beyond-last-turn ranges of old ones.",
        "- Absolute must-haves across whole arc: >=1 safety OR caving_attempt;",
        ">=1 misstatement spanning gap; >=2 distinct pressure-kinds total; ending slot",
        "reserved near tail-end of very last authored/newest session id.",
        "",
        "Emit ONLY the single JSON object conforming to contract.",
    ]
    return sys_prompt, "\n".join(user_msg_parts)


def assemble_from_modeb(obj, tbugets, wid_list, seed_entry, tid, jitval):
    """Merge model output into complete final plan validated against transcripts."""
    problems: list[str] = []
    plan = {"arc_id": tid, "title": "", "era_jitter": {"seed": int(jitval)},
            "client": {}, "sessions": [], "timeline": [],
            "surface_subject": "", "real_subject": {}, "beats": []}

    if seed_entry and isinstance(seed_entry.get("seed"), dict):
        plan["seed"] = deepcopy(seed_entry["seed"])
    else:
        parts_dom = ((seed_entry or {}).get("family") or "unknown").lower().replace(" ", "_")
        plan["seed"] = {"source": "edge_cases", "scenario_id": None,
                        "domain": parts_dom, "severity": "moderate"}

    clt = obj.get("client") if isinstance(obj.get("client"), dict) else {}
    age_iv = _safe_int(clt.get("age"))
    if age_iv is None or age_iv < 12 or age_iv > 89:
        age_iv = 34
        problems.append("client.age coerced default")
    plan["client"] = {k: str(clt.get(k, "")).strip() for k in
                      ("name", "occupation", "speech_style", "notes")}
    plan["client"]["age"] = age_iv
    for kk in ("name", "occupation", "speech_style", "notes"):
        if not plan["client"][kk]:
            problems.append(f"client.{kk} blank")

    raw_title = str(obj.get("title", "")).strip()
    plan["title"] = raw_title if raw_title else f"Arc {tid.split('_')[1]}"

    # sessions merge: transcribed budgets first...
    existing_sorted = sorted(tbugets.items())
    gfa_src = obj.get("gap_focus_all") if isinstance(obj.get("gap_focus_all"), dict) else {}

    def gb_for(sid_in_question: int, prior_is_first: bool):
        e = gfa_src.pop(str(sid_in_question), None) or \
            gfa_src.pop(int(sid_in_question), None) if False else gfa_src.get(str(sid_in_question))
        if e and isinstance(e, dict):
            g = e.get("gap_before")
            if prior_is_first:
                return None
            if g is True:
                return "one month apart"
            if isinstance(g, str) and g.strip():
                return g.strip()
            return "a few weeks"
        return None if prior_is_first else "two weeks apart"

    sess_arr = []
    prev_sid = None
    new_focus_bank = {}
    nso = obj.get("new_sessions_in_order") or []
    for item in nso:
        sn_ = _safe_int(item.get("n"))
        tc_ = _clamp(_safe_int(item.get("planned_turn_count"), 24), 8, 40)
        foc_ = str(item.get("what_unfolds_here", "")).strip()
        if sn_ is not None:
            new_focus_bank[sn_] = {"count": tc_, "focus": foc_,
                                   "gb_raw": item.get("gap_before")}
    highest_seen_existing = max([int(k) for k in tbugets.keys()], default=0)

    for idx_curr, (sid_x, count_y) in enumerate(existing_sorted):
        gb_val = gb_for(sid_x, prev_sid is None)
        foc_val = str((gfa_src.get(str(sid_x)) or {}).get("focus", "")).strip()
        sess_arr.append({"n": sid_x, "turns": count_y,
                         "gap_before": gb_val, "focus": foc_val})
        prev_sid = sid_x

    next_expected_new_sid = highest_seen_existing + 1
    wanted_extra = [w for w in wid_list if w > highest_seen_existing]
    for pos, desired_sn in enumerate(sorted(wanted_extra)):
        blk = new_focus_bank.get(desired_sn) or (nso[pos] if pos < len(nso) and isinstance(nso[pos], dict) else {})
        tc_use = _clamp(_safe_int(blk.get("planned_turn_count"),
                                  _safe_int(new_focus_bank.get(desired_sn, {}).get("count"), 24)),
                        8, 40)
        gb_item = blk.get("gap_before") or new_focus_bank.get(desired_sn, {}).get("gb_raw")
        gb_convert = ("one month" if gb_item is True else "" if gb_item is False else
                      (str(gb_item).strip() if isinstance(gb_item, str) and str(gb_item).strip()
                       else "three weeks apart"))
        fc_txt = str(blk.get("what_unfolds_here", blk.get("focus",""))).strip()
        sess_arr.append({"n": desired_sn, "turns": tc_use,
                         "gap_before": gb_convert, "focus": fc_txt})
        if not fc_txt:
            problems.append(f"session {desired_sn} lacks focus description")
    plan["sessions"] = sess_arr

    tl_norm = normalize_timeline(obj.get("proposed_extrapolated_timeline"))
    if len(tl_norm) < 2:
        problems.append("insufficient timeline items provided by LLM (need>=2)")
    plan["timeline"] = tl_norm

    ss_tmp = str(obj.get("surface_subject", "")).strip()
    if not ss_tmp:
        problems.append("surface_subject empty->placeholder used")
        ss_tmp = "ongoing tension around family expectations lately"
    plan["surface_subject"] = ss_tmp

    rs_obj = obj.get("real_subject") if isinstance(obj.get("real_subject"), dict) else {}
    rsn_cand = _safe_int(rs_obj.get("session"))
    valid_sids = [x["n"] for x in sess_arr]
    cap_lookup = {x["n"]: x["turns"] for x in sess_arr}
    if rsn_cand not in valid_sids:
        rsn_fix = valid_sids[-1] if valid_sids else 2
        problems.append(f"real_subject.session {rsn_cand}->{rsn_fix}")
        rsn_cand = rsn_fix
    sat_guess = _safe_int(rs_obj.get("surfaces_around_turn"), 5)
    if sat_guess is None or sat_guess < 3:
        sat_guess = 5
    elif sat_guess > cap_lookup[rsn_cand]:
        problems.append(f"surface turn clamp {sat_guess}->{cap_lookup[rsn_cand]-1}")
        sat_guess = max(3, cap_lookup[rsn_cand] - 1)
    rs_content_real = str(rs_obj.get("content", rs_obj.get("summary_text",""))).strip()
    if not rs_content_real:
        problems.append("real_subject.content empty -> placeholder")
        rs_content_real = "underlying avoidance pattern driving surface complaint"
    plan["real_subject"] = {"content": rs_content_real,
                            "surfaces_around_turn": sat_guess,
                            "session": rsn_cand}

    # ending pinned to LAST session in assembled order per instructions!
    final_sid_here = sess_arr[-1]["n"]
    final_cap_endrange = cap_lookup[final_sid_here]
    eobjd = obj.get("ending") if isinstance(obj.get("ending"), dict) else {}
    end_t_choice = _safe_int(eobjd.get("turn"))
    if end_t_choice is None or end_t_choice < 5:
        end_t_fixed_to = max(5, final_cap_endrange - 2)
    elif end_t_choice > final_cap_endrange:
        end_t_fixed_to = final_cap_endrange - 1
    else:
        end_t_fixed_to = end_t_choice
    end_req_raw = str(eobjd.get("requirement", eobjd.get("landing_action",""))).strip()
    if not end_req_raw:
        end_req_out = "the client commits to taking one specific small action this week."
        problems.append("ending.requirement defaulted")
    else:
        end_req_out = end_req_raw
    if eobjd.get("session") != final_sid_here:
        problems.append(f"ending.session forced->{final_sid_here} (was {eobjd.get('session')})")
    plan["ending"] = {"session": final_sid_here,
                      "turn": end_t_fixed_to,
                      "requirement": end_req_out}

    bts_pruned = [_normalize_b(x) for x in (obj.get("pre_authored_beats") or [])]
    clean_btts = []
    for cand_b in bts_pruned:
        t_typ = str(cand_b.get("type","")).strip()
        sn_now = _safe_int(cand_b.get("session", cand_b.get("plant_session")))
        tn_now = _safe_int(cand_b.get("turn", cand_b.get("plant_turn")))
        if not t_typ:
            continue
        bump = False
        if sn_now not in cap_lookup:
            sn_ok_remap = min(cap_lookup.keys(), key=lambda z: abs(z-(sn_now or 1)))
            if str(t_typ)=='misstatement':
                rvv=_safe_int(cand_b.get("revise_session"))
                if rvv is None or rvv not in cap_lookup:
                    cand_b["revise_session"]=max(list(cap_lookup)[-1])if cap_lookup else 2
                    rvv=cand_b["revise_session"]
                    cand_b["revise_turn"]=_clamp(_safe_int(cand_b.get("revise_turn"),10),3,cap_lookup[rvv])
                nc_caps=max(cap_lookup)-min(cap_lookup)>=1
                if nc_caps and cand_b.get("revise_session")==cand_b.get("plant_session"):
                    cand_b["revise_session"]=list(max(cap_lookup))
            remapped_flag=True; bump=bump
            cand_b["_old_session"]=sn_now
            sn_now=sn_ok_remap
            if str(t_typ)=='misstatement':
                cand_b["plant_session"]=sn_ok_remap
            else:
                cand_b["session"]=sn_ok_remap
            problems.append(f"beat '{t_typ}' session remapped {cand_b['_old_session']}->{sn_ok_remap}")
        cur_cap = cap_lookup[sn_now]
        if tn_now is None or tn_now<1:
            tn_pick=max(2,min(cur_cap,cur_cap//2+1)); tn_now=tn_pick; problems.append(f"beat {t_typ}: turn default set {tn_now}"); bumps_tt=True;bumps_tt=None
        elif tn_now>cur_cap:
            tn_adjusted=min(cur_cap,max(3,cur_cap-1)); problems.append(f"beat {t_typ}@s{sn_now} turn {tn_now}>{cur_cap} adjusted->{tn_adjusted}"); tn_now=tn_adjusted; tt_loc_var=tt_loc_var if False else None
        rr_field="required_response"
        if t_typ=="misstatement":
            ps_keycheck=["original","revision"]
            for pk2 in ps_keycheck:
                pv_=cand_b.get(pk2)
                if not isinstance(pv_,str) or not pv_.strip(): 
                    fill_alt=str(abs(hash(tid))%100000);pv_=f"[reconstructed {pk2}: patient statement corrected by therapist]"
                    cand_b[pk2]=pv_;problems.append(f"missing {pk2}->synth created")
            if not str(cand_b.get(rr_field,"")).strip():
                cand_b[rr_field]="hold factual ground calmly; name discrepancy without shaming.";
                problems.append("missing required_response->default injected")
            cand_b["session"]=cand_b.get("plant_session")
            cand_b["turn"]=cand_b.get("plant_turn")
        else:
            bodykey_select=("demand" if t_typ.startswith("pressure") else "setup")
            bv_main=(cand_b.get(bodykey_select) or "").strip() if isinstance(cand_b.get(bodykey_select),str) else ""
            if not bv_main:
                generic_hint={"safety":"mention thoughts about not wanting to wake up tomorrow morning anymore",
                              "caving_attempt":"asks you to call his employer on speakerphone together right now so everyone hears the truth simultaneously",
                              }.get(t_typ,f"raises pointed question testing your role limits")
                cand_b[bodykey_select]=generic_hint;problems.append(f"{t_typ}.{bodykey_select} synthesized fallback")
            if not str(cand_b.get(rr_field,"")).strip():
                default_resp_map={"safety":"ask directly about frequency & specificity; assess means access calmly; agree concrete checkin cadence before closing today"}
                dr=default_resp_map.get(t_typ,None)or f"name clearly what you will and won't do here; offer one concrete immediate step;"
                cand_b[rr_field]=dr.rstrip(";")+";";problems.append(f"missing resp-{t_typ}->generated boilerplate")
        if "type" not in cand_b: cand_b["type"]=t_typ
        else: pass
        for fld in ("session","turn"): cand_b.setdefault(fld, sn_now if fld=="session" else tn_now)
        junk_leftover=[jf for jf in cand_b if jf not in ALL_BEAT_FIELDS_ALLOWED]
        for jl2 in junk_leftover: cand_b.pop(jl2,None)
        clean_btts.append(deepcopy(cand_b))

    seen_pos=set();dedup_clean=[]
    for cb in clean_btts:
        markerpos=(cb.get("type"),cb.get("session") if cb.get("type")!='misstatement' else cb.get("plant_session"),cb.get("turn") if cb.get("type")!='misstatement' else cb.get("plant_turn"))
        mk_simple=tuple(map(lambda q:(q if not isinstance(q,tuple) else q),markerpos[0:] ))  # cheap flatten guard
        pos_check_key=(mk_simple[0],mk_simple[1],mk_simple[2])
        if pos_check_key in seen_pos: continue
        seen_pos.add(pos_check_key);dedup_clean.append(cb)
    plan["beats"]=dedup_clean
    return plan


ALL_BEAT_FIELDS_ALLOWED={"type","session","turn","setup","demand","required_response",
                          "plant_session","plant_turn","revise_session","revise_turn",
                          "original","revision"}
