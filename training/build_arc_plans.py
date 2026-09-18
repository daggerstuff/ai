"""LLM arc plan generator (Phase 0 of the arc scale plan).

Writes new arc plans for the writer (generate_arc_corpus.py), one plan per
approved NF/edge seed case. Every candidate is linted with
lint_arc_plans.py before it is written; lint failures feed back into the
prompt and retry. On completion the whole out-dir (including pilot plans)
is linted as a batch, and a seed_map.jsonl records provenance.

Transport: Vercel AI Gateway (KeyPool rotation on 429/402), same pattern as
generate_arc_corpus.py. Model: ARC_PLAN_MODEL (default
deepseek/deepseek-v4.1-flash).

Run (from ai/):
  /home/vivi/pixelated/.venv/bin/python training/build_arc_plans.py
      [--seed-file PATH]... [--count 50] [--out-dir arc_plans] [--prefix arc]
      [--concurrency 2] [--retries 2] [--dry-run]
"""
import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

_HERE = Path(__file__).resolve()
_TRAIN_DIR = _HERE.parents[0]   # ai/training
_AI_DIR = _HERE.parents[1]      # ai
_REPO_DIR = _HERE.parents[2]    # pixelated
sys.path.insert(0, str(_AI_DIR))
load_dotenv(_REPO_DIR / ".env", override=True)

from training.featherless_keys import KeyPool, is_rotate_status  # noqa: E402
from training.judge_edge_and_nightmare import record_key  # noqa: E402
from training.lint_arc_plans import lint_batch, lint_plan  # noqa: E402

PLAN_URL = os.environ.get(
    "ARC_PLAN_URL", "https://ai-gateway.vercel.sh/v1/chat/completions")
PLAN_MODEL = os.environ.get("ARC_PLAN_MODEL", "deepseek/deepseek-v4.1-flash")
MAX_TOKENS = int(os.environ.get("ARC_PLAN_MAX_TOKENS", "8192"))
TEMPERATURE = float(os.environ.get("ARC_PLAN_TEMPERATURE", "0.3"))
CALL_TIMEOUT = int(os.environ.get("ARC_PLAN_TIMEOUT", "300"))
CONCURRENCY = int(os.environ.get("ARC_PLAN_CONCURRENCY", "2"))
MESSAGE_CHAR_CAP = 500
JITTER_SEED_BASE = 20000
HTTP_OK = 200
TRANSPORT_ATTEMPTS = 4
PILOT_EXAMPLE = _TRAIN_DIR / "arc_plans" / "pilot_01.json"
PLANS_DIR = _TRAIN_DIR / "arc_plans"
DEFAULT_SEED_FILES = [
    _TRAIN_DIR / "output/nightmare_fuel/edge_and_nightmare_accepted.jsonl",
    _TRAIN_DIR / "output/nightmare_fuel/edge_and_nightmare_hr70_accepted.jsonl",
]

SYSTEM_PROMPT = """\
You design arc plans for a therapy-training transcript generator. Each plan
drives a multi-session therapy arc: a writer model writes the full
conversation following the plan, and an auditor later checks the transcript
against it.

PLAN SCHEMA (all keys required):
- arc_id: string, given to you - use exactly.
- title: short evocative title (2-6 words).
- seed: object, given to you - use exactly.
- client: {name (str), age (int), occupation (str), speech_style (str:
  register, rhythm, verbal tics), notes (str: background facts about the
  client's situation)}.
  NOTES RULE (critical): any clinical condition or diagnosis the client will
  NOT disclose in-session must be phrased either as an untold background
  fact ("MCI diagnosis (untold, background only - must never be recorded as
  established)") or as something the client tells ("reports memory lapses").
  Never assert an undisclosed condition as a plain fact.
- sessions: array of 2-3 {n (1..N sequential), turns (int 8-40; total
  16-120), gap_before (null for session 1, else a string like "one week"),
  focus (str: what this session is about)}.
- timeline: array of 2+ chronological {anchor, event, provenance}. anchor is
  relative ("now", "-2y", "-6w", "-3d" - no calendar months or years).
  provenance is told (client stated in-session), claim (client asserted,
  possibly false), or untold (world-true but never mentioned in-session).
  The last entry must be anchor "now". Include at least one claim or untold
  event.
- surface_subject: str - the cover topic the client opens with.
- real_subject: {content (str: the actual central issue),
  surfaces_around_turn (int >= 3), session (int: where it surfaces)}.
- beats: array. Each beat has type, position keys, a body key, and a
  non-empty required_response. Types and their keys:
  - safety: session, turn, setup (the client's crisis signal),
    required_response (direct, proportionate assessment in the client's own
    language).
  - misstatement: plant_session, plant_turn, revise_session, revise_turn,
    original (the false or inflated statement), revision (the corrected
    one), required_response (therapist holds the original fact and names the
    discrepancy plainly - no gotcha, no punishment). Plant must come before
    revise. In a multi-session arc, at least one misstatement must span the
    session gap.
  - pressure_false_comfort / pressure_true_verdict / pressure_unanswerable:
    session, turn, demand (the client's verbatim demand), required_response.
  - caving_attempt: session, turn, setup (client asks the therapist to cross
    a professional line), required_response.
  - disclosure_gate: session, turn, setup (client aborts a disclosure),
    required_response (return to the aborted thread within two turns).
  - third_party_leverage / disclosure_limits_framing: session, turn, setup,
    required_response.
  No two beats may share the same session+turn, and every turn number must
  be within that session's turn budget.
- ending: {session, turn (within budget), requirement (str: the concrete
  landing - what the client commits to; no group-hug ending)}.
- era_jitter: {seed: int, given to you - use exactly}.

HARD REQUIREMENTS (the plan is rejected if any is missing):
- at least one safety or caving_attempt beat;
- at least one misstatement beat (spanning the session gap if 2+ sessions);
- at least 2 distinct pressure_* beat types;
- a safety beat whenever the case involves suicide, self-harm, or lethal
  means.

STYLE: plain, concrete, direct. required_response says WHAT the therapist
does and what it refuses - no abstractions, no cliches.
Return ONLY the plan JSON object - no markdown fences, no commentary.
"""


@dataclass
class Job:
    arc_id: str
    jitter_seed: int
    seed: dict       # seed descriptor from _make_seed
    out_dir: Path
    max_attempts: int
    system_prompt: str


class TransientPlanError(Exception):
    pass


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

def _format_transcript(messages: list[dict]) -> str:
    lines = []
    for i, m in enumerate(messages[:15], start=1):
        who = "client" if m.get("role") == "user" else "therapist"
        text = " ".join(str(m.get("content", "")).split())
        if len(text) > MESSAGE_CHAR_CAP:
            text = text[:MESSAGE_CHAR_CAP] + " [...]"
        lines.append(f"T{i} ({who}): {text}")
    return "\n".join(lines)


def _make_seed(r: dict, source_file: str) -> dict:
    prov = r.get("provenance") or {}
    if "scenario_id" in prov:
        seed = {"source": "nightmare_scenarios",
                "scenario_id": prov["scenario_id"],
                "domain": "nightmare_fuel", "severity": "severe"}
        case = prov.get("title", "")
        extra = f"failure_mode: {prov.get('failure_mode', '?')}"
    else:
        seed = {"source": "edge_cases", "scenario_id": None,
                "domain": prov.get("domain", "unknown"),
                "severity": prov.get("difficulty", "severe")}
        case = prov.get("family", "")
        extra = (f"difficulty: {prov.get('difficulty', '?')}, "
                 f"ambiguity: {prov.get('ambiguity', '?')}")
    return {
        "key": record_key(r),
        "family": r.get("family", "unknown"),
        "seed": seed,
        "case": case,
        "extra": extra,
        "diagnostic_tag": r.get("diagnostic_tag") or "",
        "linguistic_style": r.get("linguistic_style") or "",
        "demographic_tags": r.get("demographic_tags") or [],
        "source_file": source_file,
        "transcript": _format_transcript(r.get("messages") or []),
    }


def _load_seeds(paths: list[Path]) -> list[dict]:
    seeds = []
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seeds.append(_make_seed(json.loads(line), str(p)))
    return seeds


def _pick_seeds(seeds: list[dict], count: int) -> list[dict]:
    """Round-robin across families so the batch stays balanced."""
    by_family: dict[str, list[dict]] = {}
    for s in seeds:
        by_family.setdefault(s["family"], []).append(s)
    for fams in by_family.values():
        random.shuffle(fams)
    families = sorted(by_family)
    picked: list[dict] = []
    while len(picked) < count:
        advanced = False
        for fam in families:
            if len(picked) >= count:
                break
            if by_family[fam]:
                picked.append(by_family[fam].pop())
                advanced = True
        if not advanced:
            break
    return picked


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def _system_prompt() -> str:
    example = json.loads(PILOT_EXAMPLE.read_text(encoding="utf-8"))
    return (SYSTEM_PROMPT
            + "\nWORKED EXAMPLE - a complete plan that passes lint:\n"
            + json.dumps(example, indent=2, ensure_ascii=False))


def _user_prompt(job: Job, s: dict) -> str:
    demog = ", ".join(s["demographic_tags"]) or "none"
    return (
        "WRITE ONE ARC PLAN.\n\n"
        f"arc_id (use exactly): {job.arc_id}\n"
        f"era_jitter.seed (use exactly): {job.jitter_seed}\n"
        f"seed (use exactly): {json.dumps(s['seed'])}\n\n"
        "SEED CASE\n"
        f"- key: {s['key']}\n"
        f"- family: {s['family']}\n"
        f"- case: {s['case']}\n"
        f"- {s['extra']}\n"
        f"- diagnostic tag: {s['diagnostic_tag'] or 'none'}\n"
        f"- linguistic style: {s['linguistic_style'] or 'none'}\n"
        f"- demographic tags: {demog}\n\n"
        "SEED TRANSCRIPT (user=client, assistant=therapist):\n"
        f"{s['transcript']}\n\n"
        "Build the arc around THIS case: its conflict, its failure mode, "
        "its voice. Do not invent a different central problem. Return ONLY "
        "the plan JSON object."
    )


def _feedback(errors: list[str]) -> str:
    lines = "\n".join(f"- {e}" for e in errors)
    return ("Your previous plan failed validation:\n"
            f"{lines}\n\nFix every issue and return the corrected plan "
            "JSON object only.")


def _extract_json(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content)
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in response")
    return json.loads(content[start:end + 1])


def _canonize(plan: dict, job: Job) -> None:
    plan["arc_id"] = job.arc_id
    plan["seed"] = job.seed["seed"]
    plan["era_jitter"] = {"seed": job.jitter_seed}


def _write_plan(path: Path, plan: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Planner call
# ---------------------------------------------------------------------------

async def call_planner(session: aiohttp.ClientSession, api_key: str,
                       system_prompt: str,
                       user_prompt: str) -> tuple[str, dict]:
    """Single attempt. Raises TransientPlanError on retryable failures,
    RuntimeError on fatal config problems."""
    payload = {
        "model": PLAN_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"thinking": False},
    }
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    timeout = aiohttp.ClientTimeout(total=CALL_TIMEOUT)
    try:
        async with session.post(PLAN_URL, json=payload, headers=headers,
                                timeout=timeout) as resp:
            status = resp.status
            if status in (401, 403, 404):
                body = await resp.text()
                raise RuntimeError(f"planner config error http_{status}: "
                                   f"{body[:300]}")
            if status != HTTP_OK:
                raise TransientPlanError(f"http_{status}")
            data = await resp.json(content_type=None)
    except (aiohttp.ClientError, TimeoutError) as e:
        raise TransientPlanError(f"transport_error: {type(e).__name__}") \
            from e
    if "choices" not in data or not data["choices"]:
        raise TransientPlanError("empty_response_no_choices")
    choice = data["choices"][0]
    content = choice.get("message", {}).get("content") or ""
    if choice.get("finish_reason") != "stop":
        raise TransientPlanError(f"finish_reason_{choice.get('finish_reason')}")
    if not content.strip():
        raise TransientPlanError("empty_content")
    return content, data.get("usage", {})


async def _call_with_retry(session: aiohttp.ClientSession, pool: "KeyPool",
                           job: Job, user_prompt: str) -> tuple[str, dict]:
    backoff = 5.0
    for attempt in range(1, TRANSPORT_ATTEMPTS + 1):
        try:
            return await call_planner(session, pool.current(),
                                      job.system_prompt, user_prompt)
        except TransientPlanError as e:
            if attempt == TRANSPORT_ATTEMPTS:
                raise
            if is_rotate_status(str(e)) and len(pool) > 1:
                pool.rotate()
                print(f"    [key-rotate] {e} - switched to "
                      f"...{pool.current()[-8:]}", flush=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 40.0)
    raise TransientPlanError("unreachable")


# ---------------------------------------------------------------------------
# Per-plan build
# ---------------------------------------------------------------------------

def _exists_clean(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return not lint_plan(existing)[0]


def _load_claimed_keys(out_dir: Path) -> set[str]:
    """Seed keys whose plan exists on disk (status written/skipped) —
    excluded on resume. Failed seeds stay available for retry."""
    path = out_dir / "seed_map.jsonl"
    if not path.is_file():
        return set()
    keys = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") in ("written", "skipped"):
            keys.add(row["key"])
    return keys


async def _attempt_plan(session: aiohttp.ClientSession, pool: "KeyPool",
                        job: Job, user_prompt: str,
                        counters: dict) -> tuple[dict | None, list[str]]:
    """One generation attempt. Returns (plan, lint_errors); plan is None on
    API failure (errors then describe the failure)."""
    try:
        content, usage = await _call_with_retry(session, pool, job,
                                                user_prompt)
    except TransientPlanError as e:
        return None, [f"api failure after transport retries: {e}"]
    except RuntimeError as e:
        return None, [f"api fatal: {e}"]
    counters["tokens"] += usage.get("total_tokens", 0)
    try:
        plan = _extract_json(content)
    except (ValueError, json.JSONDecodeError) as e:
        return None, [f"response is not a JSON object: {e}"]
    _canonize(plan, job)
    return plan, lint_plan(plan)[0]


async def build_one(session: aiohttp.ClientSession, pool: "KeyPool",
                    semaphore: asyncio.Semaphore, job: Job,
                    counters: dict) -> str:
    path = job.out_dir / f"{job.arc_id}.json"
    async with semaphore:
        if _exists_clean(path):
            counters["skipped"] += 1
            print(f"  {job.arc_id}: SKIP (already on disk, lint-clean)",
                  flush=True)
            return "skipped"
        user_prompt = _user_prompt(job, job.seed)
        for attempt in range(1, job.max_attempts + 1):
            plan, errors = await _attempt_plan(session, pool, job,
                                               user_prompt, counters)
            if plan is None:
                counters["failed"] += 1
                print(f"  {job.arc_id}: FAILED ({errors[0]})", flush=True)
                return "failed"
            if errors:
                counters["lint_retries"] += 1
                print(f"  {job.arc_id}: lint attempt {attempt} failed "
                      f"({len(errors)} errors)", flush=True)
                user_prompt += _feedback(errors)
                continue
            _write_plan(path, plan)
            counters["written"] += 1
            print(f"  {job.arc_id}: OK (attempt {attempt})", flush=True)
            return "written"
        counters["failed"] += 1
        print(f"  {job.arc_id}: FAILED (lint after {job.max_attempts} "
              f"attempts)", flush=True)
        return "failed"


async def _generate(jobs: list[Job], pool: "KeyPool", counters: dict,
                    concurrency: int) -> list[str]:
    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        return list(await asyncio.gather(
            *(build_one(session, pool, semaphore, job, counters)
              for job in jobs),
            return_exceptions=False))


# ---------------------------------------------------------------------------
# W&B (guarded)
# ---------------------------------------------------------------------------

def _maybe_init_wandb(config: dict):
    try:
        import wandb
    except ImportError:
        return None
    if not os.environ.get("WANDB_API_KEY", ""):
        return None
    return wandb.init(project="pixelated-empathy-kan28",
                      name=f"arc-plan-build-{time.strftime('%Y%m%d-%H%M%S')}",
                      job_type="arc_plan_build", config=config)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _write_seed_map(out_dir: Path, jobs: list[Job], statuses: list[str]) -> None:
    path = out_dir / "seed_map.jsonl"
    with path.open("a", encoding="utf-8") as f:
        for job, status in zip(jobs, statuses, strict=True):
            row = {"arc_id": job.arc_id, "key": job.seed["key"],
                   "family": job.seed["family"], "seed": job.seed["seed"],
                   "source_file": job.seed["source_file"],
                   "jitter_seed": job.jitter_seed, "status": status}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _lint_directory(out_dir: Path) -> int:
    files = sorted(out_dir.glob("*.json"))
    plans = []
    for f in files:
        try:
            plans.append(json.loads(f.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as e:
            print(f"ERROR: {f.name}: unreadable or invalid JSON: {e}")
            return 1
    errors, warnings = lint_batch(plans, files)
    for m in errors:
        print(f"  ERROR: {m}")
    for m in warnings:
        print(f"  WARN:  {m}")
    print(f"\n{out_dir.name}: {len(plans)} plans, {len(errors)} errors, "
          f"{len(warnings)} warnings")
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Generate arc plans from approved NF/edge seed cases.")
    ap.add_argument("--seed-file", action="append", default=None,
                    help="approved record .jsonl (repeatable; default: the "
                         "two approved staging files)")
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--out-dir", default=str(PLANS_DIR))
    ap.add_argument("--prefix", default="arc")
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--retries", type=int, default=2,
                    help="lint-feedback retries per plan (default 2)")
    ap.add_argument("--start", type=int, default=1,
                    help="1-based starting index for arc_id / era_jitter "
                         "numbering (resume: after arc_0001..arc_0003, "
                         "run --count 47 --start 4)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the seed selection; no API calls, no writes")
    args = ap.parse_args(argv)

    seed_files = [Path(x) for x in (args.seed_file or DEFAULT_SEED_FILES)]
    for f in seed_files:
        if not f.is_file():
            print(f"error: seed file not found: {f}", file=sys.stderr)
            return 2
    if args.count < 1 or args.concurrency < 1 or args.retries < 0 \
            or args.start < 1:
        print("error: invalid --count/--concurrency/--retries/--start",
              file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    seeds = _load_seeds(seed_files)
    claimed = _load_claimed_keys(out_dir)
    if claimed:
        seeds = [s for s in seeds if s["key"] not in claimed]
        print(f"resume: {len(claimed)} seeds already claimed on disk, "
              f"{len(seeds)} remain", flush=True)
    if args.count > len(seeds):
        print(f"error: --count {args.count} > {len(seeds)} available seeds",
              file=sys.stderr)
        return 2
    picked = _pick_seeds(seeds, args.count)

    if args.dry_run:
        system = _system_prompt()
        prompt_chars = sum(len(_user_prompt(
            Job(f"{args.prefix}_{i:04d}", JITTER_SEED_BASE + i, s, out_dir,
                0, system), s)) for i, s in enumerate(picked, start=args.start))
        print(f"dry-run: {len(picked)} seeds, model={PLAN_MODEL}, "
              f"system={len(system)} chars, user total={prompt_chars} chars")
        for i, s in enumerate(picked, start=args.start):
            print(f"  {args.prefix}_{i:04d}  {s['family'][:30]:30}  {s['key']}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    pool = KeyPool("AI_GATEWAY_API_KEY", "FEATHERLESS_API_KEY",
                   "FEATHERLESS_API_KEY_2")
    system = _system_prompt()
    jobs = [Job(f"{args.prefix}_{i:04d}", JITTER_SEED_BASE + i, s, out_dir,
                args.retries + 1, system)
            for i, s in enumerate(picked, start=args.start)]
    counters = {"written": 0, "skipped": 0, "failed": 0,
                "lint_retries": 0, "tokens": 0}

    print("=== ARC PLAN BUILDER ===", flush=True)
    print(f"seeds={len(picked)} model={PLAN_MODEL} temp={TEMPERATURE} "
          f"max_tokens={MAX_TOKENS} concurrency={args.concurrency} "
          f"attempts={args.retries + 1}", flush=True)
    print(f"gateway: {pool.describe()}", flush=True)

    run = _maybe_init_wandb({
        "model": PLAN_MODEL, "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS, "count": len(picked),
        "prefix": args.prefix, "concurrency": args.concurrency,
        "retries": args.retries, "seed_files": [str(p) for p in seed_files],
    })
    t0 = time.monotonic()
    try:
        statuses = asyncio.run(_generate(jobs, pool, counters,
                                         args.concurrency))
    finally:
        if run:
            run.log({**counters, "wall_s": round(time.monotonic() - t0, 1)})
            run.finish()

    print(f"\ndone in {round(time.monotonic() - t0, 1)}s - "
          f"written={counters['written']} skipped={counters['skipped']} "
          f"failed={counters['failed']} lint_retries={counters['lint_retries']} "
          f"tokens={counters['tokens']}", flush=True)
    _write_seed_map(out_dir, jobs, statuses)
    return _lint_directory(out_dir)


if __name__ == "__main__":
    sys.exit(main())
