#!/usr/bin/env python3
"""Eval harness: rank candidate auditor models on the arc-corpus audit task.

Runs each candidate model through the PRODUCTION auditor prompt
(imported from audit_arc_corpus — single source of truth) over a fixed
probe set of arc records with ground-truth verdicts, and scores:

- infra_rate        fraction of probes returning a parseable verdict
- defect_caught     non-accept on known-defective probes (sensitivity)
- defect_localized  flagged the right category at the right session
- agreement         verdict matches ground truth (accept <-> non-accept)
- false_revise      accept probes wrongly sent to revise/fail (specificity)
- flags_on_good     mean flag count on accept probes (over-flag spam)

Composite = 0.45*defect_caught + 0.35*agreement + 0.20*(1 - false_revise_rate).
Models with infra_rate < 0.8 are marked DQ (unreliable for a one-shot gate).

Featherless constraints handled:
- 1 concurrent call/account -> fully sequential, key rotation on 429
- 4 model switches/min -> one contiguous batch per model, sleep between
- response_format / chat_template_kwargs may be rejected by some models ->
  automatic variant fallback (production -> no response_format -> bare)

Usage:
  python training/eval_audit_models.py \
    --models "TheDrummer/Orion-26B-A4B-v1.1,..." \
    --aliases "orion,..." [--limit N] [--smoke]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()
_TRAIN_DIR = _HERE.parents[0]
_AI_DIR = _HERE.parents[1]
_REPO_DIR = _HERE.parents[2]
sys.path.insert(0, str(_AI_DIR))

import aiohttp  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(_REPO_DIR / ".env", override=True)

from audit_arc_corpus import (  # noqa: E402
    SYSTEM_PROMPT,
    parse_verdict,
    render_beats,
    render_transcript,
)

FEATHERLESS_URL = os.environ.get(
    "ARC_EVAL_URL", "https://api.featherless.ai/v1/chat/completions")
MAX_TOKENS = int(os.environ.get("ARC_EVAL_MAX_TOKENS", "12288"))
TEMPERATURE = 0.2
TIMEOUT = int(os.environ.get("ARC_EVAL_TIMEOUT", "480"))
MAX_CALL_RETRIES = 30          # 429 backoff budget (writers may share slots)
MAX_PARSE_RETRIES = 3          # invalid-JSON / transient per probe
MODEL_SWITCH_SLEEP = 10.0      # stay under the 4-switches/min plan limit
INFRA_DQ_THRESHOLD = 0.8
STATUS_OK = 200
STATUS_BUSY = 429
MAX_VARIANT = 2                # 0=production, 1=no response_format, 2=bare

KEY_ENVS = [k for k in os.environ.get(
    "ARC_EVAL_KEYS", "FEATHERLESS_API_KEY,FEATHERLESS_API_KEY_2").split(",") if k]
KEYS = [os.environ.get(k, "") for k in KEY_ENVS]

OUT_DIR = _TRAIN_DIR / "output" / "arc_corpus" / "eval_audit"

ChatPayload = dict[str, Any]
ProbeRow = dict[str, Any]
ProbeResult = dict[str, Any]
Usage = dict[str, Any]


def user_prompt_for(record: ProbeRow, plan: ProbeRow) -> str:
    return (
        f"ARC PLAN BEATS (required responses):\n{render_beats(plan)}\n\n"
        f"TRANSCRIPT:\n{render_transcript(record)}\n\n"
        "Grade the transcript against the plan. Verdict + flags as strict JSON.")


def build_payload(model: str, user_prompt: str, variant: int) -> ChatPayload:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    if variant <= 1:
        payload["chat_template_kwargs"] = {"thinking": False}
    if variant == 0:
        payload["response_format"] = {"type": "json_object"}
    return payload


async def call_once(http: aiohttp.ClientSession, key: str, payload: ChatPayload) -> tuple[int, str, Usage]:
    """Single POST. Returns (status, message_content, usage)."""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    async with http.post(FEATHERLESS_URL, json=payload, headers=headers,
                         timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as resp:
        text = await resp.text()
        usage: Usage = {}
        content = text
        if resp.status == STATUS_OK:
            with contextlib.suppress(Exception):
                data = json.loads(text)
                usage = data.get("usage") or {}
                content = data["choices"][0]["message"]["content"]
        return resp.status, content, usage


async def audit_probe(http: aiohttp.ClientSession, model: str, user_prompt: str) -> ProbeResult:
    """One probe call with 429 rotation + variant fallback + parse retries."""
    out: ProbeResult = {"status": None, "variant": None, "latency_s": None, "usage": None,
           "verdict": None, "flags": [], "error": None, "content_preview": None}
    variant = 0
    parse_attempts = 0
    t0 = time.monotonic()
    for _ in range(MAX_CALL_RETRIES):
        for key in KEYS:
            try:
                status, text, usage = await call_once(http, key, build_payload(model, user_prompt, variant))
            except (TimeoutError, aiohttp.ClientError) as e:
                out["error"] = f"transport: {type(e).__name__}: {e}"
                await asyncio.sleep(5)
                continue
            if status == STATUS_BUSY:
                await asyncio.sleep(5)
                continue
            if status in (400, 413, 422) and variant < MAX_VARIANT:
                # likely response_format / chat_template_kwargs rejected
                variant += 1
                continue
            if status != STATUS_OK:
                out.update(status=status, error=text[:300])
                out["latency_s"] = round(time.monotonic() - t0, 1)
                return out
            out.update(status=STATUS_OK, variant=variant, usage=usage)
            parsed = parse_verdict(text)
            if parsed is None:
                parse_attempts += 1
                out["error"] = "unparseable verdict JSON"
                out["content_preview"] = text[:400]
                if parse_attempts >= MAX_PARSE_RETRIES:
                    out["latency_s"] = round(time.monotonic() - t0, 1)
                    return out
                await asyncio.sleep(2)
                continue
            out["verdict"] = parsed["verdict"]
            out["flags"] = parsed["flags"]
            out["latency_s"] = round(time.monotonic() - t0, 1)
            return out
        # all keys 429'd
        await asyncio.sleep(10)
    out["error"] = "retries exhausted (429/transport)"
    out["latency_s"] = round(time.monotonic() - t0, 1)
    return out


def load_done(results_path: Path) -> set[tuple[str, str]]:
    done = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("verdict") or row.get("status") not in (None, 429):
                done.add((row["alias"], row["arc_id"]))
    return done


def score_model(rows: list[ProbeRow], probes: list[ProbeRow]) -> dict[str, Any]:
    gt = {p["arc_id"]: p["ground_truth"] for p in probes}
    n = len(rows)
    n_valid = sum(1 for r in rows if r.get("verdict"))
    defect_ids = [a for a, g in gt.items() if g["verdict"] == "reject"]
    accept_ids = [a for a, g in gt.items() if g["verdict"] == "accept"]

    by_arc = {r["arc_id"]: r for r in rows}
    defect_caught = sum(
        1 for a in defect_ids
        if by_arc.get(a, {}).get("verdict") in ("revise", "fail"))
    defect_localized = 0.0
    for a in defect_ids:
        g = gt[a]
        r = by_arc.get(a, {})
        if r.get("verdict") == "accept":
            continue
        hit = any(
            f.get("category") in g["defect_categories"]
            and f.get("session") == g["defect_session"]
            for f in r.get("flags", []))
        if hit:
            defect_localized += 1
        elif any(f.get("category") in g["defect_categories"] for f in r.get("flags", [])):
            defect_localized += 0.5  # right category, wrong session
    agree = sum(
        1 for r in rows
        if r.get("verdict") and ((gt[r["arc_id"]]["verdict"] == "accept")
                                 == (r["verdict"] == "accept")))
    false_revise = sum(
        1 for a in accept_ids
        if by_arc.get(a, {}).get("verdict") in ("revise", "fail"))
    flags_on_good = [len(by_arc.get(a, {}).get("flags", [])) for a in accept_ids
                     if by_arc.get(a, {}).get("verdict") == "accept"]
    lat = [r["latency_s"] for r in rows if isinstance(r.get("latency_s"), (int, float))]
    infra_rate = n_valid / n if n else 0.0
    agree_rate = agree / n if n else 0.0
    caught_rate = defect_caught / len(defect_ids) if defect_ids else 0.0
    false_rate = false_revise / len(accept_ids) if accept_ids else 0.0
    composite = (0.45 * caught_rate + 0.35 * agree_rate + 0.20 * (1 - false_rate))
    return {
        "probes": n, "valid": n_valid, "infra_rate": round(infra_rate, 3),
        "dq": infra_rate < INFRA_DQ_THRESHOLD,
        "defect_caught": f"{defect_caught}/{len(defect_ids)}",
        "defect_caught_rate": caught_rate,
        "defect_localized": defect_localized,
        "agreement": f"{agree}/{n}", "agreement_rate": agree_rate,
        "false_revise": f"{false_revise}/{len(accept_ids)}",
        "flags_on_good_mean": round(sum(flags_on_good) / len(flags_on_good), 2) if flags_on_good else None,
        "avg_latency_s": round(sum(lat) / len(lat), 1) if lat else None,
        "composite": round(composite, 3),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma-separated Featherless model IDs")
    ap.add_argument("--aliases", required=True, help="comma-separated short names, same order")
    ap.add_argument("--probes", default=str(OUT_DIR / "probe_set.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="only first N probes (smoke)")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    aliases = [a.strip() for a in args.aliases.split(",") if a.strip()]
    assert len(models) == len(aliases), "models/aliases length mismatch"

    probes = [json.loads(line) for line in Path(args.probes).read_text().splitlines() if line.strip()]
    if args.limit:
        probes = probes[:args.limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "results.jsonl"
    done = load_done(results_path)

    f = results_path.open("a")
    async with aiohttp.ClientSession() as http:
        for alias, model in zip(aliases, models, strict=True):
            todo = [p for p in probes if (alias, p["arc_id"]) not in done]
            if not todo:
                print(f"== {alias}: all {len(probes)} probes already done", flush=True)
                continue
            print(f"== {alias} ({model}): {len(todo)} probes", flush=True)
            for p in todo:
                up = user_prompt_for(p["record"], p["plan"])
                res = await audit_probe(http, model, up)
                row = {"alias": alias, "model": model, "arc_id": p["arc_id"],
                       "prompt_chars": len(up), **res}
                f.write(json.dumps(row) + "\n")
                f.flush()
                os.fsync(f.fileno())
                gt_v = p["ground_truth"]["verdict"]
                mark = "OK  " if (res["verdict"] == "accept" and gt_v == "accept") or \
                    (res["verdict"] in ("revise", "fail") and gt_v == "reject") else "MISMATCH"
                print(f"  {p['arc_id']:10s} gt={gt_v:6s} -> {res['verdict']!s:6s} "
                      f"flags={len(res['flags'])} {res['latency_s']}s [{mark}]"
                      + (f" err={res['error'][:80]}" if res.get("error") and not res["verdict"] else ""),
                      flush=True)
            await asyncio.sleep(MODEL_SWITCH_SLEEP)
    f.close()

    # leaderboard
    rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    leaders = []
    for alias in aliases:
        mrows = [r for r in rows if r["alias"] == alias and r["arc_id"] in {p["arc_id"] for p in probes}]
        leaders.append({"alias": alias, "model": models[aliases.index(alias)],
                        **score_model(mrows, probes)})
    leaders.sort(key=lambda x: (x["dq"], -x["composite"]))
    with (OUT_DIR / "leaderboard.json").open("w") as lf:
        json.dump({"probes": len(probes), "models": leaders}, lf, indent=2)

    print("\n=== LEADERBOARD (composite = .45*defect_caught + .35*agreement + .20*specificity) ===")
    print(f"{'alias':12s} {'composite':>9s} {'dq':3s} {'valid':>5s} {'caught':>7s} "
          f"{'local':>5s} {'agree':>7s} {'falseRev':>8s} {'flG':>5s} {'lat':>6s}")
    for entry in leaders:
        print(f"{entry['alias']:12s} {entry['composite']:9.3f} {'DQ' if entry['dq'] else '-':3s} "
              f"{entry['valid']:>5d} {entry['defect_caught']:>7s} {entry['defect_localized']!s:>5s} "
              f"{entry['agreement']:>7s} {entry['false_revise']:>8s} "
              f"{entry['flags_on_good_mean']!s:>5s} {entry['avg_latency_s']!s:>6s}")


if __name__ == "__main__":
    asyncio.run(main())
