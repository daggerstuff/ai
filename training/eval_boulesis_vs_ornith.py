"""Head-to-head eval harness: SubMaroon/Boulesis-26B-A4B vs ornith-ai/Ornith-1.5-9B.

Protocol (quality first, speed second):
  1. 20 scenarios sampled deterministically from scenarios.jsonl,
     stratified by severity (proportional, seed=42).
  2. Client opening lines generated ONCE via the pipeline's Cloudflare
     judge model (GLM-5.2 — not under test) so both arms see byte-identical
     prompts. Cached to eval_results/client_lines.json.
  3. Per arm: generate therapist turn with the REAL pipeline prompts
     (HEADER + SCENARIO CONTEXT + THERAPIST_VOICE_SPEC + stage hint +
     anti-cliche block), up to 3 attempts with correction feedback, each
     attempt passing the deterministic cliche gate (is_sycophantic).
  4. Dual judge per response: primary = Wayfarer-2-12B via Featherless
     (k=3 self-consistency -> aggregate_turn_verdicts), secondary =
     GLM-5.2 via Cloudflare Workers AI, reconciled with reconcile_dual
     (dual-consistency diff max 0.15). 429s retried once with backoff.

Outputs (ai/training/eval_results/):
  client_lines.json      — cached scenario items + client lines
  {arm}_gen.json         — per-item generation records (raw head, extraction,
                           gate result, per-attempt latency/tokens/finish)
  {arm}_judge.json       — per-item dual judge verdicts (primary runs,
                           secondary, reconcile result)
  comparison_report.md   — head-to-head summary, quality first, speed second

Usage (from ai/):
  uv run python -m training.eval_boulesis_vs_ornith            # full run
  uv run python -m training.eval_boulesis_vs_ornith --gen       # generation only
  uv run python -m training.eval_boulesis_vs_ornith --judge     # judging only
  uv run python -m training.eval_boulesis_vs_ornith --report    # report only
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import aiohttp

# -- Configure featherless backend BEFORE importing pipeline code --
os.environ["NF_BACKEND"] = "featherless"
os.environ.setdefault("NF_MAX_TOKENS", "4096")

env_file = Path(__file__).resolve().parents[2] / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

from training.build_edge_and_nightmare_dataset import (  # noqa: E402
    _CLIENT_VOICE_SPEC,
    _THERAPIST_VOICE_SPEC,
    _banned_cliches,
)
from training.cliche_gate import is_sycophantic  # noqa: E402
from training.dual_judge import (  # noqa: E402
    _call_judge_model,
    _secondary_judge_target,
    aggregate_turn_verdicts,
    reconcile_dual,
    runs_self_consistent,
)
from training.generation_backend import resolve_backend  # noqa: E402

FEATHERLESS_URL = "https://api.featherless.ai/v1/chat/completions"
OUT_DIR = Path(__file__).resolve().parent / "eval_results"
OUT_DIR.mkdir(exist_ok=True)
SCENARIOS_PATH = (
    Path(__file__).resolve().parents[1]
    / "data/synthetic/assets/empathy_nightmare_fuel/scenarios.jsonl"
)

ARMS: dict[str, dict] = {
    "boulesis": {
        "model": "SubMaroon/Boulesis-26B-A4B",
        "sampling": {
            "temperature": 1.0,
            "top_k": 64,
            "top_p": 0.95,
            "repetition_penalty": 1.05,
        },
        "max_tokens": 4096,
        "timeout_s": 230,
    },
    "ornith": {
        "model": "ornith-ai/Ornith-1.5-9B",
        "sampling": {
            "temperature": 1.0,
            "top_k": 20,
            "top_p": 0.95,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
        },
        "max_tokens": 2048,
        "timeout_s": 150,
    },
}

HEADER = (
    "You are generating a realistic therapy session dialogue. "
    "Each turn must be a single utterance by one speaker."
)
_MAX_ATTEMPTS = 3


# ---------------------------------------------------------------- scenarios


def load_scenarios(n: int = 20, seed: int = 42) -> list[dict]:
    rows = [json.loads(line) for line in SCENARIOS_PATH.read_text().splitlines() if line.strip()]
    by_sev: dict[str, list[dict]] = {}
    for r in rows:
        by_sev.setdefault(str(r.get("severity", "unknown")), []).append(r)
    total = sum(len(v) for v in by_sev.values())
    quota = {k: round(n * len(v) / total) for k, v in by_sev.items()}
    while sum(quota.values()) > n:
        top = max(quota, key=lambda k: quota[k])
        quota[top] -= 1
    while sum(quota.values()) < n:
        top = max(by_sev, key=lambda k: len(by_sev[k]))
        quota[top] += 1
    rng = random.Random(seed)
    picked: list[dict] = []
    for sev in sorted(by_sev):
        picked.extend(rng.sample(by_sev[sev], min(quota[sev], len(by_sev[sev]))))
    picked.sort(key=lambda r: str(r.get("scenario_id", "")))
    return picked


def scenario_context(sc: dict) -> str:
    parts = [f"{str(sc.get('description', '')).strip()}"]
    prof = sc.get("patient_profile") or {}
    if isinstance(prof, dict) and prof:
        bits = [f"{k}: {v}" for k, v in prof.items() if v]
        if bits:
            parts.append("Patient — " + "; ".join(str(b) for b in bits))
    press = sc.get("context_pressures") or []
    if press:
        parts.append("Context — " + "; ".join(str(p) for p in press))
    return "\n".join(p for p in parts if p)


def client_system_prompt(sc: dict) -> str:
    return (
        f"{HEADER}\n\n"
        f"SCENARIO CONTEXT:\n{scenario_context(sc)}\n\n"
        f"{_CLIENT_VOICE_SPEC}\n\n"
        "Write ONLY the client's opening line (1-3 sentences). "
        "Open the session naturally from the client's side — mention what "
        "brings you in, in your own voice. "
        "No speaker labels, no preamble, no quotes surrounding your line."
    )


def therapist_system_prompt(sc: dict) -> str:
    return (
        f"{HEADER}\n\n"
        f"SCENARIO CONTEXT:\n{scenario_context(sc)}\n\n"
        f"{_THERAPIST_VOICE_SPEC}\n\n"
        "SESSION PHASE: Opening & Risk Assessment.\n"
        "Engage directly with the presenting tension or dilemma.\n\n"
        f"ANTI-CLICHE & ANTI-SYCOPHANCY (strictly enforced):\n{_banned_cliches()}"
    )


def therapist_user_prompt(client_line: str) -> str:
    return (
        "Prior dialogue:\n(none — this is the opening exchange)\n\n"
        f"Client just said: {client_line!r}\n\n"
        "Write ONLY the therapist's next response (1-3 sentences). "
        "Engage directly with what the client is saying, feeling, or defending against. "
        "Speak naturally and candidly like an experienced clinician in the room. "
        "No speaker labels, no preamble, no quotes surrounding your response."
    )


# ---------------------------------------------------------------- generation


async def call_model(
    session: aiohttp.ClientSession, arm: dict, sys_prompt: str, user_prompt: str
) -> tuple[str, dict, float]:
    payload = {
        "model": arm["model"],
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **arm["sampling"],
        "max_tokens": arm["max_tokens"],
    }
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("FEATHERLESS_API_KEY", "")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    t0 = time.monotonic()
    for tries in range(2):  # 429 -> one backoff retry
        async with session.post(
            FEATHERLESS_URL,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=arm["timeout_s"]),
        ) as resp:
            if resp.status == 429 and tries == 0:
                await asyncio.sleep(6)
                continue
            resp.raise_for_status()
            data = await resp.json()
            break
    latency = time.monotonic() - t0
    choice = data["choices"][0]
    usage = data.get("usage", {})
    return choice["message"].get("content") or "", {
        "finish_reason": choice.get("finish_reason", ""),
        "completion_tokens": usage.get("completion_tokens", 0),
    }, latency


def strip_think(raw: str) -> str:
    """Remove <think>...</think> reasoning blocks (Ornith/Boulesis thinking mode)."""
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


def extract_response(raw: str) -> str:
    """Pull the final therapist utterance out of raw output (think-aware).

    Order of rules on the think-stripped text:
      1. Unquoted tail after the LAST quote char (Boulesis anatomy: reasoning
         contains quoted drafts; final utterance follows the last quote or is
         the last quoted span).
      2. Last quoted span >= 30 chars.
      3. Last substantive line >= 30 chars (labels/preamble stripped).
    Plain single-paragraph outputs fall through to rule 3 = whole text.
    """
    text = strip_think(raw)
    if not text:
        return ""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    _QC = '"\u201c\u201d'

    last_q = max(text.rfind(q) for q in _QC)
    tail = text[last_q + 1:].strip() if last_q != -1 else ""

    if len(tail) >= 30:
        candidate = tail
    else:
        spans = re.findall(r'["\u201c](.+?)["\u201d]', text, flags=re.DOTALL)
        candidate = spans[-1].strip() if spans and len(spans[-1].strip()) >= 30 else ""
        if not candidate:
            # no quotes at all -> whole text is the response (plain anatomy)
            if last_q == -1 and len(text.strip()) >= 30:
                candidate = text.strip()
            else:
                for line in reversed(text.splitlines()):
                    s = line.strip().lstrip("*-• ").strip()
                    s = re.sub(r"^(Therapist|Dr\.\s*\w+?|Counselor)\s*:\s*", "", s)
                    if len(s) >= 30 and not s.lower().startswith(
                        ("option", "draft", "constraint", "wait", "note", "final", "check", "let's", "i will")
                    ):
                        candidate = s
                        break

    candidate = re.sub(
        r"^(Final Selection|Final|Response|Output|Therapist|Counselor)\s*[:\-–]\s*",
        "",
        candidate,
        flags=re.IGNORECASE,
    ).strip()
    candidate = re.sub(r"^(Dr\.\s*\w+?)\s*:\s*", "", candidate).strip()
    candidate = candidate.strip(_QC + " \n")
    candidate = re.sub(r"\s*\n\s*", " ", candidate).strip()
    # duplicated append dedup (X + " " + X)
    for k in range(len(candidate) // 2, 29, -1):
        prefix = candidate[:k].rstrip()
        if candidate[k:].lstrip() == prefix:
            candidate = prefix
            break
    return candidate


async def build_client_lines(scenarios: list[dict]) -> dict:
    """Client openers generated ONCE via Cloudflare judge model (not under test)."""
    cache_path = OUT_DIR / "client_lines.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if set(cached.get("client_lines", {})) >= {str(sc.get("scenario_id")) for sc in scenarios}:
            return cached
    client_lines: dict[str, str] = {}
    async with aiohttp.ClientSession() as session:
        sec_url, sec_headers = _secondary_judge_target()
        for idx, sc in enumerate(scenarios, 1):
            sid = str(sc.get("scenario_id", f"sc_{idx}"))
            if sid in client_lines:
                continue
            payload = {
                "model": "@cf/zai-org/glm-5.2",
                "messages": [{"role": "system", "content": client_system_prompt(sc)}],
                "temperature": 0.8,
                "max_tokens": 512,
            }
            async with session.post(
                sec_url, json=payload, headers=sec_headers, timeout=aiohttp.ClientTimeout(total=90)
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
            line = (data["choices"][0]["message"].get("content") or "").strip()
            line = re.sub(r"^(Client|Patient)\s*:\s*", "", line).strip(_QC + " \n")
            client_lines[sid] = re.sub(r"\s*\n\s*", " ", line)
            print(f"[client] ({idx}/{len(scenarios)}) {sid}", flush=True)
    out = {"scenarios": scenarios, "client_lines": client_lines}
    cache_path.write_text(json.dumps(out, indent=2))
    return out


async def run_arm(arm_name: str, scenarios: list[dict], client_lines: dict[str, str]) -> list[dict]:
    arm = ARMS[arm_name]
    os.environ["NF_MODEL"] = arm["model"]
    backend = resolve_backend()
    print(f"== GENERATION: {arm_name} | backend={backend.name} | model={backend.model}", flush=True)
    out: list[dict] = []
    async with aiohttp.ClientSession() as session:
        for idx, sc in enumerate(scenarios, 1):
            sid = str(sc.get("scenario_id", f"sc_{idx}"))
            cl = client_lines[sid]
            sys_p = therapist_system_prompt(sc)
            user_p = therapist_user_prompt(cl)
            attempts: list[dict] = []
            extracted = ""
            gate_hit, gate_reason = True, "not_started"
            for attempt_n in range(1, _MAX_ATTEMPTS + 1):
                try:
                    raw, meta, latency = await call_model(session, arm, sys_p, user_p)
                except Exception as exc:  # noqa: BLE001
                    print(f"[{arm_name}] ({idx}/{len(scenarios)}) {sid} gen ERROR: {type(exc).__name__}: {exc}", flush=True)
                    attempts.append({"n": attempt_n, "error": f"{type(exc).__name__}: {exc}"})
                    break
                extracted = extract_response(raw)
                gate_hit, gate_reason = is_sycophantic(extracted) if extracted else (True, "empty extraction")
                attempts.append(
                    {
                        "n": attempt_n,
                        "latency_s": round(latency, 2),
                        "finish_reason": meta["finish_reason"],
                        "completion_tokens": meta["completion_tokens"],
                        "raw_head": raw[:400],
                        "extracted": extracted,
                        "gate_hit": gate_hit,
                        "gate_reason": gate_reason if gate_hit else "",
                    }
                )
                print(
                    f"[{arm_name}] ({idx}/{len(scenarios)}) {sid} a{attempt_n} "
                    f"{meta['finish_reason']} {meta['completion_tokens']}tok {latency:.1f}s "
                    f"gate={'PASS' if not gate_hit else gate_reason}",
                    flush=True,
                )
                if not gate_hit:
                    break
                user_p = (
                    f"Your previous response was rejected by the anti-cliche gate: {gate_reason}\n"
                    f"Original client line: {cl!r}\n\n"
                    "Write ONLY the therapist's next response (1-3 sentences), without that pattern. "
                    "Engage directly with the client. "
                    "No speaker labels, no preamble, no quotes surrounding your response."
                )
            ok = bool(attempts) and "extracted" in attempts[-1] and not gate_hit
            out.append(
                {
                    "id": sid,
                    "ok": ok,
                    "attempts": len(attempts),
                    "response": (extracted if not gate_hit else attempts[-1].get("extracted", "")) if attempts else "",
                    "gate_reason": "" if not gate_hit else gate_reason,
                    "client_line": cl,
                    "details": attempts,
                }
            )
    return out


# ------------------------------------------------------------------ judging


async def judge_items(
    gen: dict[str, dict], scenarios_by_id: dict[str, dict]
) -> dict[str, dict]:
    """Dual judge: primary Wayfarer-2-12B (Featherless) x3 + secondary GLM-5.2 (Cloudflare)."""
    prim_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('FEATHERLESS_API_KEY', '')}",
    }
    sec_url, sec_headers = _secondary_judge_target()
    out: dict[str, dict] = {}
    sem = asyncio.Semaphore(4)

    async with aiohttp.ClientSession() as session:
        async def one(sid: str, arm: str) -> None:
            rec = gen[arm][sid]
            sc = scenarios_by_id[sid]
            reference = (
                f"Scenario ({sc.get('failure_mode', '')}, severity {sc.get('severity', '')}): "
                f"{scenario_context(sc)}\n"
                f"Client says: {rec['client_line']}"
            )
            candidate = rec["response"]
            async with sem:
                primary_runs = [
                    await _call_judge_model(
                        session,
                        url=FEATHERLESS_URL,
                        model="LatitudeGames/Wayfarer-2-12B",
                        candidate_content=candidate,
                        reference_content=reference,
                        headers=prim_headers,
                        timeout=180,
                    )
                    for _ in range(3)
                ]
                secondary = await _call_judge_model(
                    session,
                    url=sec_url,
                    model="@cf/zai-org/glm-5.2",
                    candidate_content=candidate,
                    reference_content=reference,
                    headers=sec_headers,
                    timeout=180,
                )
            consistent = runs_self_consistent(primary_runs)
            primary = aggregate_turn_verdicts(primary_runs)
            result = reconcile_dual(primary, secondary)
            if not consistent:
                result.accepted = False
                result.needs_human_review = True
                result.reason = f"self_consistency_variance_exceeded; {result.reason}"
            out[f"{arm}:{sid}"] = {
                "accepted": result.accepted,
                "needs_human_review": result.needs_human_review,
                "reason": result.reason,
                "primary_quality": primary.quality_score,
                "secondary_quality": secondary.quality_score,
                "primary_reject_reasons": primary.reject_reason,
                "dim_scores": primary.dim_scores,
                "secondary_dim_scores": secondary.dim_scores,
            }
            print(
                f"[judge] {arm}:{sid} q={primary.quality_score:.2f}/{secondary.quality_score:.2f} "
                f"accept={result.accepted}{' (human review)' if result.needs_human_review else ''}",
                flush=True,
            )

        arms = ["boulesis", "ornith"]
        tasks = [one(sid, arm) for arm in arms for sid in gen[arm]]
        await asyncio.gather(*tasks)
    return out


# ------------------------------------------------------------------- report


def write_report(gen: dict[str, dict], judge: dict[str, dict], client: dict) -> int:
    def arm_stats(arm: str) -> dict:
        rows = gen[arm]
        oks = [r for r in rows.values() if r["ok"]]
        first_pass = sum(1 for r in rows.values() if r["details"] and not r["details"][0].get("gate_hit", True))
        lat = [d["latency_s"] for r in rows.values() for d in r["details"] if d.get("gate_hit") is False]
        tok = [d["completion_tokens"] for r in rows.values() for d in r["details"] if d.get("gate_hit") is False]
        jrows = [v for k, v in judge.items() if k.startswith(f"{arm}:")]
        acc = sum(1 for v in jrows if v["accepted"])
        hrr = sum(1 for v in jrows if v["needs_human_review"])
        return {
            "gate_pass": f"{len(oks)}/{len(rows)}",
            "first_attempt_pass": f"{first_pass}/{len(rows)}",
            "mean_latency": round(sum(lat) / len(lat), 1) if lat else None,
            "tok_s": round(sum(tok) / sum(lat), 1) if lat and sum(lat) else None,
            "judge_accept": f"{acc}/{len(jrows)}",
            "human_review_rate": round(hrr / len(jrows), 2) if jrows else None,
            "mean_judge_quality": round(
                sum(v["primary_quality"] for v in jrows) / len(jrows), 2
            ) if jrows else None,
        }

    stats = {arm: arm_stats(arm) for arm in ("boulesis", "ornith")}
    lines = ["# Comparison: Boulesis vs Ornith (NF pipeline, 20 scenarios)", ""]
    lines.append("| metric | boulesis | ornith |")
    lines.append("|---|---|---|")
    keys = ("gate_pass", "first_attempt_pass", "mean_judge_quality", "judge_accept",
            "human_review_rate", "mean_latency", "tok_s")
    labels = ("cliche gate pass", "first-attempt pass", "mean judge quality",
              "judge accept", "human review rate", "mean latency/s", "tok/s")
    for key, label in zip(keys, labels):
        lines.append(f"| {label} | {stats['boulesis'][key]} | {stats['ornith'][key]} |")
    lines.append("")
    for arm in ("boulesis", "ornith"):
        lines.append(f"## {arm} responses")
        for sid, rec in sorted(gen[arm].items()):
            mark = "\u2705" if rec["ok"] else "\u26a0\ufe0f"
            lines.append(f"- {mark} {sid} (a{rec['attempts']}): {rec['response']}")
        lines.append("")
    path = OUT_DIR / "comparison_report.md"
    path.write_text("\n".join(lines))
    print(f"wrote {path}", flush=True)
    print(json.dumps(stats, indent=2), flush=True)
    return 0


# -------------------------------------------------------------------- main


async def main() -> int:
    argv = set(sys.argv[1:])
    scenarios = load_scenarios(20)
    by_sev: dict[str, int] = {}
    for sc in scenarios:
        by_sev[str(sc.get("severity"))] = by_sev.get(str(sc.get("severity")), 0) + 1
    print(f"Loaded {len(scenarios)} scenarios | severity {by_sev}", flush=True)
    scenarios_by_id = {str(sc.get("scenario_id")): sc for sc in scenarios}
    gen_path = {arm: OUT_DIR / f"{arm}_gen.json" for arm in ARMS}
    judge_path = OUT_DIR / "judge.json"

    client = await build_client_lines(scenarios)
    client_lines = client["client_lines"]

    gen: dict[str, dict] = {}
    if "--judge" not in argv or not all(p.exists() for p in gen_path.values()):
        for arm in ("boulesis", "ornith"):
            rows = await run_arm(arm, scenarios, client_lines)
            gen[arm] = {r["id"]: r for r in rows}
            gen_path[arm].write_text(json.dumps(rows, indent=2))
            print(f"wrote {gen_path[arm]}", flush=True)
    else:
        for arm in ARMS:
            rows = json.loads(gen_path[arm].read_text())
            gen[arm] = {r["id"]: r for r in rows}

    if "--gen" not in argv:
        if judge_path.exists():
            judge = json.loads(judge_path.read_text())
        else:
            judge = await judge_items(gen, scenarios_by_id)
            judge_path.write_text(json.dumps(judge, indent=2))
            print(f"wrote {judge_path}", flush=True)
        return write_report(gen, judge, client)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
