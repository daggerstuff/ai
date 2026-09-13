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

Local-serving parity check (vLLM pod):
  EVAL_ARMS=ornith,ornith_local VLLM_URL=http://localhost:8000 \
    uv run python -m training.eval_boulesis_vs_ornith
  The ornith_local arm posts to VLLM_URL (default localhost:8000) with optional
  VLLM_API_KEY auth; the judge cache partial-merges, so previously judged arms
  are not re-judged. See training/lightning_ornith_setup.md.
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
            os.environ.setdefault(k.strip(), v.strip().strip("'\"").rstrip("\r"))

from training.build_edge_and_nightmare_dataset import (  # noqa: E402
    _CLIENT_VOICE_SPEC,
    _THERAPIST_VOICE_SPEC,
    _banned_cliches,
)
from training.cliche_gate import correction_for_reason, is_sycophantic  # noqa: E402
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
    "heresy": {
        # Closest deployed substitute for the (undeployed) Wayfarer-2-12B
        # absolute-heresy variant: 12B, Mistral family, same heresy series.
        "model": "MuXodious/Mistral-Helcyon-Mercury-12b-v3.2-absolute-heresy",
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
    "ornith_local": {
        # Same weights as the `ornith` arm, served by OUR vLLM pod (bf16) —
        # migration candidate for bulk dataset generation. Sampling identical
        # to the featherless ornith arm; max_tokens/timeout raised because
        # local tokens are free and truncation is a serving artifact, not a
        # model-quality signal. Endpoint from VLLM_URL (see
        # training/lightning_ornith_setup.md).
        "model": "ornith-ai/Ornith-1.5-9B",
        "url": f"{os.environ.get('VLLM_URL', 'http://localhost:8000').rstrip('/')}/v1/chat/completions",
        "api_key_env": "VLLM_API_KEY",
        "sampling": {
            "temperature": 1.0,
            "top_k": 20,
            "top_p": 0.95,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
        },
        "max_tokens": 4096,
        "timeout_s": 240,
    },
}

HEADER = (
    "You are generating a realistic therapy session dialogue. "
    "Each turn must be a single utterance by one speaker."
)
_QC = '"\u201c\u201d'  # straight + curly double quotes
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
    session: aiohttp.ClientSession,
    arm: dict,
    sys_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int | None = None,
    timeout_s: int | None = None,
) -> tuple[str, dict, float]:
    payload = {
        "model": arm["model"],
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **arm["sampling"],
        "max_tokens": max_tokens or arm["max_tokens"],
    }
    url = arm.get("url", FEATHERLESS_URL)
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(arm.get("api_key_env", "FEATHERLESS_API_KEY"), "")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    t0 = time.monotonic()
    for tries in range(3):  # 429 -> backoff retry; 5xx -> retry once more
        async with session.post(
            url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout_s or arm["timeout_s"]),
        ) as resp:
            if resp.status == 429 and tries < 2:
                await asyncio.sleep(6)
                continue
            if resp.status >= 500 and tries < 2:
                await asyncio.sleep(4 * (tries + 1))
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

    Anatomy-aware rules on the think-stripped text:
      1. Boulesis anatomy: reasoning contains long quoted drafts and the final
         utterance is the unquoted tail after the LAST quote (or the last
         quoted span). The tail is only trusted when it starts a fresh
         sentence — a lowercase tail following a letter means we cut into
         mid-sentence prose with an inline quote (Ornith anatomy).
      2. Ornith/plain anatomy: prose with short inline quotes or no quotes at
         all -> the whole text is the response.
    """
    text = strip_think(raw)
    if not text:
        return ""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)

    last_q = max(text.rfind(q) for q in _QC)
    tail = text[last_q + 1:].strip() if last_q != -1 else ""
    before = text[:last_q].rstrip() if last_q != -1 else ""
    # fragment guard: lowercase tail right after a letter = mid-sentence cut
    tail_starts_fresh = not (tail[:1:1].islower() and before[-1:].isalpha())
    tail_ok = last_q != -1 and len(tail) >= 30 and tail_starts_fresh

    spans = re.findall(r'["\u201c](.+?)["\u201d]', text, flags=re.DOTALL)
    long_spans = [s.strip() for s in spans if len(s.strip()) >= 30]

    if tail_ok and long_spans:
        candidate = tail
    elif long_spans:
        # final utterance itself quoted (Boulesis variant)
        candidate = long_spans[-1]
    else:
        # plain prose, possibly with short inline quotes (Ornith anatomy)
        candidate = text.strip()

    candidate = re.sub(
        r"^(Final Selection|Final|Response|Output|Therapist|Counselor)\s*[:\-–]\s*",
        "",
        candidate,
        flags=re.IGNORECASE,
    ).strip()
    candidate = re.sub(r"^(Dr\.\s*\w+?)\s*:\s*", "", candidate).strip()
    candidate = candidate.strip(_QC + " \n")
    candidate = re.sub(r"\s*\n\s*", " ", candidate).strip()

    # strip leading reasoning meta-notes ("Draft A: ...", "Caving phrases banned.")
    # bounded: only short non-question leading sentences, max 4
    _META = re.compile(
        r"^(draft|option|choose|choosing|better|worse|final|check|banned|note|"
        r"caving|robotic|parroting|sycophant\w*|cliche|cliches|gate|reject\w*|"
        r"retry|rewrite|correction|hmm|okay|ok|let me|i'll|i will|now i|next|"
        r"decision|selected|pick|avoid)\b",
        re.IGNORECASE,
    )
    for _ in range(4):
        parts = re.split(r"(?<=[.!?\u201d])\s+", candidate, maxsplit=1)
        if len(parts) == 2:
            head = parts[0].strip()
            if len(head) < 60 and not head.endswith("?") and (
                _META.match(head) or head.endswith(":")
            ):
                candidate = parts[1].strip()
                continue
        break

    # duplicated append dedup (X + " " + X)
    for k in range(len(candidate) // 2, 29, -1):
        prefix = candidate[:k].rstrip()
        if candidate[k:].lstrip() == prefix:
            candidate = prefix
            break
    return candidate


async def build_client_lines(scenarios: list[dict]) -> dict:
    """Client openers generated ONCE via Cloudflare judge model (not under test).

    Merge-on-miss semantics: existing cached openers are never dropped or
    rewritten; only genuinely missing scenario ids get generated. This keeps
    small EVAL_LIMIT smoke runs from ratcheting the shared fixture down.
    """
    cache_path = OUT_DIR / "client_lines.json"
    client_lines: dict[str, str] = {}
    scen_by_id: dict[str, dict] = {}
    want: list[tuple[str, int]] = []
    for idx, sc in enumerate(scenarios, 1):
        sid = str(sc.get("scenario_id"))
        scen_by_id[sid] = sc
        want.append((sid, idx))
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
        except (OSError, ValueError):
            cached = {}
        for k, v in cached.get("client_lines", {}).items():
            client_lines.setdefault(str(k), v)
        # Preserve richer cached metadata (e.g. full earlier scenario dumps).
        scen_by_id.update({str(c.get("scenario_id")): c for c in cached.get("scenarios", [])})
    missing = [sid for sid, _ in want if sid not in client_lines]
    async with aiohttp.ClientSession() as session:
        sec_url, sec_headers = _secondary_judge_target()
        done = 0
        total_missing = len(missing)
        scenarios_gen = {sid: sc for sid, sc in zip((s for s, _ in want), scenarios)}
        for sid in missing:
            payload = {
                "model": "@cf/zai-org/glm-5.2",
                "messages": [{"role": "system", "content": client_system_prompt(scenarios_gen[sid])}],
                "temperature": 0.8,
                "max_tokens": 1024,
            }
            async with session.post(
                sec_url, json=payload, headers=sec_headers, timeout=aiohttp.ClientTimeout(total=90)
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
            line = (data["choices"][0]["message"].get("content") or "").strip()
            line = re.sub(r"^(Client|Patient)\s*:\s*", "", line).strip(_QC + " \n")
            client_lines[sid] = re.sub(r"\s*\n\s*", " ", line)
            done += 1
            print(f"[client] ({done}/{total_missing}) {sid}", flush=True)
    out = {"scenarios": list(scen_by_id.values()), "client_lines": client_lines}
    cache_path.write_text(json.dumps(out, indent=2))
    return out


async def run_arm(arm_name: str, scenarios: list[dict], client_lines: dict[str, str]) -> list[dict]:
    arm = ARMS[arm_name]
    os.environ["NF_MODEL"] = arm["model"]
    backend = resolve_backend()
    endpoint = arm.get("url", FEATHERLESS_URL)
    print(f"== GENERATION: {arm_name} | backend={backend.name} | model={backend.model} | endpoint={endpoint}", flush=True)
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
            retry_max_tokens, retry_timeout = arm["max_tokens"], arm["timeout_s"]
            for attempt_n in range(1, _MAX_ATTEMPTS + 1):
                try:
                    raw, meta, latency = await call_model(
                        session, arm, sys_p, user_p, max_tokens=retry_max_tokens, timeout_s=retry_timeout
                    )
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
                    f"Original client line: {cl!r}\n\n"
                    + correction_for_reason(gate_reason, draft=extracted)
                    + "\n\nWrite ONLY the therapist's next response (1-3 sentences). "
                    "No speaker labels, no preamble, no quotes surrounding your response."
                )
                if not extracted:
                    # thinking-mode output exhausted the token budget; give the
                    # retry more room and forbid visible reasoning
                    retry_max_tokens = max(arm["max_tokens"], 4096)
                    retry_timeout = max(arm["timeout_s"], 240)
                    user_p += (
                        "\n\nDo NOT show any reasoning, analysis, or draft options. "
                        "Output ONLY the final 1-3 sentence therapist response, nothing else."
                    )
                else:
                    retry_max_tokens, retry_timeout = arm["max_tokens"], arm["timeout_s"]
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
            if not rec.get("response"):
                out[f"{arm}:{sid}"] = {
                    "accepted": False,
                    "needs_human_review": False,
                    "reason": "empty_response (generation error) — not judged",
                    "primary_quality": 0.0,
                    "secondary_quality": 0.0,
                    "primary_reject_reasons": "",
                    "dim_scores": {},
                    "secondary_dim_scores": {},
                }
                print(f"[judge] {arm}:{sid} SKIPPED (empty response)", flush=True)
                return
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
                        force_json=True,
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

        arms = list(gen.keys())
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
        jrows = [
            v for k, v in judge.items()
            if k.startswith(f"{arm}:") and "empty_response" not in str(v.get("reason", ""))
        ]
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

    arms = list(gen.keys())
    stats = {arm: arm_stats(arm) for arm in arms}
    n_items = len(next(iter(gen.values()))) if gen else 0
    lines = [f"# Comparison: {' vs '.join(arms)} (NF pipeline, {n_items} scenarios)", ""]
    lines.append("| metric | " + " | ".join(arms) + " |")
    lines.append("|---|" + "---|" * len(arms))
    keys = ("gate_pass", "first_attempt_pass", "mean_judge_quality", "judge_accept",
            "human_review_rate", "mean_latency", "tok_s")
    labels = ("cliche gate pass", "first-attempt pass", "mean judge quality",
              "judge accept", "human review rate", "mean latency/s", "tok/s")
    for key, label in zip(keys, labels):
        lines.append(f"| {label} | " + " | ".join(str(stats[a][key]) for a in arms) + " |")
    lines.append("")
    for arm in arms:
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
    n = int(os.environ.get("EVAL_LIMIT", "20"))
    scenarios = load_scenarios(n)
    by_sev: dict[str, int] = {}
    for sc in scenarios:
        by_sev[str(sc.get("severity"))] = by_sev.get(str(sc.get("severity")), 0) + 1
    print(f"Loaded {len(scenarios)} scenarios | severity {by_sev}", flush=True)
    scenarios_by_id = {str(sc.get("scenario_id")): sc for sc in scenarios}
    gen_path = {arm: OUT_DIR / f"{arm}_gen.json" for arm in ARMS}
    judge_path = OUT_DIR / "judge.json"

    client = await build_client_lines(scenarios)
    client_lines = client["client_lines"]

    arms = [
        a.strip() for a in os.environ.get("EVAL_ARMS", "boulesis,ornith").split(",") if a.strip() in ARMS
    ] or ["boulesis", "ornith"]
    gen: dict[str, dict] = {}
    for arm in arms:
        needs_gen = "--regen" in argv or not gen_path[arm].exists()
        if needs_gen and "--judge" not in argv:
            rows = await run_arm(arm, scenarios, client_lines)
            gen[arm] = {r["id"]: r for r in rows}
            gen_path[arm].write_text(json.dumps(rows, indent=2))
            print(f"wrote {gen_path[arm]}", flush=True)
        else:
            rows = json.loads(gen_path[arm].read_text())
            gen[arm] = {r["id"]: r for r in rows}

    if "--gen" not in argv:
        judge: dict[str, dict] = {}
        if judge_path.exists():
            judge = json.loads(judge_path.read_text())
        # Only judge entries missing from the cache (partial merge) — avoids
        # re-judging unchanged arms and halves 429 exposure on the shared key.
        pending = {
            arm: {sid: rec for sid, rec in gen[arm].items() if f"{arm}:{sid}" not in judge}
            for arm in gen
        }
        pending = {arm: items for arm, items in pending.items() if items}
        if pending:
            judge.update(await judge_items(pending, scenarios_by_id))
            judge_path.write_text(json.dumps(judge, indent=2))
            print(f"wrote {judge_path}", flush=True)
        return write_report(gen, judge, client)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
