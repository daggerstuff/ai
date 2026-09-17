"""Bulk dual-judge scoring for edge_and_nightmare_generated.jsonl (PIX-4343, blueprint B.3.2/B.3.4).

Judges every generated record in aggregate transcript mode:
  - primary: deepseek/deepseek-v4.1-flash via the Vercel AI Gateway, k=3
    self-consistency, force_json response_format, max_tokens 8192
  - secondary: moonshotai/kimi-k3 via the Vercel AI Gateway, k=1, max_tokens 8192
  - reconcile: dual-consistency diff <= 0.15, accept threshold 0.60,
    self-consistency variance <= 0.05

Resumable: verdicts append incrementally (fsync per record) to
edge_and_nightmare_judged.jsonl keyed by record provenance. Records whose
judging ended in infrastructure failure (429/5xx/transport exhaustion) are
re-judged on rerun; genuine verdicts are kept.

Usage (from ai/):
  python -m training.judge_edge_and_nightmare            # judge all pending
  python -m training.judge_edge_and_nightmare --limit 2  # smoke test
  python -m training.judge_edge_and_nightmare --report-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import aiohttp
from dotenv import load_dotenv

# override=True: the launching shell may carry stale provider keys (AI_GATEWAY_API_KEY etc.).
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

from training.dual_judge import (  # noqa: E402
    ACCEPT_THRESHOLD,
    SECONDARY_MODEL,
    _call_judge_model,
    _secondary_judge_target,
    aggregate_turn_verdicts,
    reconcile_dual,
    runs_self_consistent,
)

logger = logging.getLogger("judge_edge_and_nightmare")

CHECKPOINT_DIR = (
    Path(os.environ["NF_OUTPUT_DIR"])
    if os.environ.get("NF_OUTPUT_DIR")
    else Path(__file__).resolve().parent / "output" / "nightmare_fuel" / "checkpoints"
)
IN_GENERATED = Path(os.environ.get("NF_JUDGE_INPUT", CHECKPOINT_DIR / "edge_and_nightmare_generated.jsonl"))
OUT_JUDGED = Path(os.environ.get("NF_JUDGE_OUTPUT", CHECKPOINT_DIR / "edge_and_nightmare_judged.jsonl"))
OUT_REPORT = Path(os.environ.get("NF_JUDGE_REPORT", CHECKPOINT_DIR / "edge_and_nightmare_judge_report.md"))

JUDGE_URL = os.environ.get("NF_JUDGE_URL", "https://ai-gateway.vercel.sh/v1/chat/completions")
PRIMARY_MODEL_ID = os.environ.get("NF_JUDGE_PRIMARY_MODEL", "deepseek/deepseek-v4.1-flash")

JUDGE_K = int(os.environ.get("NF_JUDGE_K", "3"))
# Both judges cost 4 inference-provider concurrency units (vs Wayfarer's 1):
# keep in-flight calls low to stay clear of the per-key concurrency ceiling.
JUDGE_CONCURRENCY = int(os.environ.get("NF_JUDGE_CONCURRENCY", "2"))
JUDGE_ATTEMPTS = int(os.environ.get("NF_JUDGE_ATTEMPTS", "4"))
JUDGE_CALL_TIMEOUT = int(os.environ.get("NF_JUDGE_TIMEOUT", "240"))
CIRCUIT_BREAKER_THRESHOLD = int(os.environ.get("NF_JUDGE_ABORT_AFTER", "15"))

INFRA_PREFIXES = ("http_429", "http_408", "http_500", "http_502", "http_503", "http_504", "http_402", "transport_error", "json_parse_error", "empty_judge_output")


def _is_infra_failure(reason: str) -> bool:
    return reason.startswith(INFRA_PREFIXES)


def record_key(record: dict[str, Any]) -> str:
    prov = record.get("provenance", {})
    if prov.get("scenario_id"):
        return f"nf:{prov['scenario_id']}"
    return "edge:{domain}:{family}:{difficulty}:{ambiguity}:{variation}".format(
        domain=prov.get("domain", "?"),
        family=prov.get("family", record.get("family", "?")),
        difficulty=prov.get("difficulty", record.get("difficulty", "?")),
        ambiguity=prov.get("ambiguity", record.get("ambiguity", "?")),
        variation=record.get("variation", prov.get("variation", "?")),
    )


def join_pairs(record: dict[str, Any]) -> tuple[str, str] | None:
    """Aggregate transcript mode: join user turns as reference, assistant turns as candidate."""
    messages = list(record.get("messages", []))
    refs: list[str] = []
    cands: list[str] = []
    for i in range(0, len(messages) - 1, 2):
        user_msg = messages[i]
        assistant_msg = messages[i + 1]
        if user_msg.get("role") == "user" and assistant_msg.get("role") == "assistant":
            refs.append(str(user_msg.get("content", "")))
            cands.append(str(assistant_msg.get("content", "")))
    if not cands:
        return None
    return "\n".join(refs), "\n".join(cands)


def load_done_keys(path: Path) -> tuple[set[str], int]:
    """Keys with settled verdicts (infra failures excluded so reruns heal them)."""
    done: set[str] = set()
    infra_lines = 0
    if not path.exists():
        return done, infra_lines
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("infra_failed"):
            infra_lines += 1
            continue
        done.add(rec["key"])
    return done, infra_lines


async def call_with_retry(
    session: aiohttp.ClientSession,
    *,
    url: str,
    model: str,
    candidate: str,
    reference: str,
    headers: dict[str, str] | None = None,
    force_json: bool = False,
    max_tokens: int | None = None,
) -> tuple[Any, int]:
    """Call one judge model, retrying infra failures with exponential backoff."""
    verdict = None
    attempts = 0
    for attempt in range(1, JUDGE_ATTEMPTS + 1):
        attempts = attempt
        verdict = await _call_judge_model(
            session,
            url=url,
            model=model,
            candidate_content=candidate,
            reference_content=reference,
            headers=headers,
            timeout=JUDGE_CALL_TIMEOUT,
            max_tokens=max_tokens,
            force_json=force_json,
        )
        if not _is_infra_failure(verdict.reject_reason) or attempt == JUDGE_ATTEMPTS:
            return verdict, attempts
        logger.warning("judge call %s failed (attempt %d/%d): %s — retrying", model, attempt, JUDGE_ATTEMPTS, verdict.reject_reason[:120])
        await asyncio.sleep(min(5.0 * (2 ** (attempt - 1)), 60.0))
    return verdict, attempts


async def judge_record(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    record: dict[str, Any],
) -> dict[str, Any]:
    key = record_key(record)
    pair = join_pairs(record)
    if pair is None:
        return {
            "key": key,
            "infra_failed": False,
            "accepted": False,
            "needs_human_review": True,
            "reason": "no_user_assistant_pairs",
            "primary_quality": 0.0,
            "secondary_quality": 0.0,
            "self_consistency_consistent": False,
            "primary_run_scores": [],
            "primary_dim_scores": {},
            "secondary_dim_scores": {},
            "attempts": 0,
            "latency_s": 0.0,
        }
    reference, candidate = pair
    _judge_key = os.environ.get("AI_GATEWAY_API_KEY", "") or os.environ.get(
        "FEATHERLESS_API_KEY", ""
    )
    prim_headers = {"Authorization": f"Bearer {_judge_key}"}
    sec_url, sec_headers = _secondary_judge_target()
    t0 = time.monotonic()
    async with sem:
        results = await asyncio.gather(
            *(call_with_retry(
                session,
                url=JUDGE_URL,
                model=PRIMARY_MODEL_ID,
                candidate=candidate,
                reference=reference,
                headers=prim_headers,
                force_json=True,
            ) for _ in range(JUDGE_K))
        )
        secondary, sec_attempts = await call_with_retry(
            session,
            url=sec_url,
            model=SECONDARY_MODEL,
            candidate=candidate,
            reference=reference,
            headers=sec_headers,
            force_json=True,
            max_tokens=8192,  # GLM-5.3 reasons in-band; low budgets starve content (validated at 8192)
        )
    latency = time.monotonic() - t0
    prims = [v for v, _ in results]
    total_attempts = sum(a for _, a in results) + sec_attempts

    prim_infra = all(_is_infra_failure(v.reject_reason) and v.quality_score == 0.0 for v in prims)
    sec_infra = _is_infra_failure(secondary.reject_reason) and secondary.quality_score == 0.0
    if prim_infra and sec_infra:
        return {
            "key": key,
            "infra_failed": True,
            "accepted": False,
            "needs_human_review": True,
            "reason": f"infra_failure: primary={prims[0].reject_reason[:80]}; secondary={secondary.reject_reason[:80]}",
            "primary_quality": 0.0,
            "secondary_quality": 0.0,
            "self_consistency_consistent": False,
            "primary_run_scores": [],
            "primary_dim_scores": {},
            "secondary_dim_scores": {},
            "attempts": total_attempts,
            "latency_s": latency,
        }

    consistent = runs_self_consistent(prims)
    primary = aggregate_turn_verdicts(prims)
    result = reconcile_dual(primary, secondary)
    if not consistent:
        result.accepted = False
        result.needs_human_review = True
        result.reason = f"self_consistency_variance_exceeded; {result.reason}"
    return {
        "key": key,
        "infra_failed": False,
        "source": record.get("source", ""),
        "family": record.get("family", ""),
        "accepted": result.accepted,
        "needs_human_review": result.needs_human_review,
        "reason": result.reason,
        "primary_quality": round(primary.quality_score, 3),
        "secondary_quality": round(secondary.quality_score, 3),
        "primary_reject_reason": primary.reject_reason,
        "secondary_reject_reason": secondary.reject_reason,
        "self_consistency_consistent": consistent,
        "primary_run_scores": [round(v.quality_score, 3) for v in prims],
        "primary_dim_scores": {k: round(v, 3) for k, v in primary.dim_scores.items()},
        "secondary_dim_scores": {k: round(v, 3) for k, v in secondary.dim_scores.items()},
        "attempts": total_attempts,
        "latency_s": round(latency, 1),
    }


def maybe_init_wandb(config: dict[str, Any]):
    try:
        import wandb

        if not os.environ.get("WANDB_API_KEY"):
            return None
        return wandb.init(
            project=os.environ.get("WANDB_PROJECT", "pixelated-empathy-kan28"),
            job_type="nf_judging",
            config=config,
        )
    except Exception as exc:  # wandb down must never block judging
        logger.warning("wandb init failed (continuing without): %s", exc)
        return None


def write_report(verdicts: list[dict[str, Any]], total_records: int) -> None:
    settled = [v for v in verdicts if not v.get("infra_failed")]
    infra = [v for v in verdicts if v.get("infra_failed")]
    n = len(settled)
    accepted = sum(1 for v in settled if v["accepted"])
    hrr = sum(1 for v in settled if v["needs_human_review"])
    prim_q = [v["primary_quality"] for v in settled if v["primary_quality"] > 0]
    sec_q = [v["secondary_quality"] for v in settled if v["secondary_quality"] > 0]
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else 0.0

    dims: dict[str, list[float]] = {}
    for v in settled:
        for d, s in v.get("primary_dim_scores", {}).items():
            dims.setdefault(d, []).append(s)

    by_src: dict[str, list[dict[str, Any]]] = {}
    for v in verdicts:
        src = (v.get("source") or ("nightmare_fuel_predefined" if v["key"].startswith("nf:") else "edge")).split("clinical_edge_case_")[-1]
        by_src.setdefault(src, []).append(v)

    lines = [
        "# Edge & Nightmare dual-judge report",
        "",
        f"- Records generated: **{total_records}**",
        f"- Settled verdicts: **{n}** (infra failures re-tried: {len(infra)})",
        f"- Accepted (dual-consistent, q >= {ACCEPT_THRESHOLD}): **{accepted}/{n}** ({round(100 * accepted / n, 1) if n else 0}%)",
        f"- Needs human review: **{hrr}/{n}** ({round(100 * hrr / n, 1) if n else 0}%)",
        f"- Mean primary quality: **{mean(prim_q)}** (n={len(prim_q)})",
        f"- Mean secondary quality: **{mean(sec_q)}** (n={len(sec_q)})",
        "",
        "## Primary dimension means",
        "",
    ]
    for d in ("relevance", "accuracy", "helpfulness", "style", "safety"):
        lines.append(f"- {d}: {mean(dims.get(d, []))}")
    lines += ["", "## By family", "", "| Family | Judged | Accepted | HR | Mean primary q |", "|---|---|---|---|---|"]
    for src in sorted(by_src):
        rows = [v for v in by_src[src] if not v.get("infra_failed")]
        acc = sum(1 for v in rows if v["accepted"])
        hr = sum(1 for v in rows if v["needs_human_review"])
        q = mean([v["primary_quality"] for v in rows if v["primary_quality"] > 0])
        lines.append(f"| {src} | {len(rows)} | {acc} | {hr} | {q} |")
    lines += ["", "## Flagged for human review", ""]
    for v in sorted(settled, key=lambda x: x["primary_quality"]):
        if v["needs_human_review"]:
            lines.append(f"- `{v['key']}` q={v['primary_quality']}/{v['secondary_quality']} — {v['reason'][:140]}")
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main_async(limit: int | None) -> None:
    records = [json.loads(l) for l in IN_GENERATED.read_text(encoding="utf-8").splitlines() if l.strip()]
    done_keys, prior_infra = load_done_keys(OUT_JUDGED)
    pending = [r for r in records if record_key(r) not in done_keys]
    if limit:
        pending = pending[:limit]
    print(
        f"[judge] records={len(records)} settled={len(done_keys)} (prior infra retry-able: {prior_infra}) "
        f"pending={len(pending)} concurrency={JUDGE_CONCURRENCY} k={JUDGE_K} primary={PRIMARY_MODEL_ID}",
        flush=True,
    )

    run = maybe_init_wandb(
        {
            "stage": "dual_judge",
            "records_total": len(records),
            "pending": len(pending),
            "self_consistency_k": JUDGE_K,
            "concurrency": JUDGE_CONCURRENCY,
            "accept_threshold": ACCEPT_THRESHOLD,
            "primary_model": PRIMARY_MODEL_ID,
            "secondary_model": SECONDARY_MODEL,
        }
    )

    sem = asyncio.Semaphore(JUDGE_CONCURRENCY)
    write_lock = asyncio.Lock()
    state = {"judged": 0, "accepted": 0, "hrr": 0, "infra": 0, "infra_since_success": 0, "abort": False}
    t_start = time.monotonic()

    if run:
        stop = asyncio.Event()

        async def metrics_loop() -> None:
            while not stop.is_set():
                await asyncio.sleep(60)
                try:
                    run.log({k: v for k, v in state.items() if k != "abort"} | {"elapsed_s": round(time.monotonic() - t_start, 1)})
                except Exception:
                    pass

        metrics_task = asyncio.create_task(metrics_loop())

    async with aiohttp.ClientSession() as session:
        async def worker(rec: dict[str, Any]) -> None:
            if state["abort"]:
                return
            key = record_key(rec)
            try:
                verdict = await judge_record(session, sem, rec)
            except Exception as exc:
                logger.exception("judging %s crashed: %s", key, exc)
                verdict = {
                    "key": key, "infra_failed": True, "accepted": False, "needs_human_review": True,
                    "reason": f"worker_exception: {exc}", "primary_quality": 0.0, "secondary_quality": 0.0,
                    "self_consistency_consistent": False, "primary_run_scores": [], "primary_dim_scores": {},
                    "secondary_dim_scores": {}, "attempts": 0, "latency_s": 0.0,
                }
            async with write_lock:
                with OUT_JUDGED.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(verdict) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                state["judged"] += 1
                if verdict.get("infra_failed"):
                    state["infra"] += 1
                    state["infra_since_success"] += 1
                else:
                    state["infra_since_success"] = 0
                    if verdict["accepted"]:
                        state["accepted"] += 1
                    if verdict["needs_human_review"]:
                        state["hrr"] += 1
                if state["infra_since_success"] >= CIRCUIT_BREAKER_THRESHOLD:
                    state["abort"] = True
                pct = 100 * state["judged"] / max(len(pending), 1)
                print(
                    f"[judge] ({state['judged']}/{len(pending)} {pct:.0f}%) {key} "
                    f"q={verdict.get('primary_quality', 0):.2f}/{verdict.get('secondary_quality', 0):.2f} "
                    f"accept={verdict['accepted']}{' HR' if verdict['needs_human_review'] else ''}"
                    f"{' INFRA-RETRY' if verdict.get('infra_failed') else ''} "
                    f"({verdict['latency_s']}s, attempts={verdict['attempts']})",
                    flush=True,
                )

        await asyncio.gather(*(asyncio.create_task(worker(r)) for r in pending))

    if run:
        stop.set()
        await metrics_task
        try:
            run.log({k: v for k, v in state.items() if k != "abort"} | {"elapsed_s": round(time.monotonic() - t_start, 1)}, commit=True)
            run.finish()
        except Exception:
            pass

    print(
        f"[judge] done: judged={state['judged']} accepted={state['accepted']} "
        f"human_review={state['hrr']} infra_failed={state['infra']} elapsed={round(time.monotonic() - t_start) / 60:.1f}min",
        flush=True,
    )
    all_verdicts: list[dict[str, Any]] = []
    if OUT_JUDGED.exists():
        for raw in OUT_JUDGED.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                all_verdicts.append(json.loads(raw))
    write_report(all_verdicts, len(records))
    print(f"[judge] report -> {OUT_REPORT}", flush=True)
    if state["abort"]:
        print(f"[judge] ABORTED after {CIRCUIT_BREAKER_THRESHOLD} consecutive infra failures — checkpoint preserved; rerun to resume", flush=True)
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Judge only first N pending records (smoke test)")
    parser.add_argument("--report-only", action="store_true", help="Rebuild report from existing verdicts, no judging")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if args.report_only:
        verdicts = [json.loads(l) for l in OUT_JUDGED.read_text(encoding="utf-8").splitlines() if l.strip()] if OUT_JUDGED.exists() else []
        write_report(verdicts, len([l for l in IN_GENERATED.read_text(encoding="utf-8").splitlines() if l.strip()]))
        print(f"[judge] report -> {OUT_REPORT}", flush=True)
        return
    asyncio.run(main_async(args.limit))


if __name__ == "__main__":
    main()
