"""Third-judge tie-break pass over dual-judge human-review rows (blueprint B.3.2 close-out).

The dual judge (deepseek/deepseek-v4.1-flash k=3 self-consistency vs
moonshotai/kimi-k3 k=1) flags a record for human review when
|primary - secondary| > 0.15. This pass asks a third, family-independent
judge (minimax/minimax-m3 via the Vercel AI Gateway) to score the same
aggregate transcript, then resolves each contested row by 2-of-3 agreement
(either pair of judges may form the agreeing pair):

  - top pair agrees within 0.15, consensus (middle score) >= 0.60,
    and no safety-related reject flag on any judge   -> tiebreak_accept
  - top pair agrees within 0.15, consensus < 0.60    -> tiebreak_reject
  - lower pair agrees, consensus < 0.60 (lone high outlier) -> tiebreak_reject
  - lower pair agrees at >= 0.60 with a high outlier  -> tiebreak_manual
  - no pair of judges agrees within 0.15             -> tiebreak_contested (manual review)
  - agreeing pair at >= 0.60 but any judge raised a safety flag -> tiebreak_manual

Safety flags (crisis, risk, self-harm, means, escalation, ...) always force
manual review even on 2/3 agreement: a safety red flag on therapeutic
training data is never settled by vote.

Resumable: rows with a settled tiebreak verdict are skipped on rerun;
third-judge infrastructure failures are re-tried. Incremental rows append to
edge_and_nightmare_judged_v3.jsonl (fsync per record); at the end the full
judged_v2 file is merged so v3 carries all 212 rows (HR rows enriched with
third_* / tiebreak_* fields, others unchanged).

Usage (from ai/, repo-root venv so wandb is available):
  python -m training.tiebreak_hr --probe    # judge first HR row, print verdict
  python -m training.tiebreak_hr            # full pass
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import wandb
except ImportError:
    wandb = None

try:
    import pandas as pd
except ImportError:
    pd = None

import aiohttp
from dotenv import load_dotenv

# override=True: the launching shell may carry stale provider keys (AI_GATEWAY_API_KEY etc.).
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

from training.dual_judge import (  # noqa: E402
    ACCEPT_THRESHOLD,
    DUAL_CONSISTENCY_DIFF_MAX,
    _call_judge_model,
    _judge_auth_header,
)
from training.judge_edge_and_nightmare import (  # noqa: E402
    CHECKPOINT_DIR,
    IN_GENERATED,
    INFRA_PREFIXES,
    JUDGE_URL,
    join_pairs,
    record_key,
)

logger = logging.getLogger("tiebreak_hr")

THIRD_MODEL_ID = os.environ.get("NF_TIEBREAK_MODEL", "minimax/minimax-m3")
TIEBREAK_CONCURRENCY = int(os.environ.get("NF_TIEBREAK_CONCURRENCY", "2"))
TIEBREAK_ATTEMPTS = int(os.environ.get("NF_TIEBREAK_ATTEMPTS", "4"))
TIEBREAK_INPUT = Path(os.environ.get("NF_TIEBREAK_INPUT", CHECKPOINT_DIR / "edge_and_nightmare_judged_v2.jsonl"))
OUT_V3 = Path(os.environ.get("NF_TIEBREAK_OUTPUT", CHECKPOINT_DIR / "edge_and_nightmare_judged_v3.jsonl"))

# A reject flag containing any of these substrings is a safety red flag.
SAFETY_FLAG_KEYWORDS = ("crisis", "safety", "risk", "self_harm", "suicid", "lethal", "means", "escalat", "harm")

TIEBREAK_SETTLED = ("tiebreak_accept", "tiebreak_reject", "tiebreak_contested", "tiebreak_manual")


def _is_infra_failure(reason: str) -> bool:
    return reason.startswith(INFRA_PREFIXES)


def _row_is_hr(row: dict[str, Any]) -> bool:
    disposition = row.get("disposition")
    if disposition is not None:
        return disposition == "human_review"
    return bool(row.get("needs_human_review")) and not row.get("accepted") and not row.get("infra_failed")


def _row_quality(row: dict[str, Any]) -> tuple[Any, Any, str, str]:
    primary_q = row.get("primary_quality")
    if primary_q is None:
        primary_q = row.get("primary_overall")
    secondary_q = row.get("secondary_quality")
    if secondary_q is None:
        secondary_q = row.get("secondary_overall")
    flags = ";".join(row.get("flags") or [])
    primary_reason = row.get("primary_reject_reason", "") or flags
    secondary_reason = row.get("secondary_reject_reason", "") or flags
    return primary_q, secondary_q, primary_reason, secondary_reason


def _safety_flags(*reject_reasons: str) -> list[str]:
    flags: list[str] = []
    for raw in reject_reasons:
        for flag in (f.strip() for f in raw.split(";") if f.strip()):
            if any(k in flag for k in SAFETY_FLAG_KEYWORDS):
                flags.append(flag)
    return sorted(set(flags))


def tiebreak_decide(
    primary_q: float,
    secondary_q: float,
    third_q: float,
    reject_reasons: list[str],
) -> tuple[str, str]:
    """2-of-3 agreement rule (either agreeing pair counts). Returns (verdict, reason).

    Note: minimax-m3 quantizes its scores into ~8 bands (observed
    {0.18, 0.32, 0.42, 0.62, 0.72, 0.78, 0.82, 0.91}), so "agreement within
    0.15" with the third judge is effectively a band match.
    """
    a, b, c = sorted((primary_q, secondary_q, third_q), reverse=True)
    if a - b <= DUAL_CONSISTENCY_DIFF_MAX:
        # top pair agrees
        if b < ACCEPT_THRESHOLD:
            return "tiebreak_reject", f"top-two agree at {b:.3f} below accept threshold {ACCEPT_THRESHOLD}"
        flags = _safety_flags(*reject_reasons)
        if flags:
            return (
                "tiebreak_manual",
                f"top-two agree at {b:.3f} >= {ACCEPT_THRESHOLD} "
                f"but safety flags present: {', '.join(flags)}",
            )
        return "tiebreak_accept", f"top-two agree at {b:.3f} >= {ACCEPT_THRESHOLD}, no safety flags"
    if b - c <= DUAL_CONSISTENCY_DIFF_MAX:
        # lower pair agrees; a is the high outlier
        if b >= ACCEPT_THRESHOLD:
            return (
                "tiebreak_manual",
                f"lower pair agrees at {c:.3f}-{b:.3f} "
                f"but top judge {a:.3f} is high outlier; human decides",
            )
        return (
            "tiebreak_reject",
            f"lower pair agrees at {c:.3f}-{b:.3f}, consensus {b:.3f} below "
            f"threshold (top judge {a:.3f} alone above)",
        )
    return (
        "tiebreak_contested",
        f"no pair of 3 judges agrees within {DUAL_CONSISTENCY_DIFF_MAX} "
        f"(scores {a:.3f}/{b:.3f}/{c:.3f})",
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _load_generated_by_key(path: Path) -> dict[str, dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for record in _load_jsonl(path):
        by_key[record_key(record)] = record
    return by_key


async def _call_third_with_retry(
    session: aiohttp.ClientSession, *, url: str, model: str, candidate: str, reference: str, headers: dict[str, str]
) -> tuple[Any, int]:
    verdict = None
    attempts = 0
    for attempt in range(1, TIEBREAK_ATTEMPTS + 1):
        attempts = attempt
        verdict = await _call_judge_model(
            session,
            url=url,
            model=model,
            candidate_content=candidate,
            reference_content=reference,
            headers=headers,
            force_json=True,
        )
        if not _is_infra_failure(verdict.reject_reason):
            return verdict, attempts
        logger.warning("third judge %s failed (attempt %d/%d): %s — retrying", model, attempt, TIEBREAK_ATTEMPTS, verdict.reject_reason[:120])
        await asyncio.sleep(min(5.0 * (2 ** (attempt - 1)), 60.0))
    return verdict, attempts


async def _tiebreak_one(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    row: dict[str, Any],
    record: dict[str, Any],
    headers: dict[str, str],
    url: str,
) -> dict[str, Any]:
    key = row["key"]
    pair = join_pairs(record)
    if pair is None:
        row.update(
            tiebreak_verdict="tiebreak_contested",
            tiebreak_reason="no_user_assistant_pairs_in_generated_record",
            third_quality=None,
            third_reject_reason="no_user_assistant_pairs",
            third_dim_scores=None,
            tiebreak_model=THIRD_MODEL_ID,
        )
        return row
    reference, candidate = pair
    async with sem:
        verdict, attempts = await _call_third_with_retry(
            session, url=url, model=THIRD_MODEL_ID, candidate=candidate, reference=reference, headers=headers
        )
    if _is_infra_failure(verdict.reject_reason):
        row.update(
            tiebreak_verdict="infra_failed",
            tiebreak_reason=verdict.reject_reason,
            third_quality=None,
            third_reject_reason=verdict.reject_reason,
            third_dim_scores=None,
            third_attempts=attempts,
            tiebreak_model=THIRD_MODEL_ID,
        )
        return row
    third_q = round(verdict.quality_score, 3)
    row.update(
        third_quality=third_q,
        third_reject_reason=verdict.reject_reason,
        third_dim_scores=verdict.dim_scores,
        third_reasoning=verdict.reasoning,
        third_attempts=attempts,
        tiebreak_model=THIRD_MODEL_ID,
    )
    primary_q, secondary_q, primary_reason, secondary_reason = _row_quality(row)
    verdict_name, reason = tiebreak_decide(
        primary_q, secondary_q, third_q,
        [primary_reason, secondary_reason, verdict.reject_reason],
    )
    row.update(tiebreak_verdict=verdict_name, tiebreak_reason=reason)
    return row


def _init_wandb() -> Any | None:
    if wandb is None:
        return None
    try:
        return wandb.init(
            project=os.environ.get("WANDB_PROJECT"),
            name="hr70_tiebreak",
            config={
                "third_model": THIRD_MODEL_ID,
                "url": JUDGE_URL,
                "concurrency": TIEBREAK_CONCURRENCY,
                "input": str(TIEBREAK_INPUT),
            },
        )
    except Exception as exc:
        logger.warning("wandb init failed, continuing without tracing: %s", exc)
        return None


async def main_async(limit: int | None, probe: bool) -> None:
    generated = _load_generated_by_key(IN_GENERATED)
    judged = _load_jsonl(TIEBREAK_INPUT)
    hr_rows = [r for r in judged if _row_is_hr(r)]
    logger.info("judged=%d hr=%d generated=%d", len(judged), len(hr_rows), len(generated))

    settled: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(OUT_V3):
        if row.get("tiebreak_verdict") in TIEBREAK_SETTLED and not _is_infra_failure(row.get("tiebreak_reason", "") or ""):
            settled[row["key"]] = row
    pending = [r for r in hr_rows if r["key"] not in settled]
    missing = [r["key"] for r in pending if r["key"] not in generated]
    if missing:
        raise SystemExit(f"{len(missing)} HR keys missing from generated file, e.g. {missing[:3]}")
    if probe:
        pending = pending[:1]
    if limit is not None:
        pending = pending[:limit]
    logger.info("settled=%d pending=%d model=%s", len(settled), len(pending), THIRD_MODEL_ID)
    if not pending:
        _merge_v3(judged, settled)
        _print_summary(settled)
        return

    headers = _judge_auth_header()
    sem = asyncio.Semaphore(TIEBREAK_CONCURRENCY)
    wandb_run = _init_wandb()
    done = 0
    started = time.monotonic()
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:

            async def work(row: dict[str, Any]) -> dict[str, Any]:
                return await _tiebreak_one(session, sem, row, generated[row["key"]], headers, JUDGE_URL)

            for coro in asyncio.as_completed([work(r) for r in pending]):
                row = await coro
                done += 1
                _append_row(OUT_V3, row)
                if row["tiebreak_verdict"] == "infra_failed":
                    logger.error("[%3d/%d] %s INFRA: %s", done, len(pending), row["key"], row["tiebreak_reason"][:100])
                else:
                    logger.info(
                        "[%3d/%d] %s p=%s s=%s t=%s -> %s",
                        done, len(pending), row["key"],
                        row["primary_quality"], row["secondary_quality"], row["third_quality"], row["tiebreak_verdict"],
                    )
                if wandb_run and done % 10 == 0:
                    wandb_run.log({"tiebreak_progress": done, "tiebreak_total": len(pending)})
    finally:
        if wandb_run:
            elapsed = time.monotonic() - started
            _merge_v3(judged, None, incremental=True)
            _log_wandb_summary(wandb_run, _load_final(), elapsed)
            wandb_run.finish()

    _merge_v3(judged, None)
    _print_summary(_load_final())


def _load_final() -> dict[str, dict[str, Any]]:
    """All settled (and infra-failed) tiebreak rows currently in the incremental file."""
    out: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(OUT_V3):
        if row.get("tiebreak_verdict") in TIEBREAK_SETTLED or row.get("tiebreak_verdict") == "infra_failed":
            out[row["key"]] = row
    return out


def _merge_v3(
    judged: list[dict[str, Any]],
    settled: dict[str, dict[str, Any]] | None,
    incremental: bool = False,
) -> None:
    """Write the full v3 file: every judged row, HR rows enriched with tiebreak fields."""
    enriched: dict[str, dict[str, Any]] = dict(settled or {})
    for row in _load_jsonl(OUT_V3):
        if row.get("key") and row.get("tiebreak_verdict") in TIEBREAK_SETTLED:
            enriched[row["key"]] = row
    rows_out: list[dict[str, Any]] = []
    for row in judged:
        if row["key"] in enriched:
            merged = {**row, **{k: v for k, v in enriched[row["key"]].items() if k != "messages"}}
            rows_out.append(merged)
        else:
            rows_out.append(row)
    tmp = OUT_V3.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    if not incremental:
        tmp.replace(OUT_V3)
    logger.info("merged v3 -> %s (%d rows)", OUT_V3, len(rows_out))


def _print_summary(settled: dict[str, dict[str, Any]]) -> None:
    counts = Counter(v["tiebreak_verdict"] for v in settled.values())
    logger.info("=== tie-break summary (%d HR rows settled) ===", len(settled))
    for name in ("tiebreak_accept", "tiebreak_reject", "tiebreak_contested", "tiebreak_manual", "infra_failed"):
        logger.info("  %-22s %d", name, counts.get(name, 0))
    accepted = sorted(v["key"] for v in settled.values() if v["tiebreak_verdict"] == "tiebreak_accept")
    if accepted:
        logger.info("accepted keys (%d):", len(accepted))
        for k in accepted:
            logger.info("  %s", k)
    for name in ("tiebreak_manual", "tiebreak_contested"):
        keys = sorted(v["key"] for v in settled.values() if v["tiebreak_verdict"] == name)
        if keys:
            logger.info("%s keys (%d):", name, len(keys))
            for k in keys:
                v = settled[k]
                logger.info(
                    "  %s  p=%s s=%s t=%s  %s",
                    k,
                    v["primary_quality"],
                    v["secondary_quality"],
                    v.get("third_quality"),
                    v["tiebreak_reason"][:110],
                )


def _log_wandb_summary(wandb_run: Any, settled: dict[str, dict[str, Any]], elapsed: float) -> None:
    if wandb is None:
        return
    try:
        counts = Counter(v["tiebreak_verdict"] for v in settled.values())
        rows = [
            {
                "key": v["key"],
                "primary_quality": v.get("primary_quality"),
                "secondary_quality": v.get("secondary_quality"),
                "third_quality": v.get("third_quality"),
                "tiebreak_verdict": v.get("tiebreak_verdict"),
                "tiebreak_reason": (v.get("tiebreak_reason") or "")[:200],
                "wall_s_total": elapsed,
            }
            for v in sorted(settled.values(), key=lambda x: x["key"])
        ]
        wandb_run.log(
            {
                "tiebreak_accept": counts.get("tiebreak_accept", 0),
                "tiebreak_reject": counts.get("tiebreak_reject", 0),
                "tiebreak_contested": counts.get("tiebreak_contested", 0),
                "tiebreak_manual": counts.get("tiebreak_manual", 0),
                "infra_failed": counts.get("infra_failed", 0),
                "elapsed_s": elapsed,
            },
        )
        wandb_run.log({"tiebreak_rows": wandb.Table(dataframe=_to_frame(rows))})
    except Exception as exc:
        logger.warning("wandb summary log failed: %s", exc)


def _to_frame(rows: list[dict[str, Any]]) -> Any:
    if pd is None:
        return rows
    return pd.DataFrame(rows)


def readjudicate() -> None:
    """Re-derive tiebreak verdicts from stored p/s/t scores (no LLM calls).

    Used after the decide-rule changes so already-judged rows are
    re-adjudicated deterministically without re-hitting the gateway.
    """
    rows = _load_jsonl(OUT_V3)
    changed = 0
    for row in rows:
        if row.get("tiebreak_verdict") in TIEBREAK_SETTLED and row.get("third_quality") is not None:
            primary_q, secondary_q, primary_reason, secondary_reason = _row_quality(row)
            verdict_name, reason = tiebreak_decide(
                primary_q,
                secondary_q,
                row["third_quality"],
                [primary_reason, secondary_reason, row.get("third_reject_reason", "")],
            )
            if verdict_name != row["tiebreak_verdict"]:
                logger.info("readjudicate %s: %s -> %s", row["key"], row["tiebreak_verdict"], verdict_name)
                changed += 1
            row["tiebreak_verdict"] = verdict_name
            row["tiebreak_reason"] = reason
    tmp = OUT_V3.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(OUT_V3)
    logger.info("readjudicate: %d rows rewritten, %d verdicts changed", len(rows), changed)
    _print_summary(_load_final())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Tie-break only first N pending HR rows (smoke test)")
    parser.add_argument("--probe", action="store_true", help="Judge only the first pending HR row and print the verdict")
    parser.add_argument("--readjudicate", action="store_true", help="Re-derive verdicts from stored scores, no LLM calls")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if args.readjudicate:
        readjudicate()
        return
    asyncio.run(main_async(args.limit, args.probe))


if __name__ == "__main__":
    main()
