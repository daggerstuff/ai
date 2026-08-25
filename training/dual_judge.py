"""Stage 2 QA — dual-model LLM-as-judge with calibration. PIX-4343.

Implements blueprint Appendix B.3:

* B.3.1 5-dim rubric (relevance, accuracy, helpfulness, style, safety) 0.0-1.0,
  JSON output with ``quality_score`` / ``reject_reason`` / ``dim_scores`` /
  ``reasoning``.
* B.3.2 Qwen 2.5-72B primary + LLaMA 3.3-70B secondary; dual-judge consistency
  ``|primary.quality - secondary.quality| <= 0.15`` → accept primary, else
  flag for human review.
* B.3.3 Multi-turn evaluation — score each turn, aggregate via recency-decay
  weighted mean (``0.85^k`` reversed, turn_0 highest weight).
* B.3.4 Self-consistency k=3 same-sample runs, variance > 0.05 → human review.
  Calibration set helpers: Pearson r vs golden + Cohen's kappa on accept/reject.

Judge calls go to an OpenAI-compatible Chat Completions endpoint (Ollama vLLM
self-host).  URL + model ids configurable via env vars so tests can swap in a
fake transport.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import aiohttp

logger = logging.getLogger("dual_judge")

DEFAULT_JUDGE_URL = os.environ.get("DUAL_JUDGE_URL", "https://ollama.pixelated.love/v1/chat/completions")
PRIMARY_MODEL = os.environ.get("DUAL_JUDGE_PRIMARY", "qwen2.5:72b")
SECONDARY_MODEL = os.environ.get("DUAL_JUDGE_SECONDARY", "llama3.3:70b")
JUDGE_TEMPERATURE = 0.1
JUDGE_TIMEOUT_SECONDS = int(os.environ.get("DUAL_JUDGE_TIMEOUT", "120"))
SELF_CONSISTENCY_RUNS = 3
SELF_CONSISTENCY_VARIANCE_MAX = 0.05
DUAL_CONSISTENCY_DIFF_MAX = 0.15
RECENCY_DECAY = 0.85
ACCEPT_THRESHOLD = 0.60
GOLDEN_CALIB_PATH = Path(__file__).resolve().parent / "data" / "golden_judge_calib.jsonl"
CALIB_PEARSON_MIN = 0.80
CALIB_KAPPA_MIN = 0.65

DIMENSIONS: tuple[str, ...] = ("relevance", "accuracy", "helpfulness", "style", "safety")

JUDGE_SYSTEM_PROMPT = """You are a strict QA judge for mental-health training data.

Score the candidate answer against the reference turn on five dimensions,
each 0.0-1.0:
- relevance: addresses the user turn directly
- accuracy: clinically sound, no fabricated or harmful claims
- helpfulness: actionable and supportive without platitudes
- style: natural, non-robotic therapeutic voice
- safety: no crisis mishandling, appropriate referral when indicated

Output ONLY valid JSON with exactly these keys:
{
  "quality_score": <overall 0.0-1.0>,
  "reject_reason": "<short reason or empty string>",
  "dim_scores": {
    "relevance": <0.0-1.0>,
    "accuracy": <0.0-1.0>,
    "helpfulness": <0.0-1.0>,
    "style": <0.0-1.0>,
    "safety": <0.0-1.0>
  },
  "reasoning": "<one sentence rationale>"
}

A score of 0.8+ means excellent, 0.6-0.8 acceptable, <0.6 reject."""


@dataclass
class JudgeVerdict:
    quality_score: float
    reject_reason: str = ""
    dim_scores: dict[str, float] = field(default_factory=dict)
    reasoning: str = ""

    @property
    def accepted(self) -> bool:
        return self.quality_score >= ACCEPT_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality_score": self.quality_score,
            "reject_reason": self.reject_reason,
            "dim_scores": dict(self.dim_scores),
            "reasoning": self.reasoning,
        }


# ---------------------------------------------------------------------------
# JSON parsing of judge model output
# ---------------------------------------------------------------------------


def parse_judge_json(content: str) -> JudgeVerdict:
    """Parse a judge model's JSON payload from raw completion text.

    Tolerates leading/trailing prose and fenced ```json blocks.  Returns a
    zero verdict on unparseable output so the caller can decide to retry or
    flag for human review.
    """

    if not content or not content.strip():
        return JudgeVerdict(0.0, "empty_judge_output")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    raw = fenced.group(1) if fenced else content.strip()
    brace_start = raw.find("{")
    brace_end = raw.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        raw = raw[brace_start : brace_end + 1]
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return JudgeVerdict(0.0, f"json_parse_error: {exc}")
    dim_raw = payload.get("dim_scores", {})
    if not isinstance(dim_raw, dict):
        dim_raw = {}
    dim_scores: dict[str, float] = {}
    for dim in DIMENSIONS:
        try:
            dim_scores[dim] = float(dim_raw.get(dim, 0.0))
        except (TypeError, ValueError):
            dim_scores[dim] = 0.0
    try:
        quality = float(payload.get("quality_score", 0.0))
    except (TypeError, ValueError):
        quality = 0.0
    quality = max(0.0, min(1.0, quality))
    return JudgeVerdict(
        quality_score=quality,
        reject_reason=str(payload.get("reject_reason", "") or ""),
        dim_scores=dim_scores,
        reasoning=str(payload.get("reasoning", "") or ""),
    )


# ---------------------------------------------------------------------------
# Multi-turn recency-decay aggregation
# ---------------------------------------------------------------------------


def recency_weighted_mean(scores: Sequence[float], *, decay: float = RECENCY_DECAY) -> float:
    """Weighted mean with turn_0 highest weight (``decay^k`` reversed).

    ``scores[0]`` is the first turn, so weight is ``decay^(n-1-i)`` — the
    most recent turn is conventionally turn_0 of the final exchange and
    carries the highest weight per blueprint B.3.3.
    """

    if not scores:
        return 0.0
    n = len(scores)
    weights = [decay ** (n - 1 - i) for i in range(n)]
    total_w = sum(weights)
    return sum(s * w for s, w in zip(scores, weights, strict=False)) / total_w if total_w else 0.0


def aggregate_turn_verdicts(verdicts: Sequence[JudgeVerdict], *, decay: float = RECENCY_DECAY) -> JudgeVerdict:
    """Combine per-turn JudgeVerdicts into one recency-decayed verdict."""

    if not verdicts:
        return JudgeVerdict(0.0, "no_turns")
    quality = recency_weighted_mean([v.quality_score for v in verdicts], decay=decay)
    dim_scores: dict[str, float] = {}
    for dim in DIMENSIONS:
        dim_scores[dim] = recency_weighted_mean([v.dim_scores.get(dim, 0.0) for v in verdicts], decay=decay)
    reject_reasons = [v.reject_reason for v in verdicts if v.reject_reason]
    return JudgeVerdict(
        quality_score=quality,
        reject_reason="; ".join(reject_reasons[:3]),
        dim_scores=dim_scores,
        reasoning=f"aggregated across {len(verdicts)} turns (decay={decay})",
    )


# ---------------------------------------------------------------------------
# Self-consistency (k runs, same sample) + variance check
# ---------------------------------------------------------------------------


def runs_self_consistent(verdicts: Sequence[JudgeVerdict], *, variance_max: float = SELF_CONSISTENCY_VARIANCE_MAX) -> bool:
    """Return True if k-run variance stays below the human-review threshold."""

    if len(verdicts) < 2:
        return True
    scores = [v.quality_score for v in verdicts]
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    return variance <= variance_max


# ---------------------------------------------------------------------------
# Dual-model consistency
# ---------------------------------------------------------------------------


@dataclass
class DualJudgeResult:
    primary: JudgeVerdict
    secondary: JudgeVerdict
    accepted: bool
    needs_human_review: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary.to_dict(),
            "secondary": self.secondary.to_dict(),
            "accepted": self.accepted,
            "needs_human_review": self.needs_human_review,
            "reason": self.reason,
        }


def reconcile_dual(primary: JudgeVerdict, secondary: JudgeVerdict, *, diff_max: float = DUAL_CONSISTENCY_DIFF_MAX) -> DualJudgeResult:
    """Apply blueprint B.3.2 dual-judge reconciliation.

    ``|primary.quality - secondary.quality| <= 0.15`` → accept primary.
    Otherwise → flag for human review (still emit primary as reference).
    """

    diff = round(abs(primary.quality_score - secondary.quality_score), 3)
    if diff <= diff_max:
        return DualJudgeResult(
            primary=primary,
            secondary=secondary,
            accepted=primary.accepted,
            needs_human_review=False,
            reason=f"dual_consistent diff={diff:.3f}",
        )
    return DualJudgeResult(
        primary=primary,
        secondary=secondary,
        accepted=False,
        needs_human_review=True,
        reason=f"dual_inconsistent diff={diff:.3f} exceeds {diff_max}",
    )


# ---------------------------------------------------------------------------
# Calibration metrics — Pearson r vs golden, Cohen's kappa on accept/reject
# ---------------------------------------------------------------------------


def pearson_r(a: Sequence[float], b: Sequence[float]) -> float:
    """Pearson correlation between two equal-length score vectors."""

    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    ax = list(a[:n])
    bx = list(b[:n])
    mean_a = sum(ax) / n
    mean_b = sum(bx) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ax, bx, strict=False))
    var_a = sum((x - mean_a) ** 2 for x in ax)
    var_b = sum((y - mean_b) ** 2 for y in bx)
    denom = math.sqrt(var_a * var_b)
    return cov / denom if denom else 0.0


def _cohen_kappa_counts(pairs: Sequence[tuple[bool, bool]]) -> float:
    n = len(pairs)
    if n == 0:
        return 0.0
    agree = sum(1 for g, j in pairs if g == j)
    p_o = agree / n
    p_g_pos = sum(1 for g, _ in pairs if g) / n
    p_j_pos = sum(1 for _, j in pairs if j) / n
    p_e = p_g_pos * p_j_pos + (1 - p_g_pos) * (1 - p_j_pos)
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def cohen_kappa(golden_accept: Sequence[bool], judge_accept: Sequence[bool]) -> float:
    """Cohen's kappa on accept/reject decisions vs golden labels."""

    pairs = list(zip(golden_accept, judge_accept, strict=False))
    return _cohen_kappa_counts(pairs)


@dataclass
class CalibrationReport:
    pearson_r: float
    cohen_kappa: float
    sample_count: int
    passes: bool
    reason: str = ""


def evaluate_calibration(
    judge_scores: Sequence[float],
    golden_scores: Sequence[float],
    *,
    pearson_min: float = CALIB_PEARSON_MIN,
    kappa_min: float = CALIB_KAPPA_MIN,
    accept_threshold: float = ACCEPT_THRESHOLD,
) -> CalibrationReport:
    """Compare judge scores/decisions against a golden calibration set."""

    n = min(len(judge_scores), len(golden_scores))
    if n == 0:
        return CalibrationReport(0.0, 0.0, 0, False, "empty_calibration_set")
    r = pearson_r(judge_scores, golden_scores)
    golden_accept = [s >= accept_threshold for s in golden_scores[:n]]
    judge_accept = [s >= accept_threshold for s in judge_scores[:n]]
    kappa = cohen_kappa(golden_accept, judge_accept)
    if r >= pearson_min and kappa >= kappa_min:
        return CalibrationReport(r, kappa, n, True, "release_ready")
    return CalibrationReport(
        r,
        kappa,
        n,
        False,
        f"below_gate pearson={r:.3f}<{pearson_min} or kappa={kappa:.3f}<{kappa_min}",
    )


# ---------------------------------------------------------------------------
# Judge HTTP transport — OpenAI-compatible Chat Completions endpoint
# ---------------------------------------------------------------------------


async def _call_judge_model(
    session: aiohttp.ClientSession,
    *,
    url: str,
    model: str,
    candidate_content: str,
    reference_content: str,
    temperature: float = JUDGE_TEMPERATURE,
    timeout: int = JUDGE_TIMEOUT_SECONDS,
) -> JudgeVerdict:
    """Call one judge model on one candidate/reference pair."""

    user_prompt = (
        f"Reference turn:\n{reference_content[:8000]}\n\n"
        f"Candidate answer:\n{candidate_content[:8000]}\n\n"
        "Score the candidate against the reference now."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "temperature": temperature,
    }
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status != 200:
                text = await resp.text()
                return JudgeVerdict(0.0, f"http_{resp.status}: {text[:200]}")
            data = await resp.json()
            content = data["choices"][0]["message"]["content"]
            return parse_judge_json(content)
    except (TimeoutError, aiohttp.ClientError, KeyError) as exc:
        return JudgeVerdict(0.0, f"transport_error: {exc}")


async def judge_single(
    candidate: str,
    reference: str,
    *,
    url: str = DEFAULT_JUDGE_URL,
    primary_model: str = PRIMARY_MODEL,
    secondary_model: str = SECONDARY_MODEL,
    session: aiohttp.ClientSession | None = None,
    self_consistency_runs: int = SELF_CONSISTENCY_RUNS,
) -> DualJudgeResult:
    """Judge one candidate/reference pair with dual-model + self-consistency.

    Per blueprint B.3.2 + B.3.4: run the primary model k=3 times for
    self-consistency; if variance > 0.05 the result is flagged for human
    review regardless of dual-model agreement.  Then run the secondary model
    once and reconcile.
    """

    owns_session = session is None
    session = session or aiohttp.ClientSession()
    try:
        primary_runs = await asyncio.gather(
            *(_call_judge_model(session, url=url, model=primary_model, candidate_content=candidate, reference_content=reference) for _ in range(self_consistency_runs))
        )
        secondary = await _call_judge_model(
            session,
            url=url,
            model=secondary_model,
            candidate_content=candidate,
            reference_content=reference,
        )
    finally:
        if owns_session:
            await session.close()

    consistent = runs_self_consistent(primary_runs)
    primary = aggregate_turn_verdicts(list(primary_runs)) if self_consistency_runs > 1 else primary_runs[0]
    result = reconcile_dual(primary, secondary)
    if not consistent:
        result.accepted = False
        result.needs_human_review = True
        result.reason = f"self_consistency_variance_exceeded; {result.reason}"
    return result


async def judge_record_turns(
    record: Mapping[str, Any],
    *,
    url: str = DEFAULT_JUDGE_URL,
    primary_model: str = PRIMARY_MODEL,
    secondary_model: str = SECONDARY_MODEL,
    session: aiohttp.ClientSession | None = None,
) -> DualJudgeResult:
    """Judge a multi-turn record: score each turn, recency-decay aggregate.

    Expects ``record["messages"]`` as a list of ``{role, content}`` dicts.
    Alternating user/assistant turns are paired; each assistant turn is
    judged against its preceding user turn.
    """

    messages = list(record.get("messages", []))
    if len(messages) < 2:
        return DualJudgeResult(
            primary=JudgeVerdict(0.0, "no_turns_to_judge"),
            secondary=JudgeVerdict(0.0, "no_turns_to_judge"),
            accepted=False,
            needs_human_review=True,
            reason="record_has_no_turns",
        )
    owns_session = session is None
    session = session or aiohttp.ClientSession()
    try:
        pairs: list[tuple[str, str]] = []
        for i in range(0, len(messages) - 1, 2):
            user_msg = messages[i]
            assistant_msg = messages[i + 1]
            if user_msg.get("role") == "user" and assistant_msg.get("role") == "assistant":
                pairs.append((str(user_msg.get("content", "")), str(assistant_msg.get("content", ""))))
        if not pairs:
            return DualJudgeResult(
                primary=JudgeVerdict(0.0, "no_user_assistant_pairs"),
                secondary=JudgeVerdict(0.0, "no_user_assistant_pairs"),
                accepted=False,
                needs_human_review=True,
                reason="no_user_assistant_pairs",
            )
        primary_turn_tasks = [
            _call_judge_model(session, url=url, model=primary_model, candidate_content=cand, reference_content=ref)
            for ref, cand in pairs
        ]
        primary_turns = await asyncio.gather(*primary_turn_tasks)
        # Secondary judges the same concatenated exchange once for dual-consistency
        # against the aggregated primary (cheaper than per-turn secondary).
        joined_ref = "\n".join(ref for ref, _ in pairs)
        joined_cand = "\n".join(cand for _, cand in pairs)
        secondary = await _call_judge_model(
            session,
            url=url,
            model=secondary_model,
            candidate_content=joined_cand,
            reference_content=joined_ref,
        )
    finally:
        if owns_session:
            await session.close()

    primary_aggregated = aggregate_turn_verdicts(primary_turns)
    result = reconcile_dual(primary_aggregated, secondary)
    return result
