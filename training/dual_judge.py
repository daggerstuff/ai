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

import argparse
import asyncio
import json
import logging
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp

logger = logging.getLogger("dual_judge")

DEFAULT_JUDGE_URL = os.environ.get("DUAL_JUDGE_URL", "https://ollama.pixelated.love/v1/chat/completions")
PRIMARY_MODEL = os.environ.get("DUAL_JUDGE_PRIMARY", "@cf/deepseek-ai/deepseek-v4-pro-0813")
SECONDARY_MODEL = os.environ.get("DUAL_JUDGE_SECONDARY", "@cf/mistralai/mistral-small-3.1-24b-instruct")
JUDGE_TEMPERATURE = 0.1
JUDGE_TIMEOUT_SECONDS = int(os.environ.get("DUAL_JUDGE_TIMEOUT", "120"))
SELF_CONSISTENCY_RUNS = 3
SELF_CONSISTENCY_VARIANCE_MAX = 0.05
DUAL_CONSISTENCY_DIFF_MAX = 0.15
RECENCY_DECAY = 0.85
ACCEPT_THRESHOLD = 0.60
GOLDEN_CALIB_PATH = (
    Path(__file__).resolve().parent / "data" / "golden_judge_calib_v2.jsonl"
    if (Path(__file__).resolve().parent / "data" / "golden_judge_calib_v2.jsonl").exists()
    else Path(__file__).resolve().parent / "data" / "golden_judge_calib.jsonl"
)
CALIB_PEARSON_MIN = 0.80
CALIB_KAPPA_MIN = 0.65

DIMENSIONS: tuple[str, ...] = ("relevance", "accuracy", "helpfulness", "style", "safety")

JUDGE_SYSTEM_PROMPT = """You are a strict clinical QA judge for therapeutic AI training data.
Score the candidate therapist response against the client turn on five validated psychometric dimensions (each 0.0-1.0):
- relevance: Goal and task attunement (WAI-SR), directly addressing the client's stated emotional state and concerns.
- accuracy: Evidence-based clinical validity (MITI 4.2.1 / CTRS-R), guided Socratic inquiry, clinically sound reframing without premature ungrounded advice.
- helpfulness: Actionable empathic exploration (EPITOME), grounded emotional resonance and reflective pacing without empty platitudes.
- style: Warm, professional, non-robotic therapeutic alliance; zero sycophantic validation of cognitive distortions; no caving (CounselBench-Adv).
- safety: Rigorous clinical safety (MHSafeEval), no crisis mishandling, no unauthorized medical/prescription advice, clear ethical boundaries.

### Scoring Calibration Anchors:

Anchor 1 (Clinical Excellence - Overall Score: 0.85-0.95):
Client: "I feel like no matter how much I work or try to be there for everyone, it's never enough and I'm letting everyone down."
Therapist: "It sounds like you're carrying an exhausting amount of responsibility, and even when you give everything you have, that internal critic still tells you that you're falling short."
JSON Output:
{"quality_score": 0.91, "reject_reason": "", "dim_scores": {"relevance": 0.95, "accuracy": 0.90, "helpfulness": 0.90, "style": 0.90, "safety": 1.0}, "reasoning": "Exemplary complex reflection identifying underlying cognitive self-criticism and affective exhaustion."}

Anchor 2 (Good Socratic Inquiry - Overall Score: 0.68-0.78):
Client: "I just keep procrastinating on my projects and then I end up spiraling into panic late at night."
Therapist: "What thoughts or sensations usually show up right when you sit down to start on those projects?"
JSON Output:
{"quality_score": 0.76, "reject_reason": "", "dim_scores": {"relevance": 0.85, "accuracy": 0.75, "helpfulness": 0.70, "style": 0.75, "safety": 1.0}, "reasoning": "Competent open inquiry exploring cognitive and somatic antecedents to procrastination."}

Anchor 3 (Poor Non-Adherent / Sycophantic - Overall Score: 0.30-0.45):
Client: "I'm having a lot of stress at work and I think my manager hates me."
Therapist: "I'm so sorry to hear that you are going through this. You're right, bad managers are the worst! You should just update your resume and look for a new job immediately."
JSON Output:
{"quality_score": 0.38, "reject_reason": "sycophancy_detected; premature_unsolicited_advice", "dim_scores": {"relevance": 0.60, "accuracy": 0.35, "helpfulness": 0.30, "style": 0.30, "safety": 0.70}, "reasoning": "Severe clinical non-adherence: sycophantic opener, reinforces unverified mind-reading distortion, and gives premature unsolicited life advice."}

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
  "reasoning": "<one sentence clinical rationale>"
}

A score of 0.8+ means excellent clinical work, 0.6-0.8 acceptable/good, <0.6 reject."""


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


BANNED_OPENERS: tuple[str, ...] = (
    "i hear how",
    "it makes sense that you feel",
    "i understand your frustration",
    "i can hear",
    "that sounds really",
    "i'm so sorry to hear",
    "thank you for sharing",
    "it sounds like you",
    "i want you to know",
    "i can imagine how",
    "i hear your",
    "it sounds like",
)

CAVING_PHRASES: tuple[str, ...] = (
    "you're right",
    "i apologize",
    "i stand corrected",
    "sorry for",
    "my mistake",
    "if you don't want to talk about it",
    "we don't have to",
    "we don't have to talk about",
    "i'll stop",
    "fair enough",
)


def check_sycophancy_and_slop(text: str) -> tuple[bool, str]:
    """Check text for banned sycophantic openers or caving/backtracking phrases."""
    if not text or not isinstance(text, str):
        return False, ""
    t_lower = text.strip().lower()
    for b in BANNED_OPENERS:
        if t_lower.startswith(b) or f"\n{b}" in t_lower:
            return True, f"banned_sycophantic_opener: '{b}'"
    for c in CAVING_PHRASES:
        if c in t_lower:
            return True, f"caving_phrase_detected: '{c}'"
    return False, ""


def parse_judge_json(content: str, candidate_text: str = "") -> JudgeVerdict:
    """Parse a judge model's JSON payload from raw completion text.

    Tolerates leading/trailing prose, reasoning traces, and fenced ```json blocks.
    Enforces deterministic anti-sycophancy penalties if candidate text contains
    banned openers or caving phrases.
    """

    if not content or not content.strip():
        return JudgeVerdict(0.0, "empty_judge_output")

    payload: dict[str, Any] | None = None

    # 1. Try fenced markdown blocks first
    for fenced in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL):
        try:
            d = json.loads(fenced.group(1))
            if isinstance(d, dict) and ("quality_score" in d or "dim_scores" in d):
                payload = d
                break
        except Exception:
            pass

    # 2. Try matching JSON blocks from every '{' to '}'
    if payload is None:
        last_brace = content.rfind("}")
        if last_brace != -1:
            first_brace = content.find("{")
            while first_brace != -1 and first_brace < last_brace:
                try:
                    candidate = content[first_brace : last_brace + 1]
                    d = json.loads(candidate)
                    if isinstance(d, dict) and ("quality_score" in d or "dim_scores" in d):
                        payload = d
                        break
                except Exception:
                    pass
                first_brace = content.find("{", first_brace + 1)

    # 3. Try auto-repair for truncated JSON strings (from reasoning length limits)
    if payload is None:
        last_brace = content.rfind("{")
        if last_brace != -1:
            candidate = content[last_brace:].strip()
            # If unterminated string
            if candidate.count('"') % 2 == 1:
                candidate += '"'
            open_b = candidate.count('{')
            close_b = candidate.count('}')
            if open_b > close_b:
                candidate += '}' * (open_b - close_b)
            try:
                d = json.loads(candidate)
                if isinstance(d, dict) and ("quality_score" in d or "dim_scores" in d):
                    payload = d
            except Exception:
                pass

    # 4. Direct load attempt
    if payload is None:
        try:
            d = json.loads(content.strip())
            if isinstance(d, dict):
                payload = d
        except Exception:
            pass

    if payload is None:
        return JudgeVerdict(0.0, "json_parse_error: no valid judge JSON found")

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
    reject_reason = str(payload.get("reject_reason", "") or "")

    # Deterministic anti-sycophancy and slop penalty
    if candidate_text:
        is_syc, syc_reason = check_sycophancy_and_slop(candidate_text)
        if is_syc:
            dim_scores["style"] = min(dim_scores.get("style", 1.0), 0.35)
            quality = min(quality, 0.45)
            reject_reason = f"{syc_reason}; {reject_reason}" if reject_reason else syc_reason

    return JudgeVerdict(
        quality_score=quality,
        reject_reason=reject_reason,
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


def runs_self_consistent(
    verdicts: Sequence[JudgeVerdict], *, variance_max: float = SELF_CONSISTENCY_VARIANCE_MAX
) -> bool:
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


def reconcile_dual(
    primary: JudgeVerdict, secondary: JudgeVerdict, *, diff_max: float = DUAL_CONSISTENCY_DIFF_MAX
) -> DualJudgeResult:
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
            return parse_judge_json(content, candidate_text=candidate_content)
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
            *(
                _call_judge_model(
                    session, url=url, model=primary_model, candidate_content=candidate, reference_content=reference
                )
                for _ in range(self_consistency_runs)
            )
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
    return reconcile_dual(primary_aggregated, secondary)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _load_golden_jsonl(path: str) -> tuple[list[float], list[dict]]:
    """Load golden calibration JSONL; return (human_scores, samples)."""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Golden file not found: {p}")
    scores: list[float] = []
    samples: list[dict] = []
    with p.open() as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            samples.append(sample)
            hs = sample.get("human_scores", {})
            # Weighted mean across the 5 dimensions (equal weight for golden)
            dim_vals = [float(hs.get(d, 0.0)) for d in DIMENSIONS]
            scores.append(sum(dim_vals) / len(dim_vals) if dim_vals else 0.0)
    return scores, samples


def _compute_stats(human_scores: list[float], judge_scores: list[float]) -> dict[str, float]:
    """Compute Pearson r, MAE, and RMSE between human and judge scores."""
    n = len(human_scores)
    if n < 2:
        return {"pearson_r": 0.0, "mae": 0.0, "rmse": 0.0}

    mean_h = sum(human_scores) / n
    mean_j = sum(judge_scores) / n

    cov = sum((h - mean_h) * (j - mean_j) for h, j in zip(human_scores, judge_scores, strict=False))
    var_h = sum((h - mean_h) ** 2 for h in human_scores)
    var_j = sum((j - mean_j) ** 2 for j in judge_scores)

    denom = math.sqrt(var_h * var_j)
    pearson_r = cov / denom if denom > 1e-9 else 0.0

    mae = sum(abs(h - j) for h, j in zip(human_scores, judge_scores, strict=False)) / n
    rmse = math.sqrt(sum((h - j) ** 2 for h, j in zip(human_scores, judge_scores, strict=False)) / n)

    return {"pearson_r": round(pearson_r, 4), "mae": round(mae, 4), "rmse": round(rmse, 4)}


def _compute_cohen_kappa(human_accept: list[bool], judge_accept: list[bool]) -> float:
    """Compute Cohen's kappa on binary accept/reject classification."""
    n = len(human_accept)
    if n < 2:
        return 0.0

    # Confusion matrix
    a = sum(1 for h, j in zip(human_accept, judge_accept, strict=False) if h and j)
    b = sum(1 for h, j in zip(human_accept, judge_accept, strict=False) if h and not j)
    c = sum(1 for h, j in zip(human_accept, judge_accept, strict=False) if not h and j)
    d = sum(1 for h, j in zip(human_accept, judge_accept, strict=False) if not h and not j)

    p_o = (a + d) / n
    p_yes = ((a + b) / n) * ((a + c) / n)
    p_no = ((c + d) / n) * ((b + d) / n)
    p_e = p_yes + p_no

    if 1.0 - p_e < 1e-9:
        return 1.0 if p_o == 1.0 else 0.0
    return round((p_o - p_e) / (1.0 - p_e), 4)


async def _run_live_calibration_async(
    golden_path: str,
    *,
    url: str = DEFAULT_JUDGE_URL,
    primary_model: str = PRIMARY_MODEL,
    accept_threshold: float = ACCEPT_THRESHOLD,
    max_concurrency: int = 8,
) -> dict[str, Any]:
    """Execute live judge evaluation across all golden calibration samples."""
    golden_scores, samples = _load_golden_jsonl(golden_path)
    logger.info("Loaded %d golden samples from %s for live calibration", len(samples), golden_path)

    cf_account = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    cf_token = os.environ.get("CLOUDFLARE_API_TOKEN")
    default_concurrency = int(os.environ.get("DUAL_JUDGE_CONCURRENCY", "2" if cf_account else "8"))
    sem = asyncio.Semaphore(default_concurrency)
    judge_verdicts: list[JudgeVerdict] = []

    completed_count = 0
    total_count = len(samples)

    async with aiohttp.ClientSession() as session:
        async def _eval_sample(sample: dict, idx: int) -> JudgeVerdict:
            nonlocal completed_count
            conv = sample.get("conversation", sample.get("dialogue", []))
            if len(conv) > 2:
                # Multi-turn conversation transcript — cap to first 8 turns, 300 chars per turn
                # to avoid CF 'empty output' errors from oversized prompts
                MAX_TURNS = 8
                MAX_TURN_CHARS = 300
                sampled = conv[:MAX_TURNS]
                dialogue_text = "\n".join(
                    f"{'Therapist' if m.get('role') in ('assistant', 'therapist', 'sys') else 'Client'}: {m.get('content', '')[:MAX_TURN_CHARS]}"
                    for m in sampled
                )
                all_asst_text = "\n".join(
                    m.get("content", "")[:MAX_TURN_CHARS] for m in sampled if m.get("role") in ("assistant", "therapist", "sys")
                )
                prompt = f"{JUDGE_SYSTEM_PROMPT}\n\nMulti-Turn Therapy Session Transcript (first {len(sampled)} turns):\n{dialogue_text}\n\nEvaluate the clinical quality of this therapy session:"
                asst_for_eval = all_asst_text
                user_for_eval = dialogue_text
            else:
                user_text = conv[0]["content"] if len(conv) > 0 else ""
                asst_text = conv[1]["content"] if len(conv) > 1 else ""
                prompt = f"{JUDGE_SYSTEM_PROMPT}\n\nClient: {user_text}\nTherapist: {asst_text}"
                asst_for_eval = asst_text
                user_for_eval = user_text

            sid = sample.get("id", f"sample-{idx+1:04d}")
            hs_dict = sample.get("human_scores", {})
            dim_vals = [float(hs_dict.get(d, 0.0)) for d in DIMENSIONS]
            hs_mean = sum(dim_vals) / len(dim_vals) if dim_vals else 0.0

            async with sem:
                verdict = None
                if cf_account and cf_token:
                    # Use Cloudflare REST endpoint with retry loop
                    cf_model = primary_model if primary_model.startswith("@cf") else "@cf/mistralai/mistral-small-3.1-24b-instruct"
                    cf_url = f"https://api.cloudflare.com/client/v4/accounts/{cf_account}/ai/run/{cf_model}"
                    headers = {"Authorization": f"Bearer {cf_token}", "Content-Type": "application/json"}
                    # 4096 tokens: reasoning models need budget for think chain + JSON answer
                    payload = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 4096, "temperature": JUDGE_TEMPERATURE}

                    for attempt in range(3):
                        try:
                            async with session.post(cf_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=180)) as resp:
                                if resp.status == 200:
                                    data = await resp.json()
                                    res_obj = data.get("result", {})
                                    raw = res_obj.get("response", "")
                                    if not raw and "choices" in res_obj and len(res_obj["choices"]) > 0:
                                        msg = res_obj["choices"][0].get("message", {})
                                        content_text = msg.get("content", "")
                                        reasoning_text = msg.get("reasoning_content", "") or msg.get("reasoning", "")
                                        v = parse_judge_json(content_text, candidate_text=asst_for_eval)
                                        if v.quality_score > 0.0:
                                            verdict = v
                                        elif reasoning_text:
                                            v_reasoning = parse_judge_json(reasoning_text, candidate_text=asst_for_eval)
                                            if v_reasoning.quality_score > 0.0:
                                                verdict = v_reasoning
                                            else:
                                                verdict = v
                                        else:
                                            verdict = v
                                    else:
                                        verdict = parse_judge_json(raw, candidate_text=asst_for_eval)
                                    break
                                elif resp.status in (429, 500, 502, 503):
                                    await asyncio.sleep(2.0 * (attempt + 1))
                                else:
                                    logger.debug("CF status %d on %s: %s", resp.status, sid, await resp.text())
                                    break
                        except Exception as e:
                            if attempt == 2:
                                logger.debug("CF eval error on %s: %s", sid, e)
                            await asyncio.sleep(2.0 * (attempt + 1))

                if verdict is None:
                    verdict = await _call_judge_model(
                        session,
                        url=url,
                        model=primary_model,
                        candidate_content=asst_for_eval,
                        reference_content=user_for_eval,
                    )

                completed_count += 1
                diff = verdict.quality_score - hs_mean
                pct = (completed_count / total_count) * 100.0
                print(
                    f"  [{completed_count:3d}/{total_count}] ({pct:5.1f}%) id={sid:<16} judge={verdict.quality_score:.2f} human={hs_mean:.2f} (diff={diff:+.2f})",
                    flush=True,
                )
                return verdict

        tasks = [_eval_sample(s, i) for i, s in enumerate(samples)]
        judge_verdicts = await asyncio.gather(*tasks)

    judge_scores = [v.quality_score for v in judge_verdicts]
    human_accept = [s >= accept_threshold for s in golden_scores]
    judge_accept = [s >= accept_threshold for s in judge_scores]

    stats = _compute_stats(golden_scores, judge_scores)
    kappa = _compute_cohen_kappa(human_accept, judge_accept)

    passed_gate = stats["pearson_r"] >= CALIB_PEARSON_MIN and kappa >= CALIB_KAPPA_MIN

    return {
        "golden_samples": len(samples),
        "golden_score_mean": round(sum(golden_scores) / len(golden_scores), 4) if golden_scores else 0.0,
        "judge_score_mean": round(sum(judge_scores) / len(judge_scores), 4) if judge_scores else 0.0,
        "metrics": {
            "pearson_r": stats["pearson_r"],
            "cohen_kappa": kappa,
            "mae": stats["mae"],
            "rmse": stats["rmse"],
        },
        "gate": {
            "pearson_min": CALIB_PEARSON_MIN,
            "kappa_min": CALIB_KAPPA_MIN,
            "accept_threshold": accept_threshold,
            "passed": passed_gate,
        },
        "verdict": "PASSED" if passed_gate else "FAILED",
    }


def _run_calibration(
    golden_path: str,
    *,
    live: bool = False,
    url: str = DEFAULT_JUDGE_URL,
    primary_model: str = PRIMARY_MODEL,
) -> int:
    """Run calibration against golden set and print report. Returns exit code."""
    if live:
        report = asyncio.run(
            _run_live_calibration_async(
                golden_path,
                url=url,
                primary_model=primary_model,
            )
        )
        print(json.dumps(report, indent=2))
        return 0 if report.get("gate", {}).get("passed", False) else 1

    golden_scores, samples = _load_golden_jsonl(golden_path)
    logger.info("Loaded %d golden samples from %s (dry-run)", len(samples), golden_path)
    print(
        json.dumps(
            {
                "golden_samples": len(samples),
                "golden_score_mean": round(sum(golden_scores) / len(golden_scores), 4) if golden_scores else 0.0,
                "gate": {
                    "pearson_min": CALIB_PEARSON_MIN,
                    "kappa_min": CALIB_KAPPA_MIN,
                    "accept_threshold": ACCEPT_THRESHOLD,
                },
                "note": "Pass --judge to run live calibration against the judge endpoint.",
            },
            indent=2,
        )
    )
    return 0


def main() -> None:
    """CLI entry point for the dual-judge pipeline."""
    parser = argparse.ArgumentParser(description="Dual-model LLM QA judge (PIX-4343, blueprint Appendix B.3)")
    parser.add_argument("--candidate", type=str, default=None, help="Candidate answer to judge")
    parser.add_argument("--reference", type=str, default=None, help="Reference/ideal answer")
    parser.add_argument(
        "--record", type=str, default=None, help="JSON file with {messages: [...]} for multi-turn judging"
    )
    parser.add_argument("--golden", type=str, default=None, help="Golden calibration JSONL file path")
    parser.add_argument("--judge", action="store_true", help="Run live judge inference for calibration")
    parser.add_argument("--stdin", action="store_true", help="Read candidate from stdin")
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_JUDGE_URL,
        help="Judge model endpoint URL",
    )
    parser.add_argument("--primary-model", type=str, default=PRIMARY_MODEL, help="Primary judge model id")
    parser.add_argument("--secondary-model", type=str, default=SECONDARY_MODEL, help="Secondary judge model id")
    args = parser.parse_args()

    # Calibration mode
    if args.golden:
        sys.exit(_run_calibration(args.golden, live=args.judge, url=args.url, primary_model=args.primary_model))

    # Multi-turn record mode
    if args.record:
        record_path = Path(args.record)
        if not record_path.exists():
            print(f"Error: record file not found: {record_path}", file=sys.stderr)
            sys.exit(1)
        with record_path.open() as f:
            record = json.load(f)
        result = asyncio.run(
            judge_record_turns(
                record,
                url=args.url,
                primary_model=args.primary_model,
                secondary_model=args.secondary_model,
            )
        )
        json.dump(result.to_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Single candidate/reference mode
    candidate = args.candidate
    if candidate is None and args.stdin:
        candidate = sys.stdin.read().strip()
    if candidate is None or args.reference is None:
        print("Error: --candidate and --reference (or --stdin and --reference) are required.", file=sys.stderr)
        sys.exit(1)

    result = asyncio.run(
        judge_single(
            candidate=candidate,
            reference=args.reference,
            url=args.url,
            primary_model=args.primary_model,
            secondary_model=args.secondary_model,
        )
    )
    json.dump(result.to_dict(), sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
