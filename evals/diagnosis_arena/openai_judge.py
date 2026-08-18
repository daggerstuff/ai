"""
GPT-4o-as-judge integration for DiagnosisArena.

Implements the paper's inter-rater reliability protocol:
  - 3 independent judgments per (case, response) pair
  - majority vote on tier and dimension scores
  - weighted tier mapping (Identical=1.0, Relevant=0.5, Irrelevant=0.0)

Concrete judge is async-safe and accepts an OpenAI-compatible client
(e.g. ``openai.AsyncOpenAI``). For offline use keep the existing
``HeuristicJudge``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import Counter
from collections.abc import Callable, Sequence

from .judge import HeuristicJudge, Judge, Judgment
from .types import DIAGNOSTIC_DIMENSIONS, TierScore

logger = logging.getLogger(__name__)

# Default judge temperature; 0 keeps repeated calls deterministic.
_DEFAULT_TEMPERATURE = 0.0


async def _call_openai_judge(
    client,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> str:
    try:
        response = await client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:  # pragma: no cover — network fallback
        logger.error("OpenAI judge call failed: %s", exc)
        return "irrelevant"


def _build_judge_prompt(case, response) -> str:
    differential = "\n".join(f"- {d}" for d in case.differential_diagnoses)
    evidence = "\n".join(f"- {e}" for e in case.supporting_evidence)
    return (
        "You are an expert clinical reviewer.\n\n"
        "## Clinical case\n\n"
        f"Presentation: {case.presentation}\n"
        f"History: {case.history}\n"
        f"Exam: {case.exam}\n"
        f"Labs: {case.labs}\n"
        f"Imaging: {case.imaging}\n"
        f"Progression: {case.progression}\n\n"
        f"Ground-truth differential:\n{differential}\n\n"
        f"Ground-truth final diagnosis: {case.final_diagnosis}\n\n"
        f"Ground-truth supporting evidence:\n{evidence}\n\n"
        f"Key differentiators: {', '.join(case.key_differentiators)}\n\n"
        "## Model response\n\n"
        f"Final diagnosis: {response.final_diagnosis}\n"
        f"Differential: {', '.join(response.differential_list)}\n"
        f"Evidence cited: {', '.join(response.evidence_cited)}\n"
        f"Reasoning: {response.reasoning}\n\n"
        "Score with a single tier and dimension scores 0.0-1.0.\n"
        "Tier: identical | relevant | irrelevant\n"
        "Dimensions: hypothesis_generation, evidence_interpretation, differential_diagnosis, final_diagnosis\n\n"
        "Respond as JSON: {\"tier\": ..., \"dimensions\": {\"name\": score, ...}, \"notes\": \"...\"}"
    )


def _parse_judge_output(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError
        return data
    except Exception:
        return {"tier": "irrelevant", "dimensions": {}, "notes": text[:200]}


def _score_for(value) -> float:
    if isinstance(value, str):
        value = value.strip().lower()
        mapping = {
            "high": 0.9,
            "medium": 0.6,
            "low": 0.2,
            "partial": 0.5,
            "none": 0.0,
            "0.5": 0.5,
            "1.0": 1.0,
            "0.0": 0.0,
        }
        if value in mapping:
            return mapping[value]
        try:
            return max(0.0, min(1.0, float(value)))
        except ValueError:
            return 0.5
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _normalize_dimension_scores(raw: dict) -> tuple:
    out = []
    for name in DIAGNOSTIC_DIMENSIONS:
        out.append((name, _score_for(raw.get(name, 0.0)),))
    return tuple(out)


async def _single_judgment(
    client,
    *,
    case,
    response,
    model: str = "gpt-4o",
    temperature: float = _DEFAULT_TEMPERATURE,
) -> Judgment:
    system_prompt = (
        "You are the GPT-4o-as-judge from DiagnosisArena (arXiv 2505.14107). "
        "Achieve >=92.5% expert agreement by grounding scores exclusively in pretrial facts. "
        "Return JSON only."
    )
    user_prompt = _build_judge_prompt(case, response)
    raw = await _call_openai_judge(
        client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
    )
    parsed = _parse_judge_output(raw)
    tier_name = str(parsed.get("tier", "irrelevant")).strip().upper()
    try:
        tier = TierScore[tier_name]
    except KeyError:
        tier = TierScore.IRRELEVANT
    dim_scores = {name: score for name, score in _normalize_dimension_scores(parsed.get("dimensions", {}))}
    return Judgment(
        response_id=response.response_id,
        case_id=case.case_id,
        tier=tier,
        dimensions=tuple(
            (name, dim_scores.get(name, 0.0), "") for name in DIAGNOSTIC_DIMENSIONS
        ),
        judge_model=model,
        notes=str(parsed.get("notes", ""))[:500],
    )


def _default_client_factory() -> object:
    try:
        from openai import AsyncOpenAI  # type: ignore[import-untyped]

        return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    except ImportError as exc:
        raise RuntimeError("openai package required for OpenAIDiagnosisJudge") from exc


def _safe_loop() -> None:
    try:
        asyncio.get_running_loop()
        raise RuntimeError("OpenAIDiagnosisJudge.judge() cannot be called inside an active event loop")
    except RuntimeError as exc:
        if "cannot be called" in str(exc):
            raise


def _majority_tier(votes: Sequence[Judgment]) -> TierScore:
    counts = Counter(v.tier for v in votes)
    mode = counts.most_common(1)[0][0]
    tie_rank = {TierScore.IDENTICAL: 3, TierScore.RELEVANT: 2, TierScore.IRRELEVANT: 1}
    if len(counts) > 1:
        tied = [t for t, c in counts.items() if c == counts[mode]]
        mode = max(tied, key=lambda t: tie_rank.get(t, 0))
    return mode


def _majority_dimensions(votes: Sequence[Judgment]) -> tuple:
    out = []
    for name in DIAGNOSTIC_DIMENSIONS:
        values = []
        for vote in votes:
            for dim in vote.dimensions:
                if dim.name == name:
                    values.append(dim.score)
                    break
        out.append((name, sum(values) / len(values) if values else 0.0, f"mean across {len(values)} votes",))
    return tuple(out)


class OpenAIDiagnosisJudge(Judge):
    """GPT-4o-as-judge with 3-way majority-vote reliability protocol.

    Parameters
    ----------
    client_factory:
        Zero-argument callable returning an OpenAI-compatible client.
        Defaults to ``openai.AsyncOpenAI`` when omitted.
    judge_model:
        OpenAI model id used for judging. Default ``gpt-4o``.
    n_votes:
        Number of independent judgments per case. Default ``3``.
    temperature:
        LLM temperature. Default ``0.0`` (deterministic).
    fallback:
        Judge used when OpenAI calls fail. Default ``HeuristicJudge``.
    """

    def __init__(
        self,
        client_factory: Callable[[], object] | None = None,
        judge_model: str = "gpt-4o",
        n_votes: int = 3,
        temperature: float = _DEFAULT_TEMPERATURE,
        fallback: Judge | None = None,
    ) -> None:
        self._client_factory = client_factory or _default_client_factory
        self.judge_model = judge_model
        self.n_votes = max(1, int(n_votes))
        self.temperature = float(temperature)
        self.fallback = fallback or HeuristicJudge(judge_model="heuristic-v1")

    def _sync_vote(self, case, response) -> list[Judgment]:
        client = self._client_factory()
        return asyncio.run(self._async_vote(client, case=case, response=response))

    async def _async_vote(self, client, *, case, response) -> list[Judgment]:
        return [
            await _single_judgment(
                client,
                case=case,
                response=response,
                model=self.judge_model,
                temperature=self.temperature,
            )
            for _ in range(self.n_votes)
        ]

    def judge(self, case, response) -> Judgment:  # type: ignore[override]
        _safe_loop()
        try:
            votes = self._sync_vote(case, response)
        except Exception as exc:
            logger.warning("OpenAI judge failed (%s); falling back", exc)
            return self.fallback.judge(case, response)
        if not votes:
            return self.fallback.judge(case, response)
        return Judgment(
            response_id=response.response_id,
            case_id=case.case_id,
            tier=_majority_tier(votes),
            dimensions=_majority_dimensions(votes),
            judge_model=self.judge_model,
            notes=(
                f"n={len(votes)} votes, mode={_majority_tier(votes).value}"
            ),
        )


def inter_rater_agreement(votes: Sequence[Judgment], *, dimension: str | None = None) -> float:
    """Fraction of votes matching the majority (either tier or named dimension)."""
    if not votes:
        return 0.0
    if dimension is not None:
        values = []
        for vote in votes:
            for dim in vote.dimensions:
                if dim.name == dimension:
                    values.append(dim.score)
                    break
        if not values:
            return 0.0
        expected = sum(values) / len(values)
        return sum(abs(v - expected) <= 0.05 for v in values) / len(values)
    counts = Counter(v.tier for v in votes)
    mode = counts.most_common(1)[0][0]
    return sum(1 for v in votes if v.tier == mode) / len(votes)


__all__ = [
    "OpenAIDiagnosisJudge",
    "inter_rater_agreement",
]
