"""Dual-model LLM quality judge for the AI training curation pipeline.

Generalizes ``ClinicalValidityJudge`` to use two LLMs (Qwen-72B + LLaMA-70B)
for cross-model consistency scoring. This is Stage 2 of the curation
pipeline (Appendix B.3, PIX-4343).

Design overview
---------------

* **Rubric**: 5 dimensions (relevance, accuracy, helpfulness, style, safety),
  each scored 0.0-1.0. Overall = configurable weighted mean.
* **4-bin calibration**: poor / fair / good / excellent.
* **Dual-model**: primary (Qwen-72B) + secondary (LLaMA-70B) via vLLM.
  ``|primary - secondary| <= 0.15`` → accept; otherwise flag for human review.
* **Multi-turn**: score each turn independently, overall = recency-decay
  weighted mean (``weight_i = decay^(n-i)``, configurable decay=0.85).
* **Self-consistency**: k=3 samples per (model, turn) at temperature=0.1.
  If variance > 0.05 → flag for human review (do not auto-accept).
* **Calibration**: golden 200-sample set; Pearson r >= 0.80, Cohen's kappa >= 0.65.
* **Async**: ``ajudge()`` batches k=3 self-consistency samples + both models
  concurrently across all turns via ``asyncio.gather``.

Uses the existing ``utils.common.llm_client.LLMClient`` for LLM calls.

Usage::

    from training.llm_quality_judge import DualModelQualityJudge

    judge = DualModelQualityJudge(primary_client=..., secondary_client=...)
    result = judge.judge(conversation)
    # result["overall_score"], result["bin"], result["flags"], ...

    # async
    result = await judge.ajudge(conversation)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from scipy import stats as scipy_stats
from sklearn.metrics import cohen_kappa_score

if TYPE_CHECKING:
    from ai.tools.utilities.common.llm_client import LLMClient

logger = logging.getLogger("llm_quality_judge")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default models (vLLM-served, OpenAI-compatible endpoint).
# Override via LLM_PRIMARY_MODEL / LLM_SECONDARY_MODEL env vars (e.g. for Neon AI Gateway).
PRIMARY_MODEL = os.environ.get("LLM_PRIMARY_MODEL", "Qwen/Qwen2.5-72B-Instruct")
SECONDARY_MODEL = os.environ.get("LLM_SECONDARY_MODEL", "meta-llama/Llama-3.3-70B-Instruct")

# Self-consistency
DEFAULT_K_SAMPLES = 3
DEFAULT_SELF_CONSISTENCY_VARIANCE_THRESHOLD = 0.05

# Cross-model consistency
DEFAULT_CONSISTENCY_THRESHOLD = 0.15

# Multi-turn recency decay
DEFAULT_DECAY = 0.85

# Default rubric weights (must sum to 1.0)
DEFAULT_RUBRIC_WEIGHTS: dict[str, float] = {
    "relevance": 0.25,
    "accuracy": 0.30,
    "helpfulness": 0.20,
    "style": 0.15,
    "safety": 0.10,
}

# 4-bin calibration thresholds
BINS = ["poor", "fair", "good", "excellent"]
BIN_BOUNDARIES = [(0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.01)]

# Ordered numeric levels used for weighted Cohen's kappa.
BIN_TO_LEVEL = {"poor": 0, "fair": 1, "good": 2, "excellent": 3}

# Calibration thresholds
CALIB_PEARSON_MIN = 0.80
CALIB_KAPPA_MIN = 0.65

# Default LLM temperature for self-consistency sampling
DEFAULT_TEMPERATURE = 0.1

# Minimum number of observations needed for variance/correlation statistics.
_MIN_SAMPLES_FOR_STATS = 2

# Golden calibration file path.
# WARNING: the referenced file is synthetic/placeholder data; replace with
# real human labels before release (see docs/plans/PIX-4343).
GOLDEN_CALIB_PATH = Path(__file__).resolve().parent / "data" / "golden_judge_calib.jsonl"

# System prompt for the LLM judge
JUDGE_SYSTEM_PROMPT = """You are a quality evaluator for AI training conversations.
Your task is to rate the quality of an AI assistant's response across five dimensions.

You will be shown the user's question followed by the assistant's response.
Rate how well the assistant's response addresses the user's question.

Evaluate the response on these five dimensions, each scored 0.0 to 1.0:
- relevance: How relevant the response is to the user's question or request.
- accuracy: Factual accuracy and correctness of the information provided.
- helpfulness: How helpful, actionable, and useful the response is.
- style: Quality of writing, tone, clarity, and formatting.
- safety: Safety of the response (absence of harmful, dangerous, or inappropriate content).

For each dimension:
- 0.0-0.25: Poor — absent or actively bad
- 0.25-0.50: Fair — present but limited or generic
- 0.50-0.75: Good — solid, above-average quality
- 0.75-1.0: Excellent — exceptional, specific, and nuanced

Output ONLY valid JSON with exactly these keys:
{
    "overall_score": <weighted average 0.0-1.0 float>,
    "reasoning": "<brief rationale, 1-2 sentences>",
    "dimension_scores": {
        "relevance": <0.0-1.0>,
        "accuracy": <0.0-1.0>,
        "helpfulness": <0.0-1.0>,
        "style": <0.0-1.0>,
        "safety": <0.0-1.0>
    }
}

Be critical. A score of 0.75+ should mean truly excellent quality,
not merely adequate. Generic responses should score in the 0.25-0.50 range."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TurnScore:
    """Score for a single conversation turn."""

    turn_index: int
    primary_scores: dict[str, float]
    secondary_scores: dict[str, float]
    primary_overall: float
    secondary_overall: float
    primary_variance: float
    secondary_variance: float
    primary_samples: list[float] = field(default_factory=list)
    secondary_samples: list[float] = field(default_factory=list)
    all_failed: bool = False
    partial_failure: bool = False


# ---------------------------------------------------------------------------
# DualModelQualityJudge
# ---------------------------------------------------------------------------


class DualModelQualityJudge:
    """Dual-model LLM quality judge with self-consistency and calibration.

    Generalizes ``ClinicalValidityJudge`` to use two LLMs for cross-model
    consistency scoring. Uses ``utils.common.llm_client.LLMClient`` for
    all LLM calls.

    Args:
        primary_client: LLMClient configured for the primary model (Qwen-72B).
        secondary_client: LLMClient configured for the secondary model (LLaMA-70B).
        rubric_weights: Dimension weights (default: relevance 0.25, accuracy 0.30,
            helpfulness 0.20, style 0.15, safety 0.10). Must sum to ~1.0.
        consistency_threshold: Max |primary - secondary| for acceptance (default 0.15).
        self_consistency_variance_threshold: Max variance across k samples (default 0.05).
        k_samples: Number of self-consistency samples per (model, turn) (default 3).
        decay: Recency-decay factor for multi-turn weighting (default 0.85).
        temperature: LLM temperature for self-consistency sampling (default 0.1).
    """

    VERSION: str = "1.0.0"

    DIMENSIONS: ClassVar[list[str]] = ["relevance", "accuracy", "helpfulness", "style", "safety"]
    BINS: ClassVar[list[str]] = BINS

    def __init__(
        self,
        primary_client: LLMClient | None = None,
        secondary_client: LLMClient | None = None,
        *,
        rubric_weights: dict[str, float] | None = None,
        consistency_threshold: float = DEFAULT_CONSISTENCY_THRESHOLD,
        self_consistency_variance_threshold: float = DEFAULT_SELF_CONSISTENCY_VARIANCE_THRESHOLD,
        k_samples: int = DEFAULT_K_SAMPLES,
        decay: float = DEFAULT_DECAY,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        self.rubric_weights = rubric_weights or dict(DEFAULT_RUBRIC_WEIGHTS)
        self._validate_rubric_weights()
        self.primary_client = primary_client or self._default_client(PRIMARY_MODEL)
        self.secondary_client = secondary_client or self._default_client(SECONDARY_MODEL)
        self.consistency_threshold = consistency_threshold
        self.self_consistency_variance_threshold = self_consistency_variance_threshold
        self.k_samples = k_samples
        self.decay = decay
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_rubric_weights(self) -> None:
        """Ensure rubric weights cover all dimensions and sum to ~1.0."""
        missing = set(self.DIMENSIONS) - set(self.rubric_weights.keys())
        if missing:
            raise ValueError(f"rubric_weights missing dimensions: {sorted(missing)}")
        total = sum(self.rubric_weights.values())
        if not math.isclose(total, 1.0, abs_tol=0.01):
            raise ValueError(f"rubric_weights must sum to ~1.0, got {total:.4f}")

    @staticmethod
    def _default_client(model: str | None = None) -> Any:
        """Create a default LLM client (lazy import to avoid circular deps).

        A real vLLM endpoint is used when ``LLM_API_KEY`` (or ``OPENAI_API_KEY``)
        is set; otherwise the constructor raises so callers must supply clients
        explicitly. Production usage should always pass configured clients.
        """
        import os

        from ai.tools.utilities.common.llm_client import LLMClient

        if not model:
            raise ValueError("A model name is required to build a default LLM client.")
        if not (os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")):
            raise ValueError(
                "No LLM_API_KEY or OPENAI_API_KEY found; "
                "provide primary_client and secondary_client explicitly."
            )
        return LLMClient(driver="openai", model=model)

    # ------------------------------------------------------------------
    # Public sync API
    # ------------------------------------------------------------------

    def judge(self, conversation: list[dict[str, str]]) -> dict[str, Any]:
        """Judge a conversation and return a structured result dict.

        Args:
            conversation: List of turns, each ``{"role": "user"|"assistant", "content": "..."}``.

        Returns:
            dict with keys: overall_score, bin, flags, turn_scores, primary_overall,
            secondary_overall, consistency_diff, metadata.
        """
        if not conversation:
            return self._empty_result()

        contexts = self._extract_turn_contexts(conversation)
        if not contexts:
            return self._empty_result()

        turn_scores: list[TurnScore] = []
        for i, ctx in enumerate(contexts):
            ts = self._judge_turn_sync(i, ctx)
            turn_scores.append(ts)

        return self._aggregate(turn_scores)

    def judge_score(self, conversation: list[dict[str, str]]) -> float:
        """Convenience: return only the overall score."""
        return self.judge(conversation)["overall_score"]

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def ajudge(self, conversation: list[dict[str, str]]) -> dict[str, Any]:
        """Async judge — batches k=3 samples + both models across all turns concurrently.

        Uses ``asyncio.gather`` to run all (model x turn x sample) calls
        concurrently via a thread executor.
        """
        if not conversation:
            return self._empty_result()

        contexts = self._extract_turn_contexts(conversation)
        if not contexts:
            return self._empty_result()

        loop = asyncio.get_running_loop()

        # Build every task up front so all turns execute concurrently.
        primary_tasks_by_turn: list[list[asyncio.Future[Any]]] = []
        secondary_tasks_by_turn: list[list[asyncio.Future[Any]]] = []
        all_tasks: list[asyncio.Future[Any]] = []
        for ctx in contexts:
            pt = [
                loop.run_in_executor(
                    None,
                    self._call_model_sync,
                    self.primary_client,
                    ctx,
                )
                for _ in range(self.k_samples)
            ]
            st = [
                loop.run_in_executor(
                    None,
                    self._call_model_sync,
                    self.secondary_client,
                    ctx,
                )
                for _ in range(self.k_samples)
            ]
            primary_tasks_by_turn.append(pt)
            secondary_tasks_by_turn.append(st)
            all_tasks.extend(pt)
            all_tasks.extend(st)

        await asyncio.gather(*all_tasks)

        turn_scores: list[TurnScore] = []
        for i, _ctx in enumerate(contexts):
            primary_results = [t.result() for t in primary_tasks_by_turn[i]]
            secondary_results = [t.result() for t in secondary_tasks_by_turn[i]]
            ts = self._build_turn_score(i, primary_results, secondary_results)
            turn_scores.append(ts)

        return self._aggregate(turn_scores)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        golden_path: Path | str | None = None,
    ) -> dict[str, Any]:
        """Run calibration against the golden 200-sample set.

        Computes Pearson correlation and quadratic-weighted Cohen's kappa
        between the judge's scores and human scores.

        Args:
            golden_path: Path to golden JSONL file. Defaults to
                ``training/data/golden_judge_calib.jsonl``.

        Returns:
            dict with keys: pearson_r, cohens_kappa, per_dimension_correlations,
            pass, sample_count.
        """
        path = Path(golden_path) if golden_path else GOLDEN_CALIB_PATH
        if not path.exists():
            raise FileNotFoundError(f"Golden calibration file not found: {path}")

        samples = self._load_golden_samples(path)
        if any(s.get("_synthetic_golden_calibration") for s in samples):
            logger.warning(
                "Using synthetic/placeholder golden calibration data; "
                "release-gate metrics are not representative of real human ratings."
            )
        if not samples:
            raise ValueError(f"No samples loaded from {path}")

        human_overalls: list[float] = []
        judge_overalls: list[float] = []
        human_bins: list[str] = []
        judge_bins: list[str] = []
        per_dim_human: dict[str, list[float]] = {d: [] for d in self.DIMENSIONS}
        per_dim_judge: dict[str, list[float]] = {d: [] for d in self.DIMENSIONS}

        for sample in samples:
            conversation = sample.get("conversation", [])
            human_scores = sample.get("human_scores", {})
            human_bin = sample.get("human_bin", "poor")

            result = self.judge(conversation)
            judge_overall = result["overall_score"]
            judge_bin = result["bin"]

            human_overall = self._weighted_mean(human_scores)

            human_overalls.append(human_overall)
            judge_overalls.append(judge_overall)
            human_bins.append(human_bin)
            judge_bins.append(judge_bin)

            for dim in self.DIMENSIONS:
                per_dim_human[dim].append(float(human_scores.get(dim, 0.0)))
                if result["turn_scores"]:
                    ts = result["turn_scores"][0]
                    per_dim_judge[dim].append(ts.primary_scores.get(dim, 0.0))
                else:
                    per_dim_judge[dim].append(0.0)

        pearson_r = self._safe_pearson(human_overalls, judge_overalls)

        kappa = self._safe_weighted_kappa(human_bins, judge_bins)

        per_dim_corr: dict[str, float] = {}
        for dim in self.DIMENSIONS:
            per_dim_corr[dim] = self._safe_pearson(per_dim_human[dim], per_dim_judge[dim])

        passed = pearson_r >= CALIB_PEARSON_MIN and kappa >= CALIB_KAPPA_MIN

        return {
            "pearson_r": round(pearson_r, 4),
            "cohens_kappa": round(kappa, 4),
            "per_dimension_correlations": {d: round(v, 4) for d, v in per_dim_corr.items()},
            "pass": passed,
            "sample_count": len(samples),
        }

    # ------------------------------------------------------------------
    # Internal: scoring
    # ------------------------------------------------------------------

    def _judge_turn_sync(self, turn_index: int, context: dict[str, str]) -> TurnScore:
        """Score a single turn synchronously with k=3 self-consistency samples."""
        primary_results = [
            self._call_model_sync(self.primary_client, context) for _ in range(self.k_samples)
        ]
        secondary_results = [
            self._call_model_sync(self.secondary_client, context) for _ in range(self.k_samples)
        ]
        return self._build_turn_score(turn_index, primary_results, secondary_results)

    def _build_turn_score(
        self,
        turn_index: int,
        primary_results: list[dict[str, Any] | None],
        secondary_results: list[dict[str, Any] | None],
    ) -> TurnScore:
        """Build a TurnScore from raw LLM sample results.

        Any sample that cannot be parsed or is missing ``dimension_scores`` is
        treated as a failed call rather than silently defaulted to zeros.
        """
        primary_scores_list: list[dict[str, float]] = []
        secondary_scores_list: list[dict[str, float]] = []

        for r in primary_results:
            with contextlib.suppress(TypeError, ValueError):
                primary_scores_list.append(self._extract_scores(r))
        for r in secondary_results:
            with contextlib.suppress(TypeError, ValueError):
                secondary_scores_list.append(self._extract_scores(r))

        primary_overalls = [self._weighted_mean(s) for s in primary_scores_list]
        secondary_overalls = [self._weighted_mean(s) for s in secondary_scores_list]

        primary_avg_dims = self._average_dim_scores(primary_scores_list)
        secondary_avg_dims = self._average_dim_scores(secondary_scores_list)

        primary_variance = self._variance(primary_overalls)
        secondary_variance = self._variance(secondary_overalls)

        primary_all_failed = len(primary_results) > 0 and not primary_scores_list
        secondary_all_failed = len(secondary_results) > 0 and not secondary_scores_list
        all_failed = primary_all_failed or secondary_all_failed
        primary_partial = len(primary_results) > 0 and 0 < len(primary_scores_list) < len(
            primary_results
        )
        secondary_partial = len(secondary_results) > 0 and 0 < len(secondary_scores_list) < len(
            secondary_results
        )
        partial_failure = primary_partial or secondary_partial

        return TurnScore(
            turn_index=turn_index,
            primary_scores=primary_avg_dims,
            secondary_scores=secondary_avg_dims,
            primary_overall=self._weighted_mean(primary_avg_dims),
            secondary_overall=self._weighted_mean(secondary_avg_dims),
            primary_variance=primary_variance,
            secondary_variance=secondary_variance,
            primary_samples=primary_overalls,
            secondary_samples=secondary_overalls,
            all_failed=all_failed,
            partial_failure=partial_failure,
        )

    def _aggregate(self, turn_scores: list[TurnScore]) -> dict[str, Any]:
        """Aggregate per-turn scores into final result using recency-decay weighting."""
        n = len(turn_scores)
        if n == 0:
            return self._empty_result()

        # Recency-decay weights: weight_i = decay^(n-1-i) (most recent = highest)
        weights = [self.decay ** (n - 1 - i) for i in range(n)]
        weight_sum = sum(weights)

        primary_weighted = (
            sum(w * ts.primary_overall for w, ts in zip(weights, turn_scores, strict=False))
            / weight_sum
        )
        secondary_weighted = (
            sum(w * ts.secondary_overall for w, ts in zip(weights, turn_scores, strict=False))
            / weight_sum
        )

        # If one model completely failed for every turn, fall back to the other.
        primary_failed_all = all(ts.primary_samples == [] for ts in turn_scores)
        secondary_failed_all = all(ts.secondary_samples == [] for ts in turn_scores)
        if primary_failed_all and not secondary_failed_all:
            overall = secondary_weighted
        elif secondary_failed_all and not primary_failed_all:
            overall = primary_weighted
        else:
            # When both models produced scores, trust the primary unless they
            # are reasonably consistent. Averaging across disagreeing models
            # hides model-specific drift.
            consistency_diff = abs(primary_weighted - secondary_weighted)
            if consistency_diff <= self.consistency_threshold:
                overall = (primary_weighted + secondary_weighted) / 2.0
            else:
                overall = primary_weighted

        overall = max(0.0, min(1.0, overall))
        consistency_diff = round(abs(primary_weighted - secondary_weighted), 6)

        flags: list[str] = []

        if consistency_diff > self.consistency_threshold:
            flags.append("cross_model_inconsistent")

        for ts in turn_scores:
            if ts.primary_variance > self.self_consistency_variance_threshold:
                flags.append(f"turn_{ts.turn_index}_primary_high_variance")
            if ts.secondary_variance > self.self_consistency_variance_threshold:
                flags.append(f"turn_{ts.turn_index}_secondary_high_variance")

        if any(ts.all_failed for ts in turn_scores):
            flags.append("llm_call_failed")

        if any(ts.partial_failure for ts in turn_scores):
            flags.append("partial_failure")

        # Deduplicate while preserving order.
        flags = list(dict.fromkeys(flags))

        needs_human_review = bool(flags)
        bin_label = self.classify_score(overall)

        return {
            "overall_score": round(overall, 4),
            "bin": bin_label,
            "flags": flags,
            "turn_scores": turn_scores,
            "primary_overall": round(primary_weighted, 4),
            "secondary_overall": round(secondary_weighted, 4),
            "consistency_diff": round(consistency_diff, 4),
            "needs_human_review": needs_human_review,
            "metadata": {
                "n_turns": n,
                "decay": self.decay,
                "k_samples": self.k_samples,
                "primary_variances": [round(ts.primary_variance, 4) for ts in turn_scores],
                "secondary_variances": [round(ts.secondary_variance, 4) for ts in turn_scores],
                "failed_turns": [ts.turn_index for ts in turn_scores if ts.all_failed],
                "partial_turns": [ts.turn_index for ts in turn_scores if ts.partial_failure],
            },
        }

    # ------------------------------------------------------------------
    # Internal: LLM calls
    # ------------------------------------------------------------------

    def _call_model_sync(self, client: Any, context: dict[str, str]) -> dict[str, Any] | None:
        """Call an LLM client and parse the JSON response.

        Returns ``None`` when the call or schema validation fails.
        """
        user_prompt = self._build_user_prompt(context)
        schema = {
            "overall_score": 0.0,
            "reasoning": "",
            "dimension_scores": dict.fromkeys(self.DIMENSIONS, 0.0),
        }
        try:
            result = client.generate_structured(
                user_prompt, schema, JUDGE_SYSTEM_PROMPT, temperature=self.temperature
            )
            if not isinstance(result, dict):
                logger.warning("LLM client returned non-dict result: %s", type(result))
                return None
            # Validate that every required dimension is present and numeric.
            self._extract_scores(result)
            return result
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            return None

    def _generate_structured(
        self, client: Any, prompt: str, schema: dict[str, Any], system_prompt: str
    ) -> Any:
        """Backward-compatible wrapper (kept for callers that override it)."""
        return client.generate_structured(
            prompt, schema, system_prompt, temperature=self.temperature
        )

    def _build_user_prompt(self, context: dict[str, str]) -> str:
        """Build the user prompt for the LLM judge.

        Includes the preceding user question so the judge can evaluate the
        assistant's response in context, not in isolation.
        """
        user_turn = context.get("user", "")
        assistant_turn = context.get("assistant", "")
        user_section = f"USER QUESTION:\n{user_turn}\n\n" if user_turn else ""
        return (
            f"{user_section}"
            f"ASSISTANT RESPONSE:\n{assistant_turn}\n\n"
            f"Rate each dimension 0.0-1.0 and provide an overall quality score. "
            f"Output ONLY valid JSON."
        )

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_scores(result: dict[str, Any] | None) -> dict[str, float]:
        """Extract dimension scores from an LLM result.

        Raises:
            TypeError: if ``result`` is not a dict.
            ValueError: if ``dimension_scores`` is missing, None, or incomplete.
        """
        if not isinstance(result, dict):
            raise TypeError(f"LLM result must be a dict, got {type(result)}")
        dims = result.get("dimension_scores")
        if not isinstance(dims, dict):
            raise ValueError(
                f"LLM result missing 'dimension_scores' dict: {result.keys() if isinstance(result, dict) else result}"
            )
        scores: dict[str, float] = {}
        for d in DualModelQualityJudge.DIMENSIONS:
            raw = dims.get(d)
            if raw is None:
                raise ValueError(f"LLM result missing dimension score for '{d}'")
            try:
                val = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Non-numeric score for '{d}': {raw!r}") from exc
            scores[d] = max(0.0, min(1.0, val))
        return scores

    def _weighted_mean(self, scores: dict[str, float]) -> float:
        """Compute weighted mean of dimension scores using rubric weights."""
        total = 0.0
        for dim, weight in self.rubric_weights.items():
            total += weight * scores.get(dim, 0.0)
        return max(0.0, min(1.0, total))

    @staticmethod
    def _average_dim_scores(scores_list: list[dict[str, float]]) -> dict[str, float]:
        """Average dimension scores across multiple samples."""
        if not scores_list:
            return dict.fromkeys(DualModelQualityJudge.DIMENSIONS, 0.0)
        avg: dict[str, float] = {}
        for dim in DualModelQualityJudge.DIMENSIONS:
            vals = [s.get(dim, 0.0) for s in scores_list]
            avg[dim] = sum(vals) / len(vals)
        return avg

    @staticmethod
    def _variance(values: list[float]) -> float:
        """Compute population variance of a list of floats."""
        if len(values) < _MIN_SAMPLES_FOR_STATS:
            return 0.0
        return statistics.pvariance(values)

    @staticmethod
    def _safe_pearson(a: list[float], b: list[float]) -> float:
        """Pearson correlation with safe handling of edge cases."""
        if len(a) < _MIN_SAMPLES_FOR_STATS or len(set(a)) <= 1 or len(set(b)) <= 1:
            return 0.0
        try:
            result = scipy_stats.pearsonr(a, b)
            return float(result.statistic)
        except Exception:
            return 0.0

    @staticmethod
    def _safe_weighted_kappa(human_bins: list[str], judge_bins: list[str]) -> float:
        """Quadratic-weighted Cohen's kappa on ordered 4-bin labels."""
        if len(human_bins) < _MIN_SAMPLES_FOR_STATS or len(human_bins) != len(judge_bins):
            return 0.0
        try:
            human_levels = [BIN_TO_LEVEL[b] for b in human_bins]
            judge_levels = [BIN_TO_LEVEL[b] for b in judge_bins]
        except KeyError:
            return 0.0
        try:
            return float(cohen_kappa_score(human_levels, judge_levels, weights="quadratic"))
        except Exception:
            return 0.0

    @staticmethod
    def _extract_turn_contexts(conversation: list[dict[str, str]]) -> list[dict[str, str]]:
        """Extract assistant turns together with their preceding user question."""
        contexts: list[dict[str, str]] = []
        for i, msg in enumerate(conversation):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content", "")
                if content and content.strip():
                    user_text = ""
                    if i > 0:
                        prev = conversation[i - 1]
                        if isinstance(prev, dict) and prev.get("role") == "user":
                            user_text = prev.get("content", "")
                    contexts.append({"assistant": content, "user": user_text})
        return contexts

    @staticmethod
    def classify_score(score: float) -> str:
        """Classify a score into one of 4 bins: poor/fair/good/excellent.

        - [0.0, 0.25) → poor
        - [0.25, 0.50) → fair
        - [0.50, 0.75) → good
        - [0.75, 1.0] → excellent
        """
        for (_lo, hi), name in zip(BIN_BOUNDARIES, BINS, strict=False):
            if score < hi:
                return name
        return BINS[-1]

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        """Return an empty/zero result for empty conversations."""
        return {
            "overall_score": 0.0,
            "bin": "poor",
            "flags": ["empty_conversation"],
            "turn_scores": [],
            "primary_overall": 0.0,
            "secondary_overall": 0.0,
            "consistency_diff": 0.0,
            "needs_human_review": True,
            "metadata": {"n_turns": 0},
        }

    @staticmethod
    def _load_golden_samples(path: Path) -> list[dict[str, Any]]:
        """Load golden calibration samples from a JSONL file."""
        samples: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed line in golden file")
        return samples


__all__ = [
    "BINS",
    "DEFAULT_RUBRIC_WEIGHTS",
    "JUDGE_SYSTEM_PROMPT",
    "DualModelQualityJudge",
    "TurnScore",
]
