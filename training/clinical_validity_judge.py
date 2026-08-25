"""LLM-based clinical validity judge for therapeutic training data.

Evaluates the clinical quality of therapeutic responses using a prompted
LLM judge (NVIDIA NeMo API). Falls back to the keyword-density-based
ClinicalValidityScorer when the API is unavailable.

The judge asks the LLM to rate a response across clinical dimensions
and produce a structured JSON score, providing more nuanced evaluation
than pure regex keyword matching.

Usage:
    uv run python -m training.clinical_validity_judge --text "Your text here"
    uv run python -m training.clinical_validity_judge --text "..." --detail
"""

from __future__ import annotations

import json
import logging
import re
import sys
from typing import TYPE_CHECKING, Any

from training.clinical_validity_scorer import ClinicalValidityScorer

if TYPE_CHECKING:
    from training.sdg_pipeline import NemoConfig

logger = logging.getLogger("clinical_validity_judge")

# Non-English script detection — matches the scorer
_NON_ENGLISH_RE = re.compile(
    "["
    "\u4e00-\u9fff"  # CJK Unified Ideographs
    "\u3040-\u309f"  # Hiragana
    "\u30a0-\u30ff"  # Katakana
    "\uac00-\ud7af"  # Hangul Syllables
    "\u0400-\u04ff"  # Cyrillic
    "\u0600-\u06ff"  # Arabic
    "\u0e00-\u0e7f"  # Thai
    "\u0f00-\u0fff"  # Tibetan
    "]"
)
_NON_ENGLISH_RATIO = 0.30


CLINICAL_EVALUATION_SYSTEM_PROMPT = """You are a clinical quality evaluator for therapeutic AI training data.
Your task is to rate how clinically valid a therapist's response is.

Evaluate the response on these six dimensions, each scored 0.0 to 1.0:
- technique: Use of evidence-based therapeutic techniques (CBT, DBT, MI, ACT, etc.)
- alliance: Therapeutic alliance, rapport, validation, and collaboration
- structure: Clinical session structure (assessment, intervention, planning, closure)
- cultural: Cultural competence and inclusive language
- ebp: Evidence-based practice references and clinical reasoning
- dsm5: DSM-5 diagnostic pattern recognition and appropriate clinical language

For each dimension, consider:
- 0.0-0.3: No evidence of this dimension
- 0.3-0.6: Some evidence present but limited or generic
- 0.6-0.8: Good application of this dimension
- 0.8-1.0: Excellent, specific, and nuanced application

Output ONLY valid JSON with exactly these keys:
{
    "clinical_validity_score": <overall 0.0-1.0 float>,
    "reasoning": "<brief rationale for the score, 1-2 sentences>",
    "dimension_scores": {
        "technique": <0.0-1.0>,
        "alliance": <0.0-1.0>,
        "structure": <0.0-1.0>,
        "cultural": <0.0-1.0>,
        "ebp": <0.0-1.0>,
        "dsm5": <0.0-1.0>
    }
}

Be critical — a score of 0.8+ should mean truly excellent clinical work,
not merely adequate. Generic supportive responses that lack specific
therapeutic technique should score in the 0.3-0.5 range."""


# Non-clinical generalization (blueprint B.3.1). Five domain-agnostic
# dimensions with 4-bin calibration, mirroring ``llm_quality_judge.py``.
GENERAL_EVALUATION_SYSTEM_PROMPT = """You are a quality evaluator for AI training data.
Your task is to rate how good an assistant's response is, independent of domain.

Evaluate the response on these five dimensions, each scored 0.0 to 1.0:
- relevance: How relevant the response is to the user's question or request
- accuracy: Factual accuracy and correctness of the information provided
- helpfulness: How helpful, actionable, and useful the response is
- style: Quality of writing, tone, clarity, and formatting
- safety: Safety of the response (absence of harmful or inappropriate content)

For each dimension, use 4 bins:
- 0.0-0.25: Poor — absent or actively bad
- 0.25-0.50: Fair — present but limited or generic
- 0.50-0.75: Good — solid, above-average quality
- 0.75-1.0: Excellent — exceptional, specific, and nuanced

Output ONLY valid JSON with exactly these keys:
{
    "clinical_validity_score": <overall 0.0-1.0 float>,
    "reasoning": "<brief rationale for the score, 1-2 sentences>",
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


# Domain rubric registry: dimension keys + system prompt per domain.
# ``clinical`` keeps the original 6-dim rubric; ``general`` is the
# non-clinical 5-dim rubric (blueprint B.3.1).
DOMAIN_DIMENSIONS: dict[str, tuple[str, ...]] = {
    "clinical": ("technique", "alliance", "structure", "cultural", "ebp", "dsm5"),
    "general": ("relevance", "accuracy", "helpfulness", "style", "safety"),
}

DOMAIN_SYSTEM_PROMPTS: dict[str, str] = {
    "clinical": CLINICAL_EVALUATION_SYSTEM_PROMPT,
    "general": GENERAL_EVALUATION_SYSTEM_PROMPT,
}


class ClinicalValidityJudge:
    """LLM-based clinical validity judge with regex fallback.

    Uses NeMo API as a prompted LLM judge to evaluate therapeutic
    response quality. Falls back to the keyword-density ClinicalValidityScorer
    when the API is unavailable.

    All methods are classmethods for a consistent, simple API.
    """

    VERSION: str = "1.0.0"

    # Thresholds match ClinicalValidityScorer for consistent routing
    EXCLUDE_THRESHOLD: float = 0.4
    ACCEPT_THRESHOLD: float = 0.6

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def _dimensions(cls, domain: str) -> tuple[str, ...]:
        """Return the dimension keys for a domain (clinical or general)."""
        if domain not in DOMAIN_DIMENSIONS:
            raise ValueError(f"Unknown domain {domain!r}; expected one of {sorted(DOMAIN_DIMENSIONS)}")
        return DOMAIN_DIMENSIONS[domain]

    @classmethod
    def _empty_detail(cls, domain: str) -> dict[str, float]:
        return dict.fromkeys(cls._dimensions(domain), 0.0)

    @classmethod
    def score(
        cls,
        text: str,
        nemo_config: NemoConfig | None = None,
        *,
        domain: str = "clinical",
    ) -> float:
        """Compute overall validity score in [0.0, 1.0].

        Uses LLM judge when nemo_config is provided, falls back to
        ClinicalValidityScorer otherwise (clinical domain only).
        """
        result = cls.evaluate(text, nemo_config, domain=domain)
        return result.get("validity_score", 0.0)

    @classmethod
    def evaluate(
        cls,
        text: str | None,
        nemo_config: NemoConfig | None = None,
        *,
        domain: str = "clinical",
    ) -> dict[str, Any]:
        """Evaluate text and return structured score.

        Args:
            text: Response text to evaluate.
            nemo_config: NeMo API config. ``None`` triggers regex fallback for
                the clinical domain; the general domain has no regex scorer.
            domain: ``"clinical"`` (6-dim rubric) or ``"general"`` (5-dim
                non-clinical rubric, blueprint B.3.1). Default ``"clinical"``.

        Returns:
            dict with keys: validity_score, flags (list), category (str),
            detail (dict of dimension scores).
        """
        dims = cls._dimensions(domain)

        # --- Empty/None guard ---
        if not text or not isinstance(text, str) or not text.strip():
            return {
                "validity_score": 0.0,
                "flags": ["empty_input"],
                "category": "unknown",
                "detail": dict.fromkeys(dims, 0.0),
            }

        # --- Non-English guard (skip LLM call) ---
        if cls._detect_non_english(text):
            return {
                "validity_score": 0.0,
                "flags": ["non_english_content"],
                "category": "unknown",
                "detail": dict.fromkeys(dims, 0.0),
            }

        fallback = False
        judge_result: dict[str, Any] | None = None
        if nemo_config is None:
            fallback = True
        else:
            try:
                judge_result = cls._call_judge(text, nemo_config, domain=domain)
            except Exception as e:
                logger.warning(
                    "ClinicalValidityJudge: LLM judge call failed (%s); falling back to scorer",
                    e,
                )
                fallback = True
            if judge_result is None:
                fallback = True

        if fallback:
            if domain == "clinical":
                result = ClinicalValidityScorer.score_with_flags(text)
                if "fallback_regex" not in result["flags"]:
                    result["flags"].append("fallback_regex")
                return result
            # General domain has no keyword scorer — return a zeroed result.
            return {
                "validity_score": 0.0,
                "flags": ["fallback_unavailable"],
                "category": "unknown",
                "detail": dict.fromkeys(dims, 0.0),
            }

        return judge_result

    @classmethod
    def score_with_flags(
        cls,
        text: str | None,
        nemo_config: NemoConfig | None = None,
        *,
        domain: str = "clinical",
    ) -> dict[str, Any]:
        """Alias for evaluate() — same output schema as ClinicalValidityScorer."""
        return cls.evaluate(text, nemo_config, domain=domain)

    @classmethod
    def classify_score(cls, score: float) -> str:
        """Three-tier routing matching ClinicalValidityScorer thresholds."""
        return ClinicalValidityScorer.classify_score(score)

    # ------------------------------------------------------------------
    # Internal: LLM judge
    # ------------------------------------------------------------------

    @classmethod
    def _build_evaluation_prompt(cls, text: str, domain: str = "clinical") -> str:
        """Build a structured evaluation prompt for the LLM judge."""
        label = "clinical quality of this therapist response" if domain == "clinical" else "quality of this response"
        return (
            f"Evaluate the {label}:\n\n"
            f"RESPONSE:\n{text}\n\n"
            f"Rate each dimension 0.0-1.0 and provide an overall score. "
            f"Output ONLY valid JSON."
        )

    @classmethod
    def _call_judge(
        cls,
        text: str,
        nemo_config: Any,
        *,
        domain: str = "clinical",
    ) -> dict[str, Any] | None:
        """Call NeMo API and parse the judge's evaluation.

        Returns structured dict or None on failure.
        """
        # Lazy import to avoid circular dependency with sdg_pipeline.py
        from training.sdg_pipeline import _call_nemo  # type: ignore[attr-defined]

        system_prompt = DOMAIN_SYSTEM_PROMPTS[domain]
        user_prompt = cls._build_evaluation_prompt(text, domain)

        raw_response = _call_nemo(user_prompt, nemo_config, system_prompt=system_prompt)

        if not raw_response:
            return None

        return cls._parse_judge_response(raw_response, text, domain=domain)

    @classmethod
    def _parse_judge_response(
        cls,
        raw_response: str,
        original_text: str,
        *,
        domain: str = "clinical",
    ) -> dict[str, Any] | None:
        """Parse the LLM's JSON response into the standard output schema.

        Extracts the JSON blob (handling ```json ... ``` wrapping),
        validates fields, and returns the standard dict.
        """
        dims = cls._dimensions(domain)
        text = raw_response.strip()

        # Strip markdown code fences if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(
                "ClinicalValidityJudge: failed to parse NeMo response as JSON: %s",
                raw_response[:200],
            )
            return None

        # Extract overall score
        overall = parsed.get("clinical_validity_score")
        if overall is None or not isinstance(overall, (int, float)):
            logger.warning(
                "ClinicalValidityJudge: missing or invalid clinical_validity_score in response"
            )
            return None

        overall = max(0.0, min(1.0, float(overall)))

        # Extract dimension scores
        dimension_scores = parsed.get("dimension_scores", {})
        detail: dict[str, float] = {}
        for dim in dims:
            raw = dimension_scores.get(dim, 0.0)
            detail[dim] = max(0.0, min(1.0, float(raw))) if isinstance(raw, (int, float)) else 0.0

        # Category = highest-scoring dimension
        category = "unknown"
        if detail:
            best_dim: str = max(detail, key=lambda k: detail[k])  # type: ignore[arg-type]
            if detail[best_dim] > 0.0:
                category = best_dim

        # Build flags
        flags: list[str] = []
        for dim, score in detail.items():
            if score >= 0.3:
                flags.append(f"{dim}_present")
        if overall < cls.EXCLUDE_THRESHOLD:
            flags.append("below_exclude_threshold")
        elif overall < cls.ACCEPT_THRESHOLD:
            flags.append("annotation_needed")

        return {
            "validity_score": overall,
            "flags": flags,
            "category": category,
            "detail": detail,
        }

    # ------------------------------------------------------------------
    # Internal: Non-English detection
    # ------------------------------------------------------------------

    @classmethod
    def _detect_non_english(cls, text: str) -> bool:
        """Check if text contains a significant proportion of non-English scripts."""
        if not text or not isinstance(text, str):
            return False
        non_english_chars = len(_NON_ENGLISH_RE.findall(text))
        total_chars = max(1, len(text.strip()))
        return (non_english_chars / total_chars) > _NON_ENGLISH_RATIO


# =========================================================================
# CLI entry point
# =========================================================================


def _build_nemo_config_from_env() -> Any | None:
    """Build NemoConfig from environment variables if available."""
    import os

    # Lazy import to avoid circular dependency with sdg_pipeline.py
    from training.sdg_pipeline import NemoConfig  # type: ignore[attr-defined]

    endpoint = os.getenv("NEMO_ENDPOINT", "") or os.getenv("NVIDIA_BASE_URL", "")
    api_key = os.getenv("NEMO_API_KEY", "") or os.getenv("NVIDIA_API_KEY", "")

    if not endpoint or not api_key:
        return None

    return NemoConfig(
        endpoint=endpoint,
        api_key=api_key,
        model=os.getenv("NEMO_MODEL", "mistral-nemo"),
    )


def main() -> None:
    """CLI entry point.

    Usage:
        uv run python -m training.clinical_validity_judge --text "Your text here"
        uv run python -m training.clinical_validity_judge --text "..." --detail
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate clinical validity of therapeutic text using LLM judge"
    )
    parser.add_argument("--text", type=str, default=None, help="Text to evaluate")
    parser.add_argument(
        "--detail", action="store_true", help="Include per-dimension scores in output"
    )
    args = parser.parse_args()

    text = args.text
    if text is None:
        text = sys.stdin.read().strip()

    nemo_config = _build_nemo_config_from_env()

    if nemo_config is None:
        print(
            "Error: NEMO_API_KEY (or NVIDIA_API_KEY) and NEMO_ENDPOINT (or NVIDIA_BASE_URL) must be set.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.detail:
        result = ClinicalValidityJudge.evaluate(text, nemo_config)
    else:
        score = ClinicalValidityJudge.score(text, nemo_config)
        result = {
            "validity_score": score,
            "classification": ClinicalValidityJudge.classify_score(score),
        }

    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
