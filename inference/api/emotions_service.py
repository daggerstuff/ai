"""
Emotion Analysis API Endpoint

FastAPI endpoint for LLM-based emotion analysis (GLM/Qwen/Mistral — LLaMA is
banned per standing model policy) with FHE encryption support. Wires the FHE
ciphertext hash through to the R1 receipt system.
"""

import hashlib
import json
import logging
import os
import re
import time
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field

from ai.qa.validation.inference_safety_filter import InferenceSafetyFilter, SafetyFilterMode, SafetyLevel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["emotions"])
app = FastAPI(title="Emotion Analysis Service", version="1.0.0")
app.include_router(router)

_safety_filter_state: dict[str, InferenceSafetyFilter | None] = {"filter": None}

# ── LLM client configuration (env-driven, mirrors ai.research.nvidia_llm_callback) ──

DEFAULT_EMOTION_MODEL = "z-ai/glm-5.3-flash"
DEFAULT_EMOTION_BASE_URL = "https://integrate.api.nvidia.com/v1"

_client_cache: dict[str, Any] = {"client": None, "key": None}


class EmotionAnalysisRequest(BaseModel):
    """Request body for emotion analysis."""

    text: str = Field(description="Text to analyze for emotions")
    fhe_ciphertext_hash: str | None = Field(
        default=None,
        description="SHA-256 hash of the FHE-encrypted ciphertext (optional, for R1 receipt binding)",
    )
    model: str = Field(
        default="qwen-emotion-v1.0",
        description="Model version identifier (e.g., qwen-emotion-v1.0, glm-4, mistral)",
    )
    analysis_type: str = Field(
        default="multidimensional",
        description="Type of emotion analysis to perform",
    )
    return_confidence: bool = Field(
        default=True,
        description="Include confidence scores in response",
    )
    return_dimensions: bool = Field(
        default=True,
        description="Include emotion dimensions (valence, arousal, dominance) in response",
    )


class EmotionAnalysisResponse(BaseModel):
    """Response body with emotion analysis results."""

    emotions: list[dict[str, Any]] = Field(description="Detected emotions with type, intensity, confidence")
    dimensions: dict[str, float] = Field(description="Emotion dimensions (valence, arousal, dominance)")
    confidence: float = Field(description="Overall confidence score (0.0-1.0)")
    metadata: dict[str, Any] = Field(description="Additional metadata about the analysis")
    receipt_root_hash: str | None = Field(
        default=None,
        description="R1 cryptographic receipt root hash (if receipt system is active)",
    )


class EmotionAnalyzer:
    """
    LLM-backed emotion analysis.

    Sends the (PII-scrubbed) text to an OpenAI-compatible endpoint with a
    structured-output prompt, then parses the JSON response. Falls back to
    the lexicon-based EmotionClassifier when no LLM endpoint is configured
    or the call fails.
    """

    SYSTEM_PROMPT = (
        "You are a clinical emotion-analysis engine for a mental-health "
        "platform. Analyze the emotional content of the user's text and "
        "respond with ONLY a JSON object — no markdown fences, no prose — "
        "with exactly these keys:\n"
        '{"emotions": [{"type": "<plutchik category>", "intensity": 0.0-1.0, '
        '"confidence": 0.0-1.0}], "valence": 0.0-1.0, "arousal": 0.0-1.0, '
        '"dominance": 0.0-1.0, "overall_confidence": 0.0-1.0}\n'
        "Use Plutchik primary categories where possible (joy, sadness, anger, "
        "fear, surprise, disgust, trust, anticipation); closely-related "
        "descriptors are acceptable when no primary category fits. "
        "valence: 0=negative to 1=positive. arousal: 0=calm to 1=agitated. "
        "dominance: 0=helpless to 1=in-control."
    )

    def __init__(self) -> None:
        self.base_url = os.environ.get("LLM_BASE_URL", DEFAULT_EMOTION_BASE_URL)
        self.api_key = os.environ.get("LLM_API_KEY") or os.environ.get("NVIDIA_API_KEY")
        self.model = os.environ.get("LLM_MODEL", DEFAULT_EMOTION_MODEL)
        self._fallback_classifier: Any = None

    def _client(self) -> Any:
        from openai import OpenAI  # noqa: PLC0415 - heavy SDK, imported on first use

        cached_client = _client_cache["client"]
        cache_key = (self.api_key, self.base_url)
        if cached_client is not None and _client_cache["key"] == cache_key:
            return cached_client
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        _client_cache["client"] = client
        _client_cache["key"] = cache_key
        return client

    def _fallback(self) -> Any:
        if self._fallback_classifier is None:
            from ai.research.emotion_classifier import EmotionClassifier  # noqa: PLC0415 - optional fallback path

            self._fallback_classifier = EmotionClassifier()
        return self._fallback_classifier

    def _parse_llm_json(self, content: str) -> dict[str, Any]:
        """Parse the model's JSON object, tolerating markdown fences."""
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        raw = fenced.group(1) if fenced else content
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in LLM response")
        return json.loads(raw[start : end + 1])

    @staticmethod
    def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
        try:
            return max(low, min(high, float(value)))
        except (TypeError, ValueError):
            return 0.5

    def analyze(self, text: str, model_label: str) -> dict[str, Any]:
        """Run emotion analysis; returns the response payload fields."""
        if self.api_key:
            try:
                result = self._analyze_llm(text)
                result["metadata"]["engine"] = "llm"
                return result
            except Exception as exc:
                logger.warning("LLM emotion analysis failed, falling back to lexicon: %s", exc)
        else:
            logger.debug("No LLM endpoint configured; using lexicon emotion classifier")

        fallback = self._fallback()
        classification = fallback.classify(text)
        emotions = [
            {
                "type": category,
                "intensity": classification.category_scores.get(category, 0.0),
                "confidence": classification.category_scores.get(category, 0.0),
            }
            for category in classification.categories
        ]
        return {
            "emotions": emotions,
            "dimensions": {
                "valence": classification.valence,
                "arousal": classification.arousal,
                "dominance": classification.dominance,
            },
            "confidence": classification.top_score,
            "metadata": {
                "model_version": model_label,
                "analysis_type": "multidimensional",
                "engine": "lexicon",
            },
        }

    def _analyze_llm(self, text: str) -> dict[str, Any]:
        client = self._client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("LLM returned empty content")
        parsed = self._parse_llm_json(content)

        raw_emotions = parsed.get("emotions", [])
        emotions = []
        for item in raw_emotions:
            if not isinstance(item, dict):
                continue
            emotion_type = str(item.get("type", "unknown")).strip().lower()
            emotions.append(
                {
                    "type": emotion_type,
                    "intensity": self._clamp(item.get("intensity")),
                    "confidence": self._clamp(item.get("confidence")),
                }
            )

        dimensions = {
            "valence": self._clamp(parsed.get("valence")),
            "arousal": self._clamp(parsed.get("arousal")),
            "dominance": self._clamp(parsed.get("dominance")),
        }
        confidence = self._clamp(parsed.get("overall_confidence", parsed.get("confidence", 0.5)))

        return {
            "emotions": emotions,
            "dimensions": dimensions,
            "confidence": confidence,
            "metadata": {
                "model_version": self.model,
                "analysis_type": "multidimensional",
            },
        }


_analyzer_state: dict[str, EmotionAnalyzer | None] = {"analyzer": None}


def get_analyzer() -> EmotionAnalyzer:
    """Get or create the global emotion analyzer."""
    if _analyzer_state["analyzer"] is None:
        _analyzer_state["analyzer"] = EmotionAnalyzer()
    return _analyzer_state["analyzer"]


def get_safety_filter() -> InferenceSafetyFilter:
    """Get or create the global safety filter instance."""
    if _safety_filter_state["filter"] is None:
        _safety_filter_state["filter"] = InferenceSafetyFilter(
            safety_level=SafetyLevel.MODERATE,
            filter_mode=SafetyFilterMode.FILTER_AND_WARN,
        )
    return _safety_filter_state["filter"]


def _scrub_input_text(text: str) -> tuple[str, bool]:
    """
    HIPAA guard: scrub PII/PHI from the input before it reaches the LLM.

    Returns the scrubbed text and whether redaction occurred.
    """
    from ai.inference.services.security.pii_scrubber import scrub_pii  # noqa: PLC0415 - keeps FastAPI app import light

    scrubbed = scrub_pii(text)
    return scrubbed, scrubbed != text


@router.post(
    "/emotions",
    response_model=EmotionAnalysisResponse,
    summary="Analyze emotions in text",
    description=(
        "Perform multidimensional emotion analysis on the provided text. "
        "Input is PII-scrubbed (HIPAA) before LLM inference; "
        "when FHE encryption is used, the ciphertext hash is bound to the R1 receipt."
    ),
)
async def analyze_emotions(request: EmotionAnalysisRequest) -> EmotionAnalysisResponse:
    """
    Analyze emotions in the provided text.

    This endpoint:
    1. Scrubs PII/PHI from the input (HIPAA compliance)
    2. Performs emotion analysis using the configured LLM (GLM/Qwen/Mistral)
       with lexicon fallback
    3. Runs safety checks via InferenceSafetyFilter
    4. Emits R1 cryptographic receipt with FHE ciphertext hash binding
    """
    try:
        analyzer = get_analyzer()

        scrubbed_text, pii_redacted = _scrub_input_text(request.text)

        started = time.perf_counter()
        result = analyzer.analyze(scrubbed_text, request.model)
        processing_time_ms = int((time.perf_counter() - started) * 1000)

        metadata = dict(result["metadata"])
        metadata["processing_time_ms"] = processing_time_ms
        metadata["pii_scrubbed"] = pii_redacted
        if pii_redacted:
            metadata["pii_note"] = "PHI identifiers were redacted before analysis"

        # Build request_metadata for safety filter
        # This is where the FHE ciphertext hash flows through
        request_metadata: dict[str, Any] = {
            "prompt_hash": hashlib.sha256(scrubbed_text.encode()).hexdigest(),
        }

        # Wire FHE ciphertext hash if provided
        if request.fhe_ciphertext_hash:
            request_metadata["fhe_ciphertext_hash"] = request.fhe_ciphertext_hash
            logger.debug(
                "FHE ciphertext hash provided, will bind to R1 receipt: %s...",
                request.fhe_ciphertext_hash[:16],
            )

        # Run safety filter on the response content
        analysis_output = f"Emotion analysis: {', '.join(e['type'] for e in result['emotions'])}"

        safety_filter = get_safety_filter()

        safety_result = safety_filter.filter_inference_output(
            content=analysis_output,
            user_context={"prompt": scrubbed_text},
            request_metadata=request_metadata,
            model_info={
                "name": request.model,
                "model_fingerprint": f"{request.model}:v1",
            },
        )

        # Extract receipt root hash if available
        receipt_root_hash = safety_result.receipt_root_hash

        return EmotionAnalysisResponse(
            emotions=result["emotions"],
            dimensions=result["dimensions"],
            confidence=result["confidence"],
            metadata=metadata,
            receipt_root_hash=receipt_root_hash,
        )

    except Exception as e:
        logger.error(f"Error in emotion analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Emotion analysis failed: {e!s}",
        ) from e


@router.get(
    "/emotions/health",
    summary="Emotion analysis health check",
)
async def emotions_health() -> dict[str, Any]:
    """Health check for the emotion analysis endpoint."""
    analyzer = get_analyzer()
    return {
        "status": "healthy",
        "endpoint": "/analyze/emotions",
        "safety_filter_active": _safety_filter_state["filter"] is not None,
        "llm_configured": bool(analyzer.api_key),
        "llm_model": analyzer.model,
    }
