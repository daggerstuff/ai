import json
import logging
import re
import string
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Configure logger
logger = logging.getLogger(__name__)


class TranscriptCorrector:
    """
    Utility class for correcting transcripts using a multi-pass approach:
    1. Therapeutic Terminology Validation
    2. Optional LLM-based Contextual Correction (if configured)
    3. Structural Alignment (Basic regex cleanup)
    """

    def __init__(
        self,
        config_path: str = "ai/config/therapeutic_terminology.json",
        *,
        contextual_correction_client: Callable[[str, str], str] | None = None,
    ):
        """
        Initialize the TranscriptCorrector with terminology configuration.

        Args:
            config_path: Path to the JSON configuration file containing
                therapeutic terms.
        """
        self.config_path = Path(config_path)
        self.terms: dict[str, Any] = self._load_terminology()
        self._contextual_correction_client = contextual_correction_client

    def _load_terminology(self) -> dict[str, Any]:
        """Load therapeutic terminology from JSON config."""
        try:
            # Handle relative paths from project root if needed
            if not self.config_path.exists():
                # Try relative to the current file location
                # structure is usually ai/utils/transcript_corrector.py
                # config is at ai/config/therapeutic_terminology.json
                # so we go up 2 levels
                base_path = Path(__file__).parent.parent
                alt_path = base_path / "config" / "therapeutic_terminology.json"

                if alt_path.exists():
                    self.config_path = alt_path
                else:
                    logger.warning(
                        f"Terminology config not found at {self.config_path} or {alt_path}. Using empty config."
                    )
                    return {
                        "cptsd_terms": [],
                        "medical_terms": [],
                        "common_misinterpretations": {},
                    }

            with open(self.config_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load terminology config: {e}")
            return {
                "cptsd_terms": [],
                "medical_terms": [],
                "common_misinterpretations": {},
            }

    def correct_transcript(self, text: str, context: str = "therapy_session") -> str:
        """
        Main entry point for transcript correction.

        Args:
            text: Single string containing the transcript text to correct.
            context: Context hint for LLM correction.

        Returns:
            Corrected transcript text.
        """
        if not text or not text.strip():
            return ""

        # Pass 1: Basic Structural Cleanup
        text = self._clean_structure(text)

        # Pass 2: Terminology Replacement
        text = self._apply_terminology_fixes(text)

        # Pass 3: LLM Contextual Correction (Mocked)
        return self._llm_contextual_correction(text, context)

    def _clean_structure(self, text: str) -> str:
        """Remove filler words and normalize whitespace."""
        # Common filler words in speech, optionally followed by a comma
        fillers = r"\b(um|uh|err|ah|like|you know|I mean)\b,?\s*"

        # Remove fillers (case-insensitive)
        cleaned = re.sub(fillers, "", text, flags=re.IGNORECASE)

        # Normalize whitespace (replace multiple spaces with single space)
        return re.sub(r"\s+", " ", cleaned).strip()

    def _apply_terminology_fixes(self, text: str) -> str:
        """Apply deterministic terminology fixes from config."""
        misinterpretations = self.terms.get("common_misinterpretations", {})

        for bad_term, good_term in misinterpretations.items():
            # Use word boundaries to match whole words/phrases ignoring case
            pattern = re.compile(re.escape(bad_term), re.IGNORECASE)
            text = pattern.sub(good_term, text)

        return text

    def _llm_contextual_correction(self, text: str, context: str) -> str:
        """
        Optional contextual correction layer for grammar, tense, and sentence flow.
        """
        context_hint = f"Context: {context}."
        logger.debug("Applying contextual correction with hint: %s", context_hint)

        if self._contextual_correction_client is not None:
            corrected = self._contextual_correction_client(text, context)
            if isinstance(corrected, str) and corrected.strip():
                return corrected.strip()
            logger.warning("Contextual correction client returned empty output; using deterministic fallback.")

        # Deterministic fallback when no external client is configured.
        # - collapse repeated punctuation/filler spaces
        # - normalize sentence capitalization for readability
        collapsed = re.sub(r"\s+", " ", text).strip()
        collapsed = re.sub(r"([!?\\.])\s{2,}", r"\1 ", collapsed)
        collapsed = re.sub(r"\b(i)\b(?=\s+[a-z])", "I", collapsed, flags=re.IGNORECASE)

        sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", collapsed) if segment.strip()]
        normalized_sentences = []
        for sentence in sentences:
            if not sentence:
                continue
            lowered = sentence.strip().rstrip(string.punctuation)
            if lowered:
                lowered = lowered[0].upper() + lowered[1:]
            normalized_sentences.append(lowered + (sentence[-1] if sentence and sentence[-1] in ".!?" else ""))

        result = " ".join(normalized_sentences) if normalized_sentences else collapsed
        # Lightweight CPTSD-aware punctuation/grammar safety net.
        result = result.replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!")
        result = re.sub(r"\b(im)\b", "I'm", result, flags=re.IGNORECASE)
        result = re.sub(r"\b(i'd)\b", "I'd", result, flags=re.IGNORECASE)
        result = result.strip()

        if not result:
            return text.strip()
        return result

    def validate_term_coverage(self, text: str) -> dict[str, float]:
        """
        Calculate metrics on how well the transcript effectively uses domain
        terminology. Useful for validation pass.
        """
        cptsd_terms = {t.lower() for t in self.terms.get("cptsd_terms", [])}
        medical_terms = {t.lower() for t in self.terms.get("medical_terms", [])}

        text_lower = text.lower()

        found_cptsd = sum(term in text_lower for term in cptsd_terms)
        found_medical = sum(term in text_lower for term in medical_terms)

        total_domain_terms = len(cptsd_terms) + len(medical_terms)
        found_total = found_cptsd + found_medical

        # This is a naive metric, just for basic validation
        coverage_score = found_total / total_domain_terms if total_domain_terms > 0 else 0.0

        return {
            "cptsd_term_count": found_cptsd,
            "medical_term_count": found_medical,
            "domain_coverage_score": round(coverage_score, 4),
        }
