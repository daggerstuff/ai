import json
import logging
import re
from pathlib import Path
from typing import Any, Dict

# Configure logger
logger = logging.getLogger(__name__)


class TranscriptCorrector:
    """
    Utility class for correcting transcripts using a multi-pass approach:
    1. Therapeutic Terminology Validation
    2. LLM-based Contextual Correction (Mocked for now)
    3. Structural Alignment (Basic regex cleanup)
    """

    def __init__(self, config_path: str = "ai/config/therapeutic_terminology.json"):
        """
        Initialize the TranscriptCorrector with terminology configuration.

        Args:
            config_path: Path to the JSON configuration file containing
                therapeutic terms.
        """
        self.config_path = Path(config_path)
        self.terms: Dict[str, Any] = self._load_terminology()

    def _load_terminology(self) -> Dict[str, Any]:
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
                        f"Terminology config not found at {self.config_path} or "
                        f"{alt_path}. Using empty config."
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
        text = self._llm_contextual_correction(text, context)

        return text

    def _clean_structure(self, text: str) -> str:
        """Remove filler words and normalize whitespace."""
        # Common filler words in speech, optionally followed by a comma
        fillers = r"\b(um|uh|err|ah|like|you know|I mean)\b,?\s*"

        # Remove fillers (case-insensitive)
        cleaned = re.sub(fillers, "", text, flags=re.IGNORECASE)

        # Normalize whitespace (replace multiple spaces with single space)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned

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
        Mock function for GPT-4 based correction.
        In the future, this will call the LLM service to fix grammar and nuances.
        """
        # TODO: Implement actual LLM call via external service or local model
        # For now, we just log that we would allow the LLM to process this
        # and return the text as is (or maybe apply a dummy transformation
        # for testing if needed)

        # Simulating a check for critical CPTSD terms that might be missed
        # If we had an LLM, we'd ask it: "Correct this transcript keeping CPTSD context
        # in mind."

        return text

    def validate_term_coverage(self, text: str) -> Dict[str, float]:
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
        coverage_score = (
            found_total / total_domain_terms if total_domain_terms > 0 else 0.0
        )

        return {
            "cptsd_term_count": found_cptsd,
            "medical_term_count": found_medical,
            "domain_coverage_score": round(coverage_score, 4),
        }
