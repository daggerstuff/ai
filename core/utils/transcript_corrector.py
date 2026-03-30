# Stub for ai.core.utils.transcript_corrector
# Generated for test compatibility

import json
import re
from pathlib import Path
from typing import Dict, Any


class TranscriptCorrector:
    """Stub implementation for TranscriptCorrector."""

    def __init__(self, config_path: str = None):
        """Initialize transcript corrector with config file."""
        self.config_path = config_path
        self.terms = self._load_terms(config_path)

    def _load_terms(self, config_path: str) -> Dict[str, Any]:
        """Load terms from config file."""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
                    else:
                        return {
                            "cptsd_terms": data if isinstance(data, list) else [],
                            "medical_terms": [],
                            "common_misinterpretations": {}
                        }
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        return {
            "cptsd_terms": [],
            "medical_terms": [],
            "common_misinterpretations": {}
        }

    def _clean_structure(self, text: str) -> str:
        """Clean filler words and structure from text."""
        text = re.sub(r'\b(Um|um|Uh|uh|Like|like),?\s*', '', text)
        return text

    def _apply_terminology_fixes(self, text: str) -> str:
        """Apply terminology corrections."""
        for wrong, correct in self.terms.get("common_misinterpretations", {}).items():
            text = re.sub(re.escape(wrong), correct, text, flags=re.IGNORECASE)

        if "complex ptsd" in text.lower():
            text = re.sub(r'complex ptsd', 'C-PTSD', text, flags=re.IGNORECASE)

        return text

    def correct_transcript(self, text: str, context: str = "") -> str:
        """Correct transcript text."""
        result = self._clean_structure(text)
        result = self._apply_terminology_fixes(result)
        return result

    def validate_term_coverage(self, text: str) -> Dict[str, Any]:
        """Validate term coverage in text."""
        cptsd_count = 0
        medical_count = 0

        for term in self.terms.get("cptsd_terms", []):
            if term.lower() in text.lower():
                cptsd_count += 1

        for term in self.terms.get("medical_terms", []):
            if term.lower() in text.lower():
                medical_count += 1

        total_terms = len(self.terms.get("cptsd_terms", [])) + len(self.terms.get("medical_terms", []))
        score = (cptsd_count + medical_count) / total_terms if total_terms > 0 else 0.0

        return {
            "cptsd_term_count": cptsd_count,
            "medical_term_count": medical_count,
            "domain_coverage_score": round(score, 4)
        }


__all__ = ['TranscriptCorrector']
