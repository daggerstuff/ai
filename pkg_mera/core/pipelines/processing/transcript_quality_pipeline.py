"""Transcript quality orchestration pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai.pkg_mera.core.utils.transcript_corrector import TranscriptCorrector as _TranscriptCorrector


@dataclass
class TranscriptQualityResult:
    success: bool
    original_text: str | None
    corrected_text: str | None
    alignment: dict[str, Any]
    crisis_indicators: list[str]
    metadata: dict[str, Any]


class VoiceTranscriber:
    """Compatibility shim for the expected transcriber dependency."""

    def __init__(self) -> None:
        pass

    def transcribe_audio(self, audio_path: str):
        return MagicTranscribeResult(success=False, error_message="No implementation loaded")


class NemoCuratorClient:
    """Compatibility shim for crisis narrative detection client."""

    def __init__(self) -> None:
        pass

    def detect_crisis_narratives(self, _text: str):
        return []


class NemoEvaluatorClient:
    """Compatibility shim for therapeutic alignment evaluator."""

    def __init__(self) -> None:
        pass

    def evaluate_therapeutic_alignment(self, _text: str):
        return {"score": 0.0, "status": "unknown"}


@dataclass
class MagicTranscribeResult:
    success: bool
    full_text: str = ""
    error_message: str = ""
    confidence_score: float | None = None
    model_used: str | None = None


class TranscriptCorrector(_TranscriptCorrector):
    """Compatibility alias for the expected corrector class."""


class TranscriptQualityPipeline:
    """Orchestrate transcript transcription, correction, and alignment checks."""

    def __init__(self) -> None:
        self.transcriber = VoiceTranscriber()
        self.curator = NemoCuratorClient()
        self.evaluator = NemoEvaluatorClient()
        self.corrector = TranscriptCorrector()

    def process_audio(self, audio_path: Path) -> dict[str, Any]:
        result = self.transcriber.transcribe_audio(str(audio_path))

        if not getattr(result, "success", False):
            return {
                "success": False,
                "error": f"Pass 1 failed: {getattr(result, 'error_message', 'Unknown transcription error')}",
            }

        original_text = str(getattr(result, "full_text", ""))
        alignment = self.evaluator.evaluate_therapeutic_alignment(original_text)
        corrected = self.corrector.correct_transcript(original_text, context="therapy_session")
        crisis_tags = self.curator.detect_crisis_narratives(original_text)

        return {
            "success": True,
            "original_text": original_text,
            "corrected_text": corrected,
            "alignment": alignment,
            "crisis_indicators": crisis_tags,
            "metadata": {
                "transcription": {
                    "confidence_score": getattr(result, "confidence_score", None),
                    "model_used": getattr(result, "model_used", None),
                }
            },
        }


__all__ = [
    "NemoCuratorClient",
    "NemoEvaluatorClient",
    "TranscriptCorrector",
    "TranscriptQualityPipeline",
    "TranscriptQualityResult",
    "VoiceTranscriber",
]
