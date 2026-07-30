"""Voice pipeline integration façade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .processing.transcript_ingestor import TranscriptIngestor


@dataclass
class VoicePipelineResult:
    success: bool
    transcript: str
    metadata: dict[str, Any]


class VoicePipelineIntegration:
    """Integrate STT + cleaning + quality checks in a single flow."""

    def __init__(self, ingestor: TranscriptIngestor | None = None) -> None:
        self.ingestor = ingestor or TranscriptIngestor()

    def process(self, audio_path: str, *, strict: bool = True) -> VoicePipelineResult:
        result = self.ingestor.process_audio(audio_path)
        if not result.success:
            return VoicePipelineResult(False, "", {"error": result.error_message})

        transcript = str(result.full_text)
        if strict and not transcript.strip():
            return VoicePipelineResult(False, transcript, {"error": "empty_transcript"})

        return VoicePipelineResult(True, transcript, {"char_count": len(transcript), "audio_path": audio_path})

    def process_batch(self, audio_paths: list[str], *, strict: bool = True) -> list[VoicePipelineResult]:
        return [self.process(path, strict=strict) for path in audio_paths]


__all__ = ["VoicePipelineIntegration", "VoicePipelineResult"]
